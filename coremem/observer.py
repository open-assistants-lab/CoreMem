"""Observer — single-pass fact extraction from conversations.

Uses a 3-tier alignment gate (coremem.grounding.align_quote) to
deterministically catch fabricated source_quote values. The model is
prompted via CogCanvas-style system message with 2 few-shot examples
that demonstrate the verbatim-quote contract.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, cast

from coremem.grounding import AlignmentTier, align_quote
from coremem.observer_utils import parse_json_array
from coremem.providers import create_provider
from coremem.types import Memory

logger = logging.getLogger("coremem.observer")


# ── Prompt: system message (instructions + 2 few-shot examples) ────────────


OBSERVER_SYSTEM_PROMPT = """You are an observation agent. Extract key facts from a conversation and return them as structured observations via the record_observations tool.

RULES:
- One fact per observation. Be exact with values.
- source_quote: a VERBATIM sub-string of the conversation. Copy-paste exactly — do not rephrase, do not paraphrase, do not change a single character. If you cannot find a verbatim sub-string, do not return the observation.
- Be liberal in what you consider worth recording. If the conversation contains user statements of any kind, attempt to extract at least one observation.
- If you cannot find a verbatim sub-string for a fact, do not return that observation — the alignment gate handles grounding.

IMPORTANCE: 0.8-1.0 identity/contact/job/salary.
            0.5-0.7 preferences/habits/projects.
            0.1-0.4 context/trivia.
ENTITIES: list of named entities (people, companies, products, locations).

Few-shot examples:

Example 1:
[2024-01-15T10:00:00] user: I just moved to Seattle last month for a new job at Anthropic.
[2024-01-15T10:01:00] assistant: That's exciting! What's your role?
[2024-01-15T10:02:00] user: I'm a research engineer working on alignment.

{"id": "obs_1", "content": "User moved to Seattle in January 2024", "referenced_date": "2024-01", "source_quote": "I just moved to Seattle last month", "importance": 0.7, "entities": ["Seattle"]}
{"id": "obs_2", "content": "User works at Anthropic as a research engineer", "referenced_date": "2024-01-15", "source_quote": "I'm a research engineer working on alignment", "importance": 0.9, "entities": ["Anthropic"]}

Example 2:
[2024-02-03T14:30:00] user: My favorite programming language is Rust, though I still use Python for data work.
[2024-02-03T14:31:00] assistant: Why do you prefer Rust?
[2024-02-03T14:32:00] user: The borrow checker catches bugs before runtime.

{"id": "obs_1", "content": "User's favorite programming language is Rust", "referenced_date": "2024-02-03", "source_quote": "My favorite programming language is Rust", "importance": 0.6, "entities": ["Rust"]}
{"id": "obs_2", "content": "User uses Python for data work", "referenced_date": "2024-02-03", "source_quote": "I still use Python for data work", "importance": 0.5, "entities": ["Python"]}
"""


# ── Tool schema (no `priority` field) ─────────────────────────────────────


OBSERVATION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_observations",
        "description": "Record observations extracted from the conversation",
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique ID like obs_01"},
                            "content": {"type": "string", "description": "ONE fact per observation"},
                            "referenced_date": {"type": "string"},
                            "source_quote": {"type": "string", "description": "EXACT sub-string of the conversation (prefix stripped)"},
                            "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "entities": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "id",
                            "content",
                            "referenced_date",
                            "source_quote",
                            "importance",
                            "entities",
                        ],
                    },
                },
            },
            "required": ["observations"],
        },
    },
}


# ── Observer ───────────────────────────────────────────────────────────────


class Observer:
    """Single-pass fact extraction from conversation messages.

    Makes one chat_with_tools call per invocation. Returns parsed
    observations or [] on parse failure / empty tool_calls.
    """

    def __init__(
        self,
        model: str = "ollama:llama3.2",
        enable_gleaning: bool = False,
    ):
        if enable_gleaning:
            raise NotImplementedError(
                "gleaning pass not implemented; see docs/observer-hallucination-review.md for the CogCanvas pattern"
            )
        self._provider = create_provider(model)

    async def run(
        self,
        messages: list[Memory],
        prior_observations: list[dict[str, Any]] | None = None,
        observation_date: str | None = None,
    ) -> list[dict[str, Any]]:
        prior = prior_observations or []
        date_str = observation_date or datetime.now(UTC).date().isoformat()

        llm_messages = self._build_messages(messages, prior, date_str)
        response = await self._provider.chat_with_tools(llm_messages, [OBSERVATION_TOOL])
        return self._parse_response(response)

    def _build_messages(
        self,
        messages: list[Memory],
        prior: list[dict[str, Any]],
        date_str: str,
    ) -> list[dict[str, str]]:
        """Build the native messages array (system + context + conversation)."""
        context_block = self._build_context_block(prior, date_str)
        out: list[dict[str, str]] = [
            {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
            {"role": "user", "content": context_block},
        ]
        for m in messages:
            if m.content and m.ts is not None:
                ts_str = m.ts.isoformat()[:19]
                out.append({"role": m.role, "content": f"[{ts_str}] {m.content}"})
        return out

    @staticmethod
    def _build_context_block(
        prior: list[dict[str, Any]],
        date_str: str,
    ) -> str:
        """Build the # Already extracted + # Observation date preamble."""
        if prior:
            prior_lines = "\n".join(
                f"- {o.get('content', '')}" for o in prior[:20]
            )
            already = f"# Already extracted (last 20)\n{prior_lines}\n\n"
        else:
            already = "# Already extracted (last 20)\n(none)\n\n"
        return f"{already}# Observation date\n{date_str}"

    @staticmethod
    def _parse_response(response: Any) -> list[dict[str, Any]]:
        """Extract observations from a chat_with_tools response.

        Reads payload from tool_calls[0].function.arguments (NOT content).
        Falls back to parsing content if no tool_calls.
        """
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            arguments = tool_calls[0].get("function", {}).get("arguments", "")
            if arguments:
                parsed = parse_json_array(arguments)
                if parsed and "observations" in parsed[0]:
                    observations = cast(list[dict[str, Any]], parsed[0]["observations"])
                elif parsed and isinstance(parsed[0], dict) and "content" in parsed[0]:
                    observations = parsed
                else:
                    return []
            else:
                return []
        else:
            content = getattr(response, "content", "")
            if not content:
                return []
            parsed = parse_json_array(content)
            if parsed and "observations" in parsed[0]:
                observations = cast(list[dict[str, Any]], parsed[0]["observations"])
            elif parsed:
                observations = parsed
            else:
                return []

        for obs in observations:
            obs["importance"] = None
        return observations


# ── ObserverPipeline ───────────────────────────────────────────────────────


class ObserverPipeline:
    """Per-turn fact extraction pipeline with alignment-gated verification.

    Fires after each agent turn but only runs the LLM call when both
    ``token_threshold`` new tokens have accumulated AND ``min_turns``
    have passed since the last run.
    """

    def __init__(
        self,
        core: Any,
        store: Any,
        session_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model: str = "ollama:llama3.2",
        token_threshold: int = 8000,
        min_turns: int = 3,
        max_messages: int = 500,
        enable_gleaning: bool = False,
    ):
        if enable_gleaning:
            raise NotImplementedError(
                "gleaning pass not implemented; see docs/observer-hallucination-review.md for the CogCanvas pattern"
            )
        self._core = core
        self._store = store
        self._session_id = session_id
        self._user_id = user_id
        self._agent_id = agent_id
        self._metadata = metadata
        self._observer = Observer(model=model, enable_gleaning=enable_gleaning)
        self._token_threshold = token_threshold
        self._min_turns = min_turns
        self._max_messages = max_messages

        self._last_observed_id: str | None = None
        self._turns_since_last_run: int = 0
        self._running: bool = False

    async def after_turn(self) -> list[dict[str, Any]] | None:
        """Called after each agent turn. Returns new observations or None."""
        if self._running:
            return None
        self._turns_since_last_run += 1
        try:
            self._running = True
            return await self._maybe_run()
        finally:
            self._running = False

    async def _maybe_run(self) -> list[dict[str, Any]] | None:
        messages = self._core.fetch(
            session_id=self._session_id,
            user_id=self._user_id,
            agent_id=self._agent_id,
            metadata=self._metadata,
            limit=self._max_messages,
        )

        new_messages: list[Memory] = []
        seen_watermark = self._last_observed_id is None
        for m in messages:
            if m.role == "tool":
                continue
            if not seen_watermark:
                if m.id == self._last_observed_id:
                    seen_watermark = True
                continue
            new_messages.append(m)
        if not seen_watermark:
            new_messages = [m for m in messages if m.role != "tool"]

        canonical_lines: list[str] = []
        for m in new_messages:
            if m.content and m.ts is not None:
                ts_str = m.ts.isoformat()[:19]
                canonical_lines.append(f"[{ts_str}] {m.content}")
        canonical_text = "\n".join(canonical_lines)

        new_tokens = sum(_estimate_tokens_line(line) for line in canonical_lines)
        if new_tokens < self._token_threshold or self._turns_since_last_run < self._min_turns:
            return None

        prior = self._store.get_recent_observations(days=30, limit=50)

        observations = await self._observer.run(new_messages, prior)

        new_obs: list[dict[str, Any]] = []
        for obs in observations:
            quote = obs.get("source_quote", "").strip()
            content = obs.get("content", "").strip()
            if not quote or not content:
                continue

            result = align_quote(quote, canonical_text)
            if result.tier == AlignmentTier.NONE:
                continue
            obs["alignment_tier"] = result.tier.value
            obs["alignment_confidence"] = result.confidence

            if any(_string_similarity(content, p.get("content", "")) > 0.85 for p in prior):
                continue

            obs["session_id"] = self._session_id
            obs["user_id"] = self._user_id or ""
            obs["agent_id"] = self._agent_id or ""
            new_obs.append(obs)

        if new_obs:
            self._store.insert_observations(new_obs)
        if messages:
            self._last_observed_id = messages[0].id
        self._turns_since_last_run = 0
        return new_obs


# ── Helpers ────────────────────────────────────────────────────────────────


def _estimate_tokens_line(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return max(1, len(text) // 4)


def _string_similarity(a: str, b: str) -> float:
    """Simple string similarity using difflib."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
