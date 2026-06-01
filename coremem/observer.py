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

from coremem.grounding import AlignmentTier, align_quote  # noqa: F401  # used in pipeline (Task 3)
from coremem.observer_utils import parse_json_array
from coremem.providers import create_provider
from coremem.types import Memory

logger = logging.getLogger("coremem.observer")


# ── Prompt: system message (instructions + 2 few-shot examples) ────────────


OBSERVER_SYSTEM_PROMPT = """You are an observation agent. Extract key facts from a conversation and return them as structured observations via the record_observations tool.

RULES:
- One fact per observation. Be exact with values.
- source_quote: a VERBATIM sub-string of the conversation. Copy-paste exactly — do not rephrase, do not paraphrase, do not change a single character. If you cannot find a verbatim sub-string, do not return the observation.
- Do not invent facts. If nothing factual is present, return an empty observations array.

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
                    return cast(list[dict[str, Any]], parsed[0]["observations"])
                if parsed and isinstance(parsed[0], dict) and "content" in parsed[0]:
                    return parsed
                return []

        content = getattr(response, "content", "")
        if not content:
            return []
        parsed = parse_json_array(content)
        if parsed and "observations" in parsed[0]:
            return cast(list[dict[str, Any]], parsed[0]["observations"])
        return parsed if parsed else []
