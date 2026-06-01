"""Observer — per-turn fact extraction from conversations."""

from __future__ import annotations

import logging
from typing import Any

from coremem.observer_utils import chat_messages, estimate_tokens, parse_json_array
from coremem.providers import ChatResponse, LLMProvider, create_provider

logger = logging.getLogger("coremem.observer")

SENTENCE_EXTRACT_PROMPT = """You are a sentence extractor. Your ONLY job is to copy-paste sentences from the conversation that contain facts about the user.

RULES:
- Copy-paste EXACT sentences from the conversation above. Do NOT rephrase, summarize, or change a single character.
- Include ONLY sentences that contain verifiable facts about the user (names, numbers, locations, preferences, events, projects).
- Skip: greetings, meta-commentary, assistant advice, questions the assistant asks.
- One sentence per line. No bullet points, no numbering.
- Return ONLY the sentences, nothing else.

{conversation}

Sentences containing facts about the user:"""


SENTENCE_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_sentences",
        "description": "Extract exact sentences from the conversation that contain facts about the user",
        "parameters": {
            "type": "object",
            "properties": {
                "sentences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Copy-pasted sentences from the conversation. One sentence per array item."
                }
            },
            "required": ["sentences"]
        }
    }
}

OBSERVATION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_observations",
        "description": "Record observations extracted from verified conversation sentences",
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
                            "priority": {"type": "string", "enum": ["🔴", "🟡", "🟢"]},
                            "referenced_date": {"type": "string"},
                            "source_quote": {"type": "string", "description": "EXACT sentence from the source text that proves this fact"},
                        },
                        "required": ["id", "content", "priority", "referenced_date", "source_quote"]
                    }
                }
            },
            "required": ["observations"]
        }
    }
}


OBSERVER_PROMPT = """You are an observer agent. Extract key facts from the verified sentences below and record them as precise observations.

{previous_context}

{sentences}

Return observations via the record_observations tool. One fact per observation. Be exact with values. Use 🔴 for facts with numbers/dates/names, 🟡 for preferences, 🟢 for context. source_quote must be an EXACT copy from the sentences above."""


class Observer:
    """Extract facts from conversation messages as discrete observations.

    Uses two-pass extraction to eliminate hallucination:
      Pass 1: Copy-paste factual sentences from conversation.
      Pass 2: Create structured observations from verified sentences.
    The ObserverPipeline handles scheduling, cursor tracking, token
    counting, and dedup.
    """

    def __init__(self, model: str = "ollama:llama3.2"):
        self._provider = create_provider(model)

    async def run(
        self,
        conversation: list[dict[str, Any]],
        prior_observations: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        prior = prior_observations or []
        prior_text = "\n".join(
            f"- [{o.get('priority', 'medium')}] {o.get('content', '')}"
            for o in prior[:20]
        )
        conv_text = "\n".join(
            m.get("content", "")
            for m in conversation[-300:]
        )

        # Pass 1: Extract sentences via tool call (structured output)
        sentence_prompt = SENTENCE_EXTRACT_PROMPT.format(conversation=conv_text)
        sent_response = await self._provider.chat_with_tools(
            chat_messages("", sentence_prompt), [SENTENCE_EXTRACT_TOOL],
        )
        sentences_text = sent_response.content.strip()
        if not sentences_text or sentences_text.startswith("{"):
            parsed = parse_json_array(sentences_text)
            if parsed and "sentences" in parsed[0]:
                sentences_text = "\n".join(parsed[0]["sentences"])
            else:
                sentences_text = sentences_text

        if not sentences_text or sentences_text == "{}":
            return []

        # Pass 2: Create observations via tool call (structured output)
        user_prompt = OBSERVER_PROMPT.format(
            sentences=sentences_text,
            previous_context=f"Known observations (do not repeat):\n{prior_text or '(none)'}",
        )
        response = await self._provider.chat_with_tools(
            chat_messages("", user_prompt), [OBSERVATION_TOOL],
        )
        parsed = parse_json_array(response.content)
        if parsed and "observations" in parsed[0]:
            return parsed[0]["observations"]
        return parsed


class ObserverPipeline:
    """Per-turn fact extraction pipeline.

    Fires after each agent turn but only runs the LLM call when both
    ``token_threshold`` new tokens have accumulated AND ``min_turns``
    have passed since the last run.

    Args:
        core: MemoryCore instance for reading conversation messages.
        store: MemoryStore instance for writing observations.
        session_id: Session identifier for cursor tracking and fetch filtering.
        user_id: If set, only observe messages from this user.
        agent_id: If set, only observe messages involving this agent.
        metadata: If set, only observe messages with matching metadata keys.
        model: Provider model string (default ``"ollama:llama3.2"``).
        token_threshold: Fire after this many new tokens (default 8000).
        min_turns: Minimum turns between runs (default 3).
        max_messages: Max messages to fetch per check (default 500).
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
    ):
        self._core = core
        self._store = store
        self._session_id = session_id
        self._user_id = user_id
        self._agent_id = agent_id
        self._metadata = metadata
        self._observer = Observer(model=model)
        self._token_threshold = token_threshold
        self._min_turns = min_turns
        self._max_messages = max_messages

        self._last_observed_id: str | None = None
        self._turns_since_last_run: int = 0
        self._running: bool = False

    async def after_turn(self) -> list[dict[str, Any]] | None:
        """Called after each agent turn. Returns new observations or None.

        No-op unless enough new tokens have accumulated AND min_turns passed.
        """
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

        # Filter tool messages and find cutoff
        filtered: list[dict[str, Any]] = []
        new_messages: list[dict[str, Any]] = []
        seen_watermark = self._last_observed_id is None

        for m in messages:
            if m.role == "tool":
                continue
            meta_parts: list[str] = []
            if getattr(m, "user_id", "") and getattr(m, "user_id", "") != self._session_id:
                meta_parts.append(m.user_id)
            if getattr(m, "agent_id", ""):
                meta_parts.append(m.agent_id)
            if getattr(m, "ts", None):
                ts_str = m.ts.isoformat() if hasattr(m.ts, "isoformat") else str(m.ts)
                meta_parts.append(ts_str[:19])  # YYYY-MM-DDTHH:MM:SS
            if getattr(m, "metadata", None) and m.metadata:
                if isinstance(m.metadata, dict):
                    non_empty = {k: v for k, v in m.metadata.items() if v}
                    if non_empty:
                        meta_parts.append(str(non_empty))
                elif m.metadata:
                    meta_parts.append(str(m.metadata))

            meta_str = f" | {' | '.join(meta_parts)}" if meta_parts else ""
            msg_dict: dict[str, Any] = {
                "role": m.role,
                "content": f"[{m.role}{meta_str}] {m.content}",
            }
            filtered.append(msg_dict)
            if not seen_watermark:
                if m.id == self._last_observed_id:
                    seen_watermark = True
            else:
                new_messages.append(msg_dict)

        # On first run or watermark not found: all are new
        if self._last_observed_id is None or not seen_watermark:
            new_messages = filtered

        # Count tokens in new messages
        new_tokens = sum(estimate_tokens(m["content"]) for m in new_messages)

        if new_tokens < self._token_threshold or self._turns_since_last_run < self._min_turns:
            return None

        # Fetch prior observations for dedup context
        prior = self._store.get_recent_observations(days=30, limit=50)

        # Run observer
        observations = await self._observer.run(filtered, prior)

        # Post-hoc dedup: skip near-duplicates (simple string similarity)
        # Also drop observations without valid, verifiable source quotes
        new_obs: list[dict[str, Any]] = []
        source_text = " ".join(
            m.content for m in messages if m.role != "tool"
        )
        for obs in observations:
            content = obs.get("content", "")
            source_quote = obs.get("source_quote", "")

            # Gate 1: source quote must exist in conversation AND support the claim
            if source_quote and _quote_verified(source_quote, content, source_text):
                pass  # quote is valid
            else:
                continue  # no quote, or quote doesn't appear in source — drop

            # Gate 2: string similarity dedup vs prior observations
            if any(_string_similarity(content, p.get("content", "")) > 0.85 for p in prior):
                continue

            # Tag with session/agent/user context from the pipeline
            obs["session_id"] = self._session_id
            obs["user_id"] = messages[0].user_id if hasattr(messages[0], "user_id") else ""
            obs["agent_id"] = messages[0].agent_id if hasattr(messages[0], "agent_id") else ""
            new_obs.append(obs)

        if new_obs:
            self._store.insert_observations(new_obs)
            if new_messages:
                # Update watermark from newest message fetched
                self._last_observed_id = messages[0].id

        self._turns_since_last_run = 0
        return new_obs


def _string_similarity(a: str, b: str) -> float:
    """Simple string similarity using difflib."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _quote_verified(quote: str, claim: str, source_text: str = "", threshold: float = 0.3) -> bool:
    """Check whether the source quote supports the observation.

    1. Quote must exist in the source conversation text (substring match)
    2. Claim words must overlap with the quote (internal consistency)

    If source_text is empty, skips the external verification step.
    """
    if not quote or not claim:
        return False

    # External: quote must appear in the source conversation
    if source_text:
        quote_lower = quote.lower().strip().strip('"').strip("'")
        source_lower = source_text.lower()
        if len(quote_lower) >= 10 and quote_lower not in source_lower:
            return False

    # Internal: claim words must overlap with the quote
    claim_lower = claim.lower().strip()
    quote_lower = quote.lower().strip()

    if len(quote_lower) >= 10 and claim_lower in quote_lower:
        return True

    words = [w for w in claim_lower.split() if len(w) > 3]
    if not words:
        return True

    matches = sum(1 for w in words if w in quote_lower)
    return matches / len(words) >= threshold
