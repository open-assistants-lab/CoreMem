"""Observer — per-turn fact extraction from conversations."""

from __future__ import annotations

import logging
from typing import Any

from coremem.observer_utils import chat_messages, estimate_tokens, parse_json_array
from coremem.providers import ChatResponse, LLMProvider, create_provider

logger = logging.getLogger("coremem.observer")

OBSERVER_PROMPT = """You are an observer agent. Your job is to extract key facts from a conversation and record them as precise, concise observations.

Input: A conversation log between a user and an AI assistant.

Output: A JSON array of observations. Each observation must have:
- "id": a unique ID like "obs_<uuid>"
- "content": ONE fact per observation, in plain English. Be exact with values (names, numbers, dates).
- "priority": one of "\U0001f534" (high — precise value like name, address, number), "\U0001f7e1" (medium — preference, opinion), "\U0001f7e2" (low — context, trivia)
- "referenced_date": the date mentioned in the observation content, or "" if none

CRITICAL RULES:
- One fact per observation. Do not combine multiple facts.
- Use exact values as stated. Never paraphrase numbers or proper nouns.
- If the user CORRECTS previously stated information, capture both as separate observations with different timestamps.
- Skip generic chat, greetings, and meta-commentary.
- Skip observations already observed (listed below as known).

{conversation}

{previous_context}

Return ONLY the JSON array, no markdown wrapping, no explanation."""


class Observer:
    """Extract facts from conversation messages as discrete observations.

    A single-turn LLM call. The ObserverPipeline handles scheduling, cursor
    tracking, token counting, and dedup.
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
            f"[{m.get('role', 'user')}] {m.get('content', '')}"
            for m in conversation[-300:]  # keep prompt size bounded
        )
        user_prompt = OBSERVER_PROMPT.format(
            conversation=conv_text,
            previous_context=f"Known observations (do not repeat):\n{prior_text or '(none)'}",
        )
        response = await self._provider.chat(chat_messages("", user_prompt))
        return parse_json_array(response.content)


class ObserverPipeline:
    """Per-turn fact extraction pipeline.

    Fires after each agent turn but only runs the LLM call when both
    ``token_threshold`` new tokens have accumulated AND ``min_turns``
    have passed since the last run.

    Args:
        core: MemoryCore instance for reading conversation messages.
        store: MemoryStore instance for writing observations.
        session_id: Session identifier for cursor tracking and fetch filtering.
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
        model: str = "ollama:llama3.2",
        token_threshold: int = 8000,
        min_turns: int = 3,
        max_messages: int = 500,
    ):
        self._core = core
        self._store = store
        self._session_id = session_id
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
            limit=self._max_messages,
        )

        # Filter tool messages and find cutoff
        filtered: list[dict[str, Any]] = []
        new_messages: list[dict[str, Any]] = []
        seen_watermark = self._last_observed_id is None

        for m in messages:
            if m.role == "tool":
                continue
            msg_dict: dict[str, Any] = {"role": m.role, "content": m.content}
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
        new_obs: list[dict[str, Any]] = []
        for obs in observations:
            content = obs.get("content", "")
            if any(_string_similarity(content, p.get("content", "")) > 0.85 for p in prior):
                continue
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
