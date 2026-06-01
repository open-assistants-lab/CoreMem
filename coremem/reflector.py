"""Reflector — scheduled pattern discovery from observations."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from coremem.observer_utils import chat_messages, parse_json_array
from coremem.providers import create_provider

logger = logging.getLogger("coremem.reflector")

REFLECTOR_PROMPT = """You are a reflection agent. Your job is to think about what you know about a user and discover patterns, relationships, and deeper meaning.

Input: All observations collected about the user, plus any previous reflections for context.

Output: A JSON array of reflections. Each reflection must have:
- "id": a unique ID like "refl_<uuid>"
- "content": A synthesized insight — not a fact, but what the facts MEAN when considered together. Patterns, contradictions, values, trajectories, predictions.
- "domain": Category label (preference, career, lifestyle, relationship, skill, value, habit, health, finance, etc.)
- "linked_observation_ids": List of observation IDs that support this reflection

CRITICAL RULES:
- Do NOT repeat facts. Observations already say "lives in Denver." You say WHY it matters — "Has relocated twice for family; values school quality above career."
- Discover multi-observation patterns. Single facts do not need reflection.
- If observations contradict ("lives in Seattle" vs "lives in Denver"), note the change: "Previously in Seattle, now in Denver as of DATE. Reason: ..."
- Generate predictions where patterns warrant: "May relocate again within 2 years based on past behavior."
- Workflow and process patterns are valuable. User_id and session_id indicate who/recurrence. Capture: "User checks Jira every Monday morning" or "Deploys via staging → Jenkins → production pipeline." Use domain="workflow" for these.
- Quality over quantity. 3-5 meaningful reflections are better than 15 trivial ones.

{observations}

{previous_reflections}

Return ONLY the JSON array, no markdown wrapping, no explanation."""


class Reflector:
    """Synthesize patterns from observations as high-level reflections.

    A single-turn LLM call. ReflectorPipeline handles scheduling, cursor
    tracking, quality gates, and dedup.
    """

    def __init__(self, model: str = "openai:gpt-4o"):
        self._provider = create_provider(model)

    async def run(
        self,
        observations: list[dict[str, Any]],
        prior_reflections: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        prior = prior_reflections or []
        obs_text = _format_observations(observations)
        prior_text = "\n".join(
            f"- [{r.get('domain', 'general')}] {r.get('content', '')}"
            for r in prior[:10]
        )
        user_prompt = REFLECTOR_PROMPT.format(
            observations=obs_text,
            previous_reflections=f"Prior reflections (do not repeat):\n{prior_text or '(none)'}",
        )
        response = await self._provider.chat(chat_messages("", user_prompt))
        return parse_json_array(response.content)


def _format_observations(observations: list[dict[str, Any]]) -> str:
    lines = []
    for o in observations:
        pid = o.get("id", "?")
        priority = o.get("priority", "medium")
        content = o.get("content", "")
        date = o.get("referenced_date") or o.get("observation_ts", "")
        lines.append(f"[{pid}] ({priority}) {date}: {content}")
    return "\n".join(lines)


class ReflectorPipeline:
    """Scheduled pattern discovery from observations.

    Runs on a configurable timer (default 24h). Fires additional quality
    gates (cosine similarity dedup) and priority sampling for large
    observation sets.

    Args:
        store: MemoryStore instance for reading observations + reflections.
        user_id: If set, only reflect on observations from this user.
        session_id: If set, only reflect on observations from this session.
        agent_id: If set, only reflect on observations involving this agent.
        metadata: If set, only reflect on observations with matching metadata.
        model: Provider model string (default ``"openai:gpt-4o"``).
        embedding_fn: Embedding function for cosine similarity quality gate.
        interval_hours: How often to run (default 24).
        min_observations: Skip if fewer new observations (default 10).
    """

    def __init__(
        self,
        store: Any,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model: str = "openai:gpt-4o",
        embedding_fn: Any = None,
        interval_hours: int = 24,
        min_observations: int = 10,
    ):
        self._store = store
        self._user_id = user_id
        self._session_id = session_id
        self._agent_id = agent_id
        self._metadata = metadata
        self._reflector = Reflector(model=model)
        self._embedding_fn = embedding_fn
        self._interval_hours = interval_hours
        self._min_observations = min_observations

        self._last_run_ts: float = 0.0
        self._last_run_observation_id: str | None = None

    async def maybe_run(self) -> list[dict[str, Any]] | None:
        """Run if interval has elapsed. Returns new reflections or None."""
        now = time.time()
        if now - self._last_run_ts < self._interval_hours * 3600:
            return None
        return await self.run_now()

    async def run_now(self) -> list[dict[str, Any]] | None:
        """Force a run regardless of timer."""
        observations = self._store.get_observations_since(
            last_id=self._last_run_observation_id, limit=500,
            user_id=self._user_id,
            session_id=self._session_id,
            agent_id=self._agent_id,
            metadata=self._metadata,
        )

        if len(observations) < self._min_observations:
            return None

        # Priority sampling for large observation sets
        if len(observations) > 200:
            high_med = [o for o in observations
                        if o.get("priority", "").lower() in ("high", "medium")]
            green = [o for o in observations
                     if o.get("priority", "").lower() not in ("high", "medium")]
            green = sorted(green, key=lambda o: o.get("observation_ts", ""), reverse=True)[:100]
            observations = high_med + green

        prior = self._store.get_reflections(limit=10)

        reflections = await self._reflector.run(observations, prior)

        # Quality gate: cosine similarity dedup
        good: list[dict[str, Any]] = []
        for r in reflections:
            content = r.get("content", "")
            if self._embedding_fn and _is_redundant(content, prior, self._embedding_fn):
                continue
            if self._embedding_fn:
                emb = self._embedding_fn(content)
                if hasattr(emb, "tolist"):
                    emb = emb.tolist()
                r["embedding"] = emb
            r["score"] = 1.0
            good.append(r)

        if good:
            new_ids = self._store.insert_reflections(good)
            if observations:
                self._last_run_observation_id = observations[-1].get("id")
            self._last_run_ts = time.time()

        return good


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _is_redundant(
    new_text: str,
    existing: list[dict[str, Any]],
    embedding_fn: Any,
    threshold: float = 0.9,
) -> bool:
    new_emb = embedding_fn(new_text)
    if hasattr(new_emb, "tolist"):
        new_emb = np.array(new_emb)
    for r in existing:
        stored = r.get("embedding")
        if stored is None:
            continue
        if isinstance(stored, str):
            import json
            stored = np.array(json.loads(stored))
        elif isinstance(stored, list):
            stored = np.array(stored)
        if _cosine(new_emb, stored) > threshold:
            return True
    return False
