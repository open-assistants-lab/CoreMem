"""Reflector — scheduled pattern discovery from observations."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from coremem.observer_utils import chat_messages, parse_json_array
from coremem.providers import create_provider

logger = logging.getLogger("coremem.reflector")

REFLECTOR_SYSTEM = (
    "You discover patterns and meaning from observations about a user. "
    "Return ONLY a JSON array of reflections. Each reflection is an object "
    "with keys: content (the insight), domain (one of: preference, career, "
    "lifestyle, relationship, skill, health, finance, travel, project, "
    "education, other), linked_observation_ids (array of observation IDs "
    "that support this reflection), confidence (0.0–1.0). "
    "Look for: recurring themes, behavior patterns, value shifts, goal "
    "trajectories, cross-domain connections. Do not repeat prior reflections."
)


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
        user_prompt = (
            f"Recent observations:\n\n{obs_text}\n\n"
            f"Prior reflections (do not repeat):\n{prior_text or '(none)'}"
        )
        response = await self._provider.chat(chat_messages(REFLECTOR_SYSTEM, user_prompt))
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
        model: Provider model string (default ``"openai:gpt-4o"``).
        embedding_fn: Embedding function for cosine similarity quality gate.
        interval_hours: How often to run (default 24).
        min_observations: Skip if fewer new observations (default 10).
    """

    def __init__(
        self,
        store: Any,
        model: str = "openai:gpt-4o",
        embedding_fn: Any = None,
        interval_hours: int = 24,
        min_observations: int = 10,
    ):
        self._store = store
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
