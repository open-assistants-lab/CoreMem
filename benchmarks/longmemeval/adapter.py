"""LongMemEval benchmark adapter for coremem.

Direct-injection mode — injects haystack sessions into coremem,
runs search, measures Recall@K. No LLM. No HTTP. No agent loop.
Pure retrieval benchmarking.

Usage:
    python -m coremem.benchmarks.longmemeval.eval --backend chroma --limit 5
    python -m coremem.benchmarks.longmemeval.eval --backend hybrid --limit 10
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from coremem.core import MemoryCore


class LongMemEvalAdapter:
    """Inject LongMemEval sessions and measure retrieval recall."""

    def __init__(self, core: MemoryCore):
        self._core = core

    def inject_sessions(
        self,
        haystack_sessions: list[dict],
        verbose: bool = False,
    ) -> dict[str, list[str]]:
        """Inject all haystack sessions into coremem.

        Each session is a list of message dicts. Each message gets tagged
        with its session_id for later dedup during search.

        Args:
            haystack_sessions: List of sessions, each a list of {"role", "content"} dicts.
            verbose: Print injection progress.

        Returns:
            Dict mapping session_id → list of ingested memory IDs.
        """
        session_memory_ids: dict[str, list[str]] = {}
        for si, session in enumerate(haystack_sessions):
            sid = f"session_{si:04d}"
            ids = self._core.ingest_many(session, session_id=sid)
            session_memory_ids[sid] = ids
            if verbose:
                print(f"  Session {sid}: {len(ids)}/{len(session)} messages ingested", flush=True)
        return session_memory_ids

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search and return session_ids from results."""
        results = self._core.search(query, limit=limit)
        return [
            {
                "session_id": r.memory.session_id,
                "content": r.memory.content[:200],
                "score": r.score,
                "source": r.source,
            }
            for r in results
        ]

    def recall_at_k(self, query: str, answer_session_ids: list[str], k: int = 5) -> tuple[bool, int]:
        """Check if any answer session appears in top-K results.

        Returns (is_hit, count_of_answer_sessions_found).
        """
        results = self._core.search(query, limit=k)
        found_sessions = {r.memory.session_id for r in results}
        matches = found_sessions & set(answer_session_ids)
        return len(matches) > 0, len(matches)


def load_longmemeval_questions(
    data_path: str | Path,
    question_types: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load LongMemEval questions from JSON.

    Args:
        data_path: Path to LongMemEval JSON data file.
        question_types: Optional filter by question_type field.
        limit: Optional max number of questions to load.

    Returns:
        List of question dicts with keys: question_id, question, question_type,
        answer, answer_session_id, haystack_sessions.
    """
    with open(data_path) as f:
        data = json.load(f)

    questions = data if isinstance(data, list) else data.get("questions", [])

    if question_types:
        questions = [q for q in questions if q.get("question_type") in question_types]

    if limit:
        questions = questions[:limit]

    return questions
