"""L0-L3 wake-up context stack."""

from __future__ import annotations

from typing import Any

from hybriddb import HybridDB

from coremem.heuristics import SearchHeuristics


class WakeUpContext:
    """Build the L0-L3 context stack from a HybridDB backend."""

    def __init__(self, db: HybridDB):
        self._db = db

    def essential(self, user_id: str = "default") -> str:
        parts = [f"[L0: Identity] User: {user_id}"]
        rows = self._db.raw_query(
            "SELECT * FROM messages ORDER BY ts DESC LIMIT ?", (10,)
        )
        if rows:
            snippets = []
            for r in rows[:3]:
                content = r.get("content", "")[:200]
                if len(r.get("content", "")) > 200:
                    content += "..."
                snippets.append(f"  - [{r.get('role', '?')}] {content}")
            parts.append("[L1: Essential] Recent context:\n" + "\n".join(snippets))
        return "\n".join(parts)

    def session(self, session_id: str) -> str | None:
        rows = self._db.raw_query(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
            (session_id, 20),
        )
        if not rows:
            return None
        lines = [f"[L2: On-Demand] Session {session_id}:"]
        for r in rows[:5]:
            content = r.get("content", "")[:200]
            if len(r.get("content", "")) > 200:
                content += "..."
            lines.append(f"  - [{r.get('role', '?')}] {content}")
        return "\n".join(lines)

    def deep_search(self, query: str, limit: int = 10) -> str | None:
        results = self._db.search("messages", "content", query, limit=limit)
        if not results:
            return None
        is_counting = SearchHeuristics.is_counting_question(query)
        lines = [f"[L3: Deep Search] Results for '{query}':"]
        for r in results:
            content = r.get("content", "")
            if is_counting and len(content) > SearchHeuristics.COUNTING_QUESTION_SNIPPET_LENGTH:
                content = content[:SearchHeuristics.COUNTING_QUESTION_SNIPPET_LENGTH] + "..."
            elif len(content) > 500:
                content = content[:500] + "..."
            rid = r.get("id", "?")[:12]
            role = r.get("role", "?")
            score = r.get("score", 0.0)
            lines.append(f"  {rid} [{role}] (score={score:.2f}): {content}")
        return "\n".join(lines)