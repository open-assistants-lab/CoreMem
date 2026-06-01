"""MemoryStore — observations and reflections backed by HybridDB."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from hybriddb import HybridDB


_OBSERVATIONS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "content": "LONGTEXT",
    "priority": "TEXT",
    "observation_ts": "TEXT",
    "referenced_date": "TEXT",
    "source_quote": "TEXT",
    "user_id": "TEXT",
    "agent_id": "TEXT",
    "session_id": "TEXT",
}

_REFLECTIONS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "content": "LONGTEXT",
    "domain": "TEXT",
    "linked_observation_ids": "TEXT",
    "score": "REAL",
    "embedding": "TEXT",
    "user_id": "TEXT",
    "session_id": "TEXT",
}

_HIGH_PRIORITY = {"high", "critical", "important", "urgent"}
_MEDIUM_PRIORITY = {"medium"}


def _priority_tier(priority: str) -> int:
    return 0 if priority.lower() in _HIGH_PRIORITY else 1 if priority.lower() in _MEDIUM_PRIORITY else 2


class MemoryStore:
    """Storage for observations and reflections.

    Uses a single HybridDB instance with two tables. Observations are
    extracted facts from conversation messages. Reflections are synthesized
    patterns discovered from observations.

    Args:
        path: Directory for the HybridDB data.
        embedding_fn: Optional embedding function for semantic search.
    """

    def __init__(self, path: str, embedding_fn: Any = None):
        self._db = HybridDB(path=path, embedding_fn=embedding_fn)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        existing = set(self._db.list_tables())
        if "observations" not in existing:
            self._db.create_table("observations", _OBSERVATIONS_SCHEMA)
        if "reflections" not in existing:
            self._db.create_table("reflections", _REFLECTIONS_SCHEMA)

    # ── Observations ──────────────────────────────────────────────────

    def insert_observations(self, items: list[dict[str, Any]]) -> list[str]:
        ids = []
        now = datetime.now(timezone.utc).isoformat()
        for item in items:
            oid = str(uuid.uuid4())[:12]  # ignore LLM-generated ID, use client UUID
            self._db.insert("observations", {
                "id": oid,
                "content": item.get("content", ""),
                "priority": item.get("priority", "medium"),
                "observation_ts": item.get("observation_ts", now),
                "referenced_date": item.get("referenced_date", ""),
                "source_quote": item.get("source_quote", ""),
                "user_id": item.get("user_id", ""),
                "agent_id": item.get("agent_id", ""),
                "session_id": item.get("session_id", ""),
            })
            ids.append(oid)
        return ids

    def get_observations(
        self, ts_after: str | None = None, limit: int = 50,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        where_parts: list[str] = []
        params: list[Any] = []
        if ts_after:
            where_parts.append("observation_ts > ?")
            params.append(ts_after)
        if user_id:
            where_parts.append("user_id = ?")
            params.append(user_id)
        if session_id:
            where_parts.append("session_id = ?")
            params.append(session_id)
        if agent_id:
            where_parts.append("agent_id = ?")
            params.append(agent_id)
        if metadata:
            for k, v in metadata.items():
                where_parts.append(f"json_extract(metadata, '$.{k}') = ?")
                params.append(str(v))
        where = " AND ".join(where_parts) if where_parts else ""
        return self._db.query("observations", where=where, params=tuple(params),
                              order_by="observation_ts DESC", limit=limit)

    def get_observations_since(
        self, last_id: str | None = None, limit: int = 500,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Cursor-based fetch — all observations after last_id.

        Uses observation_ts for ordering (not id) since IDs are UUIDs.
        Additional scope filters narrow by user/session/agent/metadata.
        """
        where_parts: list[str] = []
        params: list[Any] = []

        if user_id:
            where_parts.append("user_id = ?")
            params.append(user_id)
        if session_id:
            where_parts.append("session_id = ?")
            params.append(session_id)
        if agent_id:
            where_parts.append("agent_id = ?")
            params.append(agent_id)
        if metadata:
            for k, v in metadata.items():
                where_parts.append(f"json_extract(metadata, '$.{k}') = ?")
                params.append(str(v))

        if last_id:
            rows = self._db.raw_query(
                "SELECT observation_ts FROM observations WHERE id = ?", (last_id,),
            )
            if not rows:
                return []
            last_ts = rows[0]["observation_ts"]
            where_parts.append("observation_ts > ?")
            params.append(last_ts)

        where = " AND ".join(where_parts) if where_parts else ""
        return self._db.query("observations", where=where, params=tuple(params),
                              order_by="observation_ts DESC", limit=limit)

    def get_recent_observations(self, days: int = 30, limit: int = 50,
                                 user_id: str | None = None,
                                 session_id: str | None = None,
                                 agent_id: str | None = None) -> list[dict[str, Any]]:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return self.get_observations(ts_after=cutoff, limit=limit,
                                     user_id=user_id, session_id=session_id,
                                     agent_id=agent_id)

    def search_observations(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        results = self._db.search("observations", "content", query, limit=limit)
        return [dict(r) for r in results]

    # ── Reflections ──────────────────────────────────────────────────

    def insert_reflections(self, items: list[dict[str, Any]]) -> list[str]:
        ids = []
        for item in items:
            rid = str(uuid.uuid4())[:12]
            linked = item.get("linked_observation_ids", [])
            emb = item.get("embedding", "")
            self._db.insert("reflections", {
                "id": rid,
                "content": item.get("content", ""),
                "domain": item.get("domain", "general"),
                "linked_observation_ids": json.dumps(linked),
                "score": item.get("score", 1.0),
                "embedding": json.dumps(emb.tolist()) if hasattr(emb, "tolist") else str(emb),
                "user_id": item.get("user_id", ""),
                "session_id": item.get("session_id", ""),
            })
            ids.append(rid)
        return ids

    def get_reflections(self, limit: int = 10,
                        user_id: str | None = None,
                        session_id: str | None = None) -> list[dict[str, Any]]:
        where_parts: list[str] = []
        params: list[Any] = []
        if user_id:
            where_parts.append("user_id = ?")
            params.append(user_id)
        if session_id:
            where_parts.append("session_id = ?")
            params.append(session_id)
        where = " AND ".join(where_parts) if where_parts else ""
        return self._db.query("reflections", where=where, params=tuple(params),
                              order_by="score DESC", limit=limit)

    def search_reflections(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        results = self._db.search("reflections", "content", query, limit=limit)
        return [dict(r) for r in results]

    def apply_decay(self, half_life_days: int = 30) -> int:
        """Reduce scores for reflections older than N days. Returns count."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=half_life_days)).isoformat()
        rows = self._db.query("reflections", where="score > 0.1", limit=1000)
        count = 0
        for row in rows:
            new_score = float(row["score"]) * 0.9
            self._db.update("reflections", row["id"], {"score": new_score})
            count += 1
        return count
