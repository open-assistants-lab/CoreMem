"""MemoryStore — observations, metadata, and reflections backed by HybridDB."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from hybriddb import HybridDB

_OBSERVATIONS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "content": "LONGTEXT",
    "source_quote": "TEXT",
    "referenced_date": "TEXT",
    "observation_ts": "TEXT",
    "user_id": "TEXT",
    "agent_id": "TEXT",
    "session_id": "TEXT",
    "alignment_tier": "TEXT",         # NEW: 0.4.0
    "alignment_confidence": "REAL",   # NEW: 0.4.0
}

_OBSERVATION_METADATA_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "observation_id": "TEXT",
    "importance": "REAL",
    "entities": "TEXT",
    "priority": "TEXT",
    "confidence": "REAL",
    "enrichment_ts": "TEXT",
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


def _join_meta(observations: list[dict]) -> list[dict]:
    """Flatten observation + metadata into a single dict for backward compat."""
    return observations


class MemoryStore:
    """Storage for observations and reflections.

    Observations are split into two tables:
      - ``observations`` — immutable facts (content, source_quote, scope).
      - ``observation_metadata`` — mutable enrichment (importance, entities,
        priority, confidence). Enrichment rows can be added over time by
        the Observer, Reflector, or human verification.

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
        else:
            self._migrate_observations_v2()
        if "observation_metadata" not in existing:
            self._db.create_table("observation_metadata", _OBSERVATION_METADATA_SCHEMA)
        if "reflections" not in existing:
            self._db.create_table("reflections", _REFLECTIONS_SCHEMA)

    def _list_observation_columns(self) -> set[str]:
        """Return column names in the observations table via PRAGMA."""
        rows = self._db.raw_query("PRAGMA table_info(observations)")
        return {r["name"] for r in rows}

    def _migrate_observations_v2(self) -> None:
        """0.4.0 migration: add alignment_tier and alignment_confidence columns.

        Idempotent: skip if columns already exist.
        """
        existing_cols = self._list_observation_columns()
        if "alignment_tier" in existing_cols and "alignment_confidence" in existing_cols:
            return
        if hasattr(self._db, "add_column"):
            if "alignment_tier" not in existing_cols:
                self._db.add_column("observations", "alignment_tier", "TEXT")
            if "alignment_confidence" not in existing_cols:
                self._db.add_column("observations", "alignment_confidence", "REAL")
        else:
            self._migrate_via_recreate()

    def _migrate_via_recreate(self) -> None:
        """Fallback migration: rename old table, create new, copy data."""
        import uuid as _uuid
        old_name = f"observations_old_{_uuid.uuid4().hex[:8]}"
        self._db.raw_query(f"ALTER TABLE observations RENAME TO {old_name}")
        self._db.create_table("observations", _OBSERVATIONS_SCHEMA)
        copy_cols = {"id", "content", "source_quote", "referenced_date", "observation_ts",
                     "user_id", "agent_id", "session_id"}
        select_list = ", ".join(sorted(copy_cols))
        self._db.raw_query(
            f"INSERT INTO observations ({select_list}) "
            f"SELECT {select_list} FROM {old_name}"
        )
        self._db.raw_query(f"DROP TABLE {old_name}")

    # ── Observations (fact layer — immutable) ───────────────────────────

    def insert_observations(self, items: list[dict[str, Any]]) -> list[str]:
        """Insert observations + metadata. Returns observation IDs."""
        ids = []
        now = datetime.now(UTC).isoformat()
        for item in items:
            oid = str(uuid.uuid4())[:12]

            # Fact row — immutable
            self._db.insert("observations", {
                "id": oid,
                "content": item.get("content", ""),
                "source_quote": item.get("source_quote", ""),
                "referenced_date": item.get("referenced_date", ""),
                "observation_ts": item.get("observation_ts", now),
                "user_id": item.get("user_id", ""),
                "agent_id": item.get("agent_id", ""),
                "session_id": item.get("session_id", ""),
                "alignment_tier": item.get("alignment_tier", ""),
                "alignment_confidence": item.get("alignment_confidence", 0.0),
            })

            # Metadata row — initial enrichment from Observer
            mid = str(uuid.uuid4())[:12]
            self._db.insert("observation_metadata", {
                "id": mid,
                "observation_id": oid,
                "importance": item.get("importance", 0.5),
                "entities": json.dumps(item.get("entities", [])),
                "priority": item.get("priority", "medium"),
                "confidence": 1.0,
                "enrichment_ts": now,
            })

            ids.append(oid)
        return ids

    def _query_joined(
        self, table: str = "observations", where: str = "",
        params: tuple = (), order_by: str = "observation_ts DESC",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query observations LEFT JOIN metadata, returning flattened rows."""
        rows = self._db.raw_query(
            "SELECT o.*, m.importance, m.entities, m.priority, m.confidence "
            "FROM observations o "
            "LEFT JOIN observation_metadata m ON m.observation_id = o.id "
            + (f"WHERE {where} " if where else "")
            + f"ORDER BY o.{order_by} "
            + (f"LIMIT {limit}" if limit else ""),
            params,
        )
        return [dict(r) for r in rows]

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
            where_parts.append("o.observation_ts > ?")
            params.append(ts_after)
        if user_id:
            where_parts.append("o.user_id = ?")
            params.append(user_id)
        if session_id:
            where_parts.append("o.session_id = ?")
            params.append(session_id)
        if agent_id:
            where_parts.append("o.agent_id = ?")
            params.append(agent_id)
        if metadata:
            for k, v in metadata.items():
                where_parts.append(f"json_extract(o.source_quote, '$.{k}') = ?")
                params.append(str(v))
        where = " AND ".join(where_parts) if where_parts else ""
        return self._query_joined(where=where, params=tuple(params), limit=limit)

    def get_observations_since(
        self, last_id: str | None = None, limit: int = 500,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Cursor-based fetch — all observations after last_id."""
        where_parts: list[str] = []
        params: list[Any] = []

        if user_id:
            where_parts.append("o.user_id = ?")
            params.append(user_id)
        if session_id:
            where_parts.append("o.session_id = ?")
            params.append(session_id)
        if agent_id:
            where_parts.append("o.agent_id = ?")
            params.append(agent_id)
        if metadata:
            for k, v in metadata.items():
                where_parts.append(f"json_extract(o.source_quote, '$.{k}') = ?")
                params.append(str(v))

        if last_id:
            rows = self._db.raw_query(
                "SELECT observation_ts FROM observations WHERE id = ?", (last_id,),
            )
            if not rows:
                return []
            last_ts = rows[0]["observation_ts"]
            where_parts.append("o.observation_ts > ?")
            params.append(last_ts)

        where = " AND ".join(where_parts) if where_parts else ""
        return self._query_joined(where=where, params=tuple(params), limit=limit)

    def get_recent_observations(self, days: int = 30, limit: int = 50,
                                 user_id: str | None = None,
                                 session_id: str | None = None,
                                 agent_id: str | None = None) -> list[dict[str, Any]]:
        from datetime import timedelta
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
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
        cutoff = (datetime.now(UTC) - timedelta(days=half_life_days)).isoformat()  # noqa: F841
        rows = self._db.query("reflections", where="score > 0.1", limit=1000)
        count = 0
        for row in rows:
            new_score = float(row["score"]) * 0.9
            self._db.update("reflections", row["id"], {"score": new_score})
            count += 1
        return count
