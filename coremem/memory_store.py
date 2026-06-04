"""MemoryStore — observations and reflections backed by HybridDB."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hybriddb import HybridDB

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


_OBSERVATIONS_SCHEMA = {
    "id":              "TEXT PRIMARY KEY",
    "kind":            "TEXT NOT NULL DEFAULT 'fact'",
    "content":         "LONGTEXT",
    "source_quote":    "TEXT",
    "source_fact_ids": "TEXT NOT NULL DEFAULT '[]'",
    "source_message_ids": "TEXT DEFAULT '[]'",
    "referenced_date": "TEXT",
    "observation_ts":  "TEXT NOT NULL",
    "user_id":         "TEXT",
    "agent_id":        "TEXT",
    "session_id":      "TEXT",
    "alignment_tier":        "TEXT",
    "alignment_confidence":  "REAL",
    "importance":      "REAL",
    "confidence":      "REAL DEFAULT 0.800",
    "memory_type":     "TEXT",
    "durability":      "TEXT DEFAULT 'durable'",
    "sensitivity":     "TEXT DEFAULT 'normal'",
    "status":          "TEXT DEFAULT 'candidate'",
    "valid_from":      "TEXT",
    "valid_to":        "TEXT",
    "superseded_by":   "TEXT",
    "entities":        "TEXT NOT NULL DEFAULT '[]'",
    "reflected":       "INTEGER NOT NULL DEFAULT 0",
    "embedding":       "TEXT",
}

_MEMORY_EVENTS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "memory_id": "TEXT NOT NULL",
    "event_type": "TEXT NOT NULL",
    "old_value": "TEXT",
    "new_value": "TEXT",
    "source_message_id": "TEXT",
    "created_at": "TEXT NOT NULL",
}

_MEMORY_CONFLICTS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "memory_id_a": "TEXT NOT NULL",
    "memory_id_b": "TEXT NOT NULL",
    "conflict_type": "TEXT NOT NULL",
    "resolution_status": "TEXT DEFAULT 'unresolved'",
    "created_at": "TEXT NOT NULL",
    "resolved_at": "TEXT",
}


class MemoryStore:
    """Storage for observations and reflections.

    0.5.0 single-table schema: a single ``observations`` table holds both
    immutable facts and Reflector-written enrichment (importance, entities)
    plus reflection rows (kind='reflection'). The 0.4.0 split schema is
    auto-migrated on first access.

    Args:
        path: Directory for the HybridDB data.
        embedding_fn: Optional embedding function for semantic search.
    """

    def __init__(self, path: str, embedding_fn: Any = None):
        existing_db = Path(path) / "app.db"
        needs_migration = (
            self._detect_0_4_0_schema(existing_db) if existing_db.exists() else False
        )

        self._db = HybridDB(path=path, embedding_fn=embedding_fn)

        if needs_migration:
            self._migrate_to_0_5_0()
        else:
            self._ensure_tables()
            self._stamp_schema_version("0.5.0")

        # v0.5 → v0.6 migration detection
        columns = self._db.raw_query("PRAGMA table_info(observations)")
        col_names = {c["name"] for c in columns}
        if "memory_type" not in col_names:
            from coremem.migrations.v0_5_to_v0_6 import migrate
            migrate(self._db, str(path))

    @staticmethod
    def _detect_0_4_0_schema(db_path: Path) -> bool:
        """Check if the DB on disk has the 0.4.0 split schema (observation_metadata)."""
        import sqlite3
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                row = conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='observation_metadata'"
                ).fetchone()
                return row is not None
            finally:
                conn.close()
        except sqlite3.OperationalError:
            return False

    def _migrate_to_0_5_0(self) -> None:
        """Run the 0.4.0 -> 0.5.0 schema collapse.

        Idempotent: the migration script checks ``_schema_version`` and
        returns early if already at 0.5.0.
        """
        from coremem.migrations.v0_4_to_v0_5 import migrate
        migrate(self._db._db_path)

    def _stamp_schema_version(self, version: str) -> None:
        """Insert a row into ``_schema_version`` for the given version."""
        self._db.raw_query("""
            CREATE TABLE IF NOT EXISTS _schema_version (
                version TEXT PRIMARY KEY,
                migrated_at TEXT NOT NULL
            )
        """)
        self._db.raw_query(
            "INSERT OR IGNORE INTO _schema_version (version, migrated_at) VALUES (?, ?)",
            (version, datetime.now(UTC).isoformat()),
        )

    def _ensure_tables(self) -> None:
        """Create the 0.5.0 single-table schema if it doesn't exist."""
        if "observations" not in self._db.list_tables():
            self._db.create_table("observations", _OBSERVATIONS_SCHEMA)
            self._db.raw_query("CREATE INDEX IF NOT EXISTS idx_observations_kind ON observations(kind)")
            self._db.raw_query("CREATE INDEX IF NOT EXISTS idx_observations_user ON observations(user_id)")
            self._db.raw_query("CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(session_id)")
            self._db.raw_query("CREATE INDEX IF NOT EXISTS idx_observations_reflected ON observations(reflected)")
            self._db.raw_query("CREATE INDEX IF NOT EXISTS idx_observations_importance ON observations(importance)")

        if "reflections" not in self._db.list_tables():
            self._db.create_table("reflections", _REFLECTIONS_SCHEMA)

        if "memory_events" not in self._db.list_tables():
            self._db.create_table("memory_events", _MEMORY_EVENTS_SCHEMA)

        if "memory_conflicts" not in self._db.list_tables():
            self._db.create_table("memory_conflicts", _MEMORY_CONFLICTS_SCHEMA)

    # ── Observations (single-table 0.5.0) ───────────────────────────────

    def insert_observations(self, items: list[dict[str, Any]]) -> list[str]:
        """Insert observations into the single observations table.

        Returns the list of observation IDs (auto-generated if not provided).
        Each item dict may include any of the 0.5.0 fields:
        id, kind, content, source_quote, source_fact_ids, referenced_date,
        observation_ts, user_id, agent_id, session_id, alignment_tier,
        alignment_confidence, importance, entities, reflected.
        """
        now = datetime.now(UTC).isoformat()
        ids: list[str] = []
        for item in items:
            oid = item.get("id") or str(uuid.uuid4())[:12]
            ids.append(oid)

            entities = item.get("entities", [])
            if not isinstance(entities, str):
                entities = json.dumps(entities)
            source_fact_ids = item.get("source_fact_ids", [])
            if not isinstance(source_fact_ids, str):
                source_fact_ids = json.dumps(source_fact_ids)

            self._db.insert("observations", {
                "id": oid,
                "kind": item.get("kind", "fact"),
                "content": item.get("content", ""),
                "source_quote": item.get("source_quote"),
                "source_fact_ids": source_fact_ids,
                "source_message_ids": item.get("source_message_ids", "[]"),
                "referenced_date": item.get("referenced_date"),
                "observation_ts": item.get("observation_ts", now),
                "user_id": item.get("user_id"),
                "agent_id": item.get("agent_id"),
                "session_id": item.get("session_id"),
                "alignment_tier": item.get("alignment_tier"),
                "alignment_confidence": item.get("alignment_confidence"),
                "importance": item.get("importance"),
                "confidence": item.get("confidence", 0.800),
                "memory_type": item.get("memory_type", ""),
                "durability": item.get("durability", "durable"),
                "sensitivity": item.get("sensitivity", "normal"),
                "status": item.get("status", "candidate"),
                "valid_from": item.get("valid_from", ""),
                "valid_to": item.get("valid_to", ""),
                "superseded_by": item.get("superseded_by", ""),
                "entities": entities,
                "reflected": item.get("reflected", 0),
            })
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
                where_parts.append(f"json_extract(source_quote, '$.{k}') = ?")
                params.append(str(v))
        where = " AND ".join(where_parts) if where_parts else "1=1"
        sql = (
            f"SELECT * FROM observations WHERE {where} "
            f"ORDER BY observation_ts DESC LIMIT {int(limit)}"
        )
        rows = self._db.raw_query(sql, tuple(params))
        return [dict(r) for r in rows]

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
                where_parts.append(f"json_extract(source_quote, '$.{k}') = ?")
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

        where = " AND ".join(where_parts) if where_parts else "1=1"
        sql = (
            f"SELECT * FROM observations WHERE {where} "
            f"ORDER BY observation_ts DESC LIMIT {int(limit)}"
        )
        rows = self._db.raw_query(sql, tuple(params))
        return [dict(r) for r in rows]

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

    # ── Reflector helpers (0.5.0) ─────────────────────────────────────

    def get_pending_reflections(self) -> list[dict[str, Any]]:
        """Return observations that are facts (not reflections) and have
        not yet been processed by the Reflector.

        Sorted by observation_ts DESC (newest first) so the Reflector sees
        recent facts first.
        """
        rows = self._db.raw_query(
            "SELECT * FROM observations "
            "WHERE kind = 'fact' AND reflected = 0 "
            "ORDER BY observation_ts DESC"
        )
        return [dict(r) for r in rows]

    def mark_reflected(self, observation_ids: list[str]) -> None:
        """Mark observations as processed by the Reflector.

        Sets reflected=1 for the given IDs. The Reflector calls this
        after running pattern synthesis on the source facts.
        """
        if not observation_ids:
            return
        placeholders = ",".join("?" * len(observation_ids))
        self._db.raw_query(
            f"UPDATE observations SET reflected = 1 WHERE id IN ({placeholders})",
            tuple(observation_ids),
        )

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

    # ── Memory lifecycle (v0.6) ──────────────────────────────────────────

    def get_candidates(
        self, content: str, user_id: str | None = None, limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find potential duplicate/related memories by keyword overlap."""
        words = set(content.lower().split())
        if not words:
            return []

        candidates: list[dict[str, Any]] = []
        recent = self.get_recent_observations(days=30, limit=200)
        for obs in recent:
            if user_id and obs.get("user_id") != user_id:
                continue
            obs_words = set(obs.get("content", "").lower().split())
            overlap = words & obs_words
            min_len = min(len(words), len(obs_words))
            if min_len > 0 and len(overlap) >= 3 and len(overlap) / min_len > 0.5:
                candidates.append(obs)

        candidates.sort(key=lambda o: o.get("observation_ts", ""), reverse=True)
        return candidates[:limit]

    def update_observation(self, obs_id: str, updates: dict[str, Any]) -> None:
        """Update specific fields on an existing observation."""
        if not updates:
            return
        set_parts = [f"{k} = ?" for k in updates]
        values = list(updates.values()) + [obs_id]
        self._db.raw_query(
            f"UPDATE observations SET {', '.join(set_parts)} WHERE id = ?",
            tuple(values),
        )

    def insert_event(
        self,
        memory_id: str,
        event_type: str,
        old_value: str | None = None,
        new_value: str | None = None,
        source_message_id: str | None = None,
    ) -> str:
        """Log a memory lifecycle event."""
        eid = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        self._db.insert("memory_events", {
            "id": eid,
            "memory_id": memory_id,
            "event_type": event_type,
            "old_value": old_value or "",
            "new_value": new_value or "",
            "source_message_id": source_message_id or "",
            "created_at": now,
        })
        return eid

    def create_conflict(
        self, memory_id_a: str, memory_id_b: str, conflict_type: str,
    ) -> str:
        """Create a conflict record between two observations."""
        cid = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        self._db.insert("memory_conflicts", {
            "id": cid,
            "memory_id_a": memory_id_a,
            "memory_id_b": memory_id_b,
            "conflict_type": conflict_type,
            "resolution_status": "unresolved",
            "created_at": now,
            "resolved_at": "",
        })
        return cid
