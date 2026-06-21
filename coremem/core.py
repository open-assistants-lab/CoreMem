"""MemoryCore — unified memory for AI agents.

Single HybridDB instance. Messages, observations, reflections all in one DB.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from hybriddb import HybridDB

from coremem.heuristics import SearchHeuristics, _mmr_diversify
from coremem.layers import WakeUpContext
from coremem.query import LLMProvider, expand_queries
from coremem.rerank import get_cross_encoder, rerank
from coremem.types import Memory, SearchQuery, SearchResult

_DEFAULT_SEARCH_DEPTH = 5

# ── Observation schemas (moved from memory_store.py) ──────────

_OBSERVATIONS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "kind": "TEXT NOT NULL DEFAULT 'fact'",
    "content": "LONGTEXT",
    "source_quote": "TEXT",
    "source_fact_ids": "TEXT NOT NULL DEFAULT '[]'",
    "source_message_ids": "TEXT DEFAULT '[]'",
    "referenced_date": "TEXT",
    "observation_ts": "TEXT NOT NULL",
    "user_id": "TEXT",
    "agent_id": "TEXT",
    "session_id": "TEXT",
    "alignment_tier": "TEXT",
    "alignment_confidence": "REAL",
    "importance": "REAL",
    "confidence": "REAL DEFAULT 0.800",
    "memory_type": "TEXT",
    "durability": "TEXT DEFAULT 'durable'",
    "sensitivity": "TEXT DEFAULT 'normal'",
    "status": "TEXT DEFAULT 'candidate'",
    "valid_from": "TEXT",
    "valid_to": "TEXT",
    "superseded_by": "TEXT",
    "entities": "TEXT NOT NULL DEFAULT '[]'",
    "reflected": "INTEGER NOT NULL DEFAULT 0",
    "embedding": "TEXT",
    "metadata": "TEXT DEFAULT '{}'",
}

_REFLECTIONS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "content": "LONGTEXT",
    "domain": "TEXT",
    "linked_observation_ids": "TEXT",
    "score": "REAL",
    "embedding": "TEXT",
    "observation_ts": "TEXT DEFAULT ''",
    "user_id": "TEXT",
}

_OBSERVATION_EVENTS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "observation_id": "TEXT NOT NULL",
    "event_type": "TEXT NOT NULL",
    "old_value": "TEXT",
    "new_value": "TEXT",
    "source_message_id": "TEXT",
    "created_at": "TEXT NOT NULL",
}

_OBSERVATION_CONFLICTS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "observation_id_a": "TEXT NOT NULL",
    "observation_id_b": "TEXT NOT NULL",
    "conflict_type": "TEXT NOT NULL",
    "resolution_status": "TEXT DEFAULT 'unresolved'",
    "created_at": "TEXT NOT NULL",
    "resolved_at": "TEXT",
}

# ── Ingest helpers (moved from ingest.py) ─────────────────────


def _ingest_message(
    db: HybridDB, role: str, content: str,
    session_id: str | None = None,
    user_id: str = "",
    agent_id: str = "",
    ts: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    embedding: list[float] | None = None,
) -> str:
    if not content.strip():
        return ""
    mid = str(uuid.uuid4())[:12]
    row = {
        "id": mid,
        "role": role,
        "content": content,
        "user_id": user_id,
        "agent_id": agent_id,
        "session_id": session_id or "",
        "metadata": json.dumps(metadata or {}),
        "ts": (ts or datetime.now(UTC)).isoformat(),
    }
    if embedding:
        row["embedding"] = json.dumps(embedding)
    db.insert("messages", row)
    return mid


def _ingest_batch(
    db: HybridDB,
    messages: list[dict],
    session_id: str | None = None,
) -> list[str]:
    ids = []
    for msg in messages:
        mid = _ingest_message(
            db=db,
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
            session_id=session_id,
            metadata=msg.get("metadata"),
        )
        if mid:
            ids.append(mid)
    return ids


def _row_to_memory(row: dict[str, Any]) -> Memory:
    ts = None
    if row.get("ts"):
        try:
            ts = datetime.fromisoformat(row["ts"])
        except (ValueError, TypeError):
            pass
    metadata = {}
    if row.get("metadata"):
        try:
            metadata = json.loads(row["metadata"])
        except (json.JSONDecodeError, TypeError):
            pass
    return Memory(
        id=row.get("id", ""),
        content=row.get("content", ""),
        role=row.get("role", "user"),
        ts=ts,
        session_id=row.get("session_id"),
        user_id=row.get("user_id"),
        agent_id=row.get("agent_id"),
        metadata=metadata,
    )


def _row_to_search_result(row: dict[str, Any], score: float) -> SearchResult:
    return SearchResult(memory=_row_to_memory(row), score=score)


# ── MemoryCore ──────────────────────────────────────────────


class MemoryCore:
    """Unified memory for AI agents. One HybridDB, all tables.

    Usage:
        core = MemoryCore(path="./memory", enable_observations=True)
        core.ingest("user", "I like coffee")
        results = core.search("coffee")
        obs = core.observations("coffee preferences")
        refs = core.reflections()
    """

    def __init__(
        self,
        path: str,
        llm_provider: LLMProvider | None = None,
        enable_observations: bool = False,
        enable_reflections: bool = False,
        observation_model: str = "deepseek:deepseek-v4-flash",
        reflect_model: str = "openai:gpt-4o",
        observation_kwargs: dict[str, Any] | None = None,
        reflect_kwargs: dict[str, Any] | None = None,
    ):
        self._db = HybridDB(path=path)
        self._heuristics = SearchHeuristics()
        self._wakeup = WakeUpContext(self._db)
        self._llm_provider = llm_provider
        self._enable_observations = enable_observations
        self._enable_reflections = enable_reflections
        self._enable_tool_extractor = enable_observations  # gates by same flag
        self._observation_model = observation_model
        self._reflect_model = reflect_model
        self._tool_min_messages: int = 5
        self._observer_pipeline: Any = None
        self._reflector_pipeline: Any = None
        self._ensure_tables()
        if enable_observations or enable_reflections:
            self._ensure_observation_tables()
        if enable_observations:
            from coremem.observer import ObserverPipeline
            kwargs = dict(observation_kwargs or {})
            kwargs.setdefault("session_id", "")
            self._observer_pipeline = ObserverPipeline(
                memory=self,
                observation_model=observation_model,
                **kwargs,
            )
        if enable_reflections:
            from coremem.reflector import ReflectorPipeline
            self._reflector_pipeline = ReflectorPipeline(
                memory=self,
                reflect_model=reflect_model,
                **(reflect_kwargs or {}),
            )

    def _ensure_tables(self) -> None:
        if "messages" not in self._db.list_tables():
            self._db.create_table("messages", {
                "id": "TEXT PRIMARY KEY",
                "role": "TEXT NOT NULL",
                "content": "LONGTEXT",
                "user_id": "TEXT DEFAULT ''",
                "agent_id": "TEXT DEFAULT ''",
                "session_id": "TEXT DEFAULT ''",
                "metadata": "TEXT DEFAULT '{}'",
                "ts": "TEXT",
                "embedding": "TEXT",
            })

    def _ensure_observation_tables(self) -> None:
        if "observations" not in self._db.list_tables():
            self._db.create_table("observations", _OBSERVATIONS_SCHEMA)
            for idx, col in [("kind", "kind"), ("user_id", "user_id"), ("session_id", "session_id"), ("reflected", "reflected"), ("importance", "importance")]:
                self._db.raw_query(
                    f"CREATE INDEX IF NOT EXISTS idx_observations_{idx} ON observations({col})"
                )
        if "observation_events" not in self._db.list_tables():
            self._db.create_table("observation_events", _OBSERVATION_EVENTS_SCHEMA)
        if "observation_conflicts" not in self._db.list_tables():
            self._db.create_table("observation_conflicts", _OBSERVATION_CONFLICTS_SCHEMA)
        if "reflections" not in self._db.list_tables():
            self._db.create_table("reflections", _REFLECTIONS_SCHEMA)

        # v0.8.0 migration: add metadata to existing observations table
        cols = {r["name"] for r in self._db.raw_query("PRAGMA table_info(observations)")}
        if "metadata" not in cols:
            self._db.raw_query("ALTER TABLE observations ADD COLUMN metadata TEXT DEFAULT '{}'")

        # v0.9.0 migration: add observation_ts to existing reflections table
        ref_cols = {r["name"] for r in self._db.raw_query("PRAGMA table_info(reflections)")}
        if "observation_ts" not in ref_cols:
            self._db.raw_query("ALTER TABLE reflections ADD COLUMN observation_ts TEXT DEFAULT ''")

        # v0.10.0 migration: run schema migrations for upgraded databases
        self._run_schema_migrations()

    def _run_schema_migrations(self) -> None:
        """Run forward-compatible schema migrations on existing databases."""
        # Add missing observation columns (v0.6+ schema additions)
        obs_cols = {r["name"] for r in self._db.raw_query("PRAGMA table_info(observations)")}
        for col_name, col_def in [
            ("source_message_ids", "TEXT DEFAULT '[]'"),
            ("confidence", "REAL DEFAULT 0.800"),
            ("memory_type", "TEXT"),
            ("durability", "TEXT DEFAULT 'durable'"),
            ("sensitivity", "TEXT DEFAULT 'normal'"),
            ("status", "TEXT DEFAULT 'candidate'"),
            ("valid_from", "TEXT"),
            ("valid_to", "TEXT"),
            ("superseded_by", "TEXT"),
            ("embedding", "TEXT"),
        ]:
            if col_name not in obs_cols:
                try:
                    self._db.raw_query(f"ALTER TABLE observations ADD COLUMN {col_name} {col_def}")
                except Exception:
                    pass

        # Create observation_events table if missing
        tables = self._db.list_tables()
        if "observation_events" not in tables:
            self._db.create_table("observation_events", _OBSERVATION_EVENTS_SCHEMA)
            self._db.raw_query(
                "CREATE INDEX IF NOT EXISTS idx_observation_events_obs "
                "ON observation_events(observation_id)"
            )

        # Create observation_conflicts table if missing
        if "observation_conflicts" not in tables:
            self._db.create_table("observation_conflicts", _OBSERVATION_CONFLICTS_SCHEMA)
            self._db.raw_query(
                "CREATE INDEX IF NOT EXISTS idx_observation_conflicts_status "
                "ON observation_conflicts(resolution_status)"
            )

        # Create reflections table if missing
        if "reflections" not in tables:
            self._db.create_table("reflections", _REFLECTIONS_SCHEMA)

    @property
    def db(self) -> HybridDB:
        return self._db

    def warmup(self) -> None:
        get_cross_encoder()

    def ingest(
        self, role: str, content: str,
        session_id: str | None = None,
        user_id: str = "",
        agent_id: str = "",
        ts: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        return _ingest_message(
            db=self._db, role=role, content=content,
            session_id=session_id, user_id=user_id, agent_id=agent_id,
            ts=ts, metadata=metadata, embedding=embedding,
        )

    def ingest_many(self, messages: list[dict[str, Any]], session_id: str | None = None) -> list[str]:
        return _ingest_batch(db=self._db, messages=messages, session_id=session_id)

    def search(
        self, query: str, limit: int = 10,
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        hybrid_limit = limit * 3
        rows = self._db.search("messages", "content", query, limit=hybrid_limit)
        results: list[SearchResult] = []
        for row in rows:
            mem = _row_to_memory(row)
            score = row.get("score", 0.0)
            score = SearchHeuristics.apply_all(
                query=query,
                content=mem.content,
                score=score,
                ts=mem.ts.isoformat() if mem.ts else None,
            )
            results.append(SearchResult(memory=mem, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def search_enhanced(
        self, query: str, limit: int = 10,
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
        depth: int = _DEFAULT_SEARCH_DEPTH,
    ) -> list[SearchResult]:
        queries = expand_queries(query, llm_provider=self._llm_provider)
        if SearchHeuristics.is_counting_question(query):
            depth = max(depth, 10)
        elif any(cue in query.lower() for cue in ("before", "after", "since", "when did", "what year")):
            depth = max(depth, 7)
        effective_limit = limit * depth
        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()
        seen_content: set[int] = set()
        for q in queries:
            rows = self._db.search("messages", "content", q, limit=effective_limit)
            if rows:
                max_score = max(r.get("score", 0) for r in rows)
                min_score = min(r.get("score", 0) for r in rows)
                score_range = max_score - min_score
                for row in rows:
                    mem = _row_to_memory(row)
                    rid = mem.id
                    if rid and rid not in seen_ids:
                        seen_ids.add(rid)
                        ch = hash(mem.content[:200])
                        if ch not in seen_content:
                            seen_content.add(ch)
                            score = row.get("score", 0.0)
                            if score_range > 0:
                                score = (score - min_score) / score_range
                            all_results.append(SearchResult(memory=mem, score=score))
        for r in all_results:
            r.score = SearchHeuristics.apply_all(
                query=query, content=r.memory.content, score=r.score,
                ts=r.memory.ts.isoformat() if r.memory.ts else None,
            )
        all_results.sort(key=lambda r: r.score, reverse=True)
        all_results = _mmr_diversify(all_results, effective_limit)
        all_results = rerank(query, all_results)
        return all_results[:limit]

    def wake_up(self, user_id: str = "default", session_id: str | None = None) -> str:
        context = self._wakeup.essential(user_id=user_id)
        if session_id:
            l2 = self._wakeup.session(session_id=session_id)
            if l2:
                context += "\n\n" + l2
        return context

    def deep_search_context(self, query: str, limit: int = 10) -> str | None:
        return self._wakeup.deep_search(query=query, limit=limit)

    def fetch(
        self,
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[Memory]:
        where_parts: list[str] = []
        params: list[Any] = []
        if role:
            where_parts.append("role = ?")
            params.append(role)
        if session_id:
            where_parts.append("session_id = ?")
            params.append(session_id)
        if user_id:
            where_parts.append("user_id = ?")
            params.append(user_id)
        if agent_id:
            where_parts.append("agent_id = ?")
            params.append(agent_id)
        if ts_after:
            where_parts.append("ts > ?")
            params.append(ts_after)
        if ts_before:
            where_parts.append("ts < ?")
            params.append(ts_before)
        if metadata:
            for k, v in metadata.items():
                where_parts.append(f"json_extract(metadata, '$.{k}') = ?")
                params.append(v)
        where = " AND ".join(where_parts) if where_parts else "1=1"
        rows = self._db.raw_query(
            f"SELECT * FROM messages WHERE {where} ORDER BY ts DESC LIMIT ? OFFSET ?",
            tuple(params) + (limit, offset),
        )
        return [_row_to_memory(r) for r in rows]

    def fetch_all(
        self,
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[Memory]:
        return self.fetch(
            role=role, session_id=session_id, user_id=user_id, agent_id=agent_id,
            ts_after=ts_after, ts_before=ts_before,
            metadata=metadata, limit=10_000, offset=0,
        )

    def store(self, memories: list[Memory]) -> list[str]:
        ids = []
        for mem in memories:
            row = {
                "id": mem.id or str(uuid.uuid4())[:12],
                "role": mem.role,
                "content": mem.content,
                "user_id": mem.user_id or "",
                "agent_id": mem.agent_id or "",
                "session_id": mem.session_id or "",
                "metadata": json.dumps(mem.metadata or {}),
                "ts": mem.ts.isoformat() if mem.ts else datetime.now(UTC).isoformat(),
            }
            self._db.insert("messages", row)
            ids.append(row["id"])
        return ids

    def count(self) -> int:
        rows = self._db.raw_query("SELECT COUNT(*) as c FROM messages")
        return rows[0]["c"] if rows else 0

    def delete(
        self,
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        where_parts: list[str] = []
        params: list[Any] = []
        if role:
            where_parts.append("role = ?")
            params.append(role)
        if session_id:
            where_parts.append("session_id = ?")
            params.append(session_id)
        if user_id:
            where_parts.append("user_id = ?")
            params.append(user_id)
        if agent_id:
            where_parts.append("agent_id = ?")
            params.append(agent_id)
        if ts_after:
            where_parts.append("ts > ?")
            params.append(ts_after)
        if ts_before:
            where_parts.append("ts < ?")
            params.append(ts_before)
        where = " AND ".join(where_parts) if where_parts else "1=1"
        before = self._db.raw_query("SELECT COUNT(*) AS c FROM messages")
        self._db.raw_query(f"DELETE FROM messages WHERE {where}", tuple(params))
        after = self._db.raw_query("SELECT COUNT(*) AS c FROM messages")
        return (before[0]["c"] - after[0]["c"]) if before and after else 0

    def clear(self) -> None:
        self._db.raw_query("DELETE FROM messages")

    # ── Observation methods (moved from MemoryStore) ───────────

    def _check_observations_enabled(self) -> None:
        if not self._enable_observations:
            raise RuntimeError(
                "Observation methods require enable_observations=True on MemoryCore"
            )

    def insert_observations(self, items: list[dict[str, Any]]) -> list[str]:
        self._check_observations_enabled()
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
            obs_metadata = item.get("metadata", "{}")
            if not isinstance(obs_metadata, str):
                obs_metadata = json.dumps(obs_metadata)
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
                "metadata": obs_metadata,
            })
        return ids

    async def session_end(
        self, session_id: str, user_id: str,
        active_skills: list[str] | None = None,
        min_tool_messages: int | None = None,
    ) -> None:
        """Called when a session ends. Triggers ToolExtractor.

        Wrap in ``asyncio.create_task()`` to fire-and-forget from the caller.

        Args:
            session_id: Session identifier.
            user_id: User identifier.
            active_skills: Opaque list of skill names loaded during session.
                           CoreMem stores them but has no knowledge of them.
        """
        if not self._enable_tool_extractor:
            return
        from coremem.tool_extractor import ToolExtractor

        extractor = ToolExtractor(
            memory=self,
            session_id=session_id,
            user_id=user_id,
            min_tool_messages=min_tool_messages if min_tool_messages is not None else self._tool_min_messages,
            active_skills=active_skills,
        )
        await extractor.extract()

    def get_observations(
        self, ts_after: str | None = None, limit: int = 50,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self._check_observations_enabled()
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
                params.append(v)
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
        self._check_observations_enabled()
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
                params.append(v)
        if last_id:
            rows = self._db.raw_query(
                "SELECT observation_ts FROM observations WHERE id = ?", (last_id,),
            )
            if not rows:
                return []
            last_ts = rows[0]["observation_ts"]
            where_parts.append("observation_ts >= ?")
            params.append(last_ts)
            where_parts.append("id != ?")
            params.append(last_id)
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
        self._check_observations_enabled()
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        return self.get_observations(
            ts_after=cutoff, limit=limit,
            user_id=user_id, session_id=session_id, agent_id=agent_id,
        )

    def search_observations(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        self._check_observations_enabled()
        results = self._db.search("observations", "content", query, limit=limit)
        return [dict(r) for r in results]

    # ── Primary observation/reflection API ─────────────────────

    def observations(self, query: str | None = None, limit: int = 10,
                     **kwargs: Any) -> list[dict[str, Any]]:
        """Search or list stored observations.

        Args:
            query: Semantic search query. If None, returns recent observations.
            limit: Max results.
            **kwargs: Passed to get_observations() when no query given.
        """
        if query:
            results = self.search_observations(query, limit=limit)
            meta = kwargs.get("metadata")
            if meta:
                results = [r for r in results if json.loads(r.get("metadata", "{}")) == meta]
            return results
        return self.get_observations(limit=limit, **kwargs)

    def reflections(self, query: str | None = None, limit: int = 10,
                    **kwargs: Any) -> list[dict[str, Any]]:
        """Search or list stored reflections.

        Args:
            query: Semantic search query. If None, returns recent reflections.
            limit: Max results.
            **kwargs: Passed to get_reflections() when no query given.
        """
        if query:
            return self.search_reflections(query, limit=limit)
        return self.get_reflections(limit=limit, **kwargs)

    def get_pending_reflections(self) -> list[dict[str, Any]]:
        self._check_observations_enabled()
        rows = self._db.raw_query(
            "SELECT * FROM observations "
            "WHERE kind = 'fact' AND reflected = 0 "
            "ORDER BY observation_ts DESC"
        )
        return [dict(r) for r in rows]

    def mark_reflected(self, observation_ids: list[str]) -> None:
        self._check_observations_enabled()
        if not observation_ids:
            return
        placeholders = ",".join("?" * len(observation_ids))
        self._db.raw_query(
            f"UPDATE observations SET reflected = 1 WHERE id IN ({placeholders})",
            tuple(observation_ids),
        )

    def insert_reflections(self, items: list[dict[str, Any]]) -> list[str]:
        self._check_observations_enabled()
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
                "observation_ts": item.get("observation_ts", datetime.now(UTC).isoformat()),
                "user_id": item.get("user_id", ""),
            })
            ids.append(rid)
        return ids

    def get_reflections(self, limit: int = 10,
                        user_id: str | None = None) -> list[dict[str, Any]]:
        self._check_observations_enabled()
        where_parts: list[str] = []
        params: list[Any] = []
        if user_id:
            where_parts.append("user_id = ?")
            params.append(user_id)
        where = " AND ".join(where_parts) if where_parts else ""
        return self._db.query("reflections", where=where, params=tuple(params),
                              order_by="score DESC", limit=limit)

    def search_reflections(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        self._check_observations_enabled()
        results = self._db.search("reflections", "content", query, limit=limit)
        return [dict(r) for r in results]

    # ── Delete methods ───────────────────────────────────────────

    def delete_observations(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        kind: str | None = None,
        status: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
    ) -> int:
        self._check_observations_enabled()
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
        if kind:
            where_parts.append("kind = ?")
            params.append(kind)
        if status:
            where_parts.append("status = ?")
            params.append(status)
        if ts_after:
            where_parts.append("observation_ts > ?")
            params.append(ts_after)
        if ts_before:
            where_parts.append("observation_ts < ?")
            params.append(ts_before)
        if metadata:
            for k, v in metadata.items():
                where_parts.append(f"json_extract(metadata, '$.{k}') = ?")
                params.append(v)
        where = " AND ".join(where_parts) if where_parts else "1=1"
        rows = self._db.raw_query(f"SELECT id FROM observations WHERE {where}", tuple(params))
        ids = [r["id"] for r in rows]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        self._db.raw_query(
            f"DELETE FROM observation_events WHERE observation_id IN ({placeholders})",
            tuple(ids),
        )
        self._db.raw_query(
            f"DELETE FROM observation_conflicts WHERE observation_id_a IN ({placeholders}) OR observation_id_b IN ({placeholders})",
            tuple(ids) + tuple(ids),
        )
        self._db.raw_query(f"DELETE FROM observations WHERE id IN ({placeholders})", tuple(ids))
        return len(ids)

    def delete_observations_by_id(self, ids: list[str]) -> int:
        self._check_observations_enabled()
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        self._db.raw_query(
            f"DELETE FROM observation_events WHERE observation_id IN ({placeholders})",
            tuple(ids),
        )
        self._db.raw_query(
            f"DELETE FROM observation_conflicts WHERE observation_id_a IN ({placeholders}) OR observation_id_b IN ({placeholders})",
            tuple(ids) + tuple(ids),
        )
        self._db.raw_query(f"DELETE FROM observations WHERE id IN ({placeholders})", tuple(ids))
        return len(ids)

    def delete_reflections(
        self,
        user_id: str | None = None,
        domain: str | None = None,
    ) -> int:
        self._check_observations_enabled()
        where_parts: list[str] = []
        params: list[Any] = []
        if user_id:
            where_parts.append("user_id = ?")
            params.append(user_id)
        if domain:
            where_parts.append("domain = ?")
            params.append(domain)
        where = " AND ".join(where_parts) if where_parts else "1=1"
        self._db.raw_query(f"DELETE FROM reflections WHERE {where}", tuple(params))
        return self._db.raw_query("SELECT changes() AS c")[0]["c"]

    def delete_reflections_by_id(self, ids: list[str]) -> int:
        self._check_observations_enabled()
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        self._db.raw_query(f"DELETE FROM reflections WHERE id IN ({placeholders})", tuple(ids))
        return len(ids)

    def update_reflections(self, ref_id: str, updates: dict[str, Any]) -> None:
        self._check_observations_enabled()
        if not updates:
            return
        set_parts = [f"{k} = ?" for k in updates]
        values = list(updates.values()) + [ref_id]
        self._db.raw_query(
            f"UPDATE reflections SET {', '.join(set_parts)} WHERE id = ?",
            tuple(values),
        )

    def apply_decay(self, half_life_days: int = 30) -> int:
        self._check_observations_enabled()
        cutoff = (datetime.now(UTC) - timedelta(days=half_life_days)).isoformat()
        rows = self._db.query("reflections", where="score > 0.1 AND (observation_ts = '' OR observation_ts < ?)", params=(cutoff,), limit=1000)
        count = 0
        for row in rows:
            new_score = float(row["score"]) * 0.9
            self._db.update("reflections", row["id"], {"score": new_score})
            count += 1
        return count

    def get_candidates(
        self, content: str, user_id: str | None = None, limit: int = 5,
    ) -> list[dict[str, Any]]:
        self._check_observations_enabled()
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
        self._check_observations_enabled()
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
        observation_id: str,
        event_type: str,
        old_value: str | None = None,
        new_value: str | None = None,
        source_message_id: str | None = None,
    ) -> str:
        self._check_observations_enabled()
        eid = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        self._db.insert("observation_events", {
            "id": eid,
            "observation_id": observation_id,
            "event_type": event_type,
            "old_value": old_value or "",
            "new_value": new_value or "",
            "source_message_id": source_message_id or "",
            "created_at": now,
        })
        return eid

    def create_conflict(
        self, observation_id_a: str, observation_id_b: str, conflict_type: str,
    ) -> str:
        self._check_observations_enabled()
        cid = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        self._db.insert("observation_conflicts", {
            "id": cid,
            "observation_id_a": observation_id_a,
            "observation_id_b": observation_id_b,
            "conflict_type": conflict_type,
            "resolution_status": "unresolved",
            "created_at": now,
            "resolved_at": "",
        })
        return cid