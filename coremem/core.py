"""MemoryCore — unified memory for AI agents.

Single HybridDB instance. Messages stored with turn_id for AgentJournal compilation.
"""

from __future__ import annotations

import json
import hashlib
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

from coremem.agent_journal import (
    AgentJournalBundle,
    AgentJournalCompileResult,
    AgentJournalError,
    AgentJournalLLMCompiler,
    AgentJournalSearch,
    CrossEncoderReranker,
    SearchHit,
    dream,
    rebuild_index,
)

_DEFAULT_SEARCH_DEPTH = 5


def _ingest_message(
    db: HybridDB, role: str, content: str,
    session_id: str | None = None,
    user_id: str = "",
    agent_id: str = "",
    ts: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    embedding: list[float] | None = None,
    turn_id: str | None = None,
) -> str:
    if not content.strip():
        return ""
    mid = str(uuid.uuid4())[:12]
    tid = turn_id or str(uuid.uuid4())[:12]
    row = {
        "id": mid,
        "role": role,
        "content": content,
        "user_id": user_id,
        "agent_id": agent_id,
        "session_id": session_id or "",
        "turn_id": tid,
        "metadata": json.dumps(metadata or {}),
        "ts": (ts or datetime.now(UTC)).isoformat(),
    }
    if embedding:
        row["embedding"] = json.dumps(embedding)
    db.insert("messages", row)
    return tid


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


class MemoryCore:
    """Unified memory for AI agents. One HybridDB, AgentJournal for compilation.

    Usage:
        core = MemoryCore(path="./memory")
        tid = core.ingest("user", "I like coffee", session_id="s1")
        core.ingest("assistant", "Great!", session_id="s1")
        await core.compile_turn(turn_id=tid)
        results = core.search_messages("coffee")
        hits = core.search_journal("coffee")
    """

    def __init__(self, path: str, llm_provider: LLMProvider | None = None):
        self._db = HybridDB(path=path)
        self._heuristics = SearchHeuristics()
        self._wakeup = WakeUpContext(self._db)
        self._llm_provider = llm_provider
        self._ensure_tables()
        workspace_root = Path(path).resolve().parent
        self._agent_journal_root = workspace_root / "agent_journal"
        self._agent_journal_bundle = AgentJournalBundle(self._agent_journal_root)
        self._journal_compiler = AgentJournalLLMCompiler(self._agent_journal_bundle)
        self._reranker = CrossEncoderReranker()
        self._agent_journal_search = AgentJournalSearch(
            self._agent_journal_root,
            reranker=self._reranker,
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
                "turn_id": "TEXT DEFAULT ''",
                "metadata": "TEXT DEFAULT '{}'",
                "ts": "TEXT",
                "embedding": "TEXT",
            })
        cols = {r["name"] for r in self._db.raw_query("PRAGMA table_info(messages)")}
        if "turn_id" not in cols:
            self._db.raw_query("ALTER TABLE messages ADD COLUMN turn_id TEXT DEFAULT ''")
            self._db.raw_query("CREATE INDEX IF NOT EXISTS idx_messages_turn_id ON messages(turn_id)")
        self._db.raw_query(
            """
            CREATE TABLE IF NOT EXISTS compiled_turns (
                turn_id TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL,
                compiled_at TEXT NOT NULL,
                daily_path TEXT NOT NULL,
                message_count INTEGER NOT NULL
            )
            """
        )

    @property
    def db(self) -> HybridDB:
        return self._db

    def _get_last_turn_id(self, session_id: str) -> str | None:
        rows = self._db.raw_query(
            "SELECT turn_id FROM messages WHERE session_id = ? ORDER BY ts DESC LIMIT 1",
            [session_id],
        )
        return rows[0]["turn_id"] if rows else str(uuid.uuid4())[:12]

    def ingest(
        self, role: str, content: str,
        session_id: str | None = None,
        user_id: str = "",
        agent_id: str = "",
        ts: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        if role == "user":
            turn_id = str(uuid.uuid4())[:12]
        else:
            turn_id = self._get_last_turn_id(session_id or "")
        _ingest_message(
            db=self._db, role=role, content=content,
            session_id=session_id, user_id=user_id, agent_id=agent_id,
            ts=ts, metadata=metadata, embedding=embedding,
            turn_id=turn_id,
        )
        return turn_id

    def ingest_turn(self, messages: list[dict], session_id: str | None = None) -> str:
        tid = str(uuid.uuid4())[:12]
        for msg in messages:
            _ingest_message(
                db=self._db,
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                session_id=session_id,
                metadata=msg.get("metadata"),
                turn_id=tid,
            )
        return tid

    def search_messages(
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

    def search_messages_deep(
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
                "turn_id": "",
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

    # ── AgentJournal methods ───────────────────────────────────

    async def compile_turn(
        self,
        turn_id: str,
        timestamp: str | None = None,
        title: str | None = None,
        *,
        force: bool = False,
    ) -> AgentJournalCompileResult | None:
        rows = self._turn_rows(turn_id)
        if not rows:
            return None
        source_hash = self._turn_source_hash(rows)
        compiled = self._compiled_turn(turn_id)
        if compiled is not None:
            if compiled["source_hash"] == source_hash and not force:
                return None
            if not force:
                raise AgentJournalError(
                    f"turn changed after compilation: {turn_id}; pass force=True to append a new section"
                )
        session_id = rows[0]["session_id"]
        if timestamp is None:
            timestamp = self._display_timestamp(rows[0].get("ts"))
        result = await self._journal_compiler.compile_session(
            turn_id=turn_id,
            session_id=session_id,
            messages=self._turn_messages(rows),
            timestamp=timestamp,
            title=title,
        )
        self._record_compiled_turn(turn_id, source_hash, result, len(rows))
        return result

    async def compile_latest_turn(
        self,
        session_id: str,
        timestamp: str | None = None,
        title: str | None = None,
        *,
        force: bool = False,
    ) -> AgentJournalCompileResult | None:
        rows = self._db.raw_query(
            """
            SELECT turn_id FROM messages
            WHERE session_id = ? AND turn_id != ''
            GROUP BY turn_id
            ORDER BY MAX(ts) DESC
            LIMIT 1
            """,
            (session_id,),
        )
        if not rows:
            return None
        return await self.compile_turn(
            rows[0]["turn_id"],
            timestamp=timestamp,
            title=title,
            force=force,
        )

    async def compile_uncompiled_turns(
        self,
        *,
        session_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, object]:
        summary: dict[str, object] = {
            "compiled": [],
            "skipped": [],
            "changed": [],
            "errors": [],
        }
        if limit <= 0:
            return summary
        where = "turn_id != ''"
        params: list[Any] = []
        if session_id is not None:
            where += " AND session_id = ?"
            params.append(session_id)
        turn_rows = self._db.raw_query(
            f"""
            SELECT turn_id FROM messages
            WHERE {where}
            GROUP BY turn_id
            ORDER BY MIN(ts) ASC
            LIMIT ?
            """,
            tuple(params) + (limit,),
        )
        for row in turn_rows:
            turn_id = row["turn_id"]
            rows = self._turn_rows(turn_id)
            if not rows:
                summary["skipped"].append(turn_id)  # type: ignore[index, union-attr]
                continue
            source_hash = self._turn_source_hash(rows)
            compiled = self._compiled_turn(turn_id)
            if compiled is not None:
                if compiled["source_hash"] == source_hash:
                    summary["skipped"].append(turn_id)  # type: ignore[index, union-attr]
                else:
                    summary["changed"].append(turn_id)  # type: ignore[index, union-attr]
                continue
            try:
                result = await self.compile_turn(turn_id)
            except Exception as exc:
                summary["errors"].append({"turn_id": turn_id, "error": str(exc)})  # type: ignore[index, union-attr]
            else:
                if result is None:
                    summary["skipped"].append(turn_id)  # type: ignore[index, union-attr]
                else:
                    summary["compiled"].append(turn_id)  # type: ignore[index, union-attr]
        return summary

    def _turn_rows(self, turn_id: str) -> list[dict[str, Any]]:
        return self._db.raw_query(
            "SELECT id, role, content, session_id, ts FROM messages WHERE turn_id = ? ORDER BY ts",
            (turn_id,),
        )

    def _turn_messages(self, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
        return [{"message_id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]

    def _turn_source_hash(self, rows: list[dict[str, Any]]) -> str:
        payload = [
            {
                "id": row.get("id", ""),
                "role": row.get("role", ""),
                "content": row.get("content", ""),
                "session_id": row.get("session_id", ""),
                "ts": row.get("ts", ""),
            }
            for row in rows
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def _compiled_turn(self, turn_id: str) -> dict[str, Any] | None:
        rows = self._db.raw_query(
            "SELECT * FROM compiled_turns WHERE turn_id = ?",
            (turn_id,),
        )
        return rows[0] if rows else None

    def _record_compiled_turn(
        self,
        turn_id: str,
        source_hash: str,
        result: AgentJournalCompileResult,
        message_count: int,
    ) -> None:
        daily_path = str(result.written_pages[0]) if result.written_pages else ""
        self._db.raw_query(
            """
            INSERT INTO compiled_turns (turn_id, source_hash, compiled_at, daily_path, message_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(turn_id) DO UPDATE SET
                source_hash = excluded.source_hash,
                compiled_at = excluded.compiled_at,
                daily_path = excluded.daily_path,
                message_count = excluded.message_count
            """,
            (
                turn_id,
                source_hash,
                datetime.now(UTC).isoformat(),
                daily_path,
                message_count,
            ),
        )

    def _display_timestamp(self, ts: str | None) -> str | None:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts).strftime("%H:%M")
        except ValueError:
            return None

    async def dream(self) -> dict:
        return await dream(self._agent_journal_bundle)

    def rebuild_index(self) -> dict:
        return rebuild_index(self._agent_journal_bundle.root)

    def search_journal(self, query: str, limit: int = 5) -> list[SearchHit]:
        return self._agent_journal_search.search(query, limit=limit)
