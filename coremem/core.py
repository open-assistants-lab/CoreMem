"""MemoryCore — unified memory for AI agents.

Single HybridDB instance. Messages stored with turn_id for AgentJournal compilation.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from hybriddb import HybridDB

from coremem.agent_journal import (
    AgentJournalBundle,
    AgentJournalCompileResult,
    AgentJournalError,
    AgentJournalLLMCompiler,
    AgentJournalSearch,
    CrossEncoderReranker,
    dream,
    rebuild_index,
)
from coremem.agent_journal.llm_compiler import DEFAULT_AGENT_JOURNAL_MODEL
from coremem.heuristics import SearchHeuristics, _mmr_diversify
from coremem.query import LLMProvider, decompose_queries, expand_queries
from coremem.rerank import get_cross_encoder, rerank
from coremem.types import Memory, SearchQuery, SearchResult, SessionBundle

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
    if row.get("turn_id"):
        metadata.setdefault("turn_id", row["turn_id"])
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


def _matches_filters(
    mem: Memory,
    *,
    role: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    ts_after: str | None = None,
    ts_before: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    if role and mem.role != role:
        return False
    if session_id and mem.session_id != session_id:
        return False
    if user_id and mem.user_id != user_id:
        return False
    if agent_id and mem.agent_id != agent_id:
        return False
    if ts_after and (mem.ts is None or mem.ts.isoformat() <= ts_after):
        return False
    if ts_before and (mem.ts is None or mem.ts.isoformat() >= ts_before):
        return False
    if metadata:
        for key, value in metadata.items():
            if (mem.metadata or {}).get(key) != value:
                return False
    return True


def _row_to_search_result(row: dict[str, Any], score: float) -> SearchResult:
    return SearchResult(memory=_row_to_memory(row), score=score)


class MemoryCore:
    """Unified memory for AI agents. One HybridDB, AgentJournal for compilation.

    Usage:
        core = MemoryCore(path="./memory")
        tid = core.ingest("user", "I like coffee", session_id="s1")
        core.ingest("assistant", "Great!", session_id="s1")
        await core.compile_turn(turn_id=tid)
        results = core.recall("coffee")
        hits = core.search_journal("coffee")
    """

    def __init__(
        self,
        path: str,
        llm_provider: LLMProvider | None = None,
        agent_journal_model: str = DEFAULT_AGENT_JOURNAL_MODEL,
    ):
        self._db = HybridDB(path=path)
        self._heuristics = SearchHeuristics()
        self._llm_provider = llm_provider
        self._ensure_tables()
        workspace_root = Path(path).resolve().parent
        self._agent_journal_root = workspace_root / "agent_journal"
        self._agent_journal_bundle = AgentJournalBundle(self._agent_journal_root)
        self._journal_compiler = AgentJournalLLMCompiler(
            self._agent_journal_bundle,
            model=agent_journal_model,
        )
        self._reranker: CrossEncoderReranker | None = None
        self._agent_journal_search = AgentJournalSearch(
            self._agent_journal_root,
            reranker=None,
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
        if "journal_records" not in self._db.list_tables():
            self._db.create_table("journal_records", {
                "id": "TEXT PRIMARY KEY",
                "session_id": "TEXT NOT NULL",
                "content": "LONGTEXT",
                "compiled_at": "TEXT NOT NULL",
                "embedding": "TEXT",
            })
        self._db.raw_query(
            "CREATE INDEX IF NOT EXISTS idx_journal_records_session ON journal_records(session_id)"
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
        turn_id: str | None = None,
    ) -> str:
        if turn_id is None:
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

    def _search_messages(
        self, query: str, limit: int = 10,
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        has_filters = any((role, session_id, user_id, agent_id, ts_after, ts_before, metadata))
        hybrid_limit = max(limit * 20, 100) if has_filters else limit * 3
        rows = self._db.search("messages", "content", query, limit=hybrid_limit)
        results: list[SearchResult] = []
        for row in rows:
            mem = _row_to_memory(row)
            if not _matches_filters(
                mem,
                role=role,
                session_id=session_id,
                user_id=user_id,
                agent_id=agent_id,
                ts_after=ts_after,
                ts_before=ts_before,
                metadata=metadata,
            ):
                continue
            score = row.get("_score", row.get("score", 0.0))
            score = SearchHeuristics.apply_all(
                query=query,
                content=mem.content,
                score=score,
                ts=mem.ts.isoformat() if mem.ts else None,
            )
            results.append(SearchResult(memory=mem, score=score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _search_messages_llm_expansion(
        self,
        query: str,
        limit: int = 5,
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
        has_filters = any((role, session_id, user_id, agent_id, ts_after, ts_before, metadata))
        effective_limit = limit * depth
        if has_filters:
            effective_limit = max(effective_limit, limit * 20, 100)
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
                    if not _matches_filters(
                        mem,
                        role=role,
                        session_id=session_id,
                        user_id=user_id,
                        agent_id=agent_id,
                        ts_after=ts_after,
                        ts_before=ts_before,
                        metadata=metadata,
                    ):
                        continue
                    rid = mem.id
                    if rid and rid not in seen_ids:
                        seen_ids.add(rid)
                        ch = hash(mem.content[:200])
                        if ch not in seen_content:
                            seen_content.add(ch)
                            score = row.get("_score", row.get("score", 0.0))
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

    def _search_messages_decomposed(
        self,
        query: str,
        limit: int = 5,
        per_query_limit: int = 20,
        use_cross_encoder: bool = False,
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if limit <= 0 or per_query_limit <= 0:
            return []

        fused: dict[str, tuple[Memory, float]] = {}
        for query_index, variant in enumerate(decompose_queries(query)):
            weight = 2.0 if query_index == 0 else 1.0
            for rank, result in enumerate(
                self._search_messages(
                    variant, limit=per_query_limit,
                    role=role, session_id=session_id, user_id=user_id,
                    agent_id=agent_id, ts_after=ts_after, ts_before=ts_before,
                    metadata=metadata,
                ), start=1,
            ):
                memory_id = result.memory.id or ""
                if not memory_id:
                    continue
                memory, score = fused.get(memory_id, (result.memory, 0.0))
                fused[memory_id] = (memory, score + weight / (60 + rank))

        ranked = [
            SearchResult(memory=memory, score=score)
            for memory, score in fused.values()
        ]
        ranked.sort(key=lambda result: result.score, reverse=True)
        if use_cross_encoder:
            ranked = rerank(query, ranked)
        return _mmr_diversify(ranked, limit)

    def _reconstruct_sessions(
        self,
        query: str,
        session_limit: int = 5,
        max_context_chars: int = 16_000,
        short_max_messages: int = 12,
        short_max_chars: int = 8_000,
        segment_max_messages: int = 6,
        segment_max_chars: int = 4_000,
        primary_results: list[SearchResult] | None = None,
    ) -> list[SessionBundle]:
        primary = (
            primary_results
            if primary_results is not None
            else self._search_messages_decomposed(query, limit=session_limit)
        )
        bundles: list[SessionBundle] = []
        seen_sessions: set[str] = set()
        for result in primary:
            session_id = result.memory.session_id or ""
            if not session_id or session_id in seen_sessions:
                continue
            seen_sessions.add(session_id)
            rows = self._db.raw_query(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY ts ASC, rowid ASC",
                (session_id,),
            )
            messages = [_row_to_memory(row) for row in rows]
            total_chars = sum(len(message.content) for message in messages)
            if len(messages) <= short_max_messages and total_chars <= short_max_chars:
                bundles.append(SessionBundle(
                    session_id=session_id,
                    messages=messages,
                    score=result.score,
                    complete=True,
                    anchor_ids=[result.memory.id] if result.memory.id else [],
                ))
                continue

            segments: list[list[Memory]] = []
            segment: list[Memory] = []
            segment_chars = 0
            for message in messages:
                message_chars = len(message.content)
                if segment and (
                    len(segment) >= segment_max_messages
                    or segment_chars + message_chars > segment_max_chars
                ):
                    segments.append(segment)
                    segment = []
                    segment_chars = 0
                segment.append(message)
                segment_chars += message_chars
            if segment:
                segments.append(segment)

            anchor_index = next(
                (
                    index for index, candidate in enumerate(segments)
                    if any(message.id == result.memory.id for message in candidate)
                ),
                0,
            )
            selected_indexes = sorted({0, anchor_index})
            selected = [
                message
                for index in selected_indexes
                for message in segments[index]
            ]
            bundles.append(SessionBundle(
                session_id=session_id,
                messages=selected,
                score=result.score,
                complete=len(selected_indexes) == len(segments),
                anchor_ids=[result.memory.id] if result.memory.id else [],
            ))

        if not bundles or max_context_chars <= 0:
            return []
        per_bundle_budget = max_context_chars // len(bundles)
        budgeted: list[SessionBundle] = []
        for bundle in bundles:
            if sum(len(message.content) for message in bundle.messages) <= per_bundle_budget:
                budgeted.append(bundle)
                continue
            position = {message.id: index for index, message in enumerate(bundle.messages)}
            priority_ids = []
            if bundle.messages:
                priority_ids.append(bundle.messages[0].id)
            priority_ids.extend(bundle.anchor_ids)
            selected: list[Memory] = []
            selected_ids: set[str] = set()
            used_chars = 0
            by_id = {message.id: message for message in bundle.messages}
            for message_id in priority_ids:
                if message_id in selected_ids or message_id not in by_id:
                    continue
                message = by_id[message_id]
                message_chars = len(message.content)
                if used_chars + message_chars > per_bundle_budget:
                    continue
                selected.append(message)
                selected_ids.add(message_id)
                used_chars += message_chars
            for message in sorted(bundle.messages, key=lambda m: position.get(m.id, 0)):
                if message.id in selected_ids:
                    continue
                message_chars = len(message.content)
                if used_chars + message_chars > per_bundle_budget:
                    continue
                selected.append(message)
                selected_ids.add(message.id)
                used_chars += message_chars
            selected.sort(key=lambda message: position[message.id])
            budgeted.append(SessionBundle(
                session_id=bundle.session_id,
                messages=selected,
                score=bundle.score,
                complete=bundle.complete and len(selected) == len(bundle.messages),
                anchor_ids=bundle.anchor_ids,
            ))
        return budgeted

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
            timestamp = self._full_timestamp(rows[0].get("ts"))
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

    def _full_timestamp(self, ts: str | None) -> str | None:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return None

    async def dream(self) -> dict:
        return await dream(self._agent_journal_bundle)

    def rebuild_index(self) -> dict:
        return rebuild_index(self._agent_journal_bundle.root)

    # ── Context-Retrieval methods (PoC) ─────────────────────────

    def _store_journal_record(self, session_id: str, content: str) -> None:
        record_id = session_id
        self._db.raw_query(
            """INSERT OR REPLACE INTO journal_records (id, session_id, content, compiled_at)
               VALUES (?, ?, ?, ?)""",
            (record_id, session_id, content, datetime.now(UTC).isoformat()),
        )

    def recall(
        self,
        query: str,
        *,
        strategy: str = "episodic",
        limit: int = 5,
        bundles: bool = False,
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult] | list[SessionBundle]:
        if strategy == "direct":
            results = self._search_messages(
                query, limit=limit, role=role, session_id=session_id,
                user_id=user_id, agent_id=agent_id, ts_after=ts_after,
                ts_before=ts_before, metadata=metadata,
            )
            if bundles:
                return self._reconstruct_sessions(
                    query, session_limit=limit, primary_results=results,
                )
            return results

        if strategy == "expanded":
            results = self._search_messages_llm_expansion(
                query, limit=limit, role=role, session_id=session_id,
                user_id=user_id, agent_id=agent_id, ts_after=ts_after,
                ts_before=ts_before, metadata=metadata,
            )
            if bundles:
                return self._reconstruct_sessions(
                    query, session_limit=limit, primary_results=results,
                )
            return results

        if strategy == "fusion":
            results = self._search_with_fusion(query, limit=limit)
            if bundles:
                return self._reconstruct_sessions(
                    query, session_limit=limit, primary_results=results,
                )
            return results

        if strategy == "episodic":
            primary = self._search_messages_decomposed(
                query, limit=limit, per_query_limit=max(20, limit * 4),
                use_cross_encoder=True,
                role=role, session_id=session_id, user_id=user_id,
                agent_id=agent_id, ts_after=ts_after, ts_before=ts_before,
                metadata=metadata,
            )
            if bundles:
                return self._reconstruct_sessions(
                    query, session_limit=limit, primary_results=primary,
                )
            return primary

        raise ValueError(f"unknown strategy: {strategy}")

    def _search_with_fusion(
        self,
        query: str,
        limit: int = 5,
        per_query_limit: int = 20,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []

        mc_results = self._search_messages(query, limit=per_query_limit)
        er_results = self._search_messages_decomposed(
            query, limit=per_query_limit, per_query_limit=per_query_limit,
            use_cross_encoder=True,
        )

        fused: dict[str, tuple[Memory, float]] = {}
        for rank, result in enumerate(mc_results, start=1):
            memory_id = result.memory.id or ""
            if not memory_id:
                continue
            memory, score = fused.get(memory_id, (result.memory, 0.0))
            fused[memory_id] = (memory, score + 1.0 / (60 + rank))

        for rank, result in enumerate(er_results, start=1):
            memory_id = result.memory.id or ""
            if not memory_id:
                continue
            memory, score = fused.get(memory_id, (result.memory, 0.0))
            fused[memory_id] = (memory, score + 1.0 / (60 + rank))

        ranked = [
            SearchResult(memory=memory, score=score)
            for memory, score in fused.values()
        ]
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked[:limit]
