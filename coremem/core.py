"""MemoryCore — unified memory for AI agents.

Single HybridDB instance. Messages stored with turn_id for AgentJournal compilation.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import UTC, datetime
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
from coremem.retrieval import _is_preference_query, search_messages_preference_union
from coremem.rerank import get_cross_encoder, rerank
from coremem.types import Memory, SearchResult, SessionBundle

# ── Batched embedding (ingest path) ────────────────────────────────────────
#
# HybridDB's journal flush embeds one document at a time through Chroma's
# per-call embedding function (~75 ms/doc on CPU). Batched SentenceTransformer
# encoding is ~15x faster with the same all-MiniLM-L6-v2 model. We prewarm a
# content→vector cache for every pending journal document and temporarily
# shadow ``db._get_embedding`` during the flush so the journal loop picks up
# the cached vectors instead of re-embedding per document.

_batch_embed_model: Any = None
_batch_embed_lock = threading.Lock()


def _batch_embedding_model() -> Any:
    global _batch_embed_model
    if _batch_embed_model is None:
        with _batch_embed_lock:
            if _batch_embed_model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    _batch_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
                except Exception:
                    _batch_embed_model = False
    return _batch_embed_model or None


def _batch_embed_texts(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """Batch-encode texts with the same model HybridDB's default embedding fn uses."""
    model = _batch_embedding_model()
    if model is None:
        raise RuntimeError("no embedding model available")
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return [[float(v) for v in row] for row in vectors]


def _flush_journal_batched(db: HybridDB, batch_limit: int = 5000) -> int:
    """Process HybridDB's pending journal with batched embedding.

    Returns the number of journal entries processed. Falls back to the
    stock per-document embedding path if batch encoding is unavailable.
    """
    with db._connect() as cur:
        pending = [
            dict(row)
            for row in cur.execute(
                "SELECT id, data FROM _journal WHERE status = 'pending' "
                "AND op IN ('add', 'update') LIMIT ?",
                (batch_limit,),
            ).fetchall()
        ]
    if not pending:
        return db.process_journal(limit=batch_limit)
    try:
        docs = [entry["data"] or "" for entry in pending]
        vectors = _batch_embed_texts(docs)
        cache = {doc: vec for doc, vec in zip(docs, vectors)}
    except Exception:
        return db.process_journal(limit=batch_limit)
    orig = db._get_embedding

    def _cached(text: str) -> list[float]:
        cached_vec = cache.get(text)
        if cached_vec is not None:
            return cached_vec
        return orig(text)

    db._get_embedding = _cached  # type: ignore[method-assign]
    try:
        return db.process_journal(limit=batch_limit)
    finally:
        db._get_embedding = orig  # type: ignore[method-assign]

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


def _anchor_first(messages: list[Memory], anchor_ids: list[str]) -> list[Memory]:
    """Order bundle messages with the retrieved evidence (anchors) first.

    Validated on the 500-question LongMemEval-S answer eval (LLM answer +
    LLM judge): evidence-first bundle formatting avoids needle-in-haystack
    failures — the same 15k-char context that scored 0/1 on a question
    scored 1/1 after reordering. Anchors keep their relative order; the
    remaining messages stay chronological.
    """
    if not anchor_ids:
        return messages
    anchors = set(anchor_ids)
    positions = {message.id: index for index, message in enumerate(messages)}
    return sorted(messages, key=lambda m: (m.id not in anchors, positions.get(m.id, 0)))


_DEFAULT_BUNDLE_BUDGET_CHARS = 4_000


class MemoryCore:
    """Unified memory for AI agents. One HybridDB, AgentJournal for compilation.

    Usage:
        core = MemoryCore(path="./memory")
        tid = core.ingest("user", "I like coffee", session_id="s1")
        core.ingest("assistant", "Great!", session_id="s1")
        await core.compile_turn(turn_id=tid)
        results = core.recall("coffee")
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
        self._closed = False
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
        """Store a single message. Returns the **turn_id**.

        Raises ValueError if ``content`` is empty/whitespace (a silent no-op
        hides storage failures). For bulk ingestion use :meth:`ingest_many`
        (returns message ids) — empty messages are skipped there.
        """
        self._ensure_open()
        if not content or not content.strip():
            raise ValueError("ingest requires non-empty content")
        if turn_id is None:
            if role == "user":
                turn_id = str(uuid.uuid4())[:12]
            else:
                # _get_last_turn_id can return '' (e.g. rows written by store()),
                # which must never become the stored/returned turn id.
                turn_id = self._get_last_turn_id(session_id or "") or str(uuid.uuid4())[:12]
        _ingest_message(
            db=self._db, role=role, content=content,
            session_id=session_id, user_id=user_id, agent_id=agent_id,
            ts=ts, metadata=metadata, embedding=embedding,
            turn_id=turn_id,
        )
        return turn_id

    def ingest_turn(self, messages: list[dict], session_id: str | None = None) -> str:
        """Store a conversation turn. Returns the **turn_id**.

        Messages with empty content are skipped.
        """
        self._ensure_open()
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

    def ingest_many(self, messages: list[dict]) -> list[str]:
        """Batch-ingest messages in one HybridDB journal flush.

        Each dict accepts the same keys as :meth:`ingest` (``role``,
        ``content``, ``session_id``, ``user_id``, ``agent_id``, ``ts``,
        ``metadata``, ``turn_id``, ``embedding``) plus an optional ``id``
        for callers that manage their own ids (e.g. eval harnesses).
        Turn assignment mirrors :meth:`ingest`: user messages open a new
        turn, assistant messages reuse the session's latest turn.

        Returns the list of inserted message ids.
        """
        self._ensure_open()
        rows: list[dict[str, Any]] = []
        ids: list[str] = []
        last_turn: dict[str, str] = {}
        for msg in messages:
            content = msg.get("content", "")
            if not content.strip():
                continue
            mid = msg.get("id") or str(uuid.uuid4())[:12]
            session_id = msg.get("session_id") or ""
            role = msg.get("role", "user")
            tid = msg.get("turn_id")
            if not tid:
                if role == "user":
                    tid = str(uuid.uuid4())[:12]
                else:
                    tid = last_turn.get(session_id) or self._get_last_turn_id(session_id) or str(uuid.uuid4())[:12]
                last_turn[session_id] = tid
            ts_value = msg.get("ts")
            ts_iso = (
                ts_value.isoformat()
                if hasattr(ts_value, "isoformat")
                else (ts_value or datetime.now(UTC).isoformat())
            )
            row: dict[str, Any] = {
                "id": mid,
                "role": role,
                "content": content,
                "user_id": msg.get("user_id", ""),
                "agent_id": msg.get("agent_id", ""),
                "session_id": session_id,
                "turn_id": tid,
                "metadata": json.dumps(msg.get("metadata") or {}),
                "ts": ts_iso,
            }
            embedding = msg.get("embedding")
            if embedding:
                row["embedding"] = (
                    embedding if isinstance(embedding, str) else json.dumps(embedding)
                )
            rows.append(row)
            ids.append(mid)
        if not rows:
            return []
        self._db.insert_batch("messages", rows, sync=False)
        _flush_journal_batched(self._db)
        return ids

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
                max_score = max(r.get("_score", r.get("score", 0.0)) for r in rows)
                min_score = min(r.get("_score", r.get("score", 0.0)) for r in rows)
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
        session_cap: int = 1,
        allocation: str = "global",
    ) -> list[SearchResult]:
        if limit <= 0 or per_query_limit <= 0:
            return []

        # Preference queries route through the per-variant union (validated
        # on S: +0.033 session recall on the 30 preference questions). The
        # union's non-preference fallback re-enters this method without
        # recursion (the routing check fails there).
        if _is_preference_query(query):
            return search_messages_preference_union(
                self, query, limit=limit, per_variant=40,
                role=role, session_id=session_id, user_id=user_id,
                agent_id=agent_id, ts_after=ts_after, ts_before=ts_before,
                metadata=metadata,
            )

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
            if session_cap > 1:
                return self._select_with_session_cap(
                    query, ranked, limit=limit, cap=session_cap,
                    allocation=allocation,
                    role=role, session_id=session_id, user_id=user_id,
                    agent_id=agent_id, ts_after=ts_after, ts_before=ts_before,
                    metadata=metadata,
                )
        return _mmr_diversify(ranked, limit)

    def _select_with_session_cap(
        self,
        query: str,
        ranked: list[SearchResult],
        *,
        limit: int = 5,
        cap: int = 2,
        allocation: str = "global",
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Final selection: cross-encoder score every message of the top
        ``limit`` sessions, then fill the top-``limit`` slots.

        The one-message-per-session MMR cap loses answers that live in a
        second message of an already-found session (question echo + answer
        detail, user fact + assistant confirmation). Re-scoring the full
        session lets those answers surface.

        ``allocation``:
          - "global" (default): top-``limit`` slots from the pool with up to
            ``cap`` per session. Best message recall (measured on S:
            +0.123 message_recall@5) but a second message of the top
            session can displace the 5th session (−0.059 session_recall@5).
          - "anchor": the top session gets two slots, the next sessions one
            each; the extra slot always comes from the strongest context.
        """
        if not ranked or limit <= 0:
            return []
        # Top ``limit`` sessions by each result's cross-encoder score.
        best: dict[str, tuple[float, SearchResult]] = {}
        for result in ranked:
            sid = result.memory.session_id or f"_no_session_{hash(result.memory.content)}"
            score = getattr(result, "_ce_score", result.score)
            if sid not in best or score > best[sid][0]:
                best[sid] = (score, result)
        top_sessions = [sid for sid, _ in sorted(best.items(), key=lambda kv: -kv[1][0])[:limit]]

        # Pool: full message list per real session, original result otherwise.
        # Long sessions are windowed around the anchor + the recent tail so the
        # cross-encoder batch stays bounded.
        pool: list[tuple[Memory, str]] = []
        for sid in top_sessions:
            if sid.startswith("_no_session_"):
                memory = best[sid][1].memory
                pool.append((memory, sid))
                continue
            rows = self._db.raw_query(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY ts ASC, rowid ASC",
                (sid,),
            )
            messages = [
                m for m in (_row_to_memory(row) for row in rows)
                if _matches_filters(
                    m,
                    role=role, session_id=session_id, user_id=user_id,
                    agent_id=agent_id, ts_after=ts_after,
                    ts_before=ts_before, metadata=metadata,
                )
            ]
            if len(messages) > 60:
                anchor_id = best[sid][1].memory.id
                positions = {m.id: index for index, m in enumerate(messages)}
                anchor_index = positions.get(anchor_id, 0)
                window = sorted(set(
                    range(max(0, anchor_index - 15), min(len(messages), anchor_index + 16))
                ) | set(range(max(0, len(messages) - 30), len(messages)))
                )
                messages = [messages[i] for i in window]
            pool.extend((memory, sid) for memory in messages)
        if not pool:
            return []

        model = get_cross_encoder()
        if model is None:
            return _mmr_diversify(ranked, limit)
        try:
            pairs = [(query, memory.content[:512]) for memory, _ in pool]
            scores = model.predict(pairs, show_progress_bar=False, batch_size=32)
        except Exception:
            return _mmr_diversify(ranked, limit)

        scored = sorted(
            zip(pool, scores),
            key=lambda item: -float(item[1]),
        )
        if allocation == "anchor":
            return self._score_anchor_slots(scored, limit=limit, cap=cap)
        taken: dict[str, int] = {}
        selected: list[SearchResult] = []
        for (memory, sid), score in scored:
            if taken.get(sid, 0) >= cap:
                continue
            taken[sid] = taken.get(sid, 0) + 1
            result = SearchResult(memory=memory, score=float(score))
            setattr(result, "_ce_score", float(score))
            selected.append(result)
            if len(selected) >= limit:
                break
        return selected

    def _score_anchor_slots(
        self,
        scored: list[tuple[tuple[Memory, str], float]],
        *,
        limit: int = 5,
        cap: int = 2,
    ) -> list[SearchResult]:
        """Anchor-biased slot allocation: the best session gets two slots,
        the next sessions one each (session coverage preserved for the
        top contexts; the anchor's second message may surface).
        """
        groups: dict[str, list[tuple[Memory, float]]] = {}
        for (memory, sid), score in scored:
            groups.setdefault(sid, []).append((memory, float(score)))
        selected: list[SearchResult] = []
        remaining = limit
        for rank, (sid, entries) in enumerate(groups.items()):
            slots = min(cap if rank == 0 else 1, remaining, len(entries))
            for memory, score in entries[:slots]:
                result = SearchResult(memory=memory, score=score)
                setattr(result, "_ce_score", score)
                selected.append(result)
                remaining -= 1
            if remaining <= 0:
                break
        if remaining > 0:
            # Very few sessions: fill the rest best-first under the cap.
            taken = {sid: len(entries[: (cap if i == 0 else 1)]) for i, (sid, entries) in enumerate(groups.items())}
            for (memory, sid), score in scored:
                if taken.get(sid, 0) >= cap:
                    continue
                taken[sid] = taken.get(sid, 0) + 1
                result = SearchResult(memory=memory, score=float(score))
                setattr(result, "_ce_score", float(score))
                selected.append(result)
                remaining -= 1
                if remaining <= 0:
                    break
        return selected

    def _reconstruct_sessions(
        self,
        query: str,
        session_limit: int = 5,
        max_context_chars: int = _DEFAULT_BUNDLE_BUDGET_CHARS,
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
                    messages=_anchor_first(
                        messages,
                        [result.memory.id] if result.memory.id else [],
                    ),
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
                messages=_anchor_first(
                    selected,
                    [result.memory.id] if result.memory.id else [],
                ),
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
            for index, message_id in enumerate(priority_ids):
                if message_id in selected_ids or message_id not in by_id:
                    continue
                message = by_id[message_id]
                message_chars = len(message.content)
                if index > 0 or used_chars + message_chars <= per_bundle_budget:
                    # Anchors (the retrieved evidence) always survive the budget;
                    # the opening message is best-effort.
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
            selected = _anchor_first(selected, bundle.anchor_ids)
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
        self._ensure_open()
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
                escaped = k.replace("\\", "\\\\").replace('"', '\\"')
                where_parts.append("json_extract(metadata, ?) = ?")
                params.append(f'$."{escaped}"')
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
        self._ensure_open()
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
        self._ensure_open()
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
        self._ensure_open()
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
                escaped = k.replace("\\", "\\\\").replace('"', '\\"')
                where_parts.append("json_extract(metadata, ?) = ?")
                params.append(f'$."{escaped}"')
                params.append(v)
        where = " AND ".join(where_parts) if where_parts else "1=1"
        before = self._db.raw_query("SELECT COUNT(*) AS c FROM messages")
        self._db.raw_query(f"DELETE FROM messages WHERE {where}", tuple(params))
        after = self._db.raw_query("SELECT COUNT(*) AS c FROM messages")
        return (before[0]["c"] - after[0]["c"]) if before and after else 0

    def clear(self) -> None:
        self._ensure_open()
        self._db.raw_query("DELETE FROM messages")

    # ── Session inventory, memory hygiene, lifecycle ────────────────────

    def _ensure_open(self) -> None:
        """Raise if this instance has been closed (use-after-close guard)."""
        if self._closed:
            raise RuntimeError(
                "MemoryCore instance is closed; create a new MemoryCore to reopen"
            )

    def list_sessions(self) -> list[dict[str, Any]]:
        """List all sessions with message counts and last activity.

        Returns rows of ``{"session_id", "messages", "last_ts"}`` ordered by
        most recent activity first.
        """
        self._ensure_open()
        return self._db.raw_query(
            "SELECT session_id, COUNT(*) as messages, MAX(ts) as last_ts "
            "FROM messages WHERE session_id IS NOT NULL AND session_id != '' "
            "GROUP BY session_id ORDER BY last_ts DESC"
        )

    def delete_messages(self, message_ids: list[str]) -> int:
        """Delete specific messages by id. Returns the number deleted."""
        self._ensure_open()
        if not message_ids:
            return 0
        placeholders = ",".join("?" * len(message_ids))
        before = self._db.raw_query("SELECT COUNT(*) AS c FROM messages")
        self._db.raw_query(
            f"DELETE FROM messages WHERE id IN ({placeholders})",
            message_ids,
        )
        after = self._db.raw_query("SELECT COUNT(*) AS c FROM messages")
        before_n = before[0]["c"] if before else 0
        after_n = after[0]["c"] if after else 0
        return before_n - after_n

    def stats(self) -> dict[str, Any]:
        """Return basic memory statistics for health checks and agent tooling."""
        self._ensure_open()
        rows = self._db.raw_query(
            "SELECT COUNT(*) as messages, "
            "COUNT(DISTINCT CASE WHEN session_id != '' THEN session_id END) as sessions, "
            "COUNT(DISTINCT user_id) as users, MAX(ts) as last_ts "
            "FROM messages"
        )
        row = rows[0] if rows else {}
        pending = 0
        try:
            pending = self._db.journal_status().get("pending", 0)
        except Exception:
            pass
        return {
            "messages": row.get("messages", 0),
            "sessions": row.get("sessions", 0) if row.get("sessions") else 0,
            "users": row.get("users", 0) if row.get("users") else 0,
            "last_ts": row.get("last_ts"),
            "journal_pending": pending,
        }

    def close(self) -> None:
        """Release Chroma/hybrid resources held by this instance.

        Without this, pooled Chroma clients keep file handles open across
        many ``MemoryCore`` instantiations (a real failure mode: unclosed
        handles left the vector store readonly in long-running processes).
        The instance is unusable after close; create a new one to reopen.
        """
        try:
            from hybriddb.db import _chroma_client_pool, _chroma_pool_lock

            key = os.fspath(self._db._vector_path)
            with _chroma_pool_lock:
                client = _chroma_client_pool.pop(key, None)
            if client is not None:
                try:
                    client.clear_system_cache()
                except Exception:
                    pass
                try:
                    client._system.stop()
                except Exception:
                    pass
        except Exception:
            pass
        self._closed = True

    def __enter__(self) -> MemoryCore:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ── AgentJournal methods ───────────────────────────────────

    async def compile_turn(
        self,
        turn_id: str,
        timestamp: str | None = None,
        title: str | None = None,
        *,
        force: bool = False,
    ) -> AgentJournalCompileResult | None:
        self._ensure_open()
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
        session_cap: int = 1,
    ) -> list[SearchResult] | list[SessionBundle]:
        """Retrieve memories by query strategy.

        Strategies:
            - "episodic" (default): query decomposition + cross-encoder reranking, zero LLM
            - "direct": BM25+hybrid search, zero LLM
            - "expanded": LLM query rephrasing, 1 LLM call
            - "fusion": RRF fusion of direct + episodic, zero LLM

        Set bundles=True to return SessionBundle objects with surrounding context.
        Filter params (role, session_id, etc.) apply to all strategies.
        Bundles use a 4k-char total context budget with evidence-first message
        ordering (retrieved anchors lead) — validated on the 500-question S
        answer eval (LLM answer + judge): 4k bundles scored 0.678 vs 0.608 at
        16k with ~60% less context.
        ``session_cap`` (episodic only) allows up to N messages per session in
        the final ranking instead of the default one-per-session MMR cap —
        recovers answers that live in a second message of a found session
        (+0.048 answer accuracy on S at cap=2, at the cost of session recall).
        """
        if not query or not query.strip():
            raise ValueError("recall requires a non-empty query")
        self._ensure_open()
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
            results = self._search_with_fusion(
                query, limit=limit,
                role=role, session_id=session_id, user_id=user_id,
                agent_id=agent_id, ts_after=ts_after, ts_before=ts_before,
                metadata=metadata,
            )
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
                session_cap=session_cap,
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
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if limit <= 0:
            return []

        mc_results = self._search_messages(
            query, limit=per_query_limit, role=role, session_id=session_id,
            user_id=user_id, agent_id=agent_id, ts_after=ts_after,
            ts_before=ts_before, metadata=metadata,
        )
        er_results = self._search_messages_decomposed(
            query, limit=per_query_limit, per_query_limit=per_query_limit,
            use_cross_encoder=True,
            role=role, session_id=session_id, user_id=user_id,
            agent_id=agent_id, ts_after=ts_after, ts_before=ts_before,
            metadata=metadata,
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
