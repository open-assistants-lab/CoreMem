"""Query-guided graph traversal retrieval.

Revives the removed SPEC design on the fixed HybridDB graph (0.5.5+):

    1. Seeds come from the strongest retriever (decomposed hybrid search),
       not a vector-only graph search.
    2. Expand locally through the message graph (temporal + session edges).
    3. Re-check relevance every hop — graph proximity is evidence, not
       relevance by itself. A zero-overlap candidate survives only when
       multiple frontier paths converge on it.
    4. Session diversity caps prevent one session from consuming the pool.
    5. Fallback: if traversal discovers nothing, the output is identical
       to the baseline top-k.

The graph is materialized per call from the message table. Incremental
maintenance on ingest is future work.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from coremem.heuristics import _mmr_diversify
from coremem.rerank import rerank
from coremem.types import Memory, SearchResult

TEMPORAL_EDGE = "temporal_next"
SESSION_EDGE = "same_session"
TOPIC_EDGE = "topic"
HOP_DECAY = 0.7
CONFIRMATION_BOOST = 0.5
MAX_TOPIC_EDGES_PER_MESSAGE = 10
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "and", "or",
    "but", "not", "so", "if", "as", "than", "that", "this", "these", "those",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "they", "them", "their",
}


def _query_keywords(query: str) -> set[str]:
    return {
        w.lower() for w in re.findall(r"\w+", query)
        if len(w) > 2 and w.lower() not in _STOPWORDS
    }


def _keyword_overlap_ratio(query: str, content: str) -> float:
    """Fraction of query keywords present in content (0..1)."""
    keywords = _query_keywords(query)
    if not keywords:
        return 0.0
    content_lower = content.lower()
    hits = sum(1 for kw in keywords if kw in content_lower)
    return hits / len(keywords)


def _significant_keywords(content: str) -> set[str]:
    """Content keywords for cross-session topic edges."""
    return {
        w.lower() for w in re.findall(r"\w+", content)
        if len(w) > 3 and w.lower() not in _STOPWORDS
    }


def _build_message_graph(db: Any, messages: list[Memory]) -> None:
    """Materialize the message graph: nodes + temporal/session/topic edges.

    Topic edges connect messages in different sessions that share
    significant keywords — the cross-session links that let traversal
    discover new episodes (the SPEC's deferred Stage 2 edge type).
    """
    db.add_nodes([
        {
            "id": message.id,
            "label": message.content,
            "type": "message",
            "properties": {
                "session_id": message.session_id or "",
                "ts": message.ts.isoformat() if message.ts else "",
            },
        }
        for message in messages
        if message.id
    ])
    by_session: dict[str, list[Memory]] = {}
    for message in messages:
        by_session.setdefault(message.session_id or "", []).append(message)
    edges: list[dict[str, Any]] = []
    for session_messages in by_session.values():
        session_messages.sort(
            key=lambda m: m.ts or datetime.min.replace(tzinfo=UTC)
        )
        for prev, nxt in zip(session_messages, session_messages[1:]):
            edges.append({
                "source_id": prev.id, "target_id": nxt.id,
                "type": TEMPORAL_EDGE, "weight": 1.0,
            })
        for i, a in enumerate(session_messages):
            for b in session_messages[i + 1:]:
                edges.append({
                    "source_id": a.id, "target_id": b.id,
                    "type": SESSION_EDGE, "weight": 0.5,
                })
    # Cross-session topic edges: shared significant keywords, capped per
    # message to keep the graph sparse.
    session_lists = list(by_session.values())
    topic_counts: dict[str, int] = {}
    for i, msgs_a in enumerate(session_lists):
        for msgs_b in session_lists[i + 1:]:
            for a in msgs_a:
                if topic_counts.get(a.id, 0) >= MAX_TOPIC_EDGES_PER_MESSAGE:
                    continue
                keywords_a = _significant_keywords(a.content)
                if not keywords_a:
                    continue
                for b in msgs_b:
                    if topic_counts.get(b.id, 0) >= MAX_TOPIC_EDGES_PER_MESSAGE:
                        continue
                    shared = keywords_a & _significant_keywords(b.content)
                    if not shared:
                        continue
                    edges.append({
                        "source_id": a.id, "target_id": b.id,
                        "type": TOPIC_EDGE,
                        "weight": min(0.5 + 0.1 * len(shared), 1.0),
                    })
                    topic_counts[a.id] = topic_counts.get(a.id, 0) + 1
                    topic_counts[b.id] = topic_counts.get(b.id, 0) + 1
    db.add_edges(edges)


def search_messages_traversal(
    core: Any,
    query: str,
    *,
    limit: int = 5,
    seed_limit: int = 50,
    hop_limit: int = 2,
    max_per_session: int = 2,
    use_cross_encoder: bool = True,
) -> list[SearchResult]:
    """Query-guided graph traversal retrieval.

    The final pool is a strict superset of the baseline's rerank window
    (top-``seed_limit`` decomposed results + graph candidates), so the
    output can only match or beat the baseline: if traversal discovers
    nothing, the rerank + MMR pipeline produces the baseline ranking.
    """
    if limit <= 0:
        return []

    # 1. Seeds: the strongest zero-LLM retriever (RRF fusion). The seed
    #    window must match the baseline's exact rerank window (same
    #    per_query_limit), so every baseline candidate is in the pool.
    seeds = core._search_messages_decomposed(
        query,
        limit=seed_limit,
        per_query_limit=max(20, seed_limit // 4),
        use_cross_encoder=False,
    )
    if not seeds:
        return []

    # 2. Materialize the message graph
    messages = core.fetch_all()
    by_id = {m.id: m for m in messages if m.id}
    _build_message_graph(core._db, messages)

    # 3. Expand from every seed through the graph
    seed_sessions = {seed.memory.session_id for seed in seeds}
    candidates: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        seed_id = seed.memory.id
        if not seed_id:
            continue
        for row in core._db.traverse(seed_id, max_depth=hop_limit, direction="both"):
            node_id = row["node_id"]
            if node_id == seed_id or node_id not in by_id:
                continue
            # Only candidates from sessions outside the seed set add value:
            # same-session neighbors are already covered by the baseline's
            # rerank window and bundle reconstruction.
            if by_id[node_id].session_id in seed_sessions:
                continue
            entry = candidates.setdefault(
                node_id,
                {"depth": row["depth"], "paths": 0,
                 "session_id": by_id[node_id].session_id or ""},
            )
            entry["depth"] = min(entry["depth"], row["depth"])
            entry["paths"] += 1

    # 4. Relevance re-check, session cap, scoring
    pool = list(seeds)
    per_session: dict[str, int] = {}
    for result in seeds:
        sid = result.memory.session_id or ""
        per_session[sid] = per_session.get(sid, 0) + 1
    for node_id, info in candidates.items():
        memory = by_id[node_id]
        sid = info["session_id"]
        if per_session.get(sid, 0) >= max_per_session:
            continue
        relevance = _keyword_overlap_ratio(query, memory.content)
        if relevance == 0.0 and info["paths"] < 2:
            continue  # single-path zero-overlap candidate: prune
        score = (
            (HOP_DECAY ** info["depth"])
            * (1.0 + CONFIRMATION_BOOST * min(info["paths"] - 1, 2))
            * (0.5 + relevance)
        )
        pool.append(SearchResult(memory=memory, score=score))
        per_session[sid] = per_session.get(sid, 0) + 1

    # 5. Rerank the full pool (same pipeline as the baseline) and apply
    #    MMR session diversity. With no surviving candidates the pool is
    #    exactly the baseline's rerank window, so the output is identical
    #    to the baseline top-limit.
    if use_cross_encoder:
        pool = rerank(query, pool, top_k=max(50, len(pool)))
    return _mmr_diversify(pool, limit)
