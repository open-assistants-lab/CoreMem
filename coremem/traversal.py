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
TURN_QA_EDGE = "turn_qa"
UPDATE_EDGE = "update"
CAUSAL_EDGE = "causal"
SELF_REF_EDGE = "self_reference"
EMOTIONAL_EDGE = "emotional"
ENTITY_EDGE = "entity"
SEMANTIC_EDGE = "semantic"
HOP_DECAY = 0.7
CONFIRMATION_BOOST = 0.5
MAX_TOPIC_EDGES_PER_MESSAGE = 10
MAX_SEMANTIC_EDGES_PER_MESSAGE = 10
SEMANTIC_SIM_THRESHOLD = 0.55

# Edge weights encode human-memory association strength (see
# docs/graph-edges-design.md): temporal 1.0, causal 0.9, entity 0.8,
# update 0.8, turn_qa 0.7, emotional 0.6, self_reference 0.6,
# semantic 0.5, topic 0.5+.
_EDGE_WEIGHTS = {
    TURN_QA_EDGE: 0.7,
    UPDATE_EDGE: 0.8,
    CAUSAL_EDGE: 0.9,
    SELF_REF_EDGE: 0.6,
    EMOTIONAL_EDGE: 0.6,
    ENTITY_EDGE: 0.8,
    SEMANTIC_EDGE: 0.5,
}

# Pattern-based extractors (zero-LLM, consistent with CoreMem's ethos)
_UPDATE_RE = re.compile(
    r"\b(changed|switched|no longer|used to|instead of|moved to|upgraded|"
    r"downgraded|replaced|quit|stopped|gave up|now (?:use|prefer|live|work|take))\b",
    re.IGNORECASE,
)
_CAUSAL_RE = re.compile(
    r"\b(because|therefore|as a result|which is why|due to|thanks to|that's why|so that)\b",
    re.IGNORECASE,
)
_SELF_RE = re.compile(r"\b(i|me|my|mine|we|our|ours)\b", re.IGNORECASE)
_POSITIVE_WORDS = {
    "love", "loved", "great", "amazing", "wonderful", "excellent", "happy",
    "excited", "enjoy", "enjoyed", "best", "beautiful", "fantastic",
    "awesome", "perfect", "delicious", "fun", "glad", "thrilled", "proud",
}
_NEGATIVE_WORDS = {
    "hate", "hated", "terrible", "awful", "horrible", "sad", "angry",
    "frustrated", "disappointed", "worried", "stressed", "annoyed",
    "boring", "waste", "regret", "regretted", "scared", "upset", "bad",
}
_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
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


def _sentiment(content: str) -> float:
    """Lexicon sentiment in [-1, 1]; 0 when neutral."""
    words = {w.lower() for w in re.findall(r"\w+", content)}
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


_ENTITY_EXCLUSIONS = {
    "i", "the", "a", "an", "it", "we", "they", "he", "she", "my", "me",
    "you", "your", "our", "this", "that", "these", "those", "there",
    "here", "so", "but", "and", "or", "for", "with", "from", "what",
    "how", "when", "where", "why", "who", "let", "one", "two", "three",
    "great", "good", "yeah", "well", "just", "actually", "really",
    "very", "also", "then", "now", "today", "tomorrow", "yesterday",
    "last", "next", "first", "second", "new", "old", "big", "small",
    "best", "worst", "more", "most", "some", "any", "every", "all",
    "both", "each", "few", "many", "much", "no", "yes", "ok", "okay",
    "thanks", "thank", "please", "sorry", "hi", "hello", "hey", "wait",
    "look", "listen", "sure", "right", "fine", "nice", "cool", "done",
    "ready", "free", "busy", "tired", "happy", "sad", "angry", "back",
    "home", "work", "school", "week", "month", "year", "day", "time",
    "thing", "things", "way", "lot", "bit", "kind", "sort", "stuff",
}


def _entities(content: str) -> set[str]:
    """Pattern NER: capitalized words/sequences, minus noise."""
    found: set[str] = set()
    for match in _ENTITY_RE.finditer(content):
        name = match.group(1)
        words = name.split()
        if any(w.lower() in _ENTITY_EXCLUSIONS for w in words):
            continue
        if len(words) == 1 and len(words[0]) < 3:
            continue
        found.add(name.lower())
    return found


def _batch_embeddings(texts: list[str]) -> list[list[float]]:
    """Batched embeddings via ChromaDB's default function (ONNX, cached)."""
    from chromadb.utils import embedding_functions
    ef = embedding_functions.DefaultEmbeddingFunction()
    return ef(texts)


def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _build_message_graph(db: Any, messages: list[Memory]) -> dict[str, int]:
    """Materialize the message graph.

    Edge types (see docs/graph-edges-design.md):
      within-session: temporal_next, same_session, turn_qa
      cross-session:  topic, update, causal, self_reference, emotional,
                      entity, semantic
    Each message pair gets ONE edge — the strongest applicable association
    (causal 0.9 > entity/update 0.8 > turn_qa 0.7 > emotional/self_ref 0.6
    > semantic/topic 0.5) — so traversal path counts stay honest.

    Returns edge counts by type (for eval instrumentation).
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

    # ── Within-session edges ──────────────────────────────────────────────
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
        # turn_qa: user ↔ assistant within the same turn
        by_turn: dict[str, list[Memory]] = {}
        for message in session_messages:
            turn_id = (message.metadata or {}).get("turn_id") or ""
            by_turn.setdefault(turn_id, []).append(message)
        for turn_messages in by_turn.values():
            roles = {m.role for m in turn_messages}
            if len(turn_messages) >= 2 and {"user", "assistant"} <= roles:
                for i, a in enumerate(turn_messages):
                    for b in turn_messages[i + 1:]:
                        if a.role != b.role:
                            edges.append({
                                "source_id": a.id, "target_id": b.id,
                                "type": TURN_QA_EDGE, "weight": 0.7,
                            })

    # ── Cross-session edges ───────────────────────────────────────────────
    # Embeddings are computed once (batched) and similarities vectorized
    # with numpy. Keyword/entity pairs are found via inverted indexes so
    # each message pair is evaluated exactly once.
    ordered = [m for m in messages if m.id]
    embeddings = _batch_embeddings([m.content for m in ordered])
    import numpy as np
    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    emb_matrix = emb_matrix / np.maximum(norms, 1e-9)
    sim_matrix = emb_matrix @ emb_matrix.T
    emb_index = {m.id: i for i, m in enumerate(ordered)}

    features = {
        m.id: {
            "keywords": _significant_keywords(m.content),
            "entities": _entities(m.content),
            "update": bool(_UPDATE_RE.search(m.content)),
            "causal": bool(_CAUSAL_RE.search(m.content)),
            "self_ref": bool(_SELF_RE.search(m.content)),
            "sentiment": _sentiment(m.content),
            "session_id": m.session_id or "",
        }
        for m in messages
        if m.id
    }

    def _cross_session_pairs(index: dict[str, list[str]]) -> dict[tuple[str, str], set[str]]:
        """Map (a, b) message-id pairs (different sessions) to shared keys.

        Keywords appearing in more than 10% of messages are too generic to
        be distinctive ("going", "think", "want") — they would create a
        near-complete cross-session graph.
        """
        cap = max(10, len(messages) // 10)
        pairs: dict[tuple[str, str], set[str]] = {}
        for key, members in index.items():
            if len(members) > cap:
                continue
            by_session: dict[str, list[str]] = {}
            for mid in members:
                by_session.setdefault(features[mid]["session_id"], []).append(mid)
            sessions = list(by_session)
            for i in range(len(sessions)):
                for j in range(i + 1, len(sessions)):
                    for a in by_session[sessions[i]]:
                        for b in by_session[sessions[j]]:
                            pair = (a, b) if a < b else (b, a)
                            pairs.setdefault(pair, set()).add(key)
        return pairs

    kw_index: dict[str, list[str]] = {}
    ent_index: dict[str, list[str]] = {}
    for mid, feat in features.items():
        for kw in feat["keywords"]:
            kw_index.setdefault(kw, []).append(mid)
        for ent in feat["entities"]:
            ent_index.setdefault(ent, []).append(mid)
    shared_keywords = _cross_session_pairs(kw_index)
    shared_entities = _cross_session_pairs(ent_index)

    edges: list[dict[str, Any]] = []
    topic_counts: dict[str, int] = {}
    semantic_counts: dict[str, int] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for pair in shared_keywords:
        a, b = pair
        fa, fb = features[a], features[b]
        best_type: str | None = None
        best_weight = 0.0
        if pair in shared_entities:
            best_type, best_weight = ENTITY_EDGE, _EDGE_WEIGHTS[ENTITY_EDGE]
        shared = shared_keywords[pair]
        topic_weight = min(0.5 + 0.1 * len(shared), 1.0)
        if fa["causal"] or fb["causal"]:
            best_type, best_weight = CAUSAL_EDGE, _EDGE_WEIGHTS[CAUSAL_EDGE]
        elif fa["update"] or fb["update"]:
            best_type, best_weight = UPDATE_EDGE, _EDGE_WEIGHTS[UPDATE_EDGE]
        elif fa["self_ref"] and fb["self_ref"]:
            best_type, best_weight = SELF_REF_EDGE, _EDGE_WEIGHTS[SELF_REF_EDGE]
        elif (
            fa["sentiment"] and fb["sentiment"]
            and (fa["sentiment"] * fb["sentiment"]) > 0
            and min(abs(fa["sentiment"]), abs(fb["sentiment"])) >= 0.5
        ):
            best_type, best_weight = EMOTIONAL_EDGE, _EDGE_WEIGHTS[EMOTIONAL_EDGE]
        elif topic_weight > best_weight:
            best_type, best_weight = TOPIC_EDGE, topic_weight
        if best_type is not None:
            edges.append({
                "source_id": a, "target_id": b,
                "type": best_type, "weight": best_weight,
            })
            seen_pairs.add(pair)
            topic_counts[a] = topic_counts.get(a, 0) + 1
            topic_counts[b] = topic_counts.get(b, 0) + 1

    # Semantic edges: cross-session pairs with high embedding similarity and
    # no stronger edge — found via the vectorized similarity matrix.
    ids = list(emb_index)
    sim_pairs = np.argwhere(sim_matrix >= SEMANTIC_SIM_THRESHOLD)
    for i, j in sim_pairs:
        if i >= j:
            continue
        a, b = ids[i], ids[j]
        if features[a]["session_id"] == features[b]["session_id"]:
            continue
        pair = (a, b)
        if pair in seen_pairs:
            continue
        if (
            topic_counts.get(a, 0) >= MAX_TOPIC_EDGES_PER_MESSAGE
            or topic_counts.get(b, 0) >= MAX_TOPIC_EDGES_PER_MESSAGE
            or semantic_counts.get(a, 0) >= MAX_SEMANTIC_EDGES_PER_MESSAGE
            or semantic_counts.get(b, 0) >= MAX_SEMANTIC_EDGES_PER_MESSAGE
        ):
            continue
        edges.append({
            "source_id": a, "target_id": b,
            "type": SEMANTIC_EDGE, "weight": _EDGE_WEIGHTS[SEMANTIC_EDGE],
        })
        seen_pairs.add(pair)
        semantic_counts[a] = semantic_counts.get(a, 0) + 1
        semantic_counts[b] = semantic_counts.get(b, 0) + 1
    db.add_edges(edges)
    from collections import Counter
    return dict(Counter(edge["type"] for edge in edges))


def search_messages_traversal(
    core: Any,
    query: str,
    *,
    limit: int = 5,
    seed_limit: int = 50,
    hop_limit: int = 2,
    max_per_session: int = 2,
    use_cross_encoder: bool = True,
    timings: dict[str, float] | None = None,
) -> list[SearchResult]:
    """Query-guided graph traversal retrieval.

    The final pool is a strict superset of the baseline's rerank window
    (top-``seed_limit`` decomposed results + graph candidates), so the
    output can only match or beat the baseline: if traversal discovers
    nothing, the rerank + MMR pipeline produces the baseline ranking.

    When ``timings`` is provided, phase durations (seeds, graph build,
    traversal, rerank) are recorded into it for eval instrumentation.
    """
    import time
    if limit <= 0:
        return []

    # 1. Seeds: the strongest zero-LLM retriever (RRF fusion). The seed
    #    window must match the baseline's exact rerank window (same
    #    per_query_limit), so every baseline candidate is in the pool.
    _t0 = time.perf_counter()
    seeds = core._search_messages_decomposed(
        query,
        limit=seed_limit,
        per_query_limit=max(20, seed_limit // 4),
        use_cross_encoder=False,
    )
    if timings is not None:
        timings["seeds_s"] = time.perf_counter() - _t0
    if not seeds:
        return []

    # 2. Materialize the message graph
    _t0 = time.perf_counter()
    messages = core.fetch_all()
    by_id = {m.id: m for m in messages if m.id}
    _build_message_graph(core._db, messages)
    if timings is not None:
        timings["graph_build_s"] = time.perf_counter() - _t0

    # 3. Expand from every seed through the graph. Only candidates from
    #    sessions outside the seed set add value: same-session neighbors are
    #    already covered by the baseline's rerank window and bundle
    #    reconstruction, and MMR's one-result-per-session cap means a
    #    same-session candidate can never outrank its own seed.
    _t0 = time.perf_counter()
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
            if by_id[node_id].session_id in seed_sessions:
                continue
            entry = candidates.setdefault(
                node_id,
                {"depth": row["depth"], "paths": 0,
                 "session_id": by_id[node_id].session_id or ""},
            )
            entry["depth"] = min(entry["depth"], row["depth"])
            entry["paths"] += 1
    if timings is not None:
        timings["traverse_s"] = time.perf_counter() - _t0

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
    _t0 = time.perf_counter()
    if use_cross_encoder:
        pool = rerank(query, pool, top_k=max(50, len(pool)))
    if timings is not None:
        timings["rerank_s"] = time.perf_counter() - _t0
    return _mmr_diversify(pool, limit)
