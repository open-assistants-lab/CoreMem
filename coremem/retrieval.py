"""Zero-LLM retrieval improvements.

- ``search_messages_confirmed``: temporal-neighbor confirmation boost.
  A message whose session-neighbors also match the query is more likely
  the answer (temporal contiguity — the strongest human-memory retrieval
  cue). A thread of moderate matches beats an isolated high-scorer.
- ``search_messages_typo_robust``: corpus-aware fuzzy query expansion.
  The eval's questions contain typos; FTS5 is exact-match, so typo'd
  queries miss messages. Each query word is matched against the message
  corpus vocabulary (edit-distance) and corrections are added as query
  variants.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

from coremem.heuristics import _mmr_diversify
from coremem.rerank import rerank
from coremem.types import SearchResult

CONFIRM_ALPHA = 0.05
CONFIRM_WINDOW = 1
TYPO_RATIO = 0.8
TYPO_MIN_WORD_LEN = 4


# ── Temporal-neighbor confirmation ─────────────────────────────────────────


def _session_neighbors(core: Any) -> dict[str, list[str]]:
    """Map message id → ids of temporal neighbors in the same session."""
    messages = core.fetch_all()
    by_session: dict[str, list[Any]] = {}
    for message in messages:
        by_session.setdefault(message.session_id or "", []).append(message)
    neighbors: dict[str, list[str]] = {}
    for session_messages in by_session.values():
        session_messages.sort(key=lambda m: m.ts or datetime.min.replace(tzinfo=UTC))
        for i, message in enumerate(session_messages):
            window = session_messages[max(0, i - CONFIRM_WINDOW):i + CONFIRM_WINDOW + 1]
            neighbors[message.id] = [x.id for x in window if x.id != message.id]
    return neighbors


def search_messages_confirmed(
    core: Any,
    query: str,
    *,
    limit: int = 5,
    seed_limit: int = 50,
    alpha: float = CONFIRM_ALPHA,
) -> list[SearchResult]:
    """Decomposed search + temporal-neighbor confirmation boost.

    After the cross-encoder rerank, each message's score is boosted by the
    mean score of its temporal neighbors in the same session, but only
    from *positive* neighbors (the cross-encoder scores are mostly
    negative; boosting from negative neighbors reshuffles the ranking by
    neighbor structure instead of relevance):
    ``final = ce_score + alpha * max(0, neighbor_mean)``.
    """
    results = core._search_messages_decomposed(
        query,
        limit=seed_limit,
        per_query_limit=max(20, seed_limit // 4),
        use_cross_encoder=True,
    )
    if not results:
        return []
    neighbors = _session_neighbors(core)
    score_by_id = {
        r.memory.id: getattr(r, "_ce_score", r.score)
        for r in results
        if r.memory.id
    }
    for result in results:
        neighbor_scores = [
            score_by_id[nid] for nid in neighbors.get(result.memory.id, [])
            if nid in score_by_id
        ]
        if neighbor_scores:
            neighbor_mean = sum(neighbor_scores) / len(neighbor_scores)
            if neighbor_mean > 0:
                result.score = result.score + alpha * neighbor_mean
    results.sort(key=lambda r: r.score, reverse=True)
    return _mmr_diversify(results, limit)


# ── Corpus-aware typo robustness ──────────────────────────────────────────


def _corpus_vocabulary(core: Any) -> set[str]:
    words: set[str] = set()
    for message in core.fetch_all():
        for word in re.findall(r"\w+", message.content):
            if len(word) >= TYPO_MIN_WORD_LEN:
                words.add(word.lower())
    return words


def fuzzy_expand_queries(query: str, vocabulary: set[str]) -> list[str]:
    """Return query variants with corpus-aware typo corrections.

    Each query word (len >= 4) is matched against the vocabulary by
    edit-similarity; the best match with ratio >= TYPO_RATIO replaces the
    word. Clean queries produce no variants.
    """
    if not vocabulary:
        return []
    variants: list[str] = []
    for word in re.findall(r"\w+", query):
        if len(word) < TYPO_MIN_WORD_LEN:
            continue
        best: str | None = None
        best_ratio = 0.0
        for candidate in vocabulary:
            if abs(len(candidate) - len(word)) > 2:
                continue
            ratio = SequenceMatcher(None, word.lower(), candidate).ratio()
            if ratio > best_ratio:
                best, best_ratio = candidate, ratio
        if best is not None and best_ratio >= TYPO_RATIO and best != word.lower():
            variants.append(query.replace(word, best))
    return variants


def search_messages_typo_robust(
    core: Any,
    query: str,
    *,
    limit: int = 5,
    seed_limit: int = 50,
) -> list[SearchResult]:
    """Decomposed search with corpus-aware typo expansion.

    The original query plus each typo-corrected variant is run through the
    decomposed search; results are RRF-fused, then cross-encoder reranked
    and MMR-diversified.
    """
    vocabulary = _corpus_vocabulary(core)
    variants = fuzzy_expand_queries(query, vocabulary)
    queries = [query] + variants

    fused: dict[str, tuple[Any, float]] = {}
    for variant in queries:
        for rank, result in enumerate(
            core._search_messages_decomposed(
                variant,
                limit=seed_limit,
                per_query_limit=max(20, seed_limit // 4),
                use_cross_encoder=False,
            ),
            start=1,
        ):
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
    ranked = rerank(query, ranked, top_k=max(50, len(ranked)))
    return _mmr_diversify(ranked, limit)


# ── Preference union-retrieval ─────────────────────────────────────────────

_PREFERENCE_CUES = (
    "recommend", "suggest", "should i", "what kind of", "what type of",
    "can you recommend", "can you suggest", "do i like", "do i prefer",
    "what do i", "what's my favorite", "what is my favorite",
    "what does", "what do you think i", "which do i",
)


def _is_preference_query(query: str) -> bool:
    lowered = query.lower()
    return any(cue in lowered for cue in _PREFERENCE_CUES)


def search_messages_preference_union(
    core: Any,
    query: str,
    *,
    limit: int = 5,
    per_variant: int = 10,
) -> list[SearchResult]:
    """Preference queries: per-variant top-k UNION instead of RRF fusion.

    Preference answers are spread across messages ("X likes A", "X enjoys
    B", "X prefers C"). RRF's rank-based fusion lets a message that matches
    only one preference keyword variant fall out of the rerank window. The
    union collects the top-k of every variant, then lets the cross-encoder
    decide. Non-preference queries fall back to the baseline decomposed
    search.
    """
    if not _is_preference_query(query):
        return core._search_messages_decomposed(
            query, limit=limit, per_query_limit=max(20, limit * 4),
            use_cross_encoder=True,
        )
    from coremem.query import decompose_queries

    collected: dict[str, SearchResult] = {}
    for variant in decompose_queries(query):
        for result in core._search_messages(variant, limit=per_variant):
            memory_id = result.memory.id or ""
            if memory_id and memory_id not in collected:
                collected[memory_id] = result
    ranked = list(collected.values())
    ranked = rerank(query, ranked, top_k=max(50, len(ranked)))
    return _mmr_diversify(ranked, limit)
