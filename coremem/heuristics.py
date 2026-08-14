"""Deterministic post-retrieval heuristics — shared by all backends.

All heuristics are zero-LLM, purely pattern-based.
"""

import math
import re
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

# ── MMR Diversity ──────────────────────────────────────────────────────────


def _mmr_diversify(results: list[Any], k: int) -> list[Any]:
    """Session-diverse MMR reranking applied before cross-encoder.

    Iterates through score-sorted results, picking the highest-scoring
    message from each new session until k unique sessions are collected.
    Remaining slots (if fewer than k sessions exist) are filled from the
    highest-scoring results not yet selected.

    Messages without a session_id get a synthetic key based on content hash
    to prevent all session-less messages from colliding.
    """
    if not results or k <= 0:
        return results[:k]

    seen_sessions: set[str] = set()
    diverse: list[Any] = []
    overflow: list[Any] = []

    for r in results:
        sid = r.memory.session_id
        key = sid if sid else f"_no_session_{hash(r.memory.content)}"
        if key not in seen_sessions:
            diverse.append(r)
            seen_sessions.add(key)
            if len(diverse) >= k:
                break
        else:
            overflow.append(r)

    if len(diverse) < k:
        diverse.extend(overflow[:k - len(diverse)])

    return diverse[:k]


class SearchHeuristics:
    """Post-retrieval scoring heuristics based on MemPalace's proven patterns.

    Each heuristic applies a deterministic multiplier to results from
    the backend's raw search. Heuristics are additive — they boost or
    penalize scores without replacing the embedding ranking.
    """

    KEYWORD_OVERLAP_WEIGHT = 1.0
    FUZZY_THRESHOLD = 0.75
    FUZZY_WEIGHT = 0.4
    TEMPORAL_BOOST_FACTOR = 0.15
    PERSON_NAME_BOOST = 0.40
    QUOTED_PHRASE_BOOST = 0.60
    COUNTING_QUESTION_SNIPPET_LENGTH = 3000
    RECENCY_DECAY_WEIGHT = 0.1
    RECENCY_DECAY_HALF_LIFE_DAYS = 30
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "but", "not", "so", "if", "as", "than", "that",
        "this", "these", "those", "it", "its", "i", "me", "my", "we",
        "our", "you", "your", "he", "she", "they", "them", "their",
    }

    @classmethod
    def keyword_overlap(cls, query: str, content: str, score: float) -> float:
        """Boost score when query keywords appear in content.

        Exact unigram match + bigram match + fuzzy fallback for near-misses.
        fused = score * (1 + weight * keyword_overlap_ratio)
        """
        q_words = {w.lower() for w in re.findall(r"\w+", query) if len(w) > 2}
        q_words -= cls.STOP_WORDS
        if not q_words:
            return score

        c_words_lower = re.findall(r"\w+", content.lower())
        c_words = set(c_words_lower)

        # Exact unigram overlap
        exact = len(q_words & c_words) / len(q_words) if q_words else 0

        # Bigram overlap — catches "coffee creamer" vs single-word matches
        q_bigrams = {" ".join(w.lower() for w in bigram)
                     for bigram in zip(re.findall(r"\w+", query), re.findall(r"\w+", query)[1:])
                     if bigram[0] not in cls.STOP_WORDS}
        c_text = " ".join(c_words_lower)
        bigram_hits = sum(1 for bg in q_bigrams if bg in c_text)
        bigram_overlap = bigram_hits / len(q_bigrams) if q_bigrams else 0

        # Fuzzy fallback — near-misses like "creamers" vs "creamer"
        fuzzy_hits = 0
        for qw in q_words:
            if qw not in c_words:
                for cw in c_words:
                    if SequenceMatcher(None, qw, cw).ratio() >= cls.FUZZY_THRESHOLD:
                        fuzzy_hits += 1
                        break
        fuzzy_overlap = fuzzy_hits / len(q_words) if q_words else 0

        total_overlap = exact + 0.5 * bigram_overlap + cls.FUZZY_WEIGHT * fuzzy_overlap
        return score * (1 + cls.KEYWORD_OVERLAP_WEIGHT * total_overlap)

    @classmethod
    def temporal_boost(cls, query: str, content_ts: str | None, score: float) -> float:
        """Boost recent memories when query contains temporal cues.

        Detects patterns like 'current', 'latest', 'now', 'this year',
        'recently', 'these days' and boosts newer content.
        """
        temporal_cues = {
            "current", "latest", "now", "recently", "recent",
            "lately", "new", "newest", "these days", "this year",
            "nowadays", "updated", "today",
        }
        q_lower = query.lower()
        if not any(cue in q_lower for cue in temporal_cues):
            return score

        if not content_ts:
            return score

        try:
            ts = datetime.fromisoformat(content_ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            age_days = (datetime.now(UTC) - ts).days
            if age_days < 30:
                return score * (1 + cls.TEMPORAL_BOOST_FACTOR)
        except (ValueError, TypeError):
            pass
        return score

    @classmethod
    def recency_decay(cls, content_ts: str | None, score: float) -> float:
        """Unconditional mild recency boost — applied to every result.

        Uses exponential decay: score * (1 + weight * e^(-age_days / half_life))
        Very recent content gets ~10% boost, 30-day-old ~3.7%, 60-day ~1.4%.
        Always applied regardless of query content.
        """
        if not content_ts:
            return score

        try:
            ts = datetime.fromisoformat(content_ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            age_days = max(0, (datetime.now(UTC) - ts).days)
            factor = cls.RECENCY_DECAY_WEIGHT * math.exp(-age_days / cls.RECENCY_DECAY_HALF_LIFE_DAYS)
            return score * (1 + factor)
        except (ValueError, TypeError):
            pass
        return score

    @classmethod
    def person_name_boost(cls, content: str, score: float) -> float:
        """Boost content containing proper names (capitalized multi-word)."""
        names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", content)
        if names:
            return score * (1 + cls.PERSON_NAME_BOOST)
        return score

    @classmethod
    def quoted_phrase_boost(cls, query: str, content: str, score: float) -> float:
        """Boost when a quoted phrase from the query appears verbatim in content."""
        quoted = re.findall(r'"([^"]+)"', query)
        if not quoted:
            return score
        for phrase in quoted:
            if phrase.lower() in content.lower():
                return score * (1 + cls.QUOTED_PHRASE_BOOST)
        return score

    @classmethod
    def is_counting_question(cls, query: str) -> bool:
        """Detect 'how many' / 'how much total' questions."""
        q = query.lower()
        return q.startswith("how many") or "how much total" in q

    @classmethod
    def extract_date_cues(cls, query: str) -> str | None:
        """Extract a date reference from the query for temporal scoping."""
        year_match = re.search(r"\b(20\d{2})\b", query)
        if year_match:
            return year_match.group(1)

        month_match = re.search(
            r"\b(january|february|march|april|may|june|july|"
            r"august|september|october|november|december)\b",
            query, re.IGNORECASE,
        )
        if month_match:
            return month_match.group(1)

        return None

    @classmethod
    def apply_all(cls, query: str, content: str, score: float, ts: str | None = None) -> float:
        """Apply all applicable heuristics to a single result."""
        s = cls.keyword_overlap(query, content, score)
        s = cls.recency_decay(ts, s)
        s = cls.temporal_boost(query, ts, s)
        s = cls.person_name_boost(content, s)
        s = cls.quoted_phrase_boost(query, content, s)
        return s
