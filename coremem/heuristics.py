"""Deterministic post-retrieval heuristics — shared by all backends.

All heuristics are zero-LLM, purely pattern-based.
"""

import math
import re
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any

_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

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

    # ── Lever 5: deterministic time-aware range pruning ─────────────────

    @classmethod
    def parse_temporal_window(
        cls, query: str, anchor: datetime,
    ) -> tuple[datetime, datetime] | None:
        """Parse an explicit/relative temporal window from the query.

        Returns ``(start, end)`` anchored to ``anchor`` (the question's date,
        NOT now — eval haystacks are historical), or None when the query has
        no resolvable time range. Deterministic: regex + calendar arithmetic
        only, zero LLM.
        """
        q = query.lower()

        def month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
            start = anchor.replace(year=year, month=month, day=1,
                                   hour=0, minute=0, second=0, microsecond=0)
            if month == 12:
                end = start.replace(year=year + 1, month=1)
            else:
                end = start.replace(month=month + 1)
            return start, end

        # "between <month> and <month> (year?)" / "from <month> to <month>"
        m = re.search(
            r"\b(?:between|from)\s+(january|february|march|april|may|june|july|"
            r"august|september|october|november|december)\s*(?:(\d{4})\s*)?"
            r"(?:and|to)\s+(january|february|march|april|may|june|july|"
            r"august|september|october|november|december)\s*(\d{4})?",
            q, re.IGNORECASE,
        )
        if m:
            months = [mn for mn in _MONTH_NAMES]
            try:
                start_m = months.index(m.group(1).lower()) + 1
                end_m = months.index(m.group(3).lower()) + 1
                year = int(m.group(4) or m.group(2) or anchor.year)
                s, _ = month_bounds(year, start_m)
                _, e = month_bounds(year, end_m)
                return min(s, e), max(e, s)
            except ValueError:
                pass

        # "in the past N months/weeks/days" / "within the last N ..."
        m = re.search(
            r"\b(?:in the past|within the last|over the last|the last)\s+"
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
            r"(day|week|month|year)s?\b", q, re.IGNORECASE,
        )
        if m:
            raw_n = m.group(1)
            word_map = {w: i + 1 for i, w in enumerate(
                ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
            )}
            n = int(raw_n) if raw_n.isdigit() else word_map.get(raw_n.lower(), 1)
            unit = m.group(2)
            days = {"day": n, "week": 7 * n, "month": 30 * n, "year": 365 * n}[unit]
            end = anchor
            start = anchor - timedelta(days=days)
            return start, end

        # "last month" / "last week" / "last year" (anchored to question date)
        m = re.search(r"\blast\s+(month|week|year)\b", q)
        if m:
            unit = m.group(1)
            if unit == "month":
                first = anchor.replace(day=1, hour=0, minute=0, microsecond=0)
                start = (first - timedelta(days=1)).replace(day=1)
                end = first
                return start, end
            if unit == "week":
                return anchor - timedelta(days=13), anchor + timedelta(days=1)
            return anchor.replace(year=anchor.year - 1, month=1, day=1), anchor

        # "<Month> <year>" explicit
        m = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|"
            r"october|november|december)\s+(\d{4})\b", q, re.IGNORECASE,
        )
        if m:
            try:
                start, end = month_bounds(int(m.group(2)), _MONTH_NAMES.index(m.group(1).lower()) + 1)
                return start, end
            except ValueError:
                pass

        # "N days/weeks/months ago" counting questions: the event date is
        # unknown a priori (anywhere in the past), so a window filter adds no
        # signal on historical haystacks — deliberately no window here.

        return None

    @classmethod
    def has_temporal_cues(cls, query: str) -> bool:
        """True when the query references dates/relative time windows."""
        return cls.parse_temporal_window(query, datetime.now(UTC)) is not None or bool(
            re.search(
                r"\b(?:how\s+many\s+(?:days|weeks|months|years)\s+ago|"
                r"how\s+many\s+(?:days|weeks|months|years)\s+(?:have\s+)?passed|"
                r"last\s+(?:month|week|year)|in\s+the\s+past\s+\d+|"
                r"between\s+\w+\s+\d{4}|before\s+\w+)",
                query, re.IGNORECASE,
            )
        )

    @classmethod
    def temporal_window_boost(
        cls,
        query: str,
        content_ts: str | None,
        score: float,
        anchor: datetime,
        *,
        inside_factor: float = 1.30,
        outside_factor: float = 0.75,
    ) -> float:
        """Boost candidates inside the query's temporal window, penalize far
        outside ones (lever 5: deterministic time-aware range pruning).

        Anchored to the question date (not now). Applied to temporal-window
        queries only; all other queries pass through untouched.
        """
        window = cls.parse_temporal_window(query, anchor)
        if window is None or not content_ts:
            return score
        return score * cls._window_factor(window, content_ts, inside_factor=inside_factor,
                                          outside_factor=outside_factor)

    @staticmethod
    def _window_factor(
        window: tuple[datetime, datetime],
        content_ts: str | None,
        *,
        inside_factor: float = 1.30,
        outside_factor: float = 0.75,
    ) -> float:
        """Multiplicative prior for one candidate given a temporal window:
        1.30 inside, 0.75 beyond a 30-day grace margin, 1.0 otherwise."""
        if not content_ts:
            return 1.0
        try:
            ts = datetime.fromisoformat(content_ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            start, end = window
            if start <= ts <= end:
                return inside_factor
            margin = timedelta(days=30)
            if ts < start - margin or ts > end + margin:
                return outside_factor
            return 1.0
        except (ValueError, TypeError):
            return 1.0

    @classmethod
    def apply_all(cls, query: str, content: str, score: float, ts: str | None = None) -> float:
        """Apply all applicable heuristics to a single result."""
        s = cls.keyword_overlap(query, content, score)
        s = cls.recency_decay(ts, s)
        s = cls.temporal_boost(query, ts, s)
        s = cls.person_name_boost(content, s)
        s = cls.quoted_phrase_boost(query, content, s)
        return s
