"""Tests for heuristics layer."""

from coremem.heuristics import SearchHeuristics, _mmr_diversify
from coremem.types import Memory, SearchResult


def test_keyword_overlap_boosts_when_words_match():
    s = SearchHeuristics.keyword_overlap(
        query="how many model kits",
        content="I love building model kits and painting them",
        score=0.5,
    )
    assert s > 0.50


def test_keyword_overlap_no_boost_when_no_overlap():
    s = SearchHeuristics.keyword_overlap(
        query="model kits",
        content="I enjoy eating pizza with friends",
        score=0.5,
    )
    assert s == 0.50


def test_keyword_overlap_ignores_stop_words():
    s = SearchHeuristics.keyword_overlap(
        query="how many model kits have I worked on",
        content="I worked on many different things",
        score=0.5,
    )
    # "worked" is not a stop word, "many" is borderline
    assert s >= 0.50


def test_is_counting_question():
    assert SearchHeuristics.is_counting_question("How many model kits?")
    assert SearchHeuristics.is_counting_question("How many weeks did it take?")
    assert not SearchHeuristics.is_counting_question("What is my name?")
    assert not SearchHeuristics.is_counting_question("Where do I live?")


def test_extract_date_cues():
    assert SearchHeuristics.extract_date_cues("In 2025 I went skiing") == "2025"
    assert SearchHeuristics.extract_date_cues("What happened in March?") == "March"
    assert SearchHeuristics.extract_date_cues("No date here") is None


def test_person_name_boost():
    s = SearchHeuristics.person_name_boost("I met Sarah Johnson at the conference", 0.5)
    assert s > 0.50

    s2 = SearchHeuristics.person_name_boost("the quick brown fox", 0.5)
    assert s2 == 0.50


def test_quoted_phrase_boost():
    s = SearchHeuristics.quoted_phrase_boost(
        query='my "model kits" hobby',
        content="I spend weekends on model kits painting",
        score=0.5,
    )
    assert s > 0.50

    s2 = SearchHeuristics.quoted_phrase_boost(
        query='my "model kits" hobby',
        content="I enjoy painting miniatures",
        score=0.5,
    )
    assert s2 == 0.50


def test_apply_all_chains():
    s = SearchHeuristics.apply_all(
        query="how many model kits",
        content="I love building model kits. Finished a Revell F-15 Eagle and Tamiya Spitfire.",
        score=0.7,
    )
    assert s > 0.70


# ── MMR Tests ──


def _make_result(content: str, score: float, session_id: str | None = None) -> SearchResult:
    return SearchResult(
        memory=Memory(id=f"id_{hash(content) & 0xFFFF}", content=content,
                      role="user", session_id=session_id),
        score=score,
    )


def test_mmr_no_op_when_all_same_session():
    results = [
        _make_result("msg A1", 0.9, "s1"),
        _make_result("msg A2", 0.8, "s1"),
        _make_result("msg A3", 0.7, "s1"),
    ]
    diverse = _mmr_diversify(results, k=5)
    assert len(diverse) == 3  # Fewer results than K
    assert diverse[0].memory.content == "msg A1"


def test_mmr_dedups_across_sessions():
    results = [
        _make_result("A1", 0.9, "s1"),
        _make_result("A2", 0.8, "s1"),
        _make_result("B1", 0.7, "s2"),
        _make_result("C1", 0.6, "s3"),
        _make_result("A3", 0.5, "s1"),
    ]
    diverse = _mmr_diversify(results, k=3)
    sessions = {r.memory.session_id for r in diverse}
    assert len(sessions) == 3
    assert sessions == {"s1", "s2", "s3"}
    # Score order preserved: s1 first, s2 second, s3 third
    assert diverse[0].memory.content == "A1"
    assert diverse[1].memory.content == "B1"


def test_mmr_sessionless_gets_unique_keys():
    results = [
        _make_result("msg 1", 0.9, None),
        _make_result("msg 2", 0.8, None),
        _make_result("msg 3", 0.7, None),
    ]
    diverse = _mmr_diversify(results, k=3)
    assert len(diverse) == 3  # Each gets a unique synthetic key


def test_mmr_sessionless_same_content_dedups():
    results = [
        _make_result("same", 0.9, None),
        _make_result("same", 0.8, None),
        _make_result("diff", 0.7, None),
    ]
    # k=2: picks "same" + "diff" (2 unique keys). "same" duplicate goes to overflow.
    diverse = _mmr_diversify(results, k=2)
    assert len(diverse) == 2
    assert diverse[0].memory.content == "same"
    assert diverse[1].memory.content == "diff"


def test_mmr_preserves_score_order():
    # MMR expects score-sorted input (pipeline pre-sorts)
    results = [
        _make_result("A1", 0.9, "s1"),
        _make_result("B1", 0.8, "s2"),
        _make_result("C1", 0.7, "s3"),
    ]
    diverse = _mmr_diversify(results, k=3)
    scores = [r.score for r in diverse]
    assert scores == [0.9, 0.8, 0.7]  # Score-sorted within diverse set


def test_mmr_fewer_sessions_than_k():
    results = [
        _make_result("A1", 0.9, "s1"),
        _make_result("B1", 0.6, "s2"),
        _make_result("A2", 0.5, "s1"),
        _make_result("A3", 0.4, "s1"),
    ]
    diverse = _mmr_diversify(results, k=4)
    assert len(diverse) == 4
    # First 2: s1, s2 (diverse)
    # Remaining 2: A2, A3 (overflow, score-ordered)
    assert diverse[0].memory.content == "A1"
    assert diverse[1].memory.content == "B1"
    assert diverse[2].memory.content == "A2"
    assert diverse[3].memory.content == "A3"


def test_mmr_empty():
    assert _mmr_diversify([], k=5) == []


def test_mmr_k_zero():
    results = [_make_result("A", 0.9, "s1")]
    assert _mmr_diversify(results, k=0) == []
