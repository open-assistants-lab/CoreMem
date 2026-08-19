from __future__ import annotations

import shutil
import tempfile

from coremem import MemoryCore, decompose_queries
from coremem.types import Memory, SearchResult


def test_decompose_queries_extracts_quoted_events():
    queries = decompose_queries(
        "How many days passed between the 'Walk for Hunger' event "
        "and the 'Coastal Cleanup' event?"
    )

    assert "Walk for Hunger" in queries
    assert "Coastal Cleanup" in queries


def test_decompose_queries_splits_before_clause():
    queries = decompose_queries(
        "How many days before I bought my iPad did I attend the Holiday Market?"
    )

    assert "I bought my iPad" in queries
    assert "attend the Holiday Market" in queries


def test_decompose_queries_splits_from_to_clause():
    queries = decompose_queries(
        "How many days passed from the day I started watering my herb garden "
        "to the day I harvested my first tomato?"
    )

    assert any("started watering my herb garden" in q for q in queries)
    assert any("harvested my first tomato" in q for q in queries)


def test_decompose_queries_splits_since_when_clause():
    queries = decompose_queries(
        "How many days had passed since I finished reading the book "
        "when I attended the concert?"
    )

    assert any("finished reading the book" in q for q in queries)
    assert any("attended the concert" in q for q in queries)


def test_decompose_queries_cleans_ago_event_cue():
    queries = decompose_queries(
        "How many days ago did I attend the Maundy Thursday service "
        "at the Episcopal Church?"
    )

    assert any("attend the Maundy Thursday service" in q for q in queries)
    assert not any(q.strip() == "days ago" for q in queries)


def test_decomposed_search_fuses_independent_cues():
    root = tempfile.mkdtemp()
    core = MemoryCore(path=root)
    try:
        alpha = SearchResult(
            memory=Memory(id="a1", content="Alpha event", role="user", session_id="a"),
            score=1.0,
        )
        beta = SearchResult(
            memory=Memory(id="b1", content="Beta event", role="user", session_id="b"),
            score=1.0,
        )

        def search(query, limit=10, **kwargs):
            if query.lower() == "alpha":
                return [alpha]
            if query.lower() == "beta":
                return [beta]
            return [alpha]

        core._search_messages = search

        results = core._search_messages_decomposed(
            "Which happened first, 'Alpha' or 'Beta'?", limit=2,
        )

        assert {result.memory.id for result in results} == {"a1", "b1"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_decomposed_search_optionally_uses_cross_encoder(monkeypatch):
    root = tempfile.mkdtemp()
    core = MemoryCore(path=root)
    try:
        first = SearchResult(
            memory=Memory(id="a", content="first", role="user", session_id="a"),
            score=1.0,
        )
        second = SearchResult(
            memory=Memory(id="b", content="second", role="user", session_id="b"),
            score=0.5,
        )
        core._search_messages = lambda query, limit=10, **kwargs: [first, second]
        monkeypatch.setattr("coremem.core.rerank", lambda query, results: list(reversed(results)))

        results = core._search_messages_decomposed(
            "camera lens", limit=2, use_cross_encoder=True,
        )

        assert [result.memory.id for result in results] == ["b", "a"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
