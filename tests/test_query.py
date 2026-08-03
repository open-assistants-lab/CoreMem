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

        core.search_messages = search

        results = core.search_messages_decomposed(
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
        core.search_messages = lambda query, limit=10, **kwargs: [first, second]
        monkeypatch.setattr("coremem.core.rerank", lambda query, results: list(reversed(results)))

        results = core.search_messages_decomposed(
            "camera lens", limit=2, use_cross_encoder=True,
        )

        assert [result.memory.id for result in results] == ["b", "a"]
    finally:
        shutil.rmtree(root, ignore_errors=True)
