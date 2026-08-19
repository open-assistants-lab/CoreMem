"""Tests for retrieval improvements: temporal-neighbor confirmation and
corpus-aware typo robustness. Both zero-LLM."""

from __future__ import annotations

import shutil
import tempfile

from coremem import MemoryCore
from coremem.retrieval import (
    fuzzy_expand_queries,
    search_messages_confirmed,
    search_messages_preference_union,
    search_messages_typo_robust,
)


def _tmp_core() -> MemoryCore:
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


# ── Temporal-neighbor confirmation ─────────────────────────────────────────


def test_confirmation_thread_beats_isolated_high_scorer():
    """A session with a thread of moderate matches should beat a session
    with one strong isolated match (temporal contiguity)."""
    core = _tmp_core()
    try:
        # answer session: thread of moderate matches
        core.ingest("user", "Yosemite has great trails", session_id="s1")
        core.ingest("user", "Yosemite is beautiful in spring", session_id="s1")
        # distractor: one strong isolated match
        core.ingest("user", "I love hiking in Yosemite", session_id="s2")

        baseline = core._search_messages_decomposed(
            "hiking Yosemite", limit=1, per_query_limit=20, use_cross_encoder=True,
        )
        confirmed = search_messages_confirmed(core, "hiking Yosemite", limit=1)

        assert baseline[0].memory.session_id == "s2", "baseline picks the isolated high scorer"
        assert confirmed[0].memory.session_id == "s1", "confirmation picks the thread"
    finally:
        core._test_cleanup()


def test_confirmation_falls_back_to_baseline_without_neighbors():
    """No neighbors in the window → output identical to baseline."""
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("user", "I love hiking in Tahoe", session_id="s2")

        baseline = core._search_messages_decomposed(
            "hiking Yosemite", limit=2, per_query_limit=20, use_cross_encoder=True,
        )
        confirmed = search_messages_confirmed(core, "hiking Yosemite", limit=2)

        assert [r.memory.id for r in confirmed] == [r.memory.id for r in baseline]
    finally:
        core._test_cleanup()


# ── Corpus-aware typo robustness ────────────────────────────────────────────


def test_fuzzy_expand_queries_corrects_typos():
    vocabulary = {"yosemite", "hiking", "tahoe", "coffee", "beautiful"}
    variants = fuzzy_expand_queries("hiking in Yosemitee", vocabulary)
    assert any("yosemite" in v.lower() for v in variants), f"got {variants}"


def test_fuzzy_expand_queries_ignores_clean_queries():
    vocabulary = {"yosemite", "hiking", "tahoe"}
    variants = fuzzy_expand_queries("hiking in Yosemite", vocabulary)
    assert variants == []


def test_typo_robust_matches_baseline_ranking():
    """The hybrid search's vector leg already covers typos; the typo-robust
    variant must not regress the ranking (documented finding: typo
    expansion only matters for FTS5-only retrieval)."""
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("user", "I love hiking in Tahoe", session_id="s2")

        baseline = core._search_messages_decomposed(
            "hiking in Yosemitee", limit=3, per_query_limit=20, use_cross_encoder=True,
        )
        robust = search_messages_typo_robust(core, "hiking in Yosemitee", limit=3)

        assert baseline[0].memory.id == robust[0].memory.id
        assert any("Yosemite" in r.memory.content for r in robust)
    finally:
        core._test_cleanup()


# ── Preference union-retrieval ─────────────────────────────────────────────


def test_preference_union_returns_results_for_preference_query():
    core = _tmp_core()
    try:
        core.ingest("user", "I like hiking in Yosemite", session_id="s1")
        core.ingest("user", "I enjoy painting landscapes", session_id="s1")
        core.ingest("user", "I love coffee", session_id="s1")

        results = search_messages_preference_union(core, "what do I like", limit=5)

        assert results
    finally:
        core._test_cleanup()


def test_preference_union_falls_back_to_baseline_for_non_preference():
    core = _tmp_core()
    try:
        core.ingest("user", "I like hiking in Yosemite", session_id="s1")
        core.ingest("user", "I love hiking in Tahoe", session_id="s2")

        baseline = core._search_messages_decomposed(
            "hiking Yosemite", limit=2, per_query_limit=20, use_cross_encoder=True,
        )
        results = search_messages_preference_union(core, "hiking Yosemite", limit=2)

        assert [r.memory.id for r in results] == [r.memory.id for r in baseline]
    finally:
        core._test_cleanup()


def test_preference_union_surfaces_variant_only_message():
    """A message matching only one preference keyword variant ("enjoy")
    must survive into the results via the per-variant union."""
    core = _tmp_core()
    try:
        core.ingest("user", "I like hiking in Yosemite", session_id="s1")
        core.ingest("user", "I enjoy painting landscapes", session_id="s1")
        core.ingest("user", "I love coffee", session_id="s1")
        core.ingest("user", "I prefer tea over coffee", session_id="s1")

        results = search_messages_preference_union(
            core, "what do I like", limit=5, per_variant=2,
        )

        assert any("painting" in r.memory.content for r in results)
    finally:
        core._test_cleanup()


def test_decomposed_search_routes_preference_queries_to_union(monkeypatch):
    """The default decomposed search must route preference queries through
    the per-variant union (the validated fold-in)."""
    import coremem.core as coremod
    from coremem.retrieval import search_messages_preference_union

    core = _tmp_core()
    try:
        core.ingest("user", "I like hiking in Yosemite", session_id="s1")
        core.ingest("user", "I love coffee", session_id="s2")

        called: list[str] = []
        real = search_messages_preference_union

        def spy(c, query, **kwargs):
            called.append(query)
            return real(c, query, **kwargs)

        monkeypatch.setattr(coremod, "search_messages_preference_union", spy)

        core._search_messages_decomposed("what do I like", limit=2, per_query_limit=20)
        core._search_messages_decomposed("hiking Yosemite", limit=2, per_query_limit=20)

        assert called == ["what do I like"], "preference query routes to the union"
    finally:
        core._test_cleanup()
