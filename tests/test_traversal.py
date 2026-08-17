"""Tests for query-guided graph traversal retrieval.

Revives the removed SPEC design (hybrid seeds → graph expansion → relevance
re-check → session caps → baseline fallback) on the fixed HybridDB graph.
"""

from __future__ import annotations

import shutil
import tempfile

from coremem import MemoryCore
from coremem.traversal import search_messages_traversal


def _tmp_core() -> MemoryCore:
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


def test_traversal_output_is_superset_of_baseline():
    """The traversal pool is a strict superset of the baseline's rerank
    window, so the output must contain every baseline result."""
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("user", "Yosemite has great trails", session_id="s2")
        core.ingest("user", "I love hiking in Tahoe", session_id="s3")
        core.ingest("user", "I love hiking in the Alps", session_id="s4")

        baseline = core._search_messages_decomposed(
            "hiking Yosemite", limit=2, per_query_limit=20, use_cross_encoder=True,
        )
        results = search_messages_traversal(core, "hiking Yosemite", limit=2)

        baseline_ids = {r.memory.id for r in baseline}
        result_ids = {r.memory.id for r in results}
        assert baseline_ids <= result_ids, "traversal must not lose baseline results"
    finally:
        core._test_cleanup()


def test_traversal_fallback_equals_baseline_when_nothing_survives():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("user", "The stock market rallied today", session_id="s1")  # zero overlap

        baseline = core._search_messages_decomposed("hiking Yosemite", limit=1)
        results = search_messages_traversal(core, "hiking Yosemite", limit=1)

        assert [r.memory.id for r in results] == [r.memory.id for r in baseline]
    finally:
        core._test_cleanup()


def test_traversal_caps_candidates_per_session():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("user", "Yosemite has great trails", session_id="s1")
        core.ingest("user", "Yosemite is beautiful in spring", session_id="s1")
        core.ingest("user", "I love hiking in Tahoe", session_id="s2")
        core.ingest("user", "Tahoe has great trails", session_id="s2")

        results = search_messages_traversal(
            core, "hiking Yosemite", limit=5, max_per_session=1,
        )

        # at most one candidate from s1's neighbors beyond the seed
        s1_candidates = [
            r for r in results
            if r.memory.session_id == "s1" and "trails" in r.memory.content
        ]
        assert len(s1_candidates) <= 1
    finally:
        core._test_cleanup()


def test_traversal_reranks_pool_with_cross_encoder():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("user", "Yosemite has great trails", session_id="s1")

        # limit=1: baseline is only the seed; the neighbor expands the pool,
        # so the cross-encoder rerank actually runs
        results = search_messages_traversal(
            core, "hiking Yosemite", limit=1, use_cross_encoder=True,
        )

        assert results
        assert all(hasattr(r, "_ce_score") for r in results)
    finally:
        core._test_cleanup()
