"""Regression tests for the 2026-08-23 E2E bug hunt findings (B1–B8).

Each test reproduces a confirmed finding and encodes the fixed behavior:
- B1: preference queries must honor recall filters (session/role/ts/...)
- B2: session_cap selection must honor filters in its cross-encoder pool
- B3: fusion strategy must honor filters
- B4: ingest() must return the turn_id that was actually stored
- B5: public methods must raise after close()
- B6: stats() must not count the empty-session bucket
- B7: MCP metadata parsing must surface invalid JSON instead of dropping it
- B8: recall() must reject empty queries loudly
"""

from __future__ import annotations

import shutil
import tempfile

import pytest

from coremem import MemoryCore


def _tmp_core(**kwargs) -> MemoryCore:
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d, **kwargs)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


# ── B1: preference union honors filters ────────────────────────────────────


def test_preference_query_respects_session_filter():
    core = _tmp_core()
    try:
        core.ingest("user", "I like hiking and I enjoy jazz music", session_id="sessA")
        core.ingest("user", "My favorite food is sushi", session_id="sessB")
        results = core.recall("what do i like to do?", session_id="sessA", limit=5)
        assert results, "preference query with filter should still return matches"
        assert all(r.memory.session_id == "sessA" for r in results), (
            f"filter bypass: {[r.memory.session_id for r in results]}"
        )
    finally:
        core._test_cleanup()


def test_preference_query_respects_session_filter_in_bundles():
    core = _tmp_core()
    try:
        core.ingest("user", "I like hiking and I enjoy jazz music", session_id="sessA")
        core.ingest("user", "My favorite food is sushi", session_id="sessB")
        bundles = core.recall(
            "what do i like to do?", session_id="sessA", bundles=True, limit=5,
        )
        assert bundles
        assert all(b.session_id == "sessA" for b in bundles), (
            f"bundle filter bypass: {[b.session_id for b in bundles]}"
        )
    finally:
        core._test_cleanup()


def test_preference_union_fallback_respects_filters():
    """The non-preference fallback inside the union must forward filters too."""
    from coremem.retrieval import search_messages_preference_union

    core = _tmp_core()
    try:
        core.ingest("user", "hiking in Yosemite was great", session_id="sessA")
        core.ingest("user", "sushi dinner in Tokyo", session_id="sessB")
        results = search_messages_preference_union(
            core, "hiking Yosemite", limit=5, session_id="sessA",
        )
        assert all(r.memory.session_id == "sessA" for r in results), (
            f"fallback filter bypass: {[r.memory.session_id for r in results]}"
        )
    finally:
        core._test_cleanup()


# ── B2: session_cap pool honors filters ────────────────────────────────────


def test_session_cap_respects_role_filter():
    core = _tmp_core()
    try:
        core.ingest("user", "We chose the blue theme for the app redesign", session_id="cap1")
        core.ingest("assistant", "Great choice! The blue theme will look fantastic.", session_id="cap1")
        results = core.recall("blue theme", role="user", session_cap=2, limit=5)
        roles = [r.memory.role for r in results]
        assert all(role == "user" for role in roles), f"role leak under cap: {roles}"
    finally:
        core._test_cleanup()


# ── B3: fusion honors filters ──────────────────────────────────────────────


def test_fusion_respects_session_filter():
    core = _tmp_core()
    try:
        core.ingest("user", "hiking trip planning notes", session_id="sessA")
        core.ingest("user", "sushi restaurant recommendation list", session_id="sessB")
        results = core.recall("sushi restaurant", strategy="fusion", session_id="sessB", limit=5)
        sessions = {r.memory.session_id for r in results}
        assert sessions <= {"sessB"}, f"fusion filter bypass: {sessions}"
    finally:
        core._test_cleanup()


# ── B4: ingest returns the stored turn_id ──────────────────────────────────


def test_ingest_returns_usable_turn_id_after_store():
    core = _tmp_core()
    try:
        from coremem.types import Memory

        core.store([Memory(id="m1", content="stored memory", role="user", session_id="sx")])
        tid = core.ingest("assistant", "reply in sx", session_id="sx")
        assert tid, "ingest returned an empty turn_id"
        rows = core.db.raw_query(
            "SELECT turn_id FROM messages WHERE role='assistant' AND session_id='sx'"
        )
        assert rows and rows[0]["turn_id"] == tid, (
            f"returned turn_id {tid!r} != stored {rows[0]['turn_id']!r}"
        )
    finally:
        core._test_cleanup()


# ── B5: use-after-close raises ──────────────────────────────────────────────


def test_public_methods_raise_after_close():
    core = _tmp_core()
    core.ingest("user", "before close", session_id="s1")
    core.close()
    with pytest.raises(RuntimeError):
        core.ingest("user", "post-close", session_id="s1")
    with pytest.raises(RuntimeError):
        core.recall("anything")
    with pytest.raises(RuntimeError):
        core.fetch(session_id="s1")
    with pytest.raises(RuntimeError):
        core.stats()
    # close() is idempotent
    core.close()


# ── B6: stats excludes the empty-session bucket ────────────────────────────


def test_stats_excludes_empty_session_bucket():
    core = _tmp_core()
    try:
        core.ingest("user", "with session", session_id="real1")
        core.ingest("user", "without session")
        s = core.stats()
        assert s["messages"] == 2
        assert s["sessions"] == 1, f"empty bucket counted as session: {s}"
    finally:
        core._test_cleanup()


# ── B7: MCP metadata parsing surfaces errors ───────────────────────────────


def test_parse_metadata_arg_valid_and_invalid():
    from coremem.mcp_server import parse_metadata_arg

    meta, err = parse_metadata_arg('{"tone": "dark"}')
    assert err is None
    assert meta == {"tone": "dark"}

    meta, err = parse_metadata_arg("")
    assert err is None
    assert meta == {}

    meta, err = parse_metadata_arg("{not json")
    assert meta == {}
    assert err is not None and "metadata" in err.lower()

    meta, err = parse_metadata_arg('["not", "an object"]')
    assert meta == {} and err is not None


# ── B8: empty query rejected loudly ────────────────────────────────────────


@pytest.mark.parametrize("strategy", ["direct", "episodic", "fusion"])
def test_recall_rejects_empty_query(strategy):
    core = _tmp_core()
    try:
        core.ingest("user", "seed message", session_id="s")
        with pytest.raises(ValueError):
            core.recall("", strategy=strategy, limit=3)
        with pytest.raises(ValueError):
            core.recall("   ", strategy=strategy, limit=3)
    finally:
        core._test_cleanup()
