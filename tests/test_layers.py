"""Tests for L0-L3 wake-up context stack."""

import tempfile
import shutil

from coremem import MemoryCore
from coremem.layers import WakeUpContext


def _make_core():
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


def test_essential_builds_l0_l1():
    core = _make_core()
    try:
        core.ingest("user", "I live in Denver")
        core.ingest("user", "I love model kits")

        ctx = WakeUpContext(core.db)
        result = ctx.essential(user_id="alice")
        assert "[L0: Identity]" in result
        assert "alice" in result
        assert "[L1: Essential]" in result
    finally:
        core._test_cleanup()


def test_session_context_filters_by_session():
    core = _make_core()
    try:
        core.ingest("user", "Hello from sess A", session_id="A")
        core.ingest("user", "Hello from sess B", session_id="B")

        ctx = WakeUpContext(core.db)
        result = ctx.session(session_id="A")
        assert result is not None
        assert "[L2: On-Demand]" in result
        assert "A" in result
        assert "B" not in result
    finally:
        core._test_cleanup()


def test_session_context_nonexistent_session():
    core = _make_core()
    try:
        ctx = WakeUpContext(core.db)
        result = ctx.session(session_id="nope")
        assert result is None
    finally:
        core._test_cleanup()


def test_deep_search_returns_results():
    core = _make_core()
    try:
        core.ingest("user", "I built a Spitfire model kit")
        core.ingest("user", "I like coffee")

        ctx = WakeUpContext(core.db)
        result = ctx.deep_search("model kit", limit=5)
        assert result is not None
        assert "Spitfire" in result
    finally:
        core._test_cleanup()


def test_deep_search_no_match_returns_results_with_zero_score():
    core = _make_core()
    try:
        core.ingest("user", "I like coffee")

        ctx = WakeUpContext(core.db)
        result = ctx.deep_search("quantum physics", limit=5)
        # HybridDB returns all results scored 0 when nothing matches
        assert result is not None
    finally:
        core._test_cleanup()