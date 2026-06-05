"""Tests for MemoryCore with HybridDB."""

from __future__ import annotations

import json
import tempfile
import shutil

from coremem import MemoryCore
from coremem.types import Memory


def _tmp_core(**kwargs) -> MemoryCore:
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d, **kwargs)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


def test_core_ingest_and_search():
    core = _tmp_core()
    try:
        core.ingest("user", "I built a Spitfire model kit")
        core.ingest("assistant", "That sounds fun!")
        assert core.count() == 2
        results = core.search("model kits", limit=10)
        assert len(results) > 0
        assert any("model" in r.memory.content.lower() for r in results)
    finally:
        core._test_cleanup()


def test_core_ingest_single():
    core = _tmp_core()
    try:
        mid = core.ingest("user", "Hello world")
        assert mid
        assert core.count() == 1
    finally:
        core._test_cleanup()


def test_core_wake_up():
    core = _tmp_core()
    try:
        core.ingest("user", "I like coffee", session_id="s1")
        core.ingest("assistant", "Great!", session_id="s1")
        ctx = core.wake_up(user_id="alice")
        assert "[L0: Identity]" in ctx
    finally:
        core._test_cleanup()


def test_core_fetch():
    core = _tmp_core()
    try:
        core.ingest("user", "msg1", session_id="s1")
        core.ingest("user", "msg2", session_id="s1")
        msgs = core.fetch(session_id="s1")
        assert len(msgs) == 2
    finally:
        core._test_cleanup()


def test_core_fetch_all():
    core = _tmp_core()
    try:
        for i in range(5):
            core.ingest("user", f"msg{i}", session_id=f"s{i % 2}")
        msgs = core.fetch_all()
        assert len(msgs) == 5
    finally:
        core._test_cleanup()


def test_core_count():
    core = _tmp_core()
    try:
        assert core.count() == 0
        core.ingest("user", "hello")
        assert core.count() == 1
    finally:
        core._test_cleanup()


def test_core_delete():
    core = _tmp_core()
    try:
        core.ingest("user", "to_delete", session_id="s1")
        core.ingest("user", "keep", session_id="s2")
        assert core.count() == 2
        deleted = core.delete(session_id="s1")
        assert deleted == 1
        assert core.count() == 1
    finally:
        core._test_cleanup()


def test_core_clear():
    core = _tmp_core()
    try:
        core.ingest("user", "a")
        core.ingest("user", "b")
        core.clear()
        assert core.count() == 0
    finally:
        core._test_cleanup()


# ── Observation tests ──────────────────────────────────


def test_memorycore_observations_disabled():
    core = _tmp_core(enable_observations=False)
    try:
        import pytest
        with pytest.raises(RuntimeError, match="enable_observations"):
            core.get_observations()
    finally:
        core._test_cleanup()


def test_memorycore_observations_enabled():
    core = _tmp_core(enable_observations=True)
    try:
        obs = core.get_observations()
        assert obs == []
    finally:
        core._test_cleanup()


def test_insert_and_retrieve_observations():
    core = _tmp_core(enable_observations=True)
    try:
        ids = core.insert_observations([{
            "content": "User likes coffee",
            "source_quote": "I like coffee",
            "importance": 0.6,
        }])
        assert len(ids) == 1
        obs = core.get_observations(limit=10)
        assert len(obs) == 1
        assert obs[0]["content"] == "User likes coffee"
    finally:
        core._test_cleanup()


def test_search_observations():
    core = _tmp_core(enable_observations=True)
    try:
        core.insert_observations([{
            "content": "User likes coffee", "source_quote": "coffee",
            "importance": 0.6,
        }])
        results = core.search_observations("coffee", limit=5)
        assert len(results) >= 1
    finally:
        core._test_cleanup()


def test_insert_event_and_create_conflict():
    core = _tmp_core(enable_observations=True)
    try:
        oid = core.insert_observations([{
            "content": "obs A", "source_quote": "a", "importance": 0.5,
        }])[0]
        eid = core.insert_event(observation_id=oid, event_type="created")
        assert eid is not None
        cid = core.create_conflict(oid, oid, "contradiction")
        assert cid is not None
    finally:
        core._test_cleanup()


def test_get_recent_observations_filter():
    core = _tmp_core(enable_observations=True)
    try:
        core.insert_observations([{
            "content": "old fact", "source_quote": "old",
            "importance": 0.3,
            "durability": "durable",
        }])
        recent = core.get_recent_observations(days=1, limit=10)
        assert len(recent) == 1
        assert recent[0]["content"] == "old fact"
    finally:
        core._test_cleanup()


def test_memorycore_store_method():
    core = _tmp_core()
    try:
        memories = [Memory(id="", content="test", role="user", ts=None)]
        ids = core.store(memories)
        assert len(ids) == 1
        assert core.count() == 1
    finally:
        core._test_cleanup()


def test_memorycore_deep_search_context():
    core = _tmp_core()
    try:
        core.ingest("user", "I built a model kit yesterday")
        ctx = core.deep_search_context("model", limit=5)
        assert ctx is not None
        assert "model" in ctx.lower()
    finally:
        core._test_cleanup()