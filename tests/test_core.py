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