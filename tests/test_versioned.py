"""Tests for versioned memory (hybriddb 0.6.0 hash chains) — docs/versioned-memory-design.md."""

from __future__ import annotations

import json
import shutil
import tempfile

import pytest

from coremem import MemoryCore

def _make_core(**kwargs) -> MemoryCore:
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d, **kwargs)
    core._spike_dir = d  # keep a handle for cleanup
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core

def _sample(n: int = 3) -> list[dict]:
    return [
        {"id": f"m{i}", "role": "user", "content": f"memory {i} about Denver",
         "session_id": "s1", "ts": "2024-01-01T00:00:00+00:00"}
        for i in range(n)
    ]

def test_versioned_store_records_ingest_history():
    core = _make_core(versioned=True, author="agent-7")
    try:
        core.ingest_many(_sample(3))
        log = core.memory_log(limit=100)
        assert len(log) == 3
        assert all(e["op"] == "insert" for e in log)
        assert any(e["author"] == "agent-7" for e in log)
        assert core.verify_memory_chain()["messages"]["valid"]
    finally:
        core._test_cleanup()

def test_delete_messages_records_tombstone():
    core = _make_core(versioned=True)
    try:
        core.ingest_many(_sample(3))
        n = core.delete_messages(["m1"])
        assert n == 1
        events = core.memory_history("m1")
        ops = [e["op"] for e in events]
        assert "insert" in ops and "delete" in ops
        # search no longer returns the tombstoned message
        hits = core._search_messages("memory 1", limit=10)
        assert all(h.memory.id != "m1" for h in hits)
        assert core.verify_memory_chain()["messages"]["valid"]
    finally:
        core._test_cleanup()

def test_clear_records_tombstones():
    core = _make_core(versioned=True)
    try:
        core.ingest_many(_sample(3))
        core.clear()
        assert core.count() == 0
        # every message carries a tombstone in history
        for i in range(3):
            ops = [e["op"] for e in core.memory_history(f"m{i}")]
            assert "delete" in ops
        assert core.verify_memory_chain()["messages"]["valid"]
    finally:
        core._test_cleanup()

def test_checkpoint_rollback_workflow():
    core = _make_core(versioned=True)
    try:
        core.ingest_many(_sample(2))
        core.checkpoint_memory("pre-import")
        core.ingest_many([
            {"id": "bad1", "role": "user", "content": "bad import",
             "session_id": "s9", "ts": "2024-02-01T00:00:00+00:00"},
        ])
        assert core.count() == 3
        core.rollback_memory("pre-import")
        assert core.count() == 2
        # audit intact: rollback is recorded, not erased
        events = core.memory_log(limit=1000)
        ops = [e["op"] for e in events]
        assert "insert" in ops  # bad1 stays in history as a version
        hist = core.memory_history("bad1")
        assert len(hist) >= 1
        chain = core.verify_memory_chain()
        assert chain["messages"]["valid"] and chain["journal_records"]["valid"]
    finally:
        core._test_cleanup()

def test_rollback_requires_label_or_seq():
    core = _make_core(versioned=True)
    try:
        with pytest.raises(ValueError):
            core.rollback_memory()
    finally:
        core._test_cleanup()

def test_governance_methods_require_versioned_flag():
    core = _make_core()  # flag off
    try:
        with pytest.raises(RuntimeError, match="versioned=True"):
            core.memory_log()
        with pytest.raises(RuntimeError, match="versioned=True"):
            core.checkpoint_memory("x")
        with pytest.raises(RuntimeError, match="versioned=True"):
            core.verify_memory_chain()
    finally:
        core._test_cleanup()

def test_versioned_flag_on_legacy_store_warns_and_stays_unversioned(caplog):
    d = tempfile.mkdtemp()
    legacy = MemoryCore(path=d)
    legacy.ingest_many(_sample(2))
    legacy.close()
    with caplog.at_level("WARNING", logger="coremem.core"):
        core = MemoryCore(path=d, versioned=True)
        assert "predates" in caplog.text
        core.ingest_many([
            {"id": "m9", "role": "user", "content": "later memory",
             "session_id": "s1", "ts": "2024-01-02T00:00:00+00:00"},
        ])
        assert core.count() == 3
        with pytest.raises(RuntimeError, match="versioned=True"):
            core.memory_log()
    core.close()
    shutil.rmtree(d, ignore_errors=True)

def test_as_of_and_diff():
    core = _make_core(versioned=True)
    try:
        core.ingest_many(_sample(2))
        snap = core.memory_log(limit=100)
        first_seq = max(int(e["seq"]) for e in snap)
        core.ingest_many([
            {"id": "m9", "role": "user", "content": "newest memory",
             "session_id": "s1", "ts": "2024-01-02T00:00:00+00:00"},
        ])
        # as_of at the earlier seq: m9 not present, m0/m1 present
        as_of = core.as_of_memory(first_seq)
        ids = {r["id"] for r in as_of}
        assert "m9" not in ids
        diff = core.memory_diff(first_seq, first_seq + 1)
        assert diff.get("added") or diff.get("changed") or diff.get("removed") is not None
        response = json.dumps(diff, default=str)
        assert "m9" in response
    finally:
        core._test_cleanup()

def test_author_recorded():
    core = _make_core(versioned=True, author="alice")
    try:
        core.ingest_many(_sample(1))
        event = core.memory_log(limit=10)[0]
        assert event["author"] == "alice"
    finally:
        core._test_cleanup()

def test_default_store_unchanged():
    # flag off: no governance methods, delete path unchanged, tests green
    core = _make_core()
    try:
        core.ingest_many(_sample(2))
        assert core.count() == 2
        assert core.delete_messages(["m0"]) == 1
        assert core.count() == 1
        with pytest.raises(RuntimeError):
            core.memory_history("m1")
    finally:
        core._test_cleanup()
