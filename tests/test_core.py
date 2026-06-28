"""Tests for MemoryCore with HybridDB."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from coremem import MemoryCore
from coremem.agent_journal import AgentJournalCompileResult, AgentJournalError
from coremem.types import Memory


def _tmp_core(**kwargs) -> MemoryCore:
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d, **kwargs)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


def test_core_ingest_and_search_messages():
    core = _tmp_core()
    try:
        core.ingest("user", "I built a Spitfire model kit")
        core.ingest("assistant", "That sounds fun!")
        assert core.count() == 2
        results = core.search_messages("model kits", limit=10)
        assert len(results) > 0
        assert any("model" in r.memory.content.lower() for r in results)
    finally:
        core._test_cleanup()


def test_core_search_messages_deep():
    core = _tmp_core()
    try:
        core.ingest("user", "I built a Spitfire model kit")
        core.ingest("assistant", "That sounds fun!")
        results = core.search_messages_deep("model kits", limit=10)
        assert len(results) > 0
        assert any("model" in r.memory.content.lower() for r in results)
    finally:
        core._test_cleanup()


def test_core_search_messages_filters_session_when_provided():
    core = _tmp_core()
    try:
        core.ingest("user", "coffee preference from session one", session_id="s1")
        core.ingest("user", "coffee preference from session two", session_id="s2")

        scoped = core.search_messages("coffee preference", session_id="s1", limit=10)
        assert scoped
        assert all(result.memory.session_id == "s1" for result in scoped)

        global_results = core.search_messages("coffee preference", limit=10)
        assert {result.memory.session_id for result in global_results} >= {"s1", "s2"}
    finally:
        core._test_cleanup()


def test_old_search_names_are_removed():
    core = _tmp_core()
    try:
        assert not hasattr(core, "search")
        assert not hasattr(core, "search_enhanced")
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


def test_compile_turn_derives_timestamp_and_allows_default_title():
    core = _tmp_core()
    calls = []

    async def fake_compile_session(**kwargs):
        calls.append(kwargs)
        return AgentJournalCompileResult(written_pages=(Path("daily.md"),), boot_pages=())

    try:
        core._journal_compiler.compile_session = fake_compile_session
        ts = datetime(2026, 6, 28, 14, 5, tzinfo=UTC)
        turn_id = core.ingest("user", "I like coffee", session_id="s1", ts=ts)
        core.ingest("assistant", "Noted.", session_id="s1", ts=ts + timedelta(minutes=1))

        result = asyncio.run(core.compile_turn(turn_id))

        assert calls
        assert result is not None
        call = calls[0]
        assert call["turn_id"] == turn_id
        assert call["session_id"] == "s1"
        assert call["timestamp"] == "14:05"
        assert call["title"] is None
        assert [m["role"] for m in call["messages"]] == ["user", "assistant"]
    finally:
        core._test_cleanup()


def test_compile_turn_records_and_skips_unchanged_turn():
    core = _tmp_core()
    calls = []

    async def fake_compile_session(**kwargs):
        calls.append(kwargs)
        return AgentJournalCompileResult(written_pages=(Path("daily.md"),), boot_pages=())

    try:
        core._journal_compiler.compile_session = fake_compile_session
        turn_id = core.ingest("user", "I like coffee", session_id="s1")
        core.ingest("assistant", "Noted.", session_id="s1")

        first = asyncio.run(core.compile_turn(turn_id))
        second = asyncio.run(core.compile_turn(turn_id))

        assert first is not None
        assert second is None
        assert len(calls) == 1
        rows = core.db.raw_query("SELECT * FROM compiled_turns WHERE turn_id = ?", (turn_id,))
        assert len(rows) == 1
        assert rows[0]["message_count"] == 2
    finally:
        core._test_cleanup()


def test_compile_turn_raises_when_compiled_source_changed_without_force():
    core = _tmp_core()
    calls = []

    async def fake_compile_session(**kwargs):
        calls.append(kwargs)
        return AgentJournalCompileResult(written_pages=(Path("daily.md"),), boot_pages=())

    try:
        core._journal_compiler.compile_session = fake_compile_session
        turn_id = core.ingest("user", "I like coffee", session_id="s1")
        asyncio.run(core.compile_turn(turn_id))
        core.ingest("assistant", "Noted.", session_id="s1")

        try:
            asyncio.run(core.compile_turn(turn_id))
        except AgentJournalError as exc:
            assert "turn changed after compilation" in str(exc)
        else:
            raise AssertionError("expected AgentJournalError")
        assert len(calls) == 1
    finally:
        core._test_cleanup()


def test_compile_turn_force_recompiles_changed_turn():
    core = _tmp_core()
    calls = []

    async def fake_compile_session(**kwargs):
        calls.append(kwargs)
        return AgentJournalCompileResult(written_pages=(Path("daily.md"),), boot_pages=())

    try:
        core._journal_compiler.compile_session = fake_compile_session
        turn_id = core.ingest("user", "I like coffee", session_id="s1")
        asyncio.run(core.compile_turn(turn_id))
        core.ingest("assistant", "Noted.", session_id="s1")

        result = asyncio.run(core.compile_turn(turn_id, force=True))

        assert result is not None
        assert len(calls) == 2
        rows = core.db.raw_query("SELECT * FROM compiled_turns WHERE turn_id = ?", (turn_id,))
        assert rows[0]["message_count"] == 2
    finally:
        core._test_cleanup()


def test_compile_latest_turn_compiles_most_recent_session_turn():
    core = _tmp_core()
    calls = []

    async def fake_compile_session(**kwargs):
        calls.append(kwargs)
        return AgentJournalCompileResult(written_pages=(Path("daily.md"),), boot_pages=())

    try:
        core._journal_compiler.compile_session = fake_compile_session
        base = datetime(2026, 6, 28, 10, 0, tzinfo=UTC)
        old_turn = core.ingest("user", "old", session_id="s1", ts=base)
        new_turn = core.ingest("user", "new", session_id="s1", ts=base + timedelta(minutes=1))

        result = asyncio.run(core.compile_latest_turn("s1"))

        assert result is not None
        assert calls[0]["turn_id"] == new_turn
        assert calls[0]["turn_id"] != old_turn
    finally:
        core._test_cleanup()


def test_compile_uncompiled_turns_compiles_each_turn_once():
    core = _tmp_core()
    calls = []

    async def fake_compile_session(**kwargs):
        calls.append(kwargs)
        return AgentJournalCompileResult(written_pages=(Path("daily.md"),), boot_pages=())

    try:
        core._journal_compiler.compile_session = fake_compile_session
        first = core.ingest("user", "first", session_id="s1")
        second = core.ingest("user", "second", session_id="s1")

        summary = asyncio.run(core.compile_uncompiled_turns())
        again = asyncio.run(core.compile_uncompiled_turns())

        assert summary["compiled"] == [first, second]
        assert again["compiled"] == []
        assert again["skipped"] == [first, second]
        assert len(calls) == 2
    finally:
        core._test_cleanup()


def test_compile_uncompiled_turns_reports_errors_without_aborting_batch():
    core = _tmp_core()

    async def fake_compile_session(**kwargs):
        if kwargs["messages"][0]["content"] == "bad":
            raise RuntimeError("boom")
        return AgentJournalCompileResult(written_pages=(Path("daily.md"),), boot_pages=())

    try:
        core._journal_compiler.compile_session = fake_compile_session
        bad = core.ingest("user", "bad", session_id="s1")
        good = core.ingest("user", "good", session_id="s1")

        summary = asyncio.run(core.compile_uncompiled_turns())

        assert summary["compiled"] == [good]
        assert summary["errors"] == [{"turn_id": bad, "error": "boom"}]
    finally:
        core._test_cleanup()
