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
from coremem.types import Memory, SearchResult


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


def test_core_search_messages_llm_expansion():
    core = _tmp_core()
    try:
        core.ingest("user", "I built a Spitfire model kit")
        core.ingest("assistant", "That sounds fun!")
        results = core.search_messages_llm_expansion("model kits", limit=10)
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


def test_memorycore_accepts_custom_agent_journal_model():
    core = _tmp_core(agent_journal_model="deepseek:test-model")
    try:
        assert core._journal_compiler.provider._model == "test-model"
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
        assert call["timestamp"] == "2026-06-28 14:05"
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


def test_core_store_journal_record_and_search_with_context():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in the mountains", session_id="session_a")
        core.ingest("assistant", "Me too! The views are amazing.", session_id="session_a")
        core.ingest("user", "What about skiing?", session_id="session_b")
        core.ingest("assistant", "Skiing is great too.", session_id="session_b")

        core._store_journal_record("session_a", "# Summary\n\nThe user enjoys hiking in the mountains and the assistant agreed.")
        core._store_journal_record("session_b", "# Summary\n\nThe user asked about skiing and the assistant confirmed it's great.")

        results = core.search_with_context("hiking mountains", k_sessions=5, k_messages=5)
        assert len(results) >= 1
        assert any("hiking" in r.memory.content.lower() for r in results)

        results = core.search_with_context("skiing", k_sessions=5, k_messages=5)
        assert len(results) >= 1
        assert any("Skiing" in r.memory.content for r in results)

        results = core.search_with_context("unknown_topic", k_sessions=5, k_messages=5)
        assert len(results) == 0
    finally:
        core._test_cleanup()


def test_core_journal_records_table():
    core = _tmp_core()
    try:
        assert "journal_records" in core.db.list_tables()
        core._store_journal_record("s1", "# Summary\n\nTest record")
        rows = core.db.raw_query("SELECT * FROM journal_records")
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s1"
        assert "Test record" in rows[0]["content"]
        core._store_journal_record("s1", "# Summary\n\nUpdated record")
        rows = core.db.raw_query("SELECT * FROM journal_records")
        assert len(rows) == 1, "upsert should replace, not add"
    finally:
        core._test_cleanup()


def test_search_with_context_adds_temporal_neighbors():
    core = _tmp_core()
    try:
        core.ingest("user", "We discussed planning a mountain trip.", session_id="session_a")
        core.ingest("assistant", "The Lake Tahoe permit detail is the exact answer.", session_id="session_a")
        core.ingest("user", "Remember to pack bear spray and a warm jacket.", session_id="session_a")
        core.ingest("user", "We discussed cooking pasta.", session_id="session_b")

        core._store_journal_record(
            "session_a",
            "# Summary\n\nMountain trip planning involving a Lake Tahoe permit and packing advice.",
        )
        core._store_journal_record("session_b", "# Summary\n\nCooking pasta for dinner.")

        results = core.search_with_context(
            "Lake Tahoe permit",
            k_sessions=1,
            k_messages=2,
            context_window=1,
        )
        contents = [r.memory.content for r in results]

        assert any("exact answer" in content for content in contents)
        assert any("mountain trip" in content or "bear spray" in content for content in contents)
    finally:
        core._test_cleanup()


def _insert_traversal_message(core, message_id, content, session_id):
    core.db.insert("messages", {
        "id": message_id,
        "role": "user",
        "content": content,
        "session_id": session_id,
        "turn_id": message_id,
        "metadata": "{}",
        "ts": "2025-01-01T00:00:00+00:00",
    })


def _seed(message_id, content, session_id, score):
    return SearchResult(
        memory=Memory(id=message_id, content=content, role="user", session_id=session_id),
        score=score,
    )


def test_reconstruct_sessions_returns_complete_short_session():
    core = _tmp_core()
    try:
        for mid, content in (("m1", "opening"), ("m2", "answer"), ("m3", "ending")):
            _insert_traversal_message(core, mid, content, "s1")
        core.search_messages_decomposed = lambda query, limit=5: [
            _seed("m2", "answer", "s1", 1.0),
        ]

        bundles = core.reconstruct_sessions("answer")

        assert len(bundles) == 1
        assert bundles[0].complete is True
        assert [message.id for message in bundles[0].messages] == ["m1", "m2", "m3"]
    finally:
        core._test_cleanup()


def test_reconstruct_sessions_combines_opening_and_anchor_segments():
    core = _tmp_core()
    try:
        for index in range(8):
            _insert_traversal_message(core, f"m{index}", f"message {index} " + "x" * 500, "s1")
        core.search_messages_decomposed = lambda query, limit=5: [
            _seed("m6", "answer", "s1", 1.0),
        ]

        bundles = core.reconstruct_sessions(
            "answer",
            short_max_chars=1_000,
            segment_max_messages=2,
            segment_max_chars=2_000,
        )

        assert len(bundles) == 1
        assert bundles[0].complete is False
        assert [message.id for message in bundles[0].messages] == ["m0", "m1", "m6", "m7"]
    finally:
        core._test_cleanup()


def test_reconstruct_sessions_respects_global_budget_across_sessions():
    core = _tmp_core()
    try:
        for session_id in ("s1", "s2"):
            for index in range(3):
                _insert_traversal_message(
                    core,
                    f"{session_id}-m{index}",
                    f"message {index} " + "x" * 600,
                    session_id,
                )
        primary = [
            _seed("s1-m1", "answer one", "s1", 1.0),
            _seed("s2-m1", "answer two", "s2", 0.9),
        ]

        bundles = core.reconstruct_sessions(
            "answer",
            session_limit=2,
            max_context_chars=3_000,
            primary_results=primary,
        )

        assert sum(
            len(message.content) for bundle in bundles for message in bundle.messages
        ) <= 3_000
        assert [message.id for message in bundles[0].messages] == ["s1-m0", "s1-m1"]
        assert [message.id for message in bundles[1].messages] == ["s2-m0", "s2-m1"]
    finally:
        core._test_cleanup()


def test_search_with_traversal_discovers_temporal_neighbor():
    core = _tmp_core()
    try:
        _insert_traversal_message(core, "m1", "Yosemite trip overview", "s1")
        _insert_traversal_message(core, "m2", "The Yosemite permit is the answer", "s1")
        core.search_messages = lambda query, limit=10: [
            _seed("m1", "Yosemite trip overview", "s1", 1.0),
        ]

        results = core.search_with_traversal("Yosemite permit", limit=2)

        assert [r.memory.id for r in results] == ["m1", "m2"]
    finally:
        core._test_cleanup()


def test_search_with_traversal_keeps_strong_seed_over_irrelevant_neighbor():
    core = _tmp_core()
    try:
        _insert_traversal_message(core, "m1", "Yosemite permit details", "s1")
        _insert_traversal_message(core, "m2", "Unrelated pasta recipe", "s1")
        core.search_messages = lambda query, limit=10: [
            _seed("m1", "Yosemite permit details", "s1", 1.0),
        ]

        results = core.search_with_traversal("Yosemite permit", limit=2)

        assert [r.memory.id for r in results] == ["m1"]
    finally:
        core._test_cleanup()


def test_search_with_traversal_preserves_session_diversity():
    core = _tmp_core()
    try:
        for mid, content, sid in (
            ("a1", "alpha one", "a"),
            ("a2", "alpha two", "a"),
            ("a3", "alpha three", "a"),
            ("b1", "beta starting point", "b"),
            ("b2", "beta answer", "b"),
        ):
            _insert_traversal_message(core, mid, content, sid)
        core.search_messages = lambda query, limit=10: [
            _seed("a1", "alpha one", "a", 1.0),
            _seed("a2", "alpha two", "a", 0.9),
            _seed("a3", "alpha three", "a", 0.8),
            _seed("b1", "beta starting point", "b", 0.7),
        ]

        results = core.search_with_traversal(
            "beta answer", limit=5, beam_width=3, max_per_session=2,
        )

        assert "b2" in [r.memory.id for r in results]
    finally:
        core._test_cleanup()


def test_search_with_traversal_is_bounded_and_deterministic():
    core = _tmp_core()
    try:
        for index in range(5):
            _insert_traversal_message(core, f"m{index}", f"topic {index}", "s1")
        core.search_messages = lambda query, limit=10: [
            _seed("m2", "topic 2", "s1", 1.0),
        ]

        first = core.search_with_traversal("topic", limit=3)
        second = core.search_with_traversal("topic", limit=3)

        assert len(first) <= 3
        assert [(r.memory.id, r.score) for r in first] == [
            (r.memory.id, r.score) for r in second
        ]
    finally:
        core._test_cleanup()


def test_search_with_traversal_empty():
    core = _tmp_core()
    try:
        results = core.search_with_traversal("nonexistent query", limit=5)
        assert results == []
    finally:
        core._test_cleanup()


def test_recall_direct_returns_memorycore_results():
    core = _tmp_core()
    try:
        core.ingest("user", "hello world", session_id="s1")
        results = core.recall("hello", strategy="direct", limit=5)
        assert len(results) > 0
        assert "hello" in results[0].memory.content
    finally:
        core._test_cleanup()


def test_recall_default_strategy_returns_results():
    core = _tmp_core()
    try:
        core.ingest("user", "hello world", session_id="s1")
        results = core.recall("hello", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()


def test_recall_episodic_routes_temporal_query():
    core = _tmp_core()
    try:
        core.ingest("user", "event one", session_id="s1")
        core.ingest("user", "event two", session_id="s2")
        results = core.recall("how many days before event one did event two happen", strategy="episodic", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()


def test_recall_unknown_strategy_raises():
    core = _tmp_core()
    try:
        import pytest
        with pytest.raises(ValueError, match="unknown strategy"):
            core.recall("hello", strategy="invalid")
    finally:
        core._test_cleanup()


def test_recall_bundles_returns_session_bundles():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("assistant", "Yosemite is beautiful in spring", session_id="s1")
        core.ingest("user", "What about skiing?", session_id="s2")
        bundles = core.recall("hiking Yosemite", bundles=True)
        assert len(bundles) > 0
        from coremem import SessionBundle
        assert isinstance(bundles[0], SessionBundle)
    finally:
        core._test_cleanup()


def test_recall_default_is_episodic():
    core = _tmp_core()
    try:
        core.ingest("user", "hello world", session_id="s1")
        results = core.recall("hello", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()


def test_recall_fusion_strategy():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("assistant", "Yosemite is beautiful in spring", session_id="s1")
        results = core.recall("hiking Yosemite", strategy="fusion", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()


def test_search_with_fusion_returns_results():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("assistant", "Yosemite is beautiful in spring", session_id="s1")
        results = core.search_with_fusion("hiking Yosemite", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()


def test_search_with_fusion_empty():
    core = _tmp_core()
    try:
        results = core.search_with_fusion("nonexistent query", limit=5)
        assert results == []
    finally:
        core._test_cleanup()


def test_search_with_session_reranking_returns_results():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("assistant", "Yosemite is beautiful in spring", session_id="s1")
        results = core.search_with_session_reranking("hiking Yosemite", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()


def test_search_with_session_reranking_empty():
    core = _tmp_core()
    try:
        results = core.search_with_session_reranking("nonexistent query", limit=5)
        assert results == []
    finally:
        core._test_cleanup()


def test_search_messages_decomposed_with_session_filter():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("user", "I love hiking in Tahoe", session_id="s2")
        results = core.search_messages_decomposed("hiking", limit=5, session_id="s1")
        assert len(results) > 0
        assert all(r.memory.session_id == "s1" for r in results)
    finally:
        core._test_cleanup()
