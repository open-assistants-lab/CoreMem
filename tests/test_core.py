"""Tests for MemoryCore with HybridDB."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from coremem import MemoryCore, SessionBundle
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
        results = core._search_messages("model kits", limit=10)
        assert len(results) > 0
        assert any("model" in r.memory.content.lower() for r in results)
    finally:
        core._test_cleanup()


def test_core_search_messages_llm_expansion():
    core = _tmp_core()
    try:
        core.ingest("user", "I built a Spitfire model kit")
        core.ingest("assistant", "That sounds fun!")
        results = core._search_messages_llm_expansion("model kits", limit=10)
        assert len(results) > 0
        assert any("model" in r.memory.content.lower() for r in results)
    finally:
        core._test_cleanup()


def test_search_messages_llm_expansion_normalizes_scores(monkeypatch):
    """Per-variant scores must be normalized before merging, so a variant with
    systematically lower raw scores still competes fairly."""
    import coremem.core as coremod

    core = _tmp_core()
    try:
        def _row(mid: str, content: str, score: float) -> dict:
            return {
                "id": mid, "content": content, "role": "user", "session_id": "s1",
                "ts": "2026-01-01T00:00:00+00:00", "metadata": "{}", "turn_id": "",
                "_score": score,
            }

        def fake_search(table, col, q, limit=10):
            if q == "coffee creamer sugar milk honey":
                return [_row("a", "alpha content", 0.9), _row("c", "charlie content", 0.8)]
            return [_row("b", "bravo content", 0.2), _row("a", "alpha content", 0.1)]

        core._db.search = fake_search
        # Isolate the normalization: strip heuristics/diversity/rerank to identity
        monkeypatch.setattr(coremod.SearchHeuristics, "apply_all",
                            lambda query, content, score, ts=None: score)
        monkeypatch.setattr(coremod, "_mmr_diversify", lambda results, k: results)
        monkeypatch.setattr(coremod, "rerank", lambda query, results: results)

        results = core._search_messages_llm_expansion(
            "coffee creamer sugar milk honey", limit=10
        )

        # "b" only appears in the low-scale variant (raw 0.2 vs 0.9/0.8).
        # Without normalization it would rank last with score ~0.2.
        b = next(r for r in results if r.memory.id == "b")
        assert b.score > 0.5
        assert results[0].memory.id in ("a", "b")
    finally:
        core._test_cleanup()


def test_core_search_messages_filters_session_when_provided():
    core = _tmp_core()
    try:
        core.ingest("user", "coffee preference from session one", session_id="s1")
        core.ingest("user", "coffee preference from session two", session_id="s2")

        scoped = core._search_messages("coffee preference", session_id="s1", limit=10)
        assert scoped
        assert all(result.memory.session_id == "s1" for result in scoped)

        global_results = core._search_messages("coffee preference", limit=10)
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


def test_core_fetch_metadata_filter_with_special_chars():
    core = _tmp_core()
    try:
        core.ingest("user", "msg with key", session_id="s1", metadata={"my key": "x"})
        core.ingest("user", "msg without", session_id="s1", metadata={"other": "y"})
        core.ingest("user", "msg quoted", session_id="s1", metadata={"it's": "v"})

        # key with a space must match, not silently return nothing
        rows = core.fetch(metadata={"my key": "x"})
        assert [r.content for r in rows] == ["msg with key"]

        # key with a quote must not crash the query
        rows2 = core.fetch(metadata={"it's": "v"})
        assert [r.content for r in rows2] == ["msg quoted"]

        # non-matching value still filters correctly
        rows3 = core.fetch(metadata={"my key": "nope"})
        assert rows3 == []
    finally:
        core._test_cleanup()


def test_core_delete_metadata_filter():
    core = _tmp_core()
    try:
        core.ingest("user", "keep me", session_id="s1", metadata={"kind": "keep"})
        core.ingest("user", "delete me", session_id="s1", metadata={"kind": "drop"})

        deleted = core.delete(metadata={"kind": "drop"})

        assert deleted == 1
        assert core.count() == 1
        assert core.fetch_all()[0].content == "keep me"
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


def test_core_store_journal_record_and_recall():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in the mountains", session_id="session_a")
        core.ingest("assistant", "Me too! The views are amazing.", session_id="session_a")
        core.ingest("user", "What about skiing?", session_id="session_b")
        core.ingest("assistant", "Skiing is great too.", session_id="session_b")

        core._store_journal_record("session_a", "# Summary\n\nThe user enjoys hiking in the mountains and the assistant agreed.")
        core._store_journal_record("session_b", "# Summary\n\nThe user asked about skiing and the assistant confirmed it's great.")

        results = core.recall("hiking mountains", strategy="direct", limit=5)
        assert len(results) >= 1
        assert any("hiking" in r.memory.content.lower() for r in results)

        results = core.recall("skiing", strategy="direct", limit=5)
        assert len(results) >= 1
        assert any("Skiing" in r.memory.content for r in results)
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
        core._search_messages_decomposed = lambda query, limit=5: [
            _seed("m2", "answer", "s1", 1.0),
        ]

        bundles = core._reconstruct_sessions("answer")

        assert len(bundles) == 1
        assert bundles[0].complete is True
        # anchor-first ordering: the retrieved evidence leads the bundle
        assert [message.id for message in bundles[0].messages] == ["m2", "m1", "m3"]
    finally:
        core._test_cleanup()


def test_reconstruct_sessions_combines_opening_and_anchor_segments():
    core = _tmp_core()
    try:
        for index in range(8):
            _insert_traversal_message(core, f"m{index}", f"message {index} " + "x" * 500, "s1")
        core._search_messages_decomposed = lambda query, limit=5: [
            _seed("m6", "answer", "s1", 1.0),
        ]

        bundles = core._reconstruct_sessions(
            "answer",
            short_max_chars=1_000,
            segment_max_messages=2,
            segment_max_chars=2_000,
        )

        assert len(bundles) == 1
        assert bundles[0].complete is False
        # anchor m6 leads, the rest stay chronological
        assert [message.id for message in bundles[0].messages] == ["m6", "m0", "m1", "m7"]
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

        bundles = core._reconstruct_sessions(
            "answer",
            session_limit=2,
            max_context_chars=3_000,
            primary_results=primary,
        )

        assert sum(
            len(message.content) for bundle in bundles for message in bundle.messages
        ) <= 3_000
        assert [message.id for message in bundles[0].messages] == ["s1-m1", "s1-m0"]
        assert [message.id for message in bundles[1].messages] == ["s2-m1", "s2-m0"]
    finally:
        core._test_cleanup()


def test_reconstruct_sessions_keeps_anchor_when_exceeding_budget():
    core = _tmp_core()
    try:
        for index in range(3):
            _insert_traversal_message(
                core, f"m{index}", f"message {index} " + "x" * 600, "s1"
            )
        # m1 is the anchor; at 610 chars it alone exceeds the 1000-char budget
        core._search_messages_decomposed = lambda query, limit=5: [
            _seed("m1", "answer", "s1", 1.0),
        ]

        bundles = core._reconstruct_sessions("answer", max_context_chars=1_000)

        assert len(bundles) == 1
        ids = [message.id for message in bundles[0].messages]
        assert "m1" in ids, "anchor message must not be dropped by the budget"
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
        assert isinstance(bundles[0], SessionBundle)
    finally:
        core._test_cleanup()


def test_recall_default_is_episodic():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("assistant", "Yosemite is beautiful in spring", session_id="s1")
        core.ingest("user", "What about skiing?", session_id="s2")
        default_results = core.recall("hiking Yosemite", limit=5)
        episodic_results = core.recall("hiking Yosemite", strategy="episodic", limit=5)
        assert [r.memory.id for r in default_results] == [r.memory.id for r in episodic_results]
    finally:
        core._test_cleanup()


def test_recall_fusion_returns_results():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("assistant", "Yosemite is beautiful in spring", session_id="s1")
        results = core.recall("hiking Yosemite", strategy="fusion", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()


def test_recall_fusion_empty():
    core = _tmp_core()
    try:
        results = core.recall("nonexistent query", strategy="fusion", limit=5)
        assert results == []
    finally:
        core._test_cleanup()


def test_search_messages_decomposed_with_session_filter():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("user", "I love hiking in Tahoe", session_id="s2")
        results = core._search_messages_decomposed("hiking", limit=5, session_id="s1")
        assert len(results) > 0
        assert all(r.memory.session_id == "s1" for r in results)
    finally:
        core._test_cleanup()


def test_get_core_creates_memorycore_with_defaults(tmp_path):
    import os
    old_path = os.environ.get("COREMEM_PATH")
    os.environ["COREMEM_PATH"] = str(tmp_path / "coremem-test")
    try:
        from coremem import get_core
        core = get_core()
        assert core is not None
        assert hasattr(core, "recall")
        assert hasattr(core, "ingest")
    finally:
        if old_path is None:
            del os.environ["COREMEM_PATH"]
        else:
            os.environ["COREMEM_PATH"] = old_path


def test_get_core_passes_coremem_llm_model_to_journal_compiler(tmp_path):
    import os
    old_path = os.environ.get("COREMEM_PATH")
    old_model = os.environ.get("COREMEM_LLM_MODEL")
    os.environ["COREMEM_PATH"] = str(tmp_path / "coremem-test")
    os.environ["COREMEM_LLM_MODEL"] = "ollama:llama3.2"
    try:
        from coremem import get_core
        core = get_core()
        assert core._journal_compiler.provider._model == "llama3.2"
    finally:
        if old_path is None:
            del os.environ["COREMEM_PATH"]
        else:
            os.environ["COREMEM_PATH"] = old_path
        if old_model is None:
            del os.environ["COREMEM_LLM_MODEL"]
        else:
            os.environ["COREMEM_LLM_MODEL"] = old_model


def test_get_core_default_journal_model_when_env_unset(tmp_path):
    import os
    old_path = os.environ.get("COREMEM_PATH")
    old_model = os.environ.get("COREMEM_LLM_MODEL")
    os.environ["COREMEM_PATH"] = str(tmp_path / "coremem-test")
    os.environ.pop("COREMEM_LLM_MODEL", None)
    try:
        from coremem import get_core
        core = get_core()
        assert core._journal_compiler.provider._model == "gpt-4o-mini"
    finally:
        if old_path is None:
            del os.environ["COREMEM_PATH"]
        else:
            os.environ["COREMEM_PATH"] = old_path
        if old_model is None:
            os.environ.pop("COREMEM_LLM_MODEL", None)
        else:
            os.environ["COREMEM_LLM_MODEL"] = old_model


# ── Batch ingest (ingest_many) ─────────────────────────────────────────────


def test_ingest_many_batches_and_flushes():
    core = _tmp_core()
    try:
        messages = [
            {"role": "user", "content": f"fact number {i}", "session_id": f"s{i % 2}"}
            for i in range(20)
        ]
        ids = core.ingest_many(messages)
        assert len(ids) == 20
        assert len(set(ids)) == 20
        assert core.count() == 20
        # journal drained — nothing pending
        assert core.db.journal_status()["pending"] == 0
        # searchable after flush
        results = core._search_messages("fact number 7", limit=5)
        assert any("fact number 7" in r.memory.content for r in results)
    finally:
        core._test_cleanup()


def test_ingest_many_skips_empty_content_and_mirrors_turn_logic():
    core = _tmp_core()
    try:
        ids = core.ingest_many([
            {"role": "user", "content": "hello", "session_id": "s1"},
            {"role": "assistant", "content": "hi back", "session_id": "s1"},
            {"role": "user", "content": "   ", "session_id": "s1"},
            {"role": "user", "content": "second turn", "session_id": "s1"},
        ])
        assert len(ids) == 3
        rows = core.db.raw_query("SELECT turn_id FROM messages ORDER BY ts")
        turn_ids = [r["turn_id"] for r in rows]
        # user msg 1 opens a turn, assistant reuses it, second user opens a new one
        assert turn_ids[0] == turn_ids[1]
        assert turn_ids[2] != turn_ids[0]
    finally:
        core._test_cleanup()


def test_ingest_many_preserves_explicit_ids_and_metadata():
    core = _tmp_core()
    try:
        ids = core.ingest_many([
            {"id": "custom-1", "role": "user", "content": "keep me",
             "session_id": "s1", "metadata": {"kind": "keep"}},
        ])
        assert ids == ["custom-1"]
        rows = core.db.raw_query("SELECT id, metadata FROM messages WHERE id = ?", ("custom-1",))
        assert rows and "keep" in rows[0]["metadata"]
    finally:
        core._test_cleanup()


# ── Session-cap selection (cap=2) ──────────────────────────────────────────


def test_session_cap_allows_two_messages_per_session():
    core = _tmp_core()
    try:
        for s in range(1, 7):
            core.ingest("user", f"user asks about topic alpha {s}", session_id=f"s{s}")
            core.ingest("assistant", f"assistant details topic alpha {s}", session_id=f"s{s}")

        mono = core._search_messages_decomposed(
            "topic alpha", limit=5, per_query_limit=20, use_cross_encoder=True,
            session_cap=1,
        )
        cap2 = core._search_messages_decomposed(
            "topic alpha", limit=5, per_query_limit=20, use_cross_encoder=True,
            session_cap=2,
        )
        # cap=1: one message per session (enough sessions exist)
        assert len(mono) == 5
        assert len({r.memory.session_id for r in mono}) == 5
        # cap=2: still five results, but one session may supply two
        assert len(cap2) == 5
        counts = {}
        for r in cap2:
            counts[r.memory.session_id] = counts.get(r.memory.session_id, 0) + 1
        assert max(counts.values()) == 2
        assert 3 <= len(counts) <= 4
        # cap=2 surfaces at least one message cap=1 missed
        cap1_ids = {r.memory.id for r in mono}
        cap2_extra = [r for r in cap2 if r.memory.id not in cap1_ids]
        assert cap2_extra, "cap=2 must surface at least one message cap=1 missed"
    finally:
        core._test_cleanup()


def test_session_cap_anchor_allocation_keeps_top_sessions():
    core = _tmp_core()
    try:
        for s in range(1, 7):
            core.ingest("user", f"user asks about topic alpha {s}", session_id=f"s{s}")
            core.ingest("assistant", f"assistant details topic alpha {s}", session_id=f"s{s}")

        anchor = core._search_messages_decomposed(
            "topic alpha", limit=5, per_query_limit=20, use_cross_encoder=True,
            session_cap=2, allocation="anchor",
        )
        assert len(anchor) == 5
        counts = {}
        for r in anchor:
            counts[r.memory.session_id] = counts.get(r.memory.session_id, 0) + 1
        # top session holds two slots, three other sessions one each
        assert max(counts.values()) == 2
        assert len(counts) == 4
        # the two-slot session is the highest-scoring one
        top_session = max(counts.items(), key=lambda kv: kv[1])[0]
        mono = core._search_messages_decomposed(
            "topic alpha", limit=5, per_query_limit=20, use_cross_encoder=True,
            session_cap=1,
        )
        assert mono[0].memory.session_id == top_session
    finally:
        core._test_cleanup()


def test_session_cap_default_unchanged():
    core = _tmp_core()
    try:
        core.ingest_many([
            {"role": "user", "content": f"pizza topping discussion {i}", "session_id": f"s{i % 6}"}
            for i in range(12)
        ])
        default = core.recall("pizza topping", strategy="episodic")
        assert len(default) == 5
        # default cap=1: at most one per session when enough sessions exist
        sessions = {r.memory.session_id for r in default}
        assert len(sessions) == len(default)
    finally:
        core._test_cleanup()


# ── Default bundle budget + anchor-first ordering (validated config) ───────


def test_reconstruct_sessions_default_budget_is_4k_and_anchor_first():
    core = _tmp_core()
    try:
        for index in range(8):
            _insert_traversal_message(
                core, f"m{index}", f"message {index} " + "x" * 600, "s1"
            )
        core._search_messages_decomposed = lambda query, limit=5: [
            _seed("m5", "answer", "s1", 1.0),
        ]

        bundles = core._reconstruct_sessions("answer")

        # default max_context_chars is now 4_000 (validated on the S answer
        # eval: 4k bundles scored 0.678 vs 0.608 at 16k with 60% less context)
        assert len(bundles) == 1
        total = sum(len(message.content) for message in bundles[0].messages)
        assert total <= 4_000
        # anchor-first: the retrieved evidence leads the bundle
        assert bundles[0].messages[0].id == "m5"
    finally:
        core._test_cleanup()


def test_recall_bundles_default_is_anchor_first():
    core = _tmp_core()
    try:
        core.ingest_many([
            {"id": "b1", "role": "user", "content": "context before", "session_id": "s1"},
            {"id": "b2", "role": "user", "content": "the answer is cobalt blue", "session_id": "s1"},
            {"id": "b3", "role": "user", "content": "context after", "session_id": "s1"},
        ])
        bundles = core.recall("cobalt blue", strategy="episodic", bundles=True)
        assert len(bundles) == 1
        ids = [message.id for message in bundles[0].messages]
        assert ids[0] == "b2", f"anchor should lead the bundle, got {ids}"
        assert set(ids) == {"b1", "b2", "b3"}
    finally:
        core._test_cleanup()
