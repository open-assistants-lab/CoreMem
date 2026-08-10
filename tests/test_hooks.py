"""Tests for hook handlers."""

from __future__ import annotations

import os
from pathlib import Path

from coremem.hooks.handler import (
    _format_recall_results,
    handle_pre_compact,
    handle_stop,
    handle_user_prompt_submit,
)


def _make_core(tmp_path: Path):
    from coremem import get_core

    os.environ["COREMEM_PATH"] = str(tmp_path / "hybrid")
    return get_core()


def test_format_recall_results_empty():
    assert _format_recall_results([]) == ""


def test_format_recall_results_with_hits(tmp_path):
    from coremem import get_core

    os.environ["COREMEM_PATH"] = str(tmp_path / "hybrid")
    core = get_core()
    core.ingest("user", "I love hiking in Yosemite", session_id="s1")
    results = core.recall("hiking", strategy="direct", limit=5)
    formatted = _format_recall_results(results)
    assert "## Relevant memories" in formatted
    assert "[user]" in formatted
    assert "hiking" in formatted
    assert "score:" in formatted


def test_handle_user_prompt_submit_captures_and_retrieves(tmp_path):
    core = _make_core(tmp_path)
    core.ingest("user", "I love hiking in Yosemite", session_id="s1")
    data = {"prompt": "hiking", "session_id": "sess-abc"}
    result = handle_user_prompt_submit(data, core)
    assert "hookSpecificOutput" in result
    assert result["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "## Relevant memories" in result["hookSpecificOutput"]["additionalContext"]


def test_handle_user_prompt_submit_empty_prompt(tmp_path):
    core = _make_core(tmp_path)
    result = handle_user_prompt_submit({"prompt": "", "session_id": "s1"}, core)
    assert result == {}


def test_handle_stop_captures_assistant_message(tmp_path):
    core = _make_core(tmp_path)
    core.ingest("user", "What about hiking?", session_id="s1")
    data = {"last_assistant_message": "Hiking is great!", "session_id": "s1"}
    result = handle_stop(data, core)
    assert result == {}
    results = core.recall("Hiking", strategy="direct", limit=5)
    assert any("Hiking is great" in r.memory.content for r in results)


def test_handle_stop_empty_message(tmp_path):
    core = _make_core(tmp_path)
    result = handle_stop({"last_assistant_message": "", "session_id": "s1"}, core)
    assert result == {}


def test_handle_pre_compact_noop(tmp_path):
    core = _make_core(tmp_path)
    result = handle_pre_compact({"trigger": "auto"}, core)
    assert result == {}


def test_handle_stop_skip_when_stop_hook_active(tmp_path):
    core = _make_core(tmp_path)
    core.ingest("user", "What about hiking?", session_id="s1")
    result = handle_stop({
        "last_assistant_message": "Hiking is great!",
        "session_id": "s1",
        "stop_hook_active": True,
    }, core)
    assert result == {}
    results = core.recall("Hiking", strategy="direct", limit=5)
    assert not any("Hiking is great" in r.memory.content for r in results)


def test_handle_stop_null_session_id(tmp_path):
    core = _make_core(tmp_path)
    result = handle_stop({
        "last_assistant_message": "test response",
        "session_id": None,
    }, core)
    assert result == {}
