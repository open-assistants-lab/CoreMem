"""Tests for ToolExtractor — session-end tool message analysis."""

import json
import tempfile

import pytest

from coremem import MemoryCore
from coremem.tool_extractor import (
    ToolExtractor,
    _analyze_deterministic,
    _build_trace,
    _classify_error,
)


class TestErrorClassification:
    def test_empty_content(self):
        assert _classify_error("") == (False, None)

    def test_success_content(self):
        assert _classify_error("Success") == (False, None)

    def test_error_prefix(self):
        assert _classify_error("Error: lint_failure") == (True, "Error")

    def test_failed_keyword(self):
        assert _classify_error("failed to connect") == (True, "failed")

    def test_not_found_keyword(self):
        assert _classify_error("404 not found") == (True, "not found")

    def test_could_not_keyword(self):
        assert _classify_error("could not parse file") == (True, "could not")

    def test_error_lowercase(self):
        assert _classify_error("error: timeout") == (True, "error")


class TestBuildTrace:
    def test_basic_pairing(self):
        assistant_msgs = [
            {"role": "assistant", "tool_calls": [{"id": "c1", "name": "read"}]},
            {"role": "assistant", "tool_calls": [{"id": "c2", "name": "edit"}]},
        ]
        tool_msgs = [
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "tool", "tool_call_id": "c2", "content": "Error: lint"},
        ]
        trace = _build_trace(assistant_msgs, tool_msgs)
        assert len(trace) == 2
        assert trace[0]["tool_name"] == "read"
        assert trace[0]["success"] is True
        assert trace[1]["tool_name"] == "edit"
        assert trace[1]["success"] is False
        assert trace[1]["error_type"] == "Error"

    def test_recovery_detection(self):
        assistant_msgs = [
            {"role": "assistant", "tool_calls": [{"id": "c1", "name": "edit"}]},
            {"role": "assistant", "tool_calls": [{"id": "c2", "name": "edit"}]},
        ]
        tool_msgs = [
            {"role": "tool", "tool_call_id": "c1", "content": "Error: fail"},
            {"role": "tool", "tool_call_id": "c2", "content": "success"},
        ]
        trace = _build_trace(assistant_msgs, tool_msgs)
        assert trace[0]["recovery_call_id"] == "c2"

    def test_missing_tool_call_id(self):
        assistant_msgs = [
            {"role": "assistant", "tool_calls": [{"id": "c1", "name": "read"}]},
        ]
        tool_msgs = [
            {"role": "tool", "tool_call_id": "", "content": "result"},
        ]
        trace = _build_trace(assistant_msgs, tool_msgs)
        assert len(trace) == 1
        # Tool result content is empty because tool_call_id didn't match
        assert trace[0]["result_content"] == ""

    def test_no_tool_calls_on_assistant(self):
        assistant_msgs = [
            {"role": "assistant", "content": "text only"},
        ]
        tool_msgs = []
        trace = _build_trace(assistant_msgs, tool_msgs)
        assert len(trace) == 0


class TestAnalyzeDeterministic:
    def test_counts_errors(self):
        trace = [
            {"tool_name": "read", "success": True, "recovery_call_id": None},
            {"tool_name": "edit", "success": False, "error_type": "lint", "recovery_call_id": None},
        ]
        result = _analyze_deterministic(trace)
        assert result["n_errors"] == 1
        assert "edit" in result["error_by_tool"]

    def test_recovery_by_tool(self):
        trace = [
            {"tool_name": "edit", "success": False, "error_type": "lint", "recovery_call_id": "c2"},
        ]
        result = _analyze_deterministic(trace)
        assert "edit" in result["recovery_by_tool"]
        assert result["recovery_by_tool"]["edit"] == ["recovered_via_retry"]

    def test_tool_coverage(self):
        trace = [
            {"tool_name": "read", "success": True, "recovery_call_id": None},
            {"tool_name": "grep", "success": True, "recovery_call_id": None},
            {"tool_name": "edit", "success": True, "recovery_call_id": None},
        ]
        result = _analyze_deterministic(trace)
        assert set(result["tool_coverage"]) == {"edit", "grep", "read"}

    def test_sequences_length_2_and_3(self):
        trace = [
            {"tool_name": "a", "success": True, "recovery_call_id": None},
            {"tool_name": "b", "success": True, "recovery_call_id": None},
            {"tool_name": "c", "success": True, "recovery_call_id": None},
            {"tool_name": "a", "success": True, "recovery_call_id": None},
        ]
        result = _analyze_deterministic(trace)
        assert "a→b" in result["sequences"]
        assert "a→b→c" in result["sequences"]
        # Length 4 should NOT be in sequences
        assert "a→b→c→a" not in result["sequences"]

    def test_sequences_limited_to_top_10(self):
        trace = [
            {"tool_name": chr(ord("a") + i), "success": True, "recovery_call_id": None}
            for i in range(20)
        ]
        result = _analyze_deterministic(trace)
        assert len(result["sequences"]) <= 10


@pytest.mark.asyncio
class TestToolExtractor:
    async def test_extract_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            core = MemoryCore(path=str(tmpdir), enable_observations=True)
            sid, uid = "test_ses", "test_user"
            core.ingest("user", "refactor main.py", session_id=sid, user_id=uid)
            core.ingest("assistant", "Read.", session_id=sid, user_id=uid,
                        metadata={"tool_calls": [{"id": "c1", "name": "files_read"}]})
            core.ingest("tool", "file content", session_id=sid, user_id=uid,
                        metadata={"tool_call_id": "c1", "name": "files_read"})
            core.ingest("assistant", "Edit.", session_id=sid, user_id=uid,
                        metadata={"tool_calls": [{"id": "c2", "name": "files_edit"}]})
            core.ingest("tool", "Error: lint", session_id=sid, user_id=uid,
                        metadata={"tool_call_id": "c2", "name": "files_edit"})
            core.ingest("assistant", "Fix.", session_id=sid, user_id=uid,
                        metadata={"tool_calls": [{"id": "c3", "name": "files_edit"}]})
            core.ingest("tool", "Applied edit successfully", session_id=sid, user_id=uid,
                        metadata={"tool_call_id": "c3", "name": "files_edit"})
            core.ingest("assistant", "Done.", session_id=sid, user_id=uid)

            await core.session_end(sid, uid, active_skills=["python-refactoring"], min_tool_messages=3)

            obs = core.get_observations(limit=50)
            summaries = [o for o in obs if o.get("kind") == "tool_summary"]
            assert len(summaries) == 1
            meta = json.loads(summaries[0]["metadata"])
            behavior = meta["agent_behavior"]
            assert meta["n_tool_calls"] == 3
            assert meta["n_errors"] == 1
            assert "files_edit" in behavior["error_by_tool"]
            assert "files_edit" in behavior["recovery_by_tool"]
            assert "files_read→files_edit" in behavior["sequences"]
            assert meta["active_skills"] == ["python-refactoring"]
            assert meta["error_classification"] == "heuristic"

    async def test_skips_small_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            core = MemoryCore(path=str(tmpdir), enable_observations=True)
            core.ingest("user", "hi", session_id="s1", user_id="u")
            core.ingest("assistant", "hello", session_id="s1", user_id="u")
            await core.session_end("s1", "u")
            assert len(core.get_observations(limit=10)) == 0

    async def test_session_end_resilient(self):
        """Handles missing tool_call_id gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            core = MemoryCore(path=str(tmpdir), enable_observations=True)
            core.ingest("user", "refactor", session_id="s2", user_id="u")
            core.ingest("assistant", "Read.", session_id="s2", user_id="u",
                        metadata={"tool_calls": [{"id": "c1", "name": "read"}]})
            core.ingest("tool", "content", session_id="s2", user_id="u",
                        metadata={"tool_call_id": "", "name": "read"})
            core.ingest("assistant", "Edit.", session_id="s2", user_id="u",
                        metadata={"tool_calls": [{"id": "c2", "name": "edit"}]})
            core.ingest("tool", "Error: fail", session_id="s2", user_id="u",
                        metadata={"name": "edit"})
            core.ingest("assistant", "Done.", session_id="s2", user_id="u")
            await core.session_end("s2", "u", min_tool_messages=2)
            obs = core.get_observations(limit=10)
            summaries = [o for o in obs if o.get("kind") == "tool_summary"]
            assert len(summaries) == 1

    async def test_session_end_requires_enabled(self):
        """session_end is no-op when observations not enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            core = MemoryCore(path=str(tmpdir), enable_observations=False)
            await core.session_end("s1", "u")
            # Should not raise even though observations not enabled
            assert True

    async def test_user_goal_extracted(self):
        """First user message is used as user_goal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            core = MemoryCore(path=str(tmpdir), enable_observations=True)
            core.ingest("user", "I want to deploy my docker image", session_id="s1", user_id="u")
            core.ingest("assistant", "Sure", session_id="s1", user_id="u",
                        metadata={"tool_calls": [{"id": "c1", "name": "run"}]})
            core.ingest("tool", "output", session_id="s1", user_id="u",
                        metadata={"tool_call_id": "c1", "name": "run"})
            core.ingest("assistant", "Done", session_id="s1", user_id="u")
            await core.session_end("s1", "u", min_tool_messages=1)
            obs = core.get_observations(limit=10)
            summaries = [o for o in obs if o.get("kind") == "tool_summary"]
            assert len(summaries) == 1
            meta = json.loads(summaries[0]["metadata"])
            assert "deploy" in meta["user_goal"]