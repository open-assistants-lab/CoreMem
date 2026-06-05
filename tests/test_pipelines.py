"""Integration tests for ObserverPipeline — alignment-gated."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import tempfile
import shutil

import pytest

from coremem import MemoryCore
from coremem.observer import ObserverPipeline
from coremem.providers import ChatResponse
from coremem.types import Memory

pytestmark = pytest.mark.asyncio


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_memory(enable_observations=True):
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d, enable_observations=enable_observations)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


def _make_core_with_messages(messages: list[Memory]):
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d, enable_observations=True)
    for m in messages:
        core.ingest(m.role, m.content, session_id="main",
                    user_id="alice", agent_id="a1", ts=m.ts)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


def _mock_valid_tool_response() -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[{
            "function": {
                "arguments": (
                    '{"observations": [{"id": "o1", "content": "User is a software engineer", '
                    '"source_quote": "I am a software engineer", '
                    '"referenced_date": "2026-06-01", '
                    '"importance": 0.8, "entities": []}]}'
                )
            }
        }],
    )


def _mock_fabricated_tool_response() -> ChatResponse:
    return ChatResponse(
        content="",
        tool_calls=[{
            "function": {
                "arguments": (
                    '{"observations": [{"id": "o1", "content": "User lives on Mars", '
                    '"source_quote": "I have lived on Mars for 10 years", '
                    '"referenced_date": "2026-06-01", '
                    '"importance": 0.8, "entities": ["Mars"]}]}'
                )
            }
        }],
    )


# ── Tests ──────────────────────────────────────────────────────────────────


class TestObserverPipelineAlignment:
    async def test_valid_quote_is_inserted_with_alignment_tier(self):
        ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
        messages = [
            Memory(id="m1", role="user", content="I am a software engineer", ts=ts),
        ]
        core = _make_core_with_messages(messages)
        try:
            pipeline = ObserverPipeline(
                memory=core, session_id="main",
                token_threshold=1, min_turns=1,
            )
            with patch.object(pipeline._observer, "_provider") as mock_p:
                entity_resp = ChatResponse(
                    content="", tool_calls=[{"function": {"arguments": '{"entities": ["software engineer"]}'}}],
                )
                obs_resp = _mock_valid_tool_response()
                mock_p.chat_with_tools = AsyncMock(side_effect=[entity_resp, entity_resp, entity_resp, entity_resp, entity_resp, entity_resp, entity_resp, obs_resp])
                result = await pipeline.extract()
            assert result is not None
            assert len(result) == 1
            assert result[0]["alignment_tier"] == "exact"
            assert result[0]["alignment_confidence"] == 1.0
            stored = core.get_observations()
            assert len(stored) == 1
            assert stored[0]["alignment_tier"] == "exact"
        finally:
            core._test_cleanup()

    async def test_fabricated_quote_is_dropped(self):
        ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
        messages = [
            Memory(id="m1", role="user", content="I am a software engineer", ts=ts),
        ]
        core = _make_core_with_messages(messages)
        try:
            pipeline = ObserverPipeline(
                memory=core, session_id="main",
                token_threshold=1, min_turns=1,
            )
            with patch.object(pipeline._observer, "_provider") as mock_p:
                fabricated = ChatResponse(
                    content="", tool_calls=[{"function": {"arguments": '{"observations": [{"id":"o1","content":"User is a doctor","source_quote":"I am a doctor","importance":0.8,"referenced_date":"2026-06-01","entities":["example"]}]}'}}],
                )
                entity_resp = ChatResponse(
                    content="", tool_calls=[{"function": {"arguments": '{"entities": ["example"]}'}}],
                )
                mock_p.chat_with_tools = AsyncMock(
                    side_effect=[entity_resp, entity_resp, entity_resp, entity_resp, entity_resp, entity_resp, entity_resp, fabricated],
                )
                result = await pipeline.extract()
            assert result is not None
            assert len(result) == 0
        finally:
            core._test_cleanup()

    async def test_below_token_threshold_skips(self):
        ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
        messages = [
            Memory(id="m1", role="user", content="Short", ts=ts),
        ]
        core = _make_core_with_messages(messages)
        try:
            pipeline = ObserverPipeline(
                memory=core, session_id="main",
                token_threshold=100_000, min_turns=0,
            )
            result = await pipeline.extract()
            assert result is None
        finally:
            core._test_cleanup()

    async def test_dedup_against_prior_observations(self):
        ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)
        messages = [
            Memory(id="m1", role="user", content="I am a software engineer", ts=ts),
        ]
        core = _make_core_with_messages(messages)
        try:
            core.insert_observations([{
                "content": "User is a software engineer",
                "importance": 0.8,
            }])
            pipeline = ObserverPipeline(
                memory=core, session_id="main",
                token_threshold=1, min_turns=1,
            )
            with patch.object(pipeline._observer, "_provider") as mock_p:
                mock_p.chat_with_tools = AsyncMock(
                    return_value=_mock_valid_tool_response()
                )
                result = await pipeline.extract()
            assert result is not None
            assert len(result) == 0
        finally:
            core._test_cleanup()