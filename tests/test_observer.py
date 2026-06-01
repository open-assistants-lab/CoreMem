"""Tests for the rewritten Observer (single-pass, alignment-gated)."""

from __future__ import annotations

import asyncio
from datetime import UTC
from unittest.mock import AsyncMock, patch

import pytest

from coremem.observer import OBSERVATION_TOOL, OBSERVER_SYSTEM_PROMPT, Observer
from coremem.providers import ChatResponse
from coremem.types import Memory


class TestObserverConstants:
    def test_system_prompt_has_few_shot_examples(self):
        # Two synthetic dialogues with verbatim-quote demonstrations
        assert OBSERVER_SYSTEM_PROMPT.count("source_quote") >= 4
        # Instructions present
        assert "verbatim" in OBSERVER_SYSTEM_PROMPT.lower()
        # Few-shot dialogues include timestamp prefixes that get stripped
        assert "[20" in OBSERVER_SYSTEM_PROMPT

    def test_tool_schema_has_no_priority(self):
        params = OBSERVATION_TOOL["function"]["parameters"]
        # No `priority` field anywhere in the schema
        assert "priority" not in params["properties"]
        item_props = params["properties"]["observations"]["items"]["properties"]
        assert "priority" not in item_props
        # Has the fields we keep (nested under items)
        assert "content" in item_props
        assert "source_quote" in item_props
        assert "importance" in item_props
        assert "entities" in item_props

    def test_tool_schema_required_fields(self):
        params = OBSERVATION_TOOL["function"]["parameters"]
        # Top-level required: list of field name strings
        assert "observations" in params["required"]
        # Item-level required: list of field name strings
        items = params["properties"]["observations"]["items"]
        for field in ("id", "content", "referenced_date", "source_quote", "importance", "entities"):
            assert field in items["required"], f"missing required field: {field}"


class TestObserverConstructor:
    def test_default_model(self):
        obs = Observer()
        assert obs is not None

    def test_custom_model(self):
        obs = Observer(model="openai:gpt-4o-mini")
        assert obs is not None

    def test_enable_gleaning_raises(self):
        with pytest.raises(NotImplementedError, match="gleaning"):
            Observer(enable_gleaning=True)


def _mock_tool_response(arguments: str) -> ChatResponse:
    """Simulate a tool_call response from chat_with_tools."""
    return ChatResponse(content="", tool_calls=[{"function": {"arguments": arguments}}])


def _mock_text_response(content: str) -> ChatResponse:
    """Simulate a text-only response (no tool call)."""
    return ChatResponse(content=content)


def _make_memory(role: str, content: str, ts=None) -> Memory:
    from datetime import datetime
    return Memory(
        id=f"m_{role}_{content[:10]}",
        role=role,
        content=content,
        ts=ts or datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC),
    )


class TestObserverRun:
    def test_makes_single_chat_with_tools_call(self):
        obs = Observer(model="openai:gpt-4o-mini")
        with patch.object(obs, "_provider") as mock_p:
            mock_p.chat_with_tools = AsyncMock(
                return_value=_mock_tool_response(
                    '[{"id": "o1", "content": "User is a software engineer", '
                    '"source_quote": "I am a software engineer", '
                    '"importance": 0.8, "entities": [], "referenced_date": "2026-06-01"}]'
                )
            )
            messages = [_make_memory("user", "I am a software engineer")]
            asyncio.run(obs.run(messages))
            assert mock_p.chat_with_tools.call_count == 1

    def test_returns_parsed_observations(self):
        obs = Observer(model="openai:gpt-4o-mini")
        with patch.object(obs, "_provider") as mock_p:
            mock_p.chat_with_tools = AsyncMock(
                return_value=_mock_tool_response(
                    '[{"id": "o1", "content": "User is a software engineer", '
                    '"source_quote": "I am a software engineer", '
                    '"importance": 0.8, "entities": [], "referenced_date": "2026-06-01"}]'
                )
            )
            messages = [_make_memory("user", "I am a software engineer")]
            result = asyncio.run(obs.run(messages))
            assert len(result) == 1
            assert result[0]["content"] == "User is a software engineer"

    def test_returns_observations_from_schema_compliant_args(self):
        obs = Observer(model="openai:gpt-4o-mini")
        with patch.object(obs, "_provider") as mock_p:
            mock_p.chat_with_tools = AsyncMock(
                return_value=_mock_tool_response(
                    '{"observations": [{"id": "o1", "content": "User is a software engineer", '
                    '"source_quote": "I am a software engineer", '
                    '"importance": 0.8, "entities": [], "referenced_date": "2026-06-01"}]}'
                )
            )
            messages = [_make_memory("user", "I am a software engineer")]
            result = asyncio.run(obs.run(messages))
            assert len(result) == 1
            assert result[0]["content"] == "User is a software engineer"

    def test_returns_empty_on_no_tool_calls(self):
        obs = Observer(model="openai:gpt-4o-mini")
        with patch.object(obs, "_provider") as mock_p:
            mock_p.chat_with_tools = AsyncMock(
                return_value=ChatResponse(content="")
            )
            messages = [_make_memory("user", "I am a software engineer")]
            result = asyncio.run(obs.run(messages))
            assert result == []

    def test_skips_messages_with_no_timestamp(self):
        obs = Observer(model="openai:gpt-4o-mini")
        with patch.object(obs, "_provider") as mock_p:
            mock_p.chat_with_tools = AsyncMock(
                return_value=_mock_tool_response("[]")
            )
            messages = [
                Memory(id="m1", role="user", content="Hello", ts=None),
                _make_memory("user", "I am a software engineer"),
            ]
            asyncio.run(obs.run(messages))
            call_args = mock_p.chat_with_tools.call_args
            sent_messages = call_args[0][0]
            sent_content = " ".join(m["content"] for m in sent_messages)
            assert "Hello" not in sent_content
            assert "I am a software engineer" in sent_content

    def test_messages_native_role_field(self):
        obs = Observer(model="openai:gpt-4o-mini")
        with patch.object(obs, "_provider") as mock_p:
            mock_p.chat_with_tools = AsyncMock(
                return_value=_mock_tool_response("[]")
            )
            messages = [
                _make_memory("user", "Hello"),
                _make_memory("assistant", "Hi there"),
            ]
            asyncio.run(obs.run(messages))
            call_args = mock_p.chat_with_tools.call_args
            sent_messages = call_args[0][0]
            for m in sent_messages:
                assert "role" in m
                assert "content" in m
            roles = [m["role"] for m in sent_messages]
            assert "user" in roles
            assert "assistant" in roles
