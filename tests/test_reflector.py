"""Tests for Reflector — broken priority filter fixed via importance.

ReflectorPipeline.run_now() previously sampled observations by checking
``priority`` (a legacy field that defaulted to ``"medium"`` and never matched
the intended "high/medium" split for emoji-valued legacy data). The 0.4.0
patch replaces it with an ``importance``-based filter:

  - importance >= 0.5 → keep all (high_med)
  - importance <  0.5 → sort by observation_ts DESC, cap at 100 (green)
"""

from __future__ import annotations

import shutil
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from coremem.memory_store import MemoryStore
from coremem.providers import ChatResponse
from coremem.reflector import ReflectorPipeline


pytestmark = pytest.mark.asyncio


def _make_obs(content: str, importance: float, observation_ts: str) -> dict:
    return {
        "content": content,
        "importance": importance,
        "observation_ts": observation_ts,
    }


def _mock_reflector_response() -> ChatResponse:
    return ChatResponse(
        content=(
            '[{"id": "refl_1", "content": "User demonstrates a pattern of work", '
            '"domain": "career", "linked_observation_ids": ["o_high"]}]'
        )
    )


def _user_prompt(mock_chat) -> str:
    """Pull the user-role content from the mocked chat() call."""
    messages = mock_chat.call_args[0][0]
    return messages[1]["content"]


class TestReflectorImportanceFilter:
    async def test_high_importance_kept_in_priority_sampling(self):
        """With 150 high + 200 low observations (350 total), the new filter must:
          - keep all 150 high-importance observations
          - cap the low-importance observations at 100 most recent
          - total sent to LLM = 250 (150 + 100), not 350 (broken no-op)
        """
        d = tempfile.mkdtemp()
        store = MemoryStore(path=d)
        try:
            for i in range(150):
                store.insert_observations([_make_obs(
                    f"high fact {i}",
                    importance=0.9,
                    observation_ts=f"2026-01-01T{i:03d}:00:00",
                )])
            for i in range(200):
                store.insert_observations([_make_obs(
                    f"low fact {i}",
                    importance=0.1,
                    observation_ts=f"2026-02-01T{i:03d}:00:00",
                )])
            assert len(store.get_observations(limit=1000)) == 350

            reflector = ReflectorPipeline(
                store=store, model="ollama:llama3.2", min_observations=5,
            )
            with patch.object(reflector._reflector, "_provider") as mock_p:
                mock_p.chat = AsyncMock(return_value=_mock_reflector_response())
                result = await reflector.run_now()

            mock_p.chat.assert_called_once()
            user_content = _user_prompt(mock_p.chat)

            high_count = user_content.count("high fact")
            low_count = user_content.count("low fact")
            total = high_count + low_count

            assert high_count == 150, f"expected all 150 high-importance kept, got {high_count}"
            assert low_count == 100, f"expected 100 low-importance (capped), got {low_count}"
            assert total == 250
            assert result is not None
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_only_low_importance_caps_to_100_most_recent(self):
        """With 250 low-importance observations, only the 100 most recent
        (by observation_ts DESC) should be sent to the LLM."""
        d = tempfile.mkdtemp()
        store = MemoryStore(path=d)
        try:
            for i in range(250):
                store.insert_observations([_make_obs(
                    f"low fact {i}",
                    importance=0.1,
                    observation_ts=f"2026-02-01T{i:03d}:00:00",
                )])
            assert len(store.get_observations(limit=1000)) == 250

            reflector = ReflectorPipeline(
                store=store, model="ollama:llama3.2", min_observations=5,
            )
            with patch.object(reflector._reflector, "_provider") as mock_p:
                mock_p.chat = AsyncMock(return_value=_mock_reflector_response())
                await reflector.run_now()

            mock_p.chat.assert_called_once()
            user_content = _user_prompt(mock_p.chat)

            low_count = user_content.count("low fact")
            # The 100 most recent are observation_ts 2026-02-01T150 through T249.
            # Verify both that the count is exactly 100 AND that the sent set
            # is the most-recent 100 (not an arbitrary 100).
            assert low_count == 100, f"expected 100 low-importance (capped), got {low_count}"
            assert "low fact 0" not in user_content, "oldest obs should be capped out"
            assert "low fact 249" in user_content, "newest obs should be in the cap"
        finally:
            shutil.rmtree(d, ignore_errors=True)
