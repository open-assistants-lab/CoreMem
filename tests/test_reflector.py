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
import time
from unittest.mock import AsyncMock, patch

import pytest

from coremem import MemoryCore
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
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            for i in range(150):
                memory.insert_observations([_make_obs(
                    f"high fact {i}",
                    importance=0.9,
                    observation_ts=f"2026-01-01T{i:03d}:00:00",
                )])
            for i in range(200):
                memory.insert_observations([_make_obs(
                    f"low fact {i}",
                    importance=0.1,
                    observation_ts=f"2026-02-01T{i:03d}:00:00",
                )])
            assert len(memory.get_observations(limit=1000)) == 350

            reflector = ReflectorPipeline(
                memory=memory, reflect_model="ollama:llama3.2", min_observations=5,
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
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            for i in range(250):
                memory.insert_observations([_make_obs(
                    f"low fact {i}",
                    importance=0.1,
                    observation_ts=f"2026-02-01T{i:03d}:00:00",
                )])
            assert len(memory.get_observations(limit=1000)) == 250

            reflector = ReflectorPipeline(
                memory=memory, reflect_model="ollama:llama3.2", min_observations=5,
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


class TestCountBasedTrigger:
    async def test_maybe_run_skips_when_count_below_threshold(self):
        """With 5 unreflected facts and N=50 plus a recent last_run_ts,
        neither trigger fires — maybe_run() must return None without
        calling the LLM."""
        d = tempfile.mkdtemp()
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            for i in range(5):
                memory.insert_observations([{
                    "id": f"obs_{i}",
                    "content": f"fact {i}",
                    "source_quote": f"q{i}",
                    "kind": "fact",
                    "reflected": 0,
                    "observation_ts": f"2026-01-{(i % 28) + 1:02d}T00:00:00",
                }])

            pipeline = ReflectorPipeline(
                memory=memory, reflect_model="ollama:llama3.2",
                trigger_every_n_observations=50, interval_hours=9999,
            )
            pipeline._last_run_ts = time.time()

            with patch.object(pipeline._reflector, "_provider") as mock_p:
                mock_p.chat = AsyncMock()
                result = await pipeline.maybe_run()

            assert result is None
            mock_p.chat.assert_not_called()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_maybe_run_fires_on_count_threshold(self):
        """With 50 unreflected facts and N=50 plus a recent last_run_ts
        (so the time-based trigger would NOT fire), the count trigger
        must fire and invoke the LLM."""
        d = tempfile.mkdtemp()
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            for i in range(50):
                memory.insert_observations([{
                    "id": f"obs_{i}",
                    "content": f"fact {i}",
                    "source_quote": f"q{i}",
                    "kind": "fact",
                    "reflected": 0,
                    "importance": 0.5,
                    "observation_ts": f"2026-01-{(i % 28) + 1:02d}T00:00:00",
                }])

            pipeline = ReflectorPipeline(
                memory=memory, reflect_model="ollama:llama3.2",
                trigger_every_n_observations=50, interval_hours=9999,
                min_observations=1,
            )
            pipeline._last_run_ts = time.time()

            with patch.object(pipeline._reflector, "_provider") as mock_p:
                mock_p.chat = AsyncMock(return_value=_mock_reflector_response())
                result = await pipeline.maybe_run()

            mock_p.chat.assert_called_once()
            assert result is not None
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_maybe_run_fires_on_time_even_when_count_below(self):
        """With only 5 unreflected facts and N=999 (count never hits),
        but interval_hours=1 and last_run_ts=0 (long ago), the time
        trigger fires."""
        d = tempfile.mkdtemp()
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            for i in range(5):
                memory.insert_observations([{
                    "id": f"obs_{i}",
                    "content": f"fact {i}",
                    "source_quote": f"q{i}",
                    "kind": "fact",
                    "reflected": 0,
                    "importance": 0.5,
                    "observation_ts": f"2026-01-{(i % 28) + 1:02d}T00:00:00",
                }])

            pipeline = ReflectorPipeline(
                memory=memory, reflect_model="ollama:llama3.2",
                trigger_every_n_observations=999, interval_hours=1,
                min_observations=1,
            )
            pipeline._last_run_ts = 0.0

            with patch.object(pipeline._reflector, "_provider") as mock_p:
                mock_p.chat = AsyncMock(return_value=_mock_reflector_response())
                result = await pipeline.maybe_run()

            mock_p.chat.assert_called_once()
            assert result is not None
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_maybe_run_marks_reflected_on_success(self):
        """After a successful run_now(), source facts should be marked
        reflected=1 via mark_reflected()."""
        d = tempfile.mkdtemp()
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            for i in range(5):
                memory.insert_observations([{
                    "id": f"obs_{i}",
                    "content": f"fact {i}",
                    "source_quote": f"q{i}",
                    "kind": "fact",
                    "reflected": 0,
                    "importance": 0.5,
                    "observation_ts": f"2026-01-{(i % 28) + 1:02d}T00:00:00",
                }])

            pipeline = ReflectorPipeline(
                memory=memory, reflect_model="ollama:llama3.2",
                trigger_every_n_observations=5, interval_hours=9999,
                min_observations=1,
            )
            pipeline._last_run_ts = time.time()

            with patch.object(pipeline._reflector, "_provider") as mock_p:
                mock_p.chat = AsyncMock(return_value=_mock_reflector_response())
                result = await pipeline.maybe_run()

            assert result is not None
            # After reflection, facts should be marked reflected
            pending = memory.get_pending_reflections()
            assert len(pending) == 0
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestStartStopLifecycle:
    async def test_start_creates_background_task(self):
        """start() spawns an asyncio task."""
        d = tempfile.mkdtemp()
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            pipeline = ReflectorPipeline(
                memory=memory, reflect_model="ollama:llama3.2",
                trigger_every_n_observations=50,
                interval_hours=24,
            )
            await pipeline.start()
            assert pipeline._task is not None
            assert not pipeline._task.done()
            await pipeline.stop()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_start_is_idempotent(self):
        """Calling start() twice is a no-op (doesn't spawn two tasks)."""
        d = tempfile.mkdtemp()
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            pipeline = ReflectorPipeline(memory=memory, reflect_model="ollama:llama3.2")
            await pipeline.start()
            task_1 = pipeline._task
            await pipeline.start()
            task_2 = pipeline._task
            assert task_1 is task_2
            await pipeline.stop()
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_stop_is_idempotent(self):
        """Calling stop() twice doesn't raise."""
        d = tempfile.mkdtemp()
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            pipeline = ReflectorPipeline(memory=memory, reflect_model="ollama:llama3.2")
            await pipeline.start()
            await pipeline.stop()
            await pipeline.stop()
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestImportanceAssignment:
    async def test_assigns_importance_to_null_facts(self):
        """Facts with NULL importance get importance assigned before
        the main reflection call."""
        d = tempfile.mkdtemp()
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            # Insert facts with NULL importance (as 0.5.0 Observer does)
            for i in range(5):
                memory.insert_observations([{
                    "id": f"obs_{i}",
                    "content": f"User fact {i}",
                    "source_quote": f"q{i}",
                    "kind": "fact",
                    "reflected": 0,
                    "importance": None,
                    "observation_ts": f"2026-01-{(i % 28) + 1:02d}T00:00:00",
                }])

            pipeline = ReflectorPipeline(
                memory=memory, reflect_model="ollama:llama3.2",
                trigger_every_n_observations=1, interval_hours=9999,
                min_observations=1,
            )
            pipeline._last_run_ts = time.time()

            # The importance prompt response assigns scores, then the
            # reflection prompt returns reflections
            with patch.object(pipeline._reflector, "_provider") as mock_p:
                # First call: importance assignment
                # Second call: reflection
                mock_p.chat = AsyncMock(side_effect=[
                    ChatResponse(content=(
                        '[{"id": "obs_0", "importance": 0.8}, '
                        '{"id": "obs_1", "importance": 0.6}, '
                        '{"id": "obs_2", "importance": 0.3}, '
                        '{"id": "obs_3", "importance": 0.9}, '
                        '{"id": "obs_4", "importance": 0.4}]'
                    )),
                    _mock_reflector_response(),
                ])
                result = await pipeline.maybe_run()

            assert result is not None
            # Verify importance was written to the store
            stored = memory.get_observations()
            scores = {o["id"]: o["importance"] for o in stored}
            assert scores["obs_0"] == 0.8
            assert scores["obs_1"] == 0.6
            assert scores["obs_2"] == 0.3
            assert scores["obs_3"] == 0.9
            assert scores["obs_4"] == 0.4
        finally:
            shutil.rmtree(d, ignore_errors=True)

    async def test_skips_when_no_null_importance(self):
        """If all facts already have importance, no importance prompt call."""
        d = tempfile.mkdtemp()
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            for i in range(5):
                memory.insert_observations([{
                    "id": f"obs_{i}",
                    "content": f"fact {i}",
                    "source_quote": f"q{i}",
                    "kind": "fact",
                    "reflected": 0,
                    "importance": 0.5,
                    "observation_ts": f"2026-01-{(i % 28) + 1:02d}T00:00:00",
                }])

            pipeline = ReflectorPipeline(
                memory=memory, reflect_model="ollama:llama3.2",
                trigger_every_n_observations=1, interval_hours=9999,
                min_observations=1,
            )
            pipeline._last_run_ts = time.time()

            with patch.object(pipeline._reflector, "_provider") as mock_p:
                mock_p.chat = AsyncMock(return_value=_mock_reflector_response())
                result = await pipeline.maybe_run()

            mock_p.chat.assert_called_once()  # only once = reflection, no importance
            assert result is not None
        finally:
            shutil.rmtree(d, ignore_errors=True)


    async def test_unreflected_count_excludes_already_reflected(self):
        """Facts that have already been reflected (reflected=1) must not
        count toward the trigger threshold."""
        d = tempfile.mkdtemp()
        memory = MemoryCore(path=d, enable_observations=True)
        try:
            ids = []
            for i in range(50):
                obs = {
                    "id": f"obs_{i}",
                    "content": f"fact {i}",
                    "source_quote": f"q{i}",
                    "kind": "fact",
                    "reflected": 0,
                    "observation_ts": f"2026-01-{(i % 28) + 1:02d}T00:00:00",
                }
                memory.insert_observations([obs])
                ids.append(obs["id"])
            memory.mark_reflected(ids[:40])

            pipeline = ReflectorPipeline(
                memory=memory, reflect_model="ollama:llama3.2",
                trigger_every_n_observations=50, interval_hours=9999,
            )
            pipeline._last_run_ts = time.time()

            assert len(memory.get_pending_reflections()) == 10

            with patch.object(pipeline._reflector, "_provider") as mock_p:
                mock_p.chat = AsyncMock()
                result = await pipeline.maybe_run()

            assert result is None
            mock_p.chat.assert_not_called()
        finally:
            shutil.rmtree(d, ignore_errors=True)
