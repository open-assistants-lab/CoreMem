"""Integration tests for ObserverPipeline and ReflectorPipeline.

Uses mocked providers — no real LLM API calls.
"""

from __future__ import annotations

import asyncio
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from coremem import MemoryCore, MemoryStore
from coremem.backends.hybrid import HybridBackend
from coremem.providers import ChatResponse

pytestmark = pytest.mark.asyncio


@pytest.fixture
def tmp_core():
    d = tempfile.mkdtemp()
    backend = HybridBackend(path=d)
    core = MemoryCore(backend=backend)
    yield core
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_store():
    d = tempfile.mkdtemp()
    store = MemoryStore(path=d)
    yield store
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def populated_core(tmp_core):
    """Core with 8 messages."""
    for i in range(4):
        tmp_core.ingest("user", f"User message {i} about projects and career goals", session_id="main")
        tmp_core.ingest("assistant", f"Assistant reply {i}", session_id="main")
    return tmp_core


# ── Mock helpers ───────────────────────────────────────────────────────────


def _mock_sentences_response():
    return ChatResponse(content="User message 0 about projects and career goals")


def _mock_observer_response():
    return ChatResponse(content='''[
        {"id": "obs_1", "content": "User is working on projects", "priority": "\\u2622", "referenced_date": "", "source_quote": "User message 0 about projects and career goals"},
        {"id": "obs_2", "content": "User mentions career goals", "priority": "\\u2622", "referenced_date": "", "source_quote": "User message 1 about projects and career goals"}
    ]''')


def _mock_reflector_response():
    return ChatResponse(content='''[
        {"id": "refl_1", "content": "User demonstrates a pattern of project-driven work and career ambition", "domain": "career", "linked_observation_ids": ["obs_1", "obs_2"]}
    ]''')


# ── ObserverPipeline Mock Tests ───────────────────────────────────────────


class TestObserverPipelineMock:
    async def test_after_turn_skips_below_threshold(self, populated_core, tmp_store):
        from coremem.observer import ObserverPipeline

        pipeline = ObserverPipeline(
            core=populated_core, store=tmp_store, session_id="main",
            token_threshold=100_000, min_turns=0,
        )
        result = await pipeline.after_turn()
        assert result is None

    async def test_after_turn_fires_with_llm(self, populated_core, tmp_store):
        from coremem.observer import ObserverPipeline

        pipeline = ObserverPipeline(
            core=populated_core, store=tmp_store, session_id="main",
            token_threshold=1, min_turns=1,
        )

        with patch.object(pipeline._observer, "_provider", new=AsyncMock()) as mock_p:
            mock_p.chat_with_tools.side_effect = [_mock_sentences_response(), _mock_observer_response()]
            result = await pipeline.after_turn()
            assert result is not None
            assert len(result) == 2
            assert any("projects" in o["content"] for o in result)

        obs = tmp_store.get_observations()
        assert len(obs) == 2

    async def test_after_turn_skips_when_running(self, populated_core, tmp_store):
        from coremem.observer import ObserverPipeline

        pipeline = ObserverPipeline(
            core=populated_core, store=tmp_store, session_id="main",
            token_threshold=1, min_turns=1,
        )
        pipeline._running = True
        result = await pipeline.after_turn()
        assert result is None

    async def test_cursor_tracks_last_observed_id(self, populated_core, tmp_store):
        from coremem.observer import ObserverPipeline

        pipeline = ObserverPipeline(
            core=populated_core, store=tmp_store, session_id="main",
            token_threshold=1, min_turns=1,
        )
        assert pipeline._last_observed_id is None

        with patch.object(pipeline._observer, "_provider", new=AsyncMock()) as mock_p:
            mock_p.chat_with_tools.side_effect = [_mock_sentences_response(), _mock_observer_response()]
            await pipeline.after_turn()

        assert pipeline._last_observed_id is not None
        assert pipeline._turns_since_last_run == 0

    async def test_post_dedup_skips_similar_observations(self, populated_core, tmp_store):
        from coremem.observer import ObserverPipeline

        tmp_store.insert_observations([
            {"content": "User is working on multiple projects", "priority": "high"},
        ])

        pipeline = ObserverPipeline(
            core=populated_core, store=tmp_store, session_id="main",
            token_threshold=1, min_turns=1,
        )

        with patch.object(pipeline._observer, "_provider", new=AsyncMock()) as mock_p:
            mock_p.chat_with_tools.side_effect = [_mock_sentences_response(), _mock_observer_response()]
            result = await pipeline.after_turn()

        assert result is not None
        assert len(result) == 1
        assert "career" in result[0]["content"]


# ── ReflectorPipeline Mock Tests ──────────────────────────────────────────


class TestReflectorPipelineMock:
    async def test_maybe_run_skips_before_interval(self, tmp_store):
        from coremem.reflector import ReflectorPipeline

        obs = [{"id": f"o{i}", "content": f"Activity {i}", "priority": "medium"}
               for i in range(15)]
        tmp_store.insert_observations(obs)

        reflector = ReflectorPipeline(
            store=tmp_store, model="ollama:llama3.2", min_observations=10,
        )
        reflector._last_run_ts = float("inf")
        result = await reflector.maybe_run()
        assert result is None

    async def test_run_now_fires_with_llm(self, tmp_store):
        from coremem.reflector import ReflectorPipeline

        obs = [{"id": f"o{i}", "content": f"Activity {i}", "priority": "medium"}
               for i in range(15)]
        tmp_store.insert_observations(obs)

        reflector = ReflectorPipeline(
            store=tmp_store, model="ollama:llama3.2", min_observations=5,
        )

        with patch.object(reflector._reflector, "_provider", new=AsyncMock()) as mock_p:
            mock_p.chat.return_value = _mock_reflector_response()
            result = await reflector.run_now()

            assert result is not None
            assert len(result) == 1
            assert "career" in result[0]["content"]

        refs = tmp_store.get_reflections()
        assert len(refs) == 1

    async def test_run_now_skips_below_min_observations(self, tmp_store):
        from coremem.reflector import ReflectorPipeline

        tmp_store.insert_observations([
            {"id": "o1", "content": "Single observation", "priority": "medium"},
        ])

        reflector = ReflectorPipeline(
            store=tmp_store, model="ollama:llama3.2", min_observations=10,
        )
        result = await reflector.run_now()
        assert result is None

    async def test_quality_gate_dedups_with_embedding(self, tmp_store):
        from coremem.reflector import ReflectorPipeline

        obs = [{"id": f"o{i}", "content": f"Activity {i}", "priority": "medium"}
               for i in range(15)]
        tmp_store.insert_observations(obs)

        test_embedding = [0.1] * 384

        def _mock_embed(_text: str):
            return test_embedding

        reflector = ReflectorPipeline(
            store=tmp_store, model="ollama:llama3.2",
            embedding_fn=_mock_embed, min_observations=5,
        )

        with patch.object(reflector._reflector, "_provider", new=AsyncMock()) as mock_p:
            mock_p.chat.return_value = _mock_reflector_response()
            result = await reflector.run_now()

        assert result is not None
        assert len(result) == 1
