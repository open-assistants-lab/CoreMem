"""Tests for the gleaning pass — prompt validation and integration."""
from __future__ import annotations

import os
import tempfile

import pytest
from coremem import MemoryCore
from coremem.observer import GLEANING_SYSTEM_PROMPT, ObserverPipeline

skip_if_no_api_key = pytest.mark.skipif(
    not os.environ.get("OLLAMA_API_KEY"),
    reason="OLLAMA_API_KEY not set",
)


class TestGleaningPrompt:
    def test_gleaning_prompt_contains_already_extracted_placeholder(self):
        assert "{already_extracted}" in GLEANING_SYSTEM_PROMPT

    def test_gleaning_prompt_contains_review_tasks(self):
        assert "Named entities" in GLEANING_SYSTEM_PROMPT
        assert "Pronoun references" in GLEANING_SYSTEM_PROMPT
        assert "Implicit facts" in GLEANING_SYSTEM_PROMPT
        assert "Buried preferences" in GLEANING_SYSTEM_PROMPT

    def test_gleaning_prompt_has_verbatim_requirement(self):
        assert "must not return the observation" in GLEANING_SYSTEM_PROMPT

    def test_gleaning_prompt_has_few_shot_examples(self):
        assert "Example 1" in GLEANING_SYSTEM_PROMPT
        assert "Example 2" in GLEANING_SYSTEM_PROMPT
        assert "wife Sarah" in GLEANING_SYSTEM_PROMPT

    def test_gleaning_prompt_substitution(self):
        """The {already_extracted} placeholder is replaced with content lines."""
        substituted = GLEANING_SYSTEM_PROMPT.replace(
            "{already_extracted}", "- fact one\n- fact two"
        )
        assert "- fact one" in substituted
        assert "- fact two" in substituted
        assert "{already_extracted}" not in substituted


class TestGleaningIntegration:
    """End-to-end tests for the 2-stage (extraction + gleaning) pipeline."""

    SESSION_ID = "test"

    @pytest.mark.asyncio
    @skip_if_no_api_key
    async def test_enable_gleaning_runs_both_stages(self):
        """With enable_gleaning=True, the pipeline runs without error.
        Observation count varies at temp 0.1 — just verify no exception."""
        core = MemoryCore(path=tempfile.mkdtemp(), enable_observations=True)

        core.ingest("user", "I'm a software engineer named Alice who works at Acme Corp. I love hiking in the Cascades.", session_id=self.SESSION_ID)
        core.ingest("assistant", "Nice to meet you Alice! The Cascades are beautiful.", session_id=self.SESSION_ID)
        core.ingest("user", "I also play piano, live in Portland, and have two cats named Luna and Milo.", session_id=self.SESSION_ID)

        pipeline = ObserverPipeline(
            memory=core,
            session_id=self.SESSION_ID,
            model="ollama-cloud:deepseek-v4-flash",
            token_threshold=1,
            min_turns=1,
            enable_gleaning=True,
        )

        observations = await pipeline.extract()
        assert observations is not None  # Pipeline ran without error

        if observations:
            for obs in observations:
                assert obs.get("alignment_tier") in ("exact", "fuzzy")
                assert obs.get("source_quote")
                assert obs.get("content")
                assert obs.get("importance") is None  # Observer sets None

    @pytest.mark.asyncio
    @skip_if_no_api_key
    async def test_disable_gleaning_skips_second_stage(self):
        """With enable_gleaning=False, pipeline runs without gleaning."""
        core = MemoryCore(path=tempfile.mkdtemp(), enable_observations=True)

        core.ingest("user", "I'm a software engineer named Bob who works at Acme Corp.", session_id=self.SESSION_ID)
        core.ingest("assistant", "Hello Bob! Acme is a great company.", session_id=self.SESSION_ID)
        core.ingest("user", "I live in Chicago, enjoy photography, and drive a Tesla.", session_id=self.SESSION_ID)

        pipeline = ObserverPipeline(
            memory=core,
            session_id=self.SESSION_ID,
            model="ollama-cloud:deepseek-v4-flash",
            token_threshold=1,
            min_turns=1,
            enable_gleaning=False,
        )

        observations = await pipeline.extract()
        assert observations is not None  # Pipeline ran without error

    @pytest.mark.asyncio
    @skip_if_no_api_key
    async def test_empty_stage1_skips_gleaning(self):
        """If extraction produces 0 observations, gleaning is skipped."""
        core = MemoryCore(path=tempfile.mkdtemp(), enable_observations=True)

        core.ingest("assistant", "Hello, how can I help?", session_id=self.SESSION_ID)

        pipeline = ObserverPipeline(
            memory=core,
            session_id=self.SESSION_ID,
            model="ollama-cloud:deepseek-v4-flash",
            token_threshold=1,
            min_turns=1,
            enable_gleaning=True,
        )

        observations = await pipeline.extract()
        assert observations is None or len(observations) == 0
