"""Tests for observer, reflector, providers, memory_store."""

import tempfile

import pytest

from coremem.core import MemoryCore
from coremem.backends.hybrid import HybridBackend
from coremem.memory_store import MemoryStore
from coremem.observer_utils import chat_messages, estimate_tokens, parse_json_array
from coremem.providers import ChatResponse, create_provider


# ── observer_utils ────────────────────────────────────────────────────────


class TestParseJsonArray:
    def test_plain_json_array(self):
        assert parse_json_array('[{"key": "val"}]') == [{"key": "val"}]

    def test_fenced_json(self):
        result = parse_json_array('```json\n[{"key": "val"}]\n```')
        assert result == [{"key": "val"}]

    def test_fenced_bare(self):
        result = parse_json_array('```\n[{"key": "val"}]\n```')
        assert result == [{"key": "val"}]

    def test_single_object_wraps_in_array(self):
        result = parse_json_array('{"key": "val"}')
        assert result == [{"key": "val"}]

    def test_text_with_embedded_json(self):
        result = parse_json_array('Some text [{"k": "v"}] trailing')
        assert result == [{"k": "v"}]

    def test_invalid_json_returns_empty(self):
        assert parse_json_array("not json") == []


class TestChatMessages:
    def test_builds_system_user_pair(self):
        msgs = chat_messages("System prompt", "User input")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "System prompt"}
        assert msgs[1] == {"role": "user", "content": "User input"}


class TestEstimateTokens:
    def test_typical_text(self):
        assert estimate_tokens("hello world") == 2

    def test_empty_string(self):
        assert estimate_tokens("") == 1


# ── Provider ───────────────────────────────────────────────────────────────


class TestCreateProvider:
    def test_creates_openai_provider(self):
        p = create_provider("openai:gpt-4o")
        assert p is not None
        assert hasattr(p, "chat")

    def test_creates_ollama_provider(self):
        p = create_provider("ollama:llama3.2")
        assert p is not None

    def test_unknown_prefix_falls_back_to_openai(self):
        p = create_provider("groq:mixtral-8x7b")
        assert p is not None
        assert hasattr(p, "chat")

    def test_missing_colon_raises(self):
        with pytest.raises(ValueError, match="provider:model"):
            create_provider("invalid-string-no-colon")


class TestChatResponse:
    def test_basic_response(self):
        r = ChatResponse(content="hello", model="gpt-4o")
        assert r.content == "hello"
        assert r.model == "gpt-4o"

    def test_default_usage(self):
        r = ChatResponse(content="x")
        assert r.usage == {}


# ── MemoryStore ────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_store():
    d = tempfile.mkdtemp()
    store = MemoryStore(path=d)
    yield store
    import shutil
    shutil.rmtree(d, ignore_errors=True)


class TestMemoryStore:
    def test_insert_and_get_observations(self, tmp_store):
        ids = tmp_store.insert_observations([
            {"content": "User is a software engineer", "priority": "high"},
            {"content": "User lives in San Francisco", "priority": "medium"},
        ])
        assert len(ids) == 2
        obs = tmp_store.get_observations(limit=50)
        assert len(obs) == 2
        contents = {o["content"] for o in obs}
        assert contents == {"User is a software engineer", "User lives in San Francisco"}

    def test_get_observations_since(self, tmp_store):
        tmp_store.insert_observations([
            {"id": "o1", "content": "First", "priority": "medium"},
        ])
        tmp_store.insert_observations([
            {"id": "o2", "content": "Second", "priority": "medium"},
        ])
        assert len(tmp_store.get_observations_since("o1")) == 1

    def test_get_observations_since_none(self, tmp_store):
        tmp_store.insert_observations([
            {"content": "Test", "priority": "low"},
        ])
        assert len(tmp_store.get_observations_since(None)) == 1

    def test_recent_observations(self, tmp_store):
        tmp_store.insert_observations([
            {"content": "Recent fact", "priority": "medium"},
        ])
        recent = tmp_store.get_recent_observations(days=30)
        assert len(recent) == 1

    def test_search_observations(self, tmp_store):
        tmp_store.insert_observations([
            {"content": "User enjoys hiking in the mountains", "priority": "medium"},
        ])
        results = tmp_store.search_observations("hiking")
        assert len(results) > 0

    def test_insert_and_get_reflections(self, tmp_store):
        ids = tmp_store.insert_reflections([
            {"content": "User has a pattern of outdoor hobbies", "domain": "lifestyle",
             "linked_observation_ids": ["o1"]},
        ])
        assert len(ids) == 1
        refs = tmp_store.get_reflections()
        assert len(refs) == 1
        assert refs[0]["content"] == "User has a pattern of outdoor hobbies"

    def test_search_reflections(self, tmp_store):
        tmp_store.insert_reflections([
            {"content": "Prefers early morning workouts", "domain": "lifestyle"},
        ])
        results = tmp_store.search_reflections("workouts")
        assert len(results) > 0

    def test_apply_decay(self, tmp_store):
        tmp_store.insert_reflections([
            {"content": "Old reflection", "domain": "general", "score": 1.0},
        ])
        count = tmp_store.apply_decay()
        assert count >= 0
