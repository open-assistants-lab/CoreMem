"""Tests for dedup and merge logic."""
from coremem.dedup import build_dedup_prompt, _DEDUP_PROMPT


class TestDedupPrompt:
    def test_prompt_defines_relationships(self):
        for rel in ["duplicate", "refine", "supersede", "contradict", "new"]:
            assert rel in _DEDUP_PROMPT, f"Missing: {rel}"

    def test_build_dedup_prompt_formats_pairs(self):
        pairs = [
            {
                "new_index": 0,
                "candidate": {"id": "old_1", "content": "User works at Google"},
            }
        ]
        new_obs = [{"content": "User works at Anthropic"}]
        prompt = build_dedup_prompt(pairs, new_obs)
        assert 'New[0]:"User works at Anthropic"' in prompt
        assert 'Old(old_1):"User works at Google"' in prompt


class TestDedupLogic:
    def test_parse_dedup_response_returns_pairs(self):
        from coremem.dedup import _parse_dedup_response

        class MockResponse:
            tool_calls = [{"function": {"arguments": '{"pairs": [{"new_index": 0, "relationship": "duplicate", "old_id": "old_1"}]}'}}]

        result = _parse_dedup_response(MockResponse())
        assert result[0]["relationship"] == "duplicate"
        assert result[0]["old_id"] == "old_1"

    def test_parse_dedup_response_empty_on_no_tool_calls(self):
        from coremem.dedup import _parse_dedup_response

        class MockResponse:
            tool_calls = None

        result = _parse_dedup_response(MockResponse())
        assert result == []
