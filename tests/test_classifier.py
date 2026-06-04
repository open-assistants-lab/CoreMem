"""Tests for classification phase."""
from coremem.classifier import build_classification_prompt, _CLASSIFICATION_PROMPT


class TestClassificationPrompt:
    def test_prompt_contains_12_types(self):
        for t in ["profile", "preference", "project", "decision",
                  "technical_stack", "business_context", "people",
                  "constraint", "workflow", "episodic", "procedural", "sentiment"]:
            assert t in _CLASSIFICATION_PROMPT, f"Missing type: {t}"

    def test_prompt_defines_durability(self):
        assert "durable" in _CLASSIFICATION_PROMPT
        assert "temporary" in _CLASSIFICATION_PROMPT

    def test_prompt_defines_sensitivity(self):
        assert "normal" in _CLASSIFICATION_PROMPT
        assert "personal" in _CLASSIFICATION_PROMPT
        assert "sensitive" in _CLASSIFICATION_PROMPT

    def test_build_classification_prompt_formats_indices(self):
        obs = [
            {"content": "User works at Anthropic"},
            {"content": "User likes coffee"},
        ]
        prompt = build_classification_prompt(obs)
        assert "[0] User works at Anthropic" in prompt
        assert "[1] User likes coffee" in prompt
