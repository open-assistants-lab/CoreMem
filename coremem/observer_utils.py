"""Shared LLM messaging utilities used by Observer and Reflector."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_array(text: str) -> list[dict[str, Any]]:
    """Parse LLM response as a JSON array, with markdown-fence stripping.

    Handles common LLM output patterns:
      - Bare JSON array: [{"key": "val"}, ...]
      - Fenced JSON: ```json [...] ```
      - Fenced bare: ``` [...] ```
    """
    text = text.strip()

    # Strip markdown fences
    fence_match = re.match(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find the outermost JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError:
        pass

    # Final fallback: line-by-line parsing
    items: list[dict[str, Any]] = []
    for line in text.split("\n"):
        line = line.strip().strip(",")
        if line.startswith("{") and line.endswith("}"):
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return items


def chat_messages(system: str, user: str) -> list[dict[str, str]]:
    """Build a standard system + user message pair for LLM calls."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def estimate_tokens(text: str) -> int:
    """Rough token count — ~4 chars per token for English text."""
    return max(1, len(text) // 4)
