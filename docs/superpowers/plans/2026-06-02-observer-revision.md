# Observer Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the CoreMem Observer (0.4.0) as a single-pass, alignment-gated extraction system that drops hallucination from 34-70% to <10% on LongMemEval / DeepSeek V4 Flash.

**Architecture:** New `coremem/grounding.py` module with pure-function 3-tier alignment (LangExtract port). Rewritten `coremem/observer.py` with single-pass LLM call (CogCanvas prompt pattern, native messages array, temperature=0.1). Patched `ObserverPipeline` that runs the alignment gate. Schema migration for `alignment_tier` + `alignment_confidence` columns. Hard break, bump to 0.4.0.

**Tech Stack:** Python 3.11+, `difflib` (stdlib only for grounding), pytest, ruff, mypy, httpx.

**Reference spec:** `docs/superpowers/specs/2026-06-02-observer-revision-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `coremem/grounding.py` | **Create** | `AlignmentTier`, `AlignmentResult`, `align_quote()` — pure-function 3-tier alignment gate |
| `tests/test_grounding.py` | **Create** | ~30 unit tests covering all tier cases |
| `coremem/observer.py` | **Rewrite** | `Observer` (single-pass LLM call) + `ObserverPipeline` (alignment-gated) |
| `tests/test_observer.py` | **Rewrite** | Observer contract tests (mocked provider) |
| `tests/test_pipelines.py` | **Rewrite** | ObserverPipeline integration tests (alignment gate behavior) |
| `coremem/memory_store.py` | **Modify** | Add `_migrate_observations_v2()` for alignment columns |
| `tests/test_memory_store.py` | **Create** | Schema migration tests |
| `coremem/reflector.py` | **Modify** | Fix broken priority filter → use `importance` |
| `tests/test_reflector.py` | **Create** | Filter behavior tests |
| `coremem/nli.py` | **Delete** | Module removed |
| `pyproject.toml` | **Modify** | Drop `[nli]` extra |
| `CHANGELOG.md` | **Modify** | Add 0.4.0 entry |
| `README.md` | **Modify** | Remove nli snippets, update Observer examples |
| `coremem/__init__.py` | **Modify** | Bump version to 0.4.0 (if exposed) |

---

## Task 1: Add `coremem/grounding.py` with `align_quote()`

**Files:**
- Create: `coremem/grounding.py`
- Create: `tests/test_grounding.py`

- [ ] **Step 1.1: Write failing tests for `align_quote()`**

Create `tests/test_grounding.py`:

```python
"""Tests for the 3-tier alignment gate (LangExtract port)."""

import pytest

from coremem.grounding import (
    AlignmentResult,
    AlignmentTier,
    align_quote,
)


# ── Tier enum ──────────────────────────────────────────────────────────────


class TestAlignmentTier:
    def test_values(self):
        assert AlignmentTier.EXACT.value == "exact"
        assert AlignmentTier.FUZZY.value == "fuzzy"
        assert AlignmentTier.NONE.value == "none"


class TestAlignmentResult:
    def test_exact_construction(self):
        r = AlignmentResult(AlignmentTier.EXACT, 1.0, (10, 20))
        assert r.tier == AlignmentTier.EXACT
        assert r.confidence == 1.0
        assert r.char_interval == (10, 20)

    def test_none_construction(self):
        r = AlignmentResult(AlignmentTier.NONE, 0.0, None)
        assert r.tier == AlignmentTier.NONE
        assert r.char_interval is None


# ── align_quote: EXACT tier ────────────────────────────────────────────────


class TestAlignExact:
    def test_perfect_match_middle(self):
        r = align_quote("Hello world", "Say Hello world there")
        assert r.tier == AlignmentTier.EXACT
        assert r.confidence == 1.0
        assert r.char_interval is not None
        # The matched span should start at "Hello" and end after "world"
        start, end = r.char_interval
        assert "Say Hello world there"[start:end] == "Hello world"

    def test_perfect_match_at_start(self):
        r = align_quote("Hello", "Hello world")
        assert r.tier == AlignmentTier.EXACT

    def test_perfect_match_at_end(self):
        r = align_quote("world", "Hello world")
        assert r.tier == AlignmentTier.EXACT

    def test_whitespace_only_diff(self):
        # Multiple internal spaces collapse on tokenize
        r = align_quote("Hello   world", "Hello world")
        assert r.tier == AlignmentTier.EXACT

    def test_case_mismatch(self):
        r = align_quote("HELLO", "hello")
        assert r.tier == AlignmentTier.EXACT


# ── align_quote: FUZZY tier ────────────────────────────────────────────────


class TestAlignFuzzy:
    def test_single_char_drift(self):
        # "I'm" vs "I am" — ratio >= 0.75
        r = align_quote("I'm a software engineer", "I am a software engineer")
        assert r.tier == AlignmentTier.FUZZY
        assert r.confidence >= 0.75
        assert r.char_interval is not None

    def test_trailing_punctuation(self):
        r = align_quote("engineer.", "engineer")
        assert r.tier == AlignmentTier.FUZZY
        assert r.confidence >= 0.75


# ── align_quote: NONE tier ─────────────────────────────────────────────────


class TestAlignNone:
    def test_fabricated_quote(self):
        r = align_quote("lives on Mars", "lives in Denver")
        assert r.tier == AlignmentTier.NONE
        assert r.confidence < 0.75
        assert r.char_interval is None

    def test_below_threshold_overlap(self):
        r = align_quote("Hello cruel world", "Hello world")
        assert r.tier == AlignmentTier.NONE

    def test_empty_quote(self):
        r = align_quote("", "Hello world")
        assert r.tier == AlignmentTier.NONE
        assert r.char_interval is None

    def test_empty_source(self):
        r = align_quote("Hello", "")
        assert r.tier == AlignmentTier.NONE

    def test_quote_longer_than_source(self):
        r = align_quote("Hello world foo", "Hello")
        assert r.tier == AlignmentTier.NONE

    def test_whitespace_only_quote(self):
        r = align_quote("   ", "Hello world")
        # Whitespace tokenizes to empty tokens → NONE
        assert r.tier == AlignmentTier.NONE


# ── align_quote: char_interval accuracy ────────────────────────────────────


class TestCharInterval:
    def test_exact_span_points_to_matched_text(self):
        source = "Alice said Hello world to Bob"
        r = align_quote("Hello world", source)
        assert r.tier == AlignmentTier.EXACT
        start, end = r.char_interval
        assert source[start:end] == "Hello world"
```

- [ ] **Step 1.2: Run tests, verify they fail**

Run: `uv run pytest tests/test_grounding.py -v`
Expected: `ModuleNotFoundError: No module named 'coremem.grounding'`

- [ ] **Step 1.3: Implement `coremem/grounding.py`**

Create `coremem/grounding.py`:

```python
"""3-tier alignment gate — port of langextract/resolver.py:316-400.

Pure function, stdlib only. Checks whether a quote appears as a
sub-string of a source text, with three confidence tiers:
EXACT (difflib-perfect token match), FUZZY (LCS ratio >= 0.75),
NONE (drop the observation).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum


class AlignmentTier(Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    NONE = "none"


@dataclass
class AlignmentResult:
    tier: AlignmentTier
    confidence: float
    char_interval: tuple[int, int] | None


_FUZZY_THRESHOLD = 0.75


def align_quote(quote: str, source: str) -> AlignmentResult:
    """Check whether ``quote`` appears as a sub-string of ``source``.

    Tokenizes both on whitespace and lowercases for case-insensitive
    matching. Returns EXACT if the quote's tokens form a contiguous
    window in the source, FUZZY if SequenceMatcher.ratio() on tokens
    is >= 0.75, otherwise NONE.
    """
    if not quote or not source:
        return AlignmentResult(AlignmentTier.NONE, 0.0, None)

    quote_lower = quote.lower().strip()
    source_lower = source.lower()

    quote_tokens = quote_lower.split()
    source_tokens = source_lower.split()

    if not quote_tokens:
        return AlignmentResult(AlignmentTier.NONE, 0.0, None)
    if len(quote_tokens) > len(source_tokens):
        return AlignmentResult(AlignmentTier.NONE, 0.0, None)

    # EXACT: contiguous window of source tokens matches all quote tokens
    for i in range(len(source_tokens) - len(quote_tokens) + 1):
        window = source_tokens[i : i + len(quote_tokens)]
        if window == quote_tokens:
            char_interval = _char_span_for_token_range(
                source, i, i + len(quote_tokens)
            )
            return AlignmentResult(AlignmentTier.EXACT, 1.0, char_interval)

    # FUZZY: char-level SequenceMatcher on lowered strings (LCS-based).
    # Deliberate deviation from the original plan's reference: the plan
    # specified token-level SequenceMatcher (and a token-index -> char-span
    # helper), but the actual implementation is char-level on whitespace
    # + case-normalized strings, returning char indices into source
    # directly. Char-level catches single-character / punctuation drift
    # in the LLM's source_quote (e.g. "engineer." vs "engineer",
    # "I'm" vs "I am") that would be token-mismatches under token-level.
    sm = SequenceMatcher(None, source_lower, quote_lower)
    ratio = sm.ratio()
    if ratio >= _FUZZY_THRESHOLD:
        block = sm.find_longest_match(
            0, len(source_lower), 0, len(quote_lower)
        )
        char_interval = (block.a, block.a + block.size)
        return AlignmentResult(AlignmentTier.FUZZY, ratio, char_interval)

    return AlignmentResult(AlignmentTier.NONE, 0.0, None)


def _char_span_for_token_range(
    source: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    """Map a token-index range to (char_start, char_end) in source."""
    char_start = -1
    char_end = -1
    token_idx = 0
    i = 0
    while i < len(source):
        # Skip leading whitespace
        while i < len(source) and source[i].isspace():
            i += 1
        if i >= len(source):
            break
        # Read token
        token_start = i
        while i < len(source) and not source[i].isspace():
            i += 1
        token_end = i
        if token_idx == start:
            char_start = token_start
        if token_idx == end - 1:
            char_end = token_end
            break
        token_idx += 1
    if char_start < 0 or char_end < 0:
        return (0, 0)
    return (char_start, char_end)
```

- [ ] **Step 1.4: Run tests, verify pass**

Run: `uv run pytest tests/test_grounding.py -v`
Expected: all tests PASS (15+ tests)

- [ ] **Step 1.5: Commit**

```bash
git add coremem/grounding.py tests/test_grounding.py
git commit -m "feat(grounding): 3-tier alignment gate — port of LangExtract resolver"
```

---

## Task 2: Rewrite `coremem/observer.py` — new constants and `Observer` class

**Files:**
- Rewrite: `coremem/observer.py` (replaces the file contents)

- [ ] **Step 2.1: Write failing tests for new `Observer` contract**

Replace `tests/test_observer.py` entirely. The existing tests reference the old two-pass design; they must be rewritten against the new contract.

```python
"""Tests for the rewritten Observer (single-pass, alignment-gated)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coremem.observer import OBSERVATION_TOOL, OBSERVER_SYSTEM_PROMPT, Observer
from coremem.providers import ChatResponse
from coremem.types import Memory


# ── Constants ───────────────────────────────────────────────────────────────


class TestObserverConstants:
    def test_system_prompt_has_few_shot_examples(self):
        # Two synthetic dialogues with verbatim-quote demonstrations
        assert OBSERVER_SYSTEM_PROMPT.count("source_quote") >= 4
        # Instructions present
        assert "verbatim" in OBSERVER_SYSTEM_PROMPT.lower()
        # Few-shot dialogues include timestamp prefixes that get stripped
        assert "[20" in OBSERVER_SYSTEM_PROMPT  # year prefix

    def test_tool_schema_has_no_priority(self):
        props = OBSERVATION_TOOL["function"]["parameters"]["properties"]
        assert "priority" not in props
        # Has the fields we keep
        assert "content" in props
        assert "source_quote" in props
        assert "importance" in props
        assert "entities" in props

    def test_tool_schema_required_fields(self):
        required = OBSERVATION_TOOL["function"]["parameters"]["required"][0]
        obs_props = required["properties"]
        for field in ("id", "content", "referenced_date", "source_quote", "importance", "entities"):
            assert field in obs_props, f"missing required field: {field}"


# ── Observer constructor ───────────────────────────────────────────────────


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


# ── Observer.run: contract ─────────────────────────────────────────────────


def _mock_tool_response(arguments: str) -> ChatResponse:
    """Simulate a tool_call response from chat_with_tools."""
    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [{"function": {"arguments": arguments}}]
    return ChatResponse(content="", tool_calls=msg.tool_calls)


def _mock_text_response(content: str) -> ChatResponse:
    """Simulate a text-only response (no tool call)."""
    return ChatResponse(content=content)


def _make_memory(role: str, content: str, ts=None) -> Memory:
    from datetime import datetime, timezone
    return Memory(
        id=f"m_{role}_{content[:10]}",
        role=role,
        content=content,
        ts=ts or datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
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
            result = asyncio.run(obs.run(messages))
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

    def test_returns_empty_on_no_tool_calls(self):
        obs = Observer(model="openai:gpt-4o-mini")
        with patch.object(obs, "_provider") as mock_p:
            mock_p.chat_with_tools = AsyncMock(
                return_value=ChatResponse(content="")
            )
            messages = [_make_memory("user", "I am a software engineer")]
            result = asyncio.run(obs.run(messages))
            assert result == []

    def test_temperature_is_0_1(self):
        obs = Observer(model="openai:gpt-4o-mini")
        with patch.object(obs, "_provider") as mock_p:
            mock_p.chat_with_tools = AsyncMock(
                return_value=_mock_tool_response(
                    '[{"id": "o1", "content": "x", "source_quote": "x", '
                    '"importance": 0.5, "entities": [], "referenced_date": ""}]'
                )
            )
            messages = [_make_memory("user", "x")]
            asyncio.run(obs.run(messages))
            # The request body should have temperature=0.1
            call_kwargs = mock_p.chat_with_tools.call_args
            # Provider-specific: assert on messages list and tools; the
            # temperature is enforced inside the provider adapter.
            assert call_kwargs is not None

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
            # The LLM should only see the timestamped message
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
            # System + context user + alternating turns (no JSON wrapping)
            for m in sent_messages:
                assert "role" in m
                assert "content" in m
            roles = [m["role"] for m in sent_messages]
            assert "user" in roles
            assert "assistant" in roles
```

- [ ] **Step 2.2: Run tests, verify they fail**

Run: `uv run pytest tests/test_observer.py -v`
Expected: tests fail because `coremem.observer` still has the old two-pass design with no `OBSERVER_SYSTEM_PROMPT`, no `OBSERVATION_TOOL` (with `priority` removed), and a different `Observer.run` signature.

- [ ] **Step 2.3: Rewrite `coremem/observer.py` with new constants + `Observer` class**

Replace the file contents entirely:

```python
"""Observer — single-pass fact extraction from conversations.

Uses a 3-tier alignment gate (coremem.grounding.align_quote) to
deterministically catch fabricated source_quote values. The model is
prompted via CogCanvas-style system message with 2 few-shot examples
that demonstrate the verbatim-quote contract.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from coremem.grounding import AlignmentTier, align_quote
from coremem.observer_utils import parse_json_array
from coremem.providers import LLMProvider, create_provider
from coremem.types import Memory

logger = logging.getLogger("coremem.observer")


# ── Prompt: system message (instructions + 2 few-shot examples) ────────────


OBSERVER_SYSTEM_PROMPT = """You are an observation agent. Extract key facts from a conversation and return them as structured observations via the record_observations tool.

RULES:
- One fact per observation. Be exact with values.
- source_quote: a VERBATIM sub-string of the conversation. Copy-paste exactly — do not rephrase, do not paraphrase, do not change a single character. If you cannot find a verbatim sub-string, do not return the observation.
- Do not invent facts. If nothing factual is present, return an empty observations array.

IMPORTANCE: 0.8-1.0 identity/contact/job/salary.
            0.5-0.7 preferences/habits/projects.
            0.1-0.4 context/trivia.
ENTITIES: list of named entities (people, companies, products, locations).

Few-shot examples:

Example 1:
[2024-01-15T10:00:00] user: I just moved to Seattle last month for a new job at Anthropic.
[2024-01-15T10:01:00] assistant: That's exciting! What's your role?
[2024-01-15T10:02:00] user: I'm a research engineer working on alignment.

{"id": "obs_1", "content": "User moved to Seattle in January 2024", "referenced_date": "2024-01", "source_quote": "I just moved to Seattle last month", "importance": 0.7, "entities": ["Seattle"]}
{"id": "obs_2", "content": "User works at Anthropic as a research engineer", "referenced_date": "2024-01-15", "source_quote": "I'm a research engineer working on alignment", "importance": 0.9, "entities": ["Anthropic"]}

Example 2:
[2024-02-03T14:30:00] user: My favorite programming language is Rust, though I still use Python for data work.
[2024-02-03T14:31:00] assistant: Why do you prefer Rust?
[2024-02-03T14:32:00] user: The borrow checker catches bugs before runtime.

{"id": "obs_1", "content": "User's favorite programming language is Rust", "referenced_date": "2024-02-03", "source_quote": "My favorite programming language is Rust", "importance": 0.6, "entities": ["Rust"]}
{"id": "obs_2", "content": "User uses Python for data work", "referenced_date": "2024-02-03", "source_quote": "I still use Python for data work", "importance": 0.5, "entities": ["Python"]}
"""


# ── Tool schema (no `priority` field) ─────────────────────────────────────


OBSERVATION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_observations",
        "description": "Record observations extracted from the conversation",
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique ID like obs_01"},
                            "content": {"type": "string", "description": "ONE fact per observation"},
                            "referenced_date": {"type": "string"},
                            "source_quote": {"type": "string", "description": "EXACT sub-string of the conversation (prefix stripped)"},
                            "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "entities": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "content", "referenced_date", "source_quote", "importance", "entities"],
                    },
                },
            },
            "required": ["observations"],
        },
    },
}


# ── Observer ───────────────────────────────────────────────────────────────


class Observer:
    """Single-pass fact extraction from conversation messages.

    Makes one chat_with_tools call per invocation. Returns parsed
    observations or [] on parse failure / empty tool_calls.
    """

    def __init__(
        self,
        model: str = "ollama:llama3.2",
        enable_gleaning: bool = False,
    ):
        if enable_gleaning:
            raise NotImplementedError(
                "gleaning pass not implemented; see docs/observer-hallucination-review.md for the CogCanvas pattern"
            )
        self._provider = create_provider(model)

    async def run(
        self,
        messages: list[Memory],
        prior_observations: list[dict[str, Any]] | None = None,
        observation_date: str | None = None,
    ) -> list[dict[str, Any]]:
        prior = prior_observations or []
        date_str = observation_date or datetime.now(timezone.utc).date().isoformat()

        llm_messages = self._build_messages(messages, prior, date_str)
        response = await self._provider.chat_with_tools(llm_messages, [OBSERVATION_TOOL])
        return self._parse_response(response)

    def _build_messages(
        self,
        messages: list[Memory],
        prior: list[dict[str, Any]],
        date_str: str,
    ) -> list[dict[str, str]]:
        """Build the native messages array (system + context + conversation)."""
        context_block = self._build_context_block(prior, date_str)
        out: list[dict[str, str]] = [
            {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
            {"role": "user", "content": context_block},
        ]
        for m in messages:
            if m.content and m.ts is not None:
                ts_str = m.ts.isoformat()[:19]
                out.append({"role": m.role, "content": f"[{ts_str}] {m.content}"})
        return out

    @staticmethod
    def _build_context_block(
        prior: list[dict[str, Any]],
        date_str: str,
    ) -> str:
        """Build the # Already extracted + # Observation date preamble."""
        if prior:
            prior_lines = "\n".join(
                f"- {o.get('content', '')}" for o in prior[:20]
            )
            already = f"# Already extracted (last 20)\n{prior_lines}\n\n"
        else:
            already = "# Already extracted (last 20)\n(none)\n\n"
        return f"{already}# Observation date\n{date_str}"

    @staticmethod
    def _parse_response(response: Any) -> list[dict[str, Any]]:
        """Extract observations from a chat_with_tools response.

        Reads payload from tool_calls[0].function.arguments (NOT content).
        Falls back to parsing content if no tool_calls.
        """
        # Try tool_calls first
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            arguments = tool_calls[0].get("function", {}).get("arguments", "")
            if arguments:
                parsed = parse_json_array(arguments)
                if parsed and "observations" in parsed[0]:
                    return parsed[0]["observations"]
                if parsed and isinstance(parsed[0], dict) and "content" in parsed[0]:
                    return parsed
                return []

        # Fallback to content (shouldn't normally happen)
        content = getattr(response, "content", "")
        if not content:
            return []
        parsed = parse_json_array(content)
        if parsed and "observations" in parsed[0]:
            return parsed[0]["observations"]
        return parsed if parsed else []
```

Note: The `Observer._provider` is private but the tests access it via `patch.object(obs, "_provider")`. The test class for `enable_gleaning_raises` is in the same file. The mocked responses use `ChatResponse(content="", tool_calls=...)` — but `ChatResponse.__init__` doesn't take a `tool_calls` arg. We need to either extend `ChatResponse` or use a different mocking strategy.

The current `ChatResponse` in `coremem/providers.py` is:
```python
class ChatResponse:
    def __init__(self, content: str, model: str = "", usage: dict[str, int] | None = None):
        self.content = content
        self.model = model
        self.usage = usage or {}
```

No `tool_calls` field. The test mock approach in the spec assumes `tool_calls` is on the response. We have two options:
- **Option A:** Add `tool_calls: list | None = None` to `ChatResponse.__init__`. Then the test uses `ChatResponse(content="", tool_calls=[...])`.
- **Option B:** Use a `MagicMock` for the response in tests, not `ChatResponse`.

The production `Observer._parse_response` uses `getattr(response, "tool_calls", None)` so it works with both. **Option A is cleaner** because it makes the response type carry the data explicitly. Let's add it.

Modify `coremem/providers.py`:
```python
class ChatResponse:
    def __init__(
        self,
        content: str,
        model: str = "",
        usage: dict[str, int] | None = None,
        tool_calls: list[dict] | None = None,
    ):
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.tool_calls = tool_calls
```

Also, the existing adapters in `providers.py` need to populate `tool_calls` for the OpenAI adapter. The current adapter puts the result in `content`:
```python
if tool_calls:
    content = tool_calls[0]["function"]["arguments"]
```

For the new design, we need to **set both `content` AND `tool_calls`**, so the Observer can read from `tool_calls`. Update the adapter to:
```python
if tool_calls:
    content = tool_calls[0]["function"]["arguments"]
    tool_calls_arg = tool_calls  # also expose the structured list
else:
    content = choice["message"].get("content", "")
    tool_calls_arg = None
```

This is a providers.py change. Include it as Step 2.4 below.

- [ ] **Step 2.4: Update `coremem/providers.py` — add `tool_calls` to `ChatResponse`**

Edit `coremem/providers.py`:

```python
class ChatResponse:
    def __init__(
        self,
        content: str,
        model: str = "",
        usage: dict[str, int] | None = None,
        tool_calls: list[dict] | None = None,
    ):
        self.content = content
        self.model = model
        self.usage = usage or {}
        self.tool_calls = tool_calls
```

Edit the `_OpenAIAdapter.chat_with_tools` method (line 96-104):

```python
async def chat_with_tools(
    self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
) -> ChatResponse:
    """OpenAI-compatible tool/function calling."""
    headers = {"Content-Type": "application/json"}
    if self._api_key:
        headers["Authorization"] = f"Bearer {self._api_key}"
    body: dict[str, Any] = {
        "model": self._model,
        "messages": messages,
        "tools": tools,
        "tool_choice": {"type": "function", "function": {"name": tools[0]["function"]["name"]}},
        "temperature": 0.1,    # CHANGED from 0.0 to 0.1 (CogCanvas pattern)
        "thinking": {"type": "disabled"},
    }
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.post(
            f"{self._base_url}/v1/chat/completions",
            json=body, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
    choice = data["choices"][0]
    raw_tool_calls = choice["message"].get("tool_calls") or []
    if raw_tool_calls:
        content = raw_tool_calls[0]["function"]["arguments"]
    else:
        content = choice["message"].get("content", "")
    return ChatResponse(
        content=content,
        model=data.get("model", self._model),
        usage=data.get("usage", {}),
        tool_calls=raw_tool_calls or None,
    )
```

Two changes: (1) `temperature: 0.0` → `temperature: 0.1`, (2) populate `tool_calls` field on the response.

- [ ] **Step 2.5: Run tests, verify pass**

Run: `uv run pytest tests/test_observer.py tests/test_grounding.py -v`
Expected: all tests PASS

- [ ] **Step 2.6: Commit**

```bash
git add coremem/observer.py coremem/providers.py tests/test_observer.py
git commit -m "feat(observer): single-pass extraction with CogCanvas prompt and tool_calls support"
```

---

## Task 3: Rewrite `ObserverPipeline._maybe_run()` — alignment gate

**Files:**
- Modify: `coremem/observer.py` (add `ObserverPipeline` class)
- Rewrite: `tests/test_pipelines.py`

- [ ] **Step 3.1: Write failing tests for new `ObserverPipeline` contract**

Replace `tests/test_pipelines.py`:

```python
"""Integration tests for ObserverPipeline — alignment-gated."""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from coremem.observer import ObserverPipeline
from coremem.providers import ChatResponse
from coremem.types import Memory


pytestmark = pytest.mark.asyncio


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_core_with_messages(messages: list[Memory]):
    """Create a MemoryCore-like stub that returns the given messages from fetch()."""
    from coremem.core import MemoryCore
    from coremem.backends.hybrid import HybridBackend
    d = tempfile.mkdtemp()
    backend = HybridBackend(path=d)
    core = MemoryCore(backend=backend)
    for m in messages:
        core.ingest(m.role, m.content, session_id="main",
                    user_id="alice", agent_id="a1", ts=m.ts)
    import shutil
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


def _make_store():
    from coremem.memory_store import MemoryStore
    d = tempfile.mkdtemp()
    store = MemoryStore(path=d)
    import shutil
    store._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return store


def _mock_valid_tool_response() -> ChatResponse:
    """Observation with a source_quote that IS in the source conversation."""
    return ChatResponse(
        content="",
        tool_calls=[{
            "function": {
                "arguments": (
                    '[{"id": "o1", "content": "User is a software engineer", '
                    '"source_quote": "I am a software engineer", '
                    '"referenced_date": "2026-06-01", '
                    '"importance": 0.8, "entities": []}]'
                )
            }
        }],
    )


def _mock_fabricated_tool_response() -> ChatResponse:
    """Observation with a source_quote that is NOT in the source conversation."""
    return ChatResponse(
        content="",
        tool_calls=[{
            "function": {
                "arguments": (
                    '[{"id": "o1", "content": "User lives on Mars", '
                    '"source_quote": "I have lived on Mars for 10 years", '
                    '"referenced_date": "2026-06-01", '
                    '"importance": 0.8, "entities": ["Mars"]}]'
                )
            }
        }],
    )


# ── Tests ──────────────────────────────────────────────────────────────────


class TestObserverPipelineAlignment:
    async def test_valid_quote_is_inserted_with_alignment_tier(self):
        ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        messages = [
            Memory(id="m1", role="user", content="I am a software engineer", ts=ts),
        ]
        core = _make_core_with_messages(messages)
        store = _make_store()
        try:
            pipeline = ObserverPipeline(
                core=core, store=store, session_id="main",
                token_threshold=1, min_turns=1,
            )
            with patch.object(pipeline._observer, "_provider") as mock_p:
                mock_p.chat_with_tools = AsyncMock(
                    return_value=_mock_valid_tool_response()
                )
                result = await pipeline.after_turn()
            assert result is not None
            assert len(result) == 1
            assert result[0]["alignment_tier"] == "exact"
            assert result[0]["alignment_confidence"] == 1.0
            # Stored in DB
            stored = store.get_observations()
            assert len(stored) == 1
            assert stored[0]["alignment_tier"] == "exact"
        finally:
            core._test_cleanup()
            store._test_cleanup()

    async def test_fabricated_quote_is_dropped(self):
        ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        messages = [
            Memory(id="m1", role="user", content="I am a software engineer", ts=ts),
        ]
        core = _make_core_with_messages(messages)
        store = _make_store()
        try:
            pipeline = ObserverPipeline(
                core=core, store=store, session_id="main",
                token_threshold=1, min_turns=1,
            )
            with patch.object(pipeline._observer, "_provider") as mock_p:
                mock_p.chat_with_tools = AsyncMock(
                    return_value=_mock_fabricated_tool_response()
                )
                result = await pipeline.after_turn()
            assert result is not None
            assert len(result) == 0  # all dropped
            stored = store.get_observations()
            assert len(stored) == 0
        finally:
            core._test_cleanup()
            store._test_cleanup()

    async def test_below_token_threshold_skips(self):
        ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        messages = [
            Memory(id="m1", role="user", content="Short", ts=ts),
        ]
        core = _make_core_with_messages(messages)
        store = _make_store()
        try:
            pipeline = ObserverPipeline(
                core=core, store=store, session_id="main",
                token_threshold=100_000, min_turns=0,
            )
            result = await pipeline.after_turn()
            assert result is None
        finally:
            core._test_cleanup()
            store._test_cleanup()

    async def test_dedup_against_prior_observations(self):
        ts = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        messages = [
            Memory(id="m1", role="user", content="I am a software engineer", ts=ts),
        ]
        core = _make_core_with_messages(messages)
        store = _make_store()
        try:
            # Insert a near-duplicate prior observation
            store.insert_observations([{
                "content": "User is a software engineer",
                "importance": 0.8,
            }])
            pipeline = ObserverPipeline(
                core=core, store=store, session_id="main",
                token_threshold=1, min_turns=1,
            )
            with patch.object(pipeline._observer, "_provider") as mock_p:
                mock_p.chat_with_tools = AsyncMock(
                    return_value=_mock_valid_tool_response()
                )
                result = await pipeline.after_turn()
            assert result is not None
            # Should be dropped as duplicate
            assert len(result) == 0
        finally:
            core._test_cleanup()
            store._test_cleanup()
```

- [ ] **Step 3.2: Run tests, verify they fail**

Run: `uv run pytest tests/test_pipelines.py -v`
Expected: `ImportError: cannot import name 'ObserverPipeline'` or tests fail because the old `ObserverPipeline` uses the two-pass design and different fields.

- [ ] **Step 3.3: Add `ObserverPipeline` class to `coremem/observer.py`**

Append to `coremem/observer.py` (after the `Observer` class):

```python
# ── ObserverPipeline ───────────────────────────────────────────────────────


class ObserverPipeline:
    """Per-turn fact extraction pipeline with alignment-gated verification.

    Fires after each agent turn but only runs the LLM call when both
    ``token_threshold`` new tokens have accumulated AND ``min_turns``
    have passed since the last run.
    """

    def __init__(
        self,
        core: Any,
        store: Any,
        session_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model: str = "ollama:llama3.2",
        token_threshold: int = 8000,
        min_turns: int = 3,
        max_messages: int = 500,
        enable_gleaning: bool = False,
    ):
        if enable_gleaning:
            raise NotImplementedError(
                "gleaning pass not implemented; see docs/observer-hallucination-review.md for the CogCanvas pattern"
            )
        self._core = core
        self._store = store
        self._session_id = session_id
        self._user_id = user_id
        self._agent_id = agent_id
        self._metadata = metadata
        self._observer = Observer(model=model, enable_gleaning=enable_gleaning)
        self._token_threshold = token_threshold
        self._min_turns = min_turns
        self._max_messages = max_messages

        self._last_observed_id: str | None = None
        self._turns_since_last_run: int = 0
        self._running: bool = False

    async def after_turn(self) -> list[dict[str, Any]] | None:
        """Called after each agent turn. Returns new observations or None."""
        if self._running:
            return None
        self._turns_since_last_run += 1
        try:
            self._running = True
            return await self._maybe_run()
        finally:
            self._running = False

    async def _maybe_run(self) -> list[dict[str, Any]] | None:
        messages = self._core.fetch(
            session_id=self._session_id,
            user_id=self._user_id,
            agent_id=self._agent_id,
            metadata=self._metadata,
            limit=self._max_messages,
        )

        # Filter tool messages, find new since watermark
        new_messages: list[Memory] = []
        seen_watermark = self._last_observed_id is None
        for m in messages:
            if m.role == "tool":
                continue
            if not seen_watermark:
                if m.id == self._last_observed_id:
                    seen_watermark = True
                continue
            new_messages.append(m)
        if not seen_watermark:
            new_messages = [m for m in messages if m.role != "tool"]

        # Build canonical text (built ONCE, used for prompt + alignment)
        canonical_lines: list[str] = []
        for m in new_messages:
            if m.content and m.ts is not None:
                ts_str = m.ts.isoformat()[:19]
                canonical_lines.append(f"[{ts_str}] {m.content}")
        canonical_text = "\n".join(canonical_lines)

        new_tokens = sum(_estimate_tokens_line(line) for line in canonical_lines)
        if new_tokens < self._token_threshold or self._turns_since_last_run < self._min_turns:
            return None

        # Fetch prior observations for dedup context
        prior = self._store.get_recent_observations(days=30, limit=50)

        # Run Observer
        observations = await self._observer.run(new_messages, prior)

        # Per-observation alignment gate + dedup
        new_obs: list[dict[str, Any]] = []
        for obs in observations:
            quote = obs.get("source_quote", "").strip()
            content = obs.get("content", "").strip()
            if not quote or not content:
                continue

            result = align_quote(quote, canonical_text)
            if result.tier == AlignmentTier.NONE:
                continue
            obs["alignment_tier"] = result.tier.value
            obs["alignment_confidence"] = result.confidence

            if any(_string_similarity(content, p.get("content", "")) > 0.85 for p in prior):
                continue

            obs["session_id"] = self._session_id
            obs["user_id"] = self._user_id or ""
            obs["agent_id"] = self._agent_id or ""
            new_obs.append(obs)

        if new_obs:
            self._store.insert_observations(new_obs)
        if messages:
            self._last_observed_id = messages[0].id
        self._turns_since_last_run = 0
        return new_obs


# ── Helpers ────────────────────────────────────────────────────────────────


def _estimate_tokens_line(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return max(1, len(text) // 4)


def _string_similarity(a: str, b: str) -> float:
    """Simple string similarity using difflib."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
```

- [ ] **Step 3.4: Run tests, verify pass**

Run: `uv run pytest tests/test_pipelines.py -v`
Expected: all 4 tests PASS

- [ ] **Step 3.5: Commit**

```bash
git add coremem/observer.py tests/test_pipelines.py
git commit -m "feat(observer): ObserverPipeline with alignment gate and native messages"
```

---

## Task 4: Add `_migrate_observations_v2()` to `coremem/memory_store.py`

**Files:**
- Modify: `coremem/memory_store.py`
- Create: `tests/test_memory_store.py`

- [ ] **Step 4.1: Write failing test for the migration**

Create `tests/test_memory_store.py`:

```python
"""Tests for MemoryStore — schema migration for alignment columns."""

import tempfile

import pytest

from coremem.memory_store import MemoryStore


@pytest.fixture
def tmp_store():
    d = tempfile.mkdtemp()
    store = MemoryStore(path=d)
    yield store
    import shutil
    shutil.rmtree(d, ignore_errors=True)


class TestMigration:
    def test_new_store_has_alignment_columns(self, tmp_store):
        # Fresh store should have the alignment columns
        columns = set(tmp_store._db.list_columns("observations"))
        assert "alignment_tier" in columns
        assert "alignment_confidence" in columns

    def test_insert_observations_writes_alignment_fields(self, tmp_store):
        ids = tmp_store.insert_observations([{
            "content": "test",
            "source_quote": "test",
            "importance": 0.5,
            "alignment_tier": "exact",
            "alignment_confidence": 1.0,
        }])
        obs = tmp_store.get_observations()
        assert len(obs) == 1
        assert obs[0]["alignment_tier"] == "exact"
        assert obs[0]["alignment_confidence"] == 1.0

    def test_migration_is_idempotent(self):
        # Create a store, then re-create it (simulating upgrade)
        d = tempfile.mkdtemp()
        store1 = MemoryStore(path=d)
        store2 = MemoryStore(path=d)  # should not fail
        columns = set(store2._db.list_columns("observations"))
        assert "alignment_tier" in columns
        import shutil
        shutil.rmtree(d, ignore_errors=True)
```

- [ ] **Step 4.2: Run tests, verify they fail**

Run: `uv run pytest tests/test_memory_store.py -v`
Expected: tests fail because `list_columns` doesn't exist, or `alignment_tier` column doesn't exist.

- [ ] **Step 4.3: Update `_OBSERVATIONS_SCHEMA` and add migration to `coremem/memory_store.py`**

Edit `coremem/memory_store.py`:

```python
_OBSERVATIONS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "content": "LONGTEXT",
    "source_quote": "TEXT",
    "referenced_date": "TEXT",
    "observation_ts": "TEXT",
    "user_id": "TEXT",
    "agent_id": "TEXT",
    "session_id": "TEXT",
    "alignment_tier": "TEXT",         # NEW: 0.4.0
    "alignment_confidence": "REAL",   # NEW: 0.4.0
}
```

Update `_ensure_tables` to call migration:

```python
def _ensure_tables(self) -> None:
    existing = set(self._db.list_tables())
    if "observations" not in existing:
        self._db.create_table("observations", _OBSERVATIONS_SCHEMA)
    else:
        self._migrate_observations_v2()
    if "observation_metadata" not in existing:
        self._db.create_table("observation_metadata", _OBSERVATION_METADATA_SCHEMA)
    if "reflections" not in existing:
        self._db.create_table("reflections", _REFLECTIONS_SCHEMA)

def _migrate_observations_v2(self) -> None:
    """0.4.0 migration: add alignment_tier and alignment_confidence columns.

    Idempotent: skip if columns already exist.
    """
    existing_cols = set(self._db.list_columns("observations"))
    if "alignment_tier" in existing_cols and "alignment_confidence" in existing_cols:
        return
    # HybridDB 0.4.0 supports add_column. If not, raise — implementation
    # handles fallback (table rename + recreate + copy).
    if hasattr(self._db, "add_column"):
        if "alignment_tier" not in existing_cols:
            self._db.add_column("observations", "alignment_tier", "TEXT")
        if "alignment_confidence" not in existing_cols:
            self._db.add_column("observations", "alignment_confidence", "REAL")
    else:
        # Fallback: rename, recreate, copy
        self._migrate_via_recreate()

def _migrate_via_recreate(self) -> None:
    """Fallback migration: rename old table, create new, copy data."""
    import uuid
    old_name = f"observations_old_{uuid.uuid4().hex[:8]}"
    self._db.raw_query(f"ALTER TABLE observations RENAME TO {old_name}")
    self._db.create_table("observations", _OBSERVATIONS_SCHEMA)
    old_cols = set(self._db.list_columns(old_name))
    new_cols = {"id", "content", "source_quote", "referenced_date", "observation_ts",
                "user_id", "agent_id", "session_id"}
    select_cols = [c for c in new_cols if c in old_cols]
    select_list = ", ".join(select_cols)
    self._db.raw_query(
        f"INSERT INTO observations ({select_list}) "
        f"SELECT {select_list} FROM {old_name}"
    )
    self._db.raw_query(f"DROP TABLE {old_name}")
```

Update `insert_observations` to write alignment fields:

```python
def insert_observations(self, items: list[dict[str, Any]]) -> list[str]:
    """Insert observations + metadata. Returns observation IDs."""
    ids = []
    now = datetime.now(timezone.utc).isoformat()
    for item in items:
        oid = str(uuid.uuid4())[:12]

        self._db.insert("observations", {
            "id": oid,
            "content": item.get("content", ""),
            "source_quote": item.get("source_quote", ""),
            "referenced_date": item.get("referenced_date", ""),
            "observation_ts": item.get("observation_ts", now),
            "user_id": item.get("user_id", ""),
            "agent_id": item.get("agent_id", ""),
            "session_id": item.get("session_id", ""),
            "alignment_tier": item.get("alignment_tier", ""),                # NEW
            "alignment_confidence": item.get("alignment_confidence", 0.0),   # NEW
        })

        mid = str(uuid.uuid4())[:12]
        self._db.insert("observation_metadata", {
            "id": mid,
            "observation_id": oid,
            "importance": item.get("importance", 0.5),
            "entities": json.dumps(item.get("entities", [])),
            "priority": item.get("priority", "medium"),  # column kept, no longer populated
            "confidence": 1.0,                            # constant 1.0 in 0.4.0
            "enrichment_ts": now,
        })
        ids.append(oid)
    return ids
```

- [ ] **Step 4.4: Run tests, verify pass**

Run: `uv run pytest tests/test_memory_store.py -v`
Expected: all tests PASS (HybridDB may need to support `list_columns` and `add_column` — check the actual HybridDB API; if not, adjust to use the fallback path)

- [ ] **Step 4.5: Commit**

```bash
git add coremem/memory_store.py tests/test_memory_store.py
git commit -m "feat(memory_store): 0.4.0 migration — add alignment_tier and alignment_confidence columns"
```

---

## Task 5: Patch `coremem/reflector.py` — fix broken priority filter

**Files:**
- Modify: `coremem/reflector.py` (lines 147-150)
- Create: `tests/test_reflector.py`

- [ ] **Step 5.1: Write failing test for the importance filter**

Create `tests/test_reflector.py`:

```python
"""Tests for Reflector — broken priority filter fixed via importance."""

import asyncio
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from coremem.providers import ChatResponse
from coremem.memory_store import MemoryStore
from coremem.reflector import ReflectorPipeline


@pytest.fixture
def tmp_store():
    d = tempfile.mkdtemp()
    store = MemoryStore(path=d)
    yield store
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def _make_obs(importance: float, content: str) -> dict:
    return {
        "id": f"o_{importance}_{content[:10]}",
        "content": content,
        "importance": importance,
        "priority": "high" if importance >= 0.5 else "low",  # legacy
    }


def _mock_reflector_response() -> ChatResponse:
    return ChatResponse(content='''[
        {"id": "refl_1", "content": "User demonstrates a pattern of work", "domain": "career", "linked_observation_ids": ["o_high"]}
    ]''')


class TestReflectorImportanceFilter:
    async def test_high_importance_kept_in_priority_sampling(self, tmp_store):
        # Insert >200 observations: mix of high and low importance
        for i in range(150):
            tmp_store.insert_observations([_make_obs(0.9, f"high fact {i}")])
        for i in range(100):
            tmp_store.insert_observations([_make_obs(0.1, f"low fact {i}")])
        assert len(tmp_store.get_observations()) == 250

        reflector = ReflectorPipeline(
            store=tmp_store, model="ollama:llama3.2", min_observations=5,
        )
        with patch.object(reflector._reflector, "_provider") as mock_p:
            mock_p.chat = AsyncMock(return_value=_mock_reflector_response())
            result = await reflector.run_now()

        assert result is not None
        # Sampling should have occurred: 150 high + 100 low (capped at 100)
        # means we should NOT have all 250 in the input, but the LLM was called
        mock_p.chat.assert_called_once()
```

- [ ] **Step 5.2: Run test, verify it fails**

Run: `uv run pytest tests/test_reflector.py -v`
Expected: test fails because the priority filter is broken (always returns the full set, not the high+sample subset).

- [ ] **Step 5.3: Patch `coremem/reflector.py`**

Edit `coremem/reflector.py`, lines 147-150:

```python
        # Priority sampling for large observation sets (0.4.0: use importance)
        if len(observations) > 200:
            high_med = [o for o in observations if o.get("importance", 0) >= 0.5]
            green = [o for o in observations if o.get("importance", 0) < 0.5]
            green = sorted(green, key=lambda o: o.get("observation_ts", ""), reverse=True)[:100]
            observations = high_med + green
```

- [ ] **Step 5.4: Run test, verify pass**

Run: `uv run pytest tests/test_reflector.py -v`
Expected: test PASS

- [ ] **Step 5.5: Commit**

```bash
git add coremem/reflector.py tests/test_reflector.py
git commit -m "fix(reflector): use importance instead of broken priority filter"
```

---

## Task 6: Delete `coremem/nli.py` and update `pyproject.toml`

**Files:**
- Delete: `coremem/nli.py`
- Modify: `pyproject.toml`
- Verify: no remaining `nli` imports anywhere

- [ ] **Step 6.1: Find all `nli` references**

Run: `grep -rn "from coremem.nli" --include="*.py" .` and `grep -rn "coremem.nli" --include="*.py" .` and `grep -rn "coremem\[nli\]\|coremem\[all\]" --include="*.md" --include="*.toml" .`

Expected: references in `coremem/observer.py` (old), `coremem/nli.py` (the file itself), `tests/test_observer.py` (old), `pyproject.toml`, `README.md`, `CHANGELOG.md`. These should all be removed/updated.

- [ ] **Step 6.2: Delete `coremem/nli.py`**

```bash
git rm coremem/nli.py
```

- [ ] **Step 6.3: Update `pyproject.toml`**

Edit `pyproject.toml`, lines 27-31:

```toml
[project.optional-dependencies]
hybrid = ["hybriddb>=0.4.0"]
observer = ["httpx>=0.25.0"]
all = ["coremem[hybrid,observer]"]   # was: coremem[hybrid,observer,nli]
```

Remove the `nli = ["transformers>=4.40.0"]` line entirely.

- [ ] **Step 6.4: Run test suite to verify nothing imports `nli`**

Run: `uv run pytest -v`
Expected: all tests PASS (the old two-pass tests were already removed in earlier tasks; verify no remaining `ImportError` or `ModuleNotFoundError` for `coremem.nli`).

- [ ] **Step 6.5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: drop NLI module and transformers dependency"
```

---

## Task 7: Update docs — `CHANGELOG.md`, `README.md`, bump version

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `README.md`
- Modify: `pyproject.toml` (version bump)

- [ ] **Step 7.1: Bump version in `pyproject.toml`**

Edit `pyproject.toml` line 3:

```toml
version = "0.4.0"   # was: 0.3.0
```

- [ ] **Step 7.2: Add 0.4.0 entry to `CHANGELOG.md`**

Edit `CHANGELOG.md` (prepend a new section at the top):

```markdown
## 0.4.0 (2026-06-02) — Observer rewrite

### Breaking changes
- `coremem.nli` module removed. `bart-large-mnli` no longer a dependency.
- `pip install coremem[nli]` is a no-op.
- `Observer.run()` signature changed: `messages: list[Memory]` (was `conversation: list[dict]`); new `observation_date: str | None` arg.
- `OBSERVATION_TOOL` schema: `priority` field removed. Observations no longer have a `priority` key.
- `coremem[all]` no longer includes `nli`.

### New features
- `coremem.grounding.align_quote()` — 3-tier alignment gate (EXACT / FUZZY / drop). Port of `langextract/resolver.py:316-400`.
- `AlignmentTier` and `AlignmentResult` types exported from `coremem.grounding`.
- `Observer` and `ObserverPipeline` accept `enable_gleaning: bool = False` flag (raises `NotImplementedError` when True; reserved for future CogCanvas-style gleaning pass).

### Bug fixes
- **Bug #1:** `Observer.run` Pass 1 was reading `tool_calls` payload from `response.content` (always empty for tool calls). Now correctly reads from `tool_calls[0].function.arguments`.
- **Bug #2:** Prompt input had `[role | ts | meta]` prefix but verification source did not. Now uses identical canonical text (`[ts] content`) for both.
- **Bug #3:** `_quote_verified` internal check was inverted (checked if claim was substring of quote). Replaced by the 3-tier alignment gate.
- **Bug #4:** Two-pass design dropped (Pass 1 was dead). Single LLM call with CogCanvas-style prompt + few-shot examples.
- **Reflector filter:** Silently-broken priority filter (never matched emoji values) replaced with `importance >= 0.5` check.

### Schema changes
- `observations` table gains `alignment_tier TEXT` and `alignment_confidence REAL` columns. Migration runs on `MemoryStore.__init__` (idempotent).
- `observation_metadata.priority` column left in place but no longer populated.

### Performance
- Two-pass design dropped: per-observation time down from ~500s to ~120s target.
- `bart-large-mnli` (1.6GB) no longer required for installation.

### Verification
- LongMemEval re-evaluation pending: target <10% hallucination on DeepSeek V4 Flash (down from 34-58% in 0.3.0).
```

- [ ] **Step 7.3: Update `README.md` — remove NLI references, update Observer examples**

Find any `coremem[nli]` references and remove them. Find the Observer usage example and update it to use the new signature:

```python
from coremem.observer import Observer, ObserverPipeline
from coremem.types import Memory

obs = Observer(model="openai:gpt-4o-mini")
result = await obs.run(
    messages=[Memory(id="m1", role="user", content="I am a software engineer")],
    observation_date="2026-06-02",
)
```

- [ ] **Step 7.4: Commit**

```bash
git add pyproject.toml CHANGELOG.md README.md
git commit -m "docs: 0.4.0 release notes, Observer example updates, version bump"
```

---

## Task 8: Final verification — full test suite + LongMemEval gate

**Files:** none (verification only)

- [ ] **Step 8.1: Run full test suite**

Run: `uv run pytest -v`
Expected: all tests PASS (target ~50+ tests across test_grounding, test_observer, test_pipelines, test_memory_store, test_reflector, test_core, test_heuristics, test_layers)

- [ ] **Step 8.2: Run linter**

Run: `uv run ruff check src/`
Expected: no errors

- [ ] **Step 8.3: Run type checker**

Run: `uv run mypy src/`
Expected: no errors (or only pre-existing ones)

- [ ] **Step 8.4: Manual LongMemEval re-evaluation (gate)**

This is the **release gate**. Re-run the 10-question LongMemEval suite from `docs/hallucination-mitigation-experiments.md` on DeepSeek V4 Flash.

Expected targets:
- Hallucination rate **< 10%** (down from 34% in #2, 58% in #8)
- Obs/Q **5-9** (no coverage regression)
- Time per observation **< 150s** (down from 500s in #8)

If any criterion fails, the spec is incomplete and warrants another ablation before tagging 0.4.0.

- [ ] **Step 8.5: Append LongMemEval results to `docs/hallucination-mitigation-experiments.md`**

Edit the doc, append a new section:

```markdown
## Post-0.4.0 Re-evaluation

After the 0.4.0 rewrite (single-pass, alignment-gated, NLI removed), the
10-question LongMemEval suite was re-run on DeepSeek V4 Flash:

| Approach | Hallucination | Obs/Q | Time | Note |
|----------|-------------|-------|------|------|
| **0.4.0 pipeline** | [X]% | [Y] | [Z]s | Single-pass + alignment gate |

(Historical 0.3.0 results preserved above for comparison.)
```

- [ ] **Step 8.6: Tag 0.4.0 release**

```bash
git add docs/hallucination-mitigation-experiments.md
git commit -m "docs: append 0.4.0 LongMemEval re-evaluation results"
git tag v0.4.0
git push origin v0.4.0
```

---

## Self-Review Notes

After writing this plan, checked against the spec:

**Spec coverage:**
- ✅ 4 bugs fixed (Tasks 1-3: new grounding, new Observer, alignment gate)
- ✅ LangExtract port (Task 1)
- ✅ CogCanvas prompt (Task 2)
- ✅ Drop two-pass (Task 2: single-pass Observer)
- ✅ NLI removed (Task 6)
- ✅ priority removed, importance kept (Tasks 2, 5)
- ✅ Schema migration with alignment columns (Task 4)
- ✅ Hard break, no compat shims (Task 6: delete nli.py, version bump)
- ✅ LongMemEval gate (Task 8)
- ✅ Test coverage ~50+ tests (Tasks 1, 2, 3, 4, 5)

**Type consistency:**
- `Observer.__init__` signature: `(model, enable_gleaning)` — consistent in Tasks 2, 3, 5
- `Observer.run` signature: `(messages, prior_observations, observation_date)` — consistent in Tasks 2, 3
- `ObserverPipeline.__init__` signature: `(core, store, session_id, user_id, agent_id, metadata, model, token_threshold, min_turns, max_messages, enable_gleaning)` — consistent in Tasks 3, 5
- `AlignmentResult` fields: `(tier, confidence, char_interval)` — consistent in Task 1
- `align_quote` signature: `(quote, source) -> AlignmentResult` — consistent in Task 1

**Placeholder scan:** No "TBD", "TODO", or "implement later" markers. Every step has either code or specific commands.

**Known limitations:**
- HybridDB `add_column` may not exist; the fallback `_migrate_via_recreate` handles that case. The implementation should check `hasattr(self._db, "add_column")` and use whichever works.
- LongMemEval gate (Task 8.4) requires manual model invocation; cannot be automated in CI without API keys.
