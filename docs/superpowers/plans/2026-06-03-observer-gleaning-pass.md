# Observer Gleaning Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CogCanvas-style gleaning pass that reviews extraction results and finds missed entities, pronouns, and implicit facts.

**Architecture:** Modify `Observer.run()` to accept gleaning context, add `GLEANING_SYSTEM_PROMPT`, and change `ObserverPipeline._maybe_run()` from 4 independent passes to 2-stage (extraction + gleaning).

**Tech Stack:** Python 3.11+, coremem proprietary (grounding.py, observer.py, providers.py)

---

### Task 1: Add GLEANING_SYSTEM_PROMPT and modify _build_context_block

**Files:**
- Modify: `coremem/observer.py:26-58` (add prompt after OBSERVER_SYSTEM_PROMPT)
- Modify: `coremem/observer.py:169-188` (_build_context_block)

- [ ] **Step 1: Add GLEANING_SYSTEM_PROMPT constant**

Add after OBSERVER_SYSTEM_PROMPT closing `"""` on line 58:

```python
GLEANING_SYSTEM_PROMPT = """You are a review agent. Read the conversation again and find facts we may have missed.

The first pass extracted these facts from the conversation:
{already_extracted}

REVIEW TASKS:
1. Named entities: read every user message looking for named people, places, companies, products, services, apps, locations. We may have missed some.
2. Pronoun references: check for "he", "she", "they", "it" — what do they refer to? These often hide facts.
3. Implicit facts: facts stated indirectly. Example: "I've been doing this for years" → user has years of experience.
4. Buried preferences: check long messages for likes/dislikes mentioned in passing.
5. Plans/past events in passing: "by the way" statements and dependent clauses often contain plans or events.
6. Relationships: connections between people and entities we may have missed.

RULES:
- One fact per observation.
- source_quote: VERBATIM sub-string. Copy-paste exactly — do not rephrase or change a single character.
- Do NOT repeat facts we already extracted.
- Only extract facts from USER messages. Skip assistant content.
- If you cannot find a verbatim sub-string, you must not return the observation.

IMPORTANCE: 0.8-1.0 identity/contact/job/salary.
            0.5-0.7 preferences/habits/projects.
            0.1-0.4 context/trivia.
ENTITIES: list of named entities (people, companies, products, locations).

Few-shot examples:

Example 1 — missed entities and implicit facts:

Already extracted:
- User works at Anthropic as a research engineer
- User moved to Seattle in January 2024

Conversation:
[2024-01-15T10:00:00] user: I just moved to Seattle last month for a new job at Anthropic. I'll be working as a research engineer on alignment.
[2024-01-15T10:01:00] assistant: Exciting move! How are you finding Seattle?
[2024-01-15T10:02:00] user: It's great! The weather is different from my hometown of Austin. I love the coffee scene here and have been exploring hiking trails on weekends. My wife Sarah also moved with me — she's an architect.

{"id": "obs_1", "content": "User previously lived in Austin", "source_quote": "my hometown of Austin", "importance": 0.6, "entities": ["Austin"]}
{"id": "obs_2", "content": "User enjoys Seattle's coffee scene", "source_quote": "I love the coffee scene", "importance": 0.4, "entities": ["Seattle"]}
{"id": "obs_3", "content": "User goes hiking on weekends", "source_quote": "exploring hiking trails on weekends", "importance": 0.5, "entities": []}
{"id": "obs_4", "content": "User's wife Sarah is an architect", "source_quote": "My wife Sarah also moved with me — she's an architect", "importance": 0.9, "entities": ["Sarah"]}

Example 2 — missed tools and plans:

Already extracted:
- User's favorite programming language is Rust
- User uses Python for data work

Conversation:
[2024-02-03T14:30:00] user: My favorite programming language is Rust, though I still use Python for data work. I'm also learning Korean in my free time — been at it about 8 months. I use Duolingo and have a language exchange partner named Ji-hye. My goal is to be conversational by the time I visit Seoul this summer for a tech conference.

{"id": "obs_1", "content": "User has been learning Korean for 8 months", "source_quote": "been at it about 8 months", "importance": 0.6, "entities": []}
{"id": "obs_2", "content": "User uses Duolingo for language learning", "source_quote": "I use Duolingo", "importance": 0.4, "entities": ["Duolingo"]}
{"id": "obs_3", "content": "User has a language exchange partner named Ji-hye", "source_quote": "a language exchange partner named Ji-hye", "importance": 0.7, "entities": ["Ji-hye"]}
{"id": "obs_4", "content": "User plans to visit Seoul this summer for a tech conference", "source_quote": "visit Seoul this summer for a tech conference", "importance": 0.7, "entities": ["Seoul"]}
"""
```

- [ ] **Step 2: Run tests to verify prompt addition doesn't break anything**

```bash
uv run pytest tests/test_observer.py -x -q
```

Expected: 118 passed (all existing tests pass)

- [ ] **Step 3: Commit**

```bash
git add coremem/observer.py
git commit -m "feat: add GLEANING_SYSTEM_PROMPT constant"
```

---

### Task 2: Add gleaning_context parameter to Observer.run()

**Files:**
- Modify: `coremem/observer.py:132-145` (Observer.run)
- Modify: `coremem/observer.py:147-167` (_build_messages)

- [ ] **Step 1: Add gleaning_context parameter**

```python
async def run(
    self,
    messages: list[Memory],
    prior_observations: list[dict[str, Any]] | None = None,
    observation_date: str | None = None,
    focus: str | None = None,
    user_only: bool = False,
    gleaning_context: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    prior = prior_observations or []
    date_str = observation_date or datetime.now(UTC).date().isoformat()

    llm_messages = self._build_messages(
        messages, prior, date_str,
        focus=focus, user_only=user_only,
        gleaning_context=gleaning_context,
    )
    response = await self._provider.chat_with_tools(llm_messages, [OBSERVATION_TOOL])
    return self._parse_response(response)
```

- [ ] **Step 2: Update _build_messages to use GLEANING_SYSTEM_PROMPT when gleaning**

When gleaning, the `prior` list (historical observations) would create a second "already extracted" list in the context block (`# Already extracted (last 20)\n...`), conflicting with the extracted facts listed in the gleaning system prompt. Suppress it by passing `prior=[]` to `_build_context_block` when gleaning:

```python
def _build_messages(
    self,
    messages: list[Memory],
    prior: list[dict[str, Any]],
    date_str: str,
    focus: str | None = None,
    user_only: bool = False,
    gleaning_context: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Build the native messages array (system + context + conversation)."""
    system_prompt = OBSERVER_SYSTEM_PROMPT
    context_prior = prior
    if gleaning_context is not None:
        extracted_lines = "\n".join(
            f"- {o.get('content', '')}" for o in gleaning_context
        )
        system_prompt = GLEANING_SYSTEM_PROMPT.replace(
            "{already_extracted}", extracted_lines or "(none)"
        )
        context_prior = []  # Suppress "already extracted" in context block — it's in the system prompt
    context_block = self._build_context_block(context_prior, date_str, focus=focus)
    out: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_block},
    ]
    for m in messages:
        if m.content and m.ts is not None:
            if user_only and m.role != "user":
                continue
            ts_str = m.ts.isoformat()[:19]
            out.append({"role": m.role, "content": f"[{ts_str}] {m.content}"})
    return out
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/ -x -q
```

Expected: 118 passed

- [ ] **Step 4: Commit**

```bash
git add coremem/observer.py
git commit -m "feat: add gleaning_context parameter to Observer.run()"
```

---

### Task 3: Lower dedup threshold from 0.85 to 0.75

**Files:**
- Modify: `coremem/observer.py:337` (dedup threshold)

- [ ] **Step 1: Update threshold**

In `_maybe_run()` (replaced in Task 4), find the dedup lines and change the threshold. The current code has:

```python
if any(_string_similarity(content, p.get("content", "")) > 0.85 for p in dedup_targets):
    continue
```

Change to 0.75:

```python
if any(_string_similarity(content, p.get("content", "")) > 0.75 for p in dedup_targets):
    continue
```

There are multiple occurrences of this dedup check. All must be updated.

- [ ] **Step 2: Verify test for dedup behavior**

Check if any test expects the old threshold:

```bash
rg "0\.85" tests/
```

- [ ] **Step 3: Commit**

```bash
git add coremem/observer.py
git commit -m "fix: lower dedup threshold from 0.85 to 0.75"
```

---

### Task 4: Rewrite ObserverPipeline._maybe_run() with error handling and guard

**Files:**
- Modify: `coremem/observer.py:279-351` (_maybe_run) — full rewrite
- Modify: `coremem/observer.py:113-130` (Observer.__init__) — remove NotImplementedError
- Modify: `coremem/observer.py:250-260` (ObserverPipeline.__init__) — remove NotImplementedError, store _enable_gleaning

- [ ] **Step 1: Remove NotImplementedError from Observer class**

At `coremem/observer.py:120-130`:

```python
def __init__(
    self,
    model: str = "ollama:llama3.2",
    tool_temp: float = 0.1,
):
    self._provider = create_provider(model, tool_temp=tool_temp)
```

Removes `enable_gleaning` parameter and the `raise NotImplementedError` block. The Observer no longer needs it — gleaning is a pipeline concern.

- [ ] **Step 2: Remove NotImplementedError from ObserverPipeline and store flag**

At `coremem/observer.py:250-260`:

```python
def __init__(
    self,
    ...,
    enable_gleaning: bool = False,
    tool_temp: float = 0.1,
):
    self._enable_gleaning = enable_gleaning
    self._core = core
    self._store = store
    ...
    self._observer = Observer(model=model, tool_temp=tool_temp)
```

Removes `enable_gleaning` from Observer constructor call and the NotImplementedError.

- [ ] **Step 3: Run quick test to verify init changes don't break things**

```bash
uv run pytest tests/ -x -q
```

Expected: 118 passed

- [ ] **Step 4: Commit init changes**

```bash
git add coremem/observer.py
git commit -m "refactor: remove enable_gleaning NotImplementedError guards"
```

- [ ] **Step 5: Replace 4-pass loop with 2-stage extraction + gleaning + error handling**

Replace the `_maybe_run()` body from line 313 onwards:

```python
prior = self._store.get_recent_observations(days=30, limit=50)
new_obs: list[dict[str, Any]] = []

# Build canonical text from full conversation
canonical_lines: list[str] = []
for m in new_messages:
    if m.content and m.ts is not None:
        ts_str = m.ts.isoformat()[:19]
        canonical_lines.append(f"[{ts_str}] {m.content}")
canonical_text = "\n".join(canonical_lines)

# Stage 1: Liberal extraction (no focus)
try:
    extraction_obs = await self._observer.run(new_messages, prior)
except Exception as e:
    logger.warning("extraction_error", {"error": str(e)}, user_id=self._user_id)
    extraction_obs = []

stage1_obs: list[dict[str, Any]] = []
for obs in extraction_obs:
    quote = obs.get("source_quote", "").strip()
    content = obs.get("content", "").strip()
    if not quote or not content:
        continue
    result = align_quote(quote, canonical_text)
    if result.tier == AlignmentTier.NONE:
        continue
    obs["alignment_tier"] = result.tier.value
    obs["alignment_confidence"] = result.confidence
    if any(_string_similarity(content, p.get("content", "")) > 0.75 for p in prior):
        continue
    obs["session_id"] = self._session_id
    obs["user_id"] = self._user_id or ""
    obs["agent_id"] = self._agent_id or ""
    obs.pop("id", None)
    stage1_obs.append(obs)

# Stage 2: Gleaning pass — only if enabled AND stage1 produced results
gleaning_obs: list[dict[str, Any]] = []
if self._enable_gleaning and stage1_obs:
    try:
        gleaning_obs = await self._observer.run(
            new_messages, prior, gleaning_context=stage1_obs,
        )
    except Exception as e:
        logger.warning("gleaning_error", {"error": str(e)}, user_id=self._user_id)

stage2_obs: list[dict[str, Any]] = []
for obs in gleaning_obs:
    quote = obs.get("source_quote", "").strip()
    content = obs.get("content", "").strip()
    if not quote or not content:
        continue
    result = align_quote(quote, canonical_text)
    if result.tier == AlignmentTier.NONE:
        continue
    obs["alignment_tier"] = result.tier.value
    obs["alignment_confidence"] = result.confidence
    dedup_targets = prior + stage1_obs + stage2_obs
    if any(_string_similarity(content, p.get("content", "")) > 0.75 for p in dedup_targets):
        continue
    obs["session_id"] = self._session_id
    obs["user_id"] = self._user_id or ""
    obs["agent_id"] = self._agent_id or ""
    obs.pop("id", None)
    stage2_obs.append(obs)

new_obs = stage1_obs + stage2_obs
```

- [ ] **Step 6: Remove dead code: _OBSERVER_FOCUSES dict, _chunk_by_user_turns, user_only parameter**

These are no longer used by the 2-stage pipeline. Remove:
- `_OBSERVER_FOCUSES` dict (lines 101-107)
- `_chunk_by_user_turns` function (lines ~359-375)
- `user_only` parameter from `Observer.run()` and `_build_messages()`
- `focus` loops (the 4-pass loop is already replaced)

Actually, keep `focus` parameter and `_OBSERVER_FOCUSES` dict — they are useful for future re-enablement or testing. Only remove `_chunk_by_user_turns` and `user_only`.

```python
# Remove from Observer.run() signature:
#   user_only: bool = False,

# Remove from _build_messages() signature and body:
#   user_only: bool = False,
#   if user_only and m.role != "user": continue

# Remove function entirely:
# def _chunk_by_user_turns(...): ...
```

- [ ] **Step 7: Run tests**

```bash
uv run pytest tests/ -x -q
```

Expected: 118 passed

- [ ] **Step 8: Commit**

```bash
git add coremem/observer.py
git commit -m "feat: implement 2-stage pipeline with gleaning and error handling"
```

---

### Task 5: Add integration test

**Files:**
- Create: `tests/test_observer_gleaning.py`

- [ ] **Step 1: Add test that validates gleaning_context substitution**

```python
from coremem.observer import GLEANING_SYSTEM_PROMPT

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
        substituted = GLEANING_SYSTEM_PROMPT.replace("{already_extracted}", "- fact one\n- fact two")
        assert "- fact one" in substituted
        assert "- fact two" in substituted
        assert "{already_extracted}" not in substituted
```

- [ ] **Step 2: Add integration test that runs 2-stage pipeline end-to-end**

```python
import os
import tempfile

import pytest
from coremem import MemoryCore, MemoryStore
from coremem.backends.hybrid import HybridBackend
from coremem.observer import ObserverPipeline

skip_if_no_api_key = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set",
)


class TestGleaningIntegration:
    """End-to-end tests for the 2-stage (extraction + gleaning) pipeline."""

    @pytest.mark.asyncio
    @skip_if_no_api_key
    async def test_enable_gleaning_runs_both_stages(self):
        """With enable_gleaning=True, the pipeline runs extraction + gleaning."""
        # Use a lightweight backend
        core = MemoryCore(backend=HybridBackend(path=tempfile.mkdtemp()))
        store = MemoryStore(path=tempfile.mkdtemp())

        core.ingest("user", "I'm a software engineer named Alice. I love hiking.")
        core.ingest("assistant", "Nice to meet you Alice!")
        core.ingest("user", "I also play piano and live in Portland.")

        pipeline = ObserverPipeline(
            core=core, store=store, session_id="test",
            model="deepseek:deepseek-chat", token_threshold=1, min_turns=1,
            enable_gleaning=True,
        )

        observations = await pipeline.after_turn()
        assert observations is not None
        assert len(observations) > 0

        # All observations should have valid alignment
        for obs in observations:
            assert obs.get("alignment_tier") in ("exact", "fuzzy")
            assert obs.get("source_quote")
            assert obs.get("content")
            assert obs.get("importance") is None  # Observer sets None

    @pytest.mark.asyncio
    @skip_if_no_api_key
    async def test_disable_gleaning_skips_second_stage(self):
        """With enable_gleaning=False, only extraction runs."""
        core = MemoryCore(backend=HybridBackend(path=tempfile.mkdtemp()))
        store = MemoryStore(path=tempfile.mkdtemp())

        core.ingest("user", "I'm a software engineer named Bob.")
        core.ingest("assistant", "Hello Bob!")

        pipeline = ObserverPipeline(
            core=core, store=store, session_id="test",
            model="deepseek:deepseek-chat", token_threshold=1, min_turns=1,
            enable_gleaning=False,
        )

        observations = await pipeline.after_turn()
        assert observations is not None
        assert len(observations) > 0

    @pytest.mark.asyncio
    @skip_if_no_api_key
    async def test_empty_stage1_skips_gleaning(self):
        """If extraction produces 0 observations, gleaning is skipped
        (the guard condition: enable_gleaning AND stage1_obs)."""
        core = MemoryCore(backend=HybridBackend(path=tempfile.mkdtemp()))
        store = MemoryStore(path=tempfile.mkdtemp())

        # No user messages with extractable facts — conversation is empty
        core.ingest("assistant", "Hello, how can I help?")

        pipeline = ObserverPipeline(
            core=core, store=store, session_id="test",
            model="deepseek:deepseek-chat", token_threshold=1, min_turns=1,
            enable_gleaning=True,
        )

        observations = await pipeline.after_turn()
        # Should be None (no new user messages) or empty
        assert observations is None or len(observations) == 0
```

Note: integration tests require `DEEPSEEK_API_KEY` in the environment and make real LLM calls (~$0.02 per test). Skip them in CI:

```python
import os
import pytest

skip_if_no_api_key = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set",
)

class TestGleaningIntegration:
    @pytest.mark.asyncio
    @skip_if_no_api_key
    async def test_enable_gleaning_runs_both_stages(self):
        ...
```

- [ ] **Step 3: Run all tests**

```bash
# Unit tests (no API key needed)
uv run pytest tests/ -v --ignore=tests/test_observer_gleaning.py

# Gleaning prompt tests (no API key needed)
uv run pytest tests/test_observer_gleaning.py::TestGleaningPrompt -v

# Integration tests (requires DEEPSEEK_API_KEY)
export $(cat .env | xargs)
uv run pytest tests/test_observer_gleaning.py::TestGleaningIntegration -v
```

Expected: 118 + 5 prompt tests = 123 passed. Integration tests pass or skip based on API key availability.

- [ ] **Step 4: Commit**

```bash
git add tests/test_observer_gleaning.py
git commit -m "test: add gleaning prompt and integration tests"
```

---

### Task 6: Wire gleaning into eval and run verification

**Files:**
- Modify: `benchmarks/longmemeval/observer_eval.py:158-161`

- [ ] **Step 1: Enable gleaning in the eval script**

```python
pipeline = ObserverPipeline(
    core=core, store=store, session_id=sid,
    model=provider, token_threshold=1, min_turns=1,
    tool_temp=0.1,
    enable_gleaning=True,
)
```

- [ ] **Step 2: Run 10-question eval**

```bash
export $(cat .env | xargs)
uv run python -m benchmarks.longmemeval.observer_eval \
  --data /Users/eddy/Developer/Python/CoreMem/results/eval/longmemeval_10q.json \
  --provider deepseek:deepseek-chat \
  --mode both \
  --limit 10 \
  --output results/eval/observer_deepseek_10q_gleaning.json
```

- [ ] **Step 3: Verify hallucination**

```bash
uv run python -c "
import json
from coremem.grounding import align_quote, AlignmentTier
...
# Check that all observations are EXACT-tier
"
```

- [ ] **Step 4: Commit results**

```bash
git add results/eval/observer_deepseek_10q_gleaning.json
git commit -m "eval: 10-question gleaning pipeline results"
```
