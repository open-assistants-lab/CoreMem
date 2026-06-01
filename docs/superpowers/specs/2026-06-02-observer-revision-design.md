# Observer Revision — 0.4.0 Hard Break

2026-06-02

## Context

The current Observer (`coremem/observer.py`) produces a 34-70% hallucination rate
on the LongMemEval 10-question suite. The recent review document
(`docs/observer-hallucination-review.md`) identified that **four concrete bugs**
in the current code, not model capability, account for the bulk of the
hallucination rate. It also identified open-source projects with proven
verbatim-grounded extraction techniques (LangExtract, CogCanvas) whose patterns
can be ported directly.

This spec implements the review's three recommendations as a hard break in
version 0.4.0. The release target is <10% hallucination on DeepSeek V4 Flash
(down from 34-58% in the documented experiments).

## Goals

- Drop the hallucination rate from 34-70% to <10% on LongMemEval / DeepSeek V4 Flash.
- Drop the two-pass design and its 500s per-observation cost.
- Drop the optional 1.6GB `bart-large-mnli` NLI dependency.
- Fix the four documented bugs structurally, not by patches.
- Preserve the 0.3.0 public API shape where possible, breaking only what is
  unavoidable.

## Non-goals

- Re-implementing Mem0-style ADD/UPDATE/DELETE merge logic. Deferred to a
  follow-up spec.
- Cross-encoder (`ms-marco-MiniLM-L-6-v2`) as a replacement NLI. The
  grounding gate is sufficient; cross-encoder is YAGNI.
- Implementing the CogCanvas gleaning pass. Hook only (`enable_gleaning` flag
  raises `NotImplementedError`).
- Schema cleanup of the `observation_metadata.priority` column. Left in place
  for a future spec.

## Architecture

### Current flow (broken)

```
fetch → LLM Pass 1 (tool-call sentence extract, reads wrong field) →
LLM Pass 2 (observation extract) →
source_quote gate (substring, inverted check) →
NLI gate (bart-large-mnli, optional) →
string-similarity dedup → insert
```

### New flow

```
fetch → build canonical text →
LLM single-pass (CogCanvas prompt, temperature=0.1) →
3-tier alignment gate (LangExtract: EXACT / FUZZY / drop) →
string-similarity dedup → insert
```

### Why this works

The four bugs in the current Observer each have a structural fix in the new
pipeline:

| Bug | Cause | Structural fix |
|---|---|---|
| 1. Pass 1 reads wrong field | Reads `response.content` from a tool call (payload is in `tool_calls[0].function.arguments`) | Pass 1 is removed; only one tool call, only one parser |
| 2. Prompt text ≠ verification text | Prompt and verification built from different code paths with different formatting | Canonical text is built once with `[ts] content` format, used for both the LLM prompt and `align_quote()` verification |
| 3. Inverted consistency check | `_quote_verified` checks if `claim in quote` (trivial) | Replaced by alignment gate (checks if `quote in source`, with tier) |
| 4. Inconclusive experiment | Bad few-shot example taught a bad pattern | New 2 few-shot examples in the system prompt; verbatim-quote contract demonstrated by example |

The dominant failure mode (co-fabrication of quote and claim) is caught
deterministically by the alignment gate. Language models can fabricate
content, but they cannot make fabricated content literally appear in the
source conversation.

## Components

### `coremem/grounding.py` (new, ~120 lines)

Pure function, stdlib only. Port of `langextract/resolver.py:316-400`.

```python
class AlignmentTier(Enum):
    EXACT = "exact"   # every token in quote appears in source in order, contiguously
    FUZZY = "fuzzy"   # LCS-based token overlap ≥ 0.75
    NONE = "none"     # below threshold — drop the observation

@dataclass
class AlignmentResult:
    tier: AlignmentTier
    confidence: float                       # 1.0 for EXACT, LCS score for FUZZY, 0.0 for NONE
    char_interval: tuple[int, int] | None   # (start, end) in `source`; None for NONE

def align_quote(quote: str, source: str) -> AlignmentResult: ...
```

Tier semantics:

- **EXACT** — `difflib.SequenceMatcher` on whitespace-tokenized strings
  returns a matching block covering the full quote. Returns `char_interval`
  pointing to that block in `source`. Confidence = 1.0.
- **FUZZY** — `SequenceMatcher.ratio()` on character sequences
  (whitespace + case normalized) ≥ 0.75. Returns `char_interval` pointing
  to the longest common substring (LCS) between `source` and `quote`.
  Chosen over token-level because LLM source_quote values drift by single
  characters or punctuation (e.g. "engineer." vs "engineer", "I'm" vs
  "I am") which would be token-mismatches but should still pass.
  Confidence = the ratio.
- **NONE** — ratio < 0.75. Drop the observation.

Edge cases handled (case-insensitive via lowercase-before-tokenize):

| Case | Tier |
|---|---|
| Perfect match | EXACT |
| Whitespace-only diff | EXACT |
| Case mismatch | EXACT |
| Single char drift | FUZZY (≥0.75) |
| Trailing punctuation | FUZZY |
| <50% overlap | NONE |
| Fabricated quote | NONE |
| Empty quote or source | NONE |
| Quote longer than source | NONE |

### `coremem/observer.py` (rewrite, ~180 lines)

```python
OBSERVER_SYSTEM_PROMPT: str   # instructions + 2 few-shot examples (static)
OBSERVATION_TOOL: dict         # tool schema; `priority` field removed

class Observer:
    def __init__(
        self,
        model: str = "ollama:llama3.2",
        enable_gleaning: bool = False,
    ): ...

    async def run(
        self,
        messages: list[Memory],
        prior_observations: list[dict] | None = None,
        observation_date: str | None = None,
    ) -> list[dict]: ...

class ObserverPipeline:
    def __init__(
        self,
        core, store, session_id,
        user_id=None, agent_id=None, metadata=None,
        model="ollama:llama3.2",
        token_threshold=8000, min_turns=3, max_messages=500,
        enable_gleaning=False,
    ): ...

    async def after_turn(self) -> list[dict] | None: ...
```

`Observer.run()` makes **one** `chat_with_tools` call, reads the payload from
`tool_calls[0].function.arguments`, returns parsed observations or `[]` on
parse failure / empty `tool_calls`. `temperature=0.1`.

`enable_gleaning=True` is **rejected at construction time** (in `__init__`)
with `NotImplementedError("gleaning pass not implemented; see docs/observer-hallucination-review.md for the CogCanvas pattern")`. Fail fast so
misconfiguration is caught before the first LLM call.

### `coremem/observer_utils.py` (unchanged)

`parse_json_array`, `chat_messages`, `estimate_tokens` are still used.

### `coremem/reflector.py` (patch)

Fix the silently-broken priority filter (two lines in
`ReflectorPipeline.run_now()`):

```python
# Was (broken — never matched emoji values):
o.get("priority", "").lower() in ("high", "medium")
o.get("priority", "").lower() not in ("high", "medium")

# Now:
o.get("importance", 0) >= 0.5
o.get("importance", 0) < 0.5
```

### `coremem/memory_store.py` (patch)

Add `_migrate_observations_v2()` to add `alignment_tier TEXT` and
`alignment_confidence REAL` columns to the `observations` table. Run on
`MemoryStore.__init__` if the columns are missing.

`observation_metadata.priority` column **left in place** but no longer
populated by new code. Future spec can drop it.

## Prompt structure (CogCanvas pattern)

### System message

Two parts: instructions + 2 few-shot examples.

**Instructions:**

```
You are an observation agent. Extract key facts from a conversation and
return them as structured observations via the record_observations tool.

RULES:
- One fact per observation. Be exact with values.
- source_quote: a VERBATIM sub-string of the conversation. Copy-paste
  exactly — do not rephrase, do not paraphrase, do not change a single
  character. If you cannot find a verbatim sub-string, do not return
  the observation.
- Do not invent facts. If nothing factual is present, return an empty
  observations array.

IMPORTANCE: 0.8-1.0 identity/contact/job/salary.
            0.5-0.7 preferences/habits/projects.
            0.1-0.4 context/trivia.
ENTITIES: list of named entities (people, companies, products, locations).
```

**Few-shot examples (2):** each is a small synthetic dialogue (2-4 turns,
inline `[ts] content` format — role is carried natively by the messages
array, not in the prefix) followed by a JSON observation whose
`source_quote` is a byte-for-byte sub-string of the dialogue **with the
`[ts]` prefix stripped**. The examples teach the verbatim-quote-without-prefix
contract by demonstration. Examples are static — embedded in source, not
constructed at runtime.

### User message (built per call)

Three sections, sent as a single user message **before** the conversation
turns:

```
# Already extracted (last 20)
- User is a software engineer
- User prefers morning workouts

# Observation date
2026-06-02
```

Then the conversation as native messages:

```python
[
    {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
    {"role": "user", "content": "Already extracted ...\n\nObservation date: ..."},
    {"role": "user", "content": "[2026-06-01T12:00:00] Hello, I am a software engineer."},
    {"role": "assistant", "content": "[2026-06-01T12:01:00] How can I help?"},
    {"role": "user", "content": "[2026-06-01T12:02:00] I'm working on a memory layer."},
]
```

Native messages array — no JSON wrapping of content. The provider's first-class
`role` field carries role information; no format-hack prefix needed for that.
The inline `[ts]` prefix carries per-message timestamp context; few-shot
teaches the model to strip it when quoting.

### `temperature=0.1`

Per CogCanvas (`cog-canvas/cogcanvas/llm/openai.py:11-90`). `0.0` makes the
model over-confident on its first sampled token; `0.1` allows minor rephrasing
that still satisfies the verbatim constraint via the alignment gate.

### What we drop

- `SENTENCE_EXTRACT_PROMPT` and `SENTENCE_EXTRACT_TOOL` — Pass 1 is dead.
- The "Fabricating is the WORST mistake" anti-fab line — review showed it has
  zero effect.
- CoT preamble — CogCanvas uses it, but the review's experiment #6 suggests
  it can hurt small models. Kept off.
- `priority` enum — replaced by `importance` float.

## Pipeline flow

```python
async def _maybe_run(self) -> list[dict] | None:
    # 1. Fetch + watermark
    messages = self._core.fetch(...)
    if not messages:
        return None
    if new_tokens < self._token_threshold or self._turns_since_last_run < self._min_turns:
        return None

    # 2. Build canonical text (built ONCE, used twice for prompt and verification)
    canonical_lines: list[str] = []
    for m in new_messages:
        if m.content and m.ts is not None:
            ts_str = m.ts.isoformat()[:19]   # YYYY-MM-DDTHH:MM:SS
            canonical_lines.append(f"[{ts_str}] {m.content}")
        elif m.content:
            # Skip messages with no timestamp — they can't be quoted deterministically
            continue
    canonical_text = "\n".join(canonical_lines)

    # 3. Single LLM call
    messages_for_llm: list[dict] = [
        {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
        {"role": "user", "content": _build_context_block(prior_observations, observation_date)},
    ]
    for m in new_messages:
        if m.content and m.ts is not None:
            ts_str = m.ts.isoformat()[:19]
            messages_for_llm.append({
                "role": m.role,
                "content": f"[{ts_str}] {m.content}",
            })
    response = await self._provider.chat_with_tools(messages_for_llm, [OBSERVATION_TOOL])
    observations = _parse_tool_response(response)

    # 4. Per-observation gates
    new_obs = []
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
        obs["user_id"] = messages[0].user_id
        obs["agent_id"] = messages[0].agent_id
        new_obs.append(obs)

    # 5. Insert + cursor advance
    if new_obs:
        self._store.insert_observations(new_obs)
    if new_messages:
        self._last_observed_id = messages[0].id
    self._turns_since_last_run = 0
    return new_obs
```

### Error handling

| Failure | Behavior | State |
|---|---|---|
| LLM API error / timeout | Return `None`, log | Cursor unchanged, retry next turn |
| Empty / unparseable `tool_calls` | Return `[]`, log | Cursor advances |
| All observations fail alignment | Return `[]` | Cursor advances |
| All observations deduped | Return `[]` | Cursor advances |
| `MemoryStore` insert error | Propagate | Cursor does **not** advance; next call retries |
| `align_quote` exception | Log + treat as NONE | Observation dropped |

## Public API changes

### Removed

```python
from coremem.nli import check_entailment, is_nli_available  # module gone
obs["priority"]                                          # field gone from tool schema
```

### Changed

```python
# Observer.run signature
# Was (0.3.0):
async def run(self, conversation: list[dict[str, Any]], prior_observations=None)

# Now (0.4.0):
async def run(self, messages: list[Memory], prior_observations=None, observation_date=None)
```

### Added

```python
# New module
from coremem.grounding import align_quote, AlignmentResult, AlignmentTier

# New optional flag (raises NotImplementedError if True)
Observer(..., enable_gleaning: bool = False)
ObserverPipeline(..., enable_gleaning: bool = False)
```

## Schema changes

`observations` table gains two columns:

```sql
ALTER TABLE observations ADD COLUMN alignment_tier TEXT;
ALTER TABLE observations ADD COLUMN alignment_confidence REAL;
```

Migration runs in `MemoryStore._ensure_tables()` via the new
`_migrate_observations_v2()` method. Idempotent (skip if columns exist).

If HybridDB 0.4.0 does not support `ALTER TABLE`, the migration renames the
old table, creates a new one with the full schema, and copies data row by
row. Heavier but acceptable at this scale (single-user SQLite-like store).

## Observation dict — top-level fields

| Field | Source | Persisted |
|---|---|---|
| `id` | LLM (new uuid) | yes (observations.id) |
| `content` | LLM | yes (observations.content) |
| `source_quote` | LLM | yes (observations.source_quote) |
| `referenced_date` | LLM | yes (observations.referenced_date) |
| `importance` | LLM | yes (observation_metadata.importance) |
| `entities` | LLM | yes (observation_metadata.entities) |
| `user_id` | Memory (propagated) | yes (observations.user_id) |
| `agent_id` | Memory (propagated) | yes (observations.agent_id) |
| `session_id` | Memory (propagated) | yes (observations.session_id) |
| `alignment_tier` | pipeline (new) | yes (observations.alignment_tier) |
| `alignment_confidence` | pipeline (new) | yes (observations.alignment_confidence) |

`Memory.metadata` and `Memory.role` are **not** propagated to the observation.
`role` is implicit in the LLM's selection of which facts to extract;
`metadata` is used for fetch-time filtering only.

### `confidence` field (legacy, now constant)

The `confidence` field on `observation_metadata` was set by the NLI gate's
`check_entailment()` return value. With NLI removed, the field is no longer
written by the pipeline. It stays at its default of `1.0` (set in
`MemoryStore.insert_observations()`). Old observations may have non-default
values from the NLI era; the column is left in place and ignored. A future
spec can drop the column or repurpose it.

## Dependency changes

```toml
# pyproject.toml
[project.optional-dependencies]
# Was:
# nli = ["transformers>=4.40.0"]
# all = ["coremem[hybrid,observer,nli]"]
# Now:
all = ["coremem[hybrid,observer]"]    # nli removed
```

`transformers` is no longer a runtime dependency of any extra.

## Testing strategy

| File | Cases | Type |
|---|---|---|
| `tests/test_grounding.py` (new) | ~30 | unit |
| `tests/test_observer.py` (rewritten) | ~10 | unit |
| `tests/test_pipelines.py` (rewritten) | ~6 | integration |

### `test_grounding.py` — exhaustive tier coverage

- EXACT: perfect match (middle / start / end of source)
- EXACT: whitespace-only diff, case mismatch
- FUZZY: single-char drift, trailing punctuation, ≥0.75 overlap
- NONE: <50% overlap, fabricated quote, empty quote/source, quote > source
- `char_interval` accuracy (regression for span positions)

### `test_observer.py` — contract

- Single LLM call (assert `chat_with_tools` called exactly once)
- Reads payload from `tool_calls[0].function.arguments`
- Returns parsed `observations` on valid tool_call
- Returns `[]` on empty `tool_calls` or unparseable content
- System prompt contains few-shot examples (string assertion)
- `temperature=0.1` set in request body
- `enable_gleaning=True` raises `NotImplementedError`

### `test_pipelines.py` — integration

- Mocked provider returns observations with **valid** `source_quote`
  (sub-string of fixture messages) — assert observations inserted with
  `alignment_tier`, `alignment_confidence` populated
- Mocked provider returns observations with **fabricated** `source_quote` —
  assert all dropped
- Existing two-pass mocks removed; `nli.py` import removed

### LongMemEval end-to-end (manual, pre-release)

Re-run the 10-question suite from
`docs/hallucination-mitigation-experiments.md` on DeepSeek V4 Flash.

**Pass criteria for 0.4.0:**

- Hallucination rate **< 10%** (down from 34% in #2, 58% in #8 two-pass)
- Obs/Q **5-9** (no coverage regression)
- Time per observation **< 150s** (down from 500s in #8)

If any criterion fails, the spec is incomplete and warrants another ablation
before tagging 0.4.0.

## Documentation changes

- `CHANGELOG.md` — add 0.4.0 entry listing the four bug fixes, the grounding
  gate, the schema migration, the priority removal, the NLI removal.
- `docs/hallucination-mitigation-experiments.md` — append
  "Post-0.4.0 Re-evaluation" section with new LongMemEval numbers. Preserve
  the original 10-row table as historical baseline.
- `docs/observer-hallucination-review.md` — already exists, no change. This
  spec implements its recommendations.
- `README.md` — remove `pip install coremem[nli]` snippets, update Observer
  examples to use `Memory` list and `observation_date` arg.

## Migration guide (for external users)

```python
# Was (0.3.0):
from coremem.observer import Observer, ObserverPipeline
from coremem.nli import check_entailment
obs = Observer(model="ollama:llama3.2")
result = obs.run(conversation=[{"role": "user", "content": "..."}])

# Now (0.4.0):
from coremem.observer import Observer, ObserverPipeline
from coremem.grounding import align_quote, AlignmentResult, AlignmentTier
obs = Observer(model="ollama:llama3.2")
result = obs.run(
    messages=[Memory(id=..., role="user", content="...")],
    observation_date="2026-06-02",
)
# observations no longer have "priority" key
# any code reading obs["priority"] must read obs["importance"] instead
# pip install coremem[nli] is a no-op; bart-large-mnli is gone
```

## References

- `docs/observer-hallucination-review.md` — the review that motivated this spec
- `docs/hallucination-mitigation-experiments.md` — the 10-experiment baseline
- `langextract/resolver.py:316-400` — ported alignment tier algorithm
- `cog-canvas/cogcanvas/llm/openai.py:11-90` — ported prompt pattern
- `mem0ai/mem0/configs/prompts.py:468-693` — borrowed: `Observation Date`,
  `Already extracted` in-prompt patterns
