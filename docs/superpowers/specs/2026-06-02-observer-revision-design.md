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

---

# 0.5.0 Deltas — Addendum

2026-06-02

## Context

The 0.4.0 release successfully reduced hallucination to 0% on a 10-question
LongMemEval sample, but introduced a structural defect: **40% of conversations
return zero observations** because the Observer prompt tells the model to
return `[]` when no "factual" content is found, and the model is conservative
in judging what counts as a fact. This produced 2.0 obs/q vs a human baseline
of ~7.0 obs/q and a 50% answer-miss rate on the 10-question suite.

Independent of the dead-output issue, the 0.4.0 design conflates extraction
with judgment — the Observer both decides "what's a fact" and "how important
is it." The Reflector `Reflector.run()` itself does real pattern synthesis
(generates higher-level insights via LLM call), but the surrounding
`ReflectorPipeline` filters its input by importance (default ≥ 0.10) and
caps the candidate set, and the 0.4.0 Observer's importance signal is
miscalibrated (Q4 dropped a discrete life event at 0.30/0.40). The result
is that high-quality facts sometimes get dropped before the Reflector ever
sees them.

This addendum specifies six small, surgical changes that move CoreMem from
"0% hallucination, 50% answer miss" toward "≤5% hallucination, ≤20% answer
miss" while preserving the existing alignment gate, the existing pipeline
skeletons, and the 0.4.0 public API shape.

## Goals

- **obs/q** on the 10-question LongMemEval sample: 2.0 → **≥ 5.0**
- **Dead output rate** (conversations returning `[]`): 40% → **< 10%**
- **Answer coverage** (answer retrievable from extracted obs): 5/10 → **≥ 8/10**
- **Hallucination rate** (ungrounded obs): 0% → **< 5%** (allow some for recall)
- **Reflector pipeline preserves the existing synthesis quality** (no
  regression to the 0.4.0 pattern-discovery behavior)

## Non-goals

- Re-implementing Mem0-style ADD/UPDATE/DELETE merge logic (deferred).
- CogCanvas gleaning pass (deferred; `enable_gleaning` stays `NotImplementedError`).
- Replacing the existing Reflector's cosine-similarity dedup algorithm.
- Changing the 0.4.0 alignment gate algorithm.
- Changing the user-data column semantics (existing `id`, `content`,
  `source_quote`, `importance` (when set), etc. are preserved across
  migration). The schema collapse drops the redundant
  `observation_metadata` table and its dead fields, but the data is
  copied first.

## Schema restructure — single table

The 0.4.0 schema split observations into two tables (`observations` +
`observation_metadata`), motivated by an outdated mental model where
humans occasionally enriched observations. In 0.5.0, the Reflector is the
sole writer of enrichment, and the 0.4.0 split causes unnecessary JOINs
on every retrieval query. Several 0.4.0 fields (`priority`, `confidence`,
the redundant `observation_metadata.id` PK) are also dead code. We
collapse to a single table that captures all observation state.

### Final 0.5.0 schema

```sql
CREATE TABLE observations (
    -- Identity
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL DEFAULT 'fact',  -- 'fact' | 'reflection'
    content         TEXT NOT NULL,
    source_quote    TEXT,                            -- verbatim; NULL for reflections
    source_fact_ids TEXT NOT NULL DEFAULT '[]',     -- JSON; non-empty for reflections
    referenced_date TEXT,
    observation_ts  TEXT NOT NULL,

    -- Scope
    user_id         TEXT,
    agent_id        TEXT,
    session_id      TEXT,

    -- Grounding (3-tier alignment, set by Observer at extraction)
    alignment_tier        TEXT,   -- 'EXACT' | 'FUZZY' | NULL
    alignment_confidence  REAL,

    -- Enrichment (set by Reflector; may be NULL initially)
    importance      REAL,                    -- 0.0-1.0, NULL until Reflector fills
    entities        TEXT NOT NULL DEFAULT '[]',  -- JSON list
    reflected       INTEGER NOT NULL DEFAULT 0   -- has Reflector seen this?
);
CREATE INDEX idx_observations_kind        ON observations(kind);
CREATE INDEX idx_observations_user        ON observations(user_id);
CREATE INDEX idx_observations_session     ON observations(session_id);
CREATE INDEX idx_observations_reflected   ON observations(reflected);
CREATE INDEX idx_observations_importance   ON observations(importance);
```

### What changed from 0.4.0

| 0.4.0 | 0.5.0 | Reason |
|---|---|---|
| `observations`: 11 cols (id, content, source_quote, referenced_date, observation_ts, user_id, agent_id, session_id, alignment_tier, alignment_confidence) | `observations`: 15 cols (+`kind`, +`source_fact_ids`, +`importance`, +`entities`, +`reflected`) | Single-table design; enrichment co-located with identity |
| `observation_metadata`: 7 cols (id PK, observation_id, importance, entities, priority, confidence, enrichment_ts) | (table removed) | Collapsed; all columns now in `observations` |
| (no indexes on `kind`, `reflected`, or `importance`) | Indexes on `kind`, `user_id`, `session_id`, `reflected`, `importance` | Required for Delta 3 (count-based trigger) and retrieval filtering |

### Why collapse (not keep the split)

The 0.4.0 split was justified by three rationales, none of which hold
strongly enough to justify the cost in 0.5.0:

1. **"Recompute enrichment without touching observations"** — true of any
   split, but the Reflector now updates `importance` rarely (only when
   `NULL` or for periodic re-calibration). The cost of updating 1-2
   columns in a 15-column row is trivial.
2. **"Audit 'what was extracted' vs 'what the Reflector thinks'"** —
   the difference is auditable via `observation_ts` (when extracted) vs
   `importance` (Reflector's score, possibly set later). No second
   table needed.
3. **"Add other writers of enrichment"** — speculative; no other writers
   are planned. YAGNI.

The benefits of collapsing are real:
- Every retrieval query loses a JOIN (saves query planning + execution time)
- Simpler model: one Observation class, one table, no FK indirection
- Simpler migration from 0.4.0: no PK rebuild, no table copy, no dedup step
- The conceptual distinction between "extracted" and "enriched" is
  preserved in the model (immutable fields vs Reflector-managed fields)
  without needing separate tables

### `user_id` is a top-level field

`Memory` (the message type) carries `user_id` as a top-level field (not
nested in `metadata`). The Observer propagates this directly to
`observations.user_id` (no extraction, no LLM involvement). This is
already the 0.4.0 behavior; we preserve it explicitly in 0.5.0.

Why this matters for the per-store decision (see Delta 4): because
`user_id` is a top-level field propagated unchanged, and `MemoryStore`
is a per-user construct, the natural unit of work for the Reflector is
"all observations for one user" — exactly what a per-store worker
processes. A per-process worker would have to partition by user anyway,
which is what the per-store model gives us for free.

### Dropped fields (with justification)

- `observation_metadata.priority` — leftover from 0.3.0 (emoji-coded
  priority values that never worked). The 0.4.0 Reflector reads it
  only to format its prompt and ignores the value. No reader cares.
- `observation_metadata.confidence` — set to 1.0 on every insert, never
  read. Dead code.
- `observation_metadata.id` PK — redundant with `observation_id`.
  Removed as part of the table collapse.
- `observation_metadata.enrichment_ts` — was set on every insert, never
  read. Dead code (timestamped via `observation_ts` on the parent row).

## Architecture — the 6 deltas

The 0.4.0 architecture (Observer + 3-tier alignment + Reflector + ReflectorPipeline + MemoryStore on HybridDB) is the baseline. The 0.5.0 deltas are six small, additive changes:

### Delta 1 — Observer prompt: remove the dead-output escape hatch

The current Observer system prompt contains the line:
> "Do not invent facts. If nothing factual is present, return an empty observations array."

This is the source of the 40% dead-output. The model interprets "factual"
conservatively and returns `[]` for preference-style conversations. The 3-tier
alignment gate is already sufficient to catch fabrications downstream.

**Change:** rewrite the prompt to:
> "Extract user-attributable facts, preferences, plans, and context. If the
> conversation contains user statements of any kind, attempt to extract at
> least one observation. If you cannot find a verbatim sub-string for a fact,
> do not return that observation — the alignment gate handles grounding. Be
> liberal in what you consider worth recording."

The "no verbatim, no observation" rule is **preserved** (it was always a
separate line in the 0.4.0 prompt); only the "return empty if nothing
factual" line is removed. This prevents the model from inventing quotes
when forced to extract at least one, while removing the conservative
"factual" gating.

**Cost:** zero code change, one prompt edit. **Impact:** single largest recall unlock.

### Delta 2 — `importance` becomes Optional

The Observer currently scores `importance` per-fact, and the Reflector filters
by threshold. The 0.4.0 Q4 finding (Observer gave 0.30/0.40 to a discrete life
event) shows the per-fact importance signal is miscalibrated by ~0.2.

**Important schema note:** in 0.4.0, `importance` lives in the
`observation_metadata` table (separate from `observations`) with type `REAL`
and **no NOT NULL constraint**. The 0.4.0 Observer always provides a value
(via `item.get("importance", 0.5)`), so existing rows have `importance=0.5` or
the model-emitted value — never NULL. The schema change is therefore **not a
SQL migration**; it's a behavior change in what the Observer emits.

**Change:**
- Observer: stops providing an `importance` value. New observations have
  `importance=NULL` in `observation_metadata`.
- Reflector: assigns `importance` to all facts with calibrated anchors
  (0.7+ identity/life event, 0.4-0.6 preferences/plans, <0.4 throwaway).
- Existing 0.4.0 rows: **unchanged** — keep their existing non-null
  `importance` values. Reflector respects non-null values and only fills
  NULL rows.
- Retrieval: ranks NULL-importance facts by recency+relevance until Reflector
  fills.

**Cost:** ~30 lines (prompt, parse, store, Reflector logic update).

**Risks:** existing 0.4.0 callers that read `importance` will see `None` for
new rows. Document this in the migration guide. The 0.4.0 default `0.5` is
a "uncalibrated" sentinel; new code should treat `None` similarly.

### Delta 3 — Reflector trigger: time-based → count-based (with time fallback)

`ReflectorPipeline.maybe_run()` currently runs on a 24h timer with a
`min_observations=10` skip rule. The user wants count-based: "every X new
unreflected facts."

**Change:** add `trigger_every_n_observations: int = 50` parameter.
`maybe_run()` fires if EITHER:
  (a) unreflected fact count since last run ≥ N, OR
  (b) time since last run ≥ 24h (existing behavior, fallback for low-volume users)

Whichever condition is hit first triggers the run.

**Cost:** ~20 lines. Default X=50, configurable per-`MemoryStore`.

### Delta 4 — Explicit Reflector worker start/stop API

`ReflectorPipeline.maybe_run()` is invoked manually today. 0.5.0 wraps it in
an asyncio task the caller can start/stop:

```python
pipeline = ReflectorPipeline(
    store,
    interval_hours=24,
    trigger_every_n_observations=50,
)
await pipeline.start()   # spawns background asyncio task
# ... conversations happen ...
await pipeline.stop()    # graceful shutdown, awaits in-flight reflection
```

**Lifecycle semantics:**
- `start()` is idempotent — calling twice is a no-op.
- `stop()` is idempotent — calling twice is a no-op.
- `stop()` awaits any in-flight reflection before returning (cancels cleanly
  if already past the 30s grace period).
- If the worker task crashes (e.g., LLM error), it auto-restarts with
  exponential backoff (1s, 2s, 4s, ..., max 60s). Status is logged but does
  not raise.
- An `asyncio.Lock` ensures only one Reflector run is in flight at a time
  (handles the count+time tie-breaker case).

**Cost:** ~50 lines (start/stop methods, lifecycle, lock, restart-on-error).

**Backward compat:** existing manual `maybe_run()` callers can keep calling
it; the count-based trigger fires when `start()` is invoked, but `maybe_run()`
remains available for callers that prefer manual scheduling.

**Per-store vs per-process (resolved):** the Reflector worker in 0.5.0 is
**per-`MemoryStore`**. Each `MemoryStore` owns its own `ReflectorPipeline`
with its own `start()/stop()` lifecycle. This decision is reversible:
a future version can add a `ReflectorCoordinator` that aggregates N
`ReflectorPipeline` instances and runs them in a single task. The
per-store API is forward-compatible with this — a coordinator would
just call `maybe_run()` on each registered `ReflectorPipeline`. Per-store
is the right call for 0.5.0 because: (1) it fits CoreMem's per-user data
model — `MemoryStore` is per-user, and `user_id` is a top-level field on
`Memory` propagated unchanged to `observations.user_id`, so partitioning
work by `MemoryStore` IS partitioning by user; (2) per-user customization
of `trigger_every_n_observations` is free; (3) failure isolation is
critical for multi-tenant deployments; (4) the typical CoreMem
embedded-process use case has 1-10 users, where N idle asyncio tasks
are negligible. Per-process scheduling is deferred to 0.5.1+ for
high-scale deployments (e.g., 100s of users in one process).

### Delta 5 — Schema collapse to single table (see Schema restructure section above)

The schema collapse is the data-model half of the 0.5.0 redesign. See the
**Schema restructure** section for the final schema, the diff table, the
collapse rationale, and the dropped-fields justification.

**Summary of the SQL migration from 0.4.0:**

```sql
-- observations: add 5 new columns (additive, default values)
ALTER TABLE observations ADD COLUMN kind TEXT NOT NULL DEFAULT 'fact';
ALTER TABLE observations ADD COLUMN source_fact_ids TEXT NOT NULL DEFAULT '[]';
ALTER TABLE observations ADD COLUMN importance REAL;          -- nullable, NULL until Reflector fills
ALTER TABLE observations ADD COLUMN entities TEXT NOT NULL DEFAULT '[]';
ALTER TABLE observations ADD COLUMN reflected INTEGER NOT NULL DEFAULT 0;

-- Copy enrichment data from observation_metadata to observations
UPDATE observations
SET importance = (SELECT importance FROM observation_metadata
                  WHERE observation_metadata.observation_id = observations.id),
    entities = (SELECT entities FROM observation_metadata
                WHERE observation_metadata.observation_id = observations.id);

-- Drop the now-redundant observation_metadata table
DROP TABLE observation_metadata;

-- Create indexes (5 total)
CREATE INDEX idx_observations_kind        ON observations(kind);
CREATE INDEX idx_observations_user        ON observations(user_id);
CREATE INDEX idx_observations_session     ON observations(session_id);
CREATE INDEX idx_observations_reflected   ON observations(reflected);
CREATE INDEX idx_observations_importance   ON observations(importance);
```

**New `MemoryStore` helpers** (added in 0.5.0, not in 0.4.0):
- `get_pending_reflections() -> list[Observation]` — returns
  `kind='fact' AND reflected=0`
- `mark_reflected(observation_ids: list[str])` — sets `reflected=1` for
  the given IDs

**Cost:** ~60 lines total (schema migration script, model fields, helpers,
indexes, UPDATE statement for data copy). Smaller than the split version
because there's no PK rebuild and no table rename.

**Risk:** the `UPDATE ... SET ... FROM observation_metadata` statement
must be wrapped in a transaction. If it fails partway, the old schema
is restored. The order of operations is:
1. ADD COLUMN (additive, safe to fail)
2. UPDATE (atomic with the surrounding transaction)
3. DROP TABLE (destructive, but only after data is copied)

### Delta 6 — `HybridBackend` becomes default

Currently `HybridBackend` is opt-in (`pip install coremem[hybrid]`). 0.5.0
promotes it to default; ChromaBackend becomes legacy.

**Change:**
- `pyproject.toml`: `hybriddb` moves from optional `[hybrid]` extra to required
  `dependencies`
- `coremem/__init__.py`: default `backend=HybridBackend` (no fallback — if
  `hybriddb` is somehow not importable, raise `ImportError` with a clear
  message; in 0.5.0 this should never happen since it's a required dep)
- `README.md`: update install snippets, remove `coremem[hybrid]` references

**Deprecation timeline:**
- 0.5.0: `HybridBackend` is default; `ChromaBackend` works with
  `DeprecationWarning` if explicitly used
- 0.6.0 (next major): `ChromaBackend` is removed; `HybridBackend` is the only
  backend. The `coremem[hybrid]` extra and the deprecation warning are
  removed.

**Cost:** ~15 lines.

## Test plan

### Unit tests (extend existing)

```python
# tests/test_observer.py — add:
def test_observer_returns_minimum_one_per_substantive_turn():
    """Preference-style conversations must produce >=1 observation."""

def test_observer_importance_is_none():
    """0.5.0 Observer does not score importance (returns None)."""

# tests/test_reflector.py — add:
def test_reflector_count_based_trigger():
    """maybe_run() fires when unreflected count >= N, not just on timer."""

def test_reflector_hybrid_trigger():
    """Time-based fires when count < N but interval has elapsed."""

def test_reflector_start_stop_lifecycle():
    """start() spawns a task; stop() cancels it cleanly and idempotently."""

def test_reflector_reflected_flag_set():
    """After reflect(), all source facts are marked reflected=1."""

# tests/test_memory_store.py — add:
def test_observation_kind_field():
def test_observation_source_fact_ids_field():
def test_observation_importance_field():
def test_observation_entities_field():
def test_observation_reflected_field():
def test_observation_metadata_table_removed():
    """observation_metadata table no longer exists in 0.5.0."""
def test_mark_reflected_helper():
def test_get_pending_reflections_helper():
def test_priority_field_dropped():  # schema migration
def test_confidence_field_dropped():  # schema migration
```

### Migration tests

```python
# tests/test_migration_0_4_to_0_5.py — new file
def test_schema_migration_preserves_existing_observations():
    """Run 0.4.0 schema, insert data, run 0.5.0 migration, verify all
    rows present with correct defaults (kind='fact', reflected=0)."""

def test_schema_migration_preserves_user_data():
    """Every observation id, content, source_quote, importance value
    from 0.4.0 is preserved in 0.5.0. Destructive changes only affect
    priority, confidence, enrichment_ts, and the observation_metadata table."""

def test_schema_migration_copies_importance_to_observations():
    """observation_metadata.importance values are copied to
    observations.importance during migration."""

def test_schema_migration_copies_entities_to_observations():
    """observation_metadata.entities values are copied to
    observations.entities during migration."""

def test_schema_migration_drops_observation_metadata_table():
    """observation_metadata table no longer exists post-migration."""

def test_schema_migration_drops_priority_column():
    """priority column no longer exists post-migration (in dropped table)."""

def test_schema_migration_drops_confidence_column():
    """confidence column no longer exists post-migration (in dropped table)."""

def test_schema_migration_is_idempotent():
    """Running the migration twice does not error or duplicate data."""

def test_schema_migration_creates_indexes():
    """All 5 indexes exist post-migration (kind, user, session, reflected, importance)."""
```

### Integration test

Re-run `benchmarks/longmemeval/observer_eval.py` against the 0.5.0 Observer.
Save results to `results/eval/observer_deepseek_10q_v0.5.0.json`. Compare
against `observer_deepseek_10q.json` baseline.

**Pre-flight check:** verify the eval script reads the new `kind` and
`reflected` fields gracefully. The script currently uses `ObserverPipeline`
and `ReflectorPipeline` directly; it does not introspect on observation
schema, so the new fields should be transparent. If the eval script DOES
break (e.g., it tries to sum `importance` and crashes on `None`), update it
to treat `None` as 0.5 (uncalibrated).

**Pass criteria (all four must hold):**
- obs/q ≥ 5.0 (was 2.0)
- dead output < 10% (was 40%)
- answer coverage ≥ 8/10 (was 5/10)
- hallucination rate < 5% (was 0%)

If any criterion fails, the release is blocked.

### Regression

All 0.4.0 tests must still pass with the new schema. Existing rows treated as
`kind="fact", reflected=False, importance=None`.

## Migration guide (0.4.0 → 0.5.0)

```python
# Was (0.4.0):
pipeline = ReflectorPipeline(store, interval_hours=24, min_observations=10)
# manually call maybe_run() from a cron or background loop

# Now (0.5.0) — recommended: use the explicit lifecycle
pipeline = ReflectorPipeline(
    store,
    interval_hours=24,                          # still works
    trigger_every_n_observations=50,            # NEW: count-based trigger
)
await pipeline.start()                         # NEW: explicit lifecycle
# ... work happens ...
await pipeline.stop()                          # NEW: graceful shutdown

# Now (0.5.0) — alternative: keep manual scheduling (backward compatible)
pipeline = ReflectorPipeline(
    store,
    interval_hours=24,
    trigger_every_n_observations=50,            # fires when you call maybe_run()
)
# existing code that calls await pipeline.maybe_run() still works
# the count-based trigger is unused in this path

# Observation schema (0.5.0):
#   observations: 15 columns (single table)
#     id              TEXT PRIMARY KEY
#     kind            TEXT NOT NULL DEFAULT 'fact'    # NEW
#     content         TEXT NOT NULL
#     source_quote    TEXT                             # NULL for reflections
#     source_fact_ids TEXT NOT NULL DEFAULT '[]'      # NEW (JSON)
#     referenced_date TEXT
#     observation_ts  TEXT NOT NULL
#     user_id, agent_id, session_id TEXT
#     alignment_tier, alignment_confidence (3-tier gate output)
#     importance      REAL                             # NEW (copied from observation_metadata)
#     entities        TEXT NOT NULL DEFAULT '[]'       # NEW (copied from observation_metadata)
#     reflected       INTEGER NOT NULL DEFAULT 0      # NEW
#   # REMOVED: observation_metadata table (collapsed into observations)
#   # DROPPED from metadata: priority, confidence, id, enrichment_ts (all dead code)
```

**SQL migration** (run on first 0.5.0 startup if migrating from 0.4.0):
```sql
ALTER TABLE observations ADD COLUMN kind TEXT NOT NULL DEFAULT 'fact';
ALTER TABLE observations ADD COLUMN source_fact_ids TEXT NOT NULL DEFAULT '[]';
ALTER TABLE observations ADD COLUMN importance REAL;
ALTER TABLE observations ADD COLUMN entities TEXT NOT NULL DEFAULT '[]';
ALTER TABLE observations ADD COLUMN reflected INTEGER NOT NULL DEFAULT 0;
UPDATE observations
SET importance = (SELECT importance FROM observation_metadata
                  WHERE observation_metadata.observation_id = observations.id),
    entities = (SELECT entities FROM observation_metadata
                WHERE observation_metadata.observation_id = observations.id);
DROP TABLE observation_metadata;
CREATE INDEX idx_observations_kind        ON observations(kind);
CREATE INDEX idx_observations_user        ON observations(user_id);
CREATE INDEX idx_observations_session     ON observations(session_id);
CREATE INDEX idx_observations_reflected   ON observations(reflected);
CREATE INDEX idx_observations_importance   ON observations(importance);
```

**Recommended migration path:**
1. Existing `maybe_run()` callers: no immediate change. Migrate to
   `start()/stop()` at your convenience; the count-based trigger only
   activates with `start()`.
2. Existing importance readers: handle `None` defensively (`x or 0.5`).
3. New 0.5.0 installs: use the `start()/stop()` lifecycle.

## Rollout plan

1. Branch: `0.5.0-observer-revision` off `main` (15 unpushed 0.4.0 commits)
2. Implement the 6 deltas in worktree (~1-2 days)
3. Run the 10-question eval, verify all 4 pass criteria
4. Bump version to 0.5.0, update CHANGELOG, push commits, tag `v0.5.0`
5. Backwards compat for user data: schema migration copies all user-data
   values from `observation_metadata` to `observations` before dropping
   the metadata table. Existing 0.4.0 observations retain their `id`,
   `content`, `source_quote`, `referenced_date`, `observation_ts`,
   `user_id`, `agent_id`, `session_id`, `alignment_tier`,
   `alignment_confidence`, `importance`, and `entities` values. The
   `observation_metadata` table is dropped post-copy. Destructive changes
   only affect documented dead fields (`priority`, `confidence`,
   `enrichment_ts`, the redundant `observation_metadata.id` PK).

**Migration auto-run:** `MemoryStore.__init__()` checks the schema version
(stored in a new `_schema_version` table) and auto-runs the migration
script if the version is < 0.5.0. The migration is idempotent and wrapped
in a transaction; on failure, the old schema is restored. Manual migration
is also possible via `await store.migrate_to_0_5_0()`.

## Cost analysis

Per typical user/day, deepseek-v3.2:cloud rates:
- Observer: 1-10 calls/day, ~$0.002/call → $0.002-$0.020
- Reflector: 1-3 calls/day, ~$0.01/call → $0.010-$0.030
- Alignment gate: deterministic, $0
- **Total: $0.012-$0.050/user/day** — fits the EA cost envelope.

## Open questions

1. The 0.4.0 Reflector dedup uses cosine similarity — should we also add
   string-similarity (already in Observer) for redundancy? Defer to 0.5.1 if
   eval shows redundancy issues.

## References

- Generative Agents reflection (Park et al. 2023): `arxiv.org/abs/2304.03442`
- A-Mem (NeurIPS 2025): `arxiv.org/abs/2502.12110`
- LangExtract resolver (Google): `github.com/google/langextract`
- LangMem hot-path + background: `github.com/langchain-ai/langmem`
- MemReader (2025): active vs passive extraction
- APEX-MEM (2025, SOTA on LongMemEval): property graph + retrieval agent
- 10-question comparison: `docs/longmemeval-0.4.0-comparison.md`
