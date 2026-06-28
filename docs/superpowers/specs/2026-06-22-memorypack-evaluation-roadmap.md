# MemoryPack Evaluation Roadmap

2026-06-22

## 1. Goal

Define a staged evaluation plan for proving or falsifying the MemoryPack
hypothesis without accidentally measuring the wrong thing.

The core hypothesis is:

```text
reference turns
  -> compiled MemoryPack pages
  -> MemoryPack-first retrieval
```

should beat raw message/session retrieval on long-horizon memory tasks by
improving retrieval compactness, citation quality, human inspectability, and
eventual answer quality.

## 2. Current Implementation Boundary

The current POC implementation includes the deterministic substrate:

- MemoryPack bundle initialization.
- Immutable reference turn writer.
- Canonical `json agent_memory-turn` payloads.
- Reference manifest hashes.
- Exact quote validation.
- Derived-summary claim validation.
- Deterministic lint for reference turns and memory pages.
- Simple markdown search over `pages/`.

The current POC does not yet include:

- LLM compiler from reference turns to MemoryPack pages.
- Observation-backed compiler.
- Answer generator.
- LongMemEval evaluator wrapper.
- Retrieval logging compatible with LongMemEval metrics.
- Temporal or knowledge-update compiler prompts.

Therefore, the next evaluation should not jump directly to full LongMemEval QA.

## 3. Evaluation Principles

### Isolate The Variable

Each evaluation stage should test one thing:

- Substrate correctness.
- Compiler quality.
- Retrieval quality.
- Reader/answer quality.

Do not combine all four too early. If a combined run fails, we cannot tell
whether the problem is page format, compiler prompt, retrieval, or answer model.

### Prevent Answer Leakage

Ground-truth evaluation fields must never be used during memory construction or
retrieval.

Forbidden during ingestion/compilation/search:

- `answer`
- `answer_session_ids`
- `has_answer`
- evaluator labels
- oracle evidence-only filtering unless explicitly running an oracle sanity test

Allowed only after retrieval/answering:

- session-level recall scoring against `answer_session_ids`
- turn-level recall scoring against `has_answer`
- QA scoring against `answer`

### Prefer Retrieval Before QA

LongMemEval full QA depends heavily on the reader model. MemoryPack should first
prove it can retrieve the right memory with less context. Full QA comes after
retrieval quality is credible.

## 4. LongMemEval Mapping

LongMemEval fields map to MemoryPack as follows:

| LongMemEval field | MemoryPack usage |
| --- | --- |
| `question_id` | evaluation instance id |
| `question_type` | stratification bucket |
| `question` | retrieval query and later QA prompt |
| `question_date` | temporal query context |
| `haystack_session_ids` | session ids for reference turns |
| `haystack_dates` | timestamps for reference turns |
| `haystack_sessions` | source messages for `references/turns/*.md` |
| `answer` | QA scoring only |
| `answer_session_ids` | retrieval scoring only |
| `has_answer` | turn-level recall scoring only |

LongMemEval abilities to track:

- `single-session-user`: information extraction from user statements.
- `single-session-assistant`: information extraction from assistant statements.
- `single-session-preference`: preference/persona memory.
- `multi-session`: synthesis across sessions.
- `temporal-reasoning`: timestamp and sequence sensitivity.
- `knowledge-update`: supersession and changed facts.
- `*_abs`: abstention when memory has no answer.

## 5. Stage 1: Internal Scripted Eval

### Purpose

Validate the MemoryPack substrate on small, controlled data before using a large
external benchmark. Compiler behavior is added to this stage only after a
compiler exists.

### Dataset

Create 20-50 scripted turns covering:

- User preferences.
- Project decisions.
- Tool-observed facts.
- Changed/superseded claims.
- Temporal facts.
- Open questions.
- Prompt-injection-looking text inside user/tool content.
- Abstention questions where no memory supports an answer.

### Modes

```text
A. raw reference-turn search
B. manually compiled MemoryPack pages
C. LLM-compiled MemoryPack pages, once compiler exists
```

### Metrics

- exact quote citation validity
- retrieval hit rate for expected page ids
- retrieval context character/token count
- context delta versus raw top-message retrieval, which may be negative on tiny
  fixtures
- stale claim count
- abstention correctness
- lint pass/fail
- human inspectability of `MEMORY.md` and `index.md`

### Success Criteria

Before LongMemEval loader/raw-baseline work starts:

- deterministic lint passes
- exact quote validation catches bad citations
- manual MemoryPack pages retrieve expected facts
- query context size is reported honestly; smaller-than-raw is not required for
  tiny fixtures where raw top-message retrieval can be extremely compact
- no system/developer prompt text leaks into memory pages

## 6. Stage 2: LongMemEval Loader And Raw Baseline Harness

### Purpose

Validate LongMemEval ingestion, ground-truth stripping, reference-turn writing,
and raw retrieval baselines before involving the MemoryPack compiler or a reader
LLM.

This stage does not prove MemoryPack compiled retrieval yet. The current POC has
no automatic compiler, so compiled `pages/` would be empty unless pages are
handwritten or generated externally. Stage 2 proves the benchmark adapter and
baseline retrieval plumbing.

### Dataset Order

Run in this order:

1. A tiny local fixture with 2-5 LongMemEval-shaped examples.
2. `longmemeval_oracle.json` for loader/scorer sanity checks only. Do not
   report oracle-file results as the primary benchmark.
3. `longmemeval_s_cleaned.json` for the first real retrieval benchmark.
4. `longmemeval_m_cleaned.json` only after `s` is stable.

### Input

For each instance:

```text
haystack_session_ids[i]
haystack_dates[i]
haystack_sessions[i]
```

are written into MemoryPack reference turns.

Reference construction rules:

- Write one reference file per LongMemEval session.
- Remap `haystack_session_ids[i]` to deterministic public ids before writing
  references. Do not expose original session ids to downstream model-facing
  artifacts; some datasets use answer-like names in session ids.
- Use `haystack_dates[i]` as the timestamp for all turns in that session when
  finer timestamps are unavailable.
- Assign deterministic message ids such as
  `<public_session_id>_turn_<turn_index>_<role>`.
- Persist only `role` and `content` into MemoryPack references.
- Strip `has_answer` before ingestion and keep it only in a separate scoring
  structure.
- Never write `answer` or `answer_session_ids` into the MemoryPack bundle.
- Treat `question_id` values ending in `_abs` as abstention examples even when
  `question_type` does not include `_abs`.

### Output JSONL

Save one JSON object per question:

```json
{
  "question_id": "example_001",
  "question_type": "multi-session",
  "query": "What did I decide about the memory architecture?",
  "mode": "raw-reference-retrieval",
  "retrieved_page_ids": [],
  "retrieved_session_ids": ["session_001"],
  "retrieved_message_ids": ["session_001_turn_3_user"],
  "context_chars": 4200,
  "used_reference_search": true
}
```

### Metrics

- session recall@k against `answer_session_ids`
- turn/message recall@k against stripped `has_answer` ground truth
- hit@k as a separate boolean "any expected evidence retrieved" metric
- mean reciprocal rank for evidence sessions
- context character/token count
- empty retrieval rate
- abstention false-positive rate for `*_abs`
- breakdown by `question_type`

### Anti-Leak Rule

The harness may load `answer_session_ids` and `has_answer` only inside the
scoring function. The ingestion and retrieval code paths should receive an
instance with those fields removed.

### Success Criteria

Before compiler-based LongMemEval retrieval starts:

- the loader writes valid MemoryPack references for a local fixture
- no ground-truth fields appear in `references/`, `pages/`, `MEMORY.md`,
  `index.md`, or `log.md`
- raw-reference retrieval emits stable `retrieved_session_ids` and
  `retrieved_message_ids`
- public retrieval logs do not include private scoring ground truth
- scoring computes session recall, turn/message recall, context size, and
  question-type breakdown

## 7. Stage 3: MemoryPack Compiler Retrieval Eval

### Purpose

Evaluate MemoryPack retrieval after the LLM compiler converts reference turns
into MemoryPack pages.

Pipeline:

```text
haystack_sessions
  -> references/turns/*.md
  -> LLM compiler
  -> pages/*.md
  -> MemoryPack search
```

### Compiler Requirements Before This Stage

- Writes only `pages/`, `MEMORY.md`, `index.md`, and `log.md`.
- Does not mutate `references/`.
- Emits structured update plans.
- Supports atomic and `derived_summary` claims.
- Runs deterministic lint after patching.
- Rejects unsupported citations.

### Metrics

- all Stage 2 retrieval metrics
- citation validity rate
- unsupported claim rejection count
- page count per instance
- average page length
- `MEMORY.md` boot size
- stale/superseded claim handling for `knowledge-update`
- temporal fact handling for `temporal-reasoning`

### Success Criteria

- Retrieval is competitive with or better than raw session search.
- Context size is materially lower than raw session search.
- Citation validity remains high.
- `knowledge-update` does not accumulate active stale claims.
- `*_abs` questions do not produce unsupported confident answers or retrievals.

## 8. Stage 4: Observation-Backed Hybrid Eval

### Purpose

Test the original hybrid theory:

```text
reference turns
  -> grounded observations
  -> MemoryPack pages
  -> MemoryPack search
```

### Modes

Compare:

```text
A. raw reference-turn search
B. direct turn-to-MemoryPack compiler
C. observation-to-MemoryPack compiler
```

### Expected Win Condition

Observation-backed MemoryPack should improve:

- citation precision
- stale claim handling
- conflict/supersession quality
- page patch stability
- unsupported claim rejection

It may cost more ingest latency. That is acceptable if retrieval quality and
trustworthiness improve.

## 9. Stage 5: Full LongMemEval QA

### Purpose

Measure end-user answer quality after retrieval is already credible.

Pipeline:

```text
LongMemEval history
  -> MemoryPack memory
  -> retrieve compact context
  -> answer generator
  -> LongMemEval evaluator
```

### Output JSONL

LongMemEval expects:

```json
{"question_id": "example_001", "hypothesis": "The answer generated by the system."}
```

The harness should also keep a richer internal log:

```json
{
  "question_id": "example_001",
  "question_type": "knowledge-update",
  "hypothesis": "...",
  "retrieved_page_ids": ["..."],
  "retrieved_session_ids": ["..."],
  "context_chars": 4200,
  "mode": "observation-backed-memorypack"
}
```

### Metrics

- LongMemEval QA score
- retrieval recall@k
- context size
- answer abstention correctness
- answer citation support rate, if citations are included
- cost and latency

## 10. Implementation Plan

### Milestone A: Internal Eval Harness

Add:

```text
scripts/eval_memorypack_internal.py
tests/test_memorypack_eval_internal.py
```

Responsibilities:

- load a tiny scripted fixture
- create MemoryPack references
- use handwritten pages or no pages depending on mode
- run `AgentMemorySearch`
- report retrieval/context/lint metrics

### Milestone B: LongMemEval Loader And Raw Baseline

Add:

```text
scripts/eval_memorypack_longmemeval.py
```

Responsibilities:

- read LongMemEval JSON
- strip ground-truth fields before ingestion/retrieval
- write MemoryPack reference turns
- run raw-reference retrieval baseline
- emit retrieval-only JSONL with session/message ids
- score retrieval after the run

### Milestone C: Compiler Adapter

Add compiler integration once the MemoryPack compiler exists.

Responsibilities:

- call compiler per instance or per session window
- run lint after compilation
- save compiler logs for failed citations and rejected claims

### Milestone D: Full QA Adapter

Add answer generation and LongMemEval evaluator compatibility after retrieval
quality is acceptable.

## 11. Current Verdict

Use LongMemEval, but do not start with full QA.

Correct order:

1. Internal scripted eval.
2. LongMemEval loader and raw retrieval baseline.
3. Direct MemoryPack compiler retrieval eval.
4. Observation-backed hybrid eval.
5. Full LongMemEval QA.

This sequence proves whether MemoryPack improves memory retrieval before testing
whether a reader model can answer from the retrieved context.
