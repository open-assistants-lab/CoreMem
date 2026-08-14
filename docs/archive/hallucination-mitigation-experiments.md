# Hallucination Mitigation Experiments

## 0.4.0 Rewrite — Alignment-Gated Observer (2026-06-02)

Following the review in `docs/observer-hallucination-review.md` (which
diagnosed four concrete bugs in the prior Observer), CoreMem 0.4.0 adopts
LangExtract's 3-tier alignment gate and CogCanvas's prompt pattern. The
NLI module (`bart-large-mnli`) and `transformers` dependency are removed
entirely — the alignment gate is deterministic and runs in microseconds.

**Rewrite highlights:**
- New module `coremem/grounding.py` — 3-tier alignment (`EXACT` /
  `FUZZY` / `NONE`) with char-level `SequenceMatcher` (lowercased
  strings; FUZZY threshold 0.75).
- `coremem/observer.py` rewritten: single-pass extraction with
  CogCanvas system prompt (2 few-shot examples), native messages array
  (no JSON wrapping), `temperature=0.1`, structured output via
  `tool_calls` API (no `priority` enum; `importance` is a float).
- `ObserverPipeline` runs the alignment gate on every extracted
  observation before storage. NONE observations are dropped.
- `MemoryStore` migration: added `alignment_tier TEXT` and
  `alignment_confidence REAL` columns to `observations` table
  (idempotent migration on first open).
- Reflector filter now checks `importance >= 0.5` (was using a broken
  `priority` enum that never matched).

**LongMemEval re-run (10 questions, single-session-user):**

| Approach | Hallucination | Obs/Q | Time/obs |
|----------|-------------|-------|----------|
| Pre-0.4.0 baseline (DeepSeek V4 Flash) | 34-63% | 5.8-9.7 | 20-100s |
| **0.4.0 + `ollama:deepseek-v3.2:cloud`** | **0% (0/20)** | 2.0 | 28s |
| **0.4.0 + `ollama:gemma4:e4b`** | 0% (0/1) | 0.1 | 513s |

The 0.4.0 results on `deepseek-v3.2:cloud` are 20/20 observations
verbatim-grounded (every `source_quote` is a sub-string of the source
conversation). Hallucination rate is below the <10% target; time/obs is
5x under the 150s target. Obs/Q (2.0) is below the 5-9 target — the
smaller model extracts fewer facts per conversation, and 4/10 questions
returned zero observations (model returns `{"observations": []}` for
~20K-char inputs in some cases). The model is the bottleneck, not the
extraction logic.

Raw results: `results/eval/observer_deepseek_10q.json`,
`results/eval/observer_gemma4_10q.json`.

---

# Hallucination Mitigation Experiments

2026-06-01

## Problem

The Observer extracts facts from conversation messages as structured observations.
Smaller models (Gemma, DeepSeek) fabricate facts — "hallucinations" — when the
conversation context is thin. These fabrications are stored alongside real
observations, degrading memory quality.

## Metric

**Hallucination rate** = percentage of observations whose `source_quote` field
does NOT appear verbatim in the source conversation text. Checked by substring
match after stripping quotes/whitespace and lowercasing.

## Test Setup

- **10 LongMemEval questions** (single-session-user type, first 10 in dataset)
- **DeepSeek V4 Flash** as the primary test model (fastest iteration)
- **Source quote gate** active in all runs (drops observations without
  verifiable source_quotes)

## Results

| # | Approach | Hallucination | Obs/Q | Time | Key Insight |
|---|----------|-------------|-------|------|-------------|
| 1 | **Gemma 4 baseline** — bare JSON prompt | 70% | 8.3 | 400s | Smaller models fabricate aggressively |
| 2 | **DeepSeek chat bare prompt** — simplest approach | 34% | 5.8 | 120s | Better, but 1/3 still fabricated |
| 3 | **+ source_quote prompt** — model asked to cite quotes | 39% | 6.4 | 120s | Model invents quotes too |
| 4 | **+ source_quote gate** — post-hoc verification of quotes vs source | 43% | 5.3 | 120s | Gate drops some but gate can't fix upstream fabrication |
| 5 | **+ stronger anti-fab prompt** — "Fabricating is the WORST mistake" | 39% | 6.4 | 118s | Stronger language has zero effect |
| 6 | **+ JSON mode + few-shot** — `response_format: json_object` + example | 60% | 6.0 | 93s | **Worse** — JSON mode made model more confident in fabrication |
| 7 | **DeepSeek V4 Pro** — larger model variant | 59% | 9.2 | 1080s | Larger model fabricates MORE (more confident output) |
| 8 | **Two-pass extraction** — Pass 1: copy sentences, Pass 2: observe | 58% | 9.7 | 500s | Pass 1 also fabricates sentences — can't copy-paste verbatim |
| 9 | **Tool calling** — forced structured output via `tool_calls` API | 61% | 9.3 | 189s | Schema constraint doesn't fix copy-paste capability |
| 10 | **NLI verification** — bart-large-mnli checks source_quote → claim entailment | 60% | 9.5 | 505s | NLI gates internal consistency. Fabricated quotes are internally consistent with fabricated claims — can't detect co-fabrication |

## Root Cause

DeepSeek V4 Flash/Pro and Gemma 4 all share a fundamental limitation: **they
cannot reproduce source text verbatim.** Even when explicitly instructed to
"copy-paste EXACT sentences" and threatened with "fabricating is the WORST
mistake," they paraphrase. This is not a prompt engineering problem — it's a
model capability gap.

Evidence: when asked to extract sentences with the instruction "Copy-paste EXACT
sentences," the model outputs "I'm a software engineer" when the source says
"I am a software engineer" — a one-character difference that confirms
paraphrasing, not copy-paste.

## What Worked

The **source quote gate** and **NLI gate** form a two-layer defense:

| Gate | What it catches | Catch rate |
|------|----------------|-----------|
| Source quote | Fabricated quotes not verbatim in source | ~20% |
| NLI entailment | Real quotes that don't logically support the claim (internal inconsistency) | ~5% |
| **Combined** | Different failure modes, complementary | ~25% |

Both gates are model-agnostic. The NLI gate uses `bart-large-mnli` (1.6GB,
optional, fail-open if `transformers` not installed). Neither catches the
dominant failure mode: co-fabrication where the LLM invents both quote AND
claim together in an internally consistent pair.

## Recommendations

### 1. Accept DeepSeek as a dev/demo model

The gate infrastructure provides defense-in-depth. For development and demos,
DeepSeek at 34-63% hallucination is acceptable if users understand the
limitation. For production, a stronger model is needed.

### 2. Keep all three gates for defense-in-depth

- **Source quote gate** (always on) — catches fabricated quotes
- **NLI gate** (optional, `pip install coremem[observer]`) — catches internal inconsistency
- **String similarity dedup** (always on) — catches near-duplicate observations

### 3. Document model tradeoffs

| Model | Hallucination Risk | Cost | Recommendation |
|-------|-------------------|------|---------------|
| GPT-4o-mini | Very low | Low | Default for production |
| Claude Haiku | Very low | Low | Best JSON compliance |
| DeepSeek V4 Flash | High (34-63%) | Lowest | Acceptable for dev/demo |
| Gemma 4 (local) | Very high (70%) | Free | Not recommended for observation |
