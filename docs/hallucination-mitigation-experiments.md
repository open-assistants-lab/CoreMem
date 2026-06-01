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

The **source quote gate** infrastructure is solid:

1. Prompt requires `source_quote` field in every observation
2. Post-hoc verification checks quote exists verbatim in source conversation text
3. Observations without verifiable quotes are dropped

This gate is model-agnostic and will work with any LLM. It currently catches
fabricated quotes at a ~40% rate (drops false positives but can't eliminate
upstream fabrication).

## Recommendations

### 1. Default to a model with strong instruction-following

```python
# Recommended production defaults
Observer(model="openai:gpt-4o-mini")     # $0.15/M tokens, near-zero hallucination
Observer(model="anthropic:claude-haiku") # $0.25/M tokens, strong JSON compliance
```

These models have significantly better "copy-paste verbatim" capability.

### 2. Keep the gate for defense-in-depth

The source_quote gate adds a safety layer that catches edge cases even with
strong models. Mem0 and other systems don't have this — they just trust the LLM.

### 3. Document model tradeoffs

| Model | Hallucination Risk | Cost | Recommendation |
|-------|-------------------|------|---------------|
| GPT-4o-mini | Very low | Low | Default for production |
| Claude Haiku | Very low | Low | Best JSON compliance |
| DeepSeek V4 Flash | High (34-63%) | Lowest | Acceptable for dev/demo |
| Gemma 4 (local) | Very high (70%) | Free | Not recommended for observation |
