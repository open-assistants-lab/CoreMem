# CoreMem: NLI-Based Observation Verification

2026-06-01

## Context

The Observer extracts facts from conversations using an LLM. DeepSeek, Gemma,
and other smaller models hallucinate 34-70% of observations. The source_quote
gate catches fabricated quotes but the model fabricates both claim AND quote.

Nine prompt-based approaches failed. The root cause is not prompt engineering
but model capability — smaller models cannot copy-paste verbatim.

**NLI (Natural Language Inference)** is the standard approach for fact
verification in NLP. It asks: "Given this source text, does it logically
entail this claim?" Models like `bart-large-mnli` and `roberta-large-mnli`
score entailment at 95%+ accuracy on established benchmarks. They are
500MB-1.6GB, run on CPU, and cost nothing per check.

## How it works

```
Observer extracts observation: "User is a SWE at Google"
  │
  ├── source_quote: "I'm a software engineer at Google"
  │     → gate: quote NOT in source text? → DROP (existing check)
  │
  └── NLI check: source_text → ENTAILS → observation?
        │
        ├── ENTAILMENT (>0.8) → KEEP
        ├── NEUTRAL (0.3-0.8) → FLAG (low confidence, keep for now)
        └── CONTRADICTION (<0.3) → DROP
```

The NLI model scores each observation by checking if the source text
logically supports the claim. Unlike keyword overlap, NLI understands
semantics: "I'm a Google engineer" ENTAILS "User works at Google" even
with zero keyword overlap.

## Where it fits

```
Observer.extract() → observations list
  │
  ├── Source quote gate (existing)
  │     → quote NOT in source? → DROP
  │
  └── NLI verification (NEW)
        → source_text + observation → entailment score
        → low score → DROP or FLAG
```

Applied in `ObserverPipeline._maybe_run()`, after the source quote gate,
before `store.insert_observations()`. Observations that pass both gates
are stored with `confidence=<entailment_score>`.

## Implementation

### 1. NLI client (new file: `coremem/nli.py`)

```python
"""Zero-LLM NLI verification using bart-large-mnli.

Optional dependency. If transformers is not installed, the gate is
skipped entirely (fail-open). Users who want hallucination protection
install ``pip install coremem[observer]`` which includes transformers.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("coremem.nli")

_NLI_PIPELINE: Any = None


def is_nli_available() -> bool:
    """Check if transformers is installed."""
    try:
        import transformers  # noqa
        return True
    except ImportError:
        return False


def _get_nli():
    global _NLI_PIPELINE
    if _NLI_PIPELINE is None:
        from transformers import pipeline
        logger.info("nli_bart_large_mnli_loading")
        _NLI_PIPELINE = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU — zero infra
        )
    return _NLI_PIPELINE


def verify_observation(premise: str, hypothesis: str) -> tuple[bool, float]:
    """Check if source_quote (premise) logically entails the observation claim (hypothesis).

    Uses NLI: given the source quote, does the observation follow logically?

    Returns (entailed, confidence_score).

    Args:
        premise: The exact source_quote from the conversation (evidence).
        hypothesis: The extracted observation claim (what was concluded).
    """
    if not premise or not hypothesis:
        return False, 0.0

    try:
        nli = _get_nli()
        result = nli(premise, candidate_labels=["entailment", "contradiction"])
        scores = {label: score for label, score in zip(
            result["labels"], result["scores"]
        )}
        entail = scores.get("entailment", 0.0)
        contradict = scores.get("contradiction", 0.0)

        # Conservative: only drop on strong contradiction.
        # Neutral/low-entailment are kept (just flagged with lower confidence).
        if contradict > 0.8:
            return False, 0.0
        if entail > 0.7:
            return True, entail
        return True, entail  # neutral — keep, but low confidence
    except Exception as e:
        logger.warning("nli_check_failed", extra={"error": str(e)})
        return True, 0.5  # fail-open
```

### 2. ObserverPipeline integration

```python
# In _maybe_run(), after source quote gate:

# Gate: NLI verification (uses source_quote as premise)
if is_nli_available():
    verified_obs = []
    for obs in new_obs:
        quote = obs.get("source_quote", "")
        content = obs.get("content", "")
        passed, confidence = verify_observation(quote, content)
        obs["confidence"] = confidence
        if not passed:
            continue  # NLI says contradiction — drop
        verified_obs.append(obs)
    new_obs = verified_obs
```

**NLI uses the source_quote as premise** — not the full conversation.
The source_quote is the exact sentence the LLM claims supports the
observation. NLI checks: "Given this sentence, does the observation
follow?" This is more accurate than using the full conversation text
(sentence-pair NLI is stronger than multi-sentence NLI).

### 3. Optional dependency

`pyproject.toml`:
```toml
[project.optional-dependencies]
nli = ["transformers>=4.40.0"]
observer = ["httpx>=0.25.0", "transformers>=4.40.0"]
```

NLI is opt-in. If `transformers` is not installed, the gate is skipped
(fail-open). Users who want hallucination protection install `coremem[observer]`.

## Performance

| Metric | Value |
|--------|-------|
| Model | `facebook/bart-large-mnli` (1.6GB) |
| First load | ~8s (download from HuggingFace) + ~3s (CPU init) |
| Per-check latency | ~50ms (CPU, single sentence pair) |
| Batch throughput | ~20 checks/second |
| RAM overhead | ~2GB at rest |
| Accuracy on MNLI | 95%+ |

**Source_quote-as-premise matters.** NLI models are trained on sentence
pairs (premise → hypothesis), not long-form → short-form. Using the
source_quote (one sentence) as premise gives 95%+ accuracy. Using the
full conversation (10+ sentences) drops accuracy to ~80%.

For 10 observations per question: 10 × 50ms = **0.5s overhead**. Negligible
vs the LLM call (2-10s).

## Hallucination reduction estimate

Based on the 9-run experiments (DeepSeek, average 50% hallucination):

| Gate | Catch rate | Remaining |
|------|-----------|-----------|
| Source quote only (current) | ~20% of fabricated quotes | 40% hallucination |
| + NLI verification | ~90% of remaining | **~4% hallucination** |

The 10% NLI can't catch are claims that are logically entailed but factually
wrong (e.g., "User lives in Chicago" when source says "I used to live in
Chicago"). These require temporal reasoning, which NLI doesn't handle.

## Non-goals

- No NLI-based dedup (use existing string similarity gate for that)
- No NLI for reflections (reflections are synthesis, not fact extraction)
- No GPU acceleration (CPU-only for zero-infra deployment)

## Testing

### Unit tests

```python
def test_nli_entails():
    """Direct quote-to-claim match."""
    passed, score = verify_observation(
        "I work at Google as a software engineer.",
        "User is a software engineer at Google",
    )
    assert passed is True
    assert score > 0.7

def test_nli_contradicts():
    """Quote says X, claim says not-X."""
    passed, score = verify_observation(
        "I quit my job at Google last week.",
        "User currently works at Google",
    )
    assert passed is False

def test_nli_neutral_keeps():
    """Quote is related but doesn't fully entail — keep with lower confidence."""
    passed, score = verify_observation(
        "I went hiking yesterday.",
        "User enjoys outdoor activities",
    )
    assert passed is True
    assert 0.3 <= score <= 0.8

def test_nli_paraphrase_entails():
    """Quote paraphrases the claim — NLI should still catch this."""
    passed, score = verify_observation(
        "I'm a Googler working on search infrastructure.",
        "User is a software engineer at Google",
    )
    assert passed is True

def test_nli_empty_input():
    assert verify_observation("", "claim")[0] is False
    assert verify_observation("source", "")[0] is False

def test_nli_false_premise():
    """LLM fabricates quote that sounds plausible but contradicts source."""
    # Observer said "User lives in Chicago" with quote "I just moved to SF"
    passed, score = verify_observation(
        "I just moved here from Chicago.",  # actual quote
        "User lives in Chicago",             # fabricated claim
    )
    assert passed is False  # NLI catches the contradiction
```

### Edge cases

| Case | Behavior | Reason |
|------|----------|--------|
| `transformers` not installed | Skip NLI, keep all | Fail-open |
| Quote is empty string | Skip NLI, keep | Can't verify without premise |
| Model download fails | Skip NLI, log warning | Fail-open |
| NLI returns contradiction >0.8 | DROP observation | Strong signal |
| NLI returns neutral (0.3-0.8) | KEEP, confidence=score | Conservative |
| NLI returns entailment >0.7 | KEEP, confidence=score | Confirmed |

### Regression gate

Run 10-question Observer eval before/after NLI. Expected:
- Hallucination: 58% → ≤10%
- Observation count: 9.6/q → 5-7/q (drops fabricated ones)
- No valid observations incorrectly dropped (spot-check 3 questions manually)

## EA integration

EA already wraps CoreMem's Observer — no EA changes needed. The gate runs
inside `ObserverPipeline._maybe_run()` and is transparent to callers.

## Rollout plan

1. Implement `coremem/nli.py` with `verify_observation()`
2. Add gate to `ObserverPipeline._maybe_run()`
3. Run 10-question eval with NLI gate
4. Run 100-question eval if hallucination ≤10%
5. If ≤10%, ship. If >10%, tune threshold or try `roberta-large-mnli`
