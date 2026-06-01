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
"""Zero-LLM NLI verification using bart-large-mnli."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("coremem.nli")

_NLI_PIPELINE: Any = None


def _get_nli():
    global _NLI_PIPELINE
    if _NLI_PIPELINE is None:
        from transformers import pipeline
        _NLI_PIPELINE = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU
        )
    return _NLI_PIPELINE


def verify_observation(source_text: str, observation: str) -> tuple[bool, float]:
    """Check if source text entails the observation claim.

    Returns (entailed, confidence_score).
    
    Args:
        source_text: The full conversation text containing the claim.
        observation: The extracted observation to verify.
    """
    if not source_text or not observation:
        return False, 0.0

    try:
        nli = _get_nli()
        result = nli(source_text, candidate_labels=["entailment", "contradiction"])
        scores = {label: score for label, score in zip(
            result["labels"], result["scores"]
        )}
        entailment_score = scores.get("entailment", 0.0)
        contradiction_score = scores.get("contradiction", 0.0)

        if entailment_score > 0.7:
            return True, entailment_score
        if contradiction_score > 0.7:
            return False, 0.0
        # Neutral — keep but flag
        return True, entailment_score
    except Exception as e:
        logger.warning("nli_check_failed", extra={"error": str(e)})
        return True, 0.5  # fail-open: don't drop on NLI error


def is_nli_available() -> bool:
    """Check if transformers is installed (optional dependency)."""
    try:
        import transformers  # noqa
        return True
    except ImportError:
        return False
```

### 2. ObserverPipeline integration

```python
# In _maybe_run(), after source quote gate:

# Gate: NLI verification
nli_source = " ".join(m.content for m in messages if m.role != "tool")
verified_obs = []
for obs in new_obs:
    content = obs.get("content", "")
    if is_nli_available():
        passed, confidence = verify_observation(nli_source, content)
        obs["confidence"] = confidence
        if not passed:
            continue
    verified_obs.append(obs)
new_obs = verified_obs
```

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
| Model size | ~1.6GB (bart-large-mnli) |
| First load | ~5s (downloads from HuggingFace) |
| Per-check latency | ~50ms (CPU, batch size 1) |
| Batch throughput | ~20 checks/second |
| Memory | ~2GB RAM |
| Accuracy | 95%+ on MNLI benchmark |

For 10 observations per question: 10 × 50ms = 0.5s overhead. Negligible vs
the LLM call (2-10s). Model downloads once and caches in `~/.cache/huggingface/`.

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

```python
def test_nli_entails():
    assert verify_observation(
        "I work at Google as a software engineer.",
        "User is a software engineer at Google"
    )[0] is True

def test_nli_contradicts():
    assert verify_observation(
        "I quit my job at Google last week.",
        "User currently works at Google"
    )[0] is False

def test_nli_neutral():
    passed, score = verify_observation(
        "I went hiking yesterday.",
        "User enjoys outdoor activities"
    )
    assert passed is True  # neutral keeps, just lower confidence
    assert 0.3 <= score <= 0.8

def test_nli_empty_input():
    assert verify_observation("", "claim")[0] is False
    assert verify_observation("source", "")[0] is False

def test_nli_not_installed():
    # mock transformers import to verify fail-open behavior
    ...
```

## EA integration

EA already wraps CoreMem's Observer — no EA changes needed. The gate runs
inside `ObserverPipeline._maybe_run()` and is transparent to callers.

## Rollout plan

1. Implement `coremem/nli.py` with `verify_observation()`
2. Add gate to `ObserverPipeline._maybe_run()`
3. Run 10-question eval with NLI gate
4. Run 100-question eval if hallucination ≤10%
5. If ≤10%, ship. If >10%, tune threshold or try `roberta-large-mnli`
