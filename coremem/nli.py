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
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _get_nli() -> Any:
    """Lazy-load the NLI pipeline."""
    global _NLI_PIPELINE
    if _NLI_PIPELINE is None:
        from transformers import pipeline  # type: ignore[import-untyped]

        logger.info("nli_bart_large_mnli_loading")
        _NLI_PIPELINE = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            device=-1,  # CPU — zero infra
        )
    return _NLI_PIPELINE


def check_entailment(premise: str, hypothesis: str) -> tuple[bool, float]:
    """Check if premise logically entails the hypothesis using NLI.

    Uses the standard NLI pipeline: premise → hypothesis.
    Returns (entailed, confidence_score).

    Args:
        premise: The source_quote from the conversation (evidence sentence).
        hypothesis: The extracted observation claim (what was concluded).

    Returns:
        (True, 0.95) if premise entails hypothesis with high confidence.
        (True, 0.40) if neutral — keep but flag as low confidence.
        (False, 0.0) if contradiction or poor input.
    """
    if not premise or not hypothesis:
        return False, 0.0

    try:
        nli = _get_nli()
        result = nli({"text": premise, "text_pair": hypothesis})
        # Single input → dict, batch input → list[dict]
        if isinstance(result, dict):
            result = [result]
        label = result[0]["label"].lower()
        score = result[0]["score"]

        if label == "entailment" and score > 0.7:
            return True, score
        if label == "contradiction" and score > 0.8:
            return False, 0.0
        # NEUTRAL → keep with score as confidence
        return True, score

    except Exception as e:
        logger.warning("nli_check_failed", extra={"error": str(e)})
        return True, 0.5  # fail-open
