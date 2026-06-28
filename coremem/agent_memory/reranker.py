"""Cross-encoder re-ranker for MemoryPack search."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from coremem.agent_memory.bundle import SearchHit

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-rank BM25 candidates using a cross-encoder model.

    Lazy-loads the model on first ``rerank()`` call. Cached for session lifetime.
    Falls back to returning candidates as-is if the model fails to load.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None

    def load(self) -> None:
        """Eagerly load the cross-encoder model."""
        self._load()

    def rerank(self, query: str, candidates: list[SearchHit], limit: int = 5) -> list[SearchHit]:
        """Re-rank candidates by cross-encoder (query, document) scores."""
        if not candidates:
            return []
        model = self._load()
        if model is None:
            return candidates[:limit]
        try:
            texts = [hit.path.read_text(encoding="utf-8")[:2000] for hit in candidates]
            pairs = [(query, t) for t in texts]
            logits = model.predict(pairs, batch_size=32, show_progress_bar=False)
            scores = [1.0 / (1.0 + math.exp(-logit)) for logit in logits]
            scored = sorted(zip(candidates, scores), key=lambda x: -x[1])
            return [SearchHit(path=hit.path, score=float(s), session_id=hit.session_id) for hit, s in scored[:limit]]
        except Exception:
            logger.warning("cross-encoder rerank failed, returning BM25 results", exc_info=True)
            return candidates[:limit]

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self._model_name, max_length=512)
            except Exception:
                logger.warning("failed to load cross-encoder model", exc_info=True)
                return None
        return self._model
