# MemoryPack Cross-Encoder Re-Ranker

2026-06-24

## 1. Goal

Add a cross-encoder re-ranker on top of BM25 retrieval. BM25 retrieves top 20 candidates, cross-encoder scores each (query, document) pair, top 5 returned. This is the standard "retrieve + re-rank" pipeline used in production search systems.

## 2. Architecture

```
query → BM25 (top 20) → cross-encoder (score each pair) → top 5
```

No change to the retrieval stage. BM25 stays as-is. The cross-encoder is a separate step that re-orders the BM25 results. If the cross-encoder fails to load, BM25 results are returned directly (graceful degradation).

## 3. Model

`cross-encoder/ms-marco-MiniLM-L-6-v2`:
- 384-dimensional, 80MB
- ~20ms per (query, doc) pair on CPU
- 20 pairs × 20ms = ~400ms per query
- Supports 512 tokens (use tokenizer for truncation)
- Trained on MS MARCO passage ranking (web search relevance)
- MIT license

## 4. Limitations

Cross-encoders improve ranking, not recall. If the correct page is not in the top 20 BM25 candidates, re-ranking cannot help. The recall ceiling is BM25's top-20 recall. Expected improvement is in precision@5 and MAP, not recall@5.

## 5. Integration

### New file: `coremem/memorypack/reranker.py`

```python
class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = None  # lazy loaded on first rerank() call, cached for session

    def rerank(self, query: str, candidates: list[SearchHit], limit: int = 5) -> list[SearchHit]:
        if not candidates:
            return []
        model = self._load()
        texts = [p.path.read_text(encoding="utf-8") for p in candidates]
        pairs = [(query, t) for t in texts]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        scored = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [SearchHit(path=hit.path, score=float(s)) for hit, s in scored[:limit]]

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, max_length=512)
        return self._model
```

### Modified: `coremem/memorypack/bundle.py`

- `AgentMemorySearch.__init__()` — optional `reranker` parameter
- `AgentMemorySearch.search()` — retrieve top 20 with BM25, if reranker is set, re-rank to top 5; if reranker fails, return BM25 results

### Modified: `scripts/eval_memorypack_llm_compiler.py`

- `run_eval()` — create `CrossEncoderReranker`, pass to `AgentMemorySearch`
- `_search_compiled_pages()` — pass through to `AgentMemorySearch`

## 6. Dependencies

`sentence-transformers` already in `pyproject.toml`. Cross-encoder models use the same `transformers` library. No new deps.

## 7. Verification

Run Stage 4 eval on 8-instance set. Compare:

| Metric | BM25 only | BM25 + cross-encoder |
|---|---|---|
| session_recall@5 | 0.812 | ? |
| session_precision@5 | 0.300 | ? |
| session_mrr | 0.854 | ? |
| session_map | 0.635 | ? |

Expected: precision@5 improves to 0.35-0.45, MAP improves. Recall@5 may stay flat or improve slightly (if re-ranking brings a correct page from rank 6-20 into top 5).

## 8. Files

- `coremem/memorypack/reranker.py` (new, ~40 lines)
- `coremem/memorypack/bundle.py` (modified, ~10 lines)
- `scripts/eval_memorypack_llm_compiler.py` (modified, ~5 lines)
- `coremem/memorypack/__init__.py` (modified, add export)
