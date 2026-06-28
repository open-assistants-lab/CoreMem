# MemoryPack Embedding + Hybrid Search

2026-06-24

## 1. Goal

Add embedding-based semantic search alongside BM25 for compiled pages only.
Raw messages stay BM25-only (baseline). Embeddings stored as numpy array +
hash manifest for integrity. Change detection via explicit rebuild, not
on every lint.

## 2. Architecture

```
pages/*.md                ← compiled memory (human-readable)
.embeddings/
  pages.npy                ← float32 array, shape (N, 384)
  pages_ids.json           ← ordered list of page_ids matching array rows
  pages_hashes.json        ← sha256 of each page content for change detection
  queries.npy              ← query embedding cache, shape (M, 384)
  queries_hashes.json      ← sha256 of each query for cache invalidation
manifest.json              ← existing, extended with embedding_version
```

Raw messages (`references/turns/*.md`) are NOT embedded. They remain
BM25-only for the baseline eval. Only compiled pages get embeddings.

## 3. Embedding Model

`sentence-transformers/all-MiniLM-L6-v2`:
- 384-dimensional embeddings
- 80MB model on disk
- ~50ms per query on CPU
- 256-token limit (use `model.max_seq_length` for truncation)
- No GPU required
- MIT license

## 4. Change Detection

NOT on every `lint()`. Separate explicit method:

```python
bundle.rebuild_embeddings()
```

Called once after all compilation is done. Flow:
1. Check if `.embeddings/` exists
2. If not, embed all pages (first run)
3. If yes, read `pages_hashes.json`, compute `sha256` of each page,
   re-embed only pages whose hash changed
4. Detect deleted pages: if a page_id in `pages_ids.json` no longer exists
   in `pages_dir`, remove its embedding row
5. Update `.embeddings/pages.npy`, `pages_ids.json`, `pages_hashes.json`

During batch compilation (380 pages), `rebuild_embeddings()` is called
once at the end, not after every `apply_plan()`.

## 5. Truncation

`all-MiniLM-L6-v2` has a 256-token limit. Compiled pages can be 1-3KB.
Truncation strategy:
- Use `model.tokenizer.encode(text, truncation=True, max_length=256)`
- This uses the model's native max_seq_length (256)
- No arbitrary threshold, no chunking needed (pages are single-topic)

## 6. Query Embedding Cache

Cache query embeddings by `sha256(query)`:
- In-memory dict: `{sha256(query): embedding_array}`
- Before embedding a query, check the dict
- If found, return cached embedding
- If not found, embed, store in dict
- Cache is in-memory only — cleared on process restart
- No disk persistence for query cache (queries are fast to embed, ~50ms)

## 7. BM25 Fallback

If the embedding model fails to load (OOM, missing file, import error):
- Log warning
- Fall back to BM25-only search
- No crash, no data loss
- User can retry with `rebuild_embeddings()` after fixing the issue

## 8. Batch Embedding

During build, embed pages in batches of 32:
- `model.encode(texts, batch_size=32)`
- 380 pages / 32 = ~12 batches
- ~6 seconds total (vs ~190s one-at-a-time)

## 9. Hybrid Search

`AgentMemorySearch.search()` becomes:

```python
def search(query, limit=5):
    bm25_hits = _bm25_search(query, limit=limit*3)
    emb_hits = _embedding_search(query, limit=limit*3)
    merged = _rrf_merge(bm25_hits, emb_hits, k=30)
    return merged[:limit]
```

Reciprocal Rank Fusion (RRF):
```
score(d) = sum over rankers r of 1 / (k + rank_r(d))
```

k=30 (tunable, start with 30 for 2 rankers on ~400 docs; k=60 is standard
for web search with many rankers but may over-dampen with only 2 rankers).

## 10. EmbeddingIndex Lifecycle

- `AgentMemoryBundle.__init__()` takes optional `embed_model: str | None`
- If provided, creates `EmbeddingIndex(self.root, model_name=embed_model)`
  but does NOT load the model or build embeddings yet (lazy init)
- `AgentMemoryBundle.rebuild_embeddings()` calls `self._embedding_index.refresh(self.pages_dir)`
  which loads the model on first call, then builds/updates embeddings
- `AgentMemorySearch.__init__()` takes optional `embedding_index` parameter
- `AgentMemorySearch.search()` checks if `embedding_index` is set and has
  embeddings loaded; if so, runs hybrid; otherwise BM25-only

In the eval script:
- `run_eval()` creates `AgentMemoryBundle` with `embed_model` parameter
- After compilation, calls `bundle.rebuild_embeddings()`
- Creates `AgentMemorySearch` with the bundle's `embedding_index`
- `_search_compiled_pages()` receives the `AgentMemorySearch` instance
  (or the `EmbeddingIndex` directly) and uses it for hybrid search

## 11. Files Changed

### New
- `coremem/memorypack/embeddings.py` — `EmbeddingIndex` class:
  - `__init__(root, model_name="all-MiniLM-L6-v2")` — lazy, no model load
  - `build(pages_dir)` — embed all pages in batches, write to `.embeddings/`
  - `search(query, limit)` — embed query (with cache), cosine sim, return top-k
  - `refresh(pages_dir)` — re-embed only changed pages, remove deleted
  - `_embed(texts)` — batch embed list of texts → numpy array
  - `_load()` / `_save()` — read/write `.embeddings/` files
  - `_truncate(text)` — use model's tokenizer, max_length=256
  - `_load_model()` — lazy load on first use

### Modified
- `coremem/memorypack/bundle.py`:
  - `AgentMemoryBundle.__init__()` — optional `embed_model` parameter
  - `AgentMemoryBundle.rebuild_embeddings()` — new method
  - `AgentMemorySearch.__init__()` — optional `embedding_index` parameter
  - `AgentMemorySearch.search()` — hybrid path (BM25 + RRF + embedding)
- `scripts/eval_memorypack_llm_compiler.py`:
  - `_search_compiled_pages()` — add hybrid path (same RRF logic)
  - `run_eval()` — pass `embed_model`, call `rebuild_embeddings()` after compile
- `scripts/eval_memorypack_longmemeval.py`:
  - `_search_reference_messages()` — UNCHANGED (raw messages stay BM25-only)

## 12. Dependencies

Add `sentence-transformers` to `pyproject.toml`:
```
sentence-transformers>=3.0.0
```

This pulls in `transformers`, `torch`, `numpy` (numpy already present in
project's dependencies — verify with `grep numpy pyproject.toml`).
Total additional disk: ~1GB (model 80MB + torch ~800MB + transformers ~150MB).

## 13. Verification

Run Stage 4 eval on 8-instance set with hybrid search. Compare:

| Metric | BM25 only | Hybrid (BM25 + embedding) |
|---|---|---|
| session_recall@5 | 0.812 | ? |
| session_precision@5 | 0.300 | ? |
| session_mrr | 0.854 | ? |
| session_map | 0.635 | ? |

Expected: recall@5 improves to 0.85-0.90, precision@5 improves to 0.35-0.40.

## 14. Future (not in this spec)

- HybridDB integration for raw message search at scale
- Multi-model support (swap all-MiniLM for OpenAI ada-002)
- GPU acceleration for batch embedding
- Chunking for multi-topic pages (not needed yet, pages are single-session)
