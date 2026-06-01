# Changelog

## [0.3.0] — 2026-06-01

### Added
- **Fuzzy keyword matching** — `difflib` (stdlib) fallback catches near-misses like "creamers" vs "creamer". +4% recall on LongMemEval.
- **Bigram keyword overlap** — "coffee creamer" matches as a phrase, not just individual words. +2% recall.
- **MMR session diversity reranking** — `_mmr_diversify()` in `search_enhanced()` prevents cross-encoder overfit by deduplicating sessions pre-rerank.
- **Score normalization** in `search_enhanced` merge — prevents one sub-query from dominating the candidate pool. +4% recall.
- **Content dedup** — hash-based deduplication before cross-encoder. +2% recall.
- **Query-type-aware depth** — counting questions get `depth=10`, temporal `depth=7`, default `depth=5`.
- **Incremental eval save/resume** — `--output` + `--resume` flags for multi-hour LongMemEval runs.
- **Per-question failure capture** — detailed diagnostics for missed questions.

### Changed
- **search 75.4% → 84.0% recall** (25-question sample). **search_enhanced 93.0% → 100%**.
- `id(r)` → `r.memory.id` merge dedup fix in `search_enhanced()`.
- Error handling in eval: ChromaDB crashes skip question but save progress.
- `seen_ids: set[int]` → `set[str]` to match TEXT PRIMARY KEY migration.

## [0.2.1] — 2026-05-31

### Changed
- **API rename**: `export()` → `fetch()`, `export_all()` → `fetch_all()`, `import_batch()` → `store()`. Old names misleadingly suggested file I/O. Backend internals (`StoreBackend.list()`, `StoreBackend.ingest_batch()`) unchanged.

## [0.2.0] — 2026-05-31

### Added
- `ts` (timestamp) parameter on `MemoryCore.ingest()` and `ingest_message()`.
- `role`, `session_id`, `user_id`, `agent_id`, `ts` stored as top-level columns in HybridBackend for filterable queries.

### Changed
- `SearchQuery.filters` renamed to `SearchQuery.metadata` for clarity.
- Text UUID primary keys (`id TEXT PRIMARY KEY`) in hybrid backend — client-side `uuid.uuid4()[:12]` generation.
- `metadata` stored as `TEXT` (serialized JSON) instead of `JSON` column type.
- Dependency caps: `hybriddb>=0.3.0,<1.0`.

## [0.1.0] — 2026-05-30

### Added
- Initial release: zero-LLM memory retrieval for AI agents.
- `MemoryCore` with `ingest()`, `ingest_many()`, `search()`, `wake_up()`, `deep_search_context()`.
- Deterministic search heuristics: keyword overlap, temporal, recency, person name, quoted phrase.
- Cross-encoder reranking (`search_enhanced`).
- Multi-query expansion (regex + optional LLM).
- L0-L3 wake-up context stack.
- Metadata/filters on `SearchQuery`.
- `StoreBackend` ABC with `ChromaBackend` and `HybridBackend`.
- `export()`, `export_all()` for paginated filter-based retrieval.
- `import_batch()` for bulk store.
- `delete()`, `count()`, `clear()` lifecycle management.
- 33 tests across backend chroma, backend hybrid, core, heuristics, layers.
