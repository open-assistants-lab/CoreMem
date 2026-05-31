# Changelog

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
