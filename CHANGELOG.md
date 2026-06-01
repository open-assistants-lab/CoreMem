# Changelog

## [0.4.0] — 2026-06-02 — Observer rewrite

### Breaking changes
- `coremem.nli` module removed. `bart-large-mnli` no longer a dependency.
- `pip install coremem[nli]` is a no-op.
- `Observer.run()` signature changed: `messages: list[Memory]` (was `conversation: list[dict]`); new `observation_date: str | None` arg.
- `OBSERVATION_TOOL` schema: `priority` field removed. Observations no longer have a `priority` key.
- `coremem[all]` no longer includes `nli`.

### Added
- `coremem.grounding.align_quote()` — 3-tier alignment gate (EXACT / FUZZY / drop). Port of `langextract/resolver.py:316-400`.
- `AlignmentTier` and `AlignmentResult` types exported from `coremem.grounding`.
- `Observer` and `ObserverPipeline` accept `enable_gleaning: bool = False` flag (raises `NotImplementedError` when True; reserved for future CogCanvas-style gleaning pass).
- Schema migration for `observations.alignment_tier` and `observations.alignment_confidence` columns (idempotent, runs on `MemoryStore.__init__`).

### Changed
- `Observer` is now single-pass: one `chat_with_tools` call per `run()` (was two-pass with NLI verification).
- Temperature default changed: 0.0 → 0.1 in `_OpenAIAdapter.chat_with_tools` (CogCanvas pattern).
- Prompt format changed: native messages array with `[ts] content` prefix (was JSON-wrapped content).
- System prompt rewritten: CogCanvas pattern with 2 few-shot examples demonstrating verbatim-quote contract.
- FUZZY tier uses character-level `SequenceMatcher.ratio()` on whitespace+case-normalized strings (chose char-level over token-level because LLM source_quote drift is typically single-character/punctuation).

### Bug fixes
- **Bug #1:** `Observer.run` Pass 1 was reading `tool_calls` payload from `response.content` (always empty for tool calls). Now correctly reads from `tool_calls[0].function.arguments`.
- **Bug #2:** Prompt input had `[role | ts | meta]` prefix but verification source did not. Now uses identical canonical text (`[ts] content`) for both.
- **Bug #3:** `_quote_verified` internal check was inverted (checked if claim was substring of quote). Replaced by the 3-tier alignment gate.
- **Bug #4:** Two-pass design dropped (Pass 1 was dead). Single LLM call with CogCanvas-style prompt + few-shot examples.
- **Reflector filter:** Silently-broken priority filter (never matched emoji values) replaced with `importance >= 0.5` check.

### Performance
- Two-pass design dropped: per-observation time target down from ~500s to ~150s.
- `bart-large-mnli` (1.6GB) no longer required for installation.

### Verification
- LongMemEval re-evaluation pending: target <10% hallucination on DeepSeek V4 Flash (down from 34-58% in 0.3.0).

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
