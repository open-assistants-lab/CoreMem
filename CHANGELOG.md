# Changelog

## [0.12.3] — Graph traversal experiment concluded (parked), hybriddb 0.5.5, instrumented eval harness

### Graph traversal (parked 2026-08-18)
- **Re-tested the falsified graph hypothesis on the fixed HybridDB graph (0.5.5)** with a corrected design (seeds = baseline's exact rerank window, candidates restricted to new sessions) and a full research-grounded edge set: topic, turn_qa, update, causal, self_reference, emotional, entity, semantic (see `docs/graph-edges-design.md`).
- **20-question subset: neutral** — identical to baseline on every metric (0.974 session recall).
- **S-scale (500 questions, ~48 sessions each): neutral-to-negative** — multi-session neutral (−0.0008 session recall at full 133), single-session exact ties, temporal-reasoning negative (−0.012 session, −0.008 message). The causal/update edges pull MMR toward graph-discovered sessions that are not temporally correct.
- The original PPR PoC (0.588/0.333) was measured against buggy graph code and overstated the harm; the corrected verdict is neutral-to-negative, not harmful. The interim +0.0053 multi-session signal at 75 questions was a small-sample artifact.
- Eval mode `memorycore_traversal_v2` remains ablation-ready; `coremem/traversal.py` keeps the edge set + timing instrumentation.

### Added
- **`scripts/eval_graph_s.py`** — resumable S-scale eval harness (checkpoint + crash-safe JSONL), per-question progress, and extensive instrumentation: ingest throughput (msgs/s), per-phase retrieval timing (seeds, graph build, traverse, rerank), graph composition (nodes, edges by type), and full retrieval metrics per mode.
- **Timing instrumentation in `coremem/traversal.py`** — `timings` dict parameter (seeds/graph_build/traverse/rerank phases); `_build_message_graph` returns edge counts by type.
- **Traversal performance fixes** — batched embeddings (9.4×), numpy-vectorized similarity, inverted-index pair construction, keyword frequency caps; 1-hop traverse is 180× cheaper than 2-hop (0.4s vs 73s per question) with identical metrics.

### Changed
- **hybriddb bumped to 0.5.5** (published to PyPI) — graph bug fixes: silent empty results on integer PKs, directed-PPR mass loss, node ID namespacing, traverse() type-filter ordering. Verified zero regression on the 20-question eval (identical metrics).
- **dev extras** now include `fastapi` and `uvicorn` (AML integration tests were skipped without them).

## [0.12.2] — Metadata filters, expanded score normalization, dream cursor, writer conflicts

### Fixed
- **`fetch()` metadata filter crashed on keys with quotes** — `json_extract(metadata, '$.{k}')` interpolated the key unescaped; a key containing `'` raised `OperationalError`, a key with a space silently returned zero rows. The JSON path is now a bind parameter with backslash-escaped quoted segments (`$."my key"`), which is injection-safe and handles spaces, quotes, and backslashes.
- **`delete()` silently ignored its `metadata` parameter** — the parameter existed in the signature but was never applied, so `delete(metadata=...)` deleted every row. The metadata filter now applies, matching `fetch()`.
- **Score normalization in `expanded` strategy was dead code** — `_search_messages_llm_expansion` computed max/min from `r.get("score", 0)` but HybridDB returns `_score`, so `score_range` was always 0 and per-variant normalization never ran. Variants with systematically lower raw scores now compete fairly.
- **`dream()` cursor advanced past failed chunks** — the cursor was written to the last pending date even when a chunk failed (LLM error / invalid output), so failed dates were never retried. The cursor now only advances past successfully processed dates; dates that already have dream entries still advance.
- **`MEMORY.md` and `index.md` had conflicting writers** — `dream()` appended promoted facts to `MEMORY.md` (compiler-owned, regenerated on every compile) and `rebuild_index()` wrote month navigation to the root `index.md` (compiler-owned page index). Promoted facts now live in `DREAMS.md`; month navigation lives in `monthly/index.md`.

### Tests
- 6 new regression tests (metadata filters with special chars, delete metadata, score normalization, dream cursor retry, dream promotion destination, rebuild index location). Suite: 148 pass.

## [0.12.1] — Bug fixes: recency heuristics, COREMEM_LLM_MODEL wiring, bundle anchor budget

### Fixed
- **Recency heuristics were dead code** — `recency_decay` and `temporal_boost` compared naive `datetime.now()` against timezone-aware timestamps, raising `TypeError` that was silently swallowed. Both now use `datetime.now(UTC)` with naive timestamps treated as UTC. The documented recency-aware rescoring now actually fires.
- **`COREMEM_LLM_MODEL` did not configure the compile model** — `get_core()` passed `llm_provider` (used only for query expansion) but the journal compiler was built with the hardcoded `openai:gpt-4o-mini` default. The env var now flows to `agent_journal_model`, and the CLI/MCP compile tools report failures cleanly instead of gating on the unrelated `_llm_provider`.
- **Bundle budget dropped the anchor message** — `_reconstruct_sessions` skipped any message exceeding the per-bundle budget, including the retrieved evidence itself. Anchors now always survive the budget; only the opening message and fill context are best-effort.

### Tests
- 9 new regression tests (recency/temporal with aware + naive timestamps, `get_core` model wiring, CLI compile error handling, anchor budget survival). Suite: 142 pass.

## [0.12.0] — MCP server, CLI, hooks

### Added
- **MCP server** (`coremem mcp`) — stdio transport, 5 tools: `recall`, `ingest`, `compile`, `rebuild_index`, `list_sessions`
- **CLI** (`coremem`) — subcommands: `recall`, `ingest`, `compile`, `rebuild`, `sessions`, `hook`, `mcp`
- **Hook handlers** for Claude Code and Codex — `UserPromptSubmit` (capture + retrieval injection), `Stop` (capture), `PreCompact` (no-op)
- **`get_core()` helper** — creates MemoryCore from `COREMEM_PATH` env var or `~/.coremem/` default
- **Integration configs** for Claude Code, Codex, and OpenCode in `integrations/`
- **`mcp` optional dependency** — `pip install coremem[mcp]`
- **`COREMEM_LLM_MODEL` env var** — configures LLM provider for `compile` tool

### Changed
- `pyproject.toml` — added `[project.scripts]` entry point, `mcp` extra

## [0.11.0] — Unified `recall()` API

### Changed
- **Unified all retrieval under `recall()`** — single method with `strategy` parameter (`direct`, `episodic` (default), `expanded`, `fusion`), `bundles` flag, and filter params (role, session_id, user_id, agent_id, ts_after, ts_before, metadata).
- **Default strategy is `episodic`** (query decomposition + cross-encoder reranking, zero LLM) — the strongest zero-LLM-retrieval mode across both oracle and S evaluations.
- **`search_messages` → `_search_messages`** (internal), `search_messages_decomposed` → `_search_messages_decomposed`, `search_messages_llm_expansion` → `_search_messages_llm_expansion`, `reconstruct_sessions` → `_reconstruct_sessions`, `search_with_fusion` → `_search_with_fusion`.
- **Episodic strategy always uses cross-encoder** — the non-reranked episodic variant is dropped (m@5: 0.867 vs 0.472 on oracle).
- **Eval script MODES** — removed `memorycore_decomposed` and `memorycore_episodic` (collapsed into `memorycore_episodic_reranked`).
- **`pyproject.toml` version** bumped to 0.11.0.

### Removed
- **`search_with_traversal`** — graph traversal, below baseline on every metric.
- **`search_with_context`** — context mode, below baseline.
- **`search_with_session_reranking`** — session reranking, session recall collapsed to 0.623.
- **`search_journal`** — journal search removed from eval.
- **Deterministic classifier in `recall(strategy="auto")`** — 78% accuracy on S, falsified as unreliable for production.
- **`SearchHit` import** in core.py — only used by removed `search_journal`.
- **8 deprecated tests** — traversal, context, session_reranking.

### Added
- **Filter params on `search_messages_decomposed`** — role, session_id, user_id, agent_id, ts_after, ts_before, metadata now pass through to underlying `search_messages` calls.
- **`bundles` flag on `recall()`** — returns `list[SessionBundle]` with surrounding context.
- **`recall()` docstring** — documents all 4 strategies, bundles flag, and filter params.

### Eval work: LongMemEval Oracle + S, streaming loader, 429 retry, S dup-session fix

#### Added
- **429 retry with exponential backoff** in `_OllamaCloudAdapter._post_with_retry()` — retries up to 10 times on 429 with exponential backoff (1s→2s→4s→…→120s cap), respecting server `Retry-After` header. Both `chat()` and `chat_with_tools()` use it.
- **Quote sanitization** in `AgentJournalLLMCompiler._fix_source_quote()` — final quote is sanitized to replace `"` → `'` and newlines → spaces before storage, preventing `_require_quote` validation failures.
- **Streaming loader** `stream_longmemeval_instances()` in eval script — uses `ijson` to yield one question at a time (2 MB peak memory vs 2.4 GB for bulk load). Enabled with `--stream` flag.
- **`--stream` CLI flag** for eval script — streams questions one at a time for large datasets (S/M variants).
- **Per-question JSONL output** `--jsonl-output` in streaming mode — appends one line per question per mode as they complete, flushed immediately. Crash-safe raw results collection.
- **`ijson` dependency** for streaming JSON parsing.

#### Fixed
- **Duplicate session IDs in S/M variants** — `_prepare_instance()` now uses position-based public_session_id (`lme_{index:04d}_session_{session_index:04d}`) instead of mapping raw_session_id to public_session_id. The S/M datasets have the same raw_session_id appearing multiple times within a question, which caused `UNIQUE constraint failed: messages.id` in SQLite.

#### Changed
- `_OllamaCloudAdapter` now imports `asyncio` for retry sleep.
- `_prepare_instance()` session ID generation now always uses position index, not raw session ID lookup.

#### LongMemEval Oracle Results (500 questions, ~2 sessions/q, k=5)
- `memorycore`: 93.8% session_recall@5, 75.4% message_recall@5 (zero LLM calls)
- `memorycore_deep`: **95.1% session_recall@5, 85.4% message_recall@5** (1 LLM call/q for query expansion)
- `memorycore_journal`: 66.6% session_recall@5, 60.0% message_recall@5 (1 LLM call/q for journal compilation)
- All 3 modes: 0% abstention false positive rate
- Results: `eval_output/lme-oracle/results.json`

#### LongMemEval S Results (500 questions, ~48 sessions/q, k=5, memorycore only)
- `memorycore`: 86.5% session_recall@5, 67.0% message_recall@5, 96.8% session_hit@5 (zero LLM calls)
- `memorycore_deep` and `memorycore_journal` not yet run on S — need bigger VM (cross-encoder ~500 MB RAM)
- Best types: single-session-assistant (1.0), single-session-user (0.969), knowledge-update (0.931)
- Hardest types: multi-session (0.779), temporal-reasoning (0.796)
- Results: `eval_output/lme-s/results.json`, `eval_output/lme-s/results.jsonl`

## [0.10.0] — 2026-06-28 — AgentJournal: replace observer/reflector with deterministic compiler + dreaming + BM25 search

### Breaking changes
- **Observer/reflector pipeline removed.** `ObserverPipeline`, `ReflectorPipeline`, `ToolExtractor` deleted. No more LLM-based observation extraction per turn.
- **`coremem/migrations/` removed.** Observer-specific schema migrations deleted.
- **Old source deleted:** `coremem/observer.py`, `coremem/observer_utils.py`, `coremem/reflector.py`, `coremem/tool_extractor.py`.
- **Old test files deleted:** `test_observer.py`, `test_observer_gleaning.py`, `test_reflector.py`, `test_pipelines.py`, `test_tool_extractor.py`.
- **Old scripts deleted:** `scripts/judge_prompts.py`, `scripts/verify_prompts.py`, `benchmarks/longmemeval/`.

### Added
- **`coremem/agent_journal/`** — new module replacing observer/reflector architecture:
  - `AgentJournalBundle` — file-based journal bundle (daily pages, agent context manifest, linting)
  - `AgentJournalCompiler` — deterministic compiler: validates exact quotes, enforces evidence-type role constraints, applies structured plans without LLM calls
  - `AgentJournalLLMCompiler` — LLM-backed compiler with retry loop, quote-fixing post-processor, caching, auto-extract fallback
  - `AgentJournalSearch` — BM25 + stemming + fuzzy matching (Levenshtein) + stopword filtering + cross-encoder re-ranking
  - `CrossEncoderReranker` — sigmoid-normalized cross-encoder re-ranker with public `load()` method
  - `dream()` — diary study consolidation: events/emotions/cognitions/behaviors/context analysis, 7-day chunking, dedup, cursor tracking
  - `rebuild_index()` — generates `weekly/`, `monthly/`, `index.md` navigation from daily pages
- **Daily journal format** — `daily/YYYY-MM-DD.md` with timestamped sections (`## HH:MM - Title`), citations excluded from BM25
- **`MemoryCore.compile_turn()`** — async method compiling a turn into a daily journal entry
- **`MemoryCore.search_journal()`** — searches compiled journal pages via BM25 + cross-encoder
- **`MemoryCore.dream()`** — async consolidation across diary pages
- **`MemoryCore.rebuild_index()`** — regenerate weekly/monthly/index navigation
- **`ingest_turn()`** — batch-ingest a list of messages under one `turn_id`; returns the `turn_id`
- **`ingest()` auto-generates `turn_id`** — user messages start a new turn, assistant/tool messages join the current turn

### Renamed
- **`AgentMemory*` → `AgentJournal*`** — `AgentMemoryBundle` → `AgentJournalBundle`, `AgentMemorySearch` → `AgentJournalSearch`, `AgentMemoryCompiler` → `AgentJournalCompiler`, `AgentMemoryLLMCompiler` → `AgentJournalLLMCompiler`, `AgentMemoryError` → `AgentJournalError`, `AgentMemoryCompileResult` → `AgentJournalCompileResult`
- **`MemoryCore.search_memory()` → `MemoryCore.search_journal()`**
- **`MemoryCore.search()` → `MemoryCore.search_messages()`**
- **`MemoryCore.search_enhanced()` → `MemoryCore.search_messages_deep()`**
- **`compile_memorypack_plan` → `compile_journal_plan`**
- **`coremem/agent_memory/` → `coremem/agent_journal/`** (directory)
- **`MemoryPack` → `AgentJournal`** in all docstrings, LLM prompts, generated file headers, error messages
- **`agent_memory-turn` code block marker → `agent_journal-turn`**
- **`SCHEMA_VERSION`**: `"memorypack-poc-0.1"` → `"agent-journal-0.1"`
- **Frontmatter field**: `agent_memory_version` → `agent_journal_version` (lint accepts both for migration)
- **Test files**: `test_memorypack*.py` → `test_agent_journal*.py`
- **Eval scripts**: `eval_memorypack*.py` → `eval_agent_journal*.py`
- **Journal path**: directory name `agent_memory/` → `agent_journal/` (still co-located with HybridDB)

### Changed
- `MemoryCore.__init__()` docstring clarifies two-tier model: HybridDB for raw messages, AgentJournal for compiled pages
- `MemoryCore.compile_turn()` now derives the display timestamp from the first message and no longer requires `timestamp` or `title` arguments; omitted titles use the generated AgentJournal page title.
- `MemoryCore.compile_turn()` is now idempotent for unchanged turns and returns `AgentJournalCompileResult | None`.
- Added `MemoryCore.compile_latest_turn(session_id=...)` and `MemoryCore.compile_uncompiled_turns(...)` for explicit compile automation.
- HybridDB table schema includes `turn_id TEXT` column + index (auto-created if missing)
- HybridDB table schema includes a `compiled_turns` ledger to prevent duplicate AgentJournal sections for unchanged turns.
- `coremem/__init__.py` exports only `MemoryCore` and top-level utilities (no AgentJournal classes)

### Fixed
- `MemoryCore.compile_turn()` now uses `HybridDB.raw_query()` for SQL lookup instead of calling `query()` with raw SQL.

### Tests
- 92 tests pass

## [0.9.1] — 2026-06-15 — Observer bug fixes: importance, watermark, decay, metadata, timestamp

### Fixed
- **Observer discards LLM-provided importance** — `_parse_response()` was unconditionally setting `importance=None` on every observation, discarding the scores the LLM was prompted to compute. Reflector's `_assign_importance_to_pending` still catches legacy null-importance observations.
- **Observer watermark logic inverted** — `_maybe_run()` watermark loop collected already-processed (older) messages while skipping new ones, because the loop iterates newest-first (`ORDER BY ts DESC`) but collected messages *after* finding the watermark. Rewritten to collect messages until the watermark is reached, then stop.
- **`apply_decay()` ignores `half_life_days`** — `cutoff` was computed from the parameter but never used in the SQL query. Added `observation_ts` column to reflections schema (with migration), populated on insert, and wired `observation_ts < ?` filter into decay query.
- **`fetch()` ignores `metadata` parameter** — `metadata` was accepted but never wired into the WHERE clause. Now uses `json_extract(metadata, '$.{k}')` matching the pattern in `get_observations()`, `get_observations_since()`, and `delete_observations()`.
- **`get_observations_since` strict `>` on timestamps** — Changed to `>=` and excludes the reference `id` to avoid missing same-timestamp records.

### Tests
- All 125 tests pass.

### Added
- `ToolExtractor` — new pipeline class for session-end tool message analysis. Reads `role='tool'` messages, pairs with assistant `tool_calls` by `tool_call_id`, produces structured `tool_summary` observations with deterministic analysis (no LLM required).
- `_classify_error()` — heuristic error detection using keyword matching (`"Error:"`, `"failed"`, `"not found"`, `"could not"`).
- `_build_trace()` — pairs tool calls with results, detects error→retry→success recovery patterns.
- `_analyze_deterministic()` — counts errors per tool, builds tool coverage, sequences of length 2-3, and recovery patterns.
- `MemoryCore.session_end()` — lifecycle hook that triggers `ToolExtractor` at session end. Accepts `session_id`, `user_id`, `active_skills`, `min_tool_messages`.
- `metadata` TEXT column on `observations` table — stores structured JSON for `tool_summary` observations.
- 21 unit/integration tests in `tests/test_tool_extractor.py`

### Changed
- `MemoryCore.__init__()` now takes `enable_tool_extractor` kwarg (defaults to `enable_observations` value).
- `__init__.py` exports `ToolExtractor`.

### Added
- `python-dotenv` dependency; `.env` auto-loaded during tests via `conftest.py`
- `tool_temp` parameter and `chat_with_tools()` method on `OllamaCloudAdapter`

### Fixed
- `ReflectorPipeline` tests (missing `store` → `memory=memory` rename after v0.6.0 refactor)
- Gleaning integration tests use `ollama-cloud:deepseek-v4-flash` and `OLLAMA_API_KEY`

## [0.6.0] — 2026-06-05 — Observer API merge, single-backend simplification

### Breaking changes
- `MemoryCore()` takes `path: str` instead of `backend: StoreBackend`. HybridDB is the only backend. ChromaBackend removed.
- `from coremem.backends.hybrid import HybridBackend` no longer needed — `MemoryCore` creates it internally.
- `MemoryStore` class deleted. All methods moved to `MemoryCore`: `get_observations()`, `search_observations()`, `insert_observations()`, `get_recent_observations()`, `insert_reflections()`, `get_reflections()`, etc.
- `ObserverPipeline(core=foo, store=bar)` → `ObserverPipeline(memory=foo)`.
- `pipeline.after_turn()` → `pipeline.extract()`.

### Added
- `pipeline.retrieve(query=None, days=30, limit=50)` — returns recent observations or semantic search results.
- `MemoryCore(path, enable_observations=True)` creates 5 tables in one HybridDB: messages, observations, observation_events, observation_conflicts, reflections.
- `observation_events` table (was `memory_events`) with `observation_id` column (was `memory_id`).
- `observation_conflicts` table (was `memory_conflicts`) with `observation_id_a`/`observation_id_b` columns (were `memory_id_a`/`memory_id_b`).

### Removed
- `coremem/backends/` directory (ChromaBackend, HybridBackend, StoreBackend ABC).
- `coremem/memory_store.py` (absorbed into MemoryCore).
- `coremem/ingest.py` (inlined into MemoryCore).
- ChromaBackend dependency (`chromadb` no longer required).
- `tests/test_memory_store.py`, `tests/test_hybrid_backend.py`, `tests/test_chroma_backend.py`, `tests/test_ingest.py`.

## [0.5.1] — 2026-06-05 — Documentation & stability

### Added
- README section documenting ObserverPipeline, the 7-LF architecture, and the tradeoff between LLM-based LFs (universal language support, 0% hallucination) and non-LLM alternatives (English-only, unverified).

### Changed
- `pyproject.toml` version bumped to 0.5.1 (matching `__init__.py`).

## [0.5.0] — 2026-06-05 — Stance extraction, classifier debias, dedup fixes

### Added
- 7th labeling function `_LF_STANCE_PROMPT` for opinion/position/belief extraction — detects hard positions ("should", "must", "ban"), values, tradeoffs, and adequacy judgments.
- `stance` as 13th memory type in classifier (alongside profile, preference, project, decision, technical_stack, business_context, people, constraint, workflow, episodic, procedural, sentiment).

### Changed
- Classifier prompt: removed "99% durable / when in doubt durable" bias. Replaced with decision rules distinguishing durable (reveals user identity/preferences) from temporary (one-off requests, session context).
- Classifier examples updated: 15 durable examples + 4 temporary examples (was 0 temporary examples).
- `_LF_ACTIONS_PROMPT` added to `_LABELING_FUNCTIONS` (was temporarily excluded).

### Bug fixes
- **Intra-turn dedup:** Step 0 added to `dedup_and_merge()` catches cross-phase near-duplicates using SequenceMatcher > 0.70 before the LLM dedup stage. Prevents duplicate observations from Phase 2 and Phase 3 producing the same fact.
- **Double-append bug:** Redundant `final.append(archived_obs)` in dedup Step 1 removed — archived observations were duplicated in the final list.
- **Test mock fix:** `test_valid_quote_is_inserted_with_alignment_tier` and `test_fabricated_quote_is_dropped` updated to expect 7 LFs (was 6).

### Performance
- 20-question LongMemEval: 417 observations, 0% hallucination, 0 duplicates, 34% temporary rate.
- Per-question average: ~35s on DeepSeek V4 Flash (7 parallel LFs + 1 batch relation extraction).

### Verification
- Human-vs-pipeline manual comparison across all 20 questions: 75% PASS, 15% MINOR, 10% FAIL.
- 2 FAIL cases (third-party events, contextual asides) are the measured cost of the hallucination gate.

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
