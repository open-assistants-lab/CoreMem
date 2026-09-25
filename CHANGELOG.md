# Changelog

## [0.17.0] — fact layer (semantic layer v0): governed facts, zero-LLM lookups

A shallow, **closed-vocabulary fact layer** over the versioned memory store.
Facts are ordinary rows in a versioned HybridDB table, so every write lands on
the audit chain and is reversible with `rollback_memory` — no second source of
truth, no new persistence layer. The read path stays **zero-LLM**: fact lookups
are deterministic queries, and the fact table never enters message search, so
recall behavior is byte-identical with facts present or absent.

Deliberately **not** a knowledge graph: relation-graph traversal was falsified
twice (PPR, query-guided v2 — see `docs/retrieval-experiments.md`). Ontology is
a flat attribute vocabulary with cardinality, not entities and edges.

### Added
- **Attribute ontology** (`coremem/ontology.py`, extensible via
  `ontology.toml`): per-attribute `cardinality` (`single` | `multi`),
  `value_type` (`text` | `number` | `date`). Unknown attributes default to
  `multi`/`text` — the safe direction (append, never silently supersede).
- **Governed fact CRUD** (requires `versioned=True`, the default since 0.15.0):
  - `add_fact(entity, attribute, value, *, user_id="", source_message_id="", valid_from=None) -> str`
    — a write to a `single`-cardinality attribute **supersedes** the current
    fact: the old row gets `valid_to` + `superseded_by`, and both events are
    verifiable on the chain. `multi` attributes append.
  - `list_facts(*, entity=None, attribute=None, user_id=None, include_expired=False, limit=200) -> list[dict]`
    — current facts only by default; expired/superseded are hidden.
  - `get_fact(fact_id) -> dict | None`
  - `fact_history(fact_id) -> list[dict]` — the full audit trail, oldest first.
- **Lifecycle verbs**: `supersede_fact(fact_id, new_value) -> str` (does *not*
  pre-expire — `add_fact`'s supersession records both events),
  `expire_fact(fact_id) -> bool` (soft-expire; the expiry is itself history, the
  chain never rewinds), `merge_facts(fact_ids, merged_value) -> str` (dedupe
  "Max" + "my dog Max"; sources stay expired-and-auditable).
- **Budget-capped fact digest** — `coremem.reading.build_facts_digest(facts, *, budget_chars=800)`.
  Renders current facts as a grouped `[FACTS]` block for the reader prompt.
  Appended **after** the bundle context: it grows the prompt and never evicts
  verbatim evidence. Empty input → empty string, so the dual path is inert
  unless facts exist.
- **Three MCP tools**: `add_fact`, `list_facts`, `fact_history`.
- **Opt-in LLM fact extraction at ingest** — `MemoryCore(..., extract_facts=True)`
  (default **False**). Extracts candidate facts from user-role messages, each
  landing in the governed table with per-message provenance. Degrade-to-no-facts
  contract: provider errors or unparseable output log a warning and never raise
  into ingest. This is the only LLM in the feature, and it is off by default.

### Extraction stays opt-in-off — the kill-gate result, quoted
`extract_facts` is **not** enabled by default. Pre-registered paired A/B,
LongMemEval S stratified-56, blind judge (`gpt-oss:120b-cloud` for reader and
judge), `episodic_4k_reranked` vs `episodic_4k_reranked_factdigest` (same 4k
bundle + ≤800-char fact digest appended; retrieval identical by construction,
1.981 s in both arms), 56/56 rows:

> control **0.554** accuracy vs factdigest **0.571**; answerable 0.500 → 0.521;
> abstention 0.875 → 0.875; context 5,114 → 5,889 chars mean.
> **Paired answer flips +2 / −1 (net +1), exact two-sided binomial p = 1.000.**
> Per type (n=8 each): multi-session 0.38 → 0.50, single-session-preference
> 0.12 → 0.25, single-session-user 0.88 → 0.75; temporal-reasoning,
> knowledge-update, single-session-assistant and abstention all unchanged.

Directionally positive with the wins exactly where the mechanism predicts
(cross-session synthesis, preference consolidation) and the single regression
being the predicted distraction case — **but p = 1.000 on 3 discordant pairs is
not evidence of a working lever.** The checkpoint trajectory (n=18: −0.056;
n=45: +0.019; n=56: +0.017) is the argument for the pre-registered rule: early
stopping would have returned the wrong call in both directions. Per the
pre-registered criterion, net > 0 means *scale up*, not *flip the default*:
full-S confirmation is specified in `docs/retrieval-experiments.md` with its
kill criteria fixed before any full-S data is seen. Phase 1 (the governance
API) is useful without extraction and ships on its own merits.

### Fixed
- **Eval harnesses leaked a pooled Chroma client per question.**
  `eval_answer_longmemeval.py` and `eval_agent_journal_longmemeval.py` (both the
  standard and `--stream` paths) build 1–2 `MemoryCore` instances per question
  and never called `close()` — the exact accumulation documented on
  `MemoryCore.close`. On the 56-question fact-digest run this was the primary
  cause of death-by-OOM around question 9–18 (six process deaths total, five
  MPS OOM, one proxy 502). Closing the cores per question doubled process
  lifespan; measurements are unaffected (rows are already scored when the close
  happens).

### Notes
- `extract_facts` was evaluated as the write-side sibling of lever 2
  (fact-augmented embeddings), which was neutral on S and LoCoMo. The digest
  variant is the read-side sibling; it has now been measured once, at
  stratified-56 scale, with the result above.
- 208 tests pass. (Four `tests/test_cli.py` subprocess tests time out on
  memory-starved hosts — a cold CLI `recall` takes ~50 s there versus the test's
  30 s timeout; unrelated to this release and reproducible on 0.16.1.)

## [0.16.1] — timestamp filters compare chronologically (mixed-format fix)

### Fixed
- **`ts_after`/`ts_before` filters now compare chronologically** across mixed
  naive/aware timestamp formats — the documented known limitation since 0.13.2.
  A naive `2024-01-01T23:00:00` message was previously **excluded** by
  `ts_after="2024-01-01T02:00:00+00:00"` (lexically `"23:…" >= "02:…+00:00"`);
  chronologically 23:00 > 02:00, so it is now correctly **kept**. Filter
  values parse via ISO (tolerant of `Z` and date-only forms, naive → UTC);
  unparseable values fall back to the legacy lexical comparison. Boundary
  semantics unchanged (both filters exclusive).
- `pyproject.toml`: the wheel now ships `CHANGELOG.md` (force-include).

### Known limitations (updated)
- Timestamp **pushdown into the Chroma scan** remains deferred: stored `ts`
  strings in pre-existing stores keep arbitrary formats, so lexical comparison
  inside Chroma would repeat the same bug at the engine layer. The Python
  post-filter is now chronologically correct; storage-side normalization is a
  prerequisite for pushing ts ranges down.

## [0.16.0] — metadata pre-filtering: recall filters pushed into the Chroma scan

Adopts hybriddb 0.7.0's `search(where=)` — CoreMem's equality filters
(`role`, `session_id`, `user_id`, `agent_id`) are now pushed into the Chroma
query **before** the vector scan instead of relying on over-fetch + Python
post-filtering.

### Changed
- **Filtered recall no longer starves.** Previously a filtered recall fetched
  the top-`max(limit×20, 100)` rows *unfiltered* and post-filtered in Python —
  ≥100 more-relevant non-matching rows could push the answer out of the window
  entirely. Now the vector scan only sees matching rows. Demonstrated: with
  150 same-phrase rows in a noise session, the old path returned **empty** for
  `recall(session_id="target")` while the pushdown returns the target as the
  top hit (`tests/test_filter_pushdown.py::test_session_filter_finds_message_beyond_overfetch_window`).
- The Python post-filter (`_matches_filters`) remains the authority (defense in
  depth, unchanged semantics) — `where=` matches its truthiness exactly.
- Query-side: the pushdown applies to every strategy (`direct`, `episodic`
  per-variant, `llm_expansion`, `fusion`, preference union) since they all
  funnel through the two `db.search` call sites.

### Not pushed down (by design)
- `ts_after`/`ts_before` (lexical ISO range — mixed-format caveat; stays
  Python post-filter) and `metadata={}` (JSON inside a TEXT column, not a
  scalar Chroma key). These filter identically to 0.15.x.
- Unfiltered recall is byte-identical (S stratified-56: 0.965/0.537 —
  pushdown only activates with filters).

## [0.15.0] — versioned memory becomes the default, hybriddb 0.7.0 floor

⚠️ **Breaking: `versioned=True` is now the default** for newly created stores,
and the hybriddb floor is `>=0.7.0`.

### Changed
- **`MemoryCore(path)` now creates versioned stores by default** (0.14.0
  shipped it opt-in; the performance gate passed and hybriddb 0.7.0 resolved
  the one open finding). New stores carry the tamper-evident hash chain:
  `checkpoint_memory` / `rollback_memory` / `memory_history` / `verify_memory_chain`
  work out of the box, and MCP gains `memory_history` / `memory_rollback` /
  `memory_verify`. Pass `versioned=False` to opt out. **Stores created before
  this release have no memory history** (open + rebuild to gain it); opening
  one logs a warning.
- **hybriddb floor `>=0.6.0` → `>=0.7.0`** — picks up the batched rollback
  (removal-heavy rollback 3.85s → 0.32s at 1k rows, resolving the one missed
  gate from 0.14.0; handoff note at HybridDB
  docs/specs/2026-08-28-rollback-performance-report.md), metadata
  pre-filtering for search (`where=`), lazy DuckDB registration (stores never
  queried with olap skip the mirror), and the keyword-mode where operator fix.

### Performance (10k messages, scripts/bench_versioned_memory.py, hybriddb 0.7.0)
- Versioned ingest overhead: **+0.2%** (was +8.2% on 0.6.0 — lazy DuckDB
  offsets the chain cost)
- Recall latency unchanged (65.6 vs 65.8 ms); storage 1.14×
- **Rollback 1k removed rows: 0.32 s** (was 3.85 s on 0.6.0 — now 4× faster
  than ingesting those rows; gate ≤1.25 s ✅); verify_chain 0.05 s @12k events
- Retrieval metrics identical on 0.5.8/0.6.0/0.7.0 (S stratified-56:
  0.965/0.537 all three)

## [0.14.0] — versioned memory (git-like governance), hybriddb 0.6.0 floor

Memory governance via hybriddb 0.6.0's versioned tables: a tamper-evident hash
chain over every memory change, point-in-time reads, checkpoints, and rollback.
Design + measured performance gate in docs/versioned-memory-design.md; this
release ships it as **opt-in** (`MemoryCore(path, versioned=True)`). Default
flip pending one more release cycle (gate already passed — see below).

⚠️ **Breaking: hybriddb floor raised `>=0.5.8` → `>=0.6.0`.** hybriddb 0.6.0 is
retrieval-neutral (S stratified-56 metrics identical on 0.5.8 vs 0.6.0).

### Added
- **`MemoryCore(path, versioned=True, author=...)`** — NEW stores only: messages
  and journal_records carry a `SHA256(prev_hash | op | pk | row_json)` chain.
  Opened-on-legacy-store warns and stays un-versioned (versioning is
  create-time only; no in-place migration in v1).
- **Governance API**: `checkpoint_memory(label)`, `rollback_memory(label|seq)`
  (chain never rewinds — rollback is recorded as new versions),
  `memory_log(limit)`, `memory_history(message_id)` (provenance timeline),
  `memory_diff(from, to)`, `as_of_memory(seq)`, `verify_memory_chain()`.
- **MCP tools**: `memory_history`, `memory_rollback` (requires `confirm=true`),
  `memory_verify` — agents and operators can self-audit.
- `delete_messages`/`clear`/filtered deletes now route through the CRUD layer on
  versioned stores so tombstones land in history (`raw_query` writes bypass it
  — found by spike, fixed).
- `scripts/bench_versioned_memory.py` — the perf-gate benchmark.

### Performance gate (10k messages, median of 3, docs/versioned-memory-design.md §7)
- Ingest overhead **+8.2%** (gate ≤15%)
- Recall latency unchanged (63.5 → 66.7 ms warm)
- Recall metrics identical (0.965/0.537); storage **1.13×**
- Rollback 3.85 s / 1k rows (~4 ms/row, interactive op — above the strictest
  self-imposed bound, documented; hybriddb batch-rollback follow-up)
- verify_chain 0.05 s @ 12k events
- **→ `versioned=True` becomes the default in 0.15.0** (legacy stores
  unaffected).

## [0.13.4] — bge-small default embedder (+0.010 recall), CoN reading helper

Five-lever improvement program concluded (docs/retrieval-experiments.md). One lever
validated and folded in; two falsified; two kept as opt-in modes. All A/Bs paired
with blind judges on identical contexts, one variable at a time.

### Changed
- **Default embedding model: all-MiniLM-L6-v2 → BAAI/bge-small-en-v1.5** (lever 3).
  Full-S paired A/B (n=500, only embedder differs): message_recall@5 0.618 → 0.628
  (+0.010), session_recall 0.952 → 0.957, session_map +0.005, **zero regressions**.
  Biggest gain on the weakest type: single-session-preference +0.066 (0.367 → 0.433);
  multi-session +0.011, temporal +0.008. 9 questions better / 2 worse / 459 identical
  (exact binomial p=0.065). Type pattern matches the MemDelta controlled study.
  Bonus: bge-small batch-encodes 2× faster than MiniLM (1225 vs 648 texts/s), same
  384 dimensions. **Existing stores must be re-embedded** (vectors differ) — the
  `COREMEM_EMBEDDING_MODEL` env var can pin the old model for migration windows.

### Added
- **`coremem.reading` module** — answer-prompt builders for consuming bundles (RAG
  reading strategies, consumer-side choice): `build_answer_prompt(chain_of_note=)`
  ships the Chain-of-Note reading prompt validated at +0.018 accuracy on full S
  (0.674 → 0.692, 17/8 flips, p=0.108) with `extract_final_answer()` for CoN
  replies and documented guidance (enable for synthesis-heavy questions; direct
  prompt when abstention fidelity or token cost dominates — CoN carries ~50%
  more output tokens).
- `COREMEM_EMBEDDING_MODEL` env var to pin/swap the embedding model (ingest and
  query sides forced to the same model).
- Eval harness: `memorycore_episodic_reranked_{factaug,timeprune}` modes,
  conversation_id-aware shared instance stores (LoCoMo per-conversation cache).

### Falsified (documented, code retained as opt-in)
- Fact-augmented key expansion (LongMemEval §5.3): retrieval-neutral on S — the
  CE reranker restores the identical top-5 from a perturbed candidate pool
  (13/56 questions changed pre-rerank, recall delta 0.000 per type).
- BGE-reranker-v2-m3: −0.040 message recall at 7× latency (same failure
  signature as the earlier L-12 A/B); keep ms-marco-MiniLM-L-6-v2.
- Deterministic temporal window pruning: only 4/133 temporal questions carry a
  parseable range — LongMemEval temporal questions are inter-event arithmetic,
  not range queries. No-op.

## [0.13.3] — hybriddb 0.5.8 floor, version-string consistency

Dependency + packaging hygiene release, validated by an A/B eval on the 20-question
LongMemEval subset (hybriddb 0.5.5 vs 0.5.8, `memorycore_episodic_reranked`, k=5):
per-question results are byte-identical (0/34 metric rows differ) — the hybriddb
0.5.6+ bugfixes (journal ordering, FTS5 backfill, custom-PK support, read_query
hardening) do not change CoreMem retrieval behavior.

### Fixed
- **`coremem.__version__` drift** — `__init__.py` still reported `0.13.1` while the
  package was 0.13.2; same version-string consistency bug class hybriddb fixed in
  0.5.8. Now matches `pyproject.toml`.

### Changed
- **`hybriddb` floor raised `>=0.5.5` → `>=0.5.8`** — ensures locked/older
  environments resolve the fixed journal write path (13× faster `insert_batch`,
  chronological last-op-wins journal ordering, TEXT-PK insert rowid semantics)
  instead of only fresh installs. Full suite 174 passed, 1 skipped (mcp extra,
  pre-existing) on 0.5.8.

## [0.13.2] — Filter plumbing fixes from E2E bug hunt, lifecycle guards

E2E bug hunt (discovery → TDD fix, 12 regression tests in `tests/test_bughunt_fixes.py`, full suite 175 passed). All retrieval-path filters are now honored by every strategy; eval results are unaffected (eval harnesses call these paths without filters).

### Fixed
- **Packaging: `[project.urls]` table placement broke editable installs** — the urls table was inserted mid-`[project]`, swallowing everything below it (`requires-python`, `license`, `authors`, `dependencies`) per TOML semantics; fresh `pip install -e .` failed with `TypeError: URL 'authors' of field 'project.urls' must be a string`. Published wheels were unaffected (non-editable builds validated differently); table moved after the project fields and the lockfile was regenerated cleanly.
- **Preference queries no longer bypass recall filters** — "what do i like"-style queries routed through the preference union silently dropped `role`/`session_id`/`ts_*`/`metadata` filters and leaked other sessions' messages (results and bundles). Filters now forward through the union path and its non-preference fallback.
- **`session_cap` selection honors filters** — the cross-encoder pool re-query (`SELECT * WHERE session_id = ?`) ignored the caller's filters; a `role="user"` recall with `session_cap=2` could return assistant messages. Pool messages are now filter-checked before anchor windowing.
- **`fusion` strategy honors filters** — previously ignored all filter params silently; they now reach both the MC and ER legs (and bundles). Docstring updated: filters apply to all strategies.
- **`ingest()` returns the turn_id actually stored** — when `_get_last_turn_id` found only rows written by `store()` (`turn_id=''`), ingest stored a fresh uuid internally but returned `''`; `compile_turn` on the returned id then found nothing.
- **Use-after-close now raises** — `close()` claimed the instance is unusable afterwards but `recall()`/`ingest()` kept working (with sqlite↔vector desync risk). Public data-path methods raise `RuntimeError` after close; `close()` is idempotent.
- **`stats()` excludes the empty-session bucket** — messages with `session_id=''` were counted as a session (`list_sessions()` already excluded them).
- **MCP `ingest` surfaces invalid metadata JSON** — invalid/non-object JSON was silently dropped (message stored without metadata, no error); the tool now returns an error via new `parse_metadata_arg()`.
- **`recall()` rejects empty queries** — empty/whitespace queries on `direct` acted as match-all returning arbitrary rows; all strategies now raise `ValueError("recall requires a non-empty query")`.

### Known limitations (documented, not fixed)
- `_flush_journal_batched` embedding-cache shadow is not thread-safe under concurrent flushes.
- A real `session_id` starting with `_no_session_` is misclassified as session-less in cap selection.
- Timestamp filters compare ISO strings lexically — mixed naive/aware formats compare incorrectly.

## [0.13.1] — API polish: memory hygiene, lifecycle, wider MCP surface

### Added
- **`MemoryCore.list_sessions()`** — session inventory `{session_id, messages, last_ts}` (most recent first); replaces raw SQL in the CLI and MCP server.
- **`MemoryCore.delete_messages(message_ids)`** — remove specific messages by id (ids now appear in recall output, so agents can act on results).
- **`MemoryCore.stats()`** — `{messages, sessions, users, last_ts, journal_pending}` for health checks.
- **`MemoryCore.close()` + context manager** — releases pooled Chroma handles (unclosed handles previously left the vector store readonly in long-running processes).
- **CLI**: `stats` and `delete <message_id...>` subcommands; recall/bundles output now includes message ids.
- **MCP server widened from 5 to 8 tools** — `delete`, `fetch_session`, `stats`; `recall` accepts `role`, `ts_after`, `ts_before`, `session_cap`; every tool description carries usage examples and cost notes (e.g. `expanded` = 1 LLM call).

### Changed
- **`ingest` raises `ValueError` on empty/whitespace content** instead of silently returning `""` (bulk paths still skip empties).
- **Return conventions documented** — `ingest`/`ingest_turn` → turn_id (for `compile`); `ingest_many`/`store` → message ids.

### Removed
- Dead `SearchQuery` type (zero usages; filters are plain kwargs).

## [0.13.0] — Preference union + temporal decomposition fold, answer-eval-validated bundle defaults, batch ingest, session-cap selection

### Changed
- **Bundle budget default 16k → 4k chars with evidence-first ordering** — the winning configuration from the 500-question S answer eval (LLM answer → LLM judge, deepseek-v4-flash both roles): 4k CE-ranked bundles score **0.678 accuracy vs 0.608** for 16k bundles with ~60% less context (6,016 vs 14,744 chars). Evidence-first ordering (retrieved anchors lead each bundle, rest chronological) fixes needle-in-haystack failures — the same 15k-char context that answered 0/1 scored 1/1 after reordering.
- **Preference union-retrieval folded into the default `episodic` path** — per-variant top-40 union for preference queries (+0.033 session recall on the 30 S preference questions).
- **Temporal query decomposition in the default path** (from/to, since/when, clean ago-event cues; +0.037 session / +0.029 message recall on the 133 temporal questions).
- Combined S-scale validation (500/500) confirmed the fold: preference stacks, L-12 cancels temporal — the default ships L-6 + preference union + temporal decomposition only.

### Added
- **`MemoryCore.ingest_many()`** — batch ingestion with a single HybridDB journal flush and batched all-MiniLM embedding (content→vector cache shadowing `_get_embedding` during the flush). 550 messages: 49.9 s → 11.5 s (4.3×) with identical retrieval. All eval harnesses use it (full S-scale eval: ~2.2 h → ~23 min on 3 shards).
- **`session_cap` parameter** (`recall(..., session_cap=2)`) + eval modes `memorycore_episodic_reranked_v3` (global) / `v4` (anchor) — after the global cross-encoder rerank, every message of the top-k sessions is CE-scored and the final top-k may hold up to 2 messages per session. Full S-scale: **+0.124 message recall@5 (+0.048 answer accuracy)** at −0.058 session recall (a second message of the top session displaces the 5th session). Not folded — opt-in, documented tradeoff.
- **`scripts/eval_answer_longmemeval.py` extended** — `episodic_cap2` mode, `--reuse` for cached per-question instances, evidence-first bundle formatting.
- **`--reuse-instances` / `--reuse` flags** on the eval harnesses — re-score from persistent per-question HybridDB instances without re-ingesting. Cache guidance: ~9.5 MB/question (500 S questions ≈ 4.3 GB, Chroma fixed overhead dominates).

### Fixed
- **AML relevance gate calibration** — preference evidence was emptied by the gate.
- **Answer-eval harness deleted reused instances after scoring** — cleanup now skipped with `--reuse`, preserving the experiment cache.

### Eval results (S, 500 questions, k=5, deepseek-v4-flash answer + judge)

| Mode | Accuracy | Context chars |
|---|---:|---:|
| 4k bundles (CE, evidence-first) — new default | 0.678 | 6,016 |
| session_cap=2 | 0.656 | 11,866 |
| llm_expansion | 0.642 | 4,587 |
| 16k bundles — previous default | 0.608 | 14,744 |
| message top-5 | 0.528 | 7,302 |

Abstention accuracy 0.867 for the top modes. Retrieval metrics (session_recall@5 0.950 answerable / message_recall@5 0.617) are unchanged.

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
