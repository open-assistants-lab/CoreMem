# Known Issues and Remaining Bugs

This file documents bugs and issues in CoreMem. Entries reference the code as
it exists today; historical entries from removed subsystems are marked as such.

## Fixed

### 1. Recency heuristics silently no-op on aware timestamps (HIGH, fixed)

**File:** `coremem/heuristics.py` (`recency_decay`, `temporal_boost`)
**Root cause:** `datetime.now()` (naive) minus a timezone-aware `ts` raises
`TypeError`, swallowed by the `except (ValueError, TypeError)` clause. All
timestamps stored by `_ingest_message` are aware (`datetime.now(UTC).isoformat()`),
so both heuristics returned the score unchanged — the README's "recency-aware
rescoring" never fired. Zero test coverage (tests passed no `ts`).
**Fix:** Compare with `datetime.now(UTC)`; naive timestamps are treated as UTC.
Regression tests: `test_recency_decay_*`, `test_temporal_boost_*`,
`test_apply_all_applies_recency_with_aware_timestamp` in `tests/test_heuristics.py`.

### 2. `COREMEM_LLM_MODEL` did not configure the compile model (HIGH, fixed)

**Files:** `coremem/__init__.py`, `coremem/__main__.py`, `coremem/mcp_server.py`
**Root cause:** `get_core()` passed `llm_provider` to `MemoryCore`, but
`compile_turn` uses `_journal_compiler`, built with the hardcoded
`DEFAULT_AGENT_JOURNAL_MODEL` (`openai:gpt-4o-mini`). The env var was ignored
for compilation, and the CLI/MCP guard checked the unrelated `_llm_provider`
(which is only used for query expansion) — refusing to compile even when
`OPENAI_API_KEY` was set, and compiling with the wrong model when
`COREMEM_LLM_MODEL` was set.
**Fix:** `get_core()` passes `agent_journal_model`; CLI/MCP compile wraps the
call in try/except and reports failures cleanly.
Regression tests: `test_get_core_passes_coremem_llm_model_to_journal_compiler`,
`test_get_core_default_journal_model_when_env_unset`, `test_cli_compile_reports_failure_without_traceback`.

### 3. Bundle budget dropped the anchor message (HIGH, fixed)

**File:** `coremem/core.py` (`_reconstruct_sessions`)
**Root cause:** In the budgeted path, the priority pass skipped any message
that exceeded the per-bundle budget — including the anchor (the retrieved
evidence itself). A long anchor (> `max_context_chars // n_bundles`) was
silently dropped from the bundle, and the fill loop could not recover it.
**Fix:** Anchors always survive the budget; only the opening message and fill
context are best-effort.
Regression test: `test_reconstruct_sessions_keeps_anchor_when_exceeding_budget`.

### 4. Observer-era bugs (historical — code removed in v0.10.0)

The following were fixed in the observer/reflector pipeline, which was
deleted in v0.10.0 (replaced by the deterministic AgentJournal compiler).
Kept for history:

- **Observer stops extracting after first run (CRITICAL)** — `_last_observed_id`
  pointed to the newest message (DESC order); next run processed zero messages.
- **Anthropic/Gemini/OllamaCloud tool calling silently failed (HIGH)** — all
  three adapters fell back to `chat()` or discarded `tool_calls`.
- **`insert_event` with empty observation ID (MEDIUM)** — orphan events.
- **Migration modules never called (MEDIUM)** — upgrades lost data.
- **README documented broken imports (MEDIUM)** — `coremem.backends.chroma`.
- **Stale `backends/` directory and `__pycache__` artifacts (LOW)**.

## Unfixed

### 5. `fetch()` metadata filter crashes on keys with quotes (MEDIUM)

**File:** `coremem/core.py` (`fetch`)
**Root cause:** `json_extract(metadata, '$.{k}')` interpolates the metadata key
unescaped. A key containing `'` raises `OperationalError: incomplete input`;
a key with a space silently returns zero rows. Not a full SQL injection
(parameter-count mismatch blocks it), but a crash + silent-wrong-results bug.
`delete()` also lacks the `metadata` filter that `fetch()` has (API inconsistency).

### 6. Score normalization in `expanded` strategy is dead code (MEDIUM)

**File:** `coremem/core.py` (`_search_messages_llm_expansion`)
**Root cause:** HybridDB returns `_score`, but max/min are computed from
`r.get("score", 0)` — always 0 — so `score_range == 0` and the per-query
normalization that was meant to balance merged variants never runs.

### 7. `dream()` cursor advances past failed chunks (MEDIUM)

**File:** `coremem/agent_journal/dreaming.py`
**Root cause:** `_write_cursor(pending[-1])` runs even when a chunk failed
(LLM error / invalid output). Failed dates are never retried — silent data
loss on transient failures.

### 8. `MEMORY.md` and `index.md` have conflicting writers (MEDIUM)

**Files:** `coremem/agent_journal/compiler.py`, `coremem/agent_journal/dreaming.py`,
`coremem/agent_journal/rebuild_index.py`
**Root cause:** `compiler._write_memory()` overwrites `MEMORY.md` (regenerated
from boot_worthy pages) while `dream()` appends promoted facts to the same
file; `compiler._write_index()` writes page links while `rebuild_index()`
writes month links to the same `index.md`. Last writer wins.

### 9. `recall(strategy="fusion")` silently ignores filter params (LOW)

**File:** `coremem/core.py` (`recall`)
**Root cause:** `_search_with_fusion` has no filter parameters; `role`,
`session_id`, etc. are silently dropped. Documented in the docstring, but a
footgun.

### 10. Zombie modules from the removed observer pipeline (LOW)

**Files:** `coremem/dedup.py`, `coremem/classifier.py`, `coremem/grounding.py`
**Root cause:** Dead code since v0.10.0, kept alive only by their own tests
(`tests/test_dedup.py`, `tests/test_classifier.py`, `tests/test_grounding.py`).
`_OpenAIAdapter.chat_with_tools` also sends the DeepSeek-specific
`"thinking": {"type": "disabled"}` body param to every OpenAI-compatible
endpoint — latent, since only the dead code calls it.

### 11. Non-deterministic `hash()` usage (LOW)

**Files:** `coremem/heuristics.py` (`_mmr_diversify`), `coremem/core.py`
(`_search_messages_llm_expansion`)
**Root cause:** Built-in `hash()` is salted per process (PYTHONHASHSEED), so
session-less dedup keys differ across runs — eval resume can give different
results.

### 12. `EmbeddingIndex` dot product vs cosine (LOW)

**File:** `coremem/agent_journal/embeddings.py`
**Root cause:** `search()` does a raw `np.dot` while the docstring says cosine
similarity (vectors are never normalized). Also, `_page_ids = [p.stem ...]`
collides for nested page dirs, which the compiler explicitly supports
(`"a.b"` → `pages/a/b.md`).

### 13. `_post_with_retry` crashes on HTTP-date `Retry-After` (LOW)

**File:** `coremem/providers.py` (`_OllamaCloudAdapter`)
**Root cause:** `int(server_hint)` raises `ValueError` if `Retry-After` is an
HTTP-date rather than seconds.

### 14. `_fix_source_quote` sanitization skips short quotes (LOW)

**File:** `coremem/agent_journal/llm_compiler.py`
**Root cause:** The final sanitize step only applies when the cleaned quote is
≥ 10 chars; shorter quotes keep invalid characters and fail `_require_quote`,
burning retries.

### 15. `expand_queries("how many")` emits an empty variant (LOW)

**File:** `coremem/query.py`
**Root cause:** `_regex_expand_queries` appends `""` when the query is exactly
a keyword like "how many". Harmless today (HybridDB handles empty queries),
but fragile.
