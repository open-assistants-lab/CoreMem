# Known Issues and Remaining Bugs

This file documents bugs and issues in CoreMem.

## Fixed in v0.10.0

### 1. Observer stops extracting after first run (CRITICAL)

**File:** `coremem/observer.py:814`
**Root cause:** `_last_observed_id = messages[0].id` pointed to the newest message (DESC order). Next run broke immediately, processing zero new messages.
**Fix:** Changed to `messages[-1].id` (oldest processed message).

### 2. Anthropic/Gemini/OllamaCloud tool calling silently fails (HIGH)

**File:** `coremem/providers.py`
**Root cause:** All three adapters' `chat_with_tools()` either fell back to `chat()` (Anthropic, Gemini) or discarded `tool_calls` from response (OllamaCloud). Observer/Classifier/Dedup pipelines returned empty results.
**Fix:** Implemented proper tool call parsing for all three adapters.

### 3. `insert_event` with empty observation ID (MEDIUM)

**File:** `coremem/dedup.py:116`
**Root cause:** `insert_event(obs.get("id", ""), "created")` called with empty string when Observer popped the `id` field. Created orphan events.
**Fix:** Only call `insert_event` when `obs_id` is truthy.

### 4. Migration modules never called (MEDIUM)

**Files:** `coremem/migrations/v0_4_to_v0_5.py`, `v0_5_to_v0_6.py`
**Root cause:** Migration modules existed but were never imported. Upgrading from 0.4.x/0.5.x lost data.
**Fix:** Added `_run_schema_migrations()` to `MemoryCore.__init__` that runs inline migrations for all missing columns/tables.

### 5. README documents broken imports (MEDIUM)

**File:** `README.md`
**Root cause:** Showed `from coremem.backends.chroma import ChromaBackend` which was deleted in v0.6.0.
**Fix:** Updated README to reflect single-backend architecture.

### 6. Stale `backends/` directory (LOW)

**Location:** `coremem/backends/`
**Root cause:** Directory contained only stale `.pyc` files after v0.6.0 refactor.
**Fix:** Removed entirely.

### 7. Stale `__pycache__` from deleted tests (LOW)

**Location:** `tests/__pycache__/`
**Root cause:** Deleted test source files left `.pyc` artifacts.
**Fix:** Cleaned up.

## Unfixed

### 8. Flaky integration test: `test_enable_gleaning_runs_both_stages`

**File:** `tests/test_observer_gleaning.py:78`
**Issue:** Asserts `obs.get("importance") is None` but the Observer tool schema requires `importance` as a field. The LLM sometimes returns a number, sometimes null. The test assertion contradicts the tool schema.
**Impact:** Only affects integration tests hitting a real LLM. Unit tests unaffected.
