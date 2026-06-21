# Known Issues and Remaining Bugs

This file documents bugs and issues that remain unfixed in CoreMem.

## Critical Bugs

### 1. Observer stops extracting after first run

**File:** `coremem/observer.py:814`
**Status:** FIXED in v0.10.0

`_last_observed_id = messages[0].id` pointed to the newest message (DESC order), causing the next run to break immediately. Fixed to `messages[-1].id`.

## High-Severity Bugs

### 2. Anthropic/Gemini/OllamaCloud tool calling silently fails

**Files:** `coremem/providers.py`
**Status:** FIXED in v0.10.0

All three adapters' `chat_with_tools()` either fell back to `chat()` (Anthropic, Gemini) or discarded `tool_calls` from the response (OllamaCloud). The Observer, Classifier, and Dedup pipelines rely on tool calling — without it, they return empty results.

Fixed by implementing proper tool call parsing for all three adapters.

## Medium-Severity Bugs

### 3. `insert_event` called with empty observation ID in dedup

**File:** `coremem/dedup.py:116`
**Status:** FIXED in v0.10.0

When dedup runs with no prior candidates (early return path), `insert_event(obs.get("id", ""), "created")` is called with an empty string. Since Observer removes the `id` field from observations (`observer.py:729`), this creates orphan events. Fixed to only call `insert_event` when `obs_id` is truthy.

### 4. Migration modules are dead code

**Files:** `coremem/migrations/v0_4_to_v0_5.py`, `v0_5_to_v0_6.py`
**Status:** PARTIALLY FIXED in v0.10.0

The migration modules existed but were never imported or called. Upgrading from 0.4.x or 0.5.x would silently lose data.

Fixed by wiring inline schema migrations into `MemoryCore._run_schema_migrations()`, which runs on every `__init__` and adds any missing columns/tables. The separate migration modules in `migrations/` are still dead code and should be cleaned up.

### 5. README documents broken import paths

**File:** `README.md`
**Status:** FIXED in v0.10.0

The README showed `from coremem.backends.chroma import ChromaBackend` which no longer exists (deleted in v0.6.0). Fixed by updating README to reflect the single-backend architecture.

## Low-Severity Issues

### 6. Stale `backends/` directory

**Location:** `coremem/backends/`
**Status:** FIXED in v0.10.0

The directory contained only stale `.pyc` files from before the v0.6.0 refactor. Removed entirely.

### 7. Stale `__pycache__` from deleted test files

**Location:** `tests/__pycache__/`
**Status:** FIXED in v0.10.0

`test_memory_store.py` and `test_migration_0_4_to_0_5.py` were deleted but their `.pyc` files remained. Cleaned up.

### 8. Flaky integration test: `test_enable_gleaning_runs_both_stages`

**File:** `tests/test_observer_gleaning.py:78`
**Status:** UNFIXED

The test asserts `obs.get("importance") is None` but the Observer tool schema requires `importance` as a field. The LLM sometimes returns a number, sometimes null. This is a pre-existing test bug — the assertion should be removed or changed to accept any value.

### 9. OllamaCloud `chat_with_tools` ignores `tool_calls` in response

**File:** `coremem/providers.py:_OllamaCloudAdapter.chat_with_tools`
**Status:** FIXED in v0.10.0

The method sent tools in the request but only read `message.content` from the response, discarding `message.tool_calls`. Fixed to parse and return tool calls in OpenAI-compatible format.

## Notable Design Decisions

### No parallel tool execution

The AgentLoop executes tools sequentially. This was a deliberate design decision (see AGENTS.md). Parallel execution is deferred.

### ToolAnnotations.auto_approval only works for non-destructive tools

A tool that is both `destructive` AND `read_only` won't interrupt (read-only wins). This is intentional — a read-only destructive tool is a contradiction that defaults to safe.

### Models.dev registry uses lazy loading

The registry fetches from `https://models.dev/api.json` on first access, caches locally with 5-min TTL, falls back to built-in subset. Never hardcode model info.
