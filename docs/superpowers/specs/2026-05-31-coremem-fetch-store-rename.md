# CoreMem: Rename `export` → `fetch`, `import_batch` → `store`

## Context

CoreMem's public API has two methods with misleading names:

- **`export()`** — retrieves stored memories by exact-match column filters (role, session_id, ts range). *"Export"* suggests writing to a file, which it doesn't do. The backend internal method is already called `list()` — the confusion started there.
- **`import_batch()`** — stores a list of `Memory` objects. *"Import"* suggests reading from a file, which it doesn't do.

The pair sits awkwardly next to `search()` (semantic/fuzzy) and `ingest()` (single-record store):

| Method | What it does | Problem |
|--------|-------------|---------|
| `ingest()` | Store one message | OK |
| `search()` | Semantic + keyword search | OK |
| `export()` | Retrieve by exact filter | Sounds like file I/O |
| `import_batch()` | Store many Memory objects | Sounds like file I/O |

## Summary

Rename `MemoryCore.export()` → `MemoryCore.fetch()` to convey "fetch me memories matching these criteria."

Rename `MemoryCore.import_batch()` → `MemoryCore.store()` to convey "store these into memory."

Treat `export_all()` as a convenience that calls `fetch()` with pagination — rename to `fetch_all()`.

## Changes

### `MemoryCore`

| Before | After |
|--------|-------|
| `core.export(role=..., session_id=..., limit=100)` | `core.fetch(role=..., session_id=..., limit=100)` |
| `core.export_all(...)` | `core.fetch_all(...)` |
| `core.import_batch([mem1, mem2])` | `core.store([mem1, mem2])` |

### `StoreBackend` (ABC)

| Before | After |
|--------|-------|
| `backend.list(...)` | unchanged (internal name is fine) |
| `backend.ingest_batch(...)` | unchanged (matches `ingest` pattern) |

Only the `MemoryCore` public API changes. Backend method names stay as-is.

## Callers

### CoreMem tests (4 files)

```
tests/test_backend_hybrid.py:       core.export(...)
tests/test_backend_hybrid.py:       core.import_batch(...)
tests/test_backend_chroma.py:       core.export(...)
tests/test_backend_chroma.py:       core.import_batch(...)
tests/test_core.py:                 core.export(...)
tests/test_core.py:                 core.import_batch(...)
tests/test_hybrid_v1_tests.py:      backend.list(...)   # internal — unchanged
```

### EA (5 files)

```
src/storage/messages.py:            self._core.export(...)      # 8 call sites
src/storage/messages.py:            self._core.export_all(...)  # 1 call site (count_messages)
src/storage/messages.py:            self._core.import_batch  — NOT used currently
src/conversation/import_export.py:  core.import_batch(...)
src/http/routers/conversation.py:   core.export_messages  — different method, unaffected
tests/storage/test_messages.py:     store._core.export(...)
```

Total: ~15 call sites across CoreMem tests + EA source + EA tests.

## Migration

Simple find-and-replace across each repo:

```
# CoreMem
core.export(          →  core.fetch(
core.export_all(      →  core.fetch_all(
core.import_batch(    →  core.store(

# EA (after CoreMem release)
self._core.export(    →  self._core.fetch(
self._core.export_all( →  self._core.fetch_all(
.import_batch(        →  .store(
```
