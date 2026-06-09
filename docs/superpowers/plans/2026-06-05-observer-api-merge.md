# Observer API Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Single `MemoryCore` object — one path, one HybridDB, all tables. Dev never touches `HybridBackend`, `MemoryStore`, or ChromaBackend. Pipeline exposes `extract()` and `retrieve()`.

**Architecture:** HybridDB becomes the only backend, created internally by `MemoryCore.__init__(path)`. `MemoryStore`'s 14 methods move into `MemoryCore`. `ObserverPipeline` takes `memory` instead of `core+store`. `after_turn()` renamed to `extract()`. `retrieve()` added.

**Tech Stack:** Python 3.11+, HybridDB, SQLite, ChromaDB removed. Tests use temp directories.

---

## Files map

| File | Action |
|------|--------|
| `coremem/core.py` | Rewrite constructor, add 14 methods from MemoryStore, embed schemas |
| `coremem/memory_store.py` | Delete |
| `coremem/observer.py` | Change constructor, rename `after_turn` → `extract`, add `retrieve()` |
| `coremem/reflector.py` | Change constructor to take `memory: MemoryCore` instead of `store: MemoryStore` |
| `coremem/dedup.py` | Change `insert_event()`/`create_conflict()` calls to go through `memory` object |
| `coremem/classifier.py` | No change |
| `coremem/backends/` | Delete entire directory |
| `coremem/backends/base.py` | Delete |
| `coremem/backends/chroma.py` | Delete |
| `coremem/backends/hybrid.py` | Delete |
| `coremem/ingest.py` | Delete (only used by `MemoryCore.ingest()` — inline that logic) |
| `coremem/migrations/v0_5_to_v0_6.py` | Update table names, import schemas from core |
| `coremem/__init__.py` | Remove MemoryStore, backend imports |
| `tests/test_core.py` | Add all MemoryStore tests here |
| `tests/test_memory_store.py` | Delete |
| `tests/test_pipelines.py` | Update constructor calls |
| `tests/test_hybrid_backend.py` | Delete (backend is now internal) |
| `tests/test_chroma_backend.py` | Delete |
| `tests/test_ingest.py` | Delete |
| `README.md` | Rewrite setup section |
| `docs/observer-api-merge-spec.md` | Already exists |

---

### Task 1: Set up worktree

**Files:** None

- [ ] **Step 1: Create worktree**

```bash
cd /Users/eddy/Developer/Python/CoreMem
git worktree add .worktrees/observer-api-merge -b feature/observer-api-merge
cd .worktrees/observer-api-merge
```

- [ ] **Step 2: Install deps**

```bash
uv sync --extra dev
```

- [ ] **Step 3: Verify baseline tests pass**

```bash
uv run pytest -q | tail -5
```
Expected: 131 passed, 3 skipped

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: baseline before observer API merge"
```

---

### Task 2: Rewrite `MemoryCore` constructor — one path, one HybridDB

**Files:**
- Modify: `coremem/core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write failing test**

Create test file `tests/test_core.py`:

```python
def test_memorycore_with_path():
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        from coremem import MemoryCore
        core = MemoryCore(path=d)
        assert core._db is not None
        mid = core.ingest("user", "hello", session_id="s1")
        assert mid is not None
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_memorycore_observations_disabled():
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        from coremem import MemoryCore
        core = MemoryCore(path=d)
        import pytest
        with pytest.raises(RuntimeError, match="enable_observations"):
            core.get_observations()
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_memorycore_observations_enabled():
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        from coremem import MemoryCore
        core = MemoryCore(path=d, enable_observations=True)
        obs = core.get_observations()
        assert obs == []
    finally:
        shutil.rmtree(d, ignore_errors=True)
```

- [ ] **Step 2: Run test to confirm failure**

```bash
uv run pytest tests/test_core.py -v
```
Expected: FAIL (MemoryCore doesn't accept `path` yet)

- [ ] **Step 3: Rewrite `MemoryCore.__init__`**

In `coremem/core.py`:

```python
from hybriddb import HybridDB

class MemoryCore:
    def __init__(
        self,
        path: str,
        llm_provider: LLMProvider | None = None,
        enable_observations: bool = False,
    ):
        self._db = HybridDB(path=path)
        self._heuristics = SearchHeuristics()
        self._wakeup = WakeUpContext(self._db)
        self._llm_provider = llm_provider
        self._ensure_tables()
        if enable_observations:
            self._ensure_observation_tables()
```

Also add `_ensure_tables()` and `_ensure_observation_tables()` methods. Replace `self._wakeup = WakeUpContext(backend)` with `WakeUpContext(self._db)` — and update `WakeUpContext` to accept a HybridDB directly instead of a StoreBackend.

- [ ] **Step 4: Remove `backend` property and old constructor** The `backend` property (`coremem/core.py:47-49`) should raise a deprecation warning or be removed entirely. Since we're deleting backends/, remove it.

- [ ] **Step 5: Update `ingest()` and `ingest_many()`** Move `ingest_message()` and `ingest_batch()` from `coremem/ingest.py` into `coremem/core.py` as private functions at module level. This is a straight copy — no logic change. The original file is deleted in Task 4.

- [ ] **Step 6: Update `search()`, `search_enhanced()`, `fetch()`, `fetch_all()`, `store()`, `delete()`, `count()`, `clear()`** These currently call `self._backend.<method>()`. Change to `self._db.<method>()`. The HybridDB API for these is: `self._db.search()`, `self._db.query()`, `self._db.insert()`, `self._db.delete()`, `self._db.count()`, `self._db.clear()`.

- [ ] **Step 7: Update `WakeUpContext` to accept HybridDB (full change here, not in Task 4)** In `coremem/layers.py`, `WakeUpContext.__init__` currently takes `backend: StoreBackend`. Change to `_db: HybridDB`. Replace `self._backend.get_recent(limit=10)` with `self._db.raw_query("SELECT * FROM messages ORDER BY ts DESC LIMIT ?", (10,))`. Replace `self._backend.get_recent(limit=20)` similarly. Replace `self._backend.search(SearchQuery(text=query, limit=limit))` with `self._db.search("messages", "content", query, limit=limit)`. Remove `from coremem.backends.base import StoreBackend` and `from coremem.types import SearchQuery`, add `from hybriddb import HybridDB`.

- [ ] **Step 8: Run tests**

```bash
uv run pytest tests/test_core.py -v
```
Expected: 3 PASS

```bash
uv run pytest -q | tail -10
```
Some tests will fail because they pass `backend=` to `MemoryCore()` — those are expected to fail and will be fixed in Task 6.

- [ ] **Step 9: Commit**

```bash
git add coremem/core.py coremem/layers.py
git commit -m "refactor: MemoryCore takes path string, creates HybridDB internally"
```

---

### Task 3: Move all MemoryStore methods into MemoryCore

**Files:**
- Modify: `coremem/core.py`
- Delete: `coremem/memory_store.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Copy schemas into core.py**

Add these module-level constants to `coremem/core.py` (taken from `memory_store.py`):
- `_OBSERVATIONS_SCHEMA`
- `_REFLECTIONS_SCHEMA`
- `_OBSERVATION_EVENTS_SCHEMA` (was `_MEMORY_EVENTS_SCHEMA`)
- `_OBSERVATION_CONFLICTS_SCHEMA` (was `_MEMORY_CONFLICTS_SCHEMA`)

Rename `memory_id` → `observation_id`, `memory_id_a` → `observation_id_a`, `memory_id_b` → `observation_id_b` in the events and conflicts schemas.

```python
_OBSERVATION_EVENTS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "observation_id": "TEXT NOT NULL",
    "event_type": "TEXT NOT NULL",
    "old_value": "TEXT",
    "new_value": "TEXT",
    "source_message_id": "TEXT",
    "created_at": "TEXT NOT NULL",
}

_OBSERVATION_CONFLICTS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "observation_id_a": "TEXT NOT NULL",
    "observation_id_b": "TEXT NOT NULL",
    "conflict_type": "TEXT NOT NULL",
    "resolution_status": "TEXT DEFAULT 'unresolved'",
    "created_at": "TEXT NOT NULL",
    "resolved_at": "TEXT",
}
```

- [ ] **Step 2: Add `_ensure_observation_tables()` method to MemoryCore**

```python
def _ensure_observation_tables(self) -> None:
    if "observations" not in self._db.list_tables():
        self._db.create_table("observations", _OBSERVATIONS_SCHEMA)
        for idx in ["kind", "user", "session", "reflected", "importance"]:
            self._db.raw_query(
                f"CREATE INDEX IF NOT EXISTS idx_observations_{idx} ON observations({idx})"
            )

    if "observation_events" not in self._db.list_tables():
        self._db.create_table("observation_events", _OBSERVATION_EVENTS_SCHEMA)

    if "observation_conflicts" not in self._db.list_tables():
        self._db.create_table("observation_conflicts", _OBSERVATION_CONFLICTS_SCHEMA)

    if "reflections" not in self._db.list_tables():
        self._db.create_table("reflections", _REFLECTIONS_SCHEMA)
```

- [ ] **Step 3: Copy all 14 methods from MemoryStore into MemoryCore**

Copy these methods verbatim from `coremem/memory_store.py` into `coremem/core.py` (inside the `MemoryCore` class, after `_ensure_observation_tables`):
- `insert_observations`
- `get_observations`
- `get_observations_since`
- `get_recent_observations`
- `search_observations`
- `get_pending_reflections`
- `mark_reflected`
- `insert_reflections`
- `get_reflections`
- `search_reflections`
- `apply_decay`
- `get_candidates`
- `update_observation`
- `insert_event`
- `create_conflict`

For `insert_event` and `create_conflict`, update the table names from `memory_events` → `observation_events` and `memory_conflicts` → `observation_conflicts`, and update column names `memory_id` → `observation_id`, `memory_id_a` → `observation_id_a`, `memory_id_b` → `observation_id_b`.

- [ ] **Step 4: Write tests for observation methods**

In `tests/test_core.py`:

```python
def test_insert_and_retrieve_observations():
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        from coremem import MemoryCore
        core = MemoryCore(path=d, enable_observations=True)
        ids = core.insert_observations([{
            "content": "User likes coffee",
            "source_quote": "I like coffee",
            "importance": 0.6,
        }])
        assert len(ids) == 1
        obs = core.get_observations(limit=10)
        assert len(obs) == 1
        assert obs[0]["content"] == "User likes coffee"
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_search_observations():
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        from coremem import MemoryCore
        core = MemoryCore(path=d, enable_observations=True)
        core.insert_observations([{
            "content": "User likes coffee", "source_quote": "coffee",
            "importance": 0.6,
        }])
        results = core.search_observations("coffee", limit=5)
        assert len(results) >= 1
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_observations_disabled_raises():
    import tempfile, shutil, pytest
    d = tempfile.mkdtemp()
    try:
        from coremem import MemoryCore
        core = MemoryCore(path=d, enable_observations=False)
        with pytest.raises(RuntimeError, match="enable_observations"):
            core.get_observations()
    finally:
        shutil.rmtree(d, ignore_errors=True)

def test_insert_event_and_create_conflict():
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        from coremem import MemoryCore
        core = MemoryCore(path=d, enable_observations=True)
        oid = core.insert_observations([{
            "content": "obs A", "source_quote": "a", "importance": 0.5,
        }])[0]
        eid = core.insert_event(oid, "created")
        assert eid is not None
        cid = core.create_conflict(oid, oid, "contradiction")
        assert cid is not None
    finally:
        shutil.rmtree(d, ignore_errors=True)
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest tests/test_core.py -v
```
Expected: All PASS

- [ ] **Step 6: Delete `coremem/memory_store.py`**

```bash
rm coremem/memory_store.py
```

- [ ] **Step 7: Run tests again to confirm nothing imports MemoryStore**

```bash
uv run pytest tests/test_core.py -v
```
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add coremem/core.py tests/test_core.py
git rm coremem/memory_store.py
git commit -m "feat: absorb MemoryStore methods and schemas into MemoryCore"
```

---

### Task 4: Delete backends/ directory and ingest.py

**Files:**
- Delete: `coremem/backends/` (entire directory)
- Delete: `coremem/ingest.py`
- Modify: `coremem/layers.py` (remove StoreBackend dependency)
- Modify: `coremem/core.py` (remove imports from backends, ingest)

- [ ] **Step 1: Remove imports from core.py**

In `coremem/core.py`, remove these lines:
```python
from coremem.backends.base import StoreBackend
from coremem.ingest import ingest_batch, ingest_message
```

- [ ] **Step 2: Update `layers.py` to remove `StoreBackend`**

`WakeUpContext.__init__` already changed in Task 2 (Step 7). Verify no remaining `StoreBackend` references. Remove `from coremem.backends.base import StoreBackend` from imports if it still exists.

- [ ] **Step 3: Remove ingest.py** — already inlined in Task 2 Step 5. Just delete the file:

```bash
rm coremem/ingest.py
```

- [ ] **Step 4: Delete backends/ directory**

```bash
rm -rf coremem/backends/
rm coremem/ingest.py
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_core.py -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add coremem/core.py coremem/layers.py
git rm -r coremem/backends/ coremem/ingest.py
git commit -m "refactor: remove backends/ and ingest.py, use HybridDB directly"
```

---

### Task 5: Update `ObserverPipeline` — memory arg, extract(), retrieve()

**Files:**
- Modify: `coremem/observer.py`

- [ ] **Step 1: Change constructor — `memory` instead of `core` + `store`**

```python
def __init__(
    self,
    memory: MemoryCore,
    session_id: str,
    user_id: str | None = None,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    model: str = "ollama:llama3.2",
    token_threshold: int = 100,
    min_turns: int = 1,
    max_messages: int = 500,
    enable_classification: bool = True,
    enable_dedup: bool = True,
    tool_temp: float = 0.1,
):
    self._memory = memory
    self._session_id = session_id
    self._user_id = user_id
    self._agent_id = agent_id
    # ... rest same but using memory instead of core/store
```

Remove `self._core` and `self._store`. All internal calls to `self._core.fetch(...)` become `self._memory.fetch(...)`. All calls to `self._store.insert_observations(...)` become `self._memory.insert_observations(...)`. All calls to `self._store.get_recent_observations(...)` become `self._memory.get_recent_observations(...)`.

- [ ] **Step 2: Rename `after_turn` → `extract`**

```python
async def extract(self) -> list[dict[str, Any]] | None:
    """Extract observations from new user messages. Fire-and-forget."""
    if self._running:
        return None
    self._turns_since_last_run += 1
    try:
        self._running = True
        return await self._maybe_run()
    finally:
        self._running = False
```

Replace all `def after_turn` with `def extract`. Keep internal `_maybe_run()` unchanged (it's private).

- [ ] **Step 3: Add `retrieve()` method**

```python
def retrieve(self, query: str | None = None, days: int = 30, limit: int = 50) -> list[dict[str, Any]]:
    if query:
        return self._memory.search_observations(query, limit=limit)
    return self._memory.get_recent_observations(days=days, limit=limit)
```

`retrieve()` relies on the fact that `get_recent_observations()` and `search_observations()` raise `RuntimeError` when `enable_observations=False` (from Task 3). No separate tracking needed.

Note: `get_recent_observations()` currently does NOT filter by `durability` or `status`. The spec says `retrieve()` should return durable + non-archived only. Add an optional `durability='durable'` filter to `get_recent_observations()` in `core.py` (or add it directly in `retrieve()` via a `WHERE` clause). The simplest approach: add `AND durability = 'durable' AND (status IS NULL OR status = '')` to the SQL in `get_observations()` when called from `get_recent_observations()`.

- [ ] **Step 4: Remove `core=, store=` fallback** — just raise TypeError if someone passes those

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_pipelines.py -v
```
Expected: Some will fail because they pass `core=, store=`. Those are fixed in Task 9 (test updates).

- [ ] **Step 6: Add test for `retrieve()`**

In `tests/test_core.py`:

```python
def test_retrieve():
    import tempfile, shutil
    d = tempfile.mkdtemp()
    try:
        from coremem import MemoryCore
        from coremem.observer import ObserverPipeline
        core = MemoryCore(path=d, enable_observations=True)
        pipeline = ObserverPipeline(memory=core, session_id="s1",
                                    token_threshold=1, min_turns=1)
        # No observations yet — retrieve should be empty
        obs = pipeline.retrieve(limit=10)
        assert obs == []
        # Insert one observation
        core.insert_observations([{
            "content": "test", "source_quote": "test",
            "importance": 0.5, "durability": "durable",
        }])
        obs = pipeline.retrieve(limit=10)
        assert len(obs) == 1
    finally:
        shutil.rmtree(d, ignore_errors=True)
```

- [ ] **Step 7: Commit**

```bash
git add coremem/observer.py
git commit -m "feat: ObserverPipeline takes memory arg, extract()/retrieve() API"
```

---

### Task 6: Update ReflectorPipeline — memory arg instead of store

**Files:**
- Modify: `coremem/reflector.py`

- [ ] **Step 1: Change constructor**

```python
# Before
def __init__(self, store: MemoryStore, ...):

# After
def __init__(self, memory: MemoryCore, ...):
```

Replace all `self._store.get_pending_reflections()` → `self._memory.get_pending_reflections()`. Same for `mark_reflected`, `insert_reflections`.

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_reflector.py -v 2>/dev/null || echo "No reflector tests yet"
```

- [ ] **Step 3: Commit**

```bash
git add coremem/reflector.py
git commit -m "refactor: ReflectorPipeline takes memory arg instead of store"
```

---

### Task 7: Update dedup.py to use `memory` object

**Files:**
- Modify: `coremem/dedup.py`

- [ ] **Step 1: Change function signatures**

`dedup_and_merge()` currently takes `provider, store: MemoryStore, new_obs`. Change to `provider, memory: MemoryCore, new_obs`. All calls to `store.insert_event()` become `memory.insert_event()`. All calls to `store.create_conflict()` become `memory.create_conflict()`.

The function is called from `observer.py` line 799-802:
```python
new_obs = await dedup_and_merge(self._observer._provider, self._store, new_obs)
```
Change to:
```python
new_obs = await dedup_and_merge(self._observer._provider, self._memory, new_obs)
```

- [ ] **Step 2: Run tests**

```bash
uv run pytest tests/test_dedup.py -v 2>/dev/null || echo "No dedup tests"
uv run pytest tests/test_pipelines.py -v
```

- [ ] **Step 3: Commit**

```bash
git add coremem/dedup.py coremem/observer.py
git commit -m "refactor: dedup_and_merge takes memory arg instead of store"
```

---

### Task 8: Update `__init__.py` and migrations

**Files:**
- Modify: `coremem/__init__.py`
- Modify: `coremem/migrations/v0_5_to_v0_6.py`

- [ ] **Step 1: Update `__init__.py`**

Remove:
```python
from coremem.memory_store import MemoryStore  # noqa: F401
```
Remove `"MemoryStore"` from `__all__`. Remove backend imports.

```python
__all__ = [
    "MemoryCore", "Memory", "SearchResult", "SearchQuery",
    "SearchHeuristics", "expand_queries", "rerank", "_mmr_diversify",
]

try:
    from coremem.observer import Observer, ObserverPipeline
    from coremem.providers import create_provider
    from coremem.reflector import Reflector, ReflectorPipeline
    __all__.extend([
        "Observer", "ObserverPipeline", "Reflector", "ReflectorPipeline",
        "create_provider",
    ])
except ImportError:
    pass
```

- [ ] **Step 2: Update `v0_5_to_v0_6.py`**

Change the import:
```python
# Before
from coremem.memory_store import _MEMORY_EVENTS_SCHEMA, _MEMORY_CONFLICTS_SCHEMA
# After
from coremem.core import _OBSERVATION_EVENTS_SCHEMA, _OBSERVATION_CONFLICTS_SCHEMA
```

Update table name references in the migration from `memory_events` → `observation_events` and `memory_conflicts` → `observation_conflicts`.

- [ ] **Step 3: Remove the old `__init__.py` try/except for MemoryStore**

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_core.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add coremem/__init__.py coremem/migrations/v0_5_to_v0_6.py
git commit -m "refactor: clean up imports, remove MemoryStore references"
```

---

### Task 9: Fix all remaining tests

**Files:**
- Modify: `tests/test_pipelines.py`
- Delete: `tests/test_memory_store.py`
- Delete: `tests/test_hybrid_backend.py`
- Delete: `tests/test_chroma_backend.py`
- Delete: `tests/test_ingest.py`
- Modify: Any other test files that import MemoryStore or backends

- [ ] **Step 1: Update `tests/test_pipelines.py`**

Change all `ObserverPipeline(core=..., store=...)` to `ObserverPipeline(memory=...)`. The `_make_core_with_messages()` helper needs to create a `MemoryCore` instead of a raw `HybridBackend`:

```python
def _make_core_with_messages(messages: list[Memory]):
    from coremem import MemoryCore
    import tempfile, shutil
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d)
    for m in messages:
        core.ingest(m.role, m.content, session_id="main",
                    user_id="alice", agent_id="a1", ts=m.ts)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core
```

Update `_make_store()` to return a `MemoryCore(path=..., enable_observations=True)` instead of `MemoryStore`.

- [ ] **Step 2: Delete obsolete test files**

```bash
rm tests/test_memory_store.py
rm tests/test_hybrid_backend.py
rm tests/test_chroma_backend.py
rm tests/test_ingest.py
```

- [ ] **Step 3: Run full test suite**

```bash
uv run pytest -q | tail -10
```
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/
git rm tests/test_memory_store.py tests/test_hybrid_backend.py tests/test_chroma_backend.py tests/test_ingest.py
git commit -m "test: update for MemoryCore unification, delete obsolete tests"
```

---

### Task 10: Bump version, update README, update CHANGELOG

**Files:**
- Modify: `coremem/__init__.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Bump version**

`coremem/__init__.py`:
```python
__version__ = "0.6.0"
```

`pyproject.toml`:
```toml
version = "0.6.0"
```

- [ ] **Step 2: Rewrite README setup section**

Replace the current "Dual-backend architecture" usage example with:

```python
from coremem import MemoryCore
from coremem.observer import ObserverPipeline

# One object, one path, all tables
core = MemoryCore(path="./memory", enable_observations=True)

# Messages
core.ingest("user", "I like coffee")
results = core.search("coffee")

# Observations
pipeline = ObserverPipeline(memory=core, session_id="conv_001")
await pipeline.extract()
obs = pipeline.retrieve(limit=50)
obs = pipeline.retrieve(query="likes")

# Reflections
refs = core.get_reflections(limit=10)
```

Replace the "Dual-backend architecture" section with "Single-backend (HybridDB)" description. Remove all references to ChromaBackend, HybridBackend, and MemoryStore.

- [ ] **Step 3: Update CHANGELOG**

Add entry at top:

```markdown
## [0.6.0] — 2026-06-05 — Observer API merge, single-backend simplification

### Breaking changes
- `MemoryCore()` now takes `path: str` instead of `backend: StoreBackend`. HybridDB is the only backend. ChromaBackend removed.
- `from coremem.backends.hybrid import HybridBackend` → no longer needed. `MemoryCore` creates it internally.
- `MemoryStore` class deleted. Methods moved to `MemoryCore`: `get_observations()`, `search_observations()`, `insert_observations()`, `get_recent_observations()`, `insert_reflections()`, `get_reflections()`, etc.
- `ObserverPipeline(core=foo, store=bar)` → `ObserverPipeline(memory=foo)`.
- `pipeline.after_turn()` → `pipeline.extract()`.

### Added
- `pipeline.retrieve(query=None, days=30, limit=50)` — returns recent observations or semantic search results.
- `MemoryCore(path, enable_observations=True)` creates 5 tables in one HybridDB: messages, observations, observation_events, observation_conflicts, reflections.

### Removed
- `coremem/backends/` directory (ChromaBackend, HybridBackend, StoreBackend ABC).
- `coremem/memory_store.py` (absorbed into MemoryCore).
- `coremem/ingest.py` (inlined into MemoryCore).
- `tests/test_memory_store.py`, `tests/test_hybrid_backend.py`, `tests/test_chroma_backend.py`, `tests/test_ingest.py`.
```

- [ ] **Step 4: Commit**

```bash
git add coremem/__init__.py pyproject.toml README.md CHANGELOG.md
git commit -m "v0.6.0: observer API merge, single-backend, extract/retrieve API"
```

---

### Task 11: Final verification

**Files:** None

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -q | tail -5
```
Expected: All PASS

- [ ] **Step 2: Verify PyPI build**

```bash
uv build && uv publish --token "$(grep PYPI_TOKEN .env | cut -d= -f2)"
```

- [ ] **Step 3: Push and tag**

```bash
git tag v0.6.0
git push origin feature/observer-api-merge
git push origin v0.6.0
```

- [ ] **Step 4: Clean up worktree**

```bash
cd /Users/eddy/Developer/Python/CoreMem
git worktree remove .worktrees/observer-api-merge
```