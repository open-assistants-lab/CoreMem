# Observer API Merge — Spec v3

## Goal

Single `MemoryCore` object. One path, one backend, all tables. Dev never touches `HybridBackend`, `MemoryStore`, or ChromaBackend. Pipeline exposes `extract()` and `retrieve()`.

## What changes

### 1. `MemoryCore` becomes the one object

```python
core = MemoryCore(path="./memory", enable_observations=True)

# Messages
core.ingest("user", "I like coffee")
core.fetch(session_id="conv_001")

# Observations (same store, same path)
await pipeline.extract()
obs = pipeline.retrieve(limit=50)
```

`MemoryCore.__init__` creates a single `HybridDB` at `path`. When `enable_observations=True`, it creates **5 tables** (messages always exists):

| Table | Purpose |
|-------|---------|
| `messages` | Conversation turns (existing) |
| `observations` | Extracted facts (from MemoryStore) |
| `observation_events` | Audit trail for dedup lifecycle (renamed from `memory_events`) |
| `observation_conflicts` | Contradiction tracking (renamed from `memory_conflicts`) |
| `reflections` | Synthesized insights (from Reflector) |

Schema creation is lazy — tables are created on first access (already the pattern for messages). `enable_observations=True` creates all 5 tables. The `enable_reflections` flag is removed — observations imply reflections, they share the same database. The Reflector pipeline itself (the synthesis logic) is unchanged in this spec.

### 2. `MemoryStore` absorbed into `MemoryCore`

All `MemoryStore` methods moved to `MemoryCore`:

| Method | Purpose |
|--------|---------|
| `core.insert_observations(items)` | Store extracted facts |
| `core.get_observations(days, limit)` | Recent durable observations |
| `core.get_observations_since(ts)` | Observations after timestamp |
| `core.search_observations(query)` | Semantic search observations |
| `core.get_pending_reflections()` | Un-reflected observations (for Reflector) |
| `core.mark_reflected(ids)` | Mark as processed (for Reflector) |
| `core.insert_reflections(items)` | Store synthesized insights (for Reflector) |
| `core.get_reflections(limit)` | Recent reflections |
| `core.search_reflections(query)` | Semantic search reflections |
| `core.apply_decay(half_life_days)` | Archive expired temporary obs |
| `core.get_candidates(...)` | Dedup candidate retrieval |
| `core.update_observation(id, updates)` | Partial update |
| `core.insert_event(...)` | Audit trail (for dedup) |
| `core.create_conflict(...)` | Contradiction tracking (for dedup) |

`coremem/memory_store.py` is deleted. Observation tests move to `tests/test_core.py`. The `reflector.py` file changes memory arg but is otherwise untouched — its methods are moved to `MemoryCore` so it continues to function.

### 3. `ObserverPipeline` simplified

```python
pipeline = ObserverPipeline(
    memory=core,          # single object
    session_id="conv_001",
    model="deepseek:deepseek-v4-flash",
    token_threshold=100,
    min_turns=1,
    enable_classification=True,
    enable_dedup=True,
)

await pipeline.extract()       # after turn, writes to core
obs = pipeline.retrieve()       # before model, reads from core
obs = pipeline.retrieve(query="likes")  # tool-based
```

No more `core=`, `store=` constructor args.

### 4. `HybridBackend` becomes internal

Devs no longer import `HybridBackend`. It's created by `MemoryCore.__init__` internally.

`MemoryCore.__init__` signature:

```python
class MemoryCore:
    def __init__(
        self,
        path: str,                    # single path for everything
        llm_provider: LLMProvider | None = None,
        enable_observations: bool = False,  # creates all 5 tables
    ):
        self._db = HybridDB(path=path)
        self._ensure_message_tables()
        if enable_observations:
            self._ensure_observation_tables()
            self._ensure_reflection_tables()
```

`ChromaBackend` removed entirely. `coremem/backends/` directory deleted.

### 5. Error handling

If `enable_observations=False`:
- `pipeline.extract()` raises `RuntimeError("Observer pipeline requires enable_observations=True")`
- `pipeline.retrieve()` raises the same
- `core.get_observations()`, `core.search_observations()`, etc. raise `RuntimeError`

### 6. `retrieve()` filter behavior

`pipeline.retrieve(query=None, days=30, limit=50)` returns only observations where `durability='durable'` and `status != 'archived'` — the "working memory" the agent needs. This mirrors `MemoryStore.get_recent_observations()` current default. Passing a `query` delegates to `search_observations` which has no durability filter (semantic search should find everything).

### 7. Table renames

| Old name | New name |
|----------|----------|
| `memory_events` | `observation_events` |
| `memory_conflicts` | `observation_conflicts` |

The `insert_event` and `create_conflict` column names (`memory_id`, `memory_id_a`, `memory_id_b`) renamed to `observation_id`, `observation_id_a`, `observation_id_b`.

## Migration

v0.5.1:
```python
from coremem.backends.hybrid import HybridBackend
from coremem.memory_store import MemoryStore
from coremem.observer import ObserverPipeline

backend = HybridBackend(path="./memory")
core = MemoryCore(backend=backend)
store = MemoryStore(path="./memory")
pipeline = ObserverPipeline(core=core, store=store, ...)
store.get_recent_observations(limit=50)
```

v0.6.0:
```python
from coremem import MemoryCore
from coremem.observer import ObserverPipeline

core = MemoryCore(path="./memory", enable_observations=True)
pipeline = ObserverPipeline(memory=core, ...)

await pipeline.extract()
obs = pipeline.retrieve(limit=50)
obs = pipeline.retrieve(query="likes")
```

## Files changed

| File | Action |
|------|--------|
| `coremem/core.py` | Rewrite constructor, add all MemoryStore methods, move schemas here |
| `coremem/memory_store.py` | Delete |
| `coremem/migrations/v0_5_to_v0_6.py` | Update table names, import schemas from core instead of memory_store |
| `coremem/observer.py` | Change constructor, rename `after_turn` → `extract`, add `retrieve()` |
| `coremem/reflector.py` | Take `memory` arg instead of `store`, use `memory.get_pending_reflections()` etc. |
| `coremem/dedup.py` | Update `insert_event()` / `create_conflict()` calls to go through `memory` |
| `coremem/classifier.py` | No change (pure function) |
| `coremem/backends/` | Delete entire directory |
| `coremem/__init__.py` | Remove MemoryStore, HybridBackend, ChromaBackend imports |
| `tests/test_core.py` | Add MemoryStore method tests |
| `tests/test_memory_store.py` | Delete |
| `tests/test_pipelines.py` | Update constructor calls |
| `README.md` | Rewrite setup section |

## Non-goals

- ChromaBackend removed (no deprecation)
- ReflectorPipeline unchanged in behavior (same pipeline class, just updated to take `memory` arg instead of `store`)
- No `Memory` rename — class stays `MemoryCore`
- No CLI or config file changes

## Decisions

- `enable_observations=True` creates all 5 tables. No separate `enable_reflections` flag.