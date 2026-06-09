# CoreMem Observation & Reflection CRUD

2026-06-09

## 1. Motivation

MemoryCore's observation/reflection API is read-heavy but missing:
- Delete for both observations and reflections
- Update for reflections
- `metadata` persistence — ObserverPipeline uses `metadata` to **filter input messages** but the **output observations lose it**. EA needs workspace-scoped observations (`metadata={"workspace_id": "work"}`), so `core.observations(metadata={"workspace_id": "work"})` must work.

Users dropping to `raw_query()` to clean up scoped data is unacceptable for the OSS library.

## 2. Current CRUD Matrix

| Operation | Messages | Observations | Reflections |
|-----------|----------|-------------|-------------|
| Create (manual) | `ingest()` | `insert_observations()` | `insert_reflections()` |
| Create (LLM) | n/a | `observer.extract()` | `reflector.run_now()` |
| Read (list) | `fetch()` | `observations()` / `get_observations()` | `reflections()` / `get_reflections()` |
| Read (search) | `search()` / `search_enhanced()` | `observations(query)` / `search_observations()` | `reflections(query)` / `search_reflections()` |
| Update | ❌ | `update_observation()` | ❌ |
| Delete | `delete()` | ❌ | ❌ |

Schema gaps (this spec fixes):

| Schema | Has `metadata`? | Has `agent_id`? |
|--------|----------------|-----------------|
| `messages` | ✅ (TEXT) | ✅ |
| `_OBSERVATIONS_SCHEMA` | ❌ | ✅ |
| `_REFLECTIONS_SCHEMA` | ❌ | ❌ (`user_id`, `session_id` only) |

## 3. Proposed API

### Schema changes

Add `metadata TEXT DEFAULT '{}'` to both `_OBSERVATIONS_SCHEMA` and `_REFLECTIONS_SCHEMA`.
Add `agent_id TEXT` to `_REFLECTIONS_SCHEMA`.

### `delete_observations(...) -> int`

Explicit keyword args mirroring `messages.delete()`. Cascades to `observation_events` and `observation_conflicts`. Does NOT cascade to reflections (dangling `linked_observation_ids` are harmless — reflection content remains valid).

```python
core.delete_observations(session_id="work")
core.delete_observations(metadata={"workspace_id": "work"})
core.delete_observations(kind="fact", status="candidate")
core.delete_observations(user_id="alice")
```

### `delete_observations_by_id(ids: list[str]) -> int`

Delete by primary key. Cascades same as above.

```python
core.delete_observations_by_id(["abc123", "def456"])
```

### `delete_reflections(...) -> int`

Explicit keyword args. No child tables.

```python
core.delete_reflections(user_id="alice")
core.delete_reflections(session_id="work")
```

### `delete_reflections_by_id(ids: list[str]) -> int`

```python
core.delete_reflections_by_id(["xyz789"])
```

### `update_reflections(ref_id, updates) -> None`

Mirror of `update_observation()`.

```python
core.update_reflections("abc123", {"score": 0.9, "content": "Updated insight"})
```

### `insert_observations()` — add `metadata` support

Add `metadata` param. If not provided, default to `{}`. Stored as JSON string in `metadata TEXT` column.

### `insert_reflections()` — add `metadata` + `agent_id` support

Add `metadata` and `agent_id` params.

### `observations()` / `reflections()` — metadata filter passthrough

Currently `observations(query, **kwargs)` only passes `**kwargs` to `get_observations()` when `query=None`. When `query="text"`, it calls `search_observations(query)` which ignores `**kwargs`. Same for `reflections()`.

Fix `observations()` to post-filter by metadata after semantic search. Note: `search_observations()` / `search_reflections()` remain metadata-unaware (HybridDB semantic search is content-only). Advanced users who call them directly will not get metadata scoping — they should use `observations(query, metadata=...)` instead.

```python
def observations(self, query=None, limit=10, **kwargs):
    if query:
        results = self.search_observations(query, limit=limit)
        meta = kwargs.get("metadata")
        if meta:
            results = [r for r in results if json.loads(r.get("metadata", "{}")) == meta]
        return results
    return self.get_observations(limit=limit, **kwargs)
```

Fix underlying getters:
1. **`get_observations()`** — already has `metadata` param but **doesn't use it** in WHERE clause. Wire in via `json_extract` per key.
2. **`get_observations_since()`** — same pre-existing gap. Wire `metadata` into WHERE.
3. **`get_reflections()`** — missing `metadata` and `agent_id` params. Add both. Uses `self._db.query()` (not `raw_query`), so metadata filter needs `where_parts` with LIKE-pattern or switch to `raw_query` for JSON extract.

## 4. Design Decisions

1. **Explicit keyword args, not `**filters`** — matches `messages.delete()` pattern. Implementation section is correct, prose was wrong.

2. **No cascade to reflections on observation delete** — reflections form independent insights. Deleting source facts doesn't invalidate the insight. Dangling `linked_observation_ids` are harmless.

3. **`metadata` stored as JSON string** — same pattern as `messages.metadata`. Simple `json.dumps()` on insert, `json_extract()` in WHERE clause for filtering.

4. **`metadata` filter uses JSON extraction** — for now, exact match via `json_extract(metadata, '$.key') = ?` for each key. More sophisticated filtering (partial match, wildcards) deferred.

5. **`agent_id` added to reflections** — fills a gap. Reflections should support the same filter fields as observations.

6. **ObserverPipeline passes `self._metadata` to stored observations** — at line 811, before calling `insert_observations()`, inject `metadata=self._metadata` into each observation dict. Same for ReflectorPipeline.

## 5. Implementation

### Schema migration (coremem/core.py)

```python
_OBSERVATIONS_SCHEMA = {
    ...
    "metadata": "TEXT DEFAULT '{}'",   # ADD
    "embedding": "TEXT",
}

_REFLECTIONS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "content": "LONGTEXT",
    "domain": "TEXT",
    "linked_observation_ids": "TEXT",
    "score": "REAL",
    "embedding": "TEXT",
    "user_id": "TEXT",
    "session_id": "TEXT",
    "agent_id": "TEXT",               # ADD
    "metadata": "TEXT DEFAULT '{}'",  # ADD
}
```

The schema dicts (`_OBSERVATIONS_SCHEMA` / `_REFLECTIONS_SCHEMA`) already include the new columns so `CREATE TABLE` handles new DBs.

For existing DBs, add `ALTER TABLE` after the `CREATE TABLE` block in `_ensure_observation_tables()`, guarded by schema version:

```python
def _ensure_observation_tables(self) -> None:
    if "observations" not in self._db.list_tables():
        # new DB — CREATE TABLE handles all columns
        ...
    # existing DB — migrate v0.7.x schemas
    cols = {r["name"] for r in self._db.raw_query("PRAGMA table_info(observations)")}
    if "metadata" not in cols:
        self._db.raw_query("ALTER TABLE observations ADD COLUMN metadata TEXT DEFAULT '{}'")
    cols = {r["name"] for r in self._db.raw_query("PRAGMA table_info(reflections)")}
    if "metadata" not in cols:
        self._db.raw_query("ALTER TABLE reflections ADD COLUMN metadata TEXT DEFAULT '{}'")
    if "agent_id" not in cols:
        self._db.raw_query("ALTER TABLE reflections ADD COLUMN agent_id TEXT")
```

This avoids fragile try/except and only runs ALTER when the column is actually missing.

### `insert_observations()` changes

```python
def insert_observations(self, items, metadata=None):
    ...
    obs_metadata = item.get("metadata", metadata or {})
    if not isinstance(obs_metadata, str):
        obs_metadata = json.dumps(obs_metadata)
    ...
    self._db.insert("observations", {
        ...
        "metadata": obs_metadata,
    })
```

### `insert_reflections()` changes

```python
def insert_reflections(self, items, metadata=None):
    ...
    ref_metadata = item.get("metadata", metadata or {})
    if not isinstance(ref_metadata, str):
        ref_metadata = json.dumps(ref_metadata)
    ...
    self._db.insert("reflections", {
        ...
        "agent_id": item.get("agent_id", ""),
        "metadata": ref_metadata,
    })
```

### `delete_observations()`

```python
def delete_observations(
    self,
    user_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    kind: str | None = None,
    status: str | None = None,
    ts_after: str | None = None,
    ts_before: str | None = None,
) -> int:
```

Internal flow:
1. Build WHERE from filters. `ts_after`/`ts_before` use `observation_ts` column. Metadata uses `json_extract(metadata, '$.key') = ?` per key.
2. `SELECT id FROM observations WHERE ...`
3. If no IDs matched, return 0 (skip event/conflict DELETEs)
4. `DELETE FROM observation_events WHERE observation_id IN (...)`
5. `DELETE FROM observation_conflicts WHERE observation_id_a IN (...) OR observation_id_b IN (...)`
6. `DELETE FROM observations WHERE ...`
7. Return deleted count

### `delete_observations_by_id()`

```python
def delete_observations_by_id(self, ids: list[str]) -> int:
    self._check_observations_enabled()
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    self._db.raw_query(f"DELETE FROM observation_events WHERE observation_id IN ({placeholders})", tuple(ids))
    self._db.raw_query(
        f"DELETE FROM observation_conflicts WHERE observation_id_a IN ({placeholders}) OR observation_id_b IN ({placeholders})",
        tuple(ids) + tuple(ids),
    )
    self._db.raw_query(f"DELETE FROM observations WHERE id IN ({placeholders})", tuple(ids))
    return len(ids)
```

### `delete_reflections_by_id()`

```python
def delete_reflections_by_id(self, ids: list[str]) -> int:
    self._check_observations_enabled()
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    self._db.raw_query(f"DELETE FROM reflections WHERE id IN ({placeholders})", tuple(ids))
    return len(ids)
```

### `delete_reflections()`

```python
def delete_reflections(
    self,
    user_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    domain: str | None = None,
) -> int:
```

### ObserverPipeline changes (observer.py)

At line 811, before calling `insert_observations()`, inject metadata:

```python
if new_obs:
    for obs in new_obs:
        obs.setdefault("metadata", self._metadata or {})
        obs.setdefault("agent_id", self._agent_id or "")
        obs.setdefault("user_id", self._user_id or "")
        obs.setdefault("session_id", self._session_id or "")
    self._memory.insert_observations(new_obs)
```

### ReflectorPipeline changes (reflector.py)

At line 205, before calling `insert_reflections()`, inject metadata:

```python
if good:
    for ref in good:
        ref.setdefault("metadata", self._metadata or {})
        ref.setdefault("user_id", self._user_id or "")
        ref.setdefault("session_id", self._session_id or "")
        ref.setdefault("agent_id", self._agent_id or "")
    new_ids = self._memory.insert_reflections(good)
```

### `get_observations()` / `get_reflections()` — metadata filter

For both getters, add `metadata` to the WHERE clause building. Iterate metadata dict and add `json_extract(metadata, '$.key') = ?` for each entry.

`get_observations()` already uses `raw_query` — straightforward WHERE addition.

`get_reflections()` currently uses `self._db.query()` which takes a bare SQL string fragment. Switch to `raw_query` to support JSON extract:

```python
def get_reflections(self, limit=10, user_id=None, session_id=None, agent_id=None, metadata=None):
    where_parts = []
    params = []
    ...
    if metadata:
        for k, v in metadata.items():
            where_parts.append(f"json_extract(metadata, '$.{k}') = ?")
            params.append(v)
    where = " AND ".join(where_parts) if where_parts else "1=1"
    rows = self._db.raw_query(
        f"SELECT * FROM reflections WHERE {where} ORDER BY score DESC LIMIT ?",
        tuple(params) + (limit,),
    )
    return [dict(r) for r in rows]
```

## 6. Tests

- `test_delete_observations_by_session` — create 3 obs with different session_ids, delete one
- `test_delete_observations_by_metadata` — create obs with metadata, delete by matching metadata
- `test_delete_observations_by_id` — delete specific IDs
- `test_delete_observations_cascades_events` — verify child table cleanup
- `test_delete_reflections_by_filter` — create reflections, delete by user_id
- `test_delete_reflections_by_id` — delete specific IDs
- `test_update_reflections` — create, update fields, verify
- `test_observations_metadata_persists` — extract via observer, verify metadata stored
- `test_reflections_metadata_persists` — run reflector, verify metadata stored
- `test_delete_disabled_when_observations_off` — verify RuntimeError
- `test_observations_query_by_metadata` — insert with metadata, query via `observations(metadata=...)`
- `test_reflections_query_by_metadata` — insert with metadata, query via `reflections(metadata=...)`