# CoreMem Metadata & Filters Design

## Context

CoreMem stores agent memories as flat records with fixed fields: `id`, `content`,
`role`, `ts`, `session_id`, `workspace_id`. There's no way for developers to attach
arbitrary tags or attributes (project name, priority, source, category) to a memory
— and no way to filter search results by those attributes.

Every competing agentic memory system has this:
- **Mem0** (53K⭐): `add(messages, metadata={"source": "email"})`, search with
  `filters={"category": {"contains": "work"}}` + AND/OR/NOT DSL
- **agentmemory** (19K⭐): facet tags (`dimension:value` pairs), facet query
- **Membrane**: scope-based access with sensitivity tags, actor identity

The gap is clear: developers can't tag memories or scope search to a subset.
This spec closes that gap with the simplest possible API — a permissive
`metadata` dict on ingest and flat equality `filters` on search. No DSL, no
nested operators, no schema enforcement.

## Summary

Add arbitrary `metadata` dict to memory ingestion, plus flat-equality `filters` on
search — matching the common API across agentic memory systems. No filter DSL
(no AND/OR/NOT/gte/in). Keep it simple.

---

## Data Model Changes

### Memory

```python
@dataclass
class Memory:
    id: str
    content: str
    role: str = "user"
    ts: datetime | None = None
    session_id: str | None = None
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "ts": self.ts.isoformat() if self.ts else None,
            "session_id": self.session_id,
            "score": self.score,
            "metadata": self.metadata,
        }
```

- `metadata` stores arbitrary key-value pairs
- Stored as opaque data by the backend, returned verbatim in results
- `workspace_id` removed from Memory — goes into `metadata["workspace_id"]`
- **Migration:** any code reading `memory.workspace_id` must change to
  `memory.metadata.get("workspace_id")`. The field no longer exists.

### SearchQuery

```python
@dataclass
class SearchQuery:
    text: str
    limit: int = 10
    filters: dict[str, Any] = field(default_factory=dict)
```

- `filters` is flat key=value equality matching
- Multiple keys combine as AND
- `wing`/`room` fields removed (unused MemPalace artifacts)
- `workspace_id` removed from SearchQuery — lives in `filters["workspace_id"]`

---

## API Surface

### Ingest — single message

```python
def ingest(
    self,
    role: str,
    content: str,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str
```

### `ingest_many` — removed from MemoryCore

`ingest_batch` remains on `StoreBackend` for bulk internal use.

### Search — fast path

```python
def search(
    self,
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[SearchResult]
```

### Search — enhanced path

```python
def search_enhanced(
    self,
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
    depth: int = 5,
) -> list[SearchResult]
```

`depth` replaces the internal `_SEARCH_DEPTH_MULTIPLIER` constant. Higher depth
means more candidates for the cross-encoder reranker.

### Export

```python
def export(
    self,
    filters: dict[str, Any] | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> list[Memory]:
    """Paginated export. Returns one page of memories matching filters."""

def export_all(self, filters: dict[str, Any] | None = None) -> list[Memory]:
    """Export all matching memories. Developer explicitly chooses this."""
```

Two methods rather than a `limit=None` magic sentinel. `export()` for paginated
iteration, `export_all()` when the developer acknowledges "all data in memory."
`export_all()` internally loops with pagination via `backend.list()` — caller does
not manage offsets.

### Import

```python
def import_batch(self, memories: list[Memory]) -> list[str]:
    """Batch import. Returns storage IDs. Delegates to backend.ingest_batch()."""
```

Backend handles any internal chunking — users don't think about it.

---

## Backend Contract

### StoreBackend (abstract)

```python
class StoreBackend(ABC):
    @abstractmethod
    def ingest(self, memory: Memory) -> str: ...

    @abstractmethod
    def ingest_batch(self, memories: list[Memory]) -> list[str]: ...

    @abstractmethod
    def search(self, query: SearchQuery) -> list[SearchResult]: ...

    @abstractmethod
    def list(self, filters: dict | None = None, limit: int | None = None, offset: int = 0) -> list[Memory]: ...

    @abstractmethod
    def get_recent(self, limit: int = 10) -> list[Memory]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...
```

New method: `list()` — returns raw memories (not scored search results), supports
filters + pagination. Backbone of `export()`.

### ChromaBackend

- `filters` → `collection.query(where=filters, ...)` — native ChromaDB AND equality
- `filters` → `collection.get(where=filters, limit=limit, offset=offset)` for `list()`
- `metadata` → flattened into top-level ChromaDB metadatas keys
  (e.g. `metadata={"project": "x"}` adds `metadatas={"project": "x", ...}`)
- **Reserved key collision**: ChromaBackend stores `role`, `session_id`, `ts` as
  top-level metadatas. If user's `metadata` contains these keys, user values win
  (no namespace prefix — ChromaDB returns merged metadatas).
- **Type constraints**: ChromaDB `where` only supports `str`, `int`, `float`, `bool`.
  Values of other types (dicts, lists, `None`) are stored but silently ignored in
  `where` clauses — not filterable in ChromaBackend.
- **Cross-backend note**: HybridBackend stores all JSON types. Applications that need
  filter portability should restrict metadata values to `str`, `int`, `float`, `bool`.
- **Implementation note**: current `ChromaBackend.ingest_batch` only stores
  `role`, `session_id`, `ts`. Must be updated to flatten user `metadata` as well.

### HybridBackend

- `filters` → appended as SQL `WHERE json_extract(metadata, '$.k') = ? AND json_extract(metadata, '$.k2') = ?` clauses
- `metadata` → stored as JSON in the existing `metadata` TEXT column
- `list()` → `SELECT * FROM messages WHERE ... ORDER BY ts DESC LIMIT ? OFFSET ?`
- `import_batch` → delegates to `self._db.insert_batch("messages", rows)`

---

## EA Changes

- `workspace_id` moves from `Memory.workspace_id` to `metadata["workspace_id"]`
- EA tools (`message_search`, `message_count`, `message_timeline`) pass
  `filters={"workspace_id": workspace_id}` to `core.search_enhanced()`
- EA reads `workspace_id` from `result.memory.metadata.get("workspace_id")`
  instead of `result.memory.workspace_id`

---

## Open Questions (Deferred)

- **Nested metadata keys** — only top-level keys supported. No dot-path access.
- **Rich operators** — not part of this spec. If demand emerges, add `gte`/`in`/`AND`
  in a future release without breaking the flat equality API.
- **Metadata value types** — ChromaDB `where` supports `str`, `int`, `float`, `bool`.
  HybridDB stores everything as JSON text. Strings are the safe cross-backend type.
  Applications that need ChromaBackend filterability should restrict to these types.

---

## Non-Goals

- No filter DSL (AND/OR/NOT/comparison operators)
- No full-text search on metadata values
- No metadata indexing configuration exposed to users
- No schema enforcement on metadata keys
