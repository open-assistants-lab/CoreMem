# AgentMemory Merge: Main Branch + AgentMemory POC

2026-06-27

## 1. Goal

Merge the AgentMemory POC into the main branch. Keep HybridDB for raw message
storage and `search_enhanced()`. Remove the observer/reflector pipeline.
Add AgentMemory's deterministic compiler, daily journal, dreaming consolidation,
and BM25 + cross-encoder search.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRIDDB (raw messages)                    │
│  • All messages stored by role, date, session                 │
│  • search() — FTS5 keyword search                             │
│  • search_enhanced() — FTS5 + embedding + query expansion     │
│  • Source of truth, immutable                                 │
│  • Schema: id, role, content, session_id, turn_id, user_id,   │
│    agent_id, ts, metadata, embedding                          │
│  • turn_id added — groups messages by conversation turn       │
│  • ingest() updated to accept turn_id parameter               │
└─────────────────────────────────────────────────────────────┘
        │
        ▼ (background, per-turn)
┌─────────────────────────────────────────────────────────────┐
│              LLM COMPILER (1 call per turn)                   │
│  • Reads messages from HybridDB via MemoryCore.compile_turn() │
│  • Message format conversion:                                 │
│    HybridDB row → {message_id: id, role, content}             │
│  • Generates timestamped section with claims + exact quotes  │
│  • Appends to {workspace}/memorypack/daily/YYYY-MM-DD.md     │
│  • Deterministic validation (quote check, role enforcement)  │
│  • Retry loop + quote-fixing post-processor                  │
│  • Cached by session content hash in .llm_cache/             │
│  • Cross-encoder model downloaded on first use (~80MB)        │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│              JOURNAL (MD files, daily atomic unit)             │
│  {workspace}/memorypack/                                      │
│  ├── daily/YYYY-MM-DD.md — timestamped sections               │
│  ├── weekly/YYYY-Www.md — navigation indexes                 │
│  ├── monthly/YYYY-MM.md — navigation indexes                 │
│  ├── index.md — table of contents                             │
│  ├── MEMORY.md — boot memory (promoted by dreaming)           │
│  ├── DREAMS.md — diary analysis output                        │
│  ├── .llm_cache/ — LLM response cache (turn_id → plan)       │
│  └── .dreaming_cursor — last processed date                   │
│                                                               │
│  • No references/ directory (HybridDB replaces it)            │
│  • No pages/ directory (daily/ replaces it)                   │
│  • AgentMemoryBundle.initialize() creates memorypack/ dir      │
│  • Path: {workspace}/memorypack/                              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼ (background, periodic — after every dreaming cycle)
┌─────────────────────────────────────────────────────────────┐
│              DREAMING CONSOLIDATION (1 call per cycle)        │
│  • Reads daily journal pages since last cursor               │
│  • Diary study analysis (events, emotions, behaviors, etc.) │
│  • Promotes durable facts to MEMORY.md                       │
│  • Appends analysis to DREAMS.md                             │
│  • After completion, calls rebuild_index()                   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    SEARCH (at query time)                      │
│  • search_enhanced() — raw messages in HybridDB              │
│  • search_memory() — compiled pages in daily/                │
│    (BM25 + stemming + fuzzy + cross-encoder)                  │
│  • Agent uses both: raw for exact recall, compiled for        │
│    distilled understanding                                   │
└─────────────────────────────────────────────────────────────┘
```

## 3. Files to Keep (from main branch)

- `coremem/core.py` — `MemoryCore` with `search()`, `search_enhanced()`, `ingest()`
- `coremem/providers.py` — LLM provider factory
- `coremem/types.py` — Memory, SearchQuery, SearchResult
- `coremem/heuristics.py` — SearchHeuristics, MMR diversification
- `coremem/query.py` — query expansion
- `coremem/rerank.py` — reranking
- `pyproject.toml` — HybridDB, sentence-transformers, numpy deps

## 4. Files to Remove (from main branch)

- `coremem/observer.py` — ObserverPipeline (9+ LLM calls/turn, no validation)
- `coremem/reflector.py` — ReflectorPipeline (separate LLM calls, speculative)
- `tests/test_observer.py` — observer tests
- `tests/test_reflector.py` — reflector tests
- Any references to observer/reflector in `coremem/core.py` and `coremem/__init__.py`

## 5. Files to Add (from AgentMemory POC)

- `coremem/memorypack/__init__.py` — exports
- `coremem/memorypack/bundle.py` — AgentMemoryBundle, AgentMemorySearch, BM25, stemming, fuzzy, _extract_sections()
- `coremem/memorypack/compiler.py` — AgentMemoryCompiler, compile_section(), plan validation
- `coremem/memorypack/llm_compiler.py` — AgentMemoryLLMCompiler, retry loop, quote fixing, caching
- `coremem/memorypack/dreaming.py` — dream(), diary analysis consolidation
- `coremem/memorypack/rebuild_index.py` — rebuild_index(), weekly/monthly/index generation
- `coremem/memorypack/reranker.py` — CrossEncoderReranker
- `coremem/memorypack/embeddings.py` — EmbeddingIndex (kept for future use, not in default search)
- `scripts/eval_memorypack_longmemeval.py` — LongMemEval baseline eval
- `scripts/eval_memorypack_llm_compiler.py` — Stage 4 LLM compiler eval
- `scripts/eval_memorypack_internal.py` — internal scripted eval
- `tests/test_memorypack.py` — bundle, search, lint tests
- `tests/test_memorypack_compiler.py` — compiler tests
- `tests/test_memorypack_eval_longmemeval.py` — eval tests
- `tests/test_memorypack_eval_internal.py` — internal eval tests
- `docs/superpowers/specs/2026-06-25-memorypack-end-to-end.md` — architecture spec

## 6. Files to Modify

### `coremem/__init__.py`

Add AgentMemory exports alongside existing MemoryCore exports:

```python
from coremem.agent_memory import (
    AgentMemoryBundle,
    AgentMemoryCompiler,
    AgentMemoryLLMCompiler,
    AgentMemorySearch,
    CrossEncoderReranker,
    dream,
    rebuild_index,
)
```

### `coremem/core.py`

- Remove observer/reflector imports and references
- Add `compile_turn()` method:
  ```python
  async def compile_turn(self, session_id: str, timestamp: str, title: str) -> None:
      rows = self._db.query(
          "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY ts",
          [session_id],
      )
      messages = [{"message_id": r["id"], "role": r["role"], "content": r["content"]} for r in rows]
      await self._llm_compiler.compile_session(
          turn_id=session_id,
          session_id=session_id,
          messages=messages,
          timestamp=timestamp,
          title=title,
      )
  ```
- Add `dream()` method:
  ```python
  async def dream(self) -> dict:
      return await dream(self._memorypack_bundle)
  ```
- Add `rebuild_index()` method:
  ```python
  def rebuild_index(self) -> dict:
      return rebuild_index(self._memorypack_bundle.root)
  ```
- Add `search_memory()` method:
  ```python
  def search_memory(self, query: str, limit: int = 5) -> list[SearchHit]:
      return self._memorypack_search.search(query, limit=limit)
  ```
- Keep `search()`, `search_enhanced()`, `ingest()` unchanged
- `MemoryCore.__init__()` creates `AgentMemoryBundle` at `{path}/memorypack/`:
  ```python
  self._memorypack_root = Path(path).parent / "memorypack"
  self._memorypack_bundle = AgentMemoryBundle(self._memorypack_root)
  self._llm_compiler = AgentMemoryLLMCompiler(self._memorypack_bundle)
  self._memorypack_search = AgentMemorySearch(
      self._memorypack_root,
      reranker=CrossEncoderReranker(),
  )
  ```

### `coremem/memorypack/bundle.py`

- `AgentMemoryBundle.initialize()` creates `daily/` directory instead of `pages/`
- Remove `references/` directory creation (HybridDB replaces it)
- `AgentMemorySearch` defaults to `daily/` directory
- Keep `pages/` fallback for backward compatibility during migration

### `pyproject.toml`

- Keep `hybriddb>=0.4.5`
- Keep `sentence-transformers>=3.0.0`
- Keep `numpy` (already present)
- No new dependencies

## 7. Integration Points

### HybridDB → LLM Compiler

The LLM compiler reads messages from HybridDB. Message format conversion:

```python
# HybridDB row → AgentMemory message dict
{
    "message_id": row["id"],       # HybridDB's UUID
    "role": row["role"],            # user, assistant, tool_result
    "content": row["content"],     # verbatim message text
}
```

HybridDB has no `turn_id` column. Messages are grouped by `session_id`.
The `compile_turn()` method queries by `session_id` and uses the session_id
as the turn_id for citation purposes.

### AgentMemory Search

`search_memory()` is a separate method on `MemoryCore`:

```python
def search_memory(self, query: str, limit: int = 5) -> list[SearchHit]:
    return self._memorypack_search.search(query, limit=limit)
```

The agent uses `search_enhanced()` for raw message recall and
`search_memory()` for compiled knowledge. Both are available.

### Dreaming → MEMORY.md

Dreaming reads daily pages, promotes facts to MEMORY.md. MEMORY.md is loaded
at agent startup (boot memory). No HybridDB interaction needed.

`rebuild_index()` is called after every dreaming cycle to regenerate
weekly/monthly/index navigation files.

### Cross-Encoder Model

`CrossEncoderReranker` downloads `cross-encoder/ms-marco-MiniLM-L-6-v2`
(~80MB) on first `rerank()` call. This is a one-time cost (~10s on first
search). Subsequent searches use the cached model.

## 8. Migration

Existing HybridDB data is unaffected. AgentMemory starts empty — it only
processes new turns after the merge. Old data can be backfilled by running
the compiler on historical sessions.

**Directory changes:**
- `references/` directory is removed — HybridDB replaces it
- `pages/` directory is removed — `daily/` replaces it
- `memorypack/` directory is created at `{workspace}/memorypack/`
- `.llm_cache/` and `.dreaming_cursor` live inside `memorypack/`

**Test changes:**
- AgentMemory tests currently use `AgentMemoryBundle` with reference turn files
- After merge, tests should use `MemoryCore.compile_turn()` which reads from HybridDB
- Test fixtures need updating to write messages to HybridDB instead of reference files
- Existing `test_memorypack_compiler.py` tests use `apply_plan()` directly — these
  still work because they test the compiler, not the storage backend

## 9. Verification

Run both evals to confirm no regression:

```bash
# Baseline (raw messages, HybridDB)
uv run python3 scripts/eval_memorypack_longmemeval.py data/longmemeval_8_remaining_subset.json --k 5

# LLM compiler (compiled pages, AgentMemory)
uv run python3 scripts/eval_memorypack_llm_compiler.py data/longmemeval_8_remaining_subset.json --k 5 --limit 2
```

Expected: baseline ≥ 0.75 recall@5, compiler ≥ 0.80 recall@5 (on 2 instances).

Eval scripts import from `coremem.memorypack` which is installed as part of
the `coremem` package. No `sys.path` manipulation needed after merge.
