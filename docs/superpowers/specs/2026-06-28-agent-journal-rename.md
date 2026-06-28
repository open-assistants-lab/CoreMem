# AgentJournal Rename: AgentMemory → AgentJournal + Path Change

2026-06-28

## 1. Problem

Two sources of naming confusion in the current API:

**A. `MemoryCore` wraps a HybridDB message store, not "memory".**
A developer calling `MemoryCore(path="./memory")` expects neural memory, not a
plain message database. The path `./memory` is misleading.

**B. `AgentMemory` (compiled markdown journal) sounds like the same thing.**
The names `AgentMemoryBundle`, `AgentMemorySearch`, `AgentMemoryCompiler` all
collide conceptually with the `MemoryCore` wrapper that owns them. A developer
cannot tell which layer does what.

**C. The directory name `agent_memory/` collides with the old `AgentMemory` class name.**
A developer scanning the filesystem sees `agent_memory/` and assumes it matches
the class they're using. Renaming both to `agent_journal` makes the two-tier
structure immediately clear.

## 2. Solution

### 2A. Rename `AgentMemory*` → `AgentJournal*`

| Current | New | Reason |
|---|---|---|
| `AgentMemoryBundle` | `AgentJournalBundle` | It's a bundle of journal files |
| `AgentMemorySearch` | `AgentJournalSearch` | Searches journal pages |
| `AgentMemoryCompiler` | `AgentJournalCompiler` | Compiles turns into journal entries |
| `AgentMemoryLLMCompiler` | `AgentJournalLLMCompiler` | LLM-powered journal compiler |
| `AgentMemoryCompileResult` | `AgentJournalCompileResult` | Result of a compile operation |
| `AgentMemoryError` | `AgentJournalError` | Error class |
| `compile_memorypack_plan` | `compile_journal_plan` | Journal plan compilation |
| `compute_agent_context_hash` | *(unchanged)* | Agent context, not journal or memory |

Module directory:
- `coremem/agent_memory/` → `coremem/agent_journal/`

Internal variable names in `core.py`:
- `self._agent_memory_root` → `self._agent_journal_root`
- `self._agent_memory_bundle` → `self._agent_journal_bundle`
- `self._llm_compiler` → `self._journal_compiler`
- `self._agent_memory_search` → `self._agent_journal_search`

Frontmatter/schema strings:
- `"agent_memory_version"` → `"agent_journal_version"`
- `"AgentMemory Page"` → `"AgentJournal Page"`
- `PROFILE_VERSION` stays `"0.1"` (version number, not name)
- `SCHEMA.md` content updated to reflect new naming

### 2B. Fix `AgentJournal` path to `workspace_root / "agent_journal"`

```python
# Before (coremem/core.py:114)
workspace_root = Path(path).resolve().parent
self._agent_memory_root = workspace_root / "agent_memory"

# After
workspace_root = Path(path).resolve().parent
self._agent_journal_root = workspace_root / "agent_journal"
```

The journal stays co-located with the HybridDB (same parent directory). Only the
directory name changes: `agent_memory/` → `agent_journal/`. This preserves the
existing behavior for both relative and absolute paths.

Future option (not for this spec): add `journal_path: str | None = None` to
`MemoryCore.__init__()` for explicit overrides.

### 2C. `MemoryCore` docstring update

```python
class MemoryCore:
    """Message store with journal compilation.

    Two tiers:
      - HybridDB db: raw message storage, search, fetch, filters
      - AgentJournal: compiled markdown pages at ./agent_journal/

    Usage:
        core = MemoryCore(path="./data")
        tid = core.ingest("user", "I like coffee", session_id="s1")
        core.ingest("assistant", "Great!", session_id="s1")
        await core.compile_turn(turn_id=tid, timestamp="10:30", title="Coffee Chat")
        results = core.search("coffee")           # raw messages
        hits = core.search_journal("coffee")       # compiled journal
    """
```

The `MemoryCore` name is kept — it's already in use and the top-level identity of
the library. The docstring clarifies the two-tier model.

### 2D. Keep `MemoryCore` + its method names

Not changing:
- `MemoryCore.__init__()` — still takes `path: str | None = None`
- `ingest()` — still returns `turn_id`
- `ingest_turn()` — already returns `tid`; dev captures or ignores
- `search()`, `search_enhanced()` — raw message search
- `fetch()`, `fetch_all()` — raw message query
- `compile_turn()` — renamed internally but public API unchanged
- `search_memory()` → `search_journal()` (see §2E)

### 2E. `search_memory()` → `search_journal()`

The only public method with "memory" in its name that should say "journal".

```python
# Before
def search_memory(self, query: str, limit: int = 5) -> list[SearchHit]:

# After
def search_journal(self, query: str, limit: int = 5) -> list[SearchHit]:
```

## 3. Files Changed

### 3.1 Renamed directory + files (git mv)

```
coremem/agent_memory/           →  coremem/agent_journal/
coremem/agent_memory/__init__.py   →  coremem/agent_journal/__init__.py
coremem/agent_memory/bundle.py     →  coremem/agent_journal/bundle.py
coremem/agent_memory/compiler.py           →  coremem/agent_journal/compiler.py
coremem/agent_memory/llm_compiler.py        →  coremem/agent_journal/llm_compiler.py
coremem/agent_memory/dreaming.py           →  coremem/agent_journal/dreaming.py
coremem/agent_memory/rebuild_index.py      →  coremem/agent_journal/rebuild_index.py
coremem/agent_memory/embeddings.py         →  coremem/agent_journal/embeddings.py
coremem/agent_memory/reranker.py           →  coremem/agent_journal/reranker.py
```

### 3.2 Modified files

| File | Change |
|---|---|
| `coremem/core.py` | imports, variables, path, `search_memory()`→`search_journal()` |
| `coremem/agent_journal/__init__.py` | class names in imports + `__all__` |
| `coremem/agent_journal/bundle.py` | class name, frontmatter strings, SCHEMA.md refs |
| `coremem/agent_journal/compiler.py` | class name |
| `coremem/agent_journal/llm_compiler.py` | class name, internal refs |
| `coremem/agent_journal/rebuild_index.py` | internal refs |
| `coremem/agent_journal/dreaming.py` | log messages, internal refs |
| `tests/test_memorypack.py` → `tests/test_agent_journal.py` (rename) | class refs |
| `tests/test_memorypack_compiler.py` → `tests/test_agent_journal_compiler.py` (rename) | class refs |
| `tests/test_memorypack_eval_internal.py` → `tests/test_agent_journal_eval_internal.py` (rename) | class refs |
| `tests/test_memorypack_eval_longmemeval.py` → `tests/test_agent_journal_eval_longmemeval.py` (rename) | class refs |
| `scripts/eval_memorypack_internal.py` → `scripts/eval_agent_journal_internal.py` (rename) | imports |
| `scripts/eval_memorypack_llm_compiler.py` → `scripts/eval_agent_journal_llm_compiler.py` (rename) | imports |
| `scripts/eval_memorypack_longmemeval.py` → `scripts/eval_agent_journal_longmemeval.py` (rename) | imports |
| `scripts/save_stage4_output.py` | internal refs |

### 3.3 Files NOT changed

- `coremem/__init__.py` — only exports `MemoryCore`, no AgentMemory names
- `coremem/types.py`, `coremem/heuristics.py`, `coremem/query.py`, etc.
- `docs/superpowers/specs/` — spec docs reference old names but are historical
- `pyproject.toml` — no class names in config
- `coremem/agent_journal/reranker.py` — `CrossEncoderReranker` already generic
- `coremem/agent_journal/embeddings.py` — `EmbeddingIndex` already generic
- `coremem/agent_journal/dreaming.py` — `dream()` function already generic
- `coremem/agent_journal/rebuild_index.py` — `rebuild_index()` function already generic
- `compute_agent_context_hash()` — about agent context, not memory

### 2F. `MemoryPack` → `AgentJournal` (old POC name)

All remaining `MemoryPack` references in docstrings, LLM prompts, comments, error
messages, and generated file headers are renamed to `AgentJournal`. This includes
LLM system prompts ("You are a MemoryPack compiler" → "You are an AgentJournal
compiler"), index headers (`# MemoryPack Index` → `# AgentJournal Index`), and
the `agent_memory-turn` JSON code block marker (`agent_memory-turn` →
`agent_journal-turn`).

The `agent_memory-turn` marker is used in reference turn files — both the writer
and parser are updated together. Existing reference turn files on disk will fail
parsing, but these are generated artifacts (not committed), so backward compat
is not required.

### 2G. `SCHEMA_VERSION` update

```
# Before (bundle.py:27)
SCHEMA_VERSION = "memorypack-poc-0.1"

# After
SCHEMA_VERSION = "agent-journal-0.1"
```

### 2H. Frontmatter migration for existing journal files

The frontmatter key `agent_memory_version` changes to `agent_journal_version`.
Existing daily pages will fail `lint()` until updated. Two-step migration:

1. **During rename**: `lint()` accepts both keys — if `agent_journal_version` is
   missing, fall back to `agent_memory_version`. This keeps old files valid.
2. **Next cycle**: recompile all turns to write `agent_journal_version` in the
   frontmatter.

## 4. Migration Order

1. `git mv coremem/agent_memory/ coremem/agent_journal/`
2. Update all class names + imports in `.py` files (mechanical find-replace)
3. Update path logic + `search_journal()` in `core.py`
4. `git mv tests/test_memorypack*.py` + update class refs
5. `git mv scripts/eval_memorypack*.py` + update imports
6. Run tests: `uv run python3 -m pytest tests/ -q` (expect no failures)

## 5. Verification

- All 97 tests pass after rename
- No imports reference `agent_memory` anywhere in `coremem/`, `tests/`, or `scripts/`
- `core.search_journal("query")` returns same results as `core.search_memory()`
- `AgentJournal(path="./test_journal").daily_dir` resolves to `./test_journal/daily/`

## 6. MemoryCore path clarification

The `path` parameter in `MemoryCore(path="./memory")` creates a HybridDB at
`./memory/` (a directory containing SQLite tables). This is a localStorage path,
not "memory" in the neural sense. The docstring now explains this clearly.

A future rename of `path` to `db_path` or `storage_path` is possible but not in
this spec — it would break every existing caller.
