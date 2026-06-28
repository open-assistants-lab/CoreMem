# Spec: Rename message search APIs and add safe AgentJournal compile automation

## Why

Developers now have two retrieval surfaces:

- Raw HybridDB message search in `MemoryCore.search()` and `MemoryCore.search_enhanced()`.
- Compiled AgentJournal markdown search in `MemoryCore.search_journal()`.

After adding AgentJournal, the bare name `search()` is ambiguous. A developer cannot tell whether it searches raw messages or compiled journal pages without reading implementation. Compilation is also still fully manual: `compile_turn(turn_id)` works, but there is no safe way to batch or automate compilation without risking duplicate daily journal entries.

## Current Code Evidence

- `coremem/core.py:187` defines `MemoryCore.search()`, which searches the `messages` HybridDB table and returns `SearchResult` objects wrapping raw `Memory` rows.
- `coremem/core.py:213` defines `MemoryCore.search_enhanced()`, which expands queries, searches raw messages more deeply, applies heuristics, MMR diversification, and reranking.
- `coremem/core.py:394` defines `MemoryCore.compile_turn()`, which loads rows by `turn_id`, derives display timestamp, and calls `AgentJournalLLMCompiler.compile_session()`.
- `coremem/agent_journal/llm_compiler.py:161` defines `compile_session()`, which writes/appends a daily journal section and returns `AgentJournalCompileResult`.
- There is no persisted record that a `turn_id` has already been compiled.

## Goals

1. Make raw message search names explicit.
2. Preserve `search_journal()` as the compiled AgentJournal search surface.
3. Keep `compile_turn()` as the manual primitive.
4. Add idempotency so automation cannot append duplicate daily sections for the same unchanged turn.
5. Add convenience automation methods that developers can call from end-of-turn hooks, batch jobs, or CLI wrappers.

## Non-Goals

- Do not auto-compile inside `ingest()` or `ingest_turn()`.
- Do not add a background scheduler or worker loop in this slice.
- Do not implement section replacement/editing inside existing daily files.
- Do not keep old `search()` or `search_enhanced()` aliases unless we decide this release must preserve external API compatibility.

## API Changes

### Message Search Rename

Rename:

```python
core.search(...)
```

to:

```python
core.search_messages(...)
```

Rename:

```python
core.search_enhanced(...)
```

to:

```python
core.search_messages_deep(...)
```

Keep:

```python
core.search_journal(...)
```

Final shape:

```python
core.search_messages("coffee")
core.search_messages_deep("when did I mention Ethiopian beans?")
core.search_journal("coffee preference")
```

### Compile Primitive

Change `compile_turn()` to return the lower-level compiler result when it writes, and `None` when there is nothing to do.

```python
async def compile_turn(
    self,
    turn_id: str,
    timestamp: str | None = None,
    title: str | None = None,
    *,
    force: bool = False,
) -> AgentJournalCompileResult | None:
```

Behavior:

- If no messages exist for `turn_id`, return `None`.
- If the same source hash is already compiled and `force=False`, return `None`.
- If the turn was compiled before but the source hash changed and `force=False`, raise `AgentJournalError` explaining that the turn changed after compilation.
- If `force=True`, append a new daily section and update the compiled-turn ledger.
- If compilation succeeds, return `AgentJournalCompileResult` and write/update the ledger.
- If LLM generation, validation, or file writing fails, do not write/update the ledger.

### Compile Convenience Methods

Add:

```python
async def compile_latest_turn(
    self,
    session_id: str,
    timestamp: str | None = None,
    title: str | None = None,
    *,
    force: bool = False,
) -> AgentJournalCompileResult | None:
```

Behavior:

- Finds the most recent `turn_id` for the given `session_id` by message timestamp.
- Calls `compile_turn()`.
- Returns `None` if the session has no messages or the latest turn is already compiled unchanged.

Add:

```python
async def compile_uncompiled_turns(
    self,
    *,
    session_id: str | None = None,
    limit: int = 50,
) -> dict[str, object]:
```

Return shape:

```python
{
    "compiled": ["turn_a", "turn_b"],
    "skipped": ["turn_c"],
    "changed": ["turn_d"],
    "errors": [{"turn_id": "turn_e", "error": "..."}],
}
```

Behavior:

- Selects candidate turn IDs from the `messages` table ordered by first message timestamp ascending.
- Filters to a single `session_id` when provided.
- Compiles never-compiled turns.
- Skips unchanged already-compiled turns.
- Reports previously compiled but changed turns under `changed` and does not recompile them.
- Catches per-turn exceptions and records them under `errors` instead of aborting the whole batch.

## Data Model

Add a `compiled_turns` table in `MemoryCore._ensure_tables()`:

```sql
CREATE TABLE compiled_turns (
    turn_id TEXT PRIMARY KEY,
    source_hash TEXT NOT NULL,
    compiled_at TEXT NOT NULL,
    daily_path TEXT NOT NULL,
    message_count INTEGER NOT NULL
)
```

Add an index if needed later, but `turn_id` primary key is enough for this slice.

## Source Hash

Compute `source_hash` from ordered turn rows:

```python
[
    {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "session_id": row["session_id"],
        "ts": row["ts"],
    }
    for row in rows_ordered_by_ts
]
```

Serialize with stable JSON and hash with SHA-256.

Do not include generated title, display timestamp override, daily file path, or LLM output in the source hash. The hash represents raw message input only.

## Implementation Plan

1. Rename `MemoryCore.search()` to `search_messages()`.
2. Rename `MemoryCore.search_enhanced()` to `search_messages_deep()`.
3. Update tests and docs that call the old names.
4. Add `compiled_turns` table creation to `_ensure_tables()`.
5. Add private helpers:
   - `_turn_rows(turn_id: str) -> list[dict[str, Any]]`
   - `_turn_source_hash(rows: list[dict[str, Any]]) -> str`
   - `_compiled_turn(turn_id: str) -> dict[str, Any] | None`
   - `_record_compiled_turn(turn_id, source_hash, result, message_count) -> None`
6. Update `compile_turn()` to be idempotent and return `AgentJournalCompileResult | None`.
7. Add `compile_latest_turn(session_id, ...)`.
8. Add `compile_uncompiled_turns(...)`.
9. Update changelog with breaking search API rename and compile automation additions.

## Tests

Add/update tests in `tests/test_core.py`:

- `test_search_messages_matches_current_search_behavior`
- `test_search_messages_deep_returns_raw_message_results`
- `test_old_search_names_are_removed`
- `test_compile_turn_records_compiled_turn`
- `test_compile_turn_skips_unchanged_compiled_turn`
- `test_compile_turn_raises_when_compiled_turn_source_changed_without_force`
- `test_compile_turn_force_recompiles_changed_turn`
- `test_compile_latest_turn_compiles_most_recent_session_turn`
- `test_compile_uncompiled_turns_compiles_each_turn_once`
- `test_compile_uncompiled_turns_reports_errors_without_aborting_batch`

Use a fake async `core._journal_compiler.compile_session` in tests to avoid real LLM calls.

## Developer Experience After This Change

Manual end-of-turn flow:

```python
turn_id = core.ingest("user", "I like coffee", session_id="s1")
core.ingest("assistant", "Noted.", session_id="s1")
await core.compile_turn(turn_id)
```

Convenience end-of-session flow:

```python
await core.compile_latest_turn(session_id="s1")
```

Batch/cron flow:

```python
summary = await core.compile_uncompiled_turns(limit=100)
```

Search flow:

```python
raw_hits = core.search_messages("coffee")
deep_raw_hits = core.search_messages_deep("when did I say I like Ethiopian beans?")
journal_hits = core.search_journal("coffee preference")
```

## Rollback

If the compile automation causes problems:

- Revert the new convenience methods and ledger checks.
- Leave `compiled_turns` table unused; it is additive.
- Manual `compile_turn()` can be restored to always append sections.

If the search rename causes too much breakage:

- Add temporary aliases:
  - `search = search_messages`
  - `search_enhanced = search_messages_deep`
- Mark aliases deprecated in docstrings and changelog.

## Self-Review

### Finding 1: The search rename is breaking

Severity: Medium.

The spec recommends removing `search()` and `search_enhanced()` instead of keeping aliases. That matches the current 0.10 breaking-change posture, but it will break any external caller already using `MemoryCore.search()`. If this package has real external consumers today, add one-release aliases and deprecation warnings instead.

Recommendation: keep the spec as breaking unless user confirms backwards compatibility matters.

### Finding 2: Changed-source behavior is intentionally conservative

Severity: Medium.

If a turn is compiled too early, then assistant/tool messages are added later, `compile_turn()` will raise on source hash mismatch. That prevents silent duplicate daily sections, but it puts recovery on the developer via `force=True`.

Recommendation: acceptable for MVP. Section replacement/editing is a separate issue.

### Finding 3: Automation still requires an external trigger

Severity: Low.

This spec does not add a scheduler or hook into `ingest()`. Developers still need to call `compile_latest_turn()` at response completion or call `compile_uncompiled_turns()` from their own worker/cron.

Recommendation: correct boundary. Hidden LLM calls inside `ingest()` would be surprising and hard to test.

### Finding 4: Batch error shape should stay stable

Severity: Low.

`compile_uncompiled_turns()` returns a plain dict. That is easy to consume but less typed than a dataclass.

Recommendation: use a dict for MVP to avoid adding public dataclasses too early. Promote to a typed result later if callers need it.

### Finding 5: Daily section duplicates are prevented only for unchanged source hashes

Severity: Low.

The ledger prevents unchanged duplicate compiles, but `force=True` can still append duplicate or superseding sections.

Recommendation: document `force=True` as explicit append/recompile behavior. Do not attempt daily-file surgery in this slice.

## Acceptance Criteria

- Developers can no longer call ambiguous `core.search()` in tests or examples.
- `core.search_messages()` returns raw message `SearchResult` values.
- `core.search_messages_deep()` returns raw message `SearchResult` values using the enhanced pipeline.
- `core.search_journal()` continues searching compiled AgentJournal files.
- Calling `compile_turn()` twice for an unchanged turn appends only one daily section.
- Calling `compile_uncompiled_turns()` twice compiles each unchanged turn only once.
- A changed compiled turn is reported clearly and not silently duplicated.
- Full test suite passes.
