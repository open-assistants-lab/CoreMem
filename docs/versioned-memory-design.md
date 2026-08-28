# Versioned Memory — Design Spec

**Status:** proposed · **Target:** CoreMem 0.14.0 (opt-in) → default after perf gate
**Depends on:** hybriddb>=0.6.0 (versioned tables, tamper-evident hash chain)
**Related:** `docs/retrieval-experiments.md` (this is a product capability, NOT a retrieval lever)

---

## 1. Goal

Give CoreMem stores git-like memory governance: a tamper-evident audit trail of
every memory change, point-in-time reads, named checkpoints, and rollback — so an
agent (or its operator) can answer "what did memory believe at time T, what has
changed since, prove nothing was tampered with, and undo a bad write."

This is the trust/differentiator layer (AML leaderboard: "memory governance",
"epistemic safety"). It does **not** move retrieval metrics — see Non-goals.

## 2. Spike findings (2026-08-28, hybriddb 0.6.0, CoreMem batched path)

De-risking experiments already run:

| Assumption | Result |
|---|---|
| CoreMem's ingest (`insert_batch(sync=False)` + `_flush_journal_batched`) records history | ✅ 200 rows → 200 chain events; `verify_chain` valid |
| Search unaffected on versioned tables | ✅ current-data-only indexing unchanged |
| `log()/history()/verify_chain()` work through CoreMem's paths | ✅ (`log()` defaults to 100 entries — pass `limit`) |
| **`raw_query("DELETE FROM messages…")` bypasses history — NO tombstone** | ❌ **must fix**: `delete_messages`/`clear()` must route through `db.delete()` which records tombstones (`db.delete('messages', pk, sync=True)` verified: tombstone + chain valid) |
| Ingest overhead on CoreMem's batched path | ✅ negligible at spike scale (embedding dominates); hybriddb's own 100k-row benchmark: ~13% |

⚠️ `raw_query` bypasses all CRUD hooks: any INSERT/UPDATE/DELETE issued through it
is **untracked**. CoreMem's tracked-write discipline (spec §5) is therefore a
correctness requirement, not a nicety — an untracked DELETE would make history
diverge from reality silently.

## 3. Non-goals (v1)

- **No recall improvement claim.** Versioning is a trust/governance capability;
  it must not be sold as a retrieval lever.
- **No in-place migration** of existing (non-versioned) stores. Legacy stores
  keep working un-versioned; a rebuild tool is future work (create versioned
  table + copy rows + re-embed).
- **No `fork`** (deferred by hybriddb itself; requires full re-embedding).
- **agent_journal / message_facts are not rolled back** (see §6.3, §6.4).

## 4. Schema decisions

| Table | Versioned | Rationale |
|---|---|---|
| `messages` | **Yes** | The memory itself — audit, rollback, provenance live here |
| `journal_records` | Yes | Compiled journals are memories too; same governance story |
| `message_facts` (side) | **No** | Derived data keyed by message id; versioning would double size for no audit value. Rollback staleness handled in §6.4 |
| `compiled_turns` (side) | No | Bookkeeping (source hashes, compilation bookkeeping) |

New stores only: `MemoryCore(versioned=True)` passes `versioned=True, hash_chain=True`
to `create_table(...)` in `_ensure_tables`. The `turn_id` column is included at
creation, so the legacy `ALTER TABLE` migration never fires on versioned stores
(its guardrail — schema changes rejected on versioned tables — is the enforcement).

**Upgrade path note:** an existing non-versioned store opened with
`versioned=True` keeps its non-versioned tables (versioning is create-time
only). `MemoryCore` logs a warning when the flag is set but the store predates
it. No error, no surprise rewrite.

## 5. Public API

```python
core = MemoryCore(path, versioned=False)   # opt-in in 0.14.0

core.checkpoint_memory(label)          # → named restore point (messages + journal_records)
core.rollback_memory(label=… | seq=…)  # → re-applies state as NEW versions; audit intact
core.memory_log(limit)                 # → change log (op, id, author, ts, seq)
core.memory_history(message_id)        # → per-row version timeline
core.memory_diff(from_seq, to_seq)     # → added/removed/changed rows
core.as_of_memory(seq)                 # → point-in-time message slice
core.verify_memory_chain()             # → {'valid': bool, 'first_broken_seq': …}
core.author = "agent-7"                # recorded per history event
```

Ingest/delete are tracked automatically once `versioned=True`:

- `ingest()`/`ingest_many()` → inserts recorded (validated in spike)
- `delete_messages()` / `clear()` → **refactor from `raw_query` to `db.delete()`**
  (tombstones land in history; spike §2 shows the bypass)

### MCP surface

Three governance tools (agents self-audit; operators audit externally):
`memory_history`, `memory_rollback(label|seq)`, `memory_verify`. Rollback via
MCP requires `confirm: true` (destructive-ish operation), mirroring
`delete_messages`' ergonomics.

## 6. Consistency & edge cases

1. **Tracked writes only.** All mutating paths must go through the CRUD layer
   (`insert_batch`, `update`, `delete`) — never `raw_query`. Add an integration
   test asserting `verify_chain` stays valid after every public mutator.
2. **Deletes + memory_facts:** facts are a side table keyed by message id; a
   tombstoned message leaves its fact behind (stale orphan). v1 behavior:
   leave them (facts are derived, unversioned, cheap to ignore); a later
   enhancement can cascade-delete facts inside the same logical action.
   Documented known limitation.
3. **Rollback + journaled vectors:** hybriddb handles Chroma restoration
   (deletions for removed rows, re-embeds for restored content). CoreMem's
   `_flush_journal_batched` shadow may cache stale vectors during a rollback —
   rollback must run with the stock embedding path (call `db.rollback(sync=True)`
   directly; do not run inside a shadowed flush).
4. **agent_journal is file-based** (`daily/*.md`) and NOT versioned — rollback
   of `messages` does not undo compiled daily pages. Document the boundary;
   recompilation is idempotent by `source_hash`, so re-running compilation
   post-rollback heals the journals.
5. **Checkpoint before bulk ingest** is the recommended workflow:
   `checkpoint_memory("pre-import")` → `ingest_many(...)` → verify →
   `rollback_memory("pre-import")` on failure.
6. **prune**: retention may drop old chain anchors; pruned checkpoints become
   un-rollbackable. Never expose `prune` via MCP.
7. **verify_chain is O(chain length)** — fine for interactive use; do not call
   per-request. Optional periodic verify in the CLI `stats` command.

## 7. Performance gate ("no huge impact"), then default flip

Benchmark on CoreMem's real paths (not raw hybriddb CRUD), versioned=False vs
True, same machine, batched ingest:

| Metric | Acceptance for default=True |
|---|---|
| `ingest_many` throughput (10k messages) | ≥ 85% of non-versioned (≤15% overhead) |
| `recall()` / `_search_messages_decomposed` latency | within ±5% (search path must be untouched) |
| recall metrics (S stratified-56 A/B) | identical (0/34 metric rows differ) |
| Storage growth after 10k-message ingest | ≤ 2.2× messages-table size (1 event/insert); measured + documented |
| `rollback_memory` (append-heavy: remove 1k ingested rows) | ≤ ingest time of those rows (Chroma deletions, few re-embeds) |
| `verify_memory_chain` at 100k events | ≤ 30 s, O(n) documented |

Method: one script (`scripts/bench_versioned_memory.py`), both arms in the same
run, medians of 3; results appended to this doc. **If all bounds pass → 0.15.0
flips `versioned=True` as the default** (legacy stores unaffected; document the
re-embed-free difference — new stores only).

## 8. Rollout

1. **0.14.0** — opt-in `MemoryCore(versioned=True)`, API + MCP tools, tracked-
   writes refactor (deletes), tests (spike assertions as regression tests),
   perf bench script + results.
2. **Perf gate** (§7). If pass → step 3; if fail → stay opt-in, document why.
3. **0.15.0** — `versioned=True` default; legacy stores unchanged; CHANGELOG
   migration note. Since backup/rollback requires the flag, the CHANGELOG must
   state clearly: *stores created before 0.15.0 have no memory history*.

## 9. Test plan

- Unit: versioned create (`versioned=True`), history recorded via
  `ingest_many`, tombstones via `delete_messages`, rollback restores prior
  message set, `verify_chain` after every public mutator, `as_of` slicing,
  `author` recorded, message_facts staleness documented (§6.2).
- Integration: MCP round-trip for the three new tools.
- Negative: `versioned=True` on pre-existing non-versioned store → reuse +
  warning (§4).
- Existing suite must stay green with the flag OFF (default behavior unchanged).

## 10. Open questions

- Default `log()` limit — expose pagination or full log? (hybriddb default 100)
- Should `checkpoint_memory` auto-checkpoint on `close()`? (probably not —
  explicit beats implicit)
- `diff` output shape for MCP (size-bounded?)
- Does `rollback` need to invalidate/refresh `message_facts` for restored rows?
  (v1: document staleness; v2: cascade via versioned facts table)