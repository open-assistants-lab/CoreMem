# Observer Pipeline Redesign: Production Memory Quality

2026-06-04

## 1. Motivation

The current Observer pipeline achieves 10.6 obs/q with 0% hallucination on LongMemEval, but extracts **all facts** from conversations — including temporary, conversational, and one-off facts that don't belong in long-term memory.

`docs/agent_memory_design.md` describes a fundamentally different approach: **extract durable memories, not every fact**. Memory types, durability filtering, semantic dedup, temporal supersession, and source traceability are the hallmarks of production-quality memory systems.

### Current vs Target

| Aspect | Current | Target |
|--------|---------|--------|
| Extraction philosophy | Extract everything | Extract durable, useful facts |
| Output classification | Importance (0.0-1.0) | Memory type (12 types) |
| Dedup | String similarity (0.75) | Search-store + LLM relationship classification |
| Fact evolution | Store all independently | Supersede, merge, contradict |
| Source trace | Not stored | `source_message_ids[]` per observation |
| Lifecycle | insert only | candidate → active → superseded → archived |

## 2. Architecture

```
Phase 1: 6-LF Parallel Enumeration (unchanged)
  ├── LF1: Named entities
  ├── LF2: Actions/habits/plans
  ├── LF3: Preferences/states
  ├── LF4: Temporal/quantitative
  ├── LF5: Sentiment/emotional
  └── LF6: Possessions/ownership
  → Union + fuzzy dedup (SequenceMatcher > 0.75)

Phase 2: Per-entity Extraction (unchanged)
  └── batch_size=4, align_quote + string dedup (0.75)
  └── source_message_ids: collected from message.id of all input messages
      (the full conversation passed to extract_relations)

Phase 3: Per-message Gleaning (unchanged)
  └── 6 calls, one per user-message pair
  └── source_message_ids: collected from message.id of the specific pair

Phase 4: Classification + Durability Filter (NEW — batched LLM call)
  ├── Batch observations from phases 2-3 into groups of 20
  ├── For each observation, classify:
      ├── memory_type (12 types)
      ├── durability (durable | temporary)
      └── sensitivity (normal | personal | sensitive)
  └── Store all observations, but set status:
      ├── durable → status = "candidate" (proceed to dedup)
      └── temporary → status = "archived" (kept for eval, excluded from retrieval)

Phase 5: Dedup + Merge (NEW — LLM-based comparison)
  For each batch of durable observations:
  1. Fetch candidate matches from existing store:
     a. Exact content match → duplicate, discard
     b. Keyword overlap (shared 3+ words, >50% word overlap) → candidate
     c. Recent observations (last 30 days, same user) → candidate
  2. For groups of 5 new-obs/candidate pairs, LLM classifies relationship:
     ├── duplicate  → discard new
     ├── refine     → merge content into old, append source_message_ids
     ├── supersede  → mark old status=superseded, set old.valid_to=now, insert new as candidate
     ├── contradict → create conflict record, flag for review
     └── new        → insert as candidate
  3. Write memory_events log for each action
```

**Pipeline ordering rationale:** Gleaning (Phase 3) produces new observations that need classification and dedup. Running gleaning before classification ensures ALL observations go through the same processing pipeline.

## 3. Memory Types

From `agent_memory_design.md` Section 4:

| Type | Signal | Example | LF mapping |
|------|--------|---------|------------|
| `profile` | Stable user identity | "User works at Anthropic" | LF1 (entities) |
| `preference` | Style, tools, workflow | "User prefers Rust over Python" | LF3 (preferences) |
| `project` | Active/historical work | "User building POS app" | LF2 (actions) |
| `decision` | Chosen direction | "User chose self-host over cloud" | LF3 (preferences) |
| `technical_stack` | Languages, tools, infra | "User uses PostgreSQL" | LF6 (possessions) |
| `business_context` | Durable business info | "Gong Cha has 160 stores" | LF1 (entities) |
| `people` | Important recurring people | "Sarah is user's wife" | LF1 (entities) |
| `constraint` | Hard restrictions | "Public AI tools need approval" | LF3 (preferences) |
| `workflow` | Repeated processes | "Inventory via stocktakes" | LF2 (actions) |
| `episodic` | Past experience/event | "Attended music festival" | LF2 (actions) |
| `procedural` | Agent behavior rules | "Ask approval for sensitive data" | n/a |
| `sentiment` | Emotional reactions | "Found festival amazing" | LF5 (sentiment) |

**Classification prompt (Phase 4):**

```
For each observation, classify and filter:

CLASSIFY:
1. memory_type: one of the 12 types above
2. durability:
   - "durable": useful beyond this conversation, persists across sessions
   - "temporary": context-specific, one-off, weather-query, single-task specific
3. sensitivity:
   - "normal": business, technical, general preferences
   - "personal": family, logistics, named people
   - "sensitive": health, legal, political, precise location, children

FILTER:
Mark temporary observations as durability="temporary" — they are still stored (status="archived") but excluded from retrieval and dedup. Only "durable" observations proceed to Phase 5.

Examples:
- "User commutes 45 minutes each way" → profile, durable, normal
- "User asks about weather this weekend" → preference, temporary, normal
- "User prefers audiobooks over e-books" → preference, durable, normal
- "User wants note-taking app recs" → preference, temporary, normal
- "User saw The Lumineers at a festival" → episodic, durable, normal
- "User found the festival amazing" → sentiment, durable, normal
- "User's address is 123 Main St" → profile, durable, sensitive
```

## 4. Supersession Patterns

Facts evolve over time. The new pipeline detects this evolution:

| Pattern | Old | New | Action |
|---------|-----|-----|--------|
| Same fact | "User uses Python" | "User uses Python" | Duplicate — discard |
| Refined | "User uses Python" | "User uses Python for data work" | Merge — enrich old |
| Superseded | "User uses n8n cloud" | "User self-hosts n8n" | Supersede — archive old |
| Contradicted | "User prefers Mac" | "User switched to Linux" | Conflict — flag for review |
| Time-invalid | "User's job: intern" | "User's job: senior eng" | Supersede (time-based) |

## 5. Schema Changes

CoreMem uses HybridDB which manages tables from Python schema dicts. `MemoryStore._OBSERVATIONS_SCHEMA` in `coremem/memory_store.py` is the single source of truth.

### New columns (add to `_OBSERVATIONS_SCHEMA`)

```python
_OBSERVATIONS_SCHEMA = {
    # ... existing columns ...
    "memory_type": "TEXT",              # NEW: one of 12 types
    "durability": "TEXT DEFAULT 'durable'",
    "sensitivity": "TEXT DEFAULT 'normal'",
    "status": "TEXT DEFAULT 'candidate'",
    "confidence": "REAL DEFAULT 0.800",
    "valid_from": "TEXT",               # ISO timestamp
    "valid_to": "TEXT",
    "source_message_ids": "TEXT DEFAULT '[]'",  # JSON array
    "superseded_by": "TEXT",            # observation ID
}
```

### New tables (via `self._db.create_table()` in MemoryStore)

```python
_MEMORY_EVENTS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "memory_id": "TEXT NOT NULL",
    "event_type": "TEXT NOT NULL",  # created, updated, merged, superseded, contradicted
    "old_value": "TEXT",            # JSON
    "new_value": "TEXT",            # JSON
    "source_message_id": "TEXT",
    "created_at": "TEXT NOT NULL",
}

_MEMORY_CONFLICTS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "memory_id_a": "TEXT NOT NULL",
    "memory_id_b": "TEXT NOT NULL",
    "conflict_type": "TEXT NOT NULL",
    "resolution_status": "TEXT DEFAULT 'unresolved'",
    "created_at": "TEXT NOT NULL",
    "resolved_at": "TEXT",
}
```

### Migration

`migrations/v0_5_to_v0_6.py`: ALTER TABLE to add new columns, CREATE TABLE for events and conflicts.

For new databases, HybridDB's `create_table` picks up the updated schema automatically. For existing databases, the migration adds columns.

### `importance` vs `confidence`

- `importance` (0.0-1.0): how significant the fact is to the user's profile. Set by Reflector. Existing field, unchanged.
- `confidence` (0.0-1.0): how explicitly the fact was stated (verbatim vs inferred). Set during classification (Phase 4) by the LLM.

Both fields coexist: a fact can be high confidence (explicitly stated) but low importance (trivial), or low confidence (inferred) but high importance (job change implicit in context).

### No embedding changes needed

HybridDB already stores `embedding` as TEXT (JSON float array) and provides `search()` with keyword + semantic search. The dedup phase uses `store.search_observations()` — no new infrastructure.

## 6. Expected Impact

| Metric | Current | After redesign |
|--------|---------|---------------|
| Obs/q (raw extraction) | 10.6 | 8-10 |
| Obs/q (after durability filter) | n/a | ~5-7 |
| Obs/q (after dedup + merge) | n/a | ~4-6 |
| Hallucination | 0% | 0% |
| Duplicate rate (string) | ~30% | 0% (semantic catches all) |
| Memory type coverage | Untagged | 12 types, every obs tagged |
| Source traceability | None | Per-observation src msg IDs |
| Supersession | None | Full lifecycle |
| Sensitivity classification | None | 3 levels |

## 7. Evaluation

LongMemEval measures fact coverage, but the redesigned pipeline optimizes for memory quality. Need separate eval dimensions:

### Extraction quality (same as before)
- 0% hallucination (alignment gate, no regression)
- Obs/q after durability filter (target: ≥4/q durable facts)

### Classification accuracy
- Manual review of 100 observations across 10 conversations
- Measure: memory_type accuracy, durability accuracy, sensitivity accuracy
- Target: ≥85% accuracy on all three

### Dedup effectiveness
- Manual review of 50 dedup decisions
- Measure: false-positive dedup rate (distinct facts merged), false-negative rate (duplicates missed)
- Target: <5% false positive, <10% false negative

### Supersession detection
- Synthetic conversations with known factual evolution
- Example: "I use n8n cloud" → next week "I migrated to self-hosted"
- Verify old fact gets `status=superseded`, new fact active, event logged

### End-to-end manual review
- 5 conversations, full pipeline run
- Human annotator reviews every observation
- Flag: wrong type, missed durable fact, incorrectly deduped, missed supersession
