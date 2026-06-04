# Observer Pipeline Redesign: Production Memory Quality

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Observer pipeline from "extract everything" to "extract durable, typed, traceable memories" with classification, durability filtering, semantic dedup, and supersession detection.

**Architecture:** Add two new phases after existing extraction: Phase 4 (classification + durability filter, batched LLM call) and Phase 5 (dedup + merge, keyword overlap candidates + LLM relationship classification). New schema columns and memory_events/memory_conflicts tables.

**Tech Stack:** Python 3.11+, coremem (HybridDB, MemoryStore, Observer), deepseek-v4-flash, pytest

**Spec:** `docs/superpowers/specs/2026-06-04-observer-pipeline-redesign.md`

**Worktree:** `/Users/eddy/Developer/Python/CoreMem/.worktrees/0.5.0-pipeline-redesign`

---

## File Map

| File | Role |
|------|------|
| `coremem/migrations/v0_5_to_v0_6.py` | Schema migration (new) |
| `coremem/memory_store.py` | Updated schema, new methods for events/conflicts/search |
| `coremem/observer.py` | Phase 4 classification, Phase 5 dedup, source_message_ids |
| `coremem/dedup.py` | Dedup + merge logic (new, extracted from observer for testability) |
| `tests/test_pipelines.py` | Updated pipeline tests |
| `tests/test_dedup.py` | Dedup + merge unit tests (new) |
| `tests/test_memory_store.py` | Updated store tests for new methods |
| `benchmarks/longmemeval/observer_eval.py` | Keep same, observe quality shift |

---

### Task 1: Schema migration — add new columns and tables

**Files:**
- Create: `coremem/migrations/v0_5_to_v0_6.py`
- Modify: `coremem/memory_store.py:25-41` (update _OBSERVATIONS_SCHEMA)
- Modify: `coremem/memory_store.py:104-130` (add events + conflicts tables)

- [ ] **Step 1: Update `_OBSERVATIONS_SCHEMA` with new columns**

```python
_OBSERVATIONS_SCHEMA = {
    "id":              "TEXT PRIMARY KEY",
    "kind":            "TEXT NOT NULL DEFAULT 'fact'",
    "content":         "LONGTEXT",
    "source_quote":    "TEXT",
    "source_fact_ids": "TEXT NOT NULL DEFAULT '[]'",
    "source_message_ids": "TEXT DEFAULT '[]'",    # NEW: trace to raw chat messages
    "referenced_date": "TEXT",
    "observation_ts":  "TEXT NOT NULL",
    "user_id":         "TEXT",
    "agent_id":        "TEXT",
    "session_id":      "TEXT",
    "alignment_tier":        "TEXT",
    "alignment_confidence":  "REAL",
    "importance":      "REAL",
    "confidence":      "REAL DEFAULT 0.800",      # NEW: how explicitly stated
    "memory_type":     "TEXT",                    # NEW: one of 12 types
    "durability":      "TEXT DEFAULT 'durable'",  # NEW: durable | temporary
    "sensitivity":     "TEXT DEFAULT 'normal'",   # NEW: normal | personal | sensitive
    "status":          "TEXT DEFAULT 'candidate'",# NEW: candidate | active | superseded | archived
    "valid_from":      "TEXT",                    # NEW: ISO timestamp
    "valid_to":        "TEXT",                    # NEW: ISO timestamp
    "superseded_by":   "TEXT",                    # NEW: observation ID
    "entities":        "TEXT NOT NULL DEFAULT '[]'",
    "reflected":       "INTEGER NOT NULL DEFAULT 0",
    "embedding":       "TEXT",
}
```

- [ ] **Step 2: Add new table schemas**

Add after `_REFLECTIONS_SCHEMA`:

```python
_MEMORY_EVENTS_SCHEMA = {
    "id": "TEXT PRIMARY KEY",
    "memory_id": "TEXT NOT NULL",
    "event_type": "TEXT NOT NULL",
    "old_value": "TEXT",
    "new_value": "TEXT",
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

- [ ] **Step 3: Create tables in `MemoryStore.__init__`**

After existing `self._db.create_table("reflections", _REFLECTIONS_SCHEMA)`:

```python
if "memory_events" not in self._db.list_tables():
    self._db.create_table("memory_events", _MEMORY_EVENTS_SCHEMA)
if "memory_conflicts" not in self._db.list_tables():
    self._db.create_table("memory_conflicts", _MEMORY_CONFLICTS_SCHEMA)
```

- [ ] **Step 4: Run auto-migration on existing databases**

After existing table creation logic in `MemoryStore.__init__`, add:

```python
# v0.5 → v0.6 migration detection
columns = self._db.raw_query("PRAGMA table_info(observations)")
col_names = {c["name"] for c in columns}
if "memory_type" not in col_names:
    from coremem.migrations.v0_5_to_v0_6 import migrate
    migrate(self._db, str(path))
```

- [ ] **Step 5: Create migration script `coremem/migrations/v0_5_to_v0_6.py`**

```python
"""Migrate observations table from v0.5 to v0.6 schema.

Adds: memory_type, durability, sensitivity, status, confidence,
      valid_from, valid_to, superseded_by, source_message_ids.
Creates: memory_events, memory_conflicts tables.
"""
from __future__ import annotations

from typing import Any

def migrate(db: Any, db_path: str) -> None:
    """Add new columns and create new tables."""
    new_columns = [
        ("source_message_ids", "TEXT DEFAULT '[]'"),
        ("confidence", "REAL DEFAULT 0.800"),
        ("memory_type", "TEXT"),
        ("durability", "TEXT DEFAULT 'durable'"),
        ("sensitivity", "TEXT DEFAULT 'normal'"),
        ("status", "TEXT DEFAULT 'candidate'"),
        ("valid_from", "TEXT"),
        ("valid_to", "TEXT"),
        ("superseded_by", "TEXT"),
    ]
    for col_name, col_def in new_columns:
        try:
            db.raw_query(f"ALTER TABLE observations ADD COLUMN {col_name} {col_def}")
        except Exception:
            pass  # Column already exists

    # Create new tables
    from coremem.memory_store import _MEMORY_EVENTS_SCHEMA, _MEMORY_CONFLICTS_SCHEMA

    db.create_table("memory_events", _MEMORY_EVENTS_SCHEMA)
    db.create_table("memory_conflicts", _MEMORY_CONFLICTS_SCHEMA)

    db.raw_query(
        "CREATE INDEX IF NOT EXISTS idx_memory_events_memory "
        "ON memory_events(memory_id)"
    )
    db.raw_query(
        "CREATE INDEX IF NOT EXISTS idx_memory_conflicts_status "
        "ON memory_conflicts(resolution_status)"
    )
```

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/ -x --ignore=tests/test_backend_chroma.py --ignore=tests/test_backend_hybrid.py --ignore=tests/test_core.py -q
```

Expected: 89 passed

- [ ] **Step 7: Commit**

```bash
git add coremem/memory_store.py coremem/migrations/v0_5_to_v0_6.py
git commit -m "feat: add v0.6 schema — memory_type, events, conflicts tables"
```

---

### Task 2: MemoryStore — new methods for dedup + events

**Files:**
- Modify: `coremem/memory_store.py` (add methods after existing ones)

- [ ] **Step 1: Add `get_candidates()` method**

```python
def get_candidates(self, content: str, user_id: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
    """Find potential duplicate/related memories by keyword overlap.

    Returns existing observations whose content shares significant words
    with the new content, ordered by recency.
    """
    words = set(content.lower().split())
    if not words:
        return []

    candidates: list[dict[str, Any]] = []
    recent = self.get_recent_observations(days=30, limit=200)
    for obs in recent:
        if user_id and obs.get("user_id") != user_id:
            continue
        obs_words = set(obs.get("content", "").lower().split())
        overlap = words & obs_words
        if len(overlap) >= 3 and len(overlap) / min(len(words), len(obs_words)) > 0.5:
            candidates.append(obs)

    candidates.sort(key=lambda o: o.get("observation_ts", ""), reverse=True)
    return candidates[:limit]
```

- [ ] **Step 2: Add `update_observation()` method**

```python
def update_observation(self, obs_id: str, updates: dict[str, Any]) -> None:
    """Update specific fields on an existing observation."""
    if not updates:
        return
    set_parts = [f"{k} = ?" for k in updates]
    values = list(updates.values()) + [obs_id]
    self._db.raw_query(
        f"UPDATE observations SET {', '.join(set_parts)} WHERE id = ?",
        tuple(values),
    )
```

- [ ] **Step 3: Add `insert_event()` method**

```python
def insert_event(
    self,
    memory_id: str,
    event_type: str,
    old_value: str | None = None,
    new_value: str | None = None,
    source_message_id: str | None = None,
) -> str:
    """Log a memory lifecycle event."""
    eid = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    self._db.insert("memory_events", {
        "id": eid,
        "memory_id": memory_id,
        "event_type": event_type,
        "old_value": old_value or "",
        "new_value": new_value or "",
        "source_message_id": source_message_id or "",
        "created_at": now,
    })
    return eid
```

- [ ] **Step 4: Add `create_conflict()` method**

```python
def create_conflict(self, memory_id_a: str, memory_id_b: str, conflict_type: str) -> str:
    """Create a conflict record between two observations."""
    cid = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    self._db.insert("memory_conflicts", {
        "id": cid,
        "memory_id_a": memory_id_a,
        "memory_id_b": memory_id_b,
        "conflict_type": conflict_type,
        "resolution_status": "unresolved",
        "created_at": now,
        "resolved_at": "",
    })
    return cid
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_memory_store.py -v
```

Expected: All memory store tests pass

- [ ] **Step 6: Commit**

```bash
git add coremem/memory_store.py
git commit -m "feat: add get_candidates, update_observation, insert_event, create_conflict"
```

---

### Task 3: source_message_ids tracking in pipeline

**Files:**
- Modify: `coremem/observer.py` (Phase 2 + Phase 3 to collect message IDs)

- [ ] **Step 1: Collect message IDs in Phase 2**

Ensure `import json` is at the top of `observer.py`. After the `new_obs.append(obs)` in Phase 2, add:

```python
# Collect source message IDs from the full conversation
src_ids = [m.id for m in new_messages if m.id]
obs["source_message_ids"] = json.dumps(src_ids)
```

- [ ] **Step 2: Collect message IDs in Phase 3 gleaning**

In the per-message gleaning loop, collect only the pair's message IDs:

```python
src_ids = [m.id for m in pair if m.id]
obs["source_message_ids"] = json.dumps(src_ids)
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/ -x --ignore=tests/test_backend_chroma.py --ignore=tests/test_backend_hybrid.py --ignore=tests/test_core.py -q
```

Expected: 89 passed (no new tests yet)

- [ ] **Step 4: Commit**

```bash
git add coremem/observer.py
git commit -m "feat: track source_message_ids in pipeline"
```

---

### Task 4: Phase 4 — Classification + Durability Filter

**Files:**
- Modify: `coremem/observer.py` (add classification method on Observer, call in _maybe_run)
- Create: `coremem/classifier.py` (classification prompt + logic, extract for testability)

- [ ] **Step 1: Create `coremem/classifier.py`**

```python
"""Phase 4: Memory classification and durability filter."""
from __future__ import annotations

import json
from typing import Any

from coremem.observer_utils import parse_json_array

_CLASSIFICATION_PROMPT = """You are a memory classifier. For each observation below, classify:

1. memory_type: profile | preference | project | decision | technical_stack | business_context | people | constraint | workflow | episodic | procedural | sentiment
2. durability: durable (persists across sessions) | temporary (context-specific, one-off question)
3. sensitivity: normal (business/technical) | personal (family, logistics, people) | sensitive (health, legal, precise location)

Guide:
- profile: identity, job, location, education, contact
- preference: likes, dislikes, preferred tools, styles
- project: active or planned work, initiatives
- decision: choices made, directions chosen
- technical_stack: languages, tools, databases, infra
- business_context: durable business facts, organization info
- people: important recurring people, relationships
- constraint: hard restrictions, musts, must-nots
- workflow: repeated processes, how things are done
- episodic: past events, experiences, specific occurrences
- procedural: agent behavior rules, interaction preferences
- sentiment: emotional reactions, feelings, evaluations
- temporary: one-off questions, weather checks, single-task context
- durable: useful beyond this conversation

For each observation, return: memory_type, durability, sensitivity, confidence (0.0-1.0).

Examples:
- "User works at Anthropic as a research engineer" → profile, durable, normal, confidence:0.95
- "User asks about weather this weekend" → preference, temporary, normal, confidence:0.90
- "User prefers audiobooks over e-books" → preference, durable, normal, confidence:0.85
- "User saw The Lumineers at a festival" → episodic, durable, normal, confidence:0.90
- "User's address is 123 Main St" → profile, durable, sensitive, confidence:0.95

Return ONLY valid JSON via the observations tool.
"""

_CLASSIFICATION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_classifications",
        "description": "Return classified observations",
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "memory_type": {"type": "string"},
                            "durability": {"type": "string"},
                            "sensitivity": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["index", "memory_type", "durability", "sensitivity", "confidence"],
                    },
                },
            },
            "required": ["observations"],
        },
    },
}


def build_classification_prompt(observations: list[dict[str, Any]]) -> str:
    """Build the user message listing all observations to classify."""
    lines = ["Classify each observation:"]
    for i, obs in enumerate(observations):
        lines.append(f"[{i}] {obs.get('content', '')}")
    return "\n".join(lines)


def parse_classifications(response: Any) -> list[dict[str, Any]]:
    """Parse LLM response into classification dicts."""
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        arguments = tool_calls[0].get("function", {}).get("arguments", "")
        if arguments:
            parsed = parse_json_array(arguments)
            if parsed and "observations" in parsed[0]:
                return parsed[0]["observations"]
    return []


async def classify_observations(
    provider: Any,
    observations: list[dict[str, Any]],
    batch_size: int = 20,
) -> list[dict[str, Any]]:
    """Classify observations in batches, returning enriched dicts."""
    classified: list[dict[str, Any]] = []
    for i in range(0, len(observations), batch_size):
        batch = observations[i : i + batch_size]
        prompt = build_classification_prompt(batch)
        messages = [
            {"role": "system", "content": _CLASSIFICATION_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = await provider.chat_with_tools(messages, [_CLASSIFICATION_TOOL])
        results = parse_classifications(response)
        for idx, obs in enumerate(batch):
            classification = next((r for r in results if r.get("index") == idx), {})
            obs["memory_type"] = classification.get("memory_type", "")
            obs["durability"] = classification.get("durability", "durable")
            obs["sensitivity"] = classification.get("sensitivity", "normal")
            obs["confidence"] = classification.get("confidence", 0.800)
            if obs["durability"] == "temporary":
                obs["status"] = "archived"
            classified.append(obs)
    return classified
```

- [ ] **Step 2: Wire Phase 4 into `ObserverPipeline._maybe_run()`**

After Phase 3 (gleaning), add:

```python
# Phase 4: Classification + Durability Filter
if self._enable_classification and new_obs:
    try:
        classified = await classify_observations(
            self._observer._provider, new_obs,
        )
        new_obs = classified
    except Exception as e:
        logger.warning("classification_error", {"error": str(e)})
```

Add `enable_classification` flag to `ObserverPipeline.__init__`:

```python
def __init__(
    self,
    ...,
    enable_gleaning: bool = False,
    enable_classification: bool = False,
    ...
):
    self._enable_classification = enable_classification
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/ -x --ignore=tests/test_backend_chroma.py --ignore=tests/test_backend_hybrid.py --ignore=tests/test_core.py -q
```

Expected: 89 passed

- [ ] **Step 4: Commit**

```bash
git add coremem/classifier.py coremem/observer.py
git commit -m "feat: add Phase 4 — classification + durability filter"
```

---

### Task 5: Phase 5 — Dedup + Merge

**Files:**
- Create: `coremem/dedup.py`
- Modify: `coremem/observer.py` (wire Phase 5 into pipeline)

- [ ] **Step 1: Create `coremem/dedup.py`**

```python
"""Phase 5: Semantic dedup and merge for observations."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from coremem.observer_utils import parse_json_array

_DEDUP_PROMPT = """You are a memory deduplicator. For each new/existing pair, classify:

1. duplicate: both say the SAME thing. Even if wording differs, the fact is identical.
2. refine: new ADDS detail to existing without contradicting (e.g., "works at Anthropic" vs "works at Anthropic as engineer").
3. supersede: new REPLACES existing (time has passed, status changed, e.g., "intern" vs "senior").
4. contradict: new CONFLICTS with existing (e.g., "prefers Mac" vs "switched to Linux").
5. new: DISTINCT fact — no meaningful relationship.

Return ONLY: relationship, and for refine/supersede, the merged content string.

Examples:
- New: "User works at Anthropic as a research engineer" vs Old: "User works at Anthropic" → refine, merged: "User works at Anthropic as a research engineer"
- New: "User self-hosts n8n" vs Old: "User uses n8n cloud" → supersede
- New: "User switched to Linux" vs Old: "User prefers Mac" → contradict
- New: "User commutes 45 min each way" vs Old: "User likes audiobooks" → new
- New: "User enjoys coffee" vs Old: "User likes the coffee scene" → duplicate
"""

_DEDUP_TOOL = {
    "type": "function",
    "function": {
        "name": "record_relationships",
        "description": "Return dedup classifications",
        "parameters": {
            "type": "object",
            "properties": {
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "new_index": {"type": "integer"},
                            "relationship": {"type": "string"},
                            "merged_content": {"type": "string"},
                            "old_id": {"type": "string"},
                        },
                        "required": ["new_index", "relationship", "old_id"],
                    },
                },
            },
            "required": ["pairs"],
        },
    },
}


def build_dedup_prompt(
    pairs: list[dict[str, Any]], new_obs_list: list[dict[str, Any]]
) -> str:
    """Build prompt listing new/old observation pairs."""
    lines = ["Classify each pair:"]
    for p in pairs:
        ni = p["new_index"]
        new_content = new_obs_list[ni].get("content", "")
        old_content = p["candidate"].get("content", "")
        lines.append(
            f"Pair {ni}: "
            f'New[{ni}]:"{new_content}" '
            f'vs Old({p["candidate"]["id"]}):"{old_content}"'
        )
    return "\n".join(lines)


async def dedup_and_merge(
    provider: Any,
    store: Any,
    new_obs: list[dict[str, Any]],
    batch_size: int = 5,
) -> list[dict[str, Any]]:
    """Run dedup + merge on observations.

    For each observation, search for keyword-overlap candidates,
    then use LLM to classify the relationship. Mutates store
    in-place for merges/supersessions.
    """
    final: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    # Step 1: Find candidates for each observation
    for i, obs in enumerate(new_obs):
        if obs.get("status") == "archived":
            final.append(obs)
            continue
        user_id = obs.get("user_id")
        candidates = store.get_candidates(obs.get("content", ""), user_id=user_id)
        for candidate in candidates:
            pairs.append({"new_index": i, "candidate": candidate})

    if not pairs:
        # No candidates — all are new
        for obs in new_obs:
            if obs.get("status") != "archived":
                store.insert_event(obs["id"], "created")
            final.append(obs)
        return final

    # Step 2: Batch pairs and classify via LLM
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        prompt = build_dedup_prompt(batch, new_obs)
        messages = [
            {"role": "system", "content": _DEDUP_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = await provider.chat_with_tools(messages, [_DEDUP_TOOL])
        classifications = _parse_dedup_response(response)

    # Step 3: Apply each classification, respecting priority order
        # Priority: supersede > contradict > refine > duplicate > new
        # A single observation may have multiple candidate pairs —
        # only the highest-priority relationship wins.
        resolved: dict[int, dict[str, Any]] = {}  # new_index → classification
        for classification in classifications:
            ni = classification["new_index"]
            ex = resolved.get(ni, {})
            ex_rel = ex.get("relationship", "new")
            new_rel = classification["relationship"]
            priority = {"supersede": 5, "contradict": 4, "refine": 3, "duplicate": 2, "new": 1}
            if priority.get(new_rel, 0) > priority.get(ex_rel, 0):
                resolved[ni] = classification

        for classification in resolved.values():
            ni = classification["new_index"]
            rel = classification["relationship"]
            old_id = classification["old_id"]
            obs = new_obs[ni]

            if rel == "duplicate":
                obs["status"] = "archived"
                obs["superseded_by"] = old_id

            elif rel == "refine":
                merged = classification.get("merged_content", obs.get("content", ""))
                store.update_observation(old_id, {"content": merged})
                old = store.get_observations(observation_ids=[old_id])
                if old:
                    old_src = json.loads(old[0].get("source_message_ids", "[]"))
                    new_src = json.loads(obs.get("source_message_ids", "[]"))
                    store.update_observation(old_id, {
                        "source_message_ids": json.dumps(old_src + new_src),
                    })
                store.insert_event(old_id, "merged", old_value=old[0]["content"] if old else "", new_value=merged)
                obs["status"] = "archived"
                obs["superseded_by"] = old_id

            elif rel == "supersede":
                valid_to = obs.get("observation_ts") or datetime.now(UTC).isoformat()
                store.update_observation(old_id, {
                    "status": "superseded",
                    "valid_to": valid_to,
                })
                store.insert_event(old_id, "superseded", old_value=old_id, new_value=obs.get("id", ""))
                obs["status"] = "candidate"
                store.insert_event(obs["id"], "created")

            elif rel == "contradict":
                store.create_conflict(old_id, obs.get("id", ""), "contradiction")
                obs["status"] = "candidate"
                store.insert_event(obs["id"], "contradicted", old_value=old_id)

            else:  # new
                store.insert_event(obs["id"], "created")

    # Collect final: archived first, then unprocessed new
    for obs in new_obs:
        if obs.get("status") == "archived":
            final.append(obs)
    for obs in new_obs:
        if obs.get("status") != "archived":
            store.insert_event(obs.get("id", ""), "created")
            final.append(obs)

    return final


def _parse_dedup_response(response: Any) -> list[dict[str, Any]]:
    """Parse LLM dedup response."""
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        arguments = tool_calls[0].get("function", {}).get("arguments", "")
        if arguments:
            parsed = parse_json_array(arguments)
            if parsed and "pairs" in parsed[0]:
                return parsed[0]["pairs"]
    return []
```

- [ ] **Step 2: Wire Phase 5 into `ObserverPipeline._maybe_run()`**

After Phase 4 classification, add:

```python
# Phase 5: Dedup + Merge
if self._enable_dedup and new_obs:
    try:
        new_obs = await dedup_and_merge(
            self._observer._provider, self._store, new_obs,
        )
    except Exception as e:
        logger.warning("dedup_error", {"error": str(e)})
```

Add `enable_dedup` to `ObserverPipeline.__init__` (default False for backward compat).

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/ -x --ignore=tests/test_backend_chroma.py --ignore=tests/test_backend_hybrid.py --ignore=tests/test_core.py -q
```

Expected: 89 passed

- [ ] **Step 4: Commit**

```bash
git add coremem/dedup.py coremem/observer.py
git commit -m "feat: add Phase 5 — semantic dedup + merge"
```

---

### Task 6: Tests — dedup, classification, integration

**Files:**
- Create: `tests/test_dedup.py`
- Create: `tests/test_classifier.py`
- Modify: `tests/test_pipelines.py` (update mocks for new phases)

- [ ] **Step 1: Create `tests/test_classifier.py`**

```python
"""Tests for classification phase."""
from coremem.classifier import build_classification_prompt, _CLASSIFICATION_PROMPT


class TestClassificationPrompt:
    def test_prompt_contains_12_types(self):
        for t in ["profile", "preference", "project", "decision",
                  "technical_stack", "business_context", "people",
                  "constraint", "workflow", "episodic", "procedural", "sentiment"]:
            assert t in _CLASSIFICATION_PROMPT, f"Missing type: {t}"

    def test_prompt_defines_durability(self):
        assert "durable" in _CLASSIFICATION_PROMPT
        assert "temporary" in _CLASSIFICATION_PROMPT

    def test_prompt_defines_sensitivity(self):
        assert "normal" in _CLASSIFICATION_PROMPT
        assert "personal" in _CLASSIFICATION_PROMPT
        assert "sensitive" in _CLASSIFICATION_PROMPT

    def test_build_classification_prompt_formats_indices(self):
        obs = [
            {"content": "User works at Anthropic"},
            {"content": "User likes coffee"},
        ]
        prompt = build_classification_prompt(obs)
        assert "[0] User works at Anthropic" in prompt
        assert "[1] User likes coffee" in prompt
```

- [ ] **Step 2: Create `tests/test_dedup.py`**

```python
"""Tests for dedup and merge logic."""
from coremem.dedup import build_dedup_prompt, _DEDUP_PROMPT


class TestDedupPrompt:
    def test_prompt_defines_relationships(self):
        for rel in ["duplicate", "refine", "supersede", "contradict", "new"]:
            assert rel in _DEDUP_PROMPT, f"Missing: {rel}"

    def test_build_dedup_prompt_formats_pairs(self):
        pairs = [
            {
                "new_index": 0,
                "candidate": {"id": "old_1", "content": "User works at Google"},
            }
        ]
        new_obs = [{"content": "User works at Anthropic"}]
        prompt = build_dedup_prompt(pairs, new_obs)
        assert 'New[0]:"User works at Anthropic"' in prompt
        assert 'Old(old_1):"User works at Google"' in prompt


class TestDedupLogic:
    def test_duplicate_sets_status_to_archived(self):
        from coremem.dedup import _parse_dedup_response
        # Mock response for duplicate classification
        class MockResponse:
            tool_calls = [{"function": {"arguments": '{"pairs": [{"new_index": 0, "relationship": "duplicate", "old_id": "old_1"}]}'}}]

        result = _parse_dedup_response(MockResponse())
        assert result[0]["relationship"] == "duplicate"
        assert result[0]["old_id"] == "old_1"
```

- [ ] **Step 3: Update pipeline test mocks**

In `tests/test_pipelines.py`, the mock `chat_with_tools` now has 6 LF calls + N Phase 2 calls + N gleaning calls. Add an extra entry for classification if `enable_classification=True`. For existing tests (enable_classification=False), no change needed.

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ -x --ignore=tests/test_backend_chroma.py --ignore=tests/test_backend_hybrid.py --ignore=tests/test_core.py -q -v
```

Expected: 89 + N new tests, all passing

- [ ] **Step 5: Commit**

```bash
git add tests/test_dedup.py tests/test_classifier.py tests/test_pipelines.py
git commit -m "test: add classification and dedup tests"
```

---

### Task 7: Integration verification — run 10-question eval

**Files:**
- Modify: `benchmarks/longmemeval/observer_eval.py` (enable classification + dedup)

- [ ] **Step 1: Enable new phases in eval**

```python
pipeline = ObserverPipeline(
    core=core, store=store, session_id=sid,
    model=provider, token_threshold=1, min_turns=1,
    tool_temp=0.1,
    enable_gleaning=True,
    enable_classification=True,
    enable_dedup=True,
)
```

- [ ] **Step 2: Run 10-question eval**

```bash
export $(cat .env | xargs)
uv run python -m benchmarks.longmemeval.observer_eval \
  --data /Users/eddy/Developer/Python/CoreMem/results/eval/longmemeval_10q.json \
  --provider deepseek:deepseek-v4-flash \
  --mode both \
  --limit 10 \
  --output results/eval/observer_deepseek_v4flash_redesign.json
```

- [ ] **Step 3: Verify results**

Check: 0% hallucination, obs/q after durability filter, answer hits, observation quality.

```bash
uv run python -c "
import json
with open('results/eval/observer_deepseek_v4flash_redesign.json') as f:
    data = json.load(f)
for r in data['results']:
    count = len(sum([s.get('observations',[]) for s in r.get('sessions',[])], []))
    durable = sum(1 for s in r.get('sessions',[]) for o in s.get('observations',[]) if o.get('durability')=='durable')
    print(f'{r[\"question_id\"]}: {count} obs, {durable} durable')
"
```

- [ ] **Step 4: Commit results**

```bash
git add results/eval/observer_deepseek_v4flash_redesign.json
git commit -m "eval: 10-question pipeline redesign results"
```
