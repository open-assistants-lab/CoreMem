# CoreMem Fact Layer (Semantic Layer v0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give CoreMem a governed, bi-temporal fact layer — a versioned `facts` table with ontology-declared cardinality, lifecycle verbs, deterministic `where=` lookups, and an optional dual-path reading section — as a zero-LLM v0, with opt-in LLM extraction as a separate, killable Phase 2 experiment.

**Architecture:** Facts live in a new versioned `facts` table (hybriddb 0.6.0+ hash chains: every fact change is auditable and reversible). A small closed **attribute ontology** (`ontology.toml` + starter defaults) declares value types and **cardinality** (single → new value supersedes old via `valid_to`; multi → appends). Facts are read through the 0.7.0 `where=` pre-filter (deterministic, zero-LLM) and surfaced to readers as a budget-capped `[FACTS]` digest section alongside verbatim session bundles (the Redis-validated dual-store pattern; the episodic/semantic memory split per Zep, arXiv 2501.13956). Extraction (message → candidate facts) is **Phase 2, opt-in, gated by a pre-registered kill criterion**.

**Tech Stack:** Python 3.11+, hybriddb>=0.7.0 (versioned tables, `where=` metadata pre-filtering), stdlib `tomllib`, pytest. No new dependencies.

**Spec:** Background & Decisions below (the self-reviewed spec — this plan is its implementation); substrate in `docs/versioned-memory-design.md`; evidence in `docs/retrieval-experiments.md` (lever-2 falsification → dual-store is the untested variant; Redis dual-store 86.1% on LongMemEval-S; Zep entity-resolution + bi-temporal sections).

## Background & Decisions (the spec, self-reviewed 2026-09-07)

1. **Zero-LLM invariant preserved.** Phase 1 adds **no LLM calls anywhere** — facts enter via the public `add_fact` API. Phase 2 (extraction) is opt-in at ingest, never on the read path.
2. **Cardinality drives supersession.** `ontology.toml` declares `{attribute: {cardinality: single|multi}}`. Single-valued: `add_fact` expires the current fact (`valid_to` + `superseded_by` link). Multi-valued: append, never supersede. **Unknown attributes default to `multi` (safe — never silently destroy).** A starter set ships in code; user TOML extends it.
3. **Honest scope (from self-review):** the supersession fix applies to *in-vocabulary* attributes only; open-world attributes behave like today's raw store (no regression, no fix). Do not promise more.
4. **Facts digest budget:** capped at 800 chars, *appended after* the bundle context — the reader prompt grows, it never evicts verbatim evidence. Phase 2's A/B validates this explicitly.
5. **Tenant scoping:** every fact carries `user_id`; fact methods filter by it.
6. **Kill criteria (pre-registered, Phase 2):** dual-path answer eval on S stratified-56 (paired: identical contexts + facts digest vs identical contexts alone, same reader/judge, blind). Net ≤ 0 on accuracy → extraction stays opt-in-**off**; the fact layer remains as the governance API. Same discipline as levers 2/4/5.
7. **Non-versioned stores:** fact methods raise with guidance (facts ride the versioned chain; pattern: `_require_versioned` in `coremem/core.py`).
8. **No traversal, no LLM-generated queries** — Zep's own guardrail (predefined Cypher "to reduce hallucinations") plus our traversal falsifications. Fact access is predefined lookups only.

## Global Constraints

- hybriddb floor `>=0.7.0` (already pinned; do not lower)
- `versioned=True` is the store default (0.15.0); fact methods require it
- Read path stays zero-LLM: no model calls in `recall`, `list_facts`, or digest assembly
- Facts digest cap: **800 chars** (`FACTS_DIGEST_BUDGET_CHARS = 800`), appended after bundle context
- All writes through the CRUD layer (`insert_batch`/`update`) — never `raw_query` (tracked-writes discipline)
- Version: **0.17.0** (new public API); both version strings bumped; suite stays green and zero behavior changes when no facts are used
- Facts table has NO LONGTEXT content beyond `search_text` (composed `entity.attribute = value`) — this is what gets embedded into Chroma for semantic fact lookup; scalar columns stay TEXT (out of the vector store)

---

### Task 1: Attribute ontology module

**Files:**
- Create: `coremem/ontology.py`
- Test: `tests/test_ontology.py`

**Interfaces:**
- Produces: `AttributeDef(name: str, cardinality: str, value_type: str)` frozen dataclass; `FactOntology(config: dict | None = None, config_path: str | Path | None = None)` with `.attribute(name: str) -> AttributeDef` (unknown → `AttributeDef(name, "multi", "text")`); `STARTER_ATTRIBUTES` dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ontology.py
from coremem.ontology import FactOntology


def test_unknown_attribute_defaults_to_multi_text():
    onto = FactOntology()
    a = onto.attribute("favorite_brand_of_soup")
    assert a.cardinality == "multi" and a.value_type == "text"


def test_starter_attributes_have_declared_cardinality():
    onto = FactOntology()
    assert onto.attribute("employer").cardinality == "single"
    assert onto.attribute("child").cardinality == "multi"


def test_toml_config_overrides_and_extends(tmp_path):
    cfg = tmp_path / "ontology.toml"
    cfg.write_text(
        '[attribute.airline]\ncardinality = "single"\nvalue_type = "text"\n'
        '[attribute.pet]\ncardinality = "multi"\n'
    )
    onto = FactOntology(config_path=str(cfg))
    assert onto.attribute("airline").cardinality == "single"
    assert onto.attribute("pet").cardinality == "multi"
    assert onto.attribute("employer").cardinality == "single"  # starter set survives merge
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_ontology.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coremem.ontology'`

- [ ] **Step 3: Write the implementation**

```python
# coremem/ontology.py
"""Fact-layer attribute ontology — a small closed vocabulary, not a knowledge graph.

Cardinality is an ontological property you cannot learn from data:
single-valued attributes supersede on update; multi-valued ones append.
Declared once in ontology.toml; the fact layer enforces it deterministically.
Traversal stays out (falsified twice — docs/retrieval-experiments.md).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttributeDef:
    name: str
    cardinality: str = "multi"  # "single" | "multi"
    value_type: str = "text"    # "text" | "number" | "date"


# Starter vocabulary for common personal-memory attributes. Extend via
# ontology.toml; unknown attributes default to multi/text (safe: append only).
STARTER_ATTRIBUTES: dict[str, dict] = {
    "employer": {"cardinality": "single"},
    "job_title": {"cardinality": "single"},
    "home_city": {"cardinality": "single"},
    "partner": {"cardinality": "single"},
    "birthday": {"cardinality": "single", "value_type": "date"},
    "child": {"cardinality": "multi"},
    "pet": {"cardinality": "multi"},
    "preference": {"cardinality": "multi"},
    "allergy": {"cardinality": "multi"},
}


class FactOntology:
    def __init__(self, config: dict | None = None, config_path: str | Path | None = None):
        # Merge order: starter set <- user TOML <- user dict.
        # TOML shape: [attribute.name] with cardinality/value_type keys.
        loaded: dict = {}
        if config_path is not None:
            with open(config_path, "rb") as fh:
                loaded = tomllib.load(fh).get("attribute", {})
        elif config:
            loaded = config.get("attribute", config)
        merged = {**STARTER_ATTRIBUTES, **(loaded or {})}
        self._attributes: dict[str, AttributeDef] = {}
        for name, spec in merged.items():
            spec = spec if isinstance(spec, dict) else {}
            self._attributes[name] = AttributeDef(
                name=name,
                cardinality=spec.get("cardinality", "multi"),
                value_type=spec.get("value_type", "text"),
            )

    def attribute(self, name: str) -> AttributeDef:
        return self._attributes.get(name, AttributeDef(name=name))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_ontology.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add coremem/ontology.py tests/test_ontology.py
git commit -m "feat: fact-layer attribute ontology (cardinality-configurable, TOML extensible)"
```

---

### Task 2: `facts` versioned table + `add_fact` with cardinality supersession

**Files:**
- Modify: `coremem/core.py` (imports, `_ensure_tables`, new methods after `verify_memory_chain`)
- Test: `tests/test_fact_layer.py`

**Interfaces:**
- Consumes: `FactOntology` (Task 1), `self._versioned`, `_flush_journal_batched`, `_require_versioned`
- Produces: `MemoryCore.add_fact(entity: str, attribute: str, value: str, *, user_id: str = "", source_message_id: str = "", valid_from: str | None = None) -> str`; instance attr `self._ontology: FactOntology`; **`verify_memory_chain`/`checkpoint_memory`/`rollback_memory` extended to cover `facts`**

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fact_layer.py
import shutil
import tempfile

import pytest

from coremem import MemoryCore


def _make_core(**kwargs) -> MemoryCore:
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d, author="test", **kwargs)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


def test_add_fact_records_on_chain_and_lists_current():
    core = _make_core(versioned=True)
    try:
        fid = core.add_fact("user", "dog", "Max", source_message_id="a1")
        facts = core.list_facts(entity="user", attribute="dog")
        assert len(facts) == 1 and facts[0]["value"] == "Max"
        assert facts[0]["valid_to"] == "" and facts[0]["source_message_id"] == "a1"
        assert core.verify_memory_chain()["facts"]["valid"]
    finally:
        core._test_cleanup()


def test_single_cardinality_supersedes_multi_appends():
    core = _make_core(versioned=True)
    try:
        core.add_fact("user", "employer", "Acme")
        core.add_fact("user", "employer", "Globex")
        current = core.list_facts(entity="user", attribute="employer")
        assert len(current) == 1 and current[0]["value"] == "Globex"
        core.add_fact("user", "child", "Emma")
        core.add_fact("user", "child", "Leo")
        kids = core.list_facts(entity="user", attribute="child")
        assert {k["value"] for k in kids} == {"Emma", "Leo"}
        assert core.verify_memory_chain()["facts"]["valid"]
    finally:
        core._test_cleanup()


def test_add_fact_requires_versioned_store():
    core = _make_core(versioned=False)
    try:
        with pytest.raises(RuntimeError, match="versioned"):
            core.add_fact("user", "dog", "Max")
    finally:
        core._test_cleanup()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_fact_layer.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'add_fact'`

- [ ] **Step 3: Write the implementation**

`coremem/core.py` — top imports: `from coremem.ontology import FactOntology`. Constructor: `self._ontology = FactOntology()` after `self._fact_augment_warned = False`. In `_ensure_tables`, after the `journal_records` block:

```python
        if self._versioned and "facts" not in self._db.list_tables():
            self._db.create_table("facts", {
                "id": "TEXT PRIMARY KEY",
                "user_id": "TEXT DEFAULT ''",
                "entity": "TEXT NOT NULL",
                "attribute": "TEXT NOT NULL",
                "value": "TEXT NOT NULL",
                "valid_from": "TEXT NOT NULL",
                "valid_to": "TEXT DEFAULT ''",
                "superseded_by": "TEXT DEFAULT ''",
                "source_message_id": "TEXT DEFAULT ''",
                "created_at": "TEXT NOT NULL",
                "search_text": "LONGTEXT",  # embedded → semantic fact lookup
            }, versioned=True)
```

New methods (place after `verify_memory_chain`):

```python
    def _require_facts(self) -> None:
        self._ensure_open()
        self._require_versioned()

    def add_fact(
        self, entity: str, attribute: str, value: str, *,
        user_id: str = "", source_message_id: str = "",
        valid_from: str | None = None,
    ) -> str:
        """Record a governed fact. Supersedes the current fact for
        single-cardinality attributes (per the attribute ontology); appends
        for multi-valued ones. Provenance rides the versioned chain."""
        self._require_facts()
        if not entity or not attribute or not value:
            raise ValueError("add_fact requires non-empty entity, attribute, value")
        attr_def = self._ontology.attribute(attribute)
        now = datetime.now(UTC).isoformat()
        fid = str(uuid.uuid4())[:12]
        if attr_def.cardinality == "single":
            current = self._db.raw_query(
                "SELECT id FROM facts WHERE user_id = ? AND entity = ? "
                "AND attribute = ? AND valid_to = '' AND superseded_by = ''",
                [user_id, entity, attribute],
            )
            for row in current:
                self._db.update("facts", row["id"],
                                {"valid_to": now, "superseded_by": fid}, sync=False)
        self._db.insert_batch("facts", [{
            "id": fid, "user_id": user_id, "entity": entity,
            "attribute": attribute, "value": value,
            "valid_from": valid_from or now, "valid_to": "",
            "superseded_by": "", "source_message_id": source_message_id,
            "created_at": now,
            "search_text": f"{entity}.{attribute} = {value}",
        }], sync=False)
        _flush_journal_batched(self._db)
        return fid
```

**Also extend the three governance methods to cover `facts`** (the tests assert
this): `verify_memory_chain` gains `"facts": self._db.verify_chain("facts")`;
`checkpoint_memory` gains `"facts": self._db.checkpoint("facts", label)`;
`rollback_memory` gains `"facts": self._db.rollback("facts", **kwargs)`.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_fact_layer.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add coremem/core.py coremem/ontology.py tests/test_fact_layer.py
git commit -m "feat: facts versioned table + add_fact with ontology cardinality supersession"
```

---

### Task 3: Fact queries — `list_facts` / `get_fact`

**Files:**
- Modify: `coremem/core.py` (append after `add_fact`)
- Test: `tests/test_fact_layer.py` (extend)

**Interfaces:**
- Consumes: facts table (Task 2)
- Produces: `MemoryCore.list_facts(*, entity=None, attribute=None, user_id=None, include_expired=False, limit=200) -> list[dict]`; `MemoryCore.get_fact(fact_id: str) -> dict | None`; `MemoryCore.fact_history(fact_id: str) -> list[dict]` (per-row chain events — Task 6's MCP tool consumes this)

- [ ] **Step 1: Write the failing test**

```python
def test_list_facts_filters_and_hides_expired():
    core = _make_core(versioned=True)
    try:
        core.add_fact("user", "employer", "Acme")
        core.add_fact("user", "employer", "Globex")   # supersedes Acme
        core.add_fact("user", "pet", "Max")
        values = {f["value"] for f in core.list_facts(entity="user")}
        assert "Globex" in values and "Acme" not in values and "Max" in values
        expired = core.list_facts(entity="user", include_expired=True)
        assert any(f["value"] == "Acme" and f["valid_to"] for f in expired)
        assert all(f["user_id"] == "" for f in expired)
    finally:
        core._test_cleanup()


def test_get_fact_returns_row_or_none():
    core = _make_core(versioned=True)
    try:
        fid = core.add_fact("user", "dog", "Max")
        assert core.get_fact(fid)["value"] == "Max"
        assert core.get_fact("nonexistent") is None
    finally:
        core._test_cleanup()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_fact_layer.py -v -k "list_facts or get_fact"`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write the implementation**

```python
    def list_facts(
        self, *, entity: str | None = None, attribute: str | None = None,
        user_id: str | None = None, include_expired: bool = False,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Current facts (deterministic, zero-LLM). Expired facts (superseded
        or expired) hidden unless include_expired=True."""
        self._require_facts()
        clauses, params = [], []
        if entity:
            clauses.append("entity = ?"); params.append(entity)
        if attribute:
            clauses.append("attribute = ?"); params.append(attribute)
        if user_id is not None:
            clauses.append("user_id = ?"); params.append(user_id)
        if not include_expired:
            clauses.append("valid_to = '' AND superseded_by = ''")
        sql = "SELECT * FROM facts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return self._db.raw_query(sql, tuple(params))

    def get_fact(self, fact_id: str) -> dict[str, Any] | None:
        self._require_facts()
        rows = self._db.raw_query("SELECT * FROM facts WHERE id = ?", [fact_id])
        return rows[0] if rows else None

    def fact_history(self, fact_id: str) -> list[dict[str, Any]]:
        """Provenance timeline of one fact (every chain event, oldest first)."""
        self._require_facts()
        return self._db.history("facts", fact_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_fact_layer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add coremem/core.py tests/test_fact_layer.py
git commit -m "feat: list_facts/get_fact — deterministic governed fact lookups"
```

---

### Task 4: Lifecycle verbs — `supersede_fact` / `expire_fact` / `merge_facts`

**Files:**
- Modify: `coremem/core.py` (append after `list_facts`)
- Test: `tests/test_fact_layer.py` (extend)

**Interfaces:**
- Consumes: `db.update`, ontology (Tasks 1–2)
- Produces: `MemoryCore.supersede_fact(fact_id: str, new_value: str) -> str`; `MemoryCore.expire_fact(fact_id: str) -> bool`; `MemoryCore.merge_facts(fact_ids: list[str], merged_value: str) -> str`

- [ ] **Step 1: Write the failing test**

```python
def test_supersede_fact_sets_valid_to_and_link():
    core = _make_core(versioned=True)
    try:
        fid = core.add_fact("user", "employer", "Acme")
        new_id = core.supersede_fact(fid, "Globex")
        old = core.get_fact(fid)
        assert old["valid_to"] and old["superseded_by"] == new_id
        assert core.get_fact(new_id)["value"] == "Globex"
        assert core.verify_memory_chain()["facts"]["valid"]
    finally:
        core._test_cleanup()


def test_merge_facts_expires_sources():
    core = _make_core(versioned=True)
    try:
        f1 = core.add_fact("user", "pet", "Max")
        f2 = core.add_fact("user", "pet", "Bella")
        merged = core.merge_facts([f1, f2], "Max and Bella")
        assert core.get_fact(f1)["valid_to"] and core.get_fact(f2)["valid_to"]
        pets = {f["value"] for f in core.list_facts(entity="user", attribute="pet")}
        assert "Max and Bella" in pets
        assert core.verify_memory_chain()["facts"]["valid"]
    finally:
        core._test_cleanup()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_fact_layer.py -v -k "supersede or merge"`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write the implementation**

```python
    def _expire(self, fact_id: str) -> bool:
        now = datetime.now(UTC).isoformat()
        row = self.get_fact(fact_id)
        if row is None or row["valid_to"]:
            return False
        self._db.update("facts", fact_id, {"valid_to": now}, sync=False)
        _flush_journal_batched(self._db)
        return True

    def supersede_fact(self, fact_id: str, new_value: str) -> str:
        """Replace a fact's value: old fact expires (superseded_by link), a new
        fact is recorded. Both events live on the audit chain."""
        self._require_facts()
        old = self.get_fact(fact_id)
        if old is None:
            raise ValueError(f"unknown fact: {fact_id}")
        if not self._expire(fact_id):
            raise ValueError(f"fact {fact_id} is already expired")
        # NOTE: the new fact's valid_from is NOW (when the superseding
        # statement was made) — do NOT inherit the old fact's valid_from,
        # that would falsify the bi-temporal timeline.
        return self.add_fact(
            old["entity"], old["attribute"], new_value,
            user_id=old["user_id"],
            source_message_id=old["source_message_id"],
        )

    def expire_fact(self, fact_id: str) -> bool:
        self._require_facts()
        return self._expire(fact_id)

    def merge_facts(self, fact_ids: list[str], merged_value: str) -> str:
        """Combine duplicate facts ('Max' + 'my dog Max') into one; sources
        stay expired-and-auditable on the chain."""
        self._require_facts()
        if len(fact_ids) < 2:
            raise ValueError("merge_facts requires at least two fact ids")
        facts = [self.get_fact(f) for f in fact_ids]
        if any(f is None for f in facts):
            raise ValueError("merge_facts: unknown fact id in list")
        first = facts[0]
        for f in facts:
            self._expire(f["id"])
        return self.add_fact(
            first["entity"], first["attribute"], merged_value,
            user_id=first["user_id"],
            source_message_id=first["source_message_id"],
        )
```

> Implementer note: `add_fact` also self-supersedes single-cardinality attributes;
> `supersede_fact` exists for explicit operator intent and preserves `valid_from`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_fact_layer.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add coremem/core.py tests/test_fact_layer.py
git commit -m "feat: fact lifecycle verbs (supersede/expire/merge) on the versioned chain"
```

---

### Task 5: Dual-path reading — `[FACTS]` digest builder

**Files:**
- Modify: `coremem/reading.py`
- Test: `tests/test_fact_layer.py` (extend)

**Interfaces:**
- Consumes: fact rows (Task 3 shape: dicts with `entity/attribute/value/valid_from`)
- Produces: `build_facts_digest(facts: list[dict], *, budget_chars: int = FACTS_DIGEST_BUDGET_CHARS) -> str` — `""` when facts empty; `[FACTS]` header + grouped lines otherwise; budget-capped with a truncation marker

- [ ] **Step 1: Write the failing test**

```python
def test_facts_digest_groups_budget_caps_and_handles_empty():
    from coremem.reading import build_facts_digest
    assert build_facts_digest([]) == ""
    facts = [
        {"entity": "user", "attribute": "dog", "value": "Max",
         "valid_from": "2024-05-01T00:00:00+00:00"},
        {"entity": "user", "attribute": "employer", "value": "Acme",
         "valid_from": "2024-03-01T00:00:00+00:00"},
    ]
    out = build_facts_digest(facts)
    assert out.startswith("[FACTS]")
    assert "user.dog = Max (since 2024-05-01)" in out
    assert "user.employer = Acme (since 2024-03-01)" in out
    tiny = build_facts_digest(facts, budget_chars=40)
    assert len(tiny) <= 40 + len("- …(facts truncated)")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_fact_layer.py -v -k digest`
Expected: FAIL with `ImportError: cannot import name 'build_facts_digest'`

- [ ] **Step 3: Write the implementation**

```python
# append to coremem/reading.py
FACTS_DIGEST_BUDGET_CHARS = 800


def build_facts_digest(facts: list[dict], *, budget_chars: int = FACTS_DIGEST_BUDGET_CHARS) -> str:
    """Render current facts as a grouped, budget-capped [FACTS] digest.
    Appended AFTER the bundle context — it grows the reader prompt, it never
    evicts verbatim evidence (docs versioned-memory/fact-layer decision §4)."""
    if not facts:
        return ""
    lines: list[str] = []
    for f in sorted(facts, key=lambda f: (f.get("entity", ""), f.get("attribute", ""))):
        line = f"{f.get('entity', '')}.{f.get('attribute', '')} = {f.get('value', '')}"
        if f.get("valid_from"):
            line += f" (since {str(f['valid_from'])[:10]})"
        lines.append(f"- {line}")
    body = "\n".join(lines)
    if len(body) > budget_chars:
        body = body[:budget_chars].rsplit("\n", 1)[0] + "\n- …(facts truncated)"
    return "[FACTS]\n" + body
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_fact_layer.py -v -k digest`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add coremem/reading.py tests/test_fact_layer.py
git commit -m "feat: budget-capped [FACTS] digest builder for dual-path reading"
```

---

### Task 6: MCP tools — `add_fact` / `list_facts` / `fact_history`

**Files:**
- Modify: `coremem/mcp_server.py` (after the `memory_verify` tool)
- Test: `tests/test_mcp_fact_layer.py`

**Interfaces:**
- Consumes: Tasks 2–4 methods
- Produces: MCP tools `add_fact(entity, attribute, value, user_id="")`, `list_facts(entity="", attribute="", user_id="")`, `fact_history(fact_id)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mcp_fact_layer.py
import json

import pytest

mcp = pytest.importorskip("mcp")


def test_add_list_history_roundtrip(tmp_path):
    # Copy the subprocess+client scaffolding verbatim from
    # tests/test_mcp_server.py (spawn run_mcp_server(path=str(tmp_path / "mem")),
    # connect a real MCP client, exercise tools).
    # 1) add_fact(entity="user", attribute="dog", value="Max")
    #    -> assert response contains "fact recorded:"
    # 2) list_facts(entity="user")
    #    -> assert response contains "user.dog = Max"
    # 3) fact_history(fact_id=<id parsed from list_facts>)
    #    -> assert response contains '"op": "insert"'
    ...
```

> Implementer: reuse `tests/test_mcp_server.py`'s existing spawn/client pattern
> unchanged; only the tool calls and assertions above are new.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_mcp_fact_layer.py -v`
Expected: FAIL (tools not exposed)

- [ ] **Step 3: Implement the three tools**

```python
    @mcp.tool()
    async def add_fact(entity: str, attribute: str, value: str, user_id: str = "") -> str:
        """Record a governed fact about an entity (versioned stores only).
        Facts are deterministic lookups with provenance — prefer this over
        free-text memories when the agent learns a durable, structured truth.
        Single-valued attributes (per the attribute ontology) supersede their
        old value; multi-valued ones append.

        Example: add_fact(entity="user", attribute="dog", value="Max")
        """
        fid = core.add_fact(entity, attribute, value, user_id=user_id)
        return f"fact recorded: {fid}"

    @mcp.tool()
    async def list_facts(entity: str = "", attribute: str = "", user_id: str = "") -> str:
        """List current governed facts (versioned stores only), optionally by
        entity/attribute. Deterministic, zero-LLM — use for direct questions
        like "what do you know about the user's pets?".

        Example: list_facts(entity="user")
        """
        facts = core.list_facts(
            entity=entity or None, attribute=attribute or None,
            user_id=user_id if user_id else None,
        )
        if not facts:
            return "no facts recorded"
        return "\n".join(
            f"{f['id']}: {f['entity']}.{f['attribute']} = {f['value']} "
            f"(since {f['valid_from'][:10]})"
            for f in facts
        )

    @mcp.tool()
    async def fact_history(fact_id: str) -> str:
        """Provenance timeline of one governed fact (versioned stores only):
        every recorded change on the audit chain, with timestamps and author.

        Example: fact_history(fact_id="a1b2c3d4e5f6")
        """
        return json.dumps(core.fact_history(fact_id), indent=1, default=str)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_mcp_fact_layer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add coremem/mcp_server.py tests/test_mcp_fact_layer.py
git commit -m "feat: MCP fact tools (add_fact/list_facts/fact_history) — governed memory interface"
```

---

### Task 7: Phase 2 (gated) — opt-in extraction experiment

**Files:**
- Modify: `coremem/core.py` (constructor `extract_facts: bool = False`), ingest path
- Test: `tests/test_fact_layer.py` (extend)

**Interfaces:**
- Consumes: `FactOntology` (Task 1), the chunked-extraction pattern from `coremem/core.py::_extract_and_store_facts` (20 msgs/call, JSON out, defensive parse, degrade-to-warning)
- Produces: constructor `extract_facts: bool = False`; extracted rows land in `facts` via the Task 2 path with `source_message_id` provenance

- [ ] **Step 1: Write the failing test**

```python
class _FactStubProvider:
    async def chat(self, messages):
        import json as _json
        payload = _json.loads(
            messages[0]["content"].split("Messages (JSON list of")[1]
            .split(":", 1)[1].rsplit("Return ONLY", 1)[0]
        )
        out = {}
        for m in payload:
            if "Denver" in m["content"]:
                out[m["id"]] = [{"entity": "user", "attribute": "home_city",
                                 "value": "Denver"}]
        return type("R", (), {"content": _json.dumps(out)})()


def test_extract_facts_ingests_candidates_flag_on():
    core = _make_core(versioned=True, llm_provider=_FactStubProvider(), extract_facts=True)
    try:
        core.ingest_many([
            {"id": "a1", "role": "user", "content": "I live in Denver.", "session_id": "s1"},
        ])
        facts = core.list_facts(entity="user")
        assert any(f["value"] == "Denver" and f["source_message_id"] == "a1" for f in facts)
        assert core.get_fact(facts[0]["id"])["attribute"] == "home_city"
    finally:
        core._test_cleanup()


def test_extract_facts_off_by_default():
    core = _make_core(versioned=True, llm_provider=_FactStubProvider())
    try:
        core.ingest_many([
            {"id": "a1", "role": "user", "content": "I live in Denver.", "session_id": "s1"},
        ])
        assert core.list_facts(entity="user") == []
    finally:
        core._test_cleanup()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_fact_layer.py -v -k extract`
Expected: FAIL (`TypeError: __init__() got an unexpected keyword argument 'extract_facts'`)

- [ ] **Step 3: Implement** — constructor flag + extraction loop inside
`ingest_many` (after rows are built, before the journal flush): chunk user
messages 20/call, prompt asks for a JSON object mapping message id → list of
`{"entity", "attribute", "value"}` objects; route each through the Task 2
insert internals (ontology cardinality applies; `source_message_id` = message
id). Extraction failures log a warning and continue (never raise into ingest) —
same degrade-to-no-facts contract as the falsified factaug path.

- [ ] **Step 4: Run test to verify it passes; run full suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add coremem/core.py tests/test_fact_layer.py
git commit -m "feat: opt-in fact extraction at ingest (Phase 2, candidates on the chain)"
```

---

### Task 8: Phase 2 kill-gate — dual-path answer eval on S stratified-56

**Files:**
- Modify: `scripts/eval_answer_longmemeval.py` (new mode `episodic_4k_reranked_factdigest`)
- Test: the eval run IS the gate

**Interfaces:**
- Consumes: `build_memorycore` (needs `extract_facts` passthrough — add kwarg like the existing `fact_augment` one), `core.list_facts`, `build_facts_digest`
- Produces: paired metrics `episodic_4k_reranked_factdigest` vs control

- [ ] **Step 1: Add the mode** — in `_score_instance_episodic`, when
`fact_digest=True`, append `build_facts_digest(core.list_facts(limit=200))`
after the bundle context (mode name `episodic_4k_reranked_factdigest`);
register in `MODES`, argparse choices, and the `_score_question` dispatch —
follow the existing `memorycore_episodic_reranked_timeprune` pattern exactly.

- [ ] **Step 2: Run the paired eval**

```bash
uv run python3 scripts/eval_answer_longmemeval.py data/longmemeval_s_stratified_56.json \
  --output results/eval_answer_s56_factdigest.json --root /tmp/coremem-factdigest \
  --answer-model "ollama:gpt-oss:120b-cloud" --judge-model "ollama:gpt-oss:120b-cloud" \
  --resume
```

- [ ] **Step 3: Apply the pre-registered kill criteria** (Background §6)

Paired comparison `episodic_4k_reranked_factdigest` vs `episodic_4k_reranked`
(same questions, same contexts + digest, blind judge):
- **Net ≤ 0** → extraction stays opt-in-**off**; record in
  `docs/retrieval-experiments.md` as the lever-2 sibling result. Phase 1 API
  remains regardless — the governance layer stands on its own.
- **Net > 0** → scale to full S, then LoCoMo stratified-30, before considering
  `extract_facts=True` as a default in a future release.

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_answer_longmemeval.py docs/retrieval-experiments.md
git commit -m "eval: fact-digest dual-path A/B on S stratified-56 (kill-gate applied)"
```

---

### Task 9: Ship 0.17.0

**Files:**
- Modify: `pyproject.toml` (version), `coremem/__init__.py` (`__version__` — BOTH, same value), `CHANGELOG.md`, `AGENTS.md` (Public API section)

- [ ] **Step 1: CHANGELOG entry** — headline: fact layer (governed, bi-temporal, provenance-chained facts + dual-path reading). Phase 1 zero-LLM API default-on for versioned stores; Phase 2 extraction opt-in with the gate result quoted verbatim; ontology.toml documented.
- [ ] **Step 2: Bump `0.17.0` in `pyproject.toml` AND `coremem/__init__.py`** (0.13.2 published a drifted `__version__` — never again).
- [ ] **Step 3: `uv lock && uv build`, verify the wheel ships `CHANGELOG.md` and `__version__` matches pyproject.**
- [ ] **Step 4: Publish** with the `.env` token; verify on PyPI (both artifacts + floor).
- [ ] **Step 5: Commit** `release: v0.17.0 — fact layer (semantic layer v0)`.