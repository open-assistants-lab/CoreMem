# Unified recall() API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify all MemoryCore retrieval under a single `recall()` method with a `strategy` parameter, removing deprecated methods and the falsified deterministic classifier.

**Architecture:** Rename `search_messages` and friends to `_`-prefixed internal methods, rewrite `recall()` as the sole public entry point with strategies `direct|episodic|expanded|fusion` and a `bundles` flag, remove 4 deprecated methods + their tests, update eval scripts and docstrings.

**Tech Stack:** Python 3.13, HybridDB, sentence-transformers cross-encoder, pytest

**Spec:** `docs/superpowers/specs/2026-08-06-unified-recall-api-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `coremem/core.py` | Modify | Rename methods to `_`-prefixed, rewrite `recall()`, remove deprecated |
| `coremem/__init__.py` | Modify | Update docstring example, bump version |
| `tests/test_core.py` | Modify | Remove 8 deprecated tests, rewrite 1, update 3 recall tests, update 2 fusion tests |
| `scripts/eval_agent_journal_longmemeval.py` | Modify | Update MODES, dispatch, helper functions |
| `scripts/eval_answer_longmemeval.py` | Modify | Update `core.search_*` calls to `core.recall()` or `core._search_*` |
| `README.md` | Modify | Remove `search_journal()` references |

---

## Task 1: Add filter params to `search_messages_decomposed`

The spec requires `_search_messages_decomposed` to accept filter params (role, session_id, etc.) and pass them through to the internal `search_messages` call. This must happen BEFORE renaming it to `_search_messages_decomposed`.

**Files:**
- Modify: `coremem/core.py:354-383`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_core.py` after the existing `test_search_messages_decomposed` tests (find them by searching for `def test_search_messages_decomposed`):

```python
def test_search_messages_decomposed_with_session_filter():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("user", "I love hiking in Tahoe", session_id="s2")
        results = core.search_messages_decomposed("hiking", limit=5, session_id="s1")
        assert len(results) > 0
        assert all(r.memory.session_id == "s1" for r in results)
    finally:
        core._test_cleanup()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_core.py::test_search_messages_decomposed_with_session_filter -v`
Expected: FAIL with `TypeError: search_messages_decomposed() got an unexpected keyword argument 'session_id'`

- [ ] **Step 3: Add filter params to `search_messages_decomposed`**

In `coremem/core.py`, change the signature of `search_messages_decomposed` (line 354) from:

```python
    def search_messages_decomposed(
        self,
        query: str,
        limit: int = 5,
        per_query_limit: int = 20,
        use_cross_encoder: bool = False,
    ) -> list[SearchResult]:
```

to:

```python
    def search_messages_decomposed(
        self,
        query: str,
        limit: int = 5,
        per_query_limit: int = 20,
        use_cross_encoder: bool = False,
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
```

Then change the internal `self.search_messages(variant, limit=per_query_limit)` call (line 368) to pass filters:

```python
            for rank, result in enumerate(
                self.search_messages(
                    variant, limit=per_query_limit,
                    role=role, session_id=session_id, user_id=user_id,
                    agent_id=agent_id, ts_after=ts_after, ts_before=ts_before,
                    metadata=metadata,
                ), start=1,
            ):
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_core.py::test_search_messages_decomposed_with_session_filter -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 122 passed (121 existing + 1 new)

- [ ] **Step 6: Commit**

```bash
git add coremem/core.py tests/test_core.py
git commit -m "feat: add filter params to search_messages_decomposed"
```

---

## Task 2: Rewrite `recall()` with new strategies and `bundles` flag

Replace the current `recall()` (which has a deterministic classifier for "auto") with the new unified version.

**Files:**
- Modify: `coremem/core.py:1029-1120`

- [ ] **Step 1: Write the failing test for bundles**

Add to `tests/test_core.py` after the existing recall tests:

```python
def test_recall_bundles_returns_session_bundles():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("assistant", "Yosemite is beautiful in spring", session_id="s1")
        core.ingest("user", "What about skiing?", session_id="s2")
        bundles = core.recall("hiking Yosemite", bundles=True)
        assert len(bundles) > 0
        from coremem import SessionBundle
        assert isinstance(bundles[0], SessionBundle)
    finally:
        core._test_cleanup()


def test_recall_default_is_episodic():
    core = _tmp_core()
    try:
        core.ingest("user", "hello world", session_id="s1")
        results = core.recall("hello", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()


def test_recall_fusion_strategy():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("assistant", "Yosemite is beautiful in spring", session_id="s1")
        results = core.recall("hiking Yosemite", strategy="fusion", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python3 -m pytest tests/test_core.py::test_recall_bundles_returns_session_bundles tests/test_core.py::test_recall_default_is_episodic tests/test_core.py::test_recall_fusion_strategy -v`
Expected: FAIL (bundles test: `recall()` doesn't accept `bundles=`; default test: uses "auto" classifier which may return empty; fusion test: `recall()` doesn't accept `strategy="fusion"`)

- [ ] **Step 3: Replace the `recall()` method**

In `coremem/core.py`, replace the entire `recall` method (lines 1029-1120) with:

```python
    def recall(
        self,
        query: str,
        *,
        strategy: str = "episodic",
        limit: int = 5,
        bundles: bool = False,
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult] | list[SessionBundle]:
        if strategy == "direct":
            if bundles:
                return self._reconstruct_sessions(
                    query, session_limit=limit,
                    primary_results=self._search_messages(
                        query, limit=limit, role=role, session_id=session_id,
                        user_id=user_id, agent_id=agent_id, ts_after=ts_after,
                        ts_before=ts_before, metadata=metadata,
                    ),
                )
            return self._search_messages(
                query, limit=limit, role=role, session_id=session_id,
                user_id=user_id, agent_id=agent_id, ts_after=ts_after,
                ts_before=ts_before, metadata=metadata,
            )

        if strategy == "expanded":
            results = self._search_messages_llm_expansion(
                query, limit=limit, role=role, session_id=session_id,
                user_id=user_id, agent_id=agent_id, ts_after=ts_after,
                ts_before=ts_before, metadata=metadata,
            )
            if bundles:
                return self._reconstruct_sessions(
                    query, session_limit=limit, primary_results=results,
                )
            return results

        if strategy == "fusion":
            results = self._search_with_fusion(query, limit=limit)
            if bundles:
                return self._reconstruct_sessions(
                    query, session_limit=limit, primary_results=results,
                )
            return results

        if strategy == "episodic":
            primary = self._search_messages_decomposed(
                query, limit=limit, per_query_limit=max(20, limit * 4),
                use_cross_encoder=True,
                role=role, session_id=session_id, user_id=user_id,
                agent_id=agent_id, ts_after=ts_after, ts_before=ts_before,
                metadata=metadata,
            )
            if bundles:
                return self._reconstruct_sessions(
                    query, session_limit=limit, primary_results=primary,
                )
            return primary

        raise ValueError(f"unknown strategy: {strategy}")
```

Note: This references `self._search_messages`, `self._search_messages_decomposed`, `self._search_messages_llm_expansion`, `self._reconstruct_sessions`, `self._search_with_fusion` — which don't exist yet (they're still public names). The rename happens in Task 3. For now, the tests will fail with `AttributeError`. That's expected — we'll fix it in Task 3.

Actually, to keep tests green between tasks, let's do the rename FIRST in this same step. Add temporary aliases at the bottom of the class (before the closing of the class, after `search_with_fusion`):

```python
    # Temporary aliases — will be removed in Task 3
    _search_messages = search_messages
    _search_messages_decomposed = search_messages_decomposed
    _search_messages_llm_expansion = search_messages_llm_expansion
    _reconstruct_sessions = reconstruct_sessions
    _search_with_fusion = search_with_fusion
```

Wait — these can't be class-level assignments referencing methods that way. Instead, keep the old method names as the actual definitions and add `_`-prefixed aliases as module-level functions... No, simplest: just have `recall()` call the public methods directly for now, and rename in Task 3.

Let me revise: in Step 3, write `recall()` calling `self.search_messages(...)` etc. (the current public names). Then Task 3 renames them and updates `recall()` to call the `_`-prefixed versions.

**Revised Step 3:** Replace `recall()` with:

```python
    def recall(
        self,
        query: str,
        *,
        strategy: str = "episodic",
        limit: int = 5,
        bundles: bool = False,
        role: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        ts_after: str | None = None,
        ts_before: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult] | list[SessionBundle]:
        if strategy == "direct":
            results = self.search_messages(
                query, limit=limit, role=role, session_id=session_id,
                user_id=user_id, agent_id=agent_id, ts_after=ts_after,
                ts_before=ts_before, metadata=metadata,
            )
            if bundles:
                return self.reconstruct_sessions(
                    query, session_limit=limit, primary_results=results,
                )
            return results

        if strategy == "expanded":
            results = self.search_messages_llm_expansion(
                query, limit=limit, role=role, session_id=session_id,
                user_id=user_id, agent_id=agent_id, ts_after=ts_after,
                ts_before=ts_before, metadata=metadata,
            )
            if bundles:
                return self.reconstruct_sessions(
                    query, session_limit=limit, primary_results=results,
                )
            return results

        if strategy == "fusion":
            results = self.search_with_fusion(query, limit=limit)
            if bundles:
                return self.reconstruct_sessions(
                    query, session_limit=limit, primary_results=results,
                )
            return results

        if strategy == "episodic":
            primary = self.search_messages_decomposed(
                query, limit=limit, per_query_limit=max(20, limit * 4),
                use_cross_encoder=True,
                role=role, session_id=session_id, user_id=user_id,
                agent_id=agent_id, ts_after=ts_after, ts_before=ts_before,
                metadata=metadata,
            )
            if bundles:
                return self.reconstruct_sessions(
                    query, session_limit=limit, primary_results=primary,
                )
            return primary

        raise ValueError(f"unknown strategy: {strategy}")
```

- [ ] **Step 4: Run new tests to verify they pass**

Run: `uv run python3 -m pytest tests/test_core.py::test_recall_bundles_returns_session_bundles tests/test_core.py::test_recall_default_is_episodic tests/test_core.py::test_recall_fusion_strategy -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 125 passed (122 + 3 new)

- [ ] **Step 6: Commit**

```bash
git add coremem/core.py tests/test_core.py
git commit -m "feat: rewrite recall() with strategies and bundles flag"
```

---

## Task 3: Rename methods to `_`-prefixed internal

Rename `search_messages` → `_search_messages`, `search_messages_decomposed` → `_search_messages_decomposed`, etc. Update all internal callers.

**Files:**
- Modify: `coremem/core.py` (multiple locations)

- [ ] **Step 1: Rename the 5 methods in core.py**

In `coremem/core.py`, rename these method definitions (use find-and-replace, one at a time):

1. `def search_messages(` → `def _search_messages(` (line ~252)
2. `def search_messages_llm_expansion(` → `def _search_messages_llm_expansion(` (line ~290)
3. `def search_messages_decomposed(` → `def _search_messages_decomposed(` (line ~354)
4. `def reconstruct_sessions(` → `def _reconstruct_sessions(` (line ~385)
5. `def search_with_fusion(` → `def _search_with_fusion(` (line ~1162)

- [ ] **Step 2: Update all internal `self.search_*` calls in core.py**

Find all calls to the old names within `core.py` and update them:

- `self.search_messages(` → `self._search_messages(` (inside `_search_messages_decomposed` at line ~368, inside `_search_with_fusion` at line ~1171, inside `search_with_traversal` at line ~906, inside `search_with_session_reranking` at lines ~1143, ~1149)
- `self.search_messages_decomposed(` → `self._search_messages_decomposed(` (inside `reconstruct_sessions` at line ~399, inside `search_with_traversal` at line ~935 if present, inside `search_with_session_reranking` at line ~1135, inside `search_with_fusion` at line ~1172)
- `self.search_messages_llm_expansion(` → `self._search_messages_llm_expansion(` (inside `recall` at the `expanded` branch)
- `self.reconstruct_sessions(` → `self._reconstruct_sessions(` (inside `recall` at all branches with `bundles=True`)
- `self.search_with_fusion(` → `self._search_with_fusion(` (inside `recall` at the `fusion` branch)

Use this command to find all occurrences:
```bash
grep -n "self\.search_messages\b\|self\.search_messages_decomposed\|self\.search_messages_llm_expansion\|self\.reconstruct_sessions\|self\.search_with_fusion" coremem/core.py
```

- [ ] **Step 3: Update `recall()` to use `_`-prefixed names**

The `recall()` method written in Task 2 calls `self.search_messages(...)` etc. Update every call in `recall()` to use `self._search_messages(...)`, `self._search_messages_decomposed(...)`, `self._search_messages_llm_expansion(...)`, `self._reconstruct_sessions(...)`, `self._search_with_fusion(...)`.

- [ ] **Step 4: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 125 passed (no behavior change, just renames)

- [ ] **Step 5: Commit**

```bash
git add coremem/core.py
git commit -m "refactor: rename search_* methods to _-prefixed internal"
```

---

## Task 4: Remove deprecated methods and update existing tests

Remove `search_with_traversal`, `search_with_context`, `search_with_session_reranking`, `search_journal`, and the `SearchHit` import. Remove/update their tests.

**Files:**
- Modify: `coremem/core.py`
- Modify: `tests/test_core.py`

- [ ] **Step 1: Remove deprecated tests from test_core.py**

Remove these test functions from `tests/test_core.py` (delete the entire function including its body):

1. `test_search_with_context_adds_temporal_neighbors` (line ~374)
2. `test_search_with_traversal_discovers_temporal_neighbor` (line ~494)
3. `test_search_with_traversal_keeps_strong_seed_over_irrelevant_neighbor` (line ~510)
4. `test_search_with_traversal_preserves_session_diversity` (line ~526)
5. `test_search_with_traversal_is_bounded_and_deterministic` (line ~553)
6. `test_search_with_traversal_empty` (line ~573)
7. `test_search_with_session_reranking_returns_results` (line ~644)
8. `test_search_with_session_reranking_empty` (line ~655)

- [ ] **Step 2: Rewrite the journal record test**

Replace `test_core_store_journal_record_and_search_with_context` (line ~333) with:

```python
def test_core_store_journal_record_and_recall():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in the mountains", session_id="session_a")
        core.ingest("assistant", "Me too! The views are amazing.", session_id="session_a")
        core.ingest("user", "What about skiing?", session_id="session_b")
        core.ingest("assistant", "Skiing is great too.", session_id="session_b")

        core._store_journal_record("session_a", "# Summary\n\nThe user enjoys hiking in the mountains and the assistant agreed.")
        core._store_journal_record("session_b", "# Summary\n\nThe user asked about skiing and the assistant confirmed it's great.")

        results = core.recall("hiking mountains", strategy="direct", limit=5)
        assert len(results) >= 1
        assert any("hiking" in r.memory.content.lower() for r in results)

        results = core.recall("skiing", strategy="direct", limit=5)
        assert len(results) >= 1
        assert any("Skiing" in r.memory.content for r in results)
    finally:
        core._test_cleanup()
```

- [ ] **Step 3: Update fusion tests to use recall()**

Replace `test_search_with_fusion_returns_results` (line ~624) with:

```python
def test_recall_fusion_returns_results():
    core = _tmp_core()
    try:
        core.ingest("user", "I love hiking in Yosemite", session_id="s1")
        core.ingest("assistant", "Yosemite is beautiful in spring", session_id="s1")
        results = core.recall("hiking Yosemite", strategy="fusion", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()
```

Replace `test_search_with_fusion_empty` (line ~635) with:

```python
def test_recall_fusion_empty():
    core = _tmp_core()
    try:
        results = core.recall("nonexistent query", strategy="fusion", limit=5)
        assert results == []
    finally:
        core._test_cleanup()
```

- [ ] **Step 4: Update existing recall tests**

Replace `test_recall_auto_routes_direct_query` (line ~593) with:

```python
def test_recall_default_returns_results():
    core = _tmp_core()
    try:
        core.ingest("user", "hello world", session_id="s1")
        results = core.recall("hello", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()
```

Replace `test_recall_auto_routes_temporal_query` (line ~603) with:

```python
def test_recall_episodic_temporal():
    core = _tmp_core()
    try:
        core.ingest("user", "event one", session_id="s1")
        core.ingest("user", "event two", session_id="s2")
        results = core.recall("how many days before event one did event two happen", strategy="episodic", limit=5)
        assert len(results) > 0
    finally:
        core._test_cleanup()
```

`test_recall_direct_returns_memorycore_results` (line ~582) and `test_recall_unknown_strategy_raises` (line ~614) are already correct — no change needed.

- [ ] **Step 5: Remove deprecated methods from core.py**

Delete these method definitions from `coremem/core.py`:

1. `search_journal` (line ~815-816) — the entire method:
```python
    def search_journal(self, query: str, limit: int = 5) -> list[SearchHit]:
        return self._agent_journal_search.search(query, limit=limit)
```

2. `search_with_context` (line ~828-889) — the entire method including the `import warnings` and `DeprecationWarning`

3. `search_with_traversal` (line ~891 to the end of the method, approximately line ~1027) — the entire method including `import warnings` and `DeprecationWarning`

4. `search_with_session_reranking` (line ~1122-1160) — the entire method including `import warnings` and `DeprecationWarning`

- [ ] **Step 6: Remove `SearchHit` from imports**

In `coremem/core.py` line 24, remove `SearchHit` from the import:

Change:
```python
from coremem.agent_journal import (
    AgentJournalBundle,
    AgentJournalCompileResult,
    AgentJournalError,
    AgentJournalLLMCompiler,
    AgentJournalSearch,
    CrossEncoderReranker,
    SearchHit,
    dream,
    rebuild_index,
)
```

to:
```python
from coremem.agent_journal import (
    AgentJournalBundle,
    AgentJournalCompileResult,
    AgentJournalError,
    AgentJournalLLMCompiler,
    AgentJournalSearch,
    CrossEncoderReranker,
    dream,
    rebuild_index,
)
```

- [ ] **Step 7: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 117 passed (125 - 8 removed) — wait, we also added 4 new tests in Task 2 and removed 8, so: 125 - 8 = 117. But we also rewrote 1 (doesn't change count) and added 3 new in Task 2. Let me recount: Start 121 + 1 (Task 1) + 3 (Task 2) = 125. Remove 8 (Step 1) = 117. Rewrite 1 (Step 2, no count change) + update 4 (Steps 3-4, no count change). Final: 117 passed.

- [ ] **Step 8: Commit**

```bash
git add coremem/core.py tests/test_core.py
git commit -m "refactor: remove deprecated methods, update tests for recall()"
```

---

## Task 5: Update eval script

Update MODES tuple, remove deprecated mode dispatch, update helper functions to use `recall()` or `_`-prefixed methods.

**Files:**
- Modify: `scripts/eval_agent_journal_longmemeval.py`

- [ ] **Step 1: Update MODES tuple**

In `scripts/eval_agent_journal_longmemeval.py` line 39, change:

```python
MODES = ("raw_bm25", "memorycore", "memorycore_llm_expansion", "memorycore_decomposed", "memorycore_episodic", "memorycore_episodic_reranked", "memorycore_episodic_reranked_4k", "memorycore_fusion")
```

to:

```python
MODES = ("raw_bm25", "memorycore", "memorycore_llm_expansion", "memorycore_episodic_reranked", "memorycore_episodic_reranked_4k", "memorycore_fusion")
```

- [ ] **Step 2: Update module docstring**

Change lines 3-10 from:

```python
"""Deterministic LongMemEval eval — BM25 baseline + MemoryCore search modes.

Modes:
  raw_bm25                        BM25 over in-memory messages (baseline)
  memorycore                      search_messages()
  memorycore_llm_expansion        search_messages_llm_expansion() (1 LLM call)
  memorycore_decomposed           search_messages_decomposed() (query decomposition)
  memorycore_episodic             search_messages_decomposed() + reconstruct_sessions()
  memorycore_episodic_reranked    + cross-encoder reranking (default)
  memorycore_episodic_reranked_4k + reranking, 4k context budget
  memorycore_fusion               RRF fusion of memorycore + episodic_reranked
"""
```

to:

```python
"""Deterministic LongMemEval eval — BM25 baseline + MemoryCore search modes.

Modes:
  raw_bm25                        BM25 over in-memory messages (baseline)
  memorycore                      recall(strategy="direct")
  memorycore_llm_expansion        recall(strategy="expanded") (1 LLM call)
  memorycore_episodic_reranked    recall(strategy="episodic") (default)
  memorycore_episodic_reranked_4k episodic with 4k context budget
  memorycore_fusion               recall(strategy="fusion") RRF
"""
```

- [ ] **Step 3: Update helper functions to use `recall()`**

Replace `_search_messages_mode` (line ~207) with:

```python
def _search_messages_mode(core: MemoryCore, query: str, k: int) -> list[RawSearchHit]:
    results = core.recall(query, strategy="direct", limit=k)
    return _results_to_hits(results)


def _search_messages_llm_expansion_mode(core: MemoryCore, query: str, k: int) -> list[RawSearchHit]:
    results = core.recall(query, strategy="expanded", limit=k)
    return _results_to_hits(results)
```

Add a shared helper function. Place it immediately before the existing `_search_messages_mode` function (currently at line ~207):

```python
def _results_to_hits(results: list) -> list[RawSearchHit]:
    return [
        RawSearchHit(
            message=ReferenceMessage(
                turn_id=str((r.memory.metadata or {}).get("turn_id") or r.memory.id or ""),
                session_id=r.memory.session_id or "",
                message_id=r.memory.id or "",
                role=r.memory.role,
                content=r.memory.content,
            ),
            score=r.score,
        )
        for r in results
    ]
```

- [ ] **Step 4: Remove `_search_messages_decomposed_mode`**

Delete the `_search_messages_decomposed_mode` function (line ~241-255) entirely.

- [ ] **Step 5: Remove `_score_instance_decomposed`**

Delete the `_score_instance_decomposed` function (line ~728-739) entirely.

- [ ] **Step 6: Update `_score_instance_memorycore` to use recall()**

Replace `_score_instance_memorycore` (line ~712) with:

```python
def _score_instance_memorycore(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
    deep: bool = False,
) -> dict[str, Any]:
    mode = "memorycore_llm_expansion" if deep else "memorycore"
    if truth.abstention_expected:
        return _empty_score(instance, truth, mode=mode)
    search_fn = _search_messages_llm_expansion_mode if deep else _search_messages_mode
    hits = search_fn(core, instance.query, k)
    return _build_scored_row(instance, truth, hits, mode=mode, k=k)
```

(This already works — `_search_messages_mode` now calls `recall(strategy="direct")` and `_search_messages_llm_expansion_mode` calls `recall(strategy="expanded")`.)

- [ ] **Step 7: Update `_score_instance_episodic` to use internal methods**

The `_score_instance_episodic` function (line ~742) currently calls `core.search_messages_decomposed` and `core.reconstruct_sessions`. Update these to `core._search_messages_decomposed` and `core._reconstruct_sessions`:

Change line ~764:
```python
    primary = core.search_messages_decomposed(
```
to:
```python
    primary = core._search_messages_decomposed(
```

Change line ~784:
```python
    bundles = core.reconstruct_sessions(
```
to:
```python
    bundles = core._reconstruct_sessions(
```

- [ ] **Step 8: Update `_score_instance_fusion` to use recall()**

Replace `_score_instance_fusion` (line ~808) with:

```python
def _score_instance_fusion(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
) -> dict[str, Any]:
    mode = "memorycore_fusion"
    if truth.abstention_expected:
        return _empty_score(instance, truth, mode=mode)
    results = core.recall(instance.query, strategy="fusion", limit=k)
    hits = _results_to_hits(results)
    return _build_scored_row(instance, truth, hits, mode=mode, k=k)
```

- [ ] **Step 9: Update `_score_question` dispatch**

Replace the `_score_question` function (line ~443) with:

```python
def _score_question(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    active_modes: Sequence[str],
    k: int,
) -> dict[str, dict[str, Any]]:
    """Score one question across all active modes. Returns {mode: row}."""
    new_rows: dict[str, dict[str, Any]] = {}
    for m in active_modes:
        if m == "memorycore":
            new_rows[m] = _score_instance_memorycore(core, instance, truth, k=k, deep=False)
        elif m == "memorycore_llm_expansion":
            new_rows[m] = _score_instance_memorycore(core, instance, truth, k=k, deep=True)
        elif m == "memorycore_episodic_reranked":
            new_rows[m] = _score_instance_episodic(
                core, instance, truth, k=k, use_cross_encoder=True,
            )
        elif m == "memorycore_episodic_reranked_4k":
            new_rows[m] = _score_instance_episodic(
                core, instance, truth, k=k, use_cross_encoder=True, max_context_chars=4_000,
            )
        elif m == "memorycore_fusion":
            new_rows[m] = _score_instance_fusion(core, instance, truth, k=k)
    return new_rows
```

- [ ] **Step 10: Verify eval script imports correctly**

Run:
```bash
uv run python3 -c "
import sys; sys.path.insert(0, 'scripts')
import eval_agent_journal_longmemeval as e
print('MODES:', e.MODES)
print('OK')
"
```
Expected: prints MODES tuple without `memorycore_decomposed` or `memorycore_episodic`, then `OK`

- [ ] **Step 11: Verify eval script --help works**

Run: `uv run python3 scripts/eval_agent_journal_longmemeval.py --help`
Expected: help output with updated mode choices

- [ ] **Step 12: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 117 passed

- [ ] **Step 13: Commit**

```bash
git add scripts/eval_agent_journal_longmemeval.py
git commit -m "refactor: update eval script for unified recall() API"
```

---

## Task 6: Update answer eval script

Update `eval_answer_longmemeval.py` to use `recall()` or `_`-prefixed methods.

**Files:**
- Modify: `scripts/eval_answer_longmemeval.py`

- [ ] **Step 1: Update all `core.search_*` and `core.reconstruct_sessions` calls**

In `scripts/eval_answer_longmemeval.py`, make these replacements:

Line ~162: `core.search_messages(` → `core.recall(query=instance.query, strategy="direct",`
Actually, the calls have different signatures. Let me be precise.

Replace lines 162-223 with:

```python
        started = time.perf_counter()
        basic = core.recall(instance.query, strategy="direct", limit=5)
        retrieval_seconds["memorycore"] = time.perf_counter() - started
        contexts["memorycore"] = _format_messages([result.memory for result in basic])

        started = time.perf_counter()
        primary = core._search_messages_decomposed(instance.query, limit=5, per_query_limit=20)
        retrieval_seconds["decomposition_only"] = time.perf_counter() - started
        contexts["decomposition_only"] = _format_messages([result.memory for result in primary])

        started = time.perf_counter()
        basic_bundles = core._reconstruct_sessions(
            instance.query,
            session_limit=5,
            max_context_chars=16_000,
            primary_results=basic,
        )
        retrieval_seconds["reconstruction_only"] = time.perf_counter() - started
        contexts["reconstruction_only"] = _format_bundles(basic_bundles)

        started = time.perf_counter()
        bundles = core._reconstruct_sessions(
            instance.query,
            session_limit=5,
            max_context_chars=16_000,
            primary_results=primary,
        )
        episodic_retrieval_seconds = time.perf_counter() - started
        retrieval_seconds["episodic_no_headers"] = episodic_retrieval_seconds
        retrieval_seconds["memorycore_episodic"] = episodic_retrieval_seconds
        contexts["episodic_no_headers"] = _format_bundles_without_headers(bundles)
        contexts["memorycore_episodic"] = _format_bundles(bundles)

        started = time.perf_counter()
        small_bundles = core._reconstruct_sessions(
            instance.query,
            session_limit=5,
            max_context_chars=4_000,
            primary_results=primary,
        )
        retrieval_seconds["episodic_4k"] = time.perf_counter() - started
        contexts["episodic_4k"] = _format_bundles(small_bundles)

        started = time.perf_counter()
        reranked_primary = core._search_messages_decomposed(
            instance.query,
            limit=5,
            per_query_limit=20,
            use_cross_encoder=True,
        )
        reranked_bundles = core._reconstruct_sessions(
            instance.query,
            session_limit=5,
            max_context_chars=4_000,
            primary_results=reranked_primary,
        )
        retrieval_seconds["episodic_4k_reranked"] = time.perf_counter() - started
        contexts["episodic_4k_reranked"] = _format_bundles(reranked_bundles)

        started = time.perf_counter()
        deep = core.recall(instance.query, strategy="expanded", limit=5)
        retrieval_seconds["memorycore_llm_expansion"] = time.perf_counter() - started
        contexts["memorycore_llm_expansion"] = _format_messages([result.memory for result in deep])
```

Note: `basic` and `deep` use `recall()` (public API). `primary`, `reranked_primary`, and all `bundles` use `_`-prefixed internal methods (because they need params like `per_query_limit`, `use_cross_encoder`, `max_context_chars` not exposed on `recall()`).

- [ ] **Step 2: Verify script imports correctly**

Run:
```bash
uv run python3 -c "
import sys; sys.path.insert(0, 'scripts')
import eval_answer_longmemeval
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 117 passed

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_answer_longmemeval.py
git commit -m "refactor: update answer eval script for unified recall() API"
```

---

## Task 7: Update docstrings and version

Update module/class docstrings and bump version to 0.11.0.

**Files:**
- Modify: `coremem/__init__.py`
- Modify: `coremem/core.py`
- Modify: `README.md`

- [ ] **Step 1: Update `coremem/__init__.py` docstring**

Change line 7 from:
```python
    results = core.search_messages("How many model kits?")
```
to:
```python
    results = core.recall("How many model kits?")
```

Change line 18 from:
```python
__version__ = "0.10.0"
```
to:
```python
__version__ = "0.11.0"
```

- [ ] **Step 2: Update `MemoryCore` class docstring**

In `coremem/core.py` lines 130-139, change:

```python
    """Unified memory for AI agents. One HybridDB, AgentJournal for compilation.

    Usage:
        core = MemoryCore(path="./memory")
        tid = core.ingest("user", "I like coffee", session_id="s1")
        core.ingest("assistant", "Great!", session_id="s1")
        await core.compile_turn(turn_id=tid)
        results = core.search_messages("coffee")
        hits = core.search_journal("coffee")
    """
```

to:

```python
    """Unified memory for AI agents. One HybridDB, AgentJournal for compilation.

    Usage:
        core = MemoryCore(path="./memory")
        tid = core.ingest("user", "I like coffee", session_id="s1")
        core.ingest("assistant", "Great!", session_id="s1")
        await core.compile_turn(turn_id=tid)
        results = core.recall("coffee")
    """
```

- [ ] **Step 3: Update README.md**

In `README.md`, remove or update references to `search_journal()`. Search for `search_journal` and either remove those table rows or replace with `recall()` references. Leave historical CHANGELOG references as-is.

Run: `grep -n "search_journal" README.md` to find references, then edit each one.

- [ ] **Step 4: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 117 passed

- [ ] **Step 5: Commit**

```bash
git add coremem/__init__.py coremem/core.py README.md
git commit -m "docs: update docstrings for unified recall() API, bump to 0.11.0"
```

---

## Task 8: Final verification

Verify everything works end-to-end.

- [ ] **Step 1: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 117 passed, 0 warnings (deprecation warnings should be gone)

- [ ] **Step 2: Verify eval script --help**

Run: `uv run python3 scripts/eval_agent_journal_longmemeval.py --help`
Expected: mode choices show `raw_bm25,memorycore,memorycore_llm_expansion,memorycore_episodic_reranked,memorycore_episodic_reranked_4k,memorycore_fusion,all`

- [ ] **Step 3: Verify no stale references to removed methods**

Run:
```bash
grep -rn "search_with_traversal\|search_with_context\|search_with_session_reranking\|search_journal\|\.search_messages\b\|\.search_messages_decomposed\|\.search_messages_llm_expansion\|\.reconstruct_sessions\|\.search_with_fusion" coremem/ scripts/ tests/ --include="*.py"
```
Expected: no matches (all references should be to `recall()` or `_`-prefixed internal methods)

Note: `grep "\.search_messages\b"` uses word boundary — it should NOT match `_search_messages`. If grep finds matches in `_search_messages_decomposed` etc., that's fine — those are the new internal names. Only flag references to the old public names without the `_` prefix.

- [ ] **Step 4: Verify import works**

Run:
```bash
uv run python3 -c "
from coremem import MemoryCore, SessionBundle, SearchResult
from coremem import MemoryCore as MC
core = MC.__init__  # just verify class loads
print('version:', __import__('coremem').__version__)
print('OK')
"
```
Expected: `version: 0.11.0` then `OK`

- [ ] **Step 5: Final commit if needed**

If any files were changed during verification:
```bash
git add -A && git commit -m "fix: final verification adjustments"
```

Otherwise, no commit needed — the implementation is complete.