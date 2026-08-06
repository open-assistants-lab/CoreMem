# Unified `recall()` API Design

**Date:** 2026-08-06
**Status:** Approved
**Supersedes:** Individual `search_*` methods on `MemoryCore`

## Motivation

The current public API has six methods for retrieval, each with different
signatures and no clear default:

- `search_messages(query, limit, ...filters)` — direct BM25+hybrid
- `search_messages_llm_expansion(query, limit, ...filters)` — 1 LLM call
- `search_messages_decomposed(query, limit, per_query_limit, use_cross_encoder)` — query decomposition + reranking
- `reconstruct_sessions(query, ...)` — session bundle reconstruction
- `search_with_fusion(query, limit)` — RRF fusion
- `recall(query, strategy, limit)` — unified entry point with deterministic classifier

Plus four deprecated methods (`search_with_traversal`, `search_with_context`,
`search_with_session_reranking`, `search_journal`) that are below baseline.

This is confusing for users. There's no obvious entry point, no clear default,
and the method names don't communicate what they do.

The eval results falsified the deterministic classifier in `recall(strategy="auto")`
(78% accuracy on S — not reliable enough for production). It should be removed.

## Decision

Unify all retrieval under a single `recall()` method with a `strategy` parameter.
The default strategy is `episodic` (episodic reranked), which is the strongest
zero-LLM-retrieval mode across both oracle and S evaluations.

## API

```python
def recall(
    self,
    query: str,
    *,
    strategy: str = "episodic",   # "direct" | "episodic" | "expanded" | "fusion"
    limit: int = 5,
    bundles: bool = False,         # return SessionBundle list instead of SearchResult list
    # filter params (passed through to direct/episodic/expanded; ignored by fusion)
    role: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    ts_after: str | None = None,
    ts_before: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[SearchResult] | list[SessionBundle]:
```

**Filter params and fusion:** The `fusion` strategy ignores filter params
because it combines `direct` + `episodic`, and applying filters to only one
side would skew the fusion. If filtered fusion is needed, the caller can
use `recall(query, strategy="direct", ...)` and `recall(query, strategy="episodic", ...)`
separately and combine results.

### Usage

```python
# Default: episodic reranking (recommended)
results = core.recall(query)

# Strategies
results = core.recall(query, strategy="direct")      # BM25+hybrid, zero LLM
results = core.recall(query, strategy="episodic")     # decomposed + cross-encoder (default)
results = core.recall(query, strategy="expanded")     # 1 LLM call for query rephrasing
results = core.recall(query, strategy="fusion")       # RRF of direct + episodic

# Session bundles
bundles = core.recall(query, bundles=True)            # returns list[SessionBundle]

# Filter params (passed through to underlying search)
results = core.recall(query, role="user", session_id="abc", ts_after="2024-01-01")

# Limit
results = core.recall(query, limit=10)
```

## Strategy Mapping

| Strategy | Internal method | LLM calls | Description |
|---|---|---|---|
| `direct` | `_search_messages` | 0 | BM25+hybrid search |
| `episodic` | `_search_messages_decomposed` + cross-encoder | 0 | Query decomposition + cross-encoder reranking |
| `expanded` | `_search_messages_llm_expansion` | 1 | LLM query rephrasing + BM25 |
| `fusion` | `_search_with_fusion` | 0 | RRF fusion of direct + episodic |

### Behavior change: episodic now always uses cross-encoder

The current `recall(strategy="episodic")` does NOT use cross-encoder — only
`strategy="auto"` with aggregation queries enables it. The new design makes
`strategy="episodic"` always use cross-encoder, matching the eval's
`memorycore_episodic_reranked` mode (the recommended default). This is a
behavior change, not just a rename. The old non-reranked episodic behavior
is dropped since the reranked version is strictly better (m@5: 0.867 vs 0.472
on oracle).

### Bundle mode (`bundles=True`)

When `bundles=True`, `recall()` runs the `episodic` strategy to get primary
results, then calls `_reconstruct_sessions()` to build `SessionBundle` objects
from the surrounding context. Returns `list[SessionBundle]`.

The `max_context_chars` budget for bundle reconstruction defaults to 16,000.
This is not exposed as a parameter — it's an internal constant. If a caller
needs a different budget, they can use the internal `_reconstruct_sessions()`
method directly.

### Filter params implementation note

`_search_messages` and `_search_messages_llm_expansion` already accept filter
params (role, session_id, etc.). However, `_search_messages_decomposed` does
NOT — it only takes `query, limit, per_query_limit, use_cross_encoder`. The
implementation must add filter params to `_search_messages_decomposed` and
pass them through to the underlying `_search_messages` calls inside it.

## What's Removed

### Methods removed from `MemoryCore`

- `search_with_traversal` — graph traversal, below baseline on every metric
- `search_with_context` — context mode, below baseline
- `search_with_session_reranking` — session reranking, session recall collapsed to 0.623
- `search_journal` — journal modes removed from eval
- The deterministic classifier in `recall(strategy="auto")` — 78% accuracy, falsified

### Methods made internal (prefixed with `_`)

- `search_messages` → `_search_messages`
- `search_messages_llm_expansion` → `_search_messages_llm_expansion`
- `search_messages_decomposed` → `_search_messages_decomposed`
- `reconstruct_sessions` → `_reconstruct_sessions`
- `search_with_fusion` → `_search_with_fusion`

### Tests removed

- `test_search_with_context_adds_temporal_neighbors` — tests removed `search_with_context`
- `test_core_store_journal_record_and_search_with_context` — rewritten as `test_core_store_journal_record_and_recall`: uses `recall()` instead of `search_with_context` to verify journal records are searchable (tests `_store_journal_record` which stays)
- `test_search_with_traversal_discovers_temporal_neighbor`
- `test_search_with_traversal_keeps_strong_seed_over_irrelevant_neighbor`
- `test_search_with_traversal_preserves_session_diversity`
- `test_search_with_traversal_is_bounded_and_deterministic`
- `test_search_with_traversal_empty`
- `test_search_with_session_reranking_returns_results`
- `test_search_with_session_reranking_empty`

### Tests updated

- `test_recall_direct_returns_memorycore_results` — calls `recall(query, strategy="direct")` (already correct)
- `test_recall_auto_routes_direct_query` — renamed to `test_recall_default_returns_results`, calls `recall(query)` with no strategy arg
- `test_recall_auto_routes_temporal_query` — renamed to `test_recall_episodic_temporal`, calls `recall(query, strategy="episodic")`
- `test_recall_unknown_strategy_raises` — still tests `recall("hello", strategy="invalid")` (already correct)

## Eval Script Impact

The eval script's `_score_question` dispatch changes from calling individual
public methods to calling `recall()` for the standard modes, with the 4k
variant using internal methods directly (since `max_context_chars` is not
exposed on `recall()`):

```python
for m in active_modes:
    if m == "memorycore":
        new_rows[m] = _score_instance_recall(core, instance, truth, k=k, strategy="direct")
    elif m == "memorycore_llm_expansion":
        new_rows[m] = _score_instance_recall(core, instance, truth, k=k, strategy="expanded")
    elif m == "memorycore_episodic_reranked":
        new_rows[m] = _score_instance_recall(core, instance, truth, k=k, strategy="episodic")
    elif m == "memorycore_episodic_reranked_4k":
        # 4k budget requires internal access — not exposed on recall()
        new_rows[m] = _score_instance_episodic(core, instance, truth, k=k,
                                               use_cross_encoder=True, max_context_chars=4_000)
    elif m == "memorycore_fusion":
        new_rows[m] = _score_instance_recall(core, instance, truth, k=k, strategy="fusion")
```

Note: `memorycore_episodic_reranked` maps to `strategy="episodic"`
since the cross-encoder is now always on. `memorycore_episodic` is removed
from `MODES` (see Eval modes collapsed below).

### Eval modes removed

- `memorycore_decomposed` — was an ablation (episodic without cross-encoder).
  Removed from `MODES` tuple since the reranked version is strictly better.
  The `_score_instance_decomposed` function and `_search_messages_decomposed_mode`
  helper are removed from the eval script.

### Eval modes collapsed

`memorycore_episodic` and `memorycore_episodic_reranked` now produce identical
results (both use `strategy="episodic"` with cross-encoder). `memorycore_episodic`
is removed from the `MODES` tuple to avoid duplicate computation in `--mode all`
runs. Users wanting the episodic reranked result should use
`--mode memorycore_episodic_reranked`.

### Eval modes needing internal access

- `memorycore_episodic_reranked_4k` — uses `max_context_chars=4_000` which is
  not exposed on `recall()`. The eval script's `_score_instance_episodic`
  function continues to call `core._search_messages_decomposed()` and
  `core._reconstruct_sessions()` directly with the 4k budget parameter. This is
  acceptable — the eval harness is internal tooling and can use `_`-prefixed
  methods.

## Public Exports

`coremem/__init__.py` stays the same:

```python
from coremem.core import MemoryCore
from coremem.heuristics import SearchHeuristics
from coremem.query import decompose_queries, expand_queries
from coremem.rerank import rerank
from coremem.types import Memory, SearchResult, SessionBundle
from coremem.providers import create_provider

__all__ = [
    "MemoryCore", "Memory", "SearchResult", "SessionBundle",
    "SearchHeuristics", "decompose_queries", "expand_queries", "rerank",
    "create_provider",
]
```

## Version Bump

`__version__` goes from `0.10.0` to `0.11.0` (breaking change: method renames).

## Other Scripts Updated

- `scripts/eval_answer_longmemeval.py` — has its own `MODES` tuple with different
  mode names (`decomposition_only`, `reconstruction_only`, etc.). It calls
  `core.search_messages`, `core.search_messages_decomposed`,
  `core.reconstruct_sessions`, and `core.search_messages_llm_expansion` directly.
  These calls must be updated to use `core.recall()` or the `_`-prefixed internal
  methods. The script's own `MODES` and dispatch logic are not otherwise changed.

## Verification

- All remaining tests pass (113 after removing 8 deprecated tests, plus 1 rewritten)
- Eval script imports correctly
- `recall()` with each strategy produces same results as the individual methods did

## Docstrings to Update

- `coremem/__init__.py` module docstring — `core.search_messages(...)` → `core.recall(...)`
- `coremem/core.py` `MemoryCore` class docstring — references `search_messages` and `search_journal`
- `scripts/eval_agent_journal_longmemeval.py` module docstring — references removed `memorycore_decomposed` mode
- `README.md` — references `search_journal()` in result tables and descriptions
- `CHANGELOG.md` — historical references to `search_journal()` (leave as-is; historical record)