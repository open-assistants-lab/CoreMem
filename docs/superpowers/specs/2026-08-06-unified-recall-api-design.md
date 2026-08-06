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
    # filter params (passed through to direct/episodic/expanded)
    role: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    ts_after: str | None = None,
    ts_before: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[SearchResult] | list[SessionBundle]:
```

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

### Bundle mode (`bundles=True`)

When `bundles=True`, `recall()` runs the `episodic` strategy to get primary
results, then calls `_reconstruct_sessions()` to build `SessionBundle` objects
from the surrounding context. Returns `list[SessionBundle]`.

The `max_context_chars` budget for bundle reconstruction defaults to 16,000.
This is not exposed as a parameter — it's an internal constant. If a caller
needs a different budget, they can use the internal `_reconstruct_sessions()`
method directly.

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

- `test_search_with_traversal_returns_results`
- `test_search_with_traversal_empty`
- `test_search_with_session_reranking_returns_results`
- `test_search_with_session_reranking_empty`

### Tests updated

- `test_recall_direct` — calls `recall(query, strategy="direct")`
- `test_recall_auto` — renamed to `test_recall_default`, calls `recall(query)` with no strategy arg
- `test_recall_auto_temporal` — renamed to `test_recall_episodic_temporal`, calls `recall(query, strategy="episodic")`
- `test_recall_invalid_strategy` — still tests `recall("hello", strategy="invalid")`

## Eval Script Impact

The eval script's `_score_question` dispatch changes from calling individual
methods to calling `recall()` with different strategies:

```python
for m in active_modes:
    if m == "memorycore":
        new_rows[m] = _score_instance_recall(core, instance, truth, k=k, strategy="direct")
    elif m == "memorycore_llm_expansion":
        new_rows[m] = _score_instance_recall(core, instance, truth, k=k, strategy="expanded")
    elif m == "memorycore_episodic":
        new_rows[m] = _score_instance_recall(core, instance, truth, k=k, strategy="episodic")
    elif m == "memorycore_episodic_reranked":
        new_rows[m] = _score_instance_recall(core, instance, truth, k=k, strategy="episodic")
    elif m == "memorycore_episodic_reranked_4k":
        new_rows[m] = _score_instance_recall(core, instance, truth, k=k, strategy="episodic")
    elif m == "memorycore_fusion":
        new_rows[m] = _score_instance_recall(core, instance, truth, k=k, strategy="fusion")
```

Note: `memorycore_episodic` and `memorycore_episodic_reranked` both map to
`strategy="episodic"` since the cross-encoder is now always on for the episodic
strategy. The eval mode names stay the same for backwards compatibility with
existing result files.

### Eval modes removed

- `memorycore_decomposed` — was an ablation (episodic without cross-encoder).
  Removed from `MODES` tuple since the reranked version is strictly better.
- `memorycore_episodic` — kept in `MODES` as an alias for
  `memorycore_episodic_reranked` for backwards-compatible result comparison.

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

## Verification

- All remaining tests pass (117 after removing 4 deprecated tests)
- Eval script imports correctly
- `recall()` with each strategy produces same results as the individual methods did