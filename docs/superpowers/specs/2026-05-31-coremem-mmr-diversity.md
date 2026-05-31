# CoreMem: MMR Diversity Reranking

2026-05-31

## Context

CoreMem is a zero-LLM message storage and retrieval layer for AI agents. It has
two search paths:

- **`search()`** — backend semantic search + deterministic heuristics (keyword
  overlap, temporal boost, recency decay, person name, quoted phrase). Returns
  ranked `SearchResult` objects. Used by `message_search` tool in EA.
- **`search_enhanced()`** — multi-query expansion + cross-encoder reranking on
  top of `search()`. Higher quality at the cost of latency (cross-encoder model
  load + rerank pass).

The HybridBackend (production backend) already session-deduplicates in its
search method (line 237-241 of `hybrid.py`). Each call to `search()` returns
at most one message per session.

But `search_enhanced()` merges results from **multiple** `search()` calls
(one per expanded query). After merging, the same session can appear in the
candidate set from different sub-queries. The cross-encoder then sees 3+
messages from the same session and can overfit — boosting that session's
vocabulary while penalizing other equally-relevant sessions.

This was discovered during LongMemEval multi-session testing:

```
Q: "How many projects have I led or am currently leading?"
   Answer: 4 sessions (clustering project, case competition, research poster, data mining)
   
   search()          → HIT (rank=1)   ✅ Backend dedup works
   search_enhanced() → MISS           ❌ Cross-encoder overfits to dominant session
```

The fix is a lightweight MMR pass before the cross-encoder: deduplicate the
merged candidate set by session, keeping the highest-scoring message per session.

## Problem

`search_enhanced()` can regress vs `search()` when the merged candidate set
contains multiple messages from the same session. The cross-encoder overfits
to the dominant session and misses answers in other sessions.

**Evidence (LongMemEval multi-session, hybrid backend, K=5):**

| Question | `search` | `search_enhanced` | Cause |
|----------|----------|-------------------|-------|
| 6d550036 | HIT (rank=1) | MISS | Cross-encoder overfits to dominant session from merged sub-queries |
| b5ef892d | HIT (rank=4) | HIT (rank=1) | Cross-encoder helped — but single miss on prior question is 20% recall loss |

The root cause: `expand_queries("How many projects...")` produces 3+ sub-queries.
Each sub-query calls `search()` (backend-deduplicated). After merging, session_A
appears 3+ times in candidates. Cross-encoder sees repeated vocabulary from
session_A and overfits.

## Solution

**Maximal Marginal Relevance (MMR) after scoring, before top-K selection.**

Core idea: iterate through scored results, greedily add the highest-scoring
result from a new session, then add the next-highest from another new session,
until top-K is full. If fewer than K unique sessions exist, fill remaining slots
with the next-highest scored results.

```
Input:  scored_results (sorted by score descending)
Output: top-K results (score-sorted within, session-diverse across)

Algorithm:
1. seen_sessions = set()
2. diverse = []
3. for r in scored_results:
      sid = r.memory.session_id or f"_no_session_{hash(r.memory.content)}"
      if sid not in seen_sessions:
          diverse.append(r)
          seen_sessions.add(sid)
          if len(diverse) >= K:
              return diverse
4. Fill remaining slots from unseen results (if needed)
5. Return diverse[:K]
```

**Session-less messages** get a synthetic key (`_no_session_{content_hash}`) to
prevent infinite dedup.

**Ordering**: results are still score-sorted within the diverse set (highest
scoring session first, then second-highest, etc.). Not shuffled.

## Where it fits

```
User query
  └─ backend.search() → raw results (HybridBackend: already deduplicated)
       └─ heuristics.apply_all() → boosted scores
            └─ [search_enhanced only] multi-query expansion merges N sub-queries
                 └─ MMR diversity → deduplicate merged candidates across sub-queries  ← NEW
                      └─ cross-encoder rerank → reranked top-K
                           └─ return
```

**`search()` is unchanged.** HybridBackend already session-deduplicates. MMR only
applies to `search_enhanced()`, where multi-query expansion can re-introduce
session duplicates — each sub-query's results are individually deduplicated by
the backend, but merging them produces duplicates across sub-queries.

## Branching behavior

**No branching.** MMR in `search_enhanced` is always-on — diverse cross-encoder
candidates help all question types. On single-session queries it's a no-op
(only one unique session in the merged result set).

## Regression prevention

### Pre-merge check: run before/after on all question types

```bash
# Full LongMemEval before MMR
uv run python -m benchmarks.longmemeval.eval \
  --backend hybrid --k 5 --search-mode search_enhanced > results/before.json

# Full LongMemEval after MMR  
uv run python -m benchmarks.longmemeval.eval \
  --backend hybrid --k 5 --search-mode search_enhanced > results/after.json

# Compare: search_enhanced recall should not regress
python -c "
import json
b = json.load(open('results/before.json'))
a = json.load(open('results/after.json'))
for qt in a.get('by_type', {}):
    for m in a['by_type'][qt]:
        br = b['by_type'][qt][m]['recall']
        ar = a['by_type'][qt][m]['recall']
        delta = ar - br
        flag = '⚠️ REGRESSION' if delta < -0.02 else '✅'
        print(f'{qt} ({m}): {br:.3f} → {ar:.3f} ({delta:+.3f}) {flag}')
"
```

Also verify `search()` is unchanged:

### Per-question diff

For each question where recall changes, print the question text, old rank, new
rank. Any regression must have a documented justification.

### New unit tests

```
test_mmr_no_op_when_all_same_session     → 5 results from 1 session, MMR selects all
test_mmr_dedups_across_sessions          → mixed sessions, MMR picks 1 per session
test_mmr_sessionless_gets_unique_keys    → null session_id → unique per content
test_mmr_preserves_score_order           → highest scoring session first
test_mmr_fewer_sessions_than_k           → 2 sessions, K=5, fills remaining
```

### Performance regression

- MMR is O(n) where n is the merged candidate set size (sub-queries × K results each).
  For 3 sub-queries at K=10, n=30 → ~30 dict lookups. Negligible vs 500ms+
  cross-encoder time.
- Benchmark: `test_mmr_performance` — 10K results, MMR to K=10, assert < 1ms.

## Non-goals

- No lambda parameter for MMR (relevance vs diversity tradeoff). Always greedily
  diverse. Can be added later if needed.
- No per-session score aggregation (avg/max of messages in a session). The
  highest-scoring message is the session representative.
- No cross-session MMR for non-session dimensions (user_id, agent_id).
  Session is the only diversity dimension for v1.

## Impact on LongMemEval recall

| Method | Expected change | Reason |
|--------|----------------|--------|
| `search()` | No change | HybridBackend already deduplicates |
| `search_enhanced()` | Possible + on multi-session | Cross-encoder currently sees duplicated sessions; MMR gives it diverse candidates |

## Implementation plan

1. Add `_mmr_diversify(results, k)` to `heuristics.py`
2. Call `_mmr_diversify()` in `MemoryCore.search_enhanced()` after query expansion merge, before cross-encoder rerank
3. `MemoryCore.search()` unchanged — HybridBackend already deduplicates
4. Run LongMemEval diff (all backends, all types — should not regress)
5. If no regression, ship
