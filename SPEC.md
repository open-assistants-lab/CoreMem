# CoreMem Query-Guided Traversal

## Goal

Add `memorycore_traversal`: an iterative retrieval mode that starts with the
existing `memorycore` hybrid search, explores nearby memories, checks every new
candidate against the original query, and preserves useful branches across
multiple sessions.

The goal is not to replace `memorycore` with graph centrality. The goal is to
use graph structure to recover relevant context that one-shot search misses.

## Motivation

Current results on the 20-question LongMemEval subset:

| Mode | session recall@5 | message recall@5 | Query-time LLM |
|---|---:|---:|---:|
| `memorycore` | 0.825 | 0.623 | 0 |
| `memorycore_deep` | 0.895 | 0.702 | 1 |
| `memorycore_context` | 0.816 | 0.281 | 0 |
| PPR traversal PoC | 0.588 | 0.333 | 0 |

The PPR PoC underperformed because:

1. HybridDB's graph search selected vector-only seeds instead of using
   `memorycore`'s hybrid keyword/vector/heuristic search.
2. PPR redistributed activation according to graph topology but did not check
   whether newly reached messages remained relevant to the query.
3. Session hubs created high-degree gravity wells that concentrated results in
   one session.
4. Temporal-only graphs were disconnected across sessions and could not
   discover new episodes.

The original idea was not global PageRank. It was traversal:

```text
cue → initial memories → neighboring memories → relevance check → repeat
```

## Research Model

The design is informed by several compatible memory models:

- **Spreading activation:** a cue activates associated memories.
- **Hippocampal pattern completion:** a partial cue reactivates details from an
  episode.
- **Hippocampal replay:** memories are replayed forward and backward through
  temporal sequences.
- **Retrieval competition and inhibition:** useful recall does not activate the
  entire network uniformly; weak or irrelevant branches are suppressed.

The important correction from the PPR PoC is that expansion must remain
query-guided. Graph proximity is evidence, not relevance by itself.

## Design Principles

1. **Start from the strongest existing retriever.** Seeds come from
   `MemoryCore.search_messages()`, not a separate vector-only search.
2. **Expand locally.** Traverse temporal neighbors first; add other edge types
   only when evidence shows they help.
3. **Re-check relevance every round.** Newly reached messages must be scored
   against the original query before they survive.
4. **Preserve multiple episodes.** One session must not consume the whole beam.
5. **Reward independent agreement.** A message reached through multiple paths
   receives a confirmation boost.
6. **Bound the search.** Use a small beam, hop limit, and round limit.
7. **Return exactly `k` ranked messages.** Context expansion must not inflate
   recall metrics by returning unranked neighbors.

## Stage 1 Scope

Stage 1 uses only temporal adjacency:

| Edge | Definition | Direction |
|---|---|---|
| `temporal_next` | Next message in the same session | Forward |
| `temporal_prev` | Previous message in the same session | Backward |

Session hubs are disabled. Topic/entity edges are deferred until temporal
traversal is understood. Temporal adjacency is derived from message order at
query time; Stage 1 does not materialize graph nodes or edges.

HybridDB remains the message store and hybrid seed retriever. Query-time
traversal is controlled by CoreMem rather than `search_graph_ppr()`.

## Algorithm

### Inputs

```python
search_with_traversal(
    query: str,
    *,
    limit: int = 5,
    seed_limit: int = 20,
    beam_width: int = 20,
    max_rounds: int = 2,
    neighbors_per_direction: int = 1,
    max_per_session: int = 2,
) -> list[SearchResult]
```

### Round 0: Seed

Use existing hybrid retrieval:

```python
seeds = self.search_messages(query, limit=seed_limit)
```

This preserves the behavior that already achieves `message_recall@5=0.623`.

Min-max normalize seed scores across `seed_limit` results:

```text
seed_score = (raw_score - min_score) / (max_score - min_score)
```

If all raw scores are equal, use deterministic reciprocal-rank scores
`1 / rank`. Max-only normalization is not sufficient because HybridDB RRF
scores are often tightly grouped and would leave every top seed near `1.0`,
preventing traversal candidates from entering the final top-k.

Keep the original top `limit` seeds as `baseline_results`. They remain eligible
for the final ranking even if session diversity excludes them from expansion.
This prevents traversal from discarding the strongest one-shot retrievals
before it has found better evidence.

Apply session diversity when selecting the initial beam:

- At most `max_per_session` seed messages from one session.
- Do not refill beyond that cap. A smaller diverse beam is preferable to one
  session consuming the frontier.
- Do not require a minimum number of sessions; the cap limits domination but
  does not invent diversity when only one session is relevant.

### Each Traversal Round

For every message in the current frontier:

1. Load `neighbors_per_direction` previous messages.
2. Load `neighbors_per_direction` next messages.
3. Skip messages expanded in an earlier round, but merge duplicate candidates
   reached by different parents in the current round.
4. Score each candidate against the original query.
5. Combine query relevance, parent activation, distance decay, and path
   convergence.
6. Merge candidates reached through multiple paths.
7. Select the next diverse beam.

### Candidate Scoring

Stage 1 uses a minimal, explicit formula:

```text
candidate_score =
    0.60 × query_relevance
  + 0.30 × parent_score × hop_decay
  + 0.10 × confirmation
```

Where:

- `query_relevance` is a message-level score against the original query.
- `parent_score` is the maximum traversal score among parents that reached it
  in the current round. Parent scores are not summed, which would reward graph
  degree rather than relevance.
- `hop_decay = 0.7 ** hop`.
- `confirmation = min(path_count - 1, 2) / 2`.

These weights are hypotheses, not permanent defaults. They must be evaluated
against the unchanged `memorycore` baseline.

### Query Relevance

Do not use graph topology as query relevance.

For Stage 1, query relevance uses the query-dependent keyword component of the
existing heuristics with a non-zero base:

```python
boosted = SearchHeuristics.keyword_overlap(
    query=query,
    content=message.content,
    score=1.0,
)
query_relevance = max(0.0, boosted - 1.0)
query_relevance = query_relevance / (1.0 + query_relevance)
```

Prune a candidate when `query_relevance == 0` and it was reached through only
one parent. Parent activation alone is not confirmation and must not let an
irrelevant neighbor displace a baseline result. A zero-overlap candidate may
survive only when multiple frontier paths converge on it.

This produces a bounded `[0, 1)` score on a consistent scale across rounds. Do
not min-max normalize each round: a weak round would otherwise assign `1.0` to
its least-bad candidate, making scores from different rounds incomparable.

This is not identical to `memorycore` scoring: `memorycore` starts from a
HybridDB hybrid score and then applies heuristics. Expanded candidates do not
have a scoped HybridDB score. The parent activation supplies the associative
signal. If textual relevance is insufficient, the next experiment is batched
query-to-candidate embedding similarity, not per-candidate database searches.

### Path Convergence

Path convergence means more than one frontier message reaches the same
candidate.

Examples:

- The same candidate is the next neighbor of one seed and the previous
  neighbor of another seed.

Track parent message IDs for each candidate:

```python
parents_by_candidate: dict[str, set[str]]
```

Messages reached through more than one distinct parent in the same expansion
round get a small boost. This is an exploration signal, not semantic
confirmation: two adjacent parents are structurally related and are not fully
independent evidence. Once a candidate has entered a completed frontier it is
visited and is not reconsidered in later rounds. XOR is not required.

### Beam Selection

After scoring a round:

1. Sort candidates by traversal score.
2. Keep at most `max_per_session` candidates per session.
3. Stop if the next beam adds no previously unseen message.
4. Otherwise continue until `max_rounds`.

Maintain `best_candidate_by_id` across all rounds. A candidate remains eligible
for final ranking even after it leaves the active beam; the beam controls
expansion, not retention. If a candidate is encountered again before being
visited, keep its highest score and merge its parents.

The final pool contains `baseline_results` plus every discovered candidate's
best score. It does not contain every raw seed from `seed_limit`. Rank that pool
and return exactly the top `limit` messages. If traversal discovers nothing,
the output is identical to the original top-`limit` `memorycore` results.

## Data Structures

Keep traversal state local to `search_with_traversal()`:

```python
@dataclass
class TraversalCandidate:
    result: SearchResult
    hop: int
    parents: set[str]
```

Traversal also maintains:

```python
visited: set[str]
best_candidate_by_id: dict[str, TraversalCandidate]
session_rows: dict[str, list[dict]]  # query-local cache
```

No new database tables or graph edges are needed. Stage 1 fetches temporal
neighbors directly using:

```sql
SELECT * FROM messages
WHERE session_id = ?
ORDER BY ts ASC, rowid ASC
```

Direct SQL is acceptable here because temporal position is already represented
by insertion order in the messages table and avoids graph-node translation
overhead. `id` cannot be used as a tie-breaker because production message IDs
are UUIDs. LongMemEval also gives every message in a session the same timestamp,
so `rowid` is required to preserve source order in the current schema.

## Implementation Plan

### 1. Replace PPR Query Logic

**File:** `coremem/core.py`

- Remove `_build_message_graph()` calls from the Stage 1 ingest/eval path.
- Do not create session hubs or temporal graph edges.
- Replace the `search_graph_ppr()` call in `search_with_traversal()` with:
  - `search_messages()` seeds
  - iterative temporal expansion
  - per-round query relevance scoring
  - diverse beam selection

### 2. Keep Eval Mode

**File:** `scripts/eval_agent_journal_longmemeval.py`

Keep `memorycore_traversal`. It should return exactly `k` `RawSearchHit`
objects, so existing recall/hit/MRR/MAP metrics remain comparable.

### 3. Tests

Add focused tests:

1. **Seed preservation:** a strong direct match remains in final results.
2. **Temporal discovery:** a relevant adjacent message not present in initial
   top-k can enter the final results.
3. **Query pruning:** an irrelevant temporal neighbor does not displace a
   strong seed.
4. **Session diversity:** one session cannot consume the full beam when equally
   relevant seeds exist elsewhere.
5. **Path confirmation:** a candidate reached through two parents scores above
   an otherwise equal single-path candidate.
6. **Bounded output:** exactly `limit` or fewer results are returned.
7. **Determinism:** repeated searches return the same IDs and scores.

## Evaluation Plan

### Phase A: Behavioral Checks

Use selected failing questions before a full run:

- `e01b8e2f`: adjacent replay should recover the answer-bearing message.
- `f0853d11`: preserve seeds from both required sessions.
- `gpt4_1e4a8aeb`: avoid one session consuming all top results.

### Phase B: 20-Question Subset

```bash
uv run python3 scripts/eval_agent_journal_longmemeval.py \
  data/longmemeval_20_baseline_subset.json \
  --mode memorycore_traversal --k 5 \
  --progress --root eval_output/traversal-guided-20 \
  --output eval_output/traversal-guided-20/results.json --overwrite
```

### Success Criteria

Primary gate:

- `message_recall@5 >= 0.623` (do not regress from `memorycore`)
- `session_recall@5 >= 0.825`

Secondary goals:

- Improve temporal-reasoning message recall above `0.479`.
- Preserve `empty_retrieval_rate <= 0.05`.
- No query-time LLM calls.

If guided traversal cannot match `memorycore`, stop expanding this approach.
Do not add entity/topic graphs or a GNN until Stage 1 proves that query-guided
temporal traversal adds value.

## Deferred Work

Only after Stage 1 passes:

- Entity/topic edges across sessions
- Journal nodes
- PPR as an additional candidate feature
- Learned edge weights
- GNN / graph attention
- Graph integrity fingerprints or XOR checks

## References

- Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of
  semantic processing. *Psychological Review*, 82(6), 407–428.
- Anderson, J. R. (1983). A spreading activation theory of memory. *Journal of
  Verbal Learning and Verbal Behavior*, 22(3), 261–295.
- Carr, M. F., Jadhav, S. P., & Frank, L. M. (2011). Hippocampal replay in the
  awake state. *Nature Neuroscience*, 14(2), 147–153.
- Gilmer, J., et al. (2017). Neural message passing for quantum chemistry.
  *ICML*.
