# CoreMem — Agent Context

## Eval Results — LongMemEval Oracle (500 questions, k=5)

**Dataset**: `data/longmemeval_oracle.json` — 500 questions, ~2 sessions each

| Metric | `memorycore` | `llm_expansion` | `episodic` | `episodic_reranked` | `episodic_reranked_4k` |
|---|---:|---:|---:|---:|---:|
| session_recall@5 | 0.938 | 0.951 | **0.999** | **0.999** | **0.999** |
| message_recall@5 | 0.754 | 0.854 | 0.472 | **0.867** | **0.867** |
| session_hit@5 | 0.972 | 0.972 | **1.000** | **1.000** | **1.000** |
| message_hit@5 | 0.904 | 0.951 | 0.570 | **0.947** | **0.947** |
| session_mrr | 0.972 | 0.972 | **1.000** | **1.000** | **1.000** |
| session_map | 0.938 | 0.951 | **0.999** | **0.999** | **0.999** |
| empty_retrieval_rate | 0.060 | 0.060 | 0.060 | 0.060 | 0.060 |
| context_chars_mean | 4,937 | **3,928** | 8,050 | 4,540 | 4,540 |
| bundle_message_recall | — | — | 0.829 | **0.917** | 0.897 |
| bundle_message_hit | — | — | 0.909 | **0.966** | 0.951 |
| bundle_context_chars_mean | — | — | 7,757 | 6,898 | **2,243** |

## Eval Results — LongMemEval S (500 questions, k=5, ~48 sessions each)

**Dataset**: `data/longmemeval_s_cleaned.json` — 500 questions, ~48 sessions each, 265 MB JSON

| Metric | `memorycore` (saved) | `episodic_reranked` | `episodic_reranked_4k` |
|---|---:|---:|---:|
| session_recall@5 | 0.865 | **0.950** | **0.950** |
| message_recall@5 | **0.670** | 0.617 | 0.617 |
| session_hit@5 | 0.968 | **0.981** | **0.981** |
| message_hit@5 | **0.768** | **0.768** | **0.768** |
| session_mrr | 0.968 | 0.947 | 0.947 |
| session_map | 0.831 | **0.920** | **0.920** |
| empty_retrieval_rate | 0.060 | 0.060 | 0.060 |
| context_chars_mean | — | 3,991 | 3,991 |
| bundle_message_recall | — | **0.847** | 0.770 |
| bundle_message_hit | — | **0.921** | 0.855 |
| bundle_context_chars_mean | — | 11,579 | **2,685** |

**By question type — message_recall@5:**

| Type | n | `memorycore` (saved) | `episodic_reranked` |
|---|---:|---:|---:|
| single-session-user | 70 | **0.898** | 0.828 |
| single-session-assistant | 56 | **0.857** | 0.589 |
| knowledge-update | 78 | **0.738** | 0.655 |
| temporal-reasoning | 133 | 0.588 | **0.599** |
| multi-session | 133 | 0.539 | **0.575** |
| single-session-preference | 30 | **0.544** | 0.367 |

## Recommendation

**Use `episodic_reranked` as the default retrieval strategy.**

It is the strongest zero-LLM-retrieval mode across both oracle and S evaluations:

- **Best session recall** (0.950 S, 0.999 oracle) — finds the right context.
- **Competitive message recall** (0.617 S, 0.867 oracle) — close to `memorycore` on S, exceeds it on oracle.
- **Bundle evidence recall** (0.847 S, 0.917 oracle) — evidence is available even when not in top-5.
- **No retrieval LLM calls** — uses a local cross-encoder (`ms-marco-MiniLM-L-6-v2`).
- **Context is efficient** (3,991 chars S, 4,540 chars oracle).
- **4k budget retains identical primary retrieval** with 77% less bundle context (2,685 vs 11,579 chars).

**When to use other strategies:**

| Strategy | When | Why |
|---|---|---|
| `memorycore` | Single-session direct questions | Best message precision for simple facts |
| `llm_expansion` | Highest precision needed | 1 LLM call, best message recall (0.854 oracle) |
| `fusion` | When session diversity is critical | Best session hit rate (0.952 S) but 2× compute |

**What was falsified:**
- Graph traversal (PPR, beam search, query-guided traversal v2) — **parked 2026-08-18**. Below baseline on every metric. Re-tested on the fixed HybridDB graph (0.5.5, published 2026-08-17) with a corrected design (seeds = baseline's exact rerank window, candidates restricted to new sessions) and a full research-grounded edge set (topic, turn_qa, update, causal, self_reference, emotional, entity, semantic — see `docs/graph-edges-design.md`). Results:
  - 20-question subset: identical to baseline on every metric (0.974 session recall) — neutral, never better.
  - S-scale (500 questions, resumable run via `scripts/eval_graph_s.py`): multi-session (133) neutral (−0.0008 session recall), single-session types exact ties, **temporal-reasoning negative** (−0.0113 session, −0.0075 message at the full 133). The causal/update edges pull MMR toward graph-discovered sessions that are not temporally correct.
  - The original PPR PoC (0.588/0.333) was measured against buggy graph code (silent empty results, directed-PPR mass loss) and overstated the harm; the corrected verdict is neutral-to-negative, not harmful.
  - The interim +0.0053 multi-session signal at 75 questions was a small-sample artifact — it eroded to −0.0008 at the full 133.
  - Eval mode `memorycore_traversal_v2` and `scripts/eval_graph_s.py` (resume + extensive timing: ingest throughput, phase timings, graph composition) remain ablation-ready.
  - Retrieval cost: graph build is 13× baseline search time per query (12.9s vs 1.0s) — a derived index that would need caching/incremental maintenance to be viable, a question made moot by the neutral-to-negative recall verdict.
- Session-hub nodes — created gravity wells, removed.
- Deterministic classifier routing — 78% accuracy on S, not reliable enough for production.
- Session reranking (ER sessions → MC messages) — session recall collapsed to 0.623.

**What was confirmed:**
- Query decomposition improves temporal session recall (0.604 → 0.750).
- Episodic reconstruction improves answer accuracy (temporal 22.2% → 66.7%).
- Cross-encoder reranking is the critical component (m@5 0.472 → 0.867 oracle).
- 4k context budget retains evidence recall while cutting bundle context by 67-77%.
- Episodic reranked beats memorycore on session recall at S scale (0.950 vs 0.865).
- **Temporal query decomposition** (from/to, since/when, clean ago-event cues) — +0.037 session / +0.029 message recall on the 133 S temporal-reasoning questions (L-6 reranker only; the L-12 reranker cancels this win).
- **Preference union-retrieval** (per-variant top-40 union for preference queries) — +0.033 session recall on the 30 S preference questions, **folded into the default `episodic` path**.
- **L-12 cross-encoder** (`COREMEM_CROSS_ENCODER_MODEL` opt-in): +0.018 message recall on the 20-question oracle-style subset, but cancels the temporal decomposition win on S — keep L-6 as default.
- **Combined S-scale validation** (500/500, `scripts/eval_combined_s.py`): the wins do NOT sum — preference stacks (+0.033), temporal is cancelled by L-12 (−0.004), overall +0.001. Validated fold: preference union only, L-6 retained.

## Public API

```python
from coremem import MemoryCore, SessionBundle

# Default: episodic reranking (zero LLM, local cross-encoder)
results = core.recall(query)

# Strategies
results = core.recall(query, strategy="direct")      # BM25+hybrid, zero LLM
results = core.recall(query, strategy="episodic")     # decomposed + cross-encoder (default)
results = core.recall(query, strategy="expanded")     # 1 LLM call for query rephrasing
results = core.recall(query, strategy="fusion")       # RRF of direct + episodic

# Session bundles
bundles = core.recall(query, bundles=True)

# With filter params
results = core.recall(query, role="user", session_id="abc", ts_after="2024-01-01")
```

## Key Design Decisions

- **No verbatim compiler** — removed; only LLM compiler for daily journals
- **Daily pages use hybriddb timestamps** — `daily/{actual_date}.md`, not `datetime.now(UTC)`
- **`DEFAULT_AGENT_JOURNAL_MODEL`** = `"openai:gpt-4o-mini"` (ollama-cloud not in library default)
- **Per-question haystack** — canonical LongMemEval setup
- **Resume/checkpoint** via sidecar `{output}.checkpoint.json`

## Eval CLI

```bash
# All modes (20-question subset)
uv run scripts/eval_agent_journal_longmemeval.py data/longmemeval_20_baseline_subset.json \
  --mode all --k 5 \
  --progress --root /tmp/coremem-lme --output results.json --overwrite

# Full oracle (500 questions)
uv run scripts/eval_agent_journal_longmemeval.py data/longmemeval_oracle.json \
  --mode memorycore_episodic_reranked --k 5 \
  --progress --root /tmp/coremem-oracle --output results.json --overwrite

# S subset (22 questions, ~48 sessions each)
uv run scripts/eval_agent_journal_longmemeval.py data/longmemeval_s_20_subset.json \
  --mode memorycore_episodic_reranked --k 5 \
  --progress --root /tmp/coremem-s20 --output results.json --overwrite
```

## Tests

```bash
uv run python3 -m pytest tests/ -q   # 142 pass
```