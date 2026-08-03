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

## Eval Results — LongMemEval S (22 questions, k=5, ~48 sessions each)

**Dataset**: `data/longmemeval_s_20_subset.json` — diverse question types

| Metric | `memorycore` | `episodic_reranked` | `episodic_reranked_4k` | `fusion` |
|---|---:|---:|---:|---:|
| session_recall@5 | 0.806 | **0.905** | **0.905** | **0.940** |
| message_recall@5 | **0.591** | 0.560 | 0.560 | 0.548 |
| session_hit@5 | 0.905 | 0.905 | 0.905 | **0.952** |
| message_hit@5 | **0.714** | 0.619 | 0.619 | 0.619 |
| session_mrr | 0.821 | **0.905** | **0.905** | 0.849 |
| session_map | 0.736 | **0.905** | **0.905** | 0.798 |
| context_chars_mean | 6,338 | 4,485 | 4,485 | **5,203** |

**By question type — message_recall@5:**

| Type | n | `memorycore` | `episodic_reranked` | `fusion` |
|---|---:|---:|---:|---:|
| knowledge-update | 4 | 0.625 | 0.500 | **0.750** |
| multi-session | 4 | 0.479 | **0.938** | 0.625 |
| single-session-assistant | 3 | **1.000** | 0.333 | 0.333 |
| single-session-preference | 3 | **0.333** | 0.000 | 0.000 |
| single-session-user | 4 | 0.333 | **0.667** | **0.667** |
| temporal-reasoning | 4 | **0.750** | **0.750** | **0.750** |

## Recommendation

**Use `episodic_reranked` as the default retrieval strategy.**

It is the strongest zero-LLM-retrieval mode across both oracle and S evaluations:

- **Best session recall** (0.905 S, 0.999 oracle) — finds the right context.
- **Competitive message recall** (0.560 S, 0.867 oracle) — close to `memorycore` on S, exceeds it on oracle.
- **Bundle evidence recall** (0.762 S, 0.917 oracle) — evidence is available even when not in top-5.
- **No retrieval LLM calls** — uses a local cross-encoder (`ms-marco-MiniLM-L-6-v2`).
- **Context is efficient** (4,485 chars S, 4,540 chars oracle).

**When to use other strategies:**

| Strategy | When | Why |
|---|---|---|
| `memorycore` | Single-session direct questions | Best message precision for simple facts |
| `llm_expansion` | Highest precision needed | 1 LLM call, best message recall (0.854 oracle) |
| `fusion` | When session diversity is critical | Best session hit rate (0.952 S) but 2× compute |

**What was falsified:**
- Graph traversal (PPR, beam search) — below baseline on every metric.
- Session-hub nodes — created gravity wells, removed.
- Deterministic classifier routing — 78% accuracy on S, not reliable enough for production.
- Session reranking (ER sessions → MC messages) — session recall collapsed to 0.623.

**What was confirmed:**
- Query decomposition improves temporal session recall (0.604 → 0.750).
- Episodic reconstruction improves answer accuracy (temporal 22.2% → 66.7%).
- Cross-encoder reranking is the critical component (m@5 0.472 → 0.867 oracle).
- 4k context budget retains evidence recall while cutting bundle context by 67%.

## Public API

```python
from coremem import MemoryCore, SessionBundle, decompose_queries

# Direct retrieval (zero LLM)
results = core.search_messages(query, limit=5)

# LLM query expansion (1 LLM call)
results = core.search_messages_llm_expansion(query, limit=5)

# Episodic reconstruction (zero LLM, local cross-encoder)
primary = core.search_messages_decomposed(query, limit=5, use_cross_encoder=True)
bundles = core.reconstruct_sessions(query, primary_results=primary, max_context_chars=4_000)

# Unified entry point
results = core.recall(query, strategy="auto")
```

## Key Design Decisions

- **No verbatim compiler** — removed; only LLM compiler for daily journals
- **Daily pages use hybriddb timestamps** — `daily/{actual_date}.md`, not `datetime.now(UTC)`
- **`DEFAULT_AGENT_JOURNAL_MODEL`** = `"openai:gpt-4o-mini"` (ollama-cloud not in library default)
- **`--journal-llm-model` required** for `memorycore_journal` mode
- **Per-question haystack** — canonical LongMemEval setup
- **Shared `CrossEncoderReranker`** across per-question cores (avoids reloading model)
- **Resume/checkpoint** via sidecar `{output}.checkpoint.json`

## Eval CLI

```bash
# All modes (20-question subset)
uv run scripts/eval_agent_journal_longmemeval.py data/longmemeval_20_baseline_subset.json \
  --mode all --k 5 --journal-llm-model ollama-cloud:deepseek-v4-flash \
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
uv run python3 -m pytest tests/ -q   # 47 pass
```