# CoreMem

> **Zero-LLM memory retrieval for AI agents.** CoreMem gives agents instant access to conversation history — semantic search plus deterministic retrieval heuristics, all without a single API call. Scores **95.1% R@5 on LongMemEval Oracle (500 questions)** with `recall(strategy="expanded")`, **93.8%** with the zero-LLM `recall(strategy="direct")` path.

> **Embedded. Local. Open source.** No external APIs, no vector DB services, no internet connection required. Runs entirely on-device with ChromaDB or HybridDB + sentence-transformers. Ships as a single Python package with zero infrastructure dependencies.

**Single-backend architecture.** HybridDB (SQLite + FTS5 + ChromaDB) is the only backend since v0.6.0. Ranking pipeline: FTS5 + vector retrieval → deterministic heuristics → MMR session diversity → recency-aware rescoring → session-deduplicated retrieval.

```python
from coremem import MemoryCore

core = MemoryCore(path="./memory")

# Ingest conversation turns
core.ingest("user", "I visited the Museum of Modern Art today")
core.ingest("assistant", "That sounds wonderful! How was it?")
core.ingest("user", "I went to an Ancient Civilizations exhibition at the Natural History Museum")

# Search with deterministic heuristic reranking
results = core.search("When did I visit art museums?")

for r in results:
    print(f"[{r.memory.ts}] [{r.memory.role}] {r.memory.content}")
```

## Why CoreMem?

Every AI agent needs memory. But cloud-based vector search is expensive, slow, and doesn't work offline. Pure embedding similarity misses keyword matches and temporal context. LLM-based memory systems cost tokens per query.

CoreMem solves all three:

| Component | What it does |
|-----------|-------------|
| **Semantic search** | Embedding similarity via ChromaDB or HybridDB |
| **Deterministic heuristics** | Keyword overlap (fuzzy + bigram), temporal recency, person-name boost, quoted-phrase matching |
| **MMR session diversity** | One result per session, preventing cross-encoder overfit |
| **Score normalization** | Per-sub-query normalization in enhanced search for balanced merging |

## LongMemEval Oracle Results (500 questions, ~2 sessions each, k=5)

| Mode | LLM calls/q | session_recall@5 | message_recall@5 | empty_retrieval |
|------|-------------|-------------------|-------------------|------------------|
| `recall(strategy="direct")` | 0 | 93.8% | 75.4% | 6.0% |
| `recall(strategy="expanded")` | 1 | **95.1%** | **85.4%** | **6.0%** |

`recall(strategy="expanded")` is the best overall — highest on every message-level metric, uses 20% less context. `recall(strategy="direct")` (zero-LLM) achieves 93.8% session recall with no API calls.

All three modes abstain correctly on unanswerable questions (0% false positive rate).

Results: `eval_output/lme-oracle/results.json`

## LongMemEval S Results (500 questions, ~48 sessions each, k=5, memorycore only)

| Mode | LLM calls/q | session_recall@5 | message_recall@5 | empty_retrieval |
|------|-------------|-------------------|-------------------|------------------|
| `recall(strategy="direct")` | 0 | **86.5%** | 67.0% | 6.0% |
| `recall(strategy="expanded")` | 1 | *not yet run* | *not yet run* | — |

Zero-LLM `recall(strategy="direct")` holds at 86.5% session recall even with ~48 sessions to search through (vs ~2 in oracle). Near-perfect on single-session types (0.97–1.0), harder on multi-session and temporal-reasoning (0.78–0.80).

| Question type | session_recall@5 | message_recall@5 | n |
|---------------|-------------------|-------------------|---|
| single-session-assistant | **100.0%** | 85.7% | 56 |
| single-session-user | 96.9% | 89.8% | 70 |
| knowledge-update | 93.1% | 73.8% | 78 |
| single-session-preference | 86.7% | 54.4% | 30 |
| temporal-reasoning | 79.6% | 58.8% | 133 |
| multi-session | 77.9% | 53.9% | 133 |

Results: `eval_output/lme-s/results.json`, `eval_output/lme-s/results.jsonl`

## Installation

```bash
pip install coremem
```

HybridBackend (HybridDB — SQLite + FTS5 + ChromaDB) is the default since 0.5.0:

> **Note on model downloads.** ChromaDB downloads a bundled MiniLM embedding model (~80MB) on first `PersistentClient()` init. The cross-encoder downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` (~500MB) on first `search_enhanced()` call. Both cache locally after download. Call `core.warmup()` at startup to pre-load models predictably.

## Core Concepts

### Backend

Since v0.6.0, CoreMem uses HybridDB internally (SQLite + FTS5 + ChromaDB). No backend selection needed:

```python
core = MemoryCore(path="./data")
```

### Ingestion

```python
# Simple ingestion
core.ingest("user", "I built a Spitfire model kit", session_id="conv_001")

# Batch ingestion
core.ingest_many([
    {"role": "user", "content": "What's the weather today?"},
    {"role": "assistant", "content": "Sunny with a high of 72°F"},
], session_id="conv_001")
```

### Search

```python
# Basic search — fast path with deterministic heuristics
results = core.search("How many model kits?", limit=10)

# Enhanced search — multi-query expansion + cross-encoder reranking
results = core.search_enhanced("What did I build recently?", limit=10)

# Cross-encoder loads on first use (~500MB download).
# Disable with DISABLE_CROSS_ENCODER=1 for eval scripts.
```

### Heuristics

Deterministic, zero-LLM scoring boosts applied to every result:

| Heuristic | What it catches |
|-----------|----------------|
| `keyword_overlap` | Exact + fuzzy (difflib) + bigram matches between query and content |
| `temporal_boost` | Queries with "latest", "current", "recently" |
| `recency_decay` | Unconditional exponential decay (30-day half-life) |
| `person_name_boost` | Proper name mentions in content |
| `quoted_phrase_boost` | Exact phrase matches in quotes |

```python
from coremem import SearchHeuristics

# Apply all heuristics to a single result
score = SearchHeuristics.apply_all(
    query="latest project",
    content="Just finished the Q3 project report",
    score=0.75,
    ts="2026-05-28T10:00:00Z",
)
```

### Enhanced Search

`search_enhanced()` adds multi-query expansion and cross-encoder reranking:

```python
results = core.search_enhanced("model kits", limit=10)
```

**Multi-query expansion.** Generates search variants for better recall. Regex expansion always active. LLM-based expansion is opt-in — pass an `llm_provider` to `MemoryCore`:

```python
core = MemoryCore(backend=..., llm_provider=my_chat_model)
```

Or set `MEMORY_EXPANSION_MODEL=ollama:llama3.2` in your environment when using MemoryCore from this project's ecosystem.

**Cross-encoder reranking.** A `cross-encoder/ms-marco-MiniLM-L-6-v2` model reranks the top results for better relevance. Loads lazily on first `search_enhanced()` call (~500MB download). Pre-load at startup with `core.warmup()` to avoid the delay during first search. Disable with:

```bash
DISABLE_CROSS_ENCODER=1 python my_script.py
```

### Observer Pipeline

The `ObserverPipeline` (v0.5.0+) extracts structured observations from conversations — identity facts, events, preferences, plans, stances — and stores them with **source-quote alignment** guaranteeing 0% hallucination:

```python
from coremem.memory_store import MemoryStore
from coremem.observer import ObserverPipeline

store = MemoryStore(path="./memory")
pipeline = ObserverPipeline(
    core=core, store=store, session_id="main",
    token_threshold=100, min_turns=1,
    enable_classification=True,
    enable_dedup=True,
)
await pipeline.after_turn()
```

**7 labeling functions (LF) in parallel** extract entities, actions, preferences, temporal facts, sentiment, possessions, and stances. All LFs are LLM-based — a deliberate choice:

| Approach | Cost | Languages | Recall | Hallucination gate |
|----------|------|-----------|--------|-------------------|
| **LLM LFs** (current) | ~7 API calls/turn | **Any language** | 97.5% | ✅ Source-quote verified |
| Non-LLM (spaCy/VADER) | ~free | English only | ~95% (unverified) | ❌ None |

Non-LLM approaches like OpenIE dependency parsing can replace entities, temporal, possessions, and actions LFs with zero API cost, but are restricted to the languages the NLP model supports (primarily English). LLM LFs handle any language out of the box — Mandarin, Arabic, Spanish, code-switching — without model swaps or quality degradation. The 2.5% miss rate (third-party events, contextual asides) is the measured cost of the hallucination gate.

## License

MIT — see [LICENSE](LICENSE).

## Author

Eddy Xu

CoreMem is the retrieval engine behind the [Executive Assistant](https://github.com/open-assistants-lab) agent system. Pairs with [HybridDB](https://github.com/open-assistants-lab) for storage and ConnectKit for real-time sync.
