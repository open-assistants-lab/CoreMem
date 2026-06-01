# CoreMem

> **Zero-LLM memory retrieval for AI agents.** CoreMem gives agents instant access to conversation history — semantic search plus deterministic retrieval heuristics, all without a single API call. Scores **93.0% R@5 on LongMemEval (500 questions)** with `search_enhanced`, **75.4%** with the zero-LLM `search()` path.

> **Embedded. Local. Open source.** No external APIs, no vector DB services, no internet connection required. Runs entirely on-device with ChromaDB or HybridDB + sentence-transformers. Ships as a single Python package with zero infrastructure dependencies.

**Dual-backend architecture.** Drop-in backends (ChromaDB baseline, HybridDB enhanced) with the same API. Ranking pipeline: backend retrieval → deterministic heuristics → MMR session diversity → recency-aware rescoring → session-deduplicated retrieval.

```python
from coremem import MemoryCore
from coremem.backends.chroma import ChromaBackend

core = MemoryCore(backend=ChromaBackend(path="./memory"))

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

## LongMemEval Results (500 questions, zero LLM tuning)

| Mode | R@5 | MRR | Rank@1 |
|------|-----|-----|--------|
| `search()` | 75.4% | 0.562 | 45.8% |
| `search_enhanced()` | **93.0%** | 0.892 | 86.6% |

| Question type | `search` | `search_enhanced` |
|---------------|----------|-------------------|
| multi-session | 76.7% | **96.2%** |
| knowledge-update | 82.1% | **97.4%** |
| single-session-user | 72.9% | **94.3%** |
| temporal-reasoning | 74.4% | **91.7%** |
| single-session-assistant | 76.8% | **89.3%** |
| single-session-preference | 60.0% | **76.7%** |

## Installation

```bash
pip install coremem
```

With HybridDB backend for enhanced FTS5 + vector hybrid search:

```bash
pip install coremem[hybrid]
```

> **Note on model downloads.** ChromaDB downloads a bundled MiniLM embedding model (~80MB) on first `PersistentClient()` init. The cross-encoder downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` (~500MB) on first `search_enhanced()` call. Both cache locally after download. Call `core.warmup()` at startup to pre-load models predictably.

## Core Concepts

### Backends

```python
# ChromaDB baseline — pure vector search
from coremem.backends.chroma import ChromaBackend
core = MemoryCore(backend=ChromaBackend(path="./data"))

# HybridDB enhanced — FTS5 + vector hybrid search
from coremem.backends.hybrid import HybridBackend
core = MemoryCore(backend=HybridBackend(path="./data"))
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

### Wake-Up Context

Give the agent instant situational awareness:

```python
context = core.wake_up(user_id="alice")
# Returns a compact string with L0 identity and L1 recent context.
```

## License

MIT — see [LICENSE](LICENSE).

## Author

Eddy Vinck

CoreMem is the retrieval engine behind the [Executive Assistant](https://github.com/open-assistants-lab) agent system. Pairs with [HybridDB](https://github.com/open-assistants-lab) for storage and ConnectKit for real-time sync.
