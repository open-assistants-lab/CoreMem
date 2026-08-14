# CoreMem

> **Zero-LLM memory retrieval for AI agents.** CoreMem gives agents instant access to conversation history — semantic search plus deterministic retrieval heuristics, all without a single API call. The default `recall(strategy="episodic")` path scores **99.9% session recall@5 on LongMemEval Oracle (500 questions)** and **95.0% on LongMemEval S (500 questions, ~48 sessions each)** with zero LLM calls.

> **Embedded. Local. Open source.** No external APIs, no vector DB services, no internet connection required. Runs entirely on-device with HybridDB (SQLite + FTS5 + ChromaDB) + sentence-transformers. Ships as a single Python package with zero infrastructure dependencies.

**Single-backend architecture.** HybridDB (SQLite + FTS5 + ChromaDB) is the only backend since v0.6.0. Retrieval pipeline: FTS5 + vector search → deterministic heuristics → query decomposition → cross-encoder reranking → MMR session diversity → session-deduplicated retrieval.

```python
from coremem import MemoryCore

core = MemoryCore(path="./memory")

# Ingest conversation turns
core.ingest("user", "I visited the Museum of Modern Art today", session_id="conv_001")
core.ingest("assistant", "That sounds wonderful! How was it?", session_id="conv_001")
core.ingest("user", "I went to an Ancient Civilizations exhibition at the Natural History Museum", session_id="conv_001")

# Retrieve with the default episodic strategy (zero LLM)
results = core.recall("When did I visit art museums?")

for r in results:
    print(f"[{r.memory.ts}] [{r.memory.role}] {r.memory.content}")
```

## Why CoreMem?

Every AI agent needs memory. But cloud-based vector search is expensive, slow, and doesn't work offline. Pure embedding similarity misses keyword matches and temporal context. LLM-based memory systems cost tokens per query.

CoreMem solves all three:

| Component | What it does |
|-----------|-------------|
| **HybridDB retrieval** | FTS5 keyword + embedding similarity via a single SQLite-backed store |
| **Deterministic heuristics** | Keyword overlap (exact + fuzzy + bigram), temporal recency, person-name boost, quoted-phrase matching |
| **Query decomposition** | Splits multi-cue relational questions ("before X did I Y") into independent search cues |
| **Cross-encoder reranking** | `ms-marco-MiniLM-L-6-v2` reranks candidates — the single biggest recall win (m@5 0.472 → 0.867 on oracle) |
| **MMR session diversity** | One result per session, preventing cross-encoder overfit |

## LongMemEval Results

### Oracle (500 questions, ~2 sessions each, k=5)

| Metric | `direct` | `expanded` | `episodic` (default) |
|---|---:|---:|---:|
| session_recall@5 | 0.938 | 0.951 | **0.999** |
| message_recall@5 | 0.754 | 0.854 | **0.867** |
| session_hit@5 | 0.972 | 0.972 | **1.000** |
| message_hit@5 | 0.904 | 0.951 | **0.947** |
| context_chars_mean | 4,937 | 3,928 | **4,540** |

### S (500 questions, ~48 sessions each, k=5)

| Metric | `direct` | `episodic` (default) |
|---|---:|---:|
| session_recall@5 | 0.865 | **0.950** |
| message_recall@5 | **0.670** | 0.617 |
| session_hit@5 | 0.968 | **0.981** |
| message_hit@5 | **0.768** | **0.768** |
| context_chars_mean | — | **3,991** |

**Recommendation: use `recall(strategy="episodic")` (the default).** It is the strongest zero-LLM mode on both evaluations — best session recall, competitive message recall, and no retrieval LLM calls. Use `direct` for single-session factual questions (best message precision), `expanded` when highest precision is needed (1 LLM call for query rephrasing), and `fusion` when session diversity is critical (2× compute).

All modes abstain correctly on unanswerable questions (0% false positive rate).

Results: `eval_output/lme-oracle/results.json`, `eval_output/lme-s/results.json`

## Installation

```bash
pip install coremem
```

Optional extras:

```bash
pip install "coremem[mcp]"    # MCP server
pip install "coremem[all]"     # all extras
```

> **Note on model downloads.** ChromaDB downloads a bundled MiniLM embedding model (~80MB) on first `PersistentClient()` init. The cross-encoder downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` (~500MB) on first `recall(strategy="episodic")` call. Both cache locally after download. Run one recall at startup to pre-load models predictably.

## Core Concepts

### Ingestion

```python
# Simple ingestion
core.ingest("user", "I built a Spitfire model kit", session_id="conv_001")

# Batch ingestion (one turn = one turn_id)
core.ingest_turn([
    {"role": "user", "content": "What's the weather today?"},
    {"role": "assistant", "content": "Sunny with a high of 72°F"},
], session_id="conv_001")
```

### Recall

`recall()` is the single retrieval entry point, with four strategies:

| Strategy | LLM calls | Pipeline |
|----------|-----------|----------|
| `episodic` (default) | 0 | Query decomposition → hybrid search per variant → RRF fusion → cross-encoder rerank → MMR diversity |
| `direct` | 0 | Single hybrid search + deterministic heuristics |
| `expanded` | 1 | LLM query rephrasing, then the direct pipeline per variant |
| `fusion` | 0 | RRF fusion of `direct` + `episodic` |

```python
results = core.recall("How many model kits?", limit=10)
results = core.recall("What did I build recently?", strategy="direct")

# Session bundles — surrounding context around each hit
bundles = core.recall("model kits", bundles=True)
for b in bundles:
    print(f"## Session {b.session_id} (complete={b.complete})")
    for m in b.messages:
        print(f"  [{m.role}] {m.content}")

# Filter params
results = core.recall("coffee", role="user", session_id="conv_001", ts_after="2024-01-01")
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

### Memory lifecycle

```python
core.fetch(session_id="conv_001")          # query with filters
core.fetch_all()                            # everything (limit 10k)
core.store([Memory(id="m1", content="...")])
core.count()
core.delete(session_id="conv_001")
core.clear()
```

### AgentJournal

The AgentJournal subsystem compiles conversation turns into dense, retrieval-optimized daily journal pages (markdown + frontmatter), with deterministic validation of every claim against its source:

```python
# Compile a turn into daily/YYYY-MM-DD.md (1 LLM call per turn)
await core.compile_turn(turn_id=tid)
await core.compile_latest_turn(session_id="conv_001")
await core.compile_uncompiled_turns()

# Dreaming consolidation — LLM analysis of daily pages, promoted facts to MEMORY.md
await core.dream()

# Rebuild weekly/monthly/index navigation files from daily pages
core.rebuild_index()
```

The LLM compiler (`openai:gpt-4o-mini` by default) produces a structured plan that the deterministic compiler validates — every claim is checked against source messages (exact quote substrings, role/evidence-type compatibility) before it is written. Set `COREMEM_LLM_MODEL` (e.g. `ollama:llama3.2`) to change the model.

### CLI, MCP, and hooks

```bash
coremem recall "model kits" --strategy direct
coremem ingest user "I built a Spitfire model kit" --session-id conv_001
coremem compile <turn_id>
coremem rebuild
coremem sessions
coremem mcp   # MCP stdio server (also the default command)
```

- **MCP server** — 5 tools: `recall`, `ingest`, `compile`, `rebuild_index`, `list_sessions`
- **Hooks** — Claude Code and Codex: `UserPromptSubmit` (capture + retrieval injection), `Stop` (capture), `PreCompact` (no-op)
- **Integration configs** in `integrations/` for Claude Code, Codex, and OpenCode

### Environment variables

| Variable | Purpose |
|----------|---------|
| `COREMEM_PATH` | Memory storage path (default `~/.coremem/hybrid`) |
| `COREMEM_LLM_MODEL` | LLM model for journal compilation (e.g. `openai:gpt-4o-mini`, `ollama:llama3.2`) |
| `DISABLE_CROSS_ENCODER` | Set to `1` to skip cross-encoder reranking (eval scripts) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OLLAMA_API_KEY` | Provider keys for LLM-backed features |

## License

MIT — see [LICENSE](LICENSE).

## Author

Eddy Xu

CoreMem is the retrieval engine behind the [Executive Assistant](https://github.com/open-assistants-lab) agent system. Pairs with [HybridDB](https://github.com/open-assistants-lab) for storage and ConnectKit for real-time sync.
