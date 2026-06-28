# MemoryPack MVP: Production Path

2026-06-25

## 1. Goal

Combine the best of both branches into a production-ready MVP:
- **Main branch**: HybridDB storage, query expansion, heuristics
- **Working tree**: MemoryPack deterministic compiler, LLM compiler, BM25 + stemming + fuzzy + cross-encoder, structured pages

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CLI (coremem)                      │
│  compile, search, lint, rebuild-embeddings, serve     │
├─────────────────────────────────────────────────────┤
│                  MemoryCore (unified API)              │
│  search(), search_enhanced(), compile(), lint()        │
├─────────────────────────────────────────────────────┤
│  MemoryPack (deterministic)  │  Search (retrieval)    │
│  ────────────────────────────│────────────────────────│
│  • Bundle (MD files)         │  • BM25 + stem + fuzzy │
│  • Compiler (plan validation) │  • Cross-encoder       │
│  • LLM compiler (prompt)     │  • Query expansion     │
│  • Lint (exact quotes, role) │  • Heuristics (recency)│
│  • Manifest (integrity)      │  • Session aggregation │
├─────────────────────────────────────────────────────┤
│                  HybridDB (storage)                   │
│  • Compiled pages (structured)                        │
│  • Reference turns (raw messages)                     │
│  • Embeddings (numpy array, not HybridDB column)      │
│  • FTS5 index for BM25                                │
└─────────────────────────────────────────────────────┘
```

## 3. Storage: HybridDB + MD Files

HybridDB replaces flat files as the primary storage backend. MD files remain as a human-readable view.

### Tables

```sql
-- Compiled MemoryPack pages (structured memory)
CREATE TABLE pages (
    page_id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    memory_kind TEXT,
    scope TEXT,
    status TEXT,
    activation TEXT,
    trust TEXT,
    safe_to_act INTEGER,
    boot_worthy INTEGER,
    summary TEXT,
    claims TEXT,       -- JSON array of claim objects
    citations TEXT,    -- JSON array of citation objects
    details TEXT,      -- JSON array
    open_questions TEXT,
    read_next TEXT,
    read_when TEXT,    -- JSON array
    agent_memory_version TEXT,
    created_at TEXT,
    updated_at TEXT,
    content TEXT,      -- Full markdown rendering (for search)
    embedding TEXT     -- JSON float array (384-dim)
);

-- Reference turns (raw messages, immutable)
CREATE TABLE turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT,
    messages TEXT,     -- JSON array of message objects
    sha256 TEXT,
    created_at TEXT
);

-- FTS5 virtual table for BM25 search
CREATE VIRTUAL TABLE pages_fts USING fts5(
    page_id UNINDEXED,
    title, description, summary, claims, content,
    tokenize='porter unicode61'
);
```

### MD File Sync

On every `compile` or `update`, the MD file is written alongside the DB entry. This keeps the human-readable view in sync without requiring a separate export step. The MD file is the canonical rendering; the DB is the canonical data.

## 4. Compilation Pipeline

```
Raw messages → LLM compiler → plan validation → HybridDB insert → MD file write → lint
```

### Changes from POC

- `AgentMemoryCompiler.apply_plan()` writes to HybridDB instead of flat files
- `AgentMemoryBundle` becomes a thin wrapper around HybridDB + MD file sync
- `AgentMemoryBundle.lint()` checks both DB integrity and MD file consistency
- `AgentMemoryBundle.rebuild_embeddings()` updates the embedding column in HybridDB

### LLM Compiler (unchanged from POC)

- System prompt with schema + few-shot examples + rules
- Retry loop with quote-fixing post-processor
- Caching by session content hash
- `--resume` flag for partial evals

## 5. Search Pipeline

```
Query → query expansion (LLM, optional) → BM25 FTS5 (top 20) → cross-encoder (re-rank to top 5) → heuristics (recency, diversification) → session aggregation
```

### Changes from POC

- BM25 uses HybridDB's FTS5 instead of custom `_bm25()` function
- Embedding search uses HybridDB's native embedding column (if populated)
- Heuristics from main branch: recency boost, counting question depth, temporal cues
- Session aggregation: group results by `session_id`, return top-k sessions
- Real abstention: if top score < threshold, return empty

### Search Modes

| Mode | Description | Latency |
|---|---|---|
| `fast` | BM25 FTS5 only | ~5ms |
| `balanced` | BM25 + cross-encoder | ~400ms |
| `deep` | Query expansion + BM25 + cross-encoder | ~2.5s |

## 6. CLI Tool

```bash
# Compile a session into a MemoryPack page
coremem compile --turn turn_id --session session_id --messages messages.json

# Search compiled pages
coremem search "how many charity events did I participate in?" --mode balanced

# Lint the bundle
coremem lint

# Rebuild embeddings
coremem rebuild-embeddings

# List pages
coremem list --kind project_fact --scope global

# Show a page
coremem show page_id

# Serve a simple HTTP API (optional, for agent integration)
coremem serve --port 8080
```

## 7. Files Changed

### New
- `coremem/cli.py` — CLI entry point (click or argparse)
- `coremem/memorypack/hybriddb.py` — HybridDB adapter for MemoryPack
- `coremem/memorypack/sync.py` — MD file ↔ HybridDB sync

### Modified
- `coremem/memorypack/bundle.py` — `AgentMemoryBundle` wraps HybridDB
- `coremem/memorypack/compiler.py` — `apply_plan()` writes to HybridDB
- `coremem/memorypack/reranker.py` — unchanged
- `coremem/memorypack/embeddings.py` — `EmbeddingIndex` writes to HybridDB column
- `coremem/memorypack/llm_compiler.py` — unchanged
- `coremem/core.py` — `MemoryCore` gets `compile()` method
- `pyproject.toml` — add `click` dependency

## 8. Dependencies

- `hybriddb>=0.4.5` (already in main branch)
- `click>=8.0` (CLI framework, lightweight, no deps)
- `sentence-transformers>=3.0.0` (already in working tree)

## 9. Verification

Run Stage 4 eval on 8-instance set with HybridDB backend. Compare:

| Metric | POC (flat files) | MVP (HybridDB) |
|---|---|---|
| session_recall@5 | 0.833 | ≥0.833 |
| session_precision@5 | 0.300 | ≥0.300 |
| session_mrr | 0.875 | ≥0.875 |
| session_map | 0.633 | ≥0.633 |

No regression expected — the search pipeline is identical, only the storage backend changes.

## 10. Future (not in this spec)

- Real-time page updates (streaming compilation)
- Multi-user isolation (project-scoped bundles)
- Web UI for browsing pages
- Fine-tuned embedding model for conversation data
- HybridDB-native hybrid search (FTS + embedding combined scoring)
