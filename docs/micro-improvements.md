# Micro-Improvements Log

2026-06-01

Low-effort, high-impact improvements identified during benchmarking and
competitor analysis. Not blocking, not urgent — queue for a future sprint.

---

## HybridDB

| # | What | Effort | Expected gain | How |
|---|------|--------|--------------|-----|
| 1 | **FTS5 porter stemming** | 1 line | +3-5% BM25 | Verify `tokenize=porter` is set on FTS5 CREATE TABLE. If using default `unicode61`, add `tokenize=porter`. Catches "running" / "run" mismatches. |
| 2 | **ChromaDB 512-token window** | Config | +1-2% vector | ChromaDB defaults to 256-token window for `all-MiniLM-L6-v2`. agentmemory uses 512. Check if ChromaDB SDK allows `model_max_length=512` override on `SentenceTransformerEmbeddingFunction`. |
| 3 | **Session-level search indexing** | Medium | +2-3% BM25 precision | Current per-message indexing vs agentmemory's per-session. Add `search_session_dedup` option that merges same-session messages into one document for BM25 query. Already partially done in HybridBackend (line 237-241). |
| 4 | **BM25 column weight tuning** | Small | +1-2% | FTS5 `bm25(fts_table, 1.0, 0.75)` uses standard k1=1.0, b=0.75. No tuning needed for most datasets. SQLite FTS5 doesn't expose column-level weights — would need separate FTS5 indexes per column for weighted fusion. |
| 5 | **Embedding model upgrade** | Config | +3-5% overall | `all-MiniLM-L6-v2` (384-dim) → `all-mpnet-base-v2` (768-dim) or `bge-small-en-v1.5`. Double the vector quality, 30% larger index. ChromaDB default is already MiniLM — would need config change. |

## CoreMem

| # | What | Effort | Expected gain | How |
|---|------|--------|--------------|-----|
| 6 | **Observer: mechanical compression** | Medium | 0% hallucination | For tool-call-driven agents (coding), skip LLM extraction. Compress `tool_name + tool_input + tool_output` into title + narrative mechanically. agentmemory's `buildSyntheticCompression()` approach. Not applicable for conversation memory. |
| 7 | **Observer: default model switch** | Config | -60% hallucination | Default from `ollama:llama3.2` → `openai:gpt-4o-mini` or `anthropic:claude-haiku-4-5`. Document tradeoffs: cost vs accuracy. Keep source_quote gate as safety net. |
| 8 | **Heuristics: negation penalty** | 5 lines | +1% | Query "not/never/without" → penalize content containing those terms. Lowers false positives on negative queries. |
| 9 | **Heuristics: entity overlap** | 20 lines | +1-2% | Match named entities (people, companies, locations) between query and content. Uses spaCy small model (~13MB). Already available via `entities` field on observations. |

## Eval Pipeline

| # | What | Effort | Expected gain | How |
|---|------|--------|--------------|-----|
| 10 | **Observer eval: hallucination ground truth** | Manual | N/A | Label 10-20 LongMemEval questions manually. Mark which observations are correct/fabricated. Use as golden set for regression testing when changing models/prompts. |

---

## Implementation order (if prioritizing)

1. Add `tokenize=porter` (1 line, instant gain)
2. Switch Observer default model to stronger one
3. ChromaDB 512-token window
4. Golden set for hallucination regression
