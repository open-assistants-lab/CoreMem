# CoreMem Backlog

Open work queue: improvement ideas identified during benchmarking, research, and
competitor analysis — not blocking, not urgent. Each entry carries its evidence
and a reason when closed, so items are not re-litigated without new data.

Related records: `docs/retrieval-experiments.md` (concluded levers),
`docs/versioned-memory-design.md`, `docs/laya-routing-spike.md`.

---

## HybridDB

| # | What | Effort | Expected gain | How |
|---|------|--------|--------------|-----|
| 1 | **FTS5 porter stemming** | 1 line | +3-5% BM25 | Verify `tokenize=porter` is set on FTS5 CREATE TABLE. If using default `unicode61`, add `tokenize=porter`. Catches "running" / "run" mismatches. |
| 2 | **ChromaDB 512-token window** | Config | +1-2% vector | ChromaDB defaults to 256-token window for `all-MiniLM-L6-v2`. agentmemory uses 512. Check if ChromaDB SDK allows `model_max_length=512` override on `SentenceTransformerEmbeddingFunction`. |
| 3 | **Session-level search indexing** | Medium | +2-3% BM25 precision | Current per-message indexing vs agentmemory's per-session. Add `search_session_dedup` option that merges same-session messages into one document for BM25 query. Already partially done in HybridBackend (line 237-241). |
| 4 | **BM25 column weight tuning** | Small | +1-2% | FTS5 `bm25(fts_table, 1.0, 0.75)` uses standard k1=1.0, b=0.75. No tuning needed for most datasets. SQLite FTS5 doesn't expose column-level weights — would need separate FTS5 indexes per column for weighted fusion. |
| 5 | ~~**Embedding model upgrade**~~ | Config | ~~+3-5% overall~~ | ✅ **DONE (0.13.4)** — shipped `BAAI/bge-small-en-v1.5` as the default (full-S paired A/B: message_recall 0.618→0.628, preference +0.066, zero regressions, 2× faster encoding than MiniLM). See `docs/retrieval-experiments.md` lever 3. |

## CoreMem

| # | What | Effort | Expected gain | How |
|---|------|--------|--------------|-----|
| 6 | **Observer: mechanical compression** | Medium | 0% hallucination | For tool-call-driven agents (coding), skip LLM extraction. Compress `tool_name + tool_input + tool_output` into title + narrative mechanically. agentmemory's `buildSyntheticCompression()` approach. Not applicable for conversation memory. |

| 7 | **Heuristics: negation penalty** | 5 lines | +1% | Query "not/never/without" → penalize content containing those terms. Lowers false positives on negative queries. |
| 8 | **Heuristics: entity overlap** | 20 lines | +1-2% | Match named entities (people, companies, locations) between query and content. Uses spaCy small model (~13MB). Already available via `entities` field on observations. |

## Eval Pipeline

| # | What | Effort | Expected gain | How |
|---|------|--------|--------------|-----|
| 9 | **Observer eval: hallucination ground truth** | Manual | N/A | Label 10-20 LongMemEval questions manually. Mark which observations are correct/fabricated. Use as golden set for regression testing when changing models/prompts. |

---

## 2026-09-07 — System One classifiers & the routing-headroom finding

From the LangChain/Jev article + the Laya spike (`docs/laya-routing-spike.md`,
`scripts/spike_laya_routing.py`) and the 500-question mode-judgment analysis.

### Open items (ranked)

| # | What | Effort | Expected gain | How |
|---|------|--------|--------------|-----|
| 10 | **Fact-candidate validation with a local System One model** | Medium | extraction precision ↑ (lever-2's real failure mode was hallucinated facts) | noul per extracted candidate: *"the text states this fact"* + *"is this durable vs transient?"* — batch at ingest, 181ms/call fine offline, local model (no memory leaves the machine). **Test corpus already exists**: the facts the kill-gate run extracted (per-question `facts` tables under `/tmp/coremem-factdigest`), with their `source_message_id` back-references |
| 11 | **Retrieval-feature router** (NOT a text classifier — small logistic regression on numeric features) | Medium | **+16.2 pts oracle headroom** (best fixed mode 0.704 → per-question oracle 0.866); 62% of questions have mode disagreement | Features: score spread of the decomposed pool, BM25-vs-vector leg agreement, CE score margin, session clustering, query length. **Caveat measured:** perfect *question-type* routing only captures +3.4 in-sample, and the detectable types (temporal, 0.875 via Laya) have zero routing gain — the signal is in retrieval results, not query text |
| 12 | **Entity resolution / merge dedup** (Laya noul) | Small | fact-layer lookup precision | *"do these two mentions refer to the same entity?"* — batch; needed before aliases (`user.dog` vs `"my dog Max"`) resolve reliably. Pair with `merge_facts` |
| 13 | **Query-variant filtering** in `decompose_queries` | Small | marginal | noul per variant: *"is this a useful search cue?"* — currently regex drops only pure time-unit phrases; needs MPS (~20–35ms) before hot-path use, or precompute |
| 14 | **Ingest-time message salience** ("worth remembering?") | Small | write-path efficiency | ⚠️ risky: false negatives lose memories permanently. Only with a conservative threshold + chain-backed review |

### Closed / not adopted (with reasons — do not re-litigate without new evidence)

- **Preference routing via classifier**: regex `_is_preference_query` wins (0.929 vs Laya's best 0.911; Laya's detected set is a subset — no ensemble gain).
- **Synthesis/multi-session routing** (to gate CoN): no signal (0.750 < 0.821 trivial baseline; P(true) 0.371 vs 0.269).
- **Question-type strategy routing**: detectable types don't align with high-gain types (temporal = +0.00 gain; abstention/preference = biggest gains but not classifier-detectable / already routed).
- **Reranker replacement**: contraindicated by the falsification record (L-12, BGE-reranker-v2-m3 both hurt; relevance scoring is where size doesn't help).
- **MCP destructive-op risk gating**: the versioned chain makes ops reversible; low value for a memory library.

### Alternatives evaluated (System One models)

| Model | Verdict | Why |
|---|---|---|
| **Laya** (Apache-2.0, 421M) | **spiked** — see `docs/laya-routing-spike.md` | Fits our decision points (text classification), runs locally on this Mac, 181ms/call CPU, measured ECE 0.081. Only the *temporal* routing probe beat our regex; preference/synthesis did not |
| **NanoJev** (MIT, 0.6B, [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev)) | **parked — not a fit for current use cases** | (1) Domain: trained on **game/embodied decision questions** (ViZDoom aim-fire, maze, snake), not text classification — our decision points are all text-over-memory; zero-shot text transfer is unpromising. (2) Footprint: **2.4 GB** safetensors (13× Laya) — this machine is already OOMing at ~1 GB eval footprints. (3) Serving is **CUDA/triton-first** (`--disable-native-triton` fallback exists; MPS/CPU path unclaimed). **Exception worth remembering:** its **Choice primitive takes 2–255 dynamic candidates** — the right structural shape for *action selection* (e.g. "which retrieval strategy?") if we ever fine-tune a router on labelled outcomes (MIT license, dataset + training recipe shipped) |
| Von, SemIf/openjev, Decider, OpenDecision, mini-jev, … | not evaluated | Survey (systemonemodels.org) found no calibration metric or explicitly disclaimed calibration for these; Von is the latency standout (sub-15ms, MPS) if latency ever becomes the binding constraint |

### Constraints for any System One adoption

- **Latency**: 181 ms/call CPU (recall is ~65 ms warm) + ~190 s cold load → **batch/ingest only**; hot-path needs MPS (~20–35 ms) verification.
- **Calibration**: the shipped Laya checkpoint logs *"temperatures outside [0.5, 5] … treat confidence from the affected buckets as uncalibrated"* — treat probabilities as ordering signals; tune thresholds per deployment.
- **Phrasing is the integration skill**: v1 meta-questions → preference recall 0.00; v2 perception phrasing → 0.75. Budget iteration on wording for every new question.

---

## Implementation order (if prioritizing)

1. Add `tokenize=porter` (1 line, instant gain)
2. ChromaDB 512-token window
3. Golden set for hallucination regression
