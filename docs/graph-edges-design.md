# Message Graph Edge Set — Design Proposal

Grounded in human memory research and agent memory architectures. The goal:
give `memorycore_traversal_v2` edges that can actually discover answers the
baseline misses, targeting the eval's hard question types (temporal-reasoning
0.599, multi-session 0.575, knowledge-update 0.655 message_recall@5 on S).

## Research foundation

### Human memory

| Phenomenon | Mechanism | Retrieval consequence |
|---|---|---|
| **Temporal contiguity + recency** (retrieved-context models: Temporal Context Model, CMR) | Items studied close in time are recalled together; a slowly drifting temporal context is the retrieval cue | Temporal adjacency is the strongest single retrieval cue |
| **Spreading activation** (Collins & Loftus 1975) | Activation spreads along associative links in a semantic network | Semantic proximity ranks candidates |
| **Cue-dependent retrieval** (Tulving) | Retrieval succeeds when cues match encoding | Multiple cue types → better recall |
| **Emotional salience** (amygdala/arousal) | Arousal enhances consolidation and attentional focusing | Emotionally charged events are better anchors |
| **Self-reference effect** (meta-analysis confirmed) | Self-referent encoding produces superior recall | First-person content is a strong cue |
| **Interference theory** | Similar memories interfere; the major cause of forgetting | Knowledge updates need contrast, not just similarity |
| **Narrative/causal structure** | Causally connected events are remembered as coherent stories | Causal links bridge episodes |
| **Social memory** | People are the strongest memory anchors | Person entities organize episodes |
| **Rehearsal/frequency** | Repetition strengthens consolidation | Repeated facts are more reliable |
| **Primacy/recency (serial position)** | First/last items of a sequence are best recalled | Session boundaries are anchors |
| **Schema/scripts** | Generalized event structures organize experience | Event templates group episodes |

### Agent memory architectures

- **MAGMA** (arXiv 2602.05665): multi-graph architecture **separating
  temporal, causal, and entity information** — argues monolithic
  semantic-similarity stores entangle these dimensions and hurt reasoning
  accuracy. Direct validation of separate edge types.
- **Zep**: temporal knowledge graph outperforms MemGPT on deep-memory
  retrieval — temporal structure is the differentiator.
- **HAGE**: retrieval as query-conditioned traversal over a weighted
  multi-relational graph — edge weights matter, traversal is query-guided.
- **LongMemEval hard types**: temporal-reasoning and multi-session questions
  are the weakest for every strategy — these need temporal + cross-session
  edges specifically.

## Proposed edge set

### Tier 1 — cheap, strong grounding, targets eval hard types

| Edge | Definition | Human grounding | Targets |
|---|---|---|---|
| `turn_qa` | user message ↔ assistant message in the same turn (turn_id already exists) | Dialogue structure; cue-dependent retrieval | single-session-assistant |
| `update` | messages linked by change language: "changed", "switched", "no longer", "now", "instead of", "used to" | Interference theory — updates need contrast with the superseded fact | knowledge-update |
| `causal` | messages linked by causal language: "because", "so", "that's why", "therefore", "as a result" | Narrative memory; MAGMA's causal graph | temporal-reasoning |
| `self_reference` | first-person content ("I", "my", "me", "we") — connects user-centric messages | Self-reference effect | single-session-user, preference |
| `emotional` | messages with sentiment intensity above threshold (lexicon/VADER) | Amygdala/arousal consolidation | general salience |

### Tier 2 — moderate cost

| Edge | Definition | Human grounding | Targets |
|---|---|---|---|
| `entity` | messages sharing a named entity (person/place/org; pattern NER: capitalized names, known-name lists) | Social/spatial memory; MAGMA's entity graph | multi-session, temporal |
| `semantic` | cross-session pairs with embedding cosine similarity above threshold (embeddings already exist in HybridDB) | Spreading activation — semantic association without word overlap | multi-session |
| `rehearsal` | repeated mention of the same fact (near-duplicate content across sessions) — reinforcement weight | Rehearsal/consolidation | knowledge-update, preference |

### Tier 3 — expensive (LLM extraction)

| Edge | Definition | Human grounding | Targets |
|---|---|---|---|
| `schema` | messages matching an event template ("planning a trip", "meeting someone", "making a purchase") | Schema/script theory | multi-session, temporal |

## Design principles (from the falsified SPEC, still valid)

1. Graph proximity is evidence, not relevance — every hop re-checks the query.
2. Seeds come from the strongest retriever (baseline's exact rerank window).
3. Candidates restricted to sessions outside the seed set (the only value-add).
4. Session caps prevent hub gravity wells.
5. Fallback: no candidates → output identical to baseline.

## Implementation notes

- All Tier 1 edges are pattern-based, zero-LLM, consistent with CoreMem's ethos.
- `turn_qa` is free: turn_id is already stored on every message.
- `semantic` reuses the embeddings HybridDB already computes — no new model.
- Edge weights encode the human-memory strength: temporal 1.0, causal 0.9,
  entity 0.8, update 0.8, turn_qa 0.7, emotional 0.6, self_reference 0.5,
  semantic 0.5, rehearsal 0.4.
- Evaluation: 20-question subset first (the v3 baseline: 0.974 session
  recall). Only a beat on the subset earns an S-scale run.

## Falsification criteria

The graph earns its place only if it beats the baseline on the 20-question
subset — the v3 result (identical to baseline) is the bar. If the richer edge
set still ties, the conclusion is final: the baseline's rerank window covers
the search space, and graph retrieval is retired with clean evidence.

## Result (2026-08-18) — 20-question subset

**v4 with the full edge set (topic, turn_qa, update, causal, self_reference,
emotional, entity, semantic): identical to baseline on every metric.**

| Metric | episodic | traversal_v4 |
|---|---:|---:|
| session_recall@5 | 0.974 | 0.974 |
| message_recall@5 | 0.605 | 0.605 |
| message_hit@5 | 0.842 | 0.842 |
| bundle_message_recall | 0.868 | 0.868 |

## Result (2026-08-18) — S-scale (resumable run, `scripts/eval_graph_s.py`)

Full 500-question run (265 MB dataset, ~48 sessions per question).
Per-type deltas (traversal − baseline) at 359/500 questions:

| Type | n | session_recall Δ | message_recall Δ |
|---|---:|---:|---:|
| multi-session | 133 (complete) | −0.0008 | +0.0012 |
| single-session-user | 70 (complete) | 0.0000 | 0.0000 |
| single-session-preference | 30 (complete) | 0.0000 | 0.0000 |
| temporal-reasoning | 126/133 | −0.0119 | −0.0079 |

**Verdict: parked 2026-08-18.** The graph is neutral-to-negative with full
statistical power:

- multi-session (the type the cross-session edges were designed for):
  neutral. The interim +0.0053 signal at 75 questions was a small-sample
  artifact — it eroded to −0.0008 at the full 133.
- temporal-reasoning (the type the causal/update edges were designed for):
  **negative** — the graph candidates displace the baseline's
  temporally-correct sessions in MMR.
- The win ratio at the session-set level (superset in most questions) is
  misleading: the graph adds different sessions, not better ones.

Root causes (consistent across both scales):

1. The baseline's rerank window (top-50 RRF + cross-encoder + MMR) already
   covers the search space; the graph has almost nothing to discover.
2. MMR's one-result-per-session cap structurally blocks same-session
   candidates (turn_qa, temporal neighbors).
3. Edges connect related messages, not answers — the cross-encoder is a
   better relevance proxy than any edge type.
4. Retrieval cost: graph build is 13× baseline search per query (12.9s vs
   1.0s) — a derived index that would need caching/incremental maintenance,
   a question made moot by the recall verdict.

The eval mode and the instrumented harness (`scripts/eval_graph_s.py` with
resume, ingest/search phase timing, graph composition) remain ablation-ready
for any future hypothesis.
