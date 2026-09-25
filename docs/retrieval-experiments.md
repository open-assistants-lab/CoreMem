# Retrieval Improvement Experiments — Running Record

Status: **lever 1 validated (full S), lever 2 in progress**

Reader/judge model for all runs below: `ollama:gpt-oss:120b-cloud` (local Ollama cloud proxy),
both answer and judge roles, blind anonymous judge. This differs from the older
`deepseek:deepseek-chat` runs in AGENTS.md — absolute numbers are NOT comparable
across reader models, only within-run paired comparisons.

---

## Lever 1 — Chain-of-Note reading (done: full S-scale)

**Claim (LongMemEval §5.5):** answering via extract-then-reason ("CoN") beats direct
answering on the same retrieved context, especially for multi-item synthesis.

**Design:** identical `episodic_4k_reranked` contexts (4k bundles, cross-encoder reranked,
evidence-first); only the answer prompt varies. `_con` = CoN prompt (notes then final
answer line). `_con_json` = same prompt with bundles serialized as JSON.

### Full S (500 questions) — `results/eval_answer_s500_con.json`

| mode | accuracy | answerable | abstention | answer_s |
|---|---:|---:|---:|---:|
| `memorycore_llm_expansion` (1 LLM call/query) | **0.704** | 0.698 | 0.800 | 1.30 |
| `episodic_4k_reranked_con` (CoN) | **0.692** | 0.681 | 0.867 | 2.13 |
| `episodic_4k_reranked` (current default) | 0.674 | 0.660 | 0.900 | 1.39 |
| `episodic_4k_reranked_con_json` | 0.674 | 0.662 | 0.867 | 2.19 |
| `episodic_cap2` (opt-in) | 0.666 | 0.662 | 0.733 | — |

- CoN vs default: **+0.018** (17 wins / 8 losses on discordant pairs, n=500)
- **McNemar p = 0.108** (continuity-corrected χ²=2.56) — suggestive, NOT significant
- Per-type deltas (CoN − direct): ss-user +3, temporal +3, multi-session +2,
  knowledge-update +1, ss-assistant +1, preference 0, **abstention −1**
- Cost: CoN answers 2.13s vs 1.39s direct (+53% tokens)

### Consistency across runs

| run | CoN wins | CoN losses | net |
|---|---:|---:|---:|
| s20 (all single-session-user) | 1 | 0 | +1 |
| stratified-56 | 4 | 1 | +3 |
| full S-500 | 17 | 8 | +9 |

Direction stable across all three; magnitude small (~+0.02), p≈0.11 at full scale.

### Verdict (lever 1)
Positive signal, mechanistically where expected (synthesis types), but not statistically
significant. One abstention regression and +53% answer tokens. **Not folded into default**;
retained as opt-in eval mode. Pair with lever 2 (orthogonal axis) before deciding.

---

## Lever 2 — Fact-augmented key expansion (in progress)

**Claim (LongMemEval §5.3):** prepending LLM-extracted user facts to each indexed
document (K = V + fact) improves recall (+9.4% recall@k / +5.4% QA in paper).

**Implementation (coremem/core.py, opt-in `fact_augment=True`):**
- `message_facts` side table (id PK, facts TEXT)
- `_extract_and_store_facts()` — batched LLM extraction (20 msgs/call) of user-role facts
- `_flush_journal_batched(augment=...)` — encodes augmented text into Chroma vectors;
  SQLite/verbatim content untouched (hybriddb search re-fetches rows from SQLite)
- Ingest-time only: 1 LLM call per 20 user messages (≈15–25 calls/question on S)
- Verified: augmented vector ≠ stock vector (cos 0.60 on smoke), verbatim content
  preserved, no fact-prefix leakage into search results

**Experiment:** stratified-56 (8/type × 7 buckets, `data/longmemeval_s_stratified_56.json`),
same reader/judge as lever 1. Mode `episodic_4k_reranked_factaug` (identical retrieval
chain, only the index differs). Control = `episodic_4k_reranked` from the same run.

- Cost note: per-question ingest is the bottleneck — extraction ≈ 15–25 LLM calls +
  double ingest (stock + factaug cores). Full-56 run ≈ 1–2 h vs ~30 s/question for
  non-augmented modes.

**Status:** run restarted after fixing `message_facts` UNIQUE collision (LLM hallucinated
ids collide across sessions; fixed by deduping against existing side-table rows). 16/56
done before crash; resumed with the partial instance for the crashed question removed.

### Lever 1 + Lever 2 combined (planned)
Run CoN answering on factaug contexts (`episodic_4k_reranked_factaug_con`) to test
whether the two wins stack. Expectation from prior combined-validation history: they
often do NOT sum — validate, don't assume.

---

## Data & scripts
- Stratified subset: `data/longmemeval_s_stratified_56.json` (reusable, 8 per type)
- Eval modes added to `scripts/eval_answer_longmemeval.py`:
  `episodic_4k_reranked_con`, `episodic_4k_reranked_con_json`, `episodic_4k_reranked_factaug`
- Results: `results/eval_answer_s500_con.json`, `results/eval_answer_s_stratified_con.json`,
  `results/eval_answer_s_stratified_factaug.json`
- Tests: `tests/test_core.py::test_fact_augment_*` (2 new, suite now 176 passed)

## LoCoMo adapter (setup complete)

`scripts/adapt_locomo.py` converts snap-research/locomo (`data/locomo/locomo10.json`,
10 conversations, 1,986 QA) into CoreMem's LongMemEval-shaped eval format.

Mapping: single-hop→single-session-user, multi-hop→multi-session,
temporal→temporal-reasoning, open-domain→open-domain (custom type),
adversarial→abstention (`_abs` suffix, empty answer). Each instance carries the
full source conversation as its per-question haystack (canonical setup).
Speaker mapping is deterministic: speaker_a→user, speaker_b→assistant.

Data quirks handled:
- evidence dia_ids with leading-zero typos (`D30:05`→`D30:5`) resolved via
  normalization so evidence matches turns
- 4 open-domain questions with empty evidence are dropped (unscorable for
  retrieval metrics; harness would misclassify as abstention)
- image turns: `blip_caption` appended as `[image: ...]`

Artifacts:
- `data/locomo/locomo_longmemeval.json` — 1,982 instances (264 MB)
- `data/locomo/locomo_stratified_30.json` — 6/type × 5 types + abstention (3 MB)
- Smoke test: `memorycore_episodic_reranked` on 3 multi-session questions runs
  end-to-end (session_hit@5 0.333, non-zero recall; abstention questions
  correctly return empty)

Cost note: ~19–35 sessions/question (~16K tokens haystack) — similar ingest
profile to LongMemEval S per question; full 1,982-question run is the heaviest
eval available in-repo. Use the stratified-30 for iteration.

## LoCoMo baseline (memorycore_episodic_reranked, k=5, stratified-30)

Overall: session_recall@5 0.722, message_recall@5 0.25, session_hit@5 0.792,
session_mrr 0.581, empty_retrieval_rate 0.20 (abstention = 6/30, correct to
return empty), abstention_false_positive_rate 0.0.

Per type (n=6 each):

| type | sess_rec | msg_rec | sess_hit | msg_hit | bundle_msg_hit |
|---|---:|---:|---:|---:|---:|
| single-session-user | 1.000 | 0.333 | 1.000 | 0.333 | 0.833 |
| temporal-reasoning | 0.833 | 0.667 | 0.833 | 0.667 | 0.667 |
| open-domain | 0.583 | 0.000 | 0.667 | 0.000 | 0.500 |
| multi-session | 0.472 | 0.000 | 0.667 | 0.000 | 0.500 |
| abstention | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Key observations (n=6/type — small, directional only):
- Session-level retrieval is strong (0.72 overall); message-level is weak (0.25).
  Same pattern as LongMemEval S (session 0.95 / message 0.62) but LoCoMo's
  evidence granularity (single turn inside a ~22-turn session) is harder.
- open-domain questions (commonsense/world knowledge, e.g. "is Melanie an ally
  to the transgender community?") have no lexical overlap with the answer turn —
  message recall 0.0 despite session hit 0.667: the right SESSION is found but
  not the exact turn. These are the retrieval-hardest questions.
- multi-session: bundle evidence hit 0.50 — bundles recover the answer turn
  even when top-k messages miss it (bundle_message_recall not stored in this
  jsonl; hit only).
- abstention 0.0 = correct (empty expected).

Files: /tmp/locomo_baseline2.json (+ .jsonl checkpoint), root /tmp/coremem-locomo-baseline2

## LoCoMo full baseline + per-conversation cache (done)

Per-conversation HybridDB cache at `data/instances_locomo/` (gitignored):
10 stores, one per conversation, **47 MB total** (vs ~5.8 GB if per-question).
Built in one pass while running the full 1,982-question baseline.

Harness change (eval_agent_journal_longmemeval.py):
- instances may carry `conversation_id`; when present, session/turn/message ids
  derive from it (stable across questions of the conversation) and the
  instance dir is `instances/{conversation_id}` — all questions share one
  ingested store. LongMemEval S (no conversation_id) is unchanged.
- `_turn_id` accepts str keys; cleanup skips shared conversation dirs.

Full LoCoMo baseline — memorycore_episodic_reranked k=5, n=1982:

| type | n | sess_rec | msg_rec | sess_hit | msg_hit | bndl_rec | bndl_hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| single-session-user | 841 | 0.867 | 0.377 | 0.868 | 0.386 | 0.733 | 0.740 |
| temporal-reasoning | 321 | 0.780 | 0.424 | 0.798 | 0.452 | 0.700 | 0.723 |
| multi-session | 282 | 0.435 | 0.117 | 0.741 | 0.252 | 0.343 | 0.649 |
| open-domain | 92 | 0.430 | 0.091 | 0.565 | 0.120 | 0.332 | 0.478 |
| abstention | 446 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **overall** | 1982 | 0.744 | 0.322 | 0.812 | 0.359 | — | — |

Overall: session_recall@5 0.744, message_recall@5 0.322, session_hit@5 0.812,
empty_retrieval_rate 0.225 (= abstention 446/1982, correct),
abstention_false_positive_rate 0.0.

Reuse check: same command with --reuse-instances --resume ran 20 questions in
~10s (model load dominated) with zero re-ingest; disk unchanged (47 MB).

Results: results/eval_locomo_full_baseline.json + .jsonl (1982 rows), log in
results/eval_locomo_full_baseline.log.

## Lever 2 — Fact-augmented key expansion: NEUTRAL on LongMemEval S (concluded)

Answer eval (stratified-56, gpt-oss-120b reader/judge): factaug 0.554 vs control
0.571 — net -1 question (multi-session 0.38→0.25; +1 ss-pref, -2 total). Not a win.

Retrieval eval (same 56 questions, harness `memorycore_episodic_reranked_factaug`
mode added to eval_agent_journal_longmemeval.py):
- session_recall@5 0.964, message_recall@5 0.526 — **identical** to stock control
- 13/56 questions retrieve different message ids (embeddings genuinely differ)
- per-type message recall delta: **0.000 across all 6 types**

Root cause (verified by direct A/B on one question): factaug changes the RAW
candidate pool (decomposed, pre-rerank), but the cross-encoder reranker restores
the identical top-5 — the CE layer is the deciding component and is robust to
the candidate-pool perturbation. Paper's +9.4% recall@k did not transfer.

Caveat: paper used Stella V5 dense retriever WITHOUT a reranker at session/round
granularity; CoreMem uses all-MiniLM + CE at message granularity. The CE
reranker appears to absorb the fact-augmentation signal.

VERDICT: factaug does not help CoreMem's default episodic pipeline on
LongMemEval S. Not folded in. (The LoCoMo run is the remaining test — its
open-domain type has near-zero lexical overlap, where fact bridging might
survive reranking. Per-conversation cache makes that run cheap.)

## Lever 3 — Embedding model swap (bge-small-en-v1.5): small positive, right types

MemDelta finding: swapping only the embedder moves LongMemEval-S accuracy
+6.2pp, biggest on temporal/multi-session. Implemented configurable embedding
model (COREMEM_EMBEDDING_MODEL env, default all-MiniLM-L6-v2) — both ingest
(batched encode) and query (hybriddb embedding_fn) use the same model.

A/B on stratified-56 (memorycore_episodic_reranked k=5, same questions):

| type | n | minilm_msg | bge_msg | delta |
|---|---:|---:|---:|---:|
| multi-session | 10 | 0.488 | 0.522 | +0.033 |
| temporal-reasoning | 8 | 0.525 | 0.550 | +0.025 |
| knowledge-update | 8 | 0.708 | 0.708 | 0.000 |
| ss-user / ss-assistant / ss-pref | 30 | flat | flat | 0.000 |
| OVERALL | 56 | 0.451 | 0.460 | +0.009 |

Harness summary: message_recall@5 0.526 -> 0.537 (+0.011); session 0.964->0.965.

Direction matches MemDelta exactly (gains on temporal + multi-session, flat on
single-session types). Magnitude small at n=56 — needs full-S to confirm
significance. bge-small-en-v1.5 is 384-dim (no collection-dimension risk) and
~130MB download. Not folded in yet.

## Lever 3 — Full-S validation (n=500, paired): CONFIRMED, right at significance edge

bge-small-en-v1.5 vs MiniLM, same harness/questions, fresh bge stores
(data/instances_s500_bge, 4.0GB):

| metric | minilm | bge | delta |
|---|---:|---:|---:|
| message_recall@5 | 0.618 | 0.628 | +0.010 |
| session_recall@5 | 0.952 | 0.957 | +0.005 |
| session_map | 0.923 | 0.928 | +0.005 |
| message_hit@5 | 0.770 | 0.774 | +0.004 |

Per-type (message recall): preference 0.367→0.433 (**+0.066** — weakest type,
biggest gain), multi-session 0.523→0.534 (+0.011), temporal 0.573→0.581
(+0.008), all others 0.000. **Zero regressions.**

Significance: 9 questions better / 2 worse / 459 same → exact binomial
p = 0.065 (right at the edge, not α=0.05).

Status: promising — the type pattern matches MemDelta (cross-session types
gain, single-session flat) AND fixes the weakest type (preference). The
stratified-56 signal (+0.011) held at full scale. Remaining question before
folding in: does the retrieval gain survive the reader (answer eval) — the
lever-1/2 lesson — and ingest cost (bge ~2.4x MiniLM encode time).

## Lever 4 — BGE-reranker-v2-m3: NEGATIVE (falsified)

Zero-code A/B (COREMEM_CROSS_ENCODER_MODEL=BAAI/bge-reranker-v2-m3) on
stratified-56, memorycore_episodic_reranked k=5:

- message_recall@5 0.526 → 0.486 (−0.040); session_recall 0.964 → 0.958
- Damage concentrated in cross-session types: multi-session 0.488→0.365
  (−0.123), temporal 0.525→0.438 (−0.088); others flat
- Latency: 109ms/candidate vs L-6's ~15ms (~7x slower, ~2.2s/query overhead)

Same failure signature as the L-12 A/B (AGENTS.md: "cancels the temporal
decomposition win"): the 568M reranker scores by general semantic relevance,
displacing verbatim evidence in synthesis-heavy types. VERDICT: keep L-6.
The MS-MARCO NDCG advantage does not transfer to conversational memory.

## Lever 5 — Deterministic time-aware range pruning: NO-OP (closed)

Implemented deterministic temporal window parsing (coremem/heuristics.py:
`parse_temporal_window(query, anchor)` + `_window_factor`) anchored to the
question date, applied as a PRIOR on the CE score ordering (post-rerank —
pre-rerank boosts are washed out, the lever-2 lesson). Mode:
`memorycore_episodic_reranked_timeprune`.

Parser handles: "in the past N months/weeks/days" (incl. number-words),
"last month/week/year", "<Month> <year>", "between <Month> and <Month>" —
7/7 targeted tests pass. "How many N ago" counting questions deliberately
get no window (event date unknown a priori; filter adds no signal).

Results:
- stratified-56: 2/56 questions have a resolvable window; retrieval changed
  0. Aggregates identical (0.964/0.526).
- ALL 133 temporal-reasoning questions (full S): only **4/133** have a
  parseable window; retrieval changed on 2; aggregates identical
  (message_recall 0.601 = 0.601, session 0.922 = 0.922).

Root cause: LongMemEval temporal questions are dominated by INTER-EVENT
ARITHMETIC ("how many days between A and B", "which happened first", 54+38
of 133) — they need BOTH event timestamps, not a filtered range. True range
queries ("last month", "in the past N months") are ~3% of the type. The
paper's +6.8–11.3% came from LLM-inferred ranges on a mix where ranges
dominate; the paper's own extractor returns N/A when no range exists.

VERDICT: no-op on this benchmark. Code retained (zero-LLM, opt-in via
anchor_ts) — the window prior is correct-by-construction and may matter on
range-heavy real-world queries, but it cannot move LongMemEval S.

## Cross-benchmark transfer — LoCoMo stratified-30 answer eval (2026-09-02)

Same harness/models as the S runs (`ollama:gpt-oss:120b-cloud`, blind judge,
LoCoMo stratified-30: 6/type × 5 types). Paired, same contexts:

| mode | acc |
|---|---:|
| `episodic_cap2` | **0.533** (+4/−0 vs direct, p=0.125) |
| direct / con / con_json / factaug | **0.400** (identical judgments) |
| `memorycore` | 0.233 |

Findings:
- **Lever 1 (CoN) does NOT transfer to LoCoMo** — zero paired flips (+0/−0),
  abstention unchanged (0.83 both). On S it was +0.018 (p=0.108). Verdict:
  benchmark-dependent, stays opt-in. The 0.16.0 `coremem.reading` guidance
  stands (enable for synthesis-heavy questions).
- **Lever 2 (factaug) doubly-falsified** — zero flips on LoCoMo too; context
  chars identical on 23/30 questions (CE reranker normalization confirmed
  cross-benchmark). CLOSED.
- **New signal: `session_cap=2` is the standout on LoCoMo** (+0.133 over
  direct, 4/0 flips) — consistent with LoCoMo's turn-level evidence inside
  ~22-turn sessions: a second message of a found session often carries the
  evidence. Stays opt-in (`session_cap=2`); worth remembering for
  conversation-memory deployments with turn-heavy sessions.
- Overall accuracy is far lower than S (0.40 vs 0.674) — LoCoMo's
  open-domain/adversarial categories and turn-level evidence granularity are
  genuinely harder for the reader.

Artifacts: results/eval_answer_locomo_stratified30_con.json (+ .log)

## Fact layer Phase 2 — fact-digest kill-gate (stratified-56, paired, blind judge)

Pre-registered criterion (read before the run): **net ≤ 0 (wins − losses on
paired answer flips) → `extract_facts` stays opt-in-off; net > 0 → scale to
full S, then LoCoMo stratified-30, before any default discussion.**

Arms (one core serves both by construction — the facts table does not affect
message search, so retrieval is byte-identical; `retrieval_seconds_mean` =
1.981s in both arms confirms it):
- control: `episodic_4k_reranked`
- treatment: `episodic_4k_reranked_factdigest` (same 4k bundle, then a ≤800-char
  ontology-tagged fact digest appended — never displaces verbatim evidence)

Reader + judge: `ollama:gpt-oss:120b-cloud`, blind, 56/56 rows, 13 modes judged
per row.

| mode | acc | answerable acc | abstention acc | context chars |
|---|---:|---:|---:|---:|
| `episodic_4k_reranked` | 0.554 | 0.500 | 0.875 | 5,114 |
| `..._factdigest` | **0.571** | **0.521** | 0.875 | 5,889 |

Paired answer flips: **+2 / −1** (net +1), exact two-sided binomial
**p = 1.000**. The three discordant questions:
- `3a704032` multi-session — digest correct, control wrong
- `32260d93` single-session-preference — digest correct, control wrong
- `51a45a95` single-session-user — digest wrong, control correct (one regression:
  the digest distracts on a simple single-session fact)

Per type (n=8 each): multi-session 0.38 → 0.50, single-session-preference
0.12 → 0.25, single-session-user 0.88 → 0.75, temporal 0.38 = 0.38,
knowledge-update 0.75 = 0.75, single-session-assistant 0.50 = 0.50,
abstention 0.88 = 0.88.

**VERDICT: net > 0 → criterion met, but directionally positive only** (+0.017
accuracy, p=1.000, 3 discordant pairs). Wins land exactly where the mechanism
predicts (cross-session synthesis, preference consolidation) and the single
regression is the predicted distraction case — the pattern is coherent, the
sample is far too small to call. Per the pre-registered rule, this is NOT a
default flip: `extract_facts` remains opt-in-off pending full-S confirmation.

The trajectory across checkpoints is itself the lesson: n=18 read **negative**
(0.722 vs 0.778), n=45 read +0.019, n=56 read +0.017. Early-stopping at the
first checkpoint would have produced the wrong call in both directions.

**Infrastructure cost (worth fixing before the full-S run):** six process
deaths over this run — 5 × MPS OOM, 1 × proxy 502 — each recovered by
`--resume` (only the in-flight question lost). Root causes: (1) the answer
eval never called `MemoryCore.close()` on its 1–2 per-question cores, leaking
a pooled Chroma client + SQLite handles per question — fixed in a05ab4a, which
doubled process lifespan (9 → 18 questions); (2) host memory exhaustion (8
Docker containers + desktop apps left ~74 MB free), worked around with
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (18 → 45 questions). The same `.close()`
gap exists in `eval_agent_journal_longmemeval.py` (1 close for 5 construction
sites) and must be fixed before the 500-question run, which would otherwise
need ~10 restart cycles.

Artifacts: `results/eval_answer_s56_factdigest.json` (+ `.log`), worktree
`fact-layer`, data `data/longmemeval_s_stratified_56.json`.
