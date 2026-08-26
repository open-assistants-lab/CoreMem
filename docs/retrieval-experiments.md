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
