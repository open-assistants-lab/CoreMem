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
