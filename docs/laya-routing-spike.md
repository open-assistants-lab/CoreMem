# Spike: Laya (local System One model) as a CoreMem query router

**Date:** 2026-09-07 · **Script:** `scripts/spike_laya_routing.py` · **Model:** `convaiinnovations/laya` 0.3.5 (Apache-2.0, 421M params, local)

## Motivation

The LangChain/Jev article argues for cheap calibrated classifiers at the decision
points *around* an agent loop. CoreMem's decision points are routing decisions
(query type → strategy), and our deterministic router was falsified at 78%
("deterministic classifier routing — 78% accuracy on S, not reliable enough").
Laya is the open, local alternative to Jev that removes the privacy objection.
This spike measures Laya against **ground-truth LongMemEval labels** (stratified-56)
and against CoreMem's regex baselines.

## Method

One `predict` call per query, 4 nouls in parallel (is_temporal / is_preference /
is_multi_session / needs_synthesis). Ground truth: `question_type` labels
(temporal-reasoning 8, single-session-preference 8, multi-session 10, others 30).
Baselines: `_is_preference_query`, the temporal cue regex
(`before/after/since/when did/what year/how long`), `is_counting_question`.

**v1 phrasing violated Laya's documented guidance** ("ask what the text *says*, not
what to do about it") — meta-questions like "does the question ask about time?"
produced near-zero probabilities (preference recall 0.00). **v2 uses perception-style
phrasings** and is reported below. The phrasing iteration is itself a finding.

## Results (v2, n=56, threshold 0.5)

| Decision | Laya acc | Regex baseline | Verdict |
|---|---:|---:|---|
| is_temporal | **0.875** | 0.786 | Laya wins (+0.089); recall 0.75, precision 0.55 |
| is_preference | 0.893 (0.911 @ thr 0.3–0.6) | **0.929** | Regex wins; Laya's positives ⊂ regex's (AND = regex) |
| is_multi_session | 0.750 | — (all-negative = 0.821) | **No signal** — worse than trivial |

Group separation (mean P(true)) — v2:

| Decision | P(true \| positive) | P(true \| negative) |
|---|---:|---:|
| temporal | 0.638 | 0.239 |
| preference | 0.608 | 0.081 |
| multi_session | 0.371 | 0.269 |

Latency: **median 181 ms/call, p90 241 ms** (CPU; Laya's page reports 21 ms on
laptop GPU). Model load ~190 s cold. Threshold sweep: temporal best at 0.5;
preference flat 0.3–0.6 then degrades.

## Findings

1. **Temporal routing is the one adoption candidate.** Laya beats the cue regex
   (+0.089) and its true-prediction set is a superset of the regex's (AND=0.875=laya,
   OR=0.786=regex) — it catches temporal questions the cues miss without adding
   false positives beyond them.
2. **Preference routing: keep the regex.** `_is_preference_query` (0.929) beats
   Laya's best operating point (0.911), and Laya's detected set is a subset — no
   ensemble gain (OR = laya, AND = regex).
3. **Synthesis routing (for CoN gating): no signal.** 0.750 < the 0.821
   all-negative base rate; the probabilities barely separate (0.371 vs 0.269).
   The CoN win on synthesis-heavy questions cannot be *routed to* with this model
   and phrasing.
4. **Phrasing is the integration skill, not an afterthought.** The v1→v2 rewrite
   moved preference recall 0.00 → 0.75 and lifted separation on every probe. Any
   production integration must budget iteration on question wording.
5. **Calibration caveat is real.** The shipped checkpoint logs
   `temperatures outside [0.5, 5] ... treat confidence from the affected buckets
   as uncalibrated` — the probabilities are usable as ordering signals, not as
   calibrated probabilities. This matches the third-party survey's warning that
   none of the Jev alternatives are trained the way Jev is.
6. **Latency**: ~180 ms/call on CPU is fine for ingest-time or offline decisions,
   too slow for per-query hot-path routing; MPS/GPU (~20–35 ms per the model card)
   would need verification before hot-path use.

## Verdict

**Not adopted now; one conditional follow-up.** Temporal routing via a local
System One model is the only decision point where the classifier beats our
deterministic baseline — and even that must clear the real bar: does
Laya-routed temporal decomposition beat cue-regex-routed decomposition on
temporal-reasoning recall (the metric that matters), not just routing accuracy?
Preference and synthesis routing are closed. The calibration warning and the
190 s load time argue for keeping any classifier strictly opt-in and offline.

## Artifacts

- `scripts/spike_laya_routing.py` (v2 phrasings), per-query results at `/tmp/spike_laya_routing.json`
- Model: `convaiinnovations/laya` (Apache-2.0); ecosystem survey: systemonemodels.org
