#!/usr/bin/env python3
"""Spike: Laya (local System One model) as a query router for CoreMem retrieval.

Tests whether an open, local decision model can replace/augment CoreMem's regex
heuristics at the routing decision points — measured against GROUND-TRUTH
LongMemEval question types and against the existing regex baselines.

Decision points probed (one predict call per query, 4 nouls in parallel):
  is_temporal       vs ground truth: question_type == temporal-reasoning
  is_preference     vs ground truth: question_type == single-session-preference
  is_multi_session  vs ground truth: question_type == multi-session
  needs_synthesis   (CoN routing proxy; multi-session + temporal)

Baselines from coremem heuristics (the falsified-deterministic-routing family):
  preference: coremem.retrieval._is_preference_query
  temporal:   the cue regex used in _search_messages_llm_expansion
  counting:   SearchHeuristics.is_counting_question

Usage: uv run python3 scripts/spike_laya_routing.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_queries() -> list[dict]:
    data = json.loads(Path("data/longmemeval_s_stratified_56.json").read_text())
    return [{"qid": r["question_id"], "query": r["question"], "type": r["question_type"]} for r in data]


# ── deterministic baselines (from coremem) ────────────────────────────────
from coremem.retrieval import _is_preference_query  # noqa: E402
from coremem.heuristics import SearchHeuristics  # noqa: E402

_TEMPORAL_CUES = ("before", "after", "since", "when did", "what year", "how long")


def baseline_temporal(q: str) -> bool:
    ql = q.lower()
    return any(c in ql for c in _TEMPORAL_CUES)


def baseline_preference(q: str) -> bool:
    return _is_preference_query(q)


def baseline_counting(q: str) -> bool:
    return SearchHeuristics.is_counting_question(q)


# ── Laya routing ─────────────────────────────────────────────────────────
# Perception-style phrasings (Laya docs: "ask what the text says, not what to
# do about it") — v2 supersedes the meta-question phrasing that produced
# near-zero probabilities in v1.
QUESTIONS = {
    "is_temporal": {
        "type": "noul",
        "instructions": (
            "The text mentions when something happened, a date or time, a duration, "
            "an amount of time passing, or the time order of events."
        ),
    },
    "is_preference": {
        "type": "noul",
        "instructions": (
            "The text asks for a recommendation, suggestion, or advice tailored to "
            "a particular person's tastes, likes, or way of doing things."
        ),
    },
    "is_multi_session": {
        "type": "noul",
        "instructions": (
            "The text talks about several different events or conversations at "
            "different times, rather than one event in one conversation."
        ),
    },
    "needs_synthesis": {
        "type": "noul",
        "instructions": (
            "The text asks to count things, compare things, or combine several "
            "pieces of information into one answer."
        ),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    queries = load_queries()
    if args.limit:
        queries = queries[: args.limit]
    print(f"loaded {len(queries)} queries")

    import laya

    t0 = time.perf_counter()
    agent = laya.load("convaiinnovations/laya")
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    rows = []
    latencies = []
    for i, item in enumerate(queries, 1):
        t0 = time.perf_counter()
        result = agent.predict(item["query"], QUESTIONS)
        latencies.append((time.perf_counter() - t0) * 1000)
        answers = result["answers"]
        rows.append({
            **item,
            "p_temporal": answers["is_temporal"]["noul"],
            "p_preference": answers["is_preference"]["noul"],
            "p_multi": answers["is_multi_session"]["noul"],
            "p_synthesis": answers["needs_synthesis"]["noul"],
            "c_temporal": answers["is_temporal"].get("confidence"),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(queries)}")

    thr = args.threshold

    def score(name: str, key: str, truth_fn, baseline_fn=None) -> None:
        tp = fp = tn = fn = 0
        for r in rows:
            truth = truth_fn(r)
            pred = r[key] >= thr
            tp += truth and pred
            fp += (not truth) and pred
            tn += (not truth) and not pred
            fn += truth and not pred
        acc = (tp + tn) / len(rows)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"\n{name}: acc={acc:.3f} prec={prec:.3f} recall={rec:.3f} (tp={tp} fp={fp} fn={fn} tn={tn})")
        if baseline_fn is not None:
            b_acc = sum(1 for r in rows if baseline_fn(r["query"]) == truth_fn(r)) / len(rows)
            print(f"  regex baseline acc={b_acc:.3f} (delta {acc - b_acc:+.3f})")

    print("\n=== Laya vs ground-truth LongMemEval labels ===")
    score("is_temporal", "p_temporal",
          lambda r: r["type"] == "temporal-reasoning", baseline_temporal)
    score("is_preference", "p_preference",
          lambda r: r["type"] == "single-session-preference", baseline_preference)
    score("is_multi_session", "p_multi",
          lambda r: r["type"] == "multi-session")

    # calibration sanity: mean noul for true vs false groups
    print("\n=== calibration (mean P(true) by ground-truth group) ===")
    for name, key, truth_fn in [
        ("temporal", "p_temporal", lambda r: r["type"] == "temporal-reasoning"),
        ("preference", "p_preference", lambda r: r["type"] == "single-session-preference"),
        ("multi_session", "p_multi", lambda r: r["type"] == "multi-session"),
    ]:
        pos = [r[key] for r in rows if truth_fn(r)]
        neg = [r[key] for r in rows if not truth_fn(r)]
        print(f"{name:14s} P(true|positive)={statistics.mean(pos):.3f}  "
              f"P(true|negative)={statistics.mean(neg):.3f}")

    print(f"\nlatency: median {statistics.median(latencies):.0f}ms, "
          f"p90 {sorted(latencies)[int(len(latencies)*0.9)]:.0f}ms, n={len(latencies)}")
    Path("/tmp/spike_laya_routing.json").write_text(json.dumps(rows, indent=1))
    print("per-query results -> /tmp/spike_laya_routing.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
