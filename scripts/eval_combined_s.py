#!/usr/bin/env python3
"""S-scale combined-improvements eval — resumable, trackable.

Runs ``memorycore_episodic_reranked_v2`` (L-12 cross-encoder + preference
union routing + temporal decomposition) on the full S dataset and compares
per-question against the saved baseline rows from
``results/eval_graph_s.json.jsonl`` (memorycore_episodic_reranked, L-6).

Usage:
    # smoke: first 2 questions
    uv run scripts/eval_combined_s.py data/longmemeval_s_cleaned.json --limit 2 --progress

    # full run
    uv run scripts/eval_combined_s.py data/longmemeval_s_cleaned.json --progress

    # resume after interruption
    uv run scripts/eval_combined_s.py data/longmemeval_s_cleaned.json --resume --progress

Outputs:
    {output}.jsonl             one line per question (crash-safe)
    {output}.checkpoint.json   resume state
    {output}                   final aggregate JSON (metrics + comparison)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_journal_longmemeval import (  # noqa: E402
    RawSearchHit,
    ReferenceMessage,
    _build_scored_row,
    _empty_score,
    _fractional_recall,
    _prepare_instance,
    stream_longmemeval_instances,
)
from coremem import MemoryCore  # noqa: E402
from coremem.retrieval import search_messages_preference_union  # noqa: E402

MODE = "memorycore_episodic_reranked_v2"
BASELINE_JSONL = Path(__file__).resolve().parents[1] / "results" / "eval_graph_s.json.jsonl"


def _ingest_instance(core: MemoryCore, instance: Any) -> int:
    count = 0
    for session in instance.sessions:
        for msg in session.messages:
            core.db.insert("messages", {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "session_id": msg.session_id or "",
                "user_id": msg.user_id or "",
                "agent_id": msg.agent_id or "",
                "turn_id": session.turn_id,
                "metadata": "{}",
                "ts": msg.ts.isoformat() if msg.ts else "1970-01-01T00:00:00Z",
            })
            count += 1
    return count


def _score(core: MemoryCore, instance: Any, truth: Any, *, k: int) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    if truth.abstention_expected:
        row = _empty_score(instance, truth, mode=MODE)
        row.update({
            "bundle_session_ids": [], "bundle_message_ids": [], "bundle_count": 0,
            "bundle_context_chars": 0, "bundle_message_recall": 0.0, "bundle_message_hit": False,
        })
        return row, time.perf_counter() - t0
    primary = search_messages_preference_union(core, instance.query, limit=k, per_variant=40)
    hits = [
        RawSearchHit(
            message=ReferenceMessage(
                turn_id=str((result.memory.metadata or {}).get("turn_id") or result.memory.id or ""),
                session_id=result.memory.session_id or "",
                message_id=result.memory.id or "",
                role=result.memory.role,
                content=result.memory.content,
            ),
            score=result.score,
        )
        for result in primary
    ]
    row = _build_scored_row(instance, truth, hits, mode=MODE, k=k)
    bundles = core._reconstruct_sessions(instance.query, session_limit=k, primary_results=primary)
    bundle_message_ids = [m.id for b in bundles for m in b.messages if m.id]
    row.update({
        "bundle_session_ids": [b.session_id for b in bundles],
        "bundle_message_ids": bundle_message_ids,
        "bundle_count": len(bundles),
        "bundle_context_chars": sum(len(m.content) for b in bundles for m in b.messages),
        "bundle_message_recall": _fractional_recall(bundle_message_ids, truth.expected_message_ids),
        "bundle_message_hit": bool(set(bundle_message_ids) & set(truth.expected_message_ids)),
    })
    return row, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="LongMemEval S JSON file")
    parser.add_argument("--limit", type=int, help="Maximum number of questions to process")
    parser.add_argument("--resume", action="store_true", help="Skip questions already completed")
    parser.add_argument("--progress", action="store_true", help="Print per-question progress")
    parser.add_argument("--output", type=Path, default=Path("results/eval_combined_s.json"))
    parser.add_argument("--root", type=Path, default=Path(tempfile.mkdtemp(prefix="coremem-combined-s-")))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output.with_suffix(output.suffix + ".jsonl")
    checkpoint_path = output.with_suffix(output.suffix + ".checkpoint.json")

    completed: set[int] = set()
    if args.resume and checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        completed = set(state.get("completed", []))
        print(f"resuming: {len(completed)} questions already completed")

    rows: list[dict[str, Any]] = []
    if args.resume and jsonl_path.exists():
        rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    t_start = time.perf_counter()
    processed = 0
    for index, raw in stream_longmemeval_instances(args.data, limit=args.limit):
        if index in completed:
            continue
        instance, truth = _prepare_instance(raw, index)
        qid = instance.question_id
        t_q = time.perf_counter()

        root = args.root / f"{index:04d}_{qid[:8]}"
        if root.exists():
            shutil.rmtree(root)
        core = MemoryCore(path=str(root / "hybrid"))

        t0 = time.perf_counter()
        n_msgs = _ingest_instance(core, instance)
        ingest_s = time.perf_counter() - t0

        row, search_s = _score(core, instance, truth, k=args.k)
        total_s = time.perf_counter() - t_q

        record = {
            "question_id": qid,
            "question_type": instance.question_type,
            "mode": MODE,
            "messages": n_msgs,
            "sessions": len(instance.sessions),
            "ingest_s": ingest_s,
            "ingest_msgs_per_s": n_msgs / ingest_s if ingest_s else 0.0,
            "search_s": search_s,
            "question_total_s": total_s,
            **row,
        }
        rows.append(record)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        completed.add(index)
        checkpoint_path.write_text(
            json.dumps({"completed": sorted(completed)}, indent=2), encoding="utf-8",
        )
        processed += 1
        if args.progress:
            s = record["scoring"]
            print(
                f"[{processed:>3}] {instance.question_type:<26} {qid[:20]} "
                f"ingest {ingest_s:5.1f}s search {search_s:5.1f}s "
                f"srec {s[f'session_recall@{args.k}']:.2f} mrec {s[f'message_recall@{args.k}']:.2f}",
                flush=True,
            )

    # ── Aggregate + compare against the saved baseline ────────────────────
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics: dict[str, Any] = {}
    for key in (
        f"session_recall@{args.k}", f"message_recall@{args.k}",
        f"session_hit@{args.k}", f"message_hit@{args.k}", "session_rr", "session_map",
    ):
        metrics[key] = _mean([r["scoring"].get(key, 0) for r in rows])
    metrics["bundle_message_recall"] = _mean([r.get("bundle_message_recall", 0) for r in rows])
    metrics["bundle_message_hit"] = _mean([r.get("bundle_message_hit", 0) for r in rows])
    metrics["n"] = len(rows)

    result = {
        "dataset": str(args.data),
        "k": args.k,
        "completed": len(completed),
        "total_time_s": time.perf_counter() - t_start,
        "metrics": metrics,
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # comparison vs saved baseline
    baseline: dict[str, dict[str, Any]] = {}
    if BASELINE_JSONL.exists():
        for line in BASELINE_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["mode"] == "memorycore_episodic_reranked":
                baseline[r["question_id"]] = r
    matched = [r for r in rows if r["question_id"] in baseline]

    print("\n=== Combined v2 vs saved baseline (L-6) ===")
    print(f"matched questions: {len(matched)} / {len(rows)}")
    print(f"{'metric':<28}{'baseline':>12}{'combined':>12}{'delta':>10}")
    for key in (
        f"session_recall@{args.k}", f"message_recall@{args.k}",
        f"session_hit@{args.k}", f"message_hit@{args.k}", "session_rr", "session_map",
    ):
        b = _mean([baseline[r["question_id"]]["scoring"].get(key, 0) for r in matched])
        c = _mean([r["scoring"].get(key, 0) for r in matched])
        print(f"{key:<28}{b:>12.4f}{c:>12.4f}{c-b:>10.4f}")
    for key in ("bundle_message_recall", "bundle_message_hit"):
        b = _mean([baseline[r["question_id"]].get(key, 0) for r in matched])
        c = _mean([r.get(key, 0) for r in matched])
        print(f"{key:<28}{b:>12.4f}{c:>12.4f}{c-b:>10.4f}")

    # by-type comparison
    from collections import defaultdict
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in matched:
        by_type[r["question_type"]].append(r)
    print(f"\n{'type':<26}{'n':>4}{'base_srec':>11}{'v2_srec':>11}{'Δ':>8}{'base_mrec':>11}{'v2_mrec':>11}{'Δ':>8}")
    for t, rs in sorted(by_type.items()):
        bs = _mean([baseline[r["question_id"]]["scoring"].get(f"session_recall@{args.k}", 0) for r in rs])
        cs = _mean([r["scoring"].get(f"session_recall@{args.k}", 0) for r in rs])
        bm = _mean([baseline[r["question_id"]]["scoring"].get(f"message_recall@{args.k}", 0) for r in rs])
        cm = _mean([r["scoring"].get(f"message_recall@{args.k}", 0) for r in rs])
        print(f"{t:<26}{len(rs):>4}{bs:>11.4f}{cs:>11.4f}{cs-bs:>+8.4f}{bm:>11.4f}{cm:>11.4f}{cm-bm:>+8.4f}")

    print(f"\nquestions completed: {len(completed)}, total time: {result['total_time_s']:.0f}s")
    print(f"results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
