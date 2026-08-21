#!/usr/bin/env python3
"""S-scale graph traversal eval — resumable, with extensive timing.

Compares ``memorycore_episodic_reranked`` (baseline) vs
``memorycore_traversal_v2`` (graph) per question, recording detailed phase
timing (ingest, seed search, graph build, traversal, rerank) and graph
composition (nodes, edges by type, candidates) alongside the standard
retrieval metrics.

Usage:
    # smoke: first 2 questions
    uv run scripts/eval_graph_s.py data/longmemeval_s_cleaned.json --limit 2 --progress

    # full run (resumable)
    uv run scripts/eval_graph_s.py data/longmemeval_s_cleaned.json --progress

    # resume after interruption
    uv run scripts/eval_graph_s.py data/longmemeval_s_cleaned.json --resume --progress

Outputs:
    {output}.jsonl             one line per question per mode (crash-safe)
    {output}.checkpoint.json   resume state (completed question ids)
    {output}                   final aggregate JSON (metrics + timing table)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_journal_longmemeval import (  # noqa: E402
    PreparedInstance,
    QuestionTruth,
    RawSearchHit,
    ReferenceMessage,
    _build_scored_row,
    _empty_score,
    _fractional_recall,
    _prepare_instance,
    _unique,
    stream_longmemeval_instances,
)
from coremem import MemoryCore  # noqa: E402
from coremem.traversal import search_messages_traversal  # noqa: E402

BASELINE_MODE = "memorycore_episodic_reranked"
TRAVERSAL_MODE = "memorycore_traversal_v2"


def _ingest_instance(core: MemoryCore, instance: PreparedInstance) -> int:
    """Canonical eval ingestion (batch insert, single journal flush)."""
    count = 0
    for session in instance.sessions:
        core.ingest_many([
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "session_id": msg.session_id or "",
                "user_id": msg.user_id or "",
                "agent_id": msg.agent_id or "",
                "turn_id": session.turn_id,
                "metadata": {},
                "ts": msg.ts.isoformat() if msg.ts else "1970-01-01T00:00:00Z",
            }
            for msg in session.messages
        ])
        count += len(session.messages)
    return count


def _hits_from_results(results: list[Any]) -> list[RawSearchHit]:
    return [
        RawSearchHit(
            message=ReferenceMessage(
                turn_id=str((r.memory.metadata or {}).get("turn_id") or r.memory.id or ""),
                session_id=r.memory.session_id or "",
                message_id=r.memory.id or "",
                role=r.memory.role,
                content=r.memory.content,
            ),
            score=r.score,
        )
        for r in results
    ]


def _bundle_metrics(core: MemoryCore, query: str, primary: list[Any], truth: QuestionTruth, k: int) -> dict[str, Any]:
    bundles = core._reconstruct_sessions(query, session_limit=k, primary_results=primary)
    bundle_message_ids = [m.id for b in bundles for m in b.messages if m.id]
    return {
        "bundle_session_ids": [b.session_id for b in bundles],
        "bundle_message_ids": bundle_message_ids,
        "bundle_count": len(bundles),
        "bundle_context_chars": sum(len(m.content) for b in bundles for m in b.messages),
        "bundle_message_recall": _fractional_recall(bundle_message_ids, truth.expected_message_ids),
        "bundle_message_hit": bool(set(bundle_message_ids) & set(truth.expected_message_ids)),
    }


def _score_question(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
) -> dict[str, dict[str, Any]]:
    """Score both modes for one question, with phase timing."""
    query = instance.query
    rows: dict[str, dict[str, Any]] = {}

    # ── Baseline ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if truth.abstention_expected:
        base_row = _empty_score(instance, truth, mode=BASELINE_MODE)
        base_row.update(_bundle_metrics(core, query, [], truth, k))
    else:
        primary = core._search_messages_decomposed(
            query, limit=k, per_query_limit=max(20, k * 4), use_cross_encoder=True,
        )
        base_row = _build_scored_row(instance, truth, _hits_from_results(primary), mode=BASELINE_MODE, k=k)
        base_row.update(_bundle_metrics(core, query, primary, truth, k))
    base_row["search_s"] = time.perf_counter() - t0
    rows[BASELINE_MODE] = base_row

    # ── Traversal ────────────────────────────────────────────────────────
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if truth.abstention_expected:
        trav_row = _empty_score(instance, truth, mode=TRAVERSAL_MODE)
        trav_row.update(_bundle_metrics(core, query, [], truth, k))
        primary = []
    else:
        primary = search_messages_traversal(
            core, query, limit=k, seed_limit=50, hop_limit=1,
            max_per_session=2, use_cross_encoder=True, timings=timings,
        )
        trav_row = _build_scored_row(instance, truth, _hits_from_results(primary), mode=TRAVERSAL_MODE, k=k)
        trav_row.update(_bundle_metrics(core, query, primary, truth, k))
    trav_row["search_s"] = time.perf_counter() - t0
    trav_row.update(timings)
    trav_row["candidates"] = len(primary) - min(len(primary), k) if primary else 0
    rows[TRAVERSAL_MODE] = trav_row
    return rows


def _graph_stats(core: MemoryCore) -> dict[str, Any]:
    nodes = core._db.raw_query("SELECT COUNT(*) AS c FROM _graph_nodes")
    edges = core._db.raw_query("SELECT type, COUNT(*) AS c FROM _graph_edges GROUP BY type")
    return {
        "graph_nodes": nodes[0]["c"] if nodes else 0,
        "graph_edges": sum(e["c"] for e in edges),
        "edges_by_type": dict(Counter({e["type"]: e["c"] for e in edges})),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="LongMemEval S JSON file")
    parser.add_argument("--limit", type=int, help="Maximum number of questions to process")
    parser.add_argument("--resume", action="store_true", help="Skip questions already completed")
    parser.add_argument("--progress", action="store_true", help="Print per-question progress")
    parser.add_argument("--output", type=Path, default=Path("results/eval_graph_s.json"))
    parser.add_argument("--root", type=Path, default=Path(tempfile.mkdtemp(prefix="coremem-graph-s-")))
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

        scored = _score_question(core, instance, truth, k=args.k)
        stats = _graph_stats(core)
        total_s = time.perf_counter() - t_q

        for mode, row in scored.items():
            record = {
                "question_id": qid,
                "question_type": instance.question_type,
                "mode": mode,
                "messages": n_msgs,
                "sessions": len(instance.sessions),
                "ingest_s": ingest_s,
                "ingest_msgs_per_s": n_msgs / ingest_s if ingest_s else 0.0,
                "question_total_s": total_s,
                **stats,
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
            base = next(r for r in rows if r["question_id"] == qid and r["mode"] == BASELINE_MODE)
            trav = next(r for r in rows if r["question_id"] == qid and r["mode"] == TRAVERSAL_MODE)
            bs, ts = base["scoring"], trav["scoring"]
            print(
                f"[{processed:>3}] {instance.question_type:<26} {qid[:20]} "
                f"ingest {ingest_s:5.1f}s ({n_msgs} msgs) "
                f"base {base['search_s']:5.1f}s trav {trav['search_s']:5.1f}s "
                f"(graph {trav.get('graph_build_s', 0):4.1f}s) "
                f"recall {bs[f'session_recall@{args.k}']:.2f}/{ts[f'session_recall@{args.k}']:.2f} "
                f"msg {bs[f'message_recall@{args.k}']:.2f}/{ts[f'message_recall@{args.k}']:.2f}",
                flush=True,
            )

    # ── Aggregate ─────────────────────────────────────────────────────────
    by_mode: dict[str, list[dict[str, Any]]] = {BASELINE_MODE: [], TRAVERSAL_MODE: []}
    for row in rows:
        by_mode[row["mode"]].append(row)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    metrics: dict[str, dict[str, Any]] = {}
    for mode, mode_rows in by_mode.items():
        m: dict[str, Any] = {}
        for key in (
            f"session_recall@{args.k}", f"message_recall@{args.k}",
            f"session_hit@{args.k}", f"message_hit@{args.k}",
            "session_rr", "session_map", "empty_retrieval",
        ):
            m[key] = _mean([r["scoring"].get(key, 0) for r in mode_rows])
        m["bundle_message_recall"] = _mean([r.get("bundle_message_recall", 0) for r in mode_rows])
        m["bundle_message_hit"] = _mean([r.get("bundle_message_hit", 0) for r in mode_rows])
        for key in (
            "ingest_s", "ingest_msgs_per_s", "search_s", "seeds_s", "graph_build_s",
            "traverse_s", "rerank_s", "question_total_s", "graph_nodes", "graph_edges",
            "candidates", "bundle_context_chars",
        ):
            m[key] = _mean([r.get(key, 0) for r in mode_rows])
        m["n"] = len(mode_rows)
        metrics[mode] = m

    result = {
        "dataset": str(args.data),
        "k": args.k,
        "completed": len(completed),
        "total_time_s": time.perf_counter() - t_start,
        "metrics": metrics,
        "modes": by_mode,
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"{'metric':<28}{'baseline':>12}{'traversal':>12}")
    for key in (
        f"session_recall@{args.k}", f"message_recall@{args.k}",
        f"session_hit@{args.k}", f"message_hit@{args.k}",
        "session_rr", "session_map", "bundle_message_recall", "bundle_message_hit",
    ):
        b = metrics[BASELINE_MODE].get(key, 0)
        t = metrics[TRAVERSAL_MODE].get(key, 0)
        print(f"{key:<28}{b:>12.4f}{t:>12.4f}")
    print("\n=== Timing (mean per question) ===")
    print(f"{'phase':<28}{'baseline':>12}{'traversal':>12}")
    for key in ("ingest_s", "ingest_msgs_per_s", "search_s", "seeds_s", "graph_build_s",
                "traverse_s", "rerank_s", "question_total_s"):
        b = metrics[BASELINE_MODE].get(key, 0)
        t = metrics[TRAVERSAL_MODE].get(key, 0)
        print(f"{key:<28}{b:>12.3f}{t:>12.3f}")
    print(f"\nquestions completed: {len(completed)}, total time: {result['total_time_s']:.0f}s")
    print(f"results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
