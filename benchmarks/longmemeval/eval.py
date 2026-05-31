"""LongMemEval retrieval benchmark runner for coremem.

Measures Recall@K without any LLM involvement. Two backends supported:
  --backend chroma  → ChromaBackend (baseline, target 95%+)
  --backend hybrid  → HybridBackend (enhanced, requires hybriddb)

Dataset format (LongMemEval):
  Each question has: question_id, question_type, question, answer_session_ids,
  haystack_session_ids, haystack_sessions.
  haystack_session_ids[i] maps to haystack_sessions[i].

Injection: sessions are injected in batch, tagged as session_{i:04d}.

Recall check: answer_session_ids[aid] → find aid in haystack_session_ids → get
index → our injected id session_{index:04d} → check if in top-K results.

Usage:
    uv run python -m coremem.benchmarks.longmemeval.eval \
        --data /tmp/lme_cache/.../longmemeval_s_cleaned.json \
        --backend chroma \
        --limit 20 \
        --k 5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from coremem.core import MemoryCore


def _load_questions(data_path: str, question_types: list[str] | None = None, limit: int | None = None) -> list[dict]:
    with open(data_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = list(data.values())

    if question_types:
        data = [q for q in data if q.get("question_type") in question_types]

    if limit:
        data = data[:limit]

    return data


def _setup_backend(backend: str, path: str):
    if backend == "chroma":
        from coremem.backends.chroma import ChromaBackend

        return ChromaBackend(path=path)
    elif backend == "hybrid":
        from coremem.backends.hybrid import HybridBackend

        return HybridBackend(path=path)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'chroma' or 'hybrid'.")


def _map_answer_sids(
    haystack_session_ids: list[str],
    answer_session_ids: list[str] | str,
) -> list[str]:
    if isinstance(answer_session_ids, str):
        answer_session_ids = [answer_session_ids]

    id_to_index = {hid: idx for idx, hid in enumerate(haystack_session_ids)}
    result = []
    for aid in answer_session_ids:
        idx = id_to_index.get(aid)
        if idx is not None:
            result.append(f"session_{idx:04d}")
    return result


def _inject_sessions_batch(core: MemoryCore, haystack_sessions: list) -> float:
    """Inject all sessions using batch insert. Returns time taken."""
    from coremem.types import Memory

    t0 = time.time()
    batch: list[Memory] = []
    for si, session_messages in enumerate(haystack_sessions):
        sid = f"session_{si:04d}"
        for msg in session_messages:
            batch.append(Memory(
                id="",
                content=msg.get("content", ""),
                role=msg.get("role", "user"),
                session_id=sid,
                user_id=msg.get("user_id", ""),
                agent_id=msg.get("agent_id", ""),
            ))
    if batch:
        core.store(batch)
    return time.time() - t0


def _find_rank(results: list[Any], answer_sids: list[str]) -> int:
    """Return the 1-based rank of the first matching answer session, or -1."""
    for idx, r in enumerate(results):
        if r.memory.session_id in answer_sids:
            return idx + 1
    return -1


def _run_search(core: MemoryCore, query: str, limit: int, mode: str = "search") -> tuple[list[Any], float]:
    """Run search with given mode and return (results, time)."""
    t0 = time.time()
    if mode == "search_enhanced":
        results = core.search_enhanced(query, limit=limit)
    else:
        results = core.search(query, limit=limit)
    return results, time.time() - t0


def _score_results(
    results: list[Any], answer_sids: list[str],
) -> tuple[bool, int]:
    """Score: (is_hit, 1-based rank or -1 for miss)."""
    found = {r.memory.session_id for r in results}
    hits = found & set(answer_sids)
    rank = _find_rank(results, answer_sids)
    return len(hits) > 0, rank


def run_retrieval_benchmark(
    data_path: str | Path,
    backend: str = "chroma",
    question_types: list[str] | None = None,
    limit: int | None = None,
    k: int = 5,
    verbose: bool = True,
    memory_base: str = "/tmp/coremem_bench",
    search_mode: str = "both",
) -> dict[str, Any]:
    questions = _load_questions(str(data_path), question_types=question_types, limit=limit)

    if not questions:
        raise ValueError(f"No questions found in {data_path}")

    results: list[dict] = []
    type_scores: dict[str, dict[str, list[bool]]] = {}

    if verbose:
        print(f"Backend: {backend}")
        print(f"Questions: {len(questions)}")
        print(f"Recall: R@{k}")
        print(f"Search modes: {search_mode}")
        print("-" * 60)

    start_time = time.time()
    modes = ["search", "search_enhanced"] if search_mode == "both" else [search_mode]

    for qi, q in enumerate(questions):
        q_id = q.get("question_id", f"q_{qi}")
        q_text = q.get("question", "")
        q_type = q.get("question_type", "unknown")
        haystack_ids = q.get("haystack_session_ids", [])
        haystack = q.get("haystack_sessions", [])

        answer_sids = _map_answer_sids(haystack_ids, q.get("answer_session_ids", []))
        if not answer_sids:
            if verbose:
                print(f"  [{qi+1}/{len(questions)}] {q_id}: SKIP (no answer IDs mapped)", flush=True)
            continue

        mem_path = f"{memory_base}_{q_id}_{os.getpid()}"
        be = _setup_backend(backend, mem_path)
        core = MemoryCore(backend=be)

        try:
            inject_time = _inject_sessions_batch(core, haystack)

            mode_stats: dict[str, Any] = {}
            for mode in modes:
                search_results, search_time = _run_search(core, q_text, k, mode)
                is_hit, rank = _score_results(search_results, answer_sids)

                mode_stats[mode] = {
                    "recall": is_hit,
                    "rank": rank,
                    "search_time_s": round(search_time, 4),
                }

                if q_type not in type_scores:
                    type_scores[q_type] = {}
                if mode not in type_scores[q_type]:
                    type_scores[q_type][mode] = []
                type_scores[q_type][mode].append(is_hit)

            result = {
                "question_id": q_id,
                "question_type": q_type,
                "sessions_injected": len(haystack),
                "answer_session_ids": sorted(answer_sids),
                "inject_time_s": round(inject_time, 3),
                "modes": mode_stats,
            }
            results.append(result)

            if verbose:
                parts = [f"  [{qi+1}/{len(questions)}] {q_id} ({q_type})"]
                for mode in modes:
                    ms = mode_stats[mode]
                    hit_str = "HIT" if ms["recall"] else "MISS"
                    rank_str = f"rank={ms['rank']}" if ms["rank"] > 0 else "rank=N/A"
                    parts.append(f"{mode}: {hit_str} ({rank_str} | {ms['search_time_s']:.4f}s)")
                parts.append(f"inject={inject_time:.1f}s")
                print(" | ".join(parts), flush=True)

        finally:
            shutil.rmtree(mem_path, ignore_errors=True)

    elapsed = time.time() - start_time

    # Compute per-mode summary
    per_mode: dict[str, dict] = {}
    for m in modes:
        hits = sum(1 for r in results if r["modes"][m]["recall"])
        total = len(results) or 1
        ranks = [r["modes"][m]["rank"] for r in results]
        mrr = sum(1.0 / r for r in ranks if r > 0) / total
        rank1 = sum(1 for r in ranks if r == 1) / total
        per_mode[m] = {
            "hits": hits, "total": total,
            "recall": hits / total,
            "mrr": mrr,
            "rank_at_1": rank1,
            "mean_rank": sum(r for r in ranks if r > 0) / max(hits, 1),
        }

    if verbose:
        print("-" * 60)
        for m in modes:
            s = per_mode[m]
            print(
                f"Overall {m} R@{k}: {s['recall']:.1%} ({s['hits']}/{s['total']}) "
                f"| MRR={s['mrr']:.3f} | Rank@1={s['rank_at_1']:.1%} "
                f"| MeanRank={s['mean_rank']:.1f}"
            )
        for qt in sorted(type_scores):
            for m in modes:
                scores = type_scores[qt].get(m, [])
                if scores:
                    s = sum(scores)
                    print(f"  {qt} ({m}): {s}/{len(scores)} = {s/len(scores):.1%}")
        print(f"Time: {elapsed:.1f}s")

    return {
        "backend": backend,
        "k": k,
        "per_mode": per_mode,
        "by_type": {
            qt: {
                m: {
                    "hits": sum(s), "total": len(s),
                    "recall": sum(s) / len(s) if s else 0,
                }
                for m, s in modes_dict.items()
            }
            for qt, modes_dict in type_scores.items()
        },
        "results": results,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="coremem LongMemEval retrieval benchmark")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to longmemeval_s_cleaned.json")
    parser.add_argument("--backend", type=str, default="chroma",
                        choices=["chroma", "hybrid"])
    parser.add_argument("--question-types", type=str, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--search-mode", type=str, default="both",
                        choices=["search", "search_enhanced", "both"])
    args = parser.parse_args()

    run_retrieval_benchmark(
        data_path=args.data,
        backend=args.backend,
        question_types=args.question_types,
        limit=args.limit,
        k=args.k,
        verbose=True,
        search_mode=args.search_mode,
    )


if __name__ == "__main__":
    main()
