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


def run_retrieval_benchmark(
    data_path: str | Path,
    backend: str = "chroma",
    question_types: list[str] | None = None,
    limit: int | None = None,
    k: int = 5,
    verbose: bool = True,
    memory_base: str = "/tmp/coremem_bench",
) -> dict[str, Any]:
    questions = _load_questions(str(data_path), question_types=question_types, limit=limit)

    if not questions:
        raise ValueError(f"No questions found in {data_path}")

    results: list[dict] = []
    type_scores: dict[str, list[bool]] = {}

    if verbose:
        print(f"Backend: {backend}")
        print(f"Questions: {len(questions)}")
        print(f"Recall: R@{k}")
        print("-" * 60)

    start_time = time.time()

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

            t0 = time.time()
            search_results = core.search(q_text, limit=k)
            found = {r.memory.session_id for r in search_results}
            hits = found & set(answer_sids)
            is_hit = len(hits) > 0
            search_time = time.time() - t0

            results.append({
                "question_id": q_id,
                "question_type": q_type,
                "recall": is_hit,
                "sessions_injected": len(haystack),
                "inject_time_s": round(inject_time, 3),
                "search_time_s": round(search_time, 4),
                "matches": sorted(hits),
            })

            if q_type not in type_scores:
                type_scores[q_type] = []
            type_scores[q_type].append(is_hit)

            if verbose:
                status = f"HIT ({len(hits)} of {len(answer_sids)})" if is_hit else "MISS"
                print(
                    f"  [{qi+1}/{len(questions)}] {q_id} ({q_type}): {status} "
                    f"| inject={inject_time:.1f}s search={search_time:.4f}s",
                    flush=True,
                )

        finally:
            shutil.rmtree(mem_path, ignore_errors=True)

    elapsed = time.time() - start_time
    total_hits = sum(r["recall"] for r in results)
    total = len(results) or 1
    overall = total_hits / total

    if verbose:
        print("-" * 60)
        print(f"Overall R@{k}: {overall:.1%} ({total_hits}/{total})")
        for qt, scores in sorted(type_scores.items()):
            s = sum(scores)
            print(f"  {qt}: {s}/{len(scores)} = {s/len(scores):.1%}")
        print(f"Time: {elapsed:.1f}s")

    return {
        "backend": backend,
        "k": k,
        "total": total,
        "hits": total_hits,
        "recall": overall,
        "by_type": {
            qt: {"hits": sum(s), "total": len(s), "recall": sum(s) / len(s) if s else 0}
            for qt, s in type_scores.items()
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
    args = parser.parse_args()

    run_retrieval_benchmark(
        data_path=args.data,
        backend=args.backend,
        question_types=args.question_types,
        limit=args.limit,
        k=args.k,
        verbose=True,
    )


if __name__ == "__main__":
    main()
