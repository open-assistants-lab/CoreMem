"""Observer & Reflector pipeline evaluation using LongMemEval questions.

For each question, injects the answer sessions' messages, runs the
Observer (and optionally Reflector), and saves observations/reflections
with source messages for human verification.

Uses incremental save/resume — same pattern as eval.py.

Usage:
    # Observer only
    uv run python -m benchmarks.longmemeval.observer_eval \
        --data .../longmemeval_s_cleaned.json \
        --provider deepseek:deepseek-chat \
        --mode observer --limit 100 --output results/observer_100.json

    # Observer + Reflector
    uv run python -m benchmarks.longmemeval.observer_eval \
        --data .../longmemeval_s_cleaned.json \
        --provider deepseek:deepseek-chat \
        --mode both --limit 100 --output results/both_100.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from coremem import MemoryCore, MemoryStore
from coremem.backends.hybrid import HybridBackend
from coremem.observer import ObserverPipeline
from coremem.reflector import ReflectorPipeline


def _load_questions(
    data_path: str,
    question_types: list[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    with open(data_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = list(data.values())
    if question_types:
        data = [q for q in data if q.get("question_type") in question_types]
    if limit:
        data = data[:limit]
    return data


def _find_answer_sessions(q: dict) -> list[tuple[int, list[dict]]]:
    """Return (session_index, session_messages) for each answer session."""
    haystack_ids = q.get("haystack_session_ids", [])
    haystack = q.get("haystack_sessions", [])
    answer_ids = q.get("answer_session_ids", [])

    if isinstance(answer_ids, str):
        answer_ids = [answer_ids]

    results = []
    for aid in answer_ids:
        try:
            idx = haystack_ids.index(aid)
        except ValueError:
            continue
        if idx < len(haystack):
            results.append((idx, haystack[idx]))
    return results


def _save_progress(output_path: Path, data: dict) -> None:
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(output_path)


async def run_observer_eval(
    data_path: str | Path,
    provider: str = "ollama:gemma4:e4b",
    mode: str = "observer",
    question_types: list[str] | None = None,
    limit: int | None = None,
    verbose: bool = True,
    output: str | None = None,
    resume: str | None = None,
) -> dict[str, Any]:
    questions = _load_questions(str(data_path), question_types=question_types, limit=limit)

    if not questions:
        raise ValueError(f"No questions found in {data_path}")

    output_path = Path(output) if output else None

    existing_results: list[dict] = []
    completed_ids: set[str] = set()
    type_stats: dict[str, dict[str, int]] = {}

    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            existing_data = json.loads(resume_path.read_text())
            existing_results = existing_data.get("results", [])
            completed_ids = {r["question_id"] for r in existing_results}
            type_stats = existing_data.get("_type_stats", {})
            if verbose:
                print(f"Resuming: {len(completed_ids)} done, {len(questions) - len(completed_ids)} remaining")

    results = list(existing_results)

    if verbose and not resume:
        print(f"Provider: {provider}")
        print(f"Questions: {len(questions)}")
        if output_path:
            print(f"Output: {output_path}")
        print("-" * 60)

    start_time = time.time()

    for qi, q in enumerate(questions):
        q_id = q.get("question_id", f"q_{qi}")

        if q_id in completed_ids:
            if verbose:
                print(f"  [{qi+1}/{len(questions)}] {q_id}: SKIP (done)", flush=True)
            continue

        q_text = q.get("question", "")
        q_type = q.get("question_type", "unknown")
        answer_sessions = _find_answer_sessions(q)

        if not answer_sessions:
            if verbose:
                print(f"  [{qi+1}/{len(questions)}] {q_id}: SKIP (no answer sessions)", flush=True)
            completed_ids.add(q_id)
            continue

        all_observations: list[dict] = []
        session_details: list[dict] = []

        d1 = tempfile.mkdtemp()
        d2 = tempfile.mkdtemp()

        try:
            core = MemoryCore(backend=HybridBackend(path=d1))
            store = MemoryStore(path=d2)

            for si, (sess_idx, sess_messages) in enumerate(answer_sessions):
                sid = f"answer_session_{si}"
                for msg in sess_messages:
                    core.ingest(msg["role"], msg["content"], session_id=sid)

                pipeline = ObserverPipeline(
                    core=core, store=store, session_id=sid,
                    model=provider, token_threshold=1, min_turns=1,
                    tool_temp=0.1,
                    enable_gleaning=True,
                )
                try:
                    obs = await pipeline.after_turn()
                except Exception as e:
                    session_details.append({
                        "session_index": sess_idx,
                        "session_id": sid,
                        "error": str(e),
                        "observations": [],
                        "message_count": len(sess_messages),
                    })
                    continue

                session_obs = obs or []
                all_observations.extend(session_obs)

                session_details.append({
                    "session_index": sess_idx,
                    "session_id": sid,
                    "observations": [
                        {
                            "priority": o.get("priority", "?"),
                            "content": o.get("content", ""),
                            "source_quote": o.get("source_quote", ""),
                            "importance": o.get("importance", 0.5),
                            "entities": o.get("entities", []),
                        }
                        for o in session_obs
                    ],
                    "message_count": len(sess_messages),
                    "messages": [
                        {"role": m["role"], "content": m["content"][:300]}
                        for m in sess_messages[:5]
                    ],
                })

            # ── Reflector (if mode=both) ─────────────────
            reflections_result: dict[str, Any] | None = None
            if mode == "both":
                try:
                    reflector = ReflectorPipeline(
                        store=store, model=provider, min_observations=3,
                    )
                    refl = await reflector.run_now()
                    if refl:
                        reflections_result = {
                            "count": len(refl),
                            "items": [
                                {"domain": r.get("domain", "?"), "content": r.get("content", "")}
                                for r in refl
                            ],
                        }
                except Exception as e:
                    reflections_result = {"error": str(e)}

        finally:
            shutil.rmtree(d1, ignore_errors=True)
            shutil.rmtree(d2, ignore_errors=True)

        result = {
            "question_id": q_id,
            "question_type": q_type,
            "question": q_text,
            "answer_session_count": len(answer_sessions),
            "total_observations": len(all_observations),
            "sessions": session_details,
        }
        if reflections_result:
            result["reflections"] = reflections_result
        results.append(result)
        completed_ids.add(q_id)

        # Track per-type stats
        if q_type not in type_stats:
            type_stats[q_type] = {"total": 0, "observations": 0, "errors": 0}
        type_stats[q_type]["total"] += 1
        type_stats[q_type]["observations"] += len(all_observations)
        if any(s.get("error") for s in session_details):
            type_stats[q_type]["errors"] += 1

        if output_path:
            _save_progress(output_path, {
                "provider": provider,
                "results": results,
                "_type_stats": type_stats,
                "_completed": len(completed_ids),
                "_total": len(questions),
            })

        if verbose:
            parts = [f"  [{qi+1}/{len(questions)}] {q_id} ({q_type})"]
            parts.append(f"obs={len(all_observations)}")
            parts.append(f"sessions={len(answer_sessions)}")
            errors = sum(1 for s in session_details if s.get("error"))
            if errors:
                parts.append(f"errors={errors}")
            if output_path:
                parts.append(f"[saved {len(completed_ids)}/{len(questions)}]")
            print(" | ".join(parts), flush=True)

    elapsed = time.time() - start_time

    summary = {
        "provider": provider,
        "total_questions": len(results),
        "total_observations": sum(r["total_observations"] for r in results),
        "avg_obs_per_question": round(
            sum(r["total_observations"] for r in results) / max(len(results), 1), 1,
        ),
        "by_type": type_stats,
        "elapsed_s": round(elapsed, 1),
    }

    if verbose:
        print("-" * 60)
        for qt, s in sorted(type_stats.items()):
            avg = round(s["observations"] / s["total"], 1) if s["total"] else 0
            print(f"  {qt}: {s['total']} questions, {s['observations']} obs ({avg}/q), {s['errors']} errors")
        print(f"Total: {summary['total_observations']} obs, {summary['avg_obs_per_question']}/q, {elapsed:.0f}s")

    if output_path:
        _save_progress(output_path, {
            "provider": provider,
            "summary": summary,
            "results": results,
            "_type_stats": type_stats,
            "_completed": len(completed_ids),
            "_total": len(questions),
        })
        if verbose:
            print(f"Saved to {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Observer/Reflector pipeline LongMemEval eval")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--provider", type=str, default="deepseek:deepseek-chat")
    parser.add_argument("--mode", type=str, default="observer",
                        choices=["observer", "both"])
    parser.add_argument("--question-types", type=str, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    asyncio.run(run_observer_eval(
        data_path=args.data,
        provider=args.provider,
        mode=args.mode,
        question_types=args.question_types,
        limit=args.limit,
        verbose=True,
        output=args.output,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
