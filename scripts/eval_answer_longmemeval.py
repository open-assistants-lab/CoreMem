#!/usr/bin/env python3
"""Evaluate answer accuracy from MemoryCore retrieval contexts."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

from coremem.providers import create_provider
from eval_agent_journal_longmemeval import build_memorycore, load_longmemeval_instances, prepare_instances


MODES = (
    "memorycore",
    "decomposition_only",
    "reconstruction_only",
    "episodic_no_headers",
    "memorycore_episodic",
    "episodic_4k",
    "episodic_4k_reranked",
    "memorycore_llm_expansion",
)


def _chat(provider: Any, prompt: str) -> str:
    response = asyncio.run(provider.chat([{"role": "user", "content": prompt}]))
    return str(response.content if hasattr(response, "content") else response).strip()


def _format_messages(messages: list[Any]) -> str:
    parts = []
    for message in messages:
        date = message.ts.date().isoformat() if message.ts else "unknown"
        parts.append(
            f"[session={message.session_id or ''} date={date} role={message.role}]\n"
            f"{message.content}"
        )
    return "\n\n".join(parts)


def _format_bundles(bundles: list[Any]) -> str:
    parts = []
    for bundle in bundles:
        message_parts = []
        for message in bundle.messages:
            date = message.ts.date().isoformat() if message.ts else "unknown"
            message_parts.append(f"[{date} {message.role}] {message.content}")
        parts.append(
            f"[session={bundle.session_id} complete={str(bundle.complete).lower()}]\n"
            + "\n".join(message_parts)
        )
    return "\n\n".join(parts)


def _format_bundles_without_headers(bundles: list[Any]) -> str:
    return "\n\n".join(
        f"[{message.ts.date().isoformat() if message.ts else 'unknown'} {message.role}] "
        f"{message.content}"
        for bundle in bundles
        for message in bundle.messages
    )


def _answer(provider: Any, question: str, context: str) -> str:
    return _chat(provider, (
        "Answer the question using only the memory context. Resolve comparisons, counts, "
        "and date differences when the context supports them. If the context is insufficient, "
        "say that the information is insufficient. Give a concise direct answer.\n\n"
        f"Question:\n{question}\n\nMemory context:\n{context}"
    ))


def _judge(provider: Any, question: str, reference: str, answers: dict[str, str]) -> dict[str, Any]:
    ordered_modes = list(MODES)
    random.Random(question).shuffle(ordered_modes)
    anonymous = {f"answer_{index}": answers[mode] for index, mode in enumerate(ordered_modes)}
    prompt = (
        "Judge each candidate answer against the reference answer for the question. Accept "
        "paraphrases and equivalent calculations. For an unanswerable reference, accept only "
        "answers that correctly state the information is insufficient. Return ONLY JSON in "
        "this form: {\"answer_0\": {\"correct\": true, \"reason\": \"...\"}, ...}.\n\n"
        f"Question: {question}\nReference answer: {reference}\n"
        f"Candidate answers: {json.dumps(anonymous, ensure_ascii=False)}"
    )
    text = _chat(provider, prompt)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge returned non-JSON: {text}")
    judged = json.loads(match.group(0))
    return {
        mode: judged.get(f"answer_{index}", {"correct": False, "reason": "missing judgment"})
        for index, mode in enumerate(ordered_modes)
    }


def _metrics(rows: list[dict[str, Any]], mode: str) -> dict[str, float]:
    mode_rows = [row for row in rows if mode in row.get("answers", {})]
    answerable = [row for row in mode_rows if not row["question_id"].endswith("_abs")]
    abstention = [row for row in mode_rows if row["question_id"].endswith("_abs")]

    def rate(items: list[dict[str, Any]]) -> float:
        return round(sum(bool(row["judgments"][mode]["correct"]) for row in items) / len(items), 3) if items else 0.0

    return {
        "accuracy": rate(mode_rows),
        "answerable_accuracy": rate(answerable),
        "abstention_accuracy": rate(abstention),
        "context_chars_mean": round(sum(row["context_chars"][mode] for row in mode_rows) / len(mode_rows), 1),
        "retrieval_seconds_mean": round(sum(row["retrieval_seconds"][mode] for row in mode_rows) / len(mode_rows), 3),
        "answer_seconds_mean": round(sum(row["answer_seconds"][mode] for row in mode_rows) / len(mode_rows), 3),
    }


def run(
    data_path: Path,
    output: Path,
    root: Path,
    *,
    answer_model: str,
    judge_model: str,
    limit: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    raw = load_longmemeval_instances(data_path, limit=limit)
    prepared, _ = prepare_instances(raw)
    references = {str(row["question_id"]): str(row.get("answer", "")) for row in raw}
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    if resume and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        rows = list(previous.get("results", []))
        completed = {row["question_id"] for row in rows}

    answer_provider = create_provider(answer_model)
    judge_provider = create_provider(judge_model)
    root.mkdir(parents=True, exist_ok=True)

    for index, instance in enumerate(prepared):
        if instance.question_id in completed:
            print(f"[{index + 1}/{len(prepared)}] {instance.question_id}: resumed", flush=True)
            continue
        print(f"[{index + 1}/{len(prepared)}] {instance.question_id}: running", flush=True)
        instance_root = root / f"{index:04d}_{instance.question_id}"
        shutil.rmtree(instance_root, ignore_errors=True)
        core = build_memorycore(
            instance_root,
            [instance],
            llm_provider=answer_provider,
        )

        contexts: dict[str, str] = {}
        retrieval_seconds: dict[str, float] = {}

        started = time.perf_counter()
        basic = core._search_messages(instance.query, limit=5)
        retrieval_seconds["memorycore"] = time.perf_counter() - started
        contexts["memorycore"] = _format_messages([result.memory for result in basic])

        started = time.perf_counter()
        primary = core._search_messages_decomposed(instance.query, limit=5, per_query_limit=20)
        retrieval_seconds["decomposition_only"] = time.perf_counter() - started
        contexts["decomposition_only"] = _format_messages([result.memory for result in primary])

        started = time.perf_counter()
        basic_bundles = core._reconstruct_sessions(
            instance.query,
            session_limit=5,
            max_context_chars=16_000,
            primary_results=basic,
        )
        retrieval_seconds["reconstruction_only"] = time.perf_counter() - started
        contexts["reconstruction_only"] = _format_bundles(basic_bundles)

        started = time.perf_counter()
        bundles = core._reconstruct_sessions(
            instance.query,
            session_limit=5,
            max_context_chars=16_000,
            primary_results=primary,
        )
        episodic_retrieval_seconds = time.perf_counter() - started
        retrieval_seconds["episodic_no_headers"] = episodic_retrieval_seconds
        retrieval_seconds["memorycore_episodic"] = episodic_retrieval_seconds
        contexts["episodic_no_headers"] = _format_bundles_without_headers(bundles)
        contexts["memorycore_episodic"] = _format_bundles(bundles)

        started = time.perf_counter()
        small_bundles = core._reconstruct_sessions(
            instance.query,
            session_limit=5,
            max_context_chars=4_000,
            primary_results=primary,
        )
        retrieval_seconds["episodic_4k"] = time.perf_counter() - started
        contexts["episodic_4k"] = _format_bundles(small_bundles)

        started = time.perf_counter()
        reranked_primary = core._search_messages_decomposed(
            instance.query,
            limit=5,
            per_query_limit=20,
            use_cross_encoder=True,
        )
        reranked_bundles = core._reconstruct_sessions(
            instance.query,
            session_limit=5,
            max_context_chars=4_000,
            primary_results=reranked_primary,
        )
        retrieval_seconds["episodic_4k_reranked"] = time.perf_counter() - started
        contexts["episodic_4k_reranked"] = _format_bundles(reranked_bundles)

        started = time.perf_counter()
        deep = core._search_messages_llm_expansion(instance.query, limit=5)
        retrieval_seconds["memorycore_llm_expansion"] = time.perf_counter() - started
        contexts["memorycore_llm_expansion"] = _format_messages([result.memory for result in deep])

        answers: dict[str, str] = {}
        answer_seconds: dict[str, float] = {}
        for mode in MODES:
            started = time.perf_counter()
            answers[mode] = _answer(answer_provider, instance.query, contexts[mode])
            answer_seconds[mode] = time.perf_counter() - started
        judgments = _judge(
            judge_provider,
            instance.query,
            references[instance.question_id],
            answers,
        )
        rows.append({
            "question_id": instance.question_id,
            "question_type": instance.question_type,
            "question": instance.query,
            "reference_answer": references[instance.question_id],
            "answers": answers,
            "judgments": judgments,
            "context_chars": {mode: len(contexts[mode]) for mode in MODES},
            "retrieval_seconds": retrieval_seconds,
            "answer_seconds": answer_seconds,
        })
        result = {
            "answer_model": answer_model,
            "judge_model": judge_model,
            "results": rows,
            "metrics": {mode: _metrics(rows, mode) for mode in MODES},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        shutil.rmtree(instance_root, ignore_errors=True)

    return {
        "answer_model": answer_model,
        "judge_model": judge_model,
        "results": rows,
        "metrics": {mode: _metrics(rows, mode) for mode in MODES},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(
        args.data,
        args.output,
        args.root,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        limit=args.limit,
        resume=args.resume,
    )
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
