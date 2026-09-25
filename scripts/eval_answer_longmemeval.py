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

from coremem import MemoryCore
from coremem.reading import build_facts_digest
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
    "episodic_cap2",
    "memorycore_llm_expansion",
    # Reading-strategy ablation: identical episodic_4k_reranked context,
    # answered with Chain-of-Note (extract-then-reason) instead of a direct
    # answer prompt. _json additionally serializes items as structured JSON
    # (LongMemEval §5.5: CoN+JSON is the strongest reader configuration).
    "episodic_4k_reranked_con",
    "episodic_4k_reranked_con_json",
    # Phase 2 kill-gate: identical bundles context + budget-capped [FACTS]
    # digest (extracted at ingest via extract_facts=True). Tests whether a
    # governed facts section lifts answer accuracy or just adds noise.
    "episodic_4k_reranked_factdigest",
    # Fact-augmented key expansion (LongMemEval §5.3): a second core instance
    # ingests with LLM-extracted user facts prepended to embedded documents.
    # Same retrieval chain as episodic_4k_reranked; only the index differs.
    "episodic_4k_reranked_factaug",
)

_CON_MODES = frozenset({"episodic_4k_reranked_con", "episodic_4k_reranked_con_json"})


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
        anchor_ids = set(bundle.anchor_ids or [])
        ordered = sorted(
            bundle.messages,
            key=lambda m: (m.id not in anchor_ids, m.ts or datetime.min.replace(tzinfo=UTC)),
        )
        message_parts = []
        for message in ordered:
            date = message.ts.date().isoformat() if message.ts else "unknown"
            message_parts.append(f"[{date} {message.role}] {message.content}")
        parts.append(
            f"[session={bundle.session_id} complete={str(bundle.complete).lower()}]\n"
            + "\n".join(message_parts)
        )
    return "\n\n".join(parts)


def _format_bundles_without_headers(bundles: list[Any]) -> str:
    parts = []
    for bundle in bundles:
        anchor_ids = set(bundle.anchor_ids or [])
        ordered = sorted(
            bundle.messages,
            key=lambda m: (m.id not in anchor_ids, m.ts or datetime.min.replace(tzinfo=UTC)),
        )
        parts.extend(
            f"[{message.ts.date().isoformat() if message.ts else 'unknown'} {message.role}] "
            f"{message.content}"
            for message in ordered
        )
    return "\n\n".join(parts)


def _ordered_bundle_messages(bundle: Any) -> list[Any]:
    anchor_ids = set(bundle.anchor_ids or [])
    return sorted(
        bundle.messages,
        key=lambda m: (m.id not in anchor_ids, m.ts or datetime.min.replace(tzinfo=UTC)),
    )


def _format_bundles_json(bundles: list[Any]) -> str:
    """Serialize bundles as structured JSON (LongMemEval reading format)."""
    sessions = []
    for bundle in bundles:
        sessions.append({
            "session_id": bundle.session_id,
            "complete": bool(bundle.complete),
            "messages": [
                {
                    "date": message.ts.date().isoformat() if message.ts else "unknown",
                    "role": message.role,
                    "content": message.content,
                }
                for message in _ordered_bundle_messages(bundle)
            ],
        })
    return json.dumps(sessions, ensure_ascii=False)


def _answer(provider: Any, question: str, context: str) -> str:
    return _chat(provider, (
        "Answer the question using only the memory context. Resolve comparisons, counts, "
        "and date differences when the context supports them. If the context is insufficient, "
        "say that the information is insufficient. Give a concise direct answer.\n\n"
        f"Question:\n{question}\n\nMemory context:\n{context}"
    ))


def _answer_con(provider: Any, question: str, context: str) -> str:
    """Chain-of-Note reading: extract evidence as notes first, then reason (LongMemEval §5.5)."""
    return _chat(provider, (
        "Answer the question using only the memory context. Work step by step: first go "
        "through each memory item and write a brief note of any information relevant to "
        "the question, then reason over your notes to reach the final answer. Resolve "
        "comparisons, counts, and date differences when the context supports them. If the "
        "context is insufficient, say that the information is insufficient. End your reply "
        "with a final line of the exact form: Final answer: <concise answer>\n\n"
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
    reuse: bool = False,
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
        core = None
        core_factaug = None
        try:
            instance_root = root / f"{index:04d}_{instance.question_id[:8]}"
            if not (reuse and (instance_root / "hybrid").exists()):
                shutil.rmtree(instance_root, ignore_errors=True)
                core = build_memorycore(
                    instance_root,
                    [instance],
                    llm_provider=answer_provider,
                    extract_facts=True,  # Phase 2: facts table for the factdigest arm
                )
            else:
                core = MemoryCore(
                    path=str(instance_root / "hybrid"),
                    llm_provider=answer_provider,
                )

            # Fact-augmented key expansion: a second instance ingests the same
            # haystack with LLM-extracted user facts prepended to embedded docs.
            # Built lazily only when the mode is active (saves ingest + LLM cost).
            core_factaug = None
            if "episodic_4k_reranked_factaug" in MODES:
                fa_root = instance_root / "fa"
                if reuse and (fa_root / "hybrid").exists():
                    core_factaug = MemoryCore(
                        path=str(fa_root / "hybrid"),
                        llm_provider=answer_provider,
                        fact_augment=True,
                    )
                else:
                    shutil.rmtree(fa_root, ignore_errors=True)
                    core_factaug = build_memorycore(
                        fa_root,
                        [instance],
                        llm_provider=answer_provider,
                        fact_augment=True,
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
            # Reading-strategy ablation reuses this exact context; only the answer
            # prompt (and for _json, the serialization) differs.
            contexts["episodic_4k_reranked_con"] = contexts["episodic_4k_reranked"]
            contexts["episodic_4k_reranked_con_json"] = _format_bundles_json(reranked_bundles)
            retrieval_seconds["episodic_4k_reranked_con"] = retrieval_seconds["episodic_4k_reranked"]
            retrieval_seconds["episodic_4k_reranked_con_json"] = retrieval_seconds["episodic_4k_reranked"]
            # Phase 2 kill-gate: same bundles + budget-capped [FACTS] digest.
            # Empty digest -> context identical to control (paired by construction).
            _digest = build_facts_digest(core.list_facts(limit=200))
            contexts["episodic_4k_reranked_factdigest"] = (
                contexts["episodic_4k_reranked"] + "\n\n" + _digest
                if _digest else contexts["episodic_4k_reranked"]
            )
            retrieval_seconds["episodic_4k_reranked_factdigest"] = retrieval_seconds["episodic_4k_reranked"]

            if core_factaug is not None:
                started = time.perf_counter()
                fa_primary = core_factaug._search_messages_decomposed(
                    instance.query,
                    limit=5,
                    per_query_limit=20,
                    use_cross_encoder=True,
                )
                fa_bundles = core_factaug._reconstruct_sessions(
                    instance.query,
                    session_limit=5,
                    max_context_chars=4_000,
                    primary_results=fa_primary,
                )
                retrieval_seconds["episodic_4k_reranked_factaug"] = time.perf_counter() - started
                contexts["episodic_4k_reranked_factaug"] = _format_bundles(fa_bundles)

            started = time.perf_counter()
            cap2_primary = core._search_messages_decomposed(
                instance.query,
                limit=5,
                per_query_limit=20,
                use_cross_encoder=True,
                session_cap=2,
            )
            cap2_bundles = core._reconstruct_sessions(
                instance.query,
                session_limit=5,
                max_context_chars=16_000,
                primary_results=cap2_primary,
            )
            retrieval_seconds["episodic_cap2"] = time.perf_counter() - started
            contexts["episodic_cap2"] = _format_bundles(cap2_bundles)

            started = time.perf_counter()
            deep = core._search_messages_llm_expansion(instance.query, limit=5)
            retrieval_seconds["memorycore_llm_expansion"] = time.perf_counter() - started
            contexts["memorycore_llm_expansion"] = _format_messages([result.memory for result in deep])

            answers: dict[str, str] = {}
            answer_seconds: dict[str, float] = {}
            for mode in MODES:
                started = time.perf_counter()
                if mode in _CON_MODES:
                    answers[mode] = _answer_con(answer_provider, instance.query, contexts[mode])
                else:
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
        finally:
            # finally, not the happy path: each question builds 1-2 cores, and a
            # provider or ingest failure must not leak a pooled Chroma client
            # (2 with the factaug arm) — see MemoryCore.close. Results are
            # already computed when this runs, so closing cannot affect the
            # measurement. Best-effort by design: cleanup never fails the eval.
            for handle in (core, core_factaug):
                if handle is None:
                    continue
                try:
                    handle.close()
                except Exception:  # noqa: BLE001
                    pass
        if not reuse:
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
    parser.add_argument("--reuse", action="store_true", help="Reuse existing per-question hybrid dirs (skip re-ingest)")
    args = parser.parse_args()
    result = run(
        args.data,
        args.output,
        args.root,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        limit=args.limit,
        resume=args.resume,
        reuse=args.reuse,
    )
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
