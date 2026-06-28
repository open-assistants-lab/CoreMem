#!/usr/bin/env python3
"""Stage 4: LLM AgentJournal compiler retrieval eval against LongMemEval.

Reuses the LongMemEval loader from Stage 2, calls the LLM compiler to
generate AgentJournal pages from reference turns, and scores retrieval.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from coremem.agent_journal import AgentJournalBundle, AgentJournalCompiler, AgentJournalLLMCompiler, AgentJournalSearch, CrossEncoderReranker

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_agent_journal_longmemeval import (  # noqa: E402
    _TURN_MESSAGES,
    PreparedInstance,
    PreparedSession,
    QuestionTruth,
    _aggregate_metrics,
    _empty_score,
    _first_rank,
    _fractional_recall,
    _fuzzy_tf,
    _stem_text,
    _terms,
    build_reference_bundle,
    load_longmemeval_instances,
    prepare_instances,
)

COMPILER_MODE = "agent-memory-llm-compiler-retrieval"


def _chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def _page_id(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^page_id:\s*([^\n]+)$", text, re.MULTILINE)
    if not match:
        return path.stem
    return match.group(1).strip().strip("\"'")


def _bm25(docs, terms, k1=1.5, b=0.75):
    N = len(docs)
    if N == 0 or not terms:
        return [(label, 0.0) for label, _ in docs]
    avgdl = sum(len(d.split()) for _, d in docs) / N
    df = {t: sum(1 for _, d in docs if t in d) for t in terms}
    idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in terms}
    scored = []
    for label, text in docs:
        words = text.split()
        doclen = len(words)
        score = 0.0
        for t in terms:
            if df[t] == 0:
                continue
            tf = _fuzzy_tf(t, words)
            score += idf[t] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doclen / avgdl))
        scored.append((label, score))
    return scored


def _search_compiled_pages(
    bundle: AgentJournalBundle, query: str, *, limit: int, search: AgentJournalSearch | None = None
) -> list[dict[str, Any]]:
    """BM25 + embedding hybrid search over compiled AgentJournal pages."""
    if search is not None:
        hits = search.search(query, limit=limit)
        return [{"page_id": _page_id(h.path), "score": h.score, "context_chars": 0} for h in hits]
    terms = _terms(query)
    if not terms:
        return []
    docs: list[tuple[str, str]] = []
    for path in sorted(bundle.pages_dir.rglob("*.md")):
        text = path.read_text(encoding="utf-8").casefold()
        pid = _page_id(path)
        docs.append((pid, _stem_text(text)))
    scored = _bm25(docs, terms)
    doc_map = {pid: t for pid, t in docs}
    hits = [{"page_id": pid, "score": s, "context_chars": len(doc_map[pid])} for pid, s in scored if s > 0]
    hits.sort(key=lambda x: (-x["score"], x["page_id"]))
    return hits[:limit]


def _session_messages(bundle: AgentJournalBundle, turn_id: str) -> list[dict[str, Any]]:
    msgs = _TURN_MESSAGES.get(turn_id, [])
    return [{"message_id": m.id, "role": m.role, "content": m.content} for m in msgs]


async def run_eval(
    data_path: str | Path,
    root: str | Path,
    *,
    k: int = 5,
    question_types: Sequence[str] | None = None,
    limit: int | None = None,
    reset: bool = False,
    model: str = "ollama-cloud:deepseek-v4-flash",
    max_retries: int = 3,
    resume: bool = False,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")

    root = Path(root)
    if reset and root.exists():
        _safe_reset_root(root)
    elif root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"bundle root already exists: {root}")

    raw_instances = load_longmemeval_instances(
        data_path, question_types=question_types, limit=limit
    )
    prepared, truth_by_question_id = prepare_instances(raw_instances)
    if resume and (root / "references" / "manifest.json").exists():
        from coremem.agent_journal import AgentJournalBundle as _MB
        bundle = _MB(root)
    else:
        bundle = build_reference_bundle(root, prepared)

    llm_compiler = AgentJournalLLMCompiler(bundle, model=model, max_retries=max_retries)
    compile_errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    sem = asyncio.Semaphore(8)

    async def _compile_one(session):
        messages = _session_messages(bundle, session.turn_id)
        if not messages:
            return None
        async with sem:
            cache_entry = llm_compiler._check_cache(session.turn_id, messages)
            if cache_entry is not None:
                return llm_compiler.compiler.apply_plan(cache_entry)
            turn_content = llm_compiler._format_turn(session.turn_id, session.session_id, messages)
            plan = await llm_compiler._generate_plan(turn_content, session.turn_id, session.session_id, messages)
            llm_compiler._save_cache(session.turn_id, messages, plan)
            return llm_compiler.compiler.apply_plan(plan)

    for instance in prepared:
        tasks: list[asyncio.Task] = []
        for session in instance.sessions:
            if session.session_id in seen:
                continue
            seen.add(session.session_id)
            if resume and llm_compiler.has_cache(session.turn_id):
                skipped += 1
                continue
            tasks.append(asyncio.create_task(_compile_one(session)))
        if not tasks:
            continue
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for session, result in zip(
            [s for s in instance.sessions if s.session_id in seen], results
        ):
            if result is None:
                compile_errors.append({
                    "question_id": instance.question_id,
                    "session_id": session.session_id,
                    "error": "no messages found in reference turn",
                })
            elif isinstance(result, Exception):
                compile_errors.append({
                    "question_id": instance.question_id,
                    "session_id": session.session_id,
                    "error": str(result),
                })

    lint_errors = bundle.lint()
    bundle.rebuild_embeddings()
    reranker = CrossEncoderReranker()
    search = AgentJournalSearch(bundle.root, embedding_index=bundle.embedding_index, reranker=reranker)
    rows = [
        _score_instance(bundle, instance, truth_by_question_id[instance.question_id], k=k, search=search)
        for instance in prepared
    ]

    return {
        "dataset": str(data_path),
        "mode": COMPILER_MODE,
        "k": k,
        "model": model,
        "lint": {"passed": not lint_errors, "errors": lint_errors},
        "compile": {
            "planned": len(seen),
            "succeeded": len(seen) - len(compile_errors) - skipped,
            "skipped": skipped,
            "errors": compile_errors,
        },
        "bundle": {
            "root": str(root),
            "reference_turn_count": sum(len(instance.sessions) for instance in prepared),
            "page_count": len(list(bundle.pages_dir.rglob("*.md"))),
        },
        "results": rows,
        "metrics": _aggregate_metrics(rows, k=k),
    }


def _score_instance(
    bundle: AgentJournalBundle,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
    search: AgentJournalSearch | None = None,
) -> dict[str, Any]:
    if truth.abstention_expected:
        return _empty_score(instance, truth, mode=COMPILER_MODE)
    hits = _search_compiled_pages(bundle, instance.query, limit=k, search=search)
    retrieved_page_ids = [h["page_id"] for h in hits]
    context_chars = sum(h["context_chars"] for h in hits)
    empty_retrieval = not hits

    expected = list(truth.expected_session_ids)
    session_rank = _first_rank(retrieved_page_ids, expected)
    session_recall = _fractional_recall(retrieved_page_ids, expected)
    abstention_false_positive = truth.abstention_expected and not empty_retrieval

    relevant = set(expected)
    top_k = retrieved_page_ids[:k]
    session_precision = sum(1 for s in top_k if s in relevant) / max(k, 1) if top_k else 0.0

    ap = 0.0
    relevant_found = 0
    for i, pid in enumerate(retrieved_page_ids):
        if pid in relevant:
            relevant_found += 1
            ap += relevant_found / (i + 1)
    session_map = ap / len(relevant) if relevant else 0.0

    return {
        "question_id": instance.question_id,
        "question_type": instance.question_type,
        "query": instance.query,
        "mode": COMPILER_MODE,
        "retrieved_page_ids": retrieved_page_ids,
        "retrieved_scores": [h["score"] for h in hits],
        "retrieved_turn_ids": [],
        "retrieved_session_ids": [],
        "retrieved_message_ids": [],
        "context_chars": context_chars,
        "used_reference_search": False,
        "top_score": hits[0]["score"] if hits else 0.0,
        "scoring": {
            "expected_session_ids": expected,
            "expected_message_ids": [],
            f"session_recall@{k}": session_recall,
            f"message_recall@{k}": 0.0,
            f"session_precision@{k}": round(session_precision, 3),
            f"session_hit@{k}": session_rank is not None,
            f"message_hit@{k}": False,
            "session_rank": session_rank,
            "message_rank": None,
            "session_rr": round(1.0 / session_rank, 3) if session_rank else 0.0,
            "message_rr": 0.0,
            "session_map": round(session_map, 3),
            "abstention_expected": truth.abstention_expected,
            "abstention_false_positive": abstention_false_positive,
            "empty_retrieval": empty_retrieval,
        },
    }


def _safe_reset_root(root: Path) -> None:
    resolved = root.resolve()
    if resolved in {Path.cwd().resolve(), Path.home().resolve(), Path("/").resolve()}:
        raise ValueError(f"refusing to overwrite unsafe root: {root}")
    if root.exists() and any(root.iterdir()):
        markers = [root / "SCHEMA.md", root / "references" / "manifest.json"]
        if not all(marker.exists() for marker in markers):
            raise ValueError(f"refusing to overwrite non-AgentJournal directory: {root}")
    shutil.rmtree(root)


def _print_result(result: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(dict(result), indent=2, sort_keys=True, default=str))
        return
    metrics = result["metrics"]
    k = result["k"]
    lint = "pass" if result["lint"]["passed"] else "fail"
    compile_ok = result["compile"]
    print(f"model: {result.get('model', '?')}")
    print(f"lint: {lint}")
    print(f"compile planned={compile_ok['planned']} succeeded={compile_ok['succeeded']} skipped={compile_ok.get('skipped', 0)}")
    print(f"questions: {metrics['question_count']}")
    print(f"session_recall@{k}: {metrics[f'session_recall@{k}']}")
    print(f"session_mrr: {metrics['session_mrr']}")
    print(f"empty_retrieval_rate: {metrics['empty_retrieval_rate']}")
    print(f"abstention_false_positive_rate: {metrics['abstention_false_positive_rate']}")


async def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage 4: LLM AgentJournal compiler retrieval eval against LongMemEval",
    )
    parser.add_argument("data", type=Path, help="Local LongMemEval-shaped JSON file")
    parser.add_argument("--root", type=Path, help="AgentJournal bundle root to write")
    parser.add_argument("--k", type=int, default=5, help="Retrieval cutoff")
    parser.add_argument("--limit", type=int, help="Maximum number of instances to load")
    parser.add_argument(
        "--model", default="ollama-cloud:deepseek-v4-flash",
        help="LLM model string (default: ollama-cloud:deepseek-v4-flash)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Maximum LLM retries per session (default: 3)",
    )
    parser.add_argument(
        "--question-type", action="append", dest="question_types",
        help="Question type filter; can be passed more than once",
    )
    parser.add_argument("--output", type=Path, help="Write full structured JSON result")
    parser.add_argument("--json", action="store_true", help="Print the full structured JSON result")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Delete --root before writing. Temp roots are always fresh.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-cached sessions on restart",
    )
    args = parser.parse_args(argv)

    if args.root is None:
        with tempfile.TemporaryDirectory(prefix="agent-memory-llm-compiler-eval-") as tmp:
            result = await run_eval(
                args.data, Path(tmp),
                k=args.k, question_types=args.question_types, limit=args.limit,
                model=args.model, max_retries=args.max_retries,
                resume=args.resume,
            )
            if args.output:
                args.output.write_text(
                    json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8",
                )
            _print_result(result, as_json=args.json)
    else:
        if args.root.exists() and any(args.root.iterdir()) and not args.overwrite and not args.resume:
            parser.error(f"--root already exists; pass --overwrite to replace it: {args.root}")
        result = await run_eval(
            args.data, args.root,
            k=args.k, question_types=args.question_types, limit=args.limit,
            reset=args.overwrite,
            model=args.model, max_retries=args.max_retries,
            resume=args.resume,
        )
        if args.output:
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                encoding="utf-8",
            )
        _print_result(result, as_json=args.json)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
