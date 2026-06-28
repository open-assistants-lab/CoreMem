#!/usr/bin/env python3
"""Deterministic LongMemEval eval — BM25 baseline + MemoryCore search modes.

Modes:
  raw_bm25           BM25 over in-memory messages (original baseline)
  memorycore         ingest via MemoryCore, search_messages()
  memorycore_deep    ingest via MemoryCore, search_messages_deep()
  memorycore_journal ingest + compile journal, search_journal()
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coremem import MemoryCore
from coremem.agent_journal import AgentJournalBundle, CrossEncoderReranker
from coremem.providers import create_provider
from coremem.types import Memory

_TURN_MESSAGES: dict[str, tuple[Memory, ...]] = {}

GROUND_TRUTH_FIELDS = {"answer", "answer_session_ids", "has_answer"}
MODES = ("raw_bm25", "memorycore", "memorycore_deep", "memorycore_journal")
JOURNAL_COMPILERS = ("none", "verbatim", "llm")
STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "back", "be", "because", "been", "being",
    "but", "by", "can", "come", "could", "did", "do", "does", "done",
    "down", "each", "end", "even", "few", "for", "from", "further",
    "get", "go", "got", "had", "has", "have", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "just", "like", "made", "make", "may", "me", "might", "more",
    "most", "much", "must", "my", "no", "nor", "not", "now", "of",
    "on", "only", "or", "other", "our", "out", "over", "own", "put",
    "round", "said", "same", "see", "shall", "she", "should", "show",
    "side", "since", "so", "some", "such", "take", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this",
    "through", "to", "too", "top", "under", "until", "up", "upon", "us",
    "very", "was", "way", "we", "well", "were", "what", "when",
    "where", "which", "while", "who", "why", "will", "with", "would",
    "yes", "yet", "you", "your",
}


@dataclass(frozen=True)
class QuestionTruth:
    expected_session_ids: tuple[str, ...]
    expected_message_ids: tuple[str, ...]
    abstention_expected: bool


@dataclass(frozen=True)
class PreparedSession:
    turn_id: str
    session_id: str
    messages: tuple[Memory, ...]


@dataclass(frozen=True)
class PreparedInstance:
    question_id: str
    question_type: str
    query: str
    stripped: Mapping[str, Any]
    sessions: tuple[PreparedSession, ...]


@dataclass(frozen=True)
class ReferenceMessage:
    turn_id: str
    session_id: str
    message_id: str
    role: str
    content: str


@dataclass(frozen=True)
class RawSearchHit:
    message: ReferenceMessage
    score: float


def load_longmemeval_instances(
    data_path: str | Path,
    *,
    question_types: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load LongMemEval-shaped JSON without downloading anything."""
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        instances = data
    elif isinstance(data, dict) and isinstance(data.get("questions"), list):
        instances = data["questions"]
    elif isinstance(data, dict):
        instances = list(data.values())
    else:
        raise ValueError("LongMemEval input must be a list or object of instances")

    if not all(isinstance(item, dict) for item in instances):
        raise ValueError("LongMemEval instances must be JSON objects")

    result = list(instances)
    if question_types:
        allowed = set(question_types)
        result = [item for item in result if item.get("question_type") in allowed]
    if limit is not None:
        result = result[:limit]
    return result


def prepare_instances(
    raw_instances: Sequence[Mapping[str, Any]],
) -> tuple[list[PreparedInstance], dict[str, QuestionTruth]]:
    """Split raw instances into stripped retrieval input and private scorer truth."""
    prepared: list[PreparedInstance] = []
    truth_by_question_id: dict[str, QuestionTruth] = {}
    for index, instance in enumerate(raw_instances):
        prepared_instance, truth = _prepare_instance(instance, index)
        prepared.append(prepared_instance)
        truth_by_question_id[prepared_instance.question_id] = truth
    return prepared, truth_by_question_id


def build_reference_bundle(root: str | Path, instances: Sequence[PreparedInstance]) -> AgentJournalBundle:
    """Store sessions in-memory for the evaluation."""
    bundle = AgentJournalBundle(root)
    bundle.initialize()
    _TURN_MESSAGES.clear()
    for instance in instances:
        for session in instance.sessions:
            _TURN_MESSAGES[session.turn_id] = tuple(session.messages)
    return bundle


def build_memorycore(
    root: str | Path,
    instances: Sequence[PreparedInstance],
    *,
    journal_reranker: CrossEncoderReranker | None = None,
) -> MemoryCore:
    """Build a MemoryCore with the provided LongMemEval haystack messages."""
    core = MemoryCore(path=str(root / "hybrid"))
    if journal_reranker is not None:
        core._reranker = journal_reranker  # noqa: SLF001 - eval harness reuses the loaded model across questions.
        core._agent_journal_search._reranker = journal_reranker  # noqa: SLF001
    for instance in instances:
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
    return core


def compile_eval_journal(core: MemoryCore, instance: PreparedInstance, *, compiler: str = "verbatim") -> dict[str, Any]:
    """Compile per-session journal pages for LongMemEval journal retrieval."""
    if compiler not in JOURNAL_COMPILERS:
        raise ValueError(f"unknown journal compiler: {compiler}; choose from {JOURNAL_COMPILERS}")
    if compiler == "none":
        return {"compiler": compiler, "pages": 0, "errors": []}
    if compiler == "llm":
        summary = asyncio.run(core.compile_uncompiled_turns(limit=max(1, len(instance.sessions))))
        compiled = summary.get("compiled", [])
        errors = summary.get("errors", [])
        return {"compiler": compiler, "pages": len(compiled), "errors": errors}

    journal_root = core._agent_journal_root  # noqa: SLF001 - eval harness writes fixture pages directly.
    pages_dir = journal_root / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for session in instance.sessions:
        page_id = _safe_identifier(session.session_id)
        path = pages_dir / f"{page_id}.md"
        path.write_text(_render_eval_journal_page(instance, session), encoding="utf-8")
        written += 1
    return {"compiler": compiler, "pages": written, "errors": []}


def _render_eval_journal_page(instance: PreparedInstance, session: PreparedSession) -> str:
    frontmatter = [
        "---",
        f"title: LongMemEval Session {session.session_id}",
        "memory_kind: transcript",
        "scope: user",
        f"question_id: {instance.question_id}",
        f"session_id: {session.session_id}",
        f"turn_id: {session.turn_id}",
        "agent_journal_version: eval-verbatim-v1",
        "---",
        "",
    ]
    lines = [
        *frontmatter,
        f"# LongMemEval Session {session.session_id}",
        "",
        f"Question ID: {instance.question_id}",
        f"Session ID: {session.session_id}",
        f"Source Turn ID: {session.turn_id}",
        "",
        "## Messages",
        "",
    ]
    for msg in session.messages:
        lines.extend([
            f"### {msg.id} ({msg.role})",
            "",
            f"Source message ID: {msg.id}",
            f"Role: {msg.role}",
            "",
            msg.content,
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _search_messages_mode(core: MemoryCore, query: str, k: int) -> list[RawSearchHit]:
    results = core.search_messages(query, limit=k)
    return [
        RawSearchHit(
            message=ReferenceMessage(
                turn_id=r.memory.id or "",
                session_id=r.memory.session_id or "",
                message_id=r.memory.id or "",
                role=r.memory.role,
                content=r.memory.content,
            ),
            score=r.score,
        )
        for r in results
    ]


def _search_messages_deep_mode(core: MemoryCore, query: str, k: int) -> list[RawSearchHit]:
    results = core.search_messages_deep(query, limit=k)
    return [
        RawSearchHit(
            message=ReferenceMessage(
                turn_id=r.memory.id or "",
                session_id=r.memory.session_id or "",
                message_id=r.memory.id or "",
                role=r.memory.role,
                content=r.memory.content,
            ),
            score=r.score,
        )
        for r in results
    ]


def run_eval(
    data_path: str | Path,
    root: str | Path,
    *,
    mode: str = "all",
    k: int = 5,
    question_types: Sequence[str] | None = None,
    limit: int | None = None,
    reset: bool = False,
    expand_model: str | None = None,
    compile_journal: bool = False,
    journal_compiler: str = "verbatim",
    resume: bool = False,
    resume_path: str | Path | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Load data, run canonical per-question LongMemEval retrieval, and score modes."""
    if k <= 0:
        raise ValueError("k must be positive")
    if mode not in MODES and mode != "all":
        raise ValueError(f"unknown mode: {mode}; choose from {MODES} or 'all'")
    if journal_compiler not in JOURNAL_COMPILERS:
        raise ValueError(f"unknown journal compiler: {journal_compiler}; choose from {JOURNAL_COMPILERS}")
    if compile_journal:
        journal_compiler = "llm"

    root = Path(root)
    if reset and root.exists():
        _safe_reset_root(root)
    elif root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"bundle root already exists: {root}")

    raw_instances = load_longmemeval_instances(data_path, question_types=question_types, limit=limit)
    prepared, truth_by_question_id = prepare_instances(raw_instances)

    llm_provider = create_provider(expand_model) if expand_model else None

    # ── BM25 baseline (separate bundle, in-memory messages) ──
    if mode == "raw_bm25":
        bundle = build_reference_bundle(root, prepared)
        lint_errors = bundle.lint()
        rows = [
            _score_instance_bm25(bundle, instance, truth_by_question_id[instance.question_id],
                                 k=k, llm_provider=llm_provider)
            for instance in prepared
        ]
        result = {
            "dataset": str(data_path),
            "evaluation_scope": "per_question_haystack",
            "mode": "raw_bm25",
            "k": k,
            "lint": {"passed": not lint_errors, "errors": lint_errors},
            "bundle": {
                "root": str(root),
                "reference_turn_count": sum(len(instance.sessions) for instance in prepared),
                "page_count": len(list(bundle.pages_dir.rglob("*.md"))) if bundle.pages_dir.exists() else 0,
            },
            "results": rows,
            "metrics": _aggregate_metrics(rows, k=k),
        }
        if resume_path is not None:
            _write_resume_checkpoint(Path(resume_path), result, complete=True)
        return result

    # ── MemoryCore modes: canonical LongMemEval per-question haystack injection ──
    active_modes = MODES if mode == "all" else (mode,)
    active_modes = tuple(m for m in active_modes if m != "raw_bm25")
    mode_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in active_modes}
    completed_question_ids: set[str] = set()
    journal_reranker = CrossEncoderReranker() if "memorycore_journal" in active_modes else None

    if resume:
        if resume_path is None:
            raise ValueError("resume=True requires resume_path")
        state = _load_resume_checkpoint(Path(resume_path))
        if state is not None:
            _validate_resume_checkpoint(
                state,
                data_path=str(data_path),
                mode=mode,
                k=k,
                active_modes=active_modes,
                journal_compiler=journal_compiler,
            )
            completed_question_ids = set(str(qid) for qid in state.get("completed_question_ids", []))
            state_modes = state.get("modes", {})
            if isinstance(state_modes, Mapping):
                for m in active_modes:
                    mode_data = state_modes.get(m, {})
                    if isinstance(mode_data, Mapping):
                        mode_rows[m] = [
                            dict(row)
                            for row in mode_data.get("results", [])
                            if isinstance(row, Mapping)
                        ]

    for index, instance in enumerate(prepared):
        if instance.question_id in completed_question_ids:
            if progress:
                print(f"[{index + 1}/{len(prepared)}] {instance.question_id}: resumed", flush=True)
            continue
        instance_root = root / "instances" / f"{index:04d}_{_safe_identifier(instance.question_id)}"
        if instance_root.exists():
            _safe_reset_root(instance_root)
        if progress:
            print(f"[{index + 1}/{len(prepared)}] {instance.question_id}: running", flush=True)
        core = build_memorycore(
            instance_root,
            instances=(instance,),
            journal_reranker=journal_reranker,
        )
        journal_summary: dict[str, Any] | None = None
        if "memorycore_journal" in active_modes:
            journal_summary = compile_eval_journal(core, instance, compiler=journal_compiler)
        truth = truth_by_question_id[instance.question_id]

        for m in active_modes:
            if m == "memorycore":
                row = _score_instance_memorycore(core, instance, truth, k=k, deep=False)
            elif m == "memorycore_deep":
                row = _score_instance_memorycore(core, instance, truth, k=k, deep=True)
            elif m == "memorycore_journal":
                row = _score_instance_journal(core, instance, truth, k=k)
                row["journal_compile"] = journal_summary or {"compiler": journal_compiler, "pages": 0, "errors": []}
            else:
                continue
            mode_rows[m].append(row)
        completed_question_ids.add(instance.question_id)
        if resume_path is not None:
            checkpoint = _build_mode_result(
                data_path=str(data_path),
                mode=mode,
                k=k,
                mode_rows=mode_rows,
                completed_question_ids=completed_question_ids,
                complete=False,
                journal_compiler=journal_compiler,
            )
            _write_resume_checkpoint(Path(resume_path), checkpoint, complete=False)

    result = _build_mode_result(
        data_path=str(data_path),
        mode=mode,
        k=k,
        mode_rows=mode_rows,
        completed_question_ids=completed_question_ids,
        complete=True,
        journal_compiler=journal_compiler,
    )
    if resume_path is not None:
        _write_resume_checkpoint(Path(resume_path), result, complete=True)
    return result


def _prepare_instance(raw: Mapping[str, Any], instance_index: int) -> tuple[PreparedInstance, QuestionTruth]:
    stripped = _strip_ground_truth(raw)
    question_id = _string_field(stripped, "question_id", f"question_{instance_index:04d}")
    question_type = _string_field(stripped, "question_type", "unknown")
    query = _string_field(stripped, "question", "")
    raw_session_ids = _session_ids(stripped)
    public_session_ids = {
        raw_session_id: f"lme_{instance_index:04d}_session_{session_index:04d}"
        for session_index, raw_session_id in enumerate(raw_session_ids)
    }
    dates = _haystack_dates(stripped)
    sessions_raw = stripped.get("haystack_sessions", [])
    if not isinstance(sessions_raw, list):
        raise ValueError(f"{question_id}: haystack_sessions must be a list")

    expected_message_ids: list[str] = []
    sessions: list[PreparedSession] = []
    for session_index, session_raw in enumerate(sessions_raw):
        if not isinstance(session_raw, list):
            raise ValueError(f"{question_id}: haystack_sessions[{session_index}] must be a list")
        raw_session_id = raw_session_ids[session_index] if session_index < len(raw_session_ids) else f"session_{session_index:04d}"
        session_id = public_session_ids.get(raw_session_id, f"lme_{instance_index:04d}_session_{session_index:04d}")
        turn_id = _turn_id(instance_index, session_id, session_index)
        ts = _parse_date(dates[session_index] if session_index < len(dates) else None)
        messages: list[Memory] = []
        for message_index, message_raw in enumerate(session_raw):
            if not isinstance(message_raw, Mapping):
                raise ValueError(
                    f"{question_id}: haystack_sessions[{session_index}][{message_index}] must be an object",
                )
            role = _normalize_role(str(message_raw.get("role", "user")))
            content = str(message_raw.get("content", ""))
            message_id = _message_id(session_id, message_index, role)
            messages.append(
                Memory(
                    id=message_id,
                    role=role,
                    content=content,
                    ts=ts,
                    session_id=session_id,
                    user_id="",
                    agent_id="",
                ),
            )
            if _has_answer(raw, session_index, message_index):
                expected_message_ids.append(message_id)
        sessions.append(PreparedSession(turn_id=turn_id, session_id=session_id, messages=tuple(messages)))

    expected_session_ids = tuple(
        public_session_ids[session_id]
        for session_id in _as_str_list(raw.get("answer_session_ids"))
        if session_id in public_session_ids
    )
    abstention_expected = question_id.endswith("_abs") or question_type.endswith("_abs") or (
        not expected_session_ids and not expected_message_ids
    )
    if abstention_expected:
        expected_session_ids = ()
        expected_message_ids = []
    return PreparedInstance(
        question_id=question_id,
        question_type=question_type,
        query=query,
        stripped=stripped,
        sessions=tuple(sessions),
    ), QuestionTruth(
            expected_session_ids=expected_session_ids,
            expected_message_ids=tuple(expected_message_ids),
            abstention_expected=abstention_expected,
    )


def _strip_ground_truth(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_ground_truth(item)
            for key, item in value.items()
            if str(key) not in GROUND_TRUTH_FIELDS
        }
    if isinstance(value, list):
        return [_strip_ground_truth(item) for item in value]
    return value


def _empty_score(instance: PreparedInstance, truth: QuestionTruth, *, mode: str, k: int = 5) -> dict[str, Any]:
    return {
        "question_id": instance.question_id,
        "question_type": instance.question_type,
        "query": instance.query,
        "mode": mode,
        "retrieved_page_ids": [],
        "retrieved_scores": [],
        "retrieved_turn_ids": [],
        "retrieved_session_ids": [],
        "retrieved_message_ids": [],
        "context_chars": 0,
        "used_reference_search": mode == "raw_bm25",
        "top_score": 0.0,
        "scoring": {
            "expected_session_ids": list(truth.expected_session_ids),
            "expected_message_ids": list(truth.expected_message_ids),
            f"session_recall@{k}": 0.0,
            f"message_recall@{k}": 0.0,
            f"session_precision@{k}": 0.0,
            f"session_hit@{k}": False,
            f"message_hit@{k}": False,
            "session_rank": None,
            "message_rank": None,
            "session_rr": 0.0,
            "message_rr": 0.0,
            "session_map": 0.0,
            "abstention_expected": truth.abstention_expected,
            "abstention_false_positive": False,
            "empty_retrieval": True,
        },
    }


def _score_instance_bm25(
    bundle: AgentJournalBundle,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
    llm_provider=None,
) -> dict[str, Any]:
    turn_ids = {session.turn_id for session in instance.sessions}
    if truth.abstention_expected:
        return _empty_score(instance, truth, mode="raw_bm25")
    hits = _search_reference_messages(bundle, instance.query, allowed_turn_ids=turn_ids, limit=k, llm_provider=llm_provider)
    return _build_scored_row(instance, truth, hits, mode="raw_bm25", k=k)


def _score_instance_memorycore(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
    deep: bool = False,
) -> dict[str, Any]:
    mode = "memorycore_deep" if deep else "memorycore"
    if truth.abstention_expected:
        return _empty_score(instance, truth, mode=mode)
    search_fn = _search_messages_deep_mode if deep else _search_messages_mode
    hits = search_fn(core, instance.query, k)
    return _build_scored_row(instance, truth, hits, mode=mode, k=k)


def _score_instance_journal(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
) -> dict[str, Any]:
    mode = "memorycore_journal"
    if truth.abstention_expected:
        return _empty_score(instance, truth, mode=mode)
    page_hits = core.search_journal(instance.query, limit=k)
    messages_by_id = {
        msg.id: (session.turn_id, msg)
        for session in instance.sessions
        for msg in session.messages
    }
    seen_message_ids: set[str] = set()
    hits: list[RawSearchHit] = []
    for page in page_hits:
        page_text = _read_text(page.path)
        for message_id, (turn_id, msg) in messages_by_id.items():
            if message_id in seen_message_ids or message_id not in page_text:
                continue
            seen_message_ids.add(message_id)
            hits.append(
                RawSearchHit(
                    message=ReferenceMessage(
                        turn_id=turn_id,
                        session_id=msg.session_id or "",
                        message_id=message_id,
                        role=msg.role,
                        content=msg.content,
                    ),
                    score=page.score,
                )
            )
    row = _build_scored_row(instance, truth, hits, mode=mode, k=k)
    row["retrieved_page_ids"] = [p.path.name for p in page_hits]
    row["journal_hit_count"] = len(page_hits)
    row["journal_top_score"] = page_hits[0].score if page_hits else 0.0
    return row


def _build_scored_row(
    instance: PreparedInstance,
    truth: QuestionTruth,
    hits: list[RawSearchHit],
    *,
    mode: str,
    k: int,
) -> dict[str, Any]:
    retrieved_message_ids = [hit.message.message_id for hit in hits]
    retrieved_turn_ids = _unique(hit.message.turn_id for hit in hits)
    retrieved_session_ids = _unique(hit.message.session_id for hit in hits)
    context_chars = sum(len(hit.message.content) for hit in hits)

    expected_sessions = list(truth.expected_session_ids)
    expected_messages = list(truth.expected_message_ids)
    session_rank = _first_rank(retrieved_session_ids, expected_sessions)
    message_rank = _first_rank(retrieved_message_ids, expected_messages)
    session_recall = _fractional_recall(retrieved_session_ids, expected_sessions)
    message_recall = _fractional_recall(retrieved_message_ids, expected_messages)
    empty_retrieval = not hits
    abstention_false_positive = truth.abstention_expected and not empty_retrieval

    relevant = set(expected_sessions)
    top_k = retrieved_session_ids[:k]
    session_precision = sum(1 for s in top_k if s in relevant) / max(k, 1) if top_k else 0.0

    ap = 0.0
    relevant_found = 0
    for i, sid in enumerate(retrieved_session_ids):
        if sid in relevant:
            relevant_found += 1
            ap += relevant_found / (i + 1)
    session_map = ap / len(relevant) if relevant else 0.0

    return {
        "question_id": instance.question_id,
        "question_type": instance.question_type,
        "query": instance.query,
        "mode": mode,
        "retrieved_page_ids": [],
        "retrieved_turn_ids": retrieved_turn_ids,
        "retrieved_session_ids": retrieved_session_ids,
        "retrieved_message_ids": retrieved_message_ids,
        "retrieved_scores": [hit.score for hit in hits],
        "context_chars": context_chars,
        "top_score": hits[0].score if hits else 0.0,
        "scoring": {
            "expected_session_ids": expected_sessions,
            "expected_message_ids": expected_messages,
            f"session_recall@{k}": session_recall,
            f"message_recall@{k}": message_recall,
            f"session_precision@{k}": round(session_precision, 3),
            f"session_hit@{k}": session_rank is not None,
            f"message_hit@{k}": message_rank is not None,
            "session_rank": session_rank,
            "message_rank": message_rank,
            "session_rr": round(1.0 / session_rank, 3) if session_rank else 0.0,
            "message_rr": round(1.0 / message_rank, 3) if message_rank else 0.0,
            "session_map": round(session_map, 3),
            "abstention_expected": truth.abstention_expected,
            "abstention_false_positive": abstention_false_positive,
            "empty_retrieval": empty_retrieval,
        },
    }


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


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _fuzzy_tf(term: str, words: list[str], max_dist: int = 1) -> int:
    count = words.count(term)
    if max_dist > 0:
        for w in set(words):
            if w != term and len(w) >= 3 and _levenshtein(term, w) <= max_dist:
                count += words.count(w)
    return count


def _search_reference_messages(
    bundle: AgentJournalBundle,
    query: str,
    *,
    allowed_turn_ids: set[str],
    limit: int,
    llm_provider=None,
) -> list[RawSearchHit]:
    """BM25 search over reference messages (stopword-filtered, with query expansion)."""
    queries = _expand_queries(query, llm_provider=llm_provider)
    seen_ids: set[str] = set()
    merged: list[RawSearchHit] = []
    for q in queries:
        terms = _terms(q)
        if not terms:
            continue
        messages = _reference_messages(bundle, allowed_turn_ids=allowed_turn_ids)
        docs = [(m, _stem_text(m.content.casefold())) for m in messages]
        scored = _bm25(docs, terms)
        for m, s in scored:
            if s > 0 and m.message_id not in seen_ids:
                seen_ids.add(m.message_id)
                merged.append(RawSearchHit(message=m, score=s))
    merged.sort(key=lambda hit: (-hit.score, hit.message.turn_id, hit.message.message_id))
    return merged[:limit]


def _expand_queries(query: str, llm_provider=None) -> list[str]:
    queries = [query]
    if llm_provider is not None:
        try:
            import asyncio
            prompt = (
                "Rephrase this search query exactly 2 different ways to improve retrieval recall. "
                "Keep the original meaning. Return ONLY a JSON array of 2 strings.\n\n"
                f"Query: {query}\n\n"
                'Format: ["rephrase 1", "rephrase 2"]'
            )
            messages = [{"role": "user", "content": prompt}]
            result = asyncio.run(llm_provider.chat(messages))
            text = str(result.content if hasattr(result, "content") else result).strip()
            if text.startswith("["):
                variants = json.loads(text)
                if isinstance(variants, list) and all(isinstance(v, str) for v in variants):
                    seen = {query.lower()}
                    for v in variants:
                        if v.lower() not in seen:
                            seen.add(v.lower())
                            queries.append(v)
        except Exception:
            pass
    return queries


def _reference_messages(
    bundle: AgentJournalBundle,
    *,
    allowed_turn_ids: set[str] | None = None,
) -> list[ReferenceMessage]:
    messages: list[ReferenceMessage] = []
    for turn_id, turn_msgs in _TURN_MESSAGES.items():
        if allowed_turn_ids is not None and turn_id not in allowed_turn_ids:
            continue
        for msg in turn_msgs:
            session_id = msg.session_id or "default"
            messages.append(
                ReferenceMessage(
                    turn_id=turn_id,
                    session_id=session_id,
                    message_id=msg.id,
                    role=msg.role,
                    content=msg.content,
                ),
            )
    return messages


def _aggregate_metrics(rows: Sequence[Mapping[str, Any]], *, k: int) -> dict[str, Any]:
    scoring_rows = [row["scoring"] for row in rows if isinstance(row.get("scoring"), Mapping)]
    session_scored = [row for row in scoring_rows if row["expected_session_ids"]]
    message_scored = [row for row in scoring_rows if row["expected_message_ids"]]
    abstention_rows = [row for row in scoring_rows if row["abstention_expected"]]
    context_chars = [int(row["context_chars"]) for row in rows]
    return {
        "question_count": len(rows),
        "answerable_question_count": len([row for row in scoring_rows if not row["abstention_expected"]]),
        "abstention_question_count": len(abstention_rows),
        "session_scored_question_count": len(session_scored),
        "message_scored_question_count": len(message_scored),
        f"session_recall@{k}": _mean([float(row[f"session_recall@{k}"]) for row in session_scored]),
        f"message_recall@{k}": _mean([float(row[f"message_recall@{k}"]) for row in message_scored]),
        f"session_precision@{k}": _mean([float(row[f"session_precision@{k}"]) for row in session_scored]),
        f"session_hit@{k}": _rate(row[f"session_hit@{k}"] for row in session_scored),
        f"message_hit@{k}": _rate(row[f"message_hit@{k}"] for row in message_scored),
        "session_mrr": _mean([float(row["session_rr"]) for row in session_scored]),
        "message_mrr": _mean([float(row["message_rr"]) for row in message_scored]),
        "session_map": _mean([float(row["session_map"]) for row in session_scored]),
        "empty_retrieval_rate": _rate(row["empty_retrieval"] for row in scoring_rows),
        "abstention_false_positive_rate": _rate(
            row["abstention_false_positive"] for row in abstention_rows
        ),
        "context_chars_mean": _mean(context_chars),
        "context_chars_total": sum(context_chars),
        "by_question_type": _breakdown_by_question_type(rows, k=k),
    }


def _build_mode_result(
    *,
    data_path: str,
    mode: str,
    k: int,
    mode_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    completed_question_ids: set[str],
    complete: bool,
    journal_compiler: str,
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode_name, rows in mode_rows.items():
        row_list = [dict(row) for row in rows]
        modes[mode_name] = {
            "results": row_list,
            "metrics": _aggregate_metrics(row_list, k=k),
        }
    return {
        "dataset": data_path,
        "evaluation_scope": "per_question_haystack",
        "mode": mode,
        "journal_compiler": journal_compiler,
        "k": k,
        "complete": complete,
        "completed_question_ids": sorted(completed_question_ids),
        "modes": modes,
    }


def _load_resume_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_resume_checkpoint(path: Path, result: Mapping[str, Any], *, complete: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload["complete"] = complete
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _validate_resume_checkpoint(
    state: Mapping[str, Any],
    *,
    data_path: str,
    mode: str,
    k: int,
    active_modes: Sequence[str],
    journal_compiler: str,
) -> None:
    if state.get("dataset") != data_path:
        raise ValueError("resume checkpoint dataset does not match")
    if state.get("mode") != mode:
        raise ValueError("resume checkpoint mode does not match")
    if state.get("k") != k:
        raise ValueError("resume checkpoint k does not match")
    if state.get("journal_compiler") != journal_compiler:
        raise ValueError("resume checkpoint journal compiler does not match")
    state_modes = set((state.get("modes") or {}).keys())
    if state_modes != set(active_modes):
        raise ValueError("resume checkpoint modes do not match")


def _breakdown_by_question_type(rows: Sequence[Mapping[str, Any]], *, k: int) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["question_type"]), []).append(row)

    breakdown: dict[str, Any] = {}
    for question_type, group in sorted(grouped.items()):
        scoring_rows = [row["scoring"] for row in group if isinstance(row.get("scoring"), Mapping)]
        session_scored = [row for row in scoring_rows if row["expected_session_ids"]]
        message_scored = [row for row in scoring_rows if row["expected_message_ids"]]
        abstention_rows = [row for row in scoring_rows if row["abstention_expected"]]
        context_chars = [int(row["context_chars"]) for row in group]
        breakdown[question_type] = {
            "count": len(group),
            f"session_recall@{k}": _mean([float(row[f"session_recall@{k}"]) for row in session_scored]),
            f"message_recall@{k}": _mean([float(row[f"message_recall@{k}"]) for row in message_scored]),
            f"session_precision@{k}": _mean([float(row[f"session_precision@{k}"]) for row in session_scored]),
            f"session_hit@{k}": _rate(row[f"session_hit@{k}"] for row in session_scored),
            f"message_hit@{k}": _rate(row[f"message_hit@{k}"] for row in message_scored),
            "session_mrr": _mean([float(row["session_rr"]) for row in session_scored]),
            "message_mrr": _mean([float(row["message_rr"]) for row in message_scored]),
            "session_map": _mean([float(row["session_map"]) for row in session_scored]),
            "empty_retrieval_rate": _rate(row["empty_retrieval"] for row in scoring_rows),
            "abstention_false_positive_rate": _rate(
                row["abstention_false_positive"] for row in abstention_rows
            ),
            "context_chars_mean": _mean(context_chars),
        }
    return breakdown


def _session_ids(instance: Mapping[str, Any]) -> list[str]:
    values = instance.get("haystack_session_ids", [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def _haystack_dates(instance: Mapping[str, Any]) -> list[str | None]:
    values = instance.get("haystack_dates", [])
    if not isinstance(values, list):
        return []
    return [str(value) if value is not None else None for value in values]


def _has_answer(raw: Mapping[str, Any], session_index: int, message_index: int) -> bool:
    sessions = raw.get("haystack_sessions", [])
    if isinstance(sessions, list) and session_index < len(sessions):
        session = sessions[session_index]
        if isinstance(session, list) and message_index < len(session):
            message = session[message_index]
            if isinstance(message, Mapping) and isinstance(message.get("has_answer"), bool):
                return bool(message["has_answer"])

    flags = raw.get("has_answer")
    if isinstance(flags, list) and session_index < len(flags):
        session_flags = flags[session_index]
        if isinstance(session_flags, list) and message_index < len(session_flags):
            return bool(session_flags[message_index])
        if isinstance(session_flags, bool) and message_index == 0:
            return session_flags
    if isinstance(flags, Mapping):
        session_ids = raw.get("haystack_session_ids", [])
        keys = [str(session_index)]
        if isinstance(session_ids, list) and session_index < len(session_ids):
            keys.append(str(session_ids[session_index]))
        for key in keys:
            session_flags = flags.get(key)
            if isinstance(session_flags, list) and message_index < len(session_flags):
                return bool(session_flags[message_index])
            if isinstance(session_flags, bool) and message_index == 0:
                return session_flags
    return False


def _turn_id(instance_index: int, session_id: str, session_index: int) -> str:
    return f"lme_{instance_index:04d}__{session_index:04d}_{_safe_identifier(session_id)}"


def _message_id(session_id: str, message_index: int, role: str) -> str:
    return f"{_safe_identifier(session_id)}_turn_{message_index:04d}_{_safe_identifier(role)}"


def _safe_identifier(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("._-")
    return safe or "item"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_date(value: str | None) -> datetime:
    if value:
        try:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed
        except ValueError:
            pass
        cleaned = re.sub(r"\s*\([A-Za-z]{3}\)\s*", " ", value).strip()
        for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(cleaned, fmt).replace(tzinfo=UTC)
            except ValueError:
                continue
    return datetime(1970, 1, 1, tzinfo=UTC)


def _normalize_role(role: str) -> str:
    normalized = role.strip().casefold()
    if normalized in {"assistant", "ai", "bot"}:
        return "assistant"
    if normalized in {"tool", "tool_result", "tool-result"}:
        return "tool_result"
    if normalized in {"tool_call", "tool-call"}:
        return "tool_call"
    if normalized in {"system", "developer", "user"}:
        return normalized
    if normalized in {"human", "customer"}:
        return "user"
    return "user"


def _string_field(instance: Mapping[str, Any], key: str, default: str) -> str:
    value = instance.get(key, default)
    return value if isinstance(value, str) else default


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return []


def _terms(query: str) -> list[str]:
    return [_stem(t) for t in re.findall(r"\w+", query.casefold()) if t not in STOPWORDS]


_IRREGULAR = {
    "bought": "buy", "brought": "bring", "built": "build", "burnt": "burn",
    "came": "come", "did": "do", "drew": "draw", "drove": "drive",
    "fell": "fall", "flew": "fly", "forgot": "forget", "gave": "give",
    "gone": "go", "grew": "grow", "had": "have", "hid": "hide",
    "knew": "know", "laid": "lay", "led": "lead", "left": "leave",
    "lent": "lend", "lost": "lose", "made": "make", "meant": "mean",
    "met": "meet", "paid": "pay", "ran": "run", "rang": "ring",
    "rose": "rise", "said": "say", "sang": "sing", "sank": "sink",
    "sat": "sit", "slept": "sleep", "spoke": "speak", "spent": "spend",
    "stood": "stand", "stole": "steal", "struck": "strike", "swam": "swim",
    "took": "take", "taught": "teach", "tore": "tear", "told": "tell",
    "thought": "think", "threw": "throw", "understood": "understand",
    "woke": "wake", "wore": "wear", "won": "win", "wrote": "write",
    "got": "get", "ate": "eat", "drank": "drink", "drove": "drive",
    "rode": "ride", "saw": "see", "sent": "send", "shook": "shake",
    "shot": "shoot", "showed": "show", "shut": "shut", "sold": "sell",
    "sought": "seek", "sped": "speed", "spun": "spin", "split": "split",
    "spread": "spread", "stuck": "stick", "stung": "sting", "stank": "stink",
    "strode": "stride", "struck": "strike", "strung": "string",
    "swept": "sweep", "swelled": "swell", "swore": "swear", "swung": "swing",
    "tore": "tear", "threw": "throw", "thrust": "thrust", "trod": "tread",
    "understood": "understand", "undertook": "undertake", "upset": "upset",
    "woke": "wake", "waylaid": "waylay", "wept": "weep", "wound": "wind",
    "withdrew": "withdraw", "withheld": "withhold", "withstood": "withstand",
    "woke": "wake", "won": "win", "wound": "wind", "wring": "wring",
    "wrote": "write",
}


def _stem(word: str) -> str:
    word = word.casefold()
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if base.endswith("nn") or base.endswith("tt") or base.endswith("mm"):
            return base[:-1]
        return base
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if base.endswith("i"):
            return base[:-1] + "y"
        if base.endswith("nn") or base.endswith("tt"):
            return base[:-1]
        return base
    if word.endswith("ly"):
        return word[:-2]
    if word.endswith("er") and len(word) > 4:
        return word[:-2]
    if word.endswith("est") and len(word) > 5:
        return word[:-3]
    return word


def _stem_text(text: str) -> str:
    return " ".join(_stem(w) for w in text.split())


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _first_rank(actual: Sequence[str], expected: Sequence[str]) -> int | None:
    expected_set = set(expected)
    if not expected_set:
        return None
    for index, value in enumerate(actual, start=1):
        if value in expected_set:
            return index
    return None


def _fractional_recall(actual: Sequence[str], expected: Sequence[str]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    return round(len(set(actual) & expected_set) / len(expected_set), 3)


def _safe_reset_root(root: Path) -> None:
    resolved = root.resolve()
    if resolved in {Path.cwd().resolve(), Path.home().resolve(), Path("/").resolve()}:
        raise ValueError(f"refusing to overwrite unsafe root: {root}")
    if root.exists() and any(root.iterdir()):
        markers = [root / "SCHEMA.md", root / "references" / "manifest.json", root / "hybrid"]
        if not any(m.exists() for m in markers):
            raise ValueError(f"refusing to overwrite non-AgentJournal/MemoryCore directory: {root}")
    shutil.rmtree(root)


def _rate(values: Iterable[bool]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(1 for item in items if item) / len(items), 3)


def _mean(values: Sequence[int | float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _build_summary(modes_result: dict[str, Any], k: int) -> dict[str, Any]:
    """Build a side-by-side comparison table of key metrics across modes."""
    summary: dict[str, Any] = {}
    for mode_name, mode_data in modes_result.items():
        metrics = mode_data["metrics"]
        summary[mode_name] = {
            f"session_recall@{k}": metrics.get(f"session_recall@{k}", 0),
            f"message_recall@{k}": metrics.get(f"message_recall@{k}", 0),
            f"session_hit@{k}": metrics.get(f"session_hit@{k}", 0),
            f"message_hit@{k}": metrics.get(f"message_hit@{k}", 0),
            "session_mrr": metrics.get("session_mrr", 0),
            "session_map": metrics.get("session_map", 0),
            "empty_retrieval_rate": metrics.get("empty_retrieval_rate", 0),
            "abstention_false_positive_rate": metrics.get("abstention_false_positive_rate", 0),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic AgentJournal LongMemEval — BM25 baseline + MemoryCore modes",
    )
    parser.add_argument("data", type=Path, help="Local LongMemEval-shaped JSON file")
    parser.add_argument("--root", type=Path, help="AgentJournal bundle root to write")
    parser.add_argument("--mode", default="all", choices=("raw_bm25", "memorycore", "memorycore_deep", "memorycore_journal", "all"),
                        help="Search mode to run (default: all)")
    parser.add_argument("--k", type=int, default=5, help="Retrieval cutoff")
    parser.add_argument("--limit", type=int, help="Maximum number of instances to load")
    parser.add_argument(
        "--question-type",
        action="append",
        dest="question_types",
        help="Question type filter; can be passed more than once",
    )
    parser.add_argument("--output", type=Path, help="Write full structured JSON result")
    parser.add_argument("--jsonl-output", type=Path, help="Write one result row JSON object per line")
    parser.add_argument("--json", action="store_true", help="Print the full structured JSON result")
    parser.add_argument(
        "--include-scoring",
        action="store_true",
        help="Include private scoring ground truth in --json/--output. JSONL is always public.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete --root before writing. Temp roots are always fresh.",
    )
    parser.add_argument(
        "--expand-model",
        help="LLM model for query expansion (e.g. ollama-cloud:deepseek-v4-flash). Default: no expansion.",
    )
    parser.add_argument(
        "--compile-journal",
        action="store_true",
        help="Deprecated alias for --journal-compiler llm.",
    )
    parser.add_argument(
        "--journal-compiler",
        default="verbatim",
        choices=JOURNAL_COMPILERS,
        help="Journal compiler for memorycore_journal: verbatim is deterministic and LLM-free (default).",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the checkpoint file if present")
    parser.add_argument("--resume-path", type=Path, help="Checkpoint path for resumable runs")
    parser.add_argument("--progress", action="store_true", help="Print per-question progress")
    args = parser.parse_args(argv)

    resume_path = args.resume_path
    if resume_path is None and args.output is not None:
        resume_path = args.output.with_suffix(args.output.suffix + ".checkpoint.json")

    if args.root is None:
        with tempfile.TemporaryDirectory(prefix="agent-memory-longmemeval-") as tmp:
            result = run_eval(
                args.data, Path(tmp),
                mode=args.mode, k=args.k,
                question_types=args.question_types, limit=args.limit,
                expand_model=args.expand_model,
                compile_journal=args.compile_journal,
                journal_compiler=args.journal_compiler,
                resume=args.resume,
                resume_path=resume_path,
                progress=args.progress,
            )
            if "modes" in result:
                result["summary"] = _build_summary(result["modes"], args.k)
            _write_outputs(
                result, output=args.output, jsonl_output=args.jsonl_output,
                include_scoring=args.include_scoring,
            )
            _print_result(result, as_json=args.json, include_scoring=args.include_scoring)
    else:
        if args.root.exists() and any(args.root.iterdir()) and not args.overwrite and not args.resume:
            parser.error(f"--root already exists; pass --overwrite to replace it: {args.root}")
        result = run_eval(
            args.data, args.root,
            mode=args.mode, k=args.k,
            question_types=args.question_types, limit=args.limit,
            reset=args.overwrite, expand_model=args.expand_model,
            compile_journal=args.compile_journal,
            journal_compiler=args.journal_compiler,
            resume=args.resume,
            resume_path=resume_path,
            progress=args.progress,
        )
        if "modes" in result:
            result["summary"] = _build_summary(result["modes"], args.k)
        _write_outputs(
            result, output=args.output, jsonl_output=args.jsonl_output,
            include_scoring=args.include_scoring,
        )
        _print_result(result, as_json=args.json, include_scoring=args.include_scoring)

    return 0


def _write_outputs(
    result: Mapping[str, Any],
    *,
    output: Path | None,
    jsonl_output: Path | None,
    include_scoring: bool,
) -> None:
    if output is not None:
        output.write_text(
            json.dumps(_public_result(result, include_scoring=include_scoring), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if jsonl_output is not None:
        rows = result.get("results", [])
        modes = result.get("modes", {})
        if modes:
            all_rows: list[dict[str, Any]] = []
            for mode_name, mode_data in modes.items():
                for row in mode_data.get("results", []):
                    r = dict(row)
                    r["mode"] = mode_name
                    all_rows.append(r)
            jsonl_output.write_text(
                "".join(json.dumps(_public_row(r), sort_keys=True) + "\n" for r in all_rows),
                encoding="utf-8",
            )
        elif isinstance(rows, list):
            jsonl_output.write_text(
                "".join(json.dumps(_public_row(r), sort_keys=True) + "\n" for r in rows),
                encoding="utf-8",
            )


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a retrieval-log row without private scoring ground truth."""
    return {
        key: value
        for key, value in row.items()
        if key != "scoring"
    }


def _public_result(result: Mapping[str, Any], *, include_scoring: bool) -> dict[str, Any]:
    if include_scoring:
        return dict(result)
    public = dict(result)
    modes = public.get("modes")
    if isinstance(modes, Mapping):
        public["modes"] = {
            mode_name: {
                "results": [_public_row(r) if isinstance(r, Mapping) else r for r in mode_data.get("results", [])],
                "metrics": mode_data.get("metrics", {}),
            }
            for mode_name, mode_data in modes.items()
        }
        public.pop("results", None)
        return public
    rows = public.get("results", [])
    if isinstance(rows, list):
        public["results"] = [_public_row(row) if isinstance(row, Mapping) else row for row in rows]
    return public


def _print_result(result: Mapping[str, Any], *, as_json: bool, include_scoring: bool) -> None:
    if as_json:
        print(json.dumps(_public_result(result, include_scoring=include_scoring), indent=2, sort_keys=True))
        return

    modes = result.get("modes", {})
    k = result["k"]

    if modes:
        print(f"\n{'Metric':<30}", end="")
        for mode_name in modes:
            print(f"{mode_name:<22}", end="")
        print()
        print("-" * (30 + 22 * len(modes)))
        metric_keys = [
            f"session_recall@{k}", f"message_recall@{k}",
            f"session_hit@{k}", f"message_hit@{k}",
            "session_mrr", "session_map",
            "empty_retrieval_rate", "abstention_false_positive_rate",
        ]
        for key in metric_keys:
            print(f"{key:<30}", end="")
            for mode_name in modes:
                val = modes[mode_name]["metrics"].get(key, "")
                print(f"{val:<22}", end="")
            print()

        summary = result.get("summary")
        if summary:
            print(f"\nBest per metric:")
            for key in metric_keys:
                best = max(
                    (v for v in (summary[m].get(key, 0) for m in summary) if v is not None),
                    default=0,
                )
                winners = [m for m in summary if summary[m].get(key, 0) == best]
                print(f"  {key}: {best} ({', '.join(winners)})")
    else:
        metrics = result["metrics"]
        lint = "pass" if result.get("lint", {}).get("passed", True) else "fail"
        print(f"mode: {result.get('mode', '?')}")
        print(f"lint: {lint}")
        print(f"questions: {metrics['question_count']}")
        print(f"session_recall@{k}: {metrics[f'session_recall@{k}']}")
        print(f"message_recall@{k}: {metrics[f'message_recall@{k}']}")
        print(f"session_mrr: {metrics['session_mrr']}")
        print(f"empty_retrieval_rate: {metrics['empty_retrieval_rate']}")


if __name__ == "__main__":
    raise SystemExit(main())
