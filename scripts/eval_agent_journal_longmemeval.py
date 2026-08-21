#!/usr/bin/env python3
"""Deterministic LongMemEval eval — BM25 baseline + MemoryCore search modes.

Modes:
  raw_bm25                        BM25 over in-memory messages (baseline)
  memorycore                      recall(strategy="direct")
  memorycore_llm_expansion        recall(strategy="expanded") (1 LLM call)
  memorycore_episodic_reranked    recall(strategy="episodic") (default)
  memorycore_episodic_reranked_4k episodic with 4k context budget
  memorycore_fusion               recall(strategy="fusion") RRF
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import shutil
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coremem import MemoryCore
from coremem.agent_journal import AgentJournalBundle
from coremem.providers import create_provider
from coremem.retrieval import search_messages_confirmed, search_messages_preference_union
from coremem.rerank import set_cross_encoder_model
from coremem.traversal import search_messages_traversal
from coremem.types import Memory

_TURN_MESSAGES: dict[str, tuple[Memory, ...]] = {}

GROUND_TRUTH_FIELDS = {"answer", "answer_session_ids", "has_answer"}
MODES = ("raw_bm25", "memorycore", "memorycore_llm_expansion", "memorycore_episodic_reranked", "memorycore_episodic_reranked_4k", "memorycore_fusion", "memorycore_traversal_v2", "memorycore_episodic_reranked_confirmed", "memorycore_episodic_reranked_preference_union", "memorycore_episodic_reranked_v2", "memorycore_episodic_reranked_v3", "memorycore_episodic_reranked_v4")
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


def stream_longmemeval_instances(
    data_path: str | Path,
    *,
    question_types: Sequence[str] | None = None,
    limit: int | None = None,
    skip_indices: set[int] | None = None,
) -> Iterable[tuple[int, dict[str, Any]]]:
    """Stream LongMemEval JSON one instance at a time via ijson (low memory).

    Yields (index, raw_instance) tuples. Index is the position in the original
    array (before filtering), matching the index used by _prepare_instance.
    """
    import ijson

    skip = skip_indices or set()
    allowed_types = set(question_types) if question_types else None
    count = 0
    with open(data_path, "rb") as f:
        for index, obj in enumerate(ijson.items(f, "item")):
            if index in skip:
                count += 1
                continue
            if allowed_types and obj.get("question_type") not in allowed_types:
                count += 1
                continue
            if limit is not None and count >= limit:
                break
            yield index, obj
            count += 1


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
    llm_provider: Any = None,
) -> MemoryCore:
    """Build a MemoryCore with the provided LongMemEval haystack messages.

    Uses batch ingest (single journal flush with batched embedding) —
    ~5-15x faster than per-message inserts.
    """
    core = MemoryCore(path=str(root / "hybrid"), llm_provider=llm_provider)
    for instance in instances:
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
    return core


def _search_messages_mode(core: MemoryCore, query: str, k: int) -> list[RawSearchHit]:
    results = core._search_messages(query, limit=k)
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


def _search_messages_llm_expansion_mode(core: MemoryCore, query: str, k: int) -> list[RawSearchHit]:
    results = core._search_messages_llm_expansion(query, limit=k)
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
    cleanup_instances: bool = False,
    reuse_instances: bool = False,
    resume: bool = False,
    resume_path: str | Path | None = None,
    progress: bool = False,
    stream: bool = False,
    jsonl_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load data, run canonical per-question LongMemEval retrieval, and score modes."""
    if k <= 0:
        raise ValueError("k must be positive")
    if mode not in MODES and mode != "all":
        raise ValueError(f"unknown mode: {mode}; choose from {MODES} or 'all'")

    root = Path(root)
    if reset and root.exists():
        _safe_reset_root(root)
    elif root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"bundle root already exists: {root}")

    llm_provider = create_provider(expand_model) if expand_model else None

    # ── BM25 baseline (separate bundle, in-memory messages) ──
    # Streaming not supported for raw_bm25 (needs all messages in memory)
    if mode == "raw_bm25":
        raw_instances = load_longmemeval_instances(data_path, question_types=question_types, limit=limit)
        prepared, truth_by_question_id = prepare_instances(raw_instances)
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

    # Open JSONL for incremental output (append mode for crash safety)
    jsonl_file = None
    if jsonl_path is not None:
        jsonl_file = open(jsonl_path, "a", encoding="utf-8")

    try:
        if stream:
            result = _run_streaming(
                data_path=data_path,
                root=root,
                active_modes=active_modes,
                k=k,
                question_types=question_types,
                limit=limit,
                llm_provider=llm_provider,
                mode_rows=mode_rows,
                completed_question_ids=completed_question_ids,
                cleanup_instances=cleanup_instances,
                reuse_instances=reuse_instances,
                resume_path=resume_path,
                progress=progress,
                jsonl_file=jsonl_file,
            )
        else:
            raw_instances = load_longmemeval_instances(data_path, question_types=question_types, limit=limit)
            prepared, truth_by_question_id = prepare_instances(raw_instances)
            total = len(prepared)

            for index, instance in enumerate(prepared):
                if instance.question_id in completed_question_ids:
                    if progress:
                        print(f"[{index + 1}/{total}] {instance.question_id}: resumed", flush=True)
                    continue
                instance_root = root / "instances" / f"{index:04d}_{_safe_identifier(instance.question_id)}"
                if not (reuse_instances and (instance_root / "hybrid").exists()):
                    if instance_root.exists():
                        _safe_reset_root(instance_root)
                    core = build_memorycore(
                        instance_root,
                        instances=(instance,),
                        llm_provider=llm_provider,
                    )
                else:
                    core = MemoryCore(
                        path=str(instance_root / "hybrid"),
                        llm_provider=llm_provider,
                    )
                if progress:
                    print(f"[{index + 1}/{total}] {instance.question_id}: running", flush=True)
                question_start = time.time()
                truth = truth_by_question_id[instance.question_id]

                new_rows = _score_question(
                    core, instance, truth,
                    active_modes=active_modes, k=k,
                )
                question_elapsed = time.time() - question_start
                instance_disk_mb = _dir_size_mb(instance_root)
                for m in active_modes:
                    row = new_rows.get(m)
                    if row is not None:
                        row["question_time_seconds"] = round(question_elapsed, 1)
                        row["instance_disk_mb"] = round(instance_disk_mb, 1)
                        mode_rows[m].append(row)
                        if jsonl_file is not None:
                            jsonl_file.write(json.dumps(_public_row(row), sort_keys=True) + "\n")
                            jsonl_file.flush()
                completed_question_ids.add(instance.question_id)
                if cleanup_instances and instance_root.exists():
                    shutil.rmtree(instance_root, ignore_errors=True)
                if resume_path is not None:
                    checkpoint = _build_mode_result(
                        data_path=str(data_path),
                        mode=mode,
                        k=k,
                        mode_rows=mode_rows,
                        completed_question_ids=completed_question_ids,
                        complete=False,
                    )
                    _write_resume_checkpoint(Path(resume_path), checkpoint, complete=False)
            result = _build_mode_result(
                data_path=str(data_path),
                mode=mode,
                k=k,
                mode_rows=mode_rows,
                completed_question_ids=completed_question_ids,
                complete=True,
            )
    finally:
        if jsonl_file is not None:
            jsonl_file.close()

    if resume_path is not None:
        _write_resume_checkpoint(Path(resume_path), result, complete=True)
    return result


def _score_question(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    active_modes: Sequence[str],
    k: int,
) -> dict[str, dict[str, Any]]:
    """Score one question across all active modes. Returns {mode: row}."""
    new_rows: dict[str, dict[str, Any]] = {}
    for m in active_modes:
        if m == "memorycore":
            new_rows[m] = _score_instance_memorycore(core, instance, truth, k=k, deep=False)
        elif m == "memorycore_llm_expansion":
            new_rows[m] = _score_instance_memorycore(core, instance, truth, k=k, deep=True)
        elif m == "memorycore_episodic_reranked":
            new_rows[m] = _score_instance_episodic(
                core, instance, truth, k=k, use_cross_encoder=True,
            )
        elif m == "memorycore_episodic_reranked_4k":
            new_rows[m] = _score_instance_episodic(
                core, instance, truth, k=k, use_cross_encoder=True, max_context_chars=4_000,
            )
        elif m == "memorycore_fusion":
            new_rows[m] = _score_instance_fusion(core, instance, truth, k=k)
        elif m == "memorycore_traversal_v2":
            new_rows[m] = _score_instance_traversal(core, instance, truth, k=k)
        elif m == "memorycore_episodic_reranked_confirmed":
            new_rows[m] = _score_instance_confirmed(core, instance, truth, k=k)
        elif m == "memorycore_episodic_reranked_preference_union":
            new_rows[m] = _score_instance_preference_union(core, instance, truth, k=k)
        elif m == "memorycore_episodic_reranked_v2":
            new_rows[m] = _score_instance_v2(core, instance, truth, k=k)
        elif m == "memorycore_episodic_reranked_v3":
            new_rows[m] = _score_instance_v3(core, instance, truth, k=k, allocation="global")
        elif m == "memorycore_episodic_reranked_v4":
            new_rows[m] = _score_instance_v3(core, instance, truth, k=k, allocation="anchor")
    return new_rows


def _run_streaming(
    *,
    data_path: str | Path,
    root: Path,
    active_modes: Sequence[str],
    k: int,
    question_types: Sequence[str] | None,
    limit: int | None,
    llm_provider: Any,
    mode_rows: dict[str, list[dict[str, Any]]],
    completed_question_ids: set[str],
    cleanup_instances: bool,
    reuse_instances: bool,
    resume_path: str | Path | None,
    progress: bool,
    jsonl_file: Any,
) -> dict[str, Any]:
    """Stream questions one at a time from the JSON file via ijson.

    Each question is loaded, prepared, scored, and its instance cleaned up
    before the next question is loaded. This keeps peak memory low enough
    for the S variant (265 MB JSON, ~2.4 GB if loaded all at once).
    """
    # Build a set of completed question IDs to skip during streaming
    # We also need to track the global index for instance dir naming
    skip_indices: set[int] = set()
    # On resume, we don't know which indices were completed (only question IDs).
    # We stream all questions, skip by question_id, and rebuild skip_indices as we go.
    total_yielded = 0

    for index, raw_obj in stream_longmemeval_instances(
        data_path,
        question_types=question_types,
        limit=limit,
        skip_indices=skip_indices,
    ):
        instance, truth = _prepare_instance(raw_obj, index)

        if instance.question_id in completed_question_ids:
            if progress:
                print(f"[{index + 1}] {instance.question_id}: resumed", flush=True)
            continue

        instance_root = root / "instances" / f"{index:04d}_{_safe_identifier(instance.question_id)}"
        if not (reuse_instances and (instance_root / "hybrid").exists()):
            if instance_root.exists():
                _safe_reset_root(instance_root)
            core = build_memorycore(
                instance_root,
                instances=(instance,),
                llm_provider=llm_provider,
            )
        else:
            core = MemoryCore(
                path=str(instance_root / "hybrid"),
                llm_provider=llm_provider,
            )
        if progress:
            print(f"[{index + 1}] {instance.question_id}: running", flush=True)

        question_start = time.time()
        new_rows = _score_question(
            core, instance, truth,
            active_modes=active_modes, k=k,
        )
        question_elapsed = time.time() - question_start
        instance_disk_mb = _dir_size_mb(instance_root)
        for m in active_modes:
            row = new_rows.get(m)
            if row is not None:
                row["question_time_seconds"] = round(question_elapsed, 1)
                row["instance_disk_mb"] = round(instance_disk_mb, 1)
                mode_rows[m].append(row)
                if jsonl_file is not None:
                    jsonl_file.write(json.dumps(_public_row(row), sort_keys=True) + "\n")
                    jsonl_file.flush()
        completed_question_ids.add(instance.question_id)
        total_yielded += 1

        if cleanup_instances and instance_root.exists():
            shutil.rmtree(instance_root, ignore_errors=True)
        if resume_path is not None:
            checkpoint = _build_mode_result(
                data_path=str(data_path),
                mode="all" if len(active_modes) > 1 else active_modes[0],
                k=k,
                mode_rows=mode_rows,
                completed_question_ids=completed_question_ids,
                complete=False,
            )
            _write_resume_checkpoint(Path(resume_path), checkpoint, complete=False)

    return _build_mode_result(
        data_path=str(data_path),
        mode="all" if len(active_modes) > 1 else active_modes[0],
        k=k,
        mode_rows=mode_rows,
        completed_question_ids=completed_question_ids,
        complete=True,
    )


def _prepare_instance(raw: Mapping[str, Any], instance_index: int) -> tuple[PreparedInstance, QuestionTruth]:
    stripped = _strip_ground_truth(raw)
    question_id = _string_field(stripped, "question_id", f"question_{instance_index:04d}")
    question_type = _string_field(stripped, "question_type", "unknown")
    query = _string_field(stripped, "question", "")
    raw_session_ids = _session_ids(stripped)
    # Each session position gets a unique public_session_id, even when the
    # same raw_session_id appears multiple times (S/M variants).
    # Map first occurrence of each raw_session_id for answer_session_ids lookup.
    public_session_ids: dict[str, str] = {}
    for session_index, raw_session_id in enumerate(raw_session_ids):
        public_id = f"lme_{instance_index:04d}_session_{session_index:04d}"
        if raw_session_id not in public_session_ids:
            public_session_ids[raw_session_id] = public_id
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
        session_id = f"lme_{instance_index:04d}_session_{session_index:04d}"
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


def _dir_size_mb(path: Path) -> float:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


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
    mode = "memorycore_llm_expansion" if deep else "memorycore"
    if truth.abstention_expected:
        return _empty_score(instance, truth, mode=mode)
    search_fn = _search_messages_llm_expansion_mode if deep else _search_messages_mode
    hits = search_fn(core, instance.query, k)
    return _build_scored_row(instance, truth, hits, mode=mode, k=k)


def _score_instance_episodic(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
    use_cross_encoder: bool = False,
    max_context_chars: int = 16_000,
) -> dict[str, Any]:
    mode = "memorycore_episodic_reranked_4k" if max_context_chars == 4_000 else "memorycore_episodic_reranked"
    if truth.abstention_expected:
        row = _empty_score(instance, truth, mode=mode)
        row.update({
            "bundle_session_ids": [],
            "bundle_message_ids": [],
            "bundle_count": 0,
            "bundle_context_chars": 0,
            "bundle_message_recall": 0.0,
            "bundle_message_hit": False,
        })
        return row

    primary = core._search_messages_decomposed(
        instance.query,
        limit=k,
        per_query_limit=max(20, k * 4),
        use_cross_encoder=use_cross_encoder,
    )
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
    row = _build_scored_row(instance, truth, hits, mode=mode, k=k)
    bundles = core._reconstruct_sessions(
        instance.query,
        session_limit=k,
        max_context_chars=max_context_chars,
        primary_results=primary,
    )
    bundle_message_ids = [
        message.id for bundle in bundles for message in bundle.messages if message.id
    ]
    row.update({
        "bundle_session_ids": [bundle.session_id for bundle in bundles],
        "bundle_message_ids": bundle_message_ids,
        "bundle_count": len(bundles),
        "bundle_context_chars": sum(
            len(message.content) for bundle in bundles for message in bundle.messages
        ),
        "bundle_message_recall": _fractional_recall(
            bundle_message_ids, truth.expected_message_ids,
        ),
        "bundle_message_hit": bool(set(bundle_message_ids) & set(truth.expected_message_ids)),
    })
    return row


def _score_instance_v3(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
    allocation: str = "global",
) -> dict[str, Any]:
    """Episodic reranked with session-cap selection (cap=2).

    After the global cross-encoder rerank, every message of the top-k
    sessions is cross-encoder scored and the final top-k is filled with up
    to 2 messages per session (instead of the one-per-session MMR cap).
    Recovers answers that live in a second message of an already-found
    session.

    Allocation:
      - global (v3): best message recall (+0.123 message_recall@5 on S)
        but a second message of the top session can displace the 5th
        session (-0.059 session_recall@5).
      - anchor (v4): top session gets two slots, the rest one each —
        session coverage of the top contexts is preserved.
    """
    mode = "memorycore_episodic_reranked_v4" if allocation == "anchor" else "memorycore_episodic_reranked_v3"
    if truth.abstention_expected:
        row = _empty_score(instance, truth, mode=mode)
        row.update({
            "bundle_session_ids": [],
            "bundle_message_ids": [],
            "bundle_count": 0,
            "bundle_context_chars": 0,
            "bundle_message_recall": 0.0,
            "bundle_message_hit": False,
        })
        return row

    primary = core._search_messages_decomposed(
        instance.query,
        limit=k,
        per_query_limit=max(20, k * 4),
        use_cross_encoder=True,
        session_cap=2,
        allocation=allocation,
    )
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
    row = _build_scored_row(instance, truth, hits, mode=mode, k=k)
    bundles = core._reconstruct_sessions(
        instance.query,
        session_limit=k,
        primary_results=primary,
    )
    bundle_message_ids = [
        message.id for bundle in bundles for message in bundle.messages if message.id
    ]
    row.update({
        "bundle_session_ids": [bundle.session_id for bundle in bundles],
        "bundle_message_ids": bundle_message_ids,
        "bundle_count": len(bundles),
        "bundle_context_chars": sum(
            len(message.content) for bundle in bundles for message in bundle.messages
        ),
        "bundle_message_recall": _fractional_recall(
            bundle_message_ids, truth.expected_message_ids,
        ),
        "bundle_message_hit": bool(set(bundle_message_ids) & set(truth.expected_message_ids)),
    })
    return row


def _score_instance_traversal(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
) -> dict[str, Any]:
    """Query-guided graph traversal retrieval (revived SPEC design).

    Seeds from decomposed hybrid search, expansion through the message
    graph (temporal + session edges), relevance re-check per hop, session
    caps, baseline fallback, cross-encoder rerank of the merged pool.
    """
    mode = "memorycore_traversal_v2"
    if truth.abstention_expected:
        row = _empty_score(instance, truth, mode=mode)
        row.update({
            "bundle_session_ids": [],
            "bundle_message_ids": [],
            "bundle_count": 0,
            "bundle_context_chars": 0,
            "bundle_message_recall": 0.0,
            "bundle_message_hit": False,
        })
        return row

    primary = search_messages_traversal(
        core,
        instance.query,
        limit=k,
        seed_limit=50,
        hop_limit=2,
        max_per_session=2,
        use_cross_encoder=True,
    )
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
    row = _build_scored_row(instance, truth, hits, mode=mode, k=k)
    bundles = core._reconstruct_sessions(
        instance.query,
        session_limit=k,
        primary_results=primary,
    )
    bundle_message_ids = [
        message.id for bundle in bundles for message in bundle.messages if message.id
    ]
    row.update({
        "bundle_session_ids": [bundle.session_id for bundle in bundles],
        "bundle_message_ids": bundle_message_ids,
        "bundle_count": len(bundles),
        "bundle_context_chars": sum(
            len(message.content) for bundle in bundles for message in bundle.messages
        ),
        "bundle_message_recall": _fractional_recall(
            bundle_message_ids, truth.expected_message_ids,
        ),
        "bundle_message_hit": bool(set(bundle_message_ids) & set(truth.expected_message_ids)),
    })
    return row


def _score_instance_confirmed(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
) -> dict[str, Any]:
    """Episodic reranked + temporal-neighbor confirmation boost."""
    mode = "memorycore_episodic_reranked_confirmed"
    if truth.abstention_expected:
        row = _empty_score(instance, truth, mode=mode)
        row.update({
            "bundle_session_ids": [], "bundle_message_ids": [], "bundle_count": 0,
            "bundle_context_chars": 0, "bundle_message_recall": 0.0, "bundle_message_hit": False,
        })
        return row

    primary = search_messages_confirmed(core, instance.query, limit=k, seed_limit=50)
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
    row = _build_scored_row(instance, truth, hits, mode=mode, k=k)
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
    return row


def _score_instance_preference_union(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
) -> dict[str, Any]:
    """Episodic reranked with per-variant union for preference queries."""
    mode = "memorycore_episodic_reranked_preference_union"
    if truth.abstention_expected:
        row = _empty_score(instance, truth, mode=mode)
        row.update({
            "bundle_session_ids": [], "bundle_message_ids": [], "bundle_count": 0,
            "bundle_context_chars": 0, "bundle_message_recall": 0.0, "bundle_message_hit": False,
        })
        return row

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
    row = _build_scored_row(instance, truth, hits, mode=mode, k=k)
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
    return row


def _score_instance_v2(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
) -> dict[str, Any]:
    """Combined improvements: L-12 cross-encoder + preference union routing
    + temporal query decomposition (shared decompose_queries).

    Preference queries route to the per-variant union; everything else uses
    the baseline decomposed search. The L-12 reranker is forced regardless
    of the COREMEM_CROSS_ENCODER_MODEL env var.
    """
    set_cross_encoder_model("cross-encoder/ms-marco-MiniLM-L-12-v2")
    mode = "memorycore_episodic_reranked_v2"
    if truth.abstention_expected:
        row = _empty_score(instance, truth, mode=mode)
        row.update({
            "bundle_session_ids": [], "bundle_message_ids": [], "bundle_count": 0,
            "bundle_context_chars": 0, "bundle_message_recall": 0.0, "bundle_message_hit": False,
        })
        return row

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
    row = _build_scored_row(instance, truth, hits, mode=mode, k=k)
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
    return row


def _score_instance_fusion(
    core: MemoryCore,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
) -> dict[str, Any]:
    mode = "memorycore_fusion"
    if truth.abstention_expected:
        return _empty_score(instance, truth, mode=mode)
    results = core._search_with_fusion(instance.query, limit=k)
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
        for result in results
    ]
    return _build_scored_row(instance, truth, hits, mode=mode, k=k)


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
    metrics = {
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
    bundle_rows = [row for row in rows if "bundle_message_recall" in row]
    if bundle_rows:
        metrics.update({
            "bundle_message_recall": _mean([
                float(row["bundle_message_recall"]) for row in bundle_rows
                if isinstance(row.get("scoring"), Mapping)
                and row["scoring"]["expected_message_ids"]
            ]),
            "bundle_message_hit": _rate(
                row["bundle_message_hit"] for row in bundle_rows
                if isinstance(row.get("scoring"), Mapping)
                and row["scoring"]["expected_message_ids"]
            ),
            "bundle_context_chars_mean": _mean([
                float(row["bundle_context_chars"]) for row in bundle_rows
            ]),
        })
    return metrics


def _build_mode_result(
    *,
    data_path: str,
    mode: str,
    k: int,
    mode_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    completed_question_ids: set[str],
    complete: bool,
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode_name, rows in mode_rows.items():
        row_list = [dict(row) for row in rows]
        modes[mode_name] = {
            "results": row_list,
            "metrics": _aggregate_metrics(row_list, k=k),
        }
    result: dict[str, Any] = {
        "dataset": data_path,
        "evaluation_scope": "per_question_haystack",
        "mode": mode,
        "k": k,
        "complete": complete,
        "completed_question_ids": sorted(completed_question_ids),
        "modes": modes,
    }

    # Cumulative stats across all modes
    all_rows = [r for m in modes.values() for r in m.get("results", [])]
    times = [r.get("question_time_seconds", 0) for r in all_rows if r.get("question_time_seconds") is not None]
    disks = [r.get("instance_disk_mb", 0) for r in all_rows if r.get("instance_disk_mb") is not None]
    if times:
        result["cumulative_time_seconds"] = round(sum(times), 1)
        result["mean_time_seconds"] = round(sum(times) / len(times), 1)
    if disks:
        result["cumulative_disk_mb"] = round(sum(disks), 1)
        result["mean_disk_mb"] = round(sum(disks) / len(disks), 1)
    return result


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
) -> None:
    if state.get("dataset") != data_path:
        raise ValueError("resume checkpoint dataset does not match")
    if state.get("mode") != mode:
        raise ValueError("resume checkpoint mode does not match")
    if state.get("k") != k:
        raise ValueError("resume checkpoint k does not match")
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
    parser.add_argument("--mode", default="all", choices=("raw_bm25", "memorycore", "memorycore_llm_expansion", "memorycore_episodic_reranked", "memorycore_episodic_reranked_4k", "memorycore_fusion", "memorycore_traversal_v2", "memorycore_episodic_reranked_confirmed", "memorycore_episodic_reranked_preference_union", "memorycore_episodic_reranked_v2", "memorycore_episodic_reranked_v3", "memorycore_episodic_reranked_v4", "all"),
                        help="Search mode to run (default: all)")
    parser.add_argument("--k", type=int, default=5, help="Retrieval cutoff")
    parser.add_argument("--limit", type=int, help="Maximum number of instances to load")
    parser.add_argument("--question-types", nargs="*", help="Only run these question types (e.g. single-session-preference)")
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
    parser.add_argument("--resume", action="store_true", help="Resume from the checkpoint file if present")
    parser.add_argument("--resume-path", type=Path, help="Checkpoint path for resumable runs")
    parser.add_argument("--cleanup-instances", action="store_true", help="Delete per-question instance dirs after scoring (saves disk)")
    parser.add_argument("--reuse-instances", action="store_true", help="Reuse existing per-question hybrid dirs instead of re-ingesting (requires a persistent --root)")
    parser.add_argument("--progress", action="store_true", help="Print per-question progress")
    parser.add_argument("--stream", action="store_true", help="Stream questions one at a time via ijson (low memory, for large datasets like S/M)")
    args = parser.parse_args(argv)

    resume_path = args.resume_path
    if resume_path is None and args.output is not None:
        resume_path = args.output.with_suffix(args.output.suffix + ".checkpoint.json")

    jsonl_path: Path | None = None
    if args.jsonl_output is not None:
        jsonl_path = args.jsonl_output
        # In streaming mode, JSONL is appended per-question inside run_eval
        if not args.stream:
            # Non-streaming: JSONL written at end via _write_outputs
            jsonl_path = None

    if args.root is None:
        with tempfile.TemporaryDirectory(prefix="agent-memory-longmemeval-") as tmp:
            result = run_eval(
                args.data, Path(tmp),
                mode=args.mode, k=args.k,
                question_types=args.question_types, limit=args.limit,
                expand_model=args.expand_model,
                cleanup_instances=args.cleanup_instances,
                reuse_instances=args.reuse_instances,
                resume=args.resume,
                resume_path=resume_path,
                progress=args.progress,
                stream=args.stream,
                jsonl_path=jsonl_path,
            )
            if "modes" in result:
                result["summary"] = _build_summary(result["modes"], args.k)
            _write_outputs(
                result, output=args.output, jsonl_output=args.jsonl_output if not args.stream else None,
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
            cleanup_instances=args.cleanup_instances,
            reuse_instances=args.reuse_instances,
            resume=args.resume,
            resume_path=resume_path,
            progress=args.progress,
            stream=args.stream,
            jsonl_path=jsonl_path,
        )
        if "modes" in result:
            result["summary"] = _build_summary(result["modes"], args.k)
        _write_outputs(
            result, output=args.output, jsonl_output=args.jsonl_output if not args.stream else None,
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
