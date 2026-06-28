#!/usr/bin/env python3
"""Deterministic LongMemEval loader and raw AgentJournal reference baseline.

This is the Stage 2 AgentJournal eval slice: adapt LongMemEval-shaped JSON to
AgentJournal reference turns, prove oracle fields are stripped before ingestion and
retrieval, then score a raw lexical reference-message/session baseline. It does
not call an LLM and does not use embedding backends.
"""

from __future__ import annotations

import argparse
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

from coremem.agent_journal import AgentJournalBundle
from coremem.providers import create_provider
from coremem.types import Memory

GROUND_TRUTH_FIELDS = {"answer", "answer_session_ids", "has_answer"}
MODE = "raw-reference-retrieval"
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
    """Write one AgentJournal reference turn per stripped haystack session."""
    bundle = AgentJournalBundle(root)
    bundle.initialize()
    for instance in instances:
        for session in instance.sessions:
            bundle.write_reference_turn(
                session.messages,
                turn_id=session.turn_id,
                session_id=session.session_id,
                agent_context_hash="sha256:longmemeval-stripped-reference-baseline",
                metadata={},
            )
    return bundle


def run_eval(
    data_path: str | Path,
    root: str | Path,
    *,
    k: int = 5,
    question_types: Sequence[str] | None = None,
    limit: int | None = None,
    reset: bool = False,
    expand_model: str | None = None,
) -> dict[str, Any]:
    """Load data, write stripped references, run raw retrieval, and score rows."""
    if k <= 0:
        raise ValueError("k must be positive")

    root = Path(root)
    if reset and root.exists():
        _safe_reset_root(root)
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError(f"bundle root already exists: {root}")

    raw_instances = load_longmemeval_instances(data_path, question_types=question_types, limit=limit)
    prepared, truth_by_question_id = prepare_instances(raw_instances)
    bundle = build_reference_bundle(root, prepared)
    lint_errors = bundle.lint()
    llm_provider = create_provider(expand_model) if expand_model else None
    rows = [
        _score_instance(bundle, instance, truth_by_question_id[instance.question_id], k=k, llm_provider=llm_provider)
        for instance in prepared
    ]

    return {
        "dataset": str(data_path),
        "mode": MODE,
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
        "used_reference_search": mode == MODE,
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


def _score_instance(
    bundle: AgentJournalBundle,
    instance: PreparedInstance,
    truth: QuestionTruth,
    *,
    k: int,
    llm_provider=None,
) -> dict[str, Any]:
    turn_ids = {session.turn_id for session in instance.sessions}
    if truth.abstention_expected:
        return _empty_score(instance, truth, mode=MODE)
    hits = _search_reference_messages(bundle, instance.query, allowed_turn_ids=turn_ids, limit=k, llm_provider=llm_provider)
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

    # Precision@k: fraction of top-k that are relevant
    relevant = set(expected_sessions)
    top_k = retrieved_session_ids[:k]
    session_precision = sum(1 for s in top_k if s in relevant) / max(k, 1) if top_k else 0.0

    # MAP: average precision at each relevant hit
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
        "mode": MODE,
        "retrieved_page_ids": [],
        "retrieved_turn_ids": retrieved_turn_ids,
        "retrieved_session_ids": retrieved_session_ids,
        "retrieved_message_ids": retrieved_message_ids,
        "retrieved_scores": [hit.score for hit in hits],
        "context_chars": context_chars,
        "used_reference_search": True,
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
    for path in sorted(bundle.turns_dir.rglob("*.md")):
        payload = _extract_turn_payload(path)
        turn_id = str(payload["turn_id"])
        if allowed_turn_ids is not None and turn_id not in allowed_turn_ids:
            continue
        session_id = str(payload["session_id"])
        for message in payload.get("messages", []):
            if not isinstance(message, dict):
                continue
            message_id = message.get("message_id")
            role = message.get("role")
            content = message.get("content")
            if isinstance(message_id, str) and isinstance(role, str) and isinstance(content, str):
                messages.append(
                    ReferenceMessage(
                        turn_id=turn_id,
                        session_id=session_id,
                        message_id=message_id,
                        role=role,
                        content=content,
                    ),
                )
    return messages


def _extract_turn_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    parts = text.split("\n# Canonical Turn Payload\n", 1)
    if len(parts) != 2:
        raise ValueError(f"missing canonical AgentJournal turn payload section: {path}")
    match = re.search(r"```json agent_journal-turn\n(.*?)\n```", parts[1], re.DOTALL)
    if not match:
        raise ValueError(f"missing canonical AgentJournal turn payload: {path}")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError(f"canonical AgentJournal turn payload must be an object: {path}")
    return payload


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
        markers = [root / "SCHEMA.md", root / "references" / "manifest.json"]
        if not all(marker.exists() for marker in markers):
            raise ValueError(f"refusing to overwrite non-AgentJournal directory: {root}")
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic AgentJournal LongMemEval raw-reference baseline",
    )
    parser.add_argument("data", type=Path, help="Local LongMemEval-shaped JSON file")
    parser.add_argument("--root", type=Path, help="AgentJournal bundle root to write")
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
    args = parser.parse_args(argv)

    if args.root is None:
        with tempfile.TemporaryDirectory(prefix="agent-memory-longmemeval-") as tmp:
            result = run_eval(
                args.data,
                Path(tmp),
                k=args.k,
                question_types=args.question_types,
                limit=args.limit,
                expand_model=args.expand_model,
            )
            _write_outputs(
                result,
                output=args.output,
                jsonl_output=args.jsonl_output,
                include_scoring=args.include_scoring,
            )
            _print_result(result, as_json=args.json, include_scoring=args.include_scoring)
    else:
        if args.root.exists() and any(args.root.iterdir()) and not args.overwrite:
            parser.error(f"--root already exists; pass --overwrite to replace it: {args.root}")
        result = run_eval(
            args.data,
            args.root,
            k=args.k,
            question_types=args.question_types,
            limit=args.limit,
            reset=args.overwrite,
            expand_model=args.expand_model,
        )
        _write_outputs(
            result,
            output=args.output,
            jsonl_output=args.jsonl_output,
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
        if not isinstance(rows, list):
            raise ValueError("result rows must be a list")
        jsonl_output.write_text(
            "".join(json.dumps(_public_row(row), sort_keys=True) + "\n" for row in rows),
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
    rows = public.get("results", [])
    if isinstance(rows, list):
        public["results"] = [_public_row(row) if isinstance(row, Mapping) else row for row in rows]
    return public


def _print_result(result: Mapping[str, Any], *, as_json: bool, include_scoring: bool) -> None:
    if as_json:
        print(json.dumps(_public_result(result, include_scoring=include_scoring), indent=2, sort_keys=True))
        return
    metrics = result["metrics"]
    k = result["k"]
    lint = "pass" if result["lint"]["passed"] else "fail"
    print(f"lint: {lint}")
    print(f"questions: {metrics['question_count']}")
    print(f"session_recall@{k}: {metrics[f'session_recall@{k}']}")
    print(f"message_recall@{k}: {metrics[f'message_recall@{k}']}")
    print(f"session_mrr: {metrics['session_mrr']}")
    print(f"empty_retrieval_rate: {metrics['empty_retrieval_rate']}")
    print(f"abstention_false_positive_rate: {metrics['abstention_false_positive_rate']}")


if __name__ == "__main__":
    raise SystemExit(main())
