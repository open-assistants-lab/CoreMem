#!/usr/bin/env python3
"""Adapt LoCoMo (snap-research) into CoreMem's LongMemEval-shaped eval format.

LoCoMo: 10 very-long conversations (avg ~27 sessions, ~16K tokens), 1,986 QA
pairs with per-turn evidence (``dia_id``) and category labels:
    1=multi-hop (282)  2=temporal (321)  3=open-domain (96)
    4=single-hop (841) 5=adversarial (446)

Mapping to LongMemEval eval types:
    single-hop    -> single-session-user
    multi-hop     -> multi-session
    temporal      -> temporal-reasoning
    open-domain   -> open-domain   (custom type; not in LongMemEval taxonomy)
    adversarial   -> abstention    (question_id gets "_abs" suffix; empty answer)

Each output instance carries the full source conversation as its per-question
haystack (canonical LongMemEval setup — same as longmemeval_s_cleaned.json).
Speakers map deterministically: speaker_a -> "user", speaker_b -> "assistant".

Usage:
    uv run python3 scripts/adapt_locomo.py \
        data/locomo/locomo10.json \
        --output data/locomo/locomo_longmemeval.json \
        --subset 40            # optional stratified subset (per-category cap)
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

CATEGORY_TO_TYPE = {
    1: "multi-session",
    2: "temporal-reasoning",
    3: "open-domain",
    4: "single-session-user",
    5: "abstention",
}

_MONTHS = {
    month.lower(): index for index, month in enumerate(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"]
    )
}


def _parse_locomo_date(value: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '2023-05-08 13:56' (ISO, UTC-less)."""
    match = re.search(
        r"(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})",
        value.strip(), re.IGNORECASE,
    )
    if not match:
        return ""
    hour, minute, meridiem, day, month_name, year = match.groups()
    hour = int(hour)
    if meridiem.lower() == "pm" and hour < 12:
        hour += 12
    if meridiem.lower() == "am" and hour == 12:
        hour = 0
    month = _MONTHS.get(month_name.lower())
    if month is None:
        return ""
    return f"{int(year):04d}-{month:02d}-{int(day):02d} {hour:02d}:{minute}"


def _build_dia_index(conversation: dict) -> dict[str, tuple[int, int]]:
    """Map dia_id -> (session_index, message_index) across all sessions."""
    index: dict[str, tuple[int, int]] = {}
    for session_index, (key, value) in enumerate(conversation.items()):
        if not key.startswith("session_") or key.endswith("date_time"):
            continue
        if not isinstance(value, list):
            continue
        for message_index, turn in enumerate(value):
            if isinstance(turn, dict) and turn.get("dia_id"):
                # Normalize turn-index leading zeros so evidence typos resolve.
                dia_id = str(turn["dia_id"])
                match = re.match(r"^(D\d+):(\d+)$", dia_id, re.IGNORECASE)
                if match:
                    dia_id = f"{match.group(1)}:{int(match.group(2))}"
                index[dia_id] = (session_index, message_index)
    return index


def _speaker_role(speaker: str, conversation: dict) -> str:
    if speaker == conversation.get("speaker_a"):
        return "user"
    return "assistant"


def _turn_content(turn: dict) -> str:
    text = str(turn.get("text", "")).strip()
    caption = turn.get("blip_caption")
    if caption:
        text = f"{text}\n[image: {caption}]" if text else f"[image: {caption}]"
    return text


def _iter_sessions(conversation: dict):
    for key, value in conversation.items():
        if key.startswith("session_") and not key.endswith("date_time") and isinstance(value, list):
            yield key, value


def _split_evidence(evidence: list) -> list[str]:
    """Evidence entries may contain space-separated dia_ids (dataset typos).

    Also normalizes turn-index leading zeros (source has ``D30:05`` for
    ``D30:5``) so typos still resolve to their turn.
    """
    out: list[str] = []
    for entry in evidence:
        for piece in str(entry).split():
            piece = piece.strip()
            if not piece:
                continue
            match = re.match(r"^(D\d+):(\d+)$", piece, re.IGNORECASE)
            if match:
                piece = f"{match.group(1)}:{int(match.group(2))}"
            out.append(piece)
    return out


def convert_sample(
    sample: dict, sample_index: int, dia_index: dict[str, tuple[int, int]],
) -> list[dict]:
    conversation = sample["conversation"]
    session_keys = [key for key, _ in _iter_sessions(conversation)]
    dates = [
        _parse_locomo_date(str(conversation.get(f"{key}_date_time", "")))
        for key in session_keys
    ]
    raw_session_ids = session_keys
    instances: list[dict] = []
    for qa_index, qa in enumerate(sample.get("qa", [])):
        category = int(qa.get("category", 4))
        qtype = CATEGORY_TO_TYPE.get(category, "single-session-user")
        is_abs = category == 5
        evidence_dia = _split_evidence(qa.get("evidence", []) or [])

        # Per-turn has_answer flags + session membership for evidence.
        has_answer_flags: list[list[bool]] = []
        evidence_session_keys: list[str] = []
        for session_index, (key, turns) in enumerate(_iter_sessions(conversation)):
            flags: list[bool] = []
            for message_index, turn in enumerate(turns):
                dia_id = str(turn.get("dia_id", ""))
                flags.append(dia_id in evidence_dia)
            if any(flags):
                evidence_session_keys.append(key)
            has_answer_flags.append(flags)

        question_id = f"locomo_{sample.get('sample_id', sample_index)}_q{qa_index:03d}"
        if is_abs:
            question_id += "_abs"
        # Non-adversarial questions with no evidence at all are unscorable for
        # retrieval metrics (no expected session/message) and the harness would
        # misclassify them as abstention via the no-evidence fallback. Drop them.
        if not is_abs and not evidence_dia:
            continue
        instances.append({
            "question_id": question_id,
            "question_type": qtype,
            "question": str(qa.get("question", "")),
            "answer": str(qa.get("answer", "")) if not is_abs else "",
            "haystack_sessions": [
                [
                    {
                        "role": _speaker_role(turn.get("speaker", ""), conversation),
                        "content": _turn_content(turn),
                        "has_answer": bool(has_answer_flags[si][mi]),
                    }
                    for mi, turn in enumerate(turns)
                ]
                for si, (_, turns) in enumerate(_iter_sessions(conversation))
            ],
            "haystack_session_ids": raw_session_ids,
            "haystack_dates": dates,
            "answer_session_ids": evidence_session_keys,
        })
    return instances


def _stratify(instances: list[dict], cap: int) -> list[dict]:
    from collections import defaultdict
    buckets: dict[str, list[dict]] = defaultdict(list)
    for instance in instances:
        buckets[instance["question_type"]].append(instance)
    selected: list[dict] = []
    for qtype in sorted(buckets):
        selected.extend(buckets[qtype][:cap])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--subset", type=int, default=0,
        help="If >0, emit a stratified subset capped at N per question type",
    )
    args = parser.parse_args()

    samples = json.loads(args.input.read_text(encoding="utf-8"))
    all_instances: list[dict] = []
    for sample_index, sample in enumerate(samples):
        dia_index = _build_dia_index(sample["conversation"])
        all_instances.extend(convert_sample(sample, sample_index, dia_index))

    if args.subset:
        all_instances = _stratify(all_instances, args.subset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(all_instances, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    from collections import Counter
    counts = Counter(inst["question_type"] for inst in all_instances)
    print(f"wrote {len(all_instances)} instances -> {args.output}")
    for qtype, count in counts.most_common():
        print(f"  {qtype:24s} {count}")
    empty_abs = sum(1 for i in all_instances if i["question_id"].endswith("_abs"))
    print(f"  abstention (_abs)     {empty_abs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
