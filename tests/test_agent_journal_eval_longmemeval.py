"""Tests for the deterministic LongMemEval AgentJournal raw baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_agent_journal_longmemeval import main, run_eval


def _fixture() -> list[dict]:
    return [
        {
            "question_id": "q_dessert",
            "question_type": "single-session-user",
            "question": "Which dessert did Maya pick for launch dinner?",
            "question_date": "2026-06-20",
            "answer": "Maya picked the pear tart.",
            "answer_session_ids": ["dessert_session"],
            "haystack_session_ids": ["dessert_session"],
            "haystack_dates": ["2026-06-18T09:00:00Z"],
            "haystack_sessions": [
                [
                    {
                        "role": "user",
                        "content": "For launch dinner, Maya picked the pear tart as dessert.",
                        "has_answer": True,
                    },
                    {"role": "assistant", "content": "I noted the launch dinner dessert choice."},
                ],
            ],
        },
        {
            "question_id": "q_update",
            "question_type": "knowledge-update",
            "question": "What is the current project codename after the update?",
            "question_date": "2026-06-20",
            "answer": "The current project codename is Quartz.",
            "answer_session_ids": ["update_new"],
            "haystack_session_ids": ["update_old", "update_new"],
            "haystack_dates": ["2026-06-16", "2026-06-19"],
            "haystack_sessions": [
                [
                    {
                        "role": "user",
                        "content": "Earlier, the project codename was Pebble.",
                    },
                ],
                [
                    {
                        "role": "user",
                        "content": "Update: the current project codename is Quartz, not Pebble.",
                        "has_answer": True,
                    },
                ],
            ],
        },
        {
            "question_id": "q_assistant",
            "question_type": "single-session-assistant",
            "question": "Where is the blue envelope backup code?",
            "question_date": "2026-06-20",
            "answer": "The backup code is in the blue envelope.",
            "answer_session_ids": ["assistant_session"],
            "has_answer": [[False, True]],
            "haystack_session_ids": ["assistant_session"],
            "haystack_dates": ["2026-06-17"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "Please store the backup code somewhere safe."},
                    {"role": "assistant", "content": "I stored the backup code in the blue envelope."},
                ],
            ],
        },
        {
            "question_id": "q_absent",
            "question_type": "single-session-user_abs",
            "question": "Which vendor owns zebra orchid billing?",
            "question_date": "2026-06-20",
            "answer": "No information is available.",
            "answer_session_ids": [],
            "haystack_session_ids": ["unrelated_session"],
            "haystack_dates": ["2026-06-15"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I prefer chamomile tea before morning walks."},
                    {"role": "assistant", "content": "I will remember the chamomile preference."},
                ],
            ],
        },
    ]


def _write_fixture(path: Path) -> Path:
    path.write_text(json.dumps(_fixture(), indent=2), encoding="utf-8")
    return path


def test_longmemeval_eval_builds_references_and_scores_metrics(tmp_path):
    data_path = _write_fixture(tmp_path / "longmemeval_fixture.json")

    result = run_eval(data_path, tmp_path / "memorypack", k=3)

    assert result["lint"] == {"passed": True, "errors": []}
    assert result["bundle"]["reference_turn_count"] == 5
    assert result["bundle"]["page_count"] == 0

    metrics = result["metrics"]
    assert metrics["question_count"] == 4
    assert metrics["answerable_question_count"] == 3
    assert metrics["abstention_question_count"] == 1
    assert metrics["session_recall@3"] == 1.0
    assert metrics["message_recall@3"] == 1.0
    assert metrics["session_mrr"] == 1.0
    assert metrics["message_mrr"] == 1.0
    assert metrics["empty_retrieval_rate"] == 0.25
    assert metrics["abstention_false_positive_rate"] == 0.0
    assert metrics["context_chars_mean"] > 0


def test_longmemeval_eval_strips_ground_truth_from_memorypack_files(tmp_path):
    data_path = _write_fixture(tmp_path / "longmemeval_fixture.json")
    root = tmp_path / "memorypack"

    run_eval(data_path, root, k=3)

    memorypack_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(root.rglob("*")) if path.is_file()
    )
    assert "has_answer" not in memorypack_text
    assert "answer_session_ids" not in memorypack_text
    assert '"answer"' not in memorypack_text
    assert "Maya picked the pear tart" in memorypack_text
    assert "The current project codename is Quartz." not in memorypack_text


def test_longmemeval_eval_returns_rows_and_breakdown(tmp_path):
    data_path = _write_fixture(tmp_path / "longmemeval_fixture.json")

    result = run_eval(data_path, tmp_path / "memorypack", k=3)
    rows = {row["question_id"]: row for row in result["results"]}

    update = rows["q_update"]
    assert update["retrieved_session_ids"][0] == "lme_0001_session_0001"
    assert update["retrieved_message_ids"][0] == "lme_0001_session_0001_turn_0000_user"
    assert update["scoring"]["expected_message_ids"] == ["lme_0001_session_0001_turn_0000_user"]
    assert update["scoring"]["session_recall@3"] == 1.0
    assert update["scoring"]["message_recall@3"] == 1.0
    assert update["scoring"]["session_hit@3"] is True
    assert update["scoring"]["message_hit@3"] is True

    absent = rows["q_absent"]
    assert absent["retrieved_session_ids"] == []
    assert absent["retrieved_message_ids"] == []
    assert absent["scoring"]["empty_retrieval"] is True
    assert absent["scoring"]["abstention_expected"] is True
    assert absent["scoring"]["abstention_false_positive"] is False

    breakdown = result["metrics"]["by_question_type"]
    assert breakdown["knowledge-update"]["count"] == 1
    assert breakdown["knowledge-update"]["session_recall@3"] == 1.0
    assert breakdown["single-session-user_abs"]["empty_retrieval_rate"] == 1.0


def test_longmemeval_eval_cli_emits_json_and_jsonl(tmp_path, capsys):
    data_path = _write_fixture(tmp_path / "longmemeval_fixture.json")
    jsonl_path = tmp_path / "rows.jsonl"

    exit_code = main([
        str(data_path),
        "--root",
        str(tmp_path / "memorypack"),
        "--k",
        "3",
        "--json",
        "--jsonl-output",
        str(jsonl_path),
    ])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    jsonl_rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]

    assert exit_code == 0
    assert payload["mode"] == "raw-reference-retrieval"
    assert payload["metrics"]["session_recall@3"] == 1.0
    assert all("scoring" not in row for row in payload["results"])
    assert len(jsonl_rows) == 4
    assert all("scoring" not in row for row in jsonl_rows)
    assert {row["question_id"] for row in jsonl_rows} == {
        "q_absent",
        "q_assistant",
        "q_dessert",
        "q_update",
    }


def test_longmemeval_eval_uses_fractional_recall_for_multiple_evidence_sessions(tmp_path):
    data = [
        {
            "question_id": "q_multi",
            "question_type": "multi-session",
            "question": "What did Ana choose?",
            "answer": "Ana chose the red notebook and Ben chose the blue folder.",
            "answer_session_ids": ["answer_ana", "answer_ben"],
            "haystack_session_ids": ["answer_ana", "answer_ben"],
            "haystack_dates": ["2023/05/20 (Sat) 01:10", "2023/05/21 (Sun) 01:10"],
            "haystack_sessions": [
                [{"role": "user", "content": "Ana chose the red notebook.", "has_answer": True}],
                [{"role": "user", "content": "Ben selected the blue folder.", "has_answer": True}],
            ],
        },
    ]
    data_path = tmp_path / "longmemeval_fixture.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")

    result = run_eval(data_path, tmp_path / "memorypack", k=1)
    row = result["results"][0]

    assert row["scoring"]["session_recall@1"] == 0.5
    assert row["scoring"]["message_recall@1"] == 0.5
    memorypack_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((tmp_path / "memorypack").rglob("*"))
        if path.is_file()
    )
    assert "answer_ana" not in memorypack_text
    assert "answer_ben" not in memorypack_text


def test_longmemeval_eval_detects_abs_suffix_on_question_id(tmp_path):
    data = [
        {
            "question_id": "q_temporal_abs",
            "question_type": "temporal-reasoning",
            "question": "Which vendor owns zebra orchid billing?",
            "answer": "No information is available.",
            "answer_session_ids": ["answer_abs_marker"],
            "haystack_session_ids": ["answer_abs_marker"],
            "haystack_dates": ["2023/05/20 (Sat) 01:10"],
            "haystack_sessions": [
                [{"role": "user", "content": "I prefer chamomile tea before walks."}],
            ],
        },
    ]
    data_path = tmp_path / "longmemeval_fixture.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")

    result = run_eval(data_path, tmp_path / "memorypack", k=3)
    row = result["results"][0]

    assert row["scoring"]["abstention_expected"] is True
    assert row["scoring"]["expected_session_ids"] == []
    assert result["metrics"]["abstention_question_count"] == 1


def test_longmemeval_eval_refuses_unsafe_overwrite(tmp_path):
    data_path = _write_fixture(tmp_path / "longmemeval_fixture.json")
    unsafe_root = tmp_path / "not_memorypack"
    unsafe_root.mkdir()
    (unsafe_root / "keep.txt").write_text("important", encoding="utf-8")

    with pytest.raises(ValueError, match="refusing to overwrite non-AgentJournal"):
        run_eval(data_path, unsafe_root, reset=True)
