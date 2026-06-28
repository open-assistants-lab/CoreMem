"""Tests for the deterministic internal AgentJournal eval harness."""

from __future__ import annotations

import json

from scripts.eval_agent_journal_internal import eval_questions, main, run_eval


def test_internal_eval_builds_valid_fixture_and_scores_retrieval(tmp_path):
    result = run_eval(tmp_path / "memorypack")

    assert result["lint"] == {"passed": True, "errors": []}
    assert result["bundle"]["reference_turn_count"] == 8
    assert result["bundle"]["page_count"] == 7
    assert result["bundle"]["stale_claim_count"] == 0
    assert result["bundle"]["memory_has_reference_links"] is False
    assert result["bundle"]["system_prompt_leak"] is False

    metrics = result["metrics"]
    assert metrics["question_count"] == len(eval_questions())
    assert metrics["answerable_question_count"] == 7
    assert metrics["raw_reference_hit_rate"] == 1.0
    assert metrics["memorypack_page_hit_rate"] == 1.0
    assert metrics["abstention_correct_rate"] == 1.0
    assert metrics["raw_context_chars_mean"] > 0
    assert metrics["memorypack_context_chars_mean"] > 0


def test_internal_eval_validates_citations_and_negative_control(tmp_path):
    result = run_eval(tmp_path / "memorypack")

    citations = result["citations"]
    assert citations["checked"] == 8
    assert citations["valid"] == 8
    assert citations["invalid"] == []
    assert citations["validity_rate"] == 1.0
    assert citations["negative_control_caught"] is True
    assert "source_quote is not an exact substring" in "\n".join(
        citations["negative_control_errors"],
    )


def test_internal_eval_returns_expected_rows_for_raw_and_page_search(tmp_path):
    result = run_eval(tmp_path / "memorypack")
    rows = {row["question_id"]: row for row in result["questions"]}

    decision = rows["q_decision"]
    assert "session_memorypack_poc" in decision["raw_reference_search"]["retrieved_session_ids"]
    assert "decision_user_001" in decision["raw_reference_search"]["retrieved_message_ids"]
    assert "decisions.deterministic-retrieval" in decision["memorypack_page_search"]["retrieved_page_ids"]

    update = rows["q_knowledge_update"]
    assert "index_new_user_001" in update["raw_reference_search"]["retrieved_message_ids"]
    assert "decisions.index-format" in update["memorypack_page_search"]["retrieved_page_ids"]

    absent = rows["q_absent"]
    assert absent["abstention_expected"] is True
    assert absent["raw_reference_search"]["retrieved_message_ids"] == []
    assert absent["raw_reference_search"]["hit"] is False
    assert absent["memorypack_page_search"]["retrieved_page_ids"] == []
    assert absent["memorypack_page_search"]["hit"] is False
    assert absent["memorypack_page_search"]["abstained"] is True


def test_internal_eval_result_is_json_serializable(tmp_path):
    result = run_eval(tmp_path / "memorypack")

    encoded = json.dumps(result, sort_keys=True)
    decoded = json.loads(encoded)

    assert decoded["fixture"] == "memorypack-internal-scripted-v1"
    assert decoded["metrics"] == result["metrics"]


def test_internal_eval_cli_emits_structured_json(tmp_path, capsys):
    exit_code = main(["--root", str(tmp_path / "memorypack"), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["lint"]["passed"] is True
    assert payload["modes"] == ["raw-reference-search", "memorypack-page-search"]
    assert payload["metrics"]["memorypack_page_hit_rate"] == 1.0
