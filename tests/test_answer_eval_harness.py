"""Tests for the answer-eval harness's per-question resource handling.

The answer eval builds 1-2 ``MemoryCore`` instances per question. Without
``close()`` the pooled Chroma clients and SQLite handles accumulate for the
life of the process — the failure mode documented on ``MemoryCore.close``, and
the reason long runs died around question 9-18 during the fact-digest gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# eval_answer_longmemeval.py imports its sibling by bare name (it is run as a
# script, where scripts/ is already on sys.path). Mirror that for import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts import eval_answer_longmemeval as harness  # noqa: E402


class FakeProvider:
    """Answers questions and judges verdicts without touching the network."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages: list[dict]) -> SimpleNamespace:
        self.calls += 1
        prompt = messages[0]["content"]
        if "Judge each candidate answer" in prompt:
            verdicts = {
                f"answer_{index}": {"correct": True, "reason": "matches"}
                for index in range(len(harness.MODES))
            }
            return SimpleNamespace(content=json.dumps(verdicts))
        return SimpleNamespace(content="Maya picked the pear tart.")


class ExplodingProvider(FakeProvider):
    async def chat(self, messages: list[dict]) -> SimpleNamespace:
        raise RuntimeError("simulated provider failure")


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
    ]


def _write_fixture(path: Path) -> Path:
    path.write_text(json.dumps(_fixture()), encoding="utf-8")
    return path


def _track_cores(monkeypatch) -> tuple[list[object], set[int]]:
    from coremem.core import MemoryCore as RealMemoryCore

    built: list[object] = []
    closed: set[int] = set()
    original_init = RealMemoryCore.__init__
    original_close = RealMemoryCore.close

    def tracking_init(self, *args, **kwargs):
        built.append(self)
        original_init(self, *args, **kwargs)

    def tracking_close(self, *args, **kwargs):
        closed.add(id(self))
        original_close(self, *args, **kwargs)

    monkeypatch.setattr(RealMemoryCore, "__init__", tracking_init)
    monkeypatch.setattr(RealMemoryCore, "close", tracking_close)
    return built, closed


def test_answer_eval_closes_its_memorycore(tmp_path, monkeypatch):
    built, closed = _track_cores(monkeypatch)
    monkeypatch.setattr(harness, "create_provider", lambda model: FakeProvider())

    data_path = _write_fixture(tmp_path / "fixture.json")
    result = harness.run(
        data_path,
        tmp_path / "out.json",
        tmp_path / "instances",
        answer_model="fake:answer",
        judge_model="fake:judge",
    )

    assert built, "expected the harness to build a core"
    assert closed == {id(core) for core in built}, (
        f"leaked {len(built) - len(closed)} of {len(built)} cores"
    )
    assert len(result["results"]) == 1


def test_answer_eval_closes_its_memorycore_when_the_provider_fails(tmp_path, monkeypatch):
    """The close belongs in a finally: a provider blow-up must not leak either."""
    built, closed = _track_cores(monkeypatch)
    monkeypatch.setattr(harness, "create_provider", lambda model: ExplodingProvider())

    data_path = _write_fixture(tmp_path / "fixture.json")
    with pytest.raises(RuntimeError, match="simulated provider failure"):
        harness.run(
            data_path,
            tmp_path / "out.json",
            tmp_path / "instances",
            answer_model="fake:answer",
            judge_model="fake:judge",
        )

    assert built, "expected the harness to build a core before answering"
    assert closed == {id(core) for core in built}, (
        f"leaked {len(built) - len(closed)} of {len(built)} cores on the error path"
    )
