"""Per-question resource handling in the S-scale eval harnesses.

``eval_combined_s.py`` and ``eval_graph_s.py`` each build a ``MemoryCore`` per
question. Neither closed it, so a 500-question run accumulated 500 pooled
Chroma clients — the failure mode documented on ``MemoryCore.close``. These
tests pin the close on both the success and the error path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scripts import eval_combined_s as combined  # noqa: E402
from scripts import eval_graph_s as graph  # noqa: E402
from coremem.core import MemoryCore as RealMemoryCore  # noqa: E402


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


def _write_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(_fixture()), encoding="utf-8")
    return path


def _tracked(monkeypatch, module) -> tuple[list[object], set[int]]:
    built: list[object] = []
    closed: set[int] = set()

    class TrackedMemoryCore(RealMemoryCore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

        def close(self):
            closed.add(id(self))
            super().close()

    monkeypatch.setattr(module, "MemoryCore", TrackedMemoryCore)
    return built, closed


def test_eval_combined_s_closes_its_memorycore(tmp_path, monkeypatch):
    built, closed = _tracked(monkeypatch, combined)
    data = _write_fixture(tmp_path)

    combined.main([
        str(data), "--limit", "1", "--k", "3",
        "--output", str(tmp_path / "out.json"), "--root", str(tmp_path / "roots"),
    ])

    assert built, "expected the harness to build a core"
    assert closed == {id(core) for core in built}, (
        f"leaked {len(built) - len(closed)} of {len(built)} cores"
    )


def test_eval_combined_s_closes_its_memorycore_when_scoring_raises(tmp_path, monkeypatch):
    built, closed = _tracked(monkeypatch, combined)

    def exploding_score(*args, **kwargs):
        raise RuntimeError("simulated scoring failure")

    monkeypatch.setattr(combined, "_score", exploding_score)
    data = _write_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="simulated scoring failure"):
        combined.main([
            str(data), "--limit", "1", "--k", "3",
            "--output", str(tmp_path / "out.json"), "--root", str(tmp_path / "roots"),
        ])

    assert built, "expected the harness to build a core before scoring"
    assert closed == {id(core) for core in built}, (
        f"leaked {len(built) - len(closed)} of {len(built)} cores on the error path"
    )


def test_eval_graph_s_closes_its_memorycore(tmp_path, monkeypatch):
    built, closed = _tracked(monkeypatch, graph)
    data = _write_fixture(tmp_path)

    graph.main([
        str(data), "--limit", "1", "--k", "3",
        "--output", str(tmp_path / "out.json"), "--root", str(tmp_path / "roots"),
    ])

    assert built, "expected the harness to build a core"
    assert closed == {id(core) for core in built}, (
        f"leaked {len(built) - len(closed)} of {len(built)} cores"
    )


def test_eval_graph_s_closes_its_memorycore_when_scoring_raises(tmp_path, monkeypatch):
    built, closed = _tracked(monkeypatch, graph)

    def exploding_score(*args, **kwargs):
        raise RuntimeError("simulated scoring failure")

    monkeypatch.setattr(graph, "_score_question", exploding_score)
    data = _write_fixture(tmp_path)

    with pytest.raises(RuntimeError, match="simulated scoring failure"):
        graph.main([
            str(data), "--limit", "1", "--k", "3",
            "--output", str(tmp_path / "out.json"), "--root", str(tmp_path / "roots"),
        ])

    assert built, "expected the harness to build a core before scoring"
    assert closed == {id(core) for core in built}, (
        f"leaked {len(built) - len(closed)} of {len(built)} cores on the error path"
    )
