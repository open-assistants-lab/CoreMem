"""Tests for the AML (Agent Memory Leaderboard) Add/Search adapter.

AML-specific: these tests exercise the integration in integrations/aml/,
which exists only to expose CoreMem through the AML evaluation contract.
Contract verified against the live page https://agentmemories.ai/api-guide
via browser automation.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def server(monkeypatch, tmp_path):
    """Fresh server module per test: env is read at import time."""
    monkeypatch.setenv("COREMEM_PATH", str(tmp_path / "memory"))
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    mod = importlib.import_module("integrations.aml.server")
    return importlib.reload(mod)


@pytest.fixture()
def client(server):
    return TestClient(server.app)


def _add(client, user_id: str, session_id: str, messages: list[dict], request_id: str = "r1"):
    return client.post("/v1/memories/add", json={
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "messages": messages,
    })


def test_add_echoes_contract_fields(client):
    resp = _add(client, "uid-1", "conv-1", [
        {"role": "user", "content": "I like hiking in Yosemite"},
    ], request_id="req-123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["request_id"] == "req-123"
    assert body["user_id"] == "uid-1"
    assert body["session_id"] == "conv-1"


def test_add_stores_messages_with_user_id_and_session_id(server, client):
    _add(client, "uid-1", "conv-1", [
        {"role": "user", "content": "I like hiking in Yosemite"},
        {"role": "assistant", "content": "Yosemite is beautiful in spring"},
    ])
    memories = server.core.fetch_all()
    assert len(memories) == 2
    assert all(m.user_id == "uid-1" for m in memories)
    assert all(m.session_id == "conv-1" for m in memories)


def test_add_converts_unix_ms_timestamp(server, client):
    _add(client, "uid-1", "conv-1", [
        {"role": "user", "content": "I visited the museum", "timestamp": 1_700_000_000_000},
    ])
    memory = server.core.fetch_all()[0]
    assert memory.ts is not None
    assert int(memory.ts.timestamp() * 1000) == 1_700_000_000_000


def test_search_returns_data_array_with_contract_fields(client):
    _add(client, "uid-1", "conv-1", [
        {"role": "user", "content": "I love hiking in Yosemite"},
        {"role": "user", "content": "I prefer coffee over tea"},
    ])
    resp = client.post("/v1/memories/search", json={
        "query": "hiking Yosemite",
        "user_id": "uid-1",
        "top_k": 5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    results = body["data"]
    assert len(results) >= 1
    first = results[0]
    assert {"id", "content"} <= set(first.keys())
    assert "Yosemite" in first["content"]
    # ordered most relevant first
    assert results[0]["content"] == "I love hiking in Yosemite"


def test_search_respects_user_id_isolation(client):
    _add(client, "uid-1", "conv-1", [{"role": "user", "content": "I love hiking in Yosemite"}])
    _add(client, "uid-2", "conv-2", [{"role": "user", "content": "I love hiking in Tahoe"}])
    resp = client.post("/v1/memories/search", json={
        "query": "hiking",
        "user_id": "uid-1",
        "top_k": 5,
    })
    results = resp.json()["data"]
    assert results
    assert all("Tahoe" not in r["content"] for r in results)


def test_search_returns_empty_data_when_no_relevant_memory(client):
    _add(client, "uid-1", "conv-1", [{"role": "user", "content": "I love hiking in Yosemite"}])
    resp = client.post("/v1/memories/search", json={
        "query": "quantum physics research",
        "user_id": "uid-1",
        "top_k": 5,
    })
    assert resp.status_code == 200
    assert resp.json()["data"] == []


def test_search_honors_top_k(client):
    for i in range(5):
        _add(client, "uid-1", f"conv-{i}", [
            {"role": "user", "content": f"I love hiking in Yosemite {i}"},
        ])
    resp = client.post("/v1/memories/search", json={
        "query": "hiking Yosemite",
        "user_id": "uid-1",
        "top_k": 2,
    })
    assert len(resp.json()["data"]) <= 2


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_add_handles_concurrent_requests(server):
    """The AML platform runs concurrent Add workers (64 by default); writes
    must be serialized instead of hanging on SQLite lock contention."""
    from concurrent.futures import ThreadPoolExecutor

    client = TestClient(server.app)
    errors: list[str] = []

    def add_worker(i: int) -> None:
        try:
            resp = client.post("/v1/memories/add", json={
                "request_id": f"r{i}", "user_id": "uid-1", "session_id": f"c{i % 3}",
                "messages": [
                    {"role": "user", "content": f"message {i} about hiking in Yosemite"}
                ] * 5,
            })
            if resp.status_code != 200:
                errors.append(f"worker {i}: HTTP {resp.status_code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"worker {i}: {type(exc).__name__}: {exc}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(add_worker, i) for i in range(16)]
        for future in futures:
            # A hung request (lock contention without the write lock) surfaces
            # here as a TimeoutError instead of hanging the whole suite.
            future.result(timeout=60)

    assert errors == []
    assert server.core.count() == 16 * 5


def test_auth_required_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("COREMEM_PATH", str(tmp_path / "memory"))
    monkeypatch.setenv("AML_API_KEY", "secret-key")
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    mod = importlib.import_module("integrations.aml.server")
    mod = importlib.reload(mod)
    client = TestClient(mod.app)

    resp = client.post("/v1/memories/search", json={"query": "x", "user_id": "u", "top_k": 5})
    assert resp.status_code == 401

    resp = client.post("/v1/memories/search", json={"query": "x", "user_id": "u", "top_k": 5},
                       headers={"Authorization": "Bearer secret-key"})
    assert resp.status_code == 200
