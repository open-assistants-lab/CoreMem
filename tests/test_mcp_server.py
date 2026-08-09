"""Tests for the MCP server."""
from __future__ import annotations

import os
import tempfile


def test_format_recall_results():
    from coremem import get_core
    from coremem.mcp_server import format_recall_results
    os.environ["COREMEM_PATH"] = str(tempfile.mkdtemp() + "/hybrid")
    core = get_core()
    core.ingest("user", "I love hiking in Yosemite", session_id="s1")
    results = core.recall("hiking", strategy="direct", limit=5)
    formatted = format_recall_results(results)
    assert "[user]" in formatted
    assert "hiking" in formatted.lower()
    assert "score:" in formatted


def test_format_bundles():
    from coremem import SessionBundle, get_core
    from coremem.mcp_server import format_bundles
    os.environ["COREMEM_PATH"] = str(tempfile.mkdtemp() + "/hybrid")
    get_core()
    bundle = SessionBundle(
        session_id="s1",
        messages=[],
        score=0.5,
        complete=True,
        anchor_ids=[],
    )
    formatted = format_bundles([bundle])
    assert "Session s1" in formatted
    assert "0.50" in formatted


def test_format_sessions():
    from coremem.mcp_server import format_sessions
    rows = [
        {"session_id": "s1", "messages": 3, "last_ts": "2024-01-01T00:00:00"},
        {"session_id": "s2", "messages": 1, "last_ts": "2024-01-02T00:00:00"},
    ]
    formatted = format_sessions(rows)
    assert "s1" in formatted
    assert "s2" in formatted
    assert "messages" in formatted
