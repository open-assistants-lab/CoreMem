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


def test_mcp_server_full_protocol_roundtrip():
    """End-to-end MCP protocol test: spawn the stdio server and exercise the
    tools as a real client (no cross-encoder needed — direct strategy)."""
    pytest = __import__("pytest")
    pytest.importorskip("mcp")

    import asyncio
    import shutil
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    memory_path = tempfile.mkdtemp(prefix="coremem-mcp-proto-") + "/hybrid"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "coremem", "mcp"],
        env={**os.environ, "COREMEM_PATH": memory_path},
    )

    async def run():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                names = [t.name for t in tools.tools]
                assert names == [
                    "recall", "ingest", "delete", "fetch_session",
                    "list_sessions", "stats", "compile", "rebuild_index",
                ], names

                text = lambda r: r.content[0].text

                # empty memory
                assert "messages: 0" in text(await session.call_tool("stats", {}))

                # ingest two sessions
                r = await session.call_tool("ingest", {
                    "role": "user", "content": "I prefer dark roast coffee",
                    "session_id": "conv_001",
                })
                assert "turn_id:" in text(r)
                await session.call_tool("ingest", {
                    "role": "assistant", "content": "Great choice!", "session_id": "conv_001",
                })
                await session.call_tool("ingest", {
                    "role": "user", "content": "I visited the Grand Canyon",
                    "session_id": "conv_002",
                })

                # recall: direct strategy, ids in output
                r = text(await session.call_tool(
                    "recall", {"query": "coffee", "strategy": "direct"},
                ))
                assert "dark roast" in r and "id:" in r

                # filters
                r = text(await session.call_tool(
                    "recall", {"query": "coffee", "session_id": "conv_001", "role": "user"},
                ))
                assert "dark roast" in r and "Great choice" not in r

                # bundles
                r = text(await session.call_tool(
                    "recall", {"query": "grand canyon", "bundles": True, "strategy": "direct"},
                ))
                assert "## Session conv_002" in r

                # list_sessions
                r = text(await session.call_tool("list_sessions", {}))
                assert "conv_001" in r and "conv_002" in r

                # fetch_session
                r = text(await session.call_tool("fetch_session", {"session_id": "conv_001"}))
                assert "dark roast" in r and "Great choice" in r

                # delete by id from recall output
                import re
                recall_out = text(await session.call_tool(
                    "recall", {"query": "grand canyon", "limit": 1, "strategy": "direct"},
                ))
                mid = re.search(r"id: ([a-z0-9-]+)", recall_out)
                assert mid
                r = text(await session.call_tool("delete", {"message_ids": [mid.group(1)]}))
                assert "deleted: 1" in r
                assert "messages: 2" in text(await session.call_tool("stats", {}))

    asyncio.run(run())
