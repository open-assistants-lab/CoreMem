"""MCP round-trip tests for the fact-layer tools (0.16.x/0.17 fact layer).

Follows the subprocess+client scaffolding from tests/test_mcp_server.py:
spawns `run_mcp_server` over stdio and exercises the tools as a real client.
"""
from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("mcp")


def test_mcp_fact_tools_roundtrip():
    """End-to-end: spawn the stdio server on a fresh versioned store, record a
    fact, list it, and read its provenance — the governed interface."""
    import asyncio
    import re
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    memory_path = tempfile.mkdtemp(prefix="coremem-mcp-facts-") + "/hybrid"
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
                for expected in ("add_fact", "list_facts", "fact_history"):
                    assert expected in names, (expected, names)

                text = lambda r: r.content[0].text

                # add_fact returns the recorded fact id
                r = text(await session.call_tool("add_fact", {
                    "entity": "user", "attribute": "dog", "value": "Max",
                }))
                assert "fact recorded:" in r

                # list_facts surfaces the fact with its id and value
                listed = text(await session.call_tool("list_facts", {"entity": "user"}))
                assert "user.dog = Max" in listed
                match = re.search(r"([a-z0-9-]+): user\.dog = Max", listed)
                assert match, listed
                fact_id = match.group(1)

                # fact_history exposes the insert event on the audit chain
                hist = text(await session.call_tool("fact_history", {"fact_id": fact_id}))
                assert '"op": "insert"' in hist

    asyncio.run(run())