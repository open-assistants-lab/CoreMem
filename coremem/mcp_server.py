"""CoreMem MCP server — exposes memory tools to AI agents via stdio."""
from __future__ import annotations

from coremem import get_core


def format_recall_results(results: list) -> str:
    if not results:
        return "No results."
    lines = []
    for r in results:
        content = r.memory.content[:200]
        lines.append(f"[{r.memory.role}] {content} (score: {r.score:.2f}, session: {r.memory.session_id})")
    return "\n".join(lines)


def format_bundles(bundles: list) -> str:
    if not bundles:
        return "No bundles."
    lines = []
    for b in bundles:
        lines.append(f"## Session {b.session_id} (score: {b.score:.2f})")
        for m in b.messages:
            lines.append(f"  [{m.role}] {m.content[:200]}")
    return "\n".join(lines)


def format_sessions(rows: list[dict]) -> str:
    if not rows:
        return "No sessions."
    lines = ["session_id | messages | last_ts", "--- | --- | ---"]
    for row in rows:
        lines.append(f"{row['session_id']} | {row['messages']} | {row['last_ts'] or ''}")
    return "\n".join(lines)


def run_mcp_server(path: str | None = None) -> None:
    """Start the MCP stdio server."""
    from mcp.server import MCPServer

    core = get_core(path=path)
    mcp = MCPServer("coremem")

    @mcp.tool()
    async def recall(
        query: str,
        strategy: str = "episodic",
        limit: int = 5,
        bundles: bool = False,
        session_id: str = "",
    ) -> str:
        """Retrieve memories by query strategy.

        Strategies: "episodic" (default), "direct", "expanded", "fusion".
        Set bundles=True for session context around each hit.
        """
        results = core.recall(
            query, strategy=strategy, limit=limit,
            bundles=bundles, session_id=session_id or None,
        )
        if bundles:
            return format_bundles(results)
        return format_recall_results(results)

    @mcp.tool()
    async def ingest(role: str, content: str, session_id: str = "", user_id: str = "") -> str:
        """Store a message in memory. Returns the turn_id."""
        turn_id = core.ingest(role, content, session_id=session_id or None, user_id=user_id)
        return f"turn_id: {turn_id}"

    @mcp.tool()
    async def compile(turn_id: str) -> str:
        """Compile a turn into daily journal pages."""
        try:
            result = await core.compile_turn(turn_id=turn_id)
        except Exception as exc:
            return f"error: compile failed: {exc}"
        pages = len(result.written_pages) if hasattr(result, 'written_pages') else 0
        return f"compiled: {pages} pages"

    @mcp.tool()
    async def rebuild_index() -> str:
        """Rebuild the agent journal search index."""
        result = core.rebuild_index()
        pages = result.get("pages", 0) if isinstance(result, dict) else 0
        return f"rebuilt: {pages} pages"

    @mcp.tool()
    async def list_sessions() -> str:
        """List all sessions with message counts."""
        rows = core._db.raw_query(
            "SELECT session_id, COUNT(*) as messages, MAX(ts) as last_ts "
            "FROM messages WHERE session_id IS NOT NULL GROUP BY session_id "
            "ORDER BY last_ts DESC"
        )
        return format_sessions(rows)

    mcp.run(transport="stdio")
