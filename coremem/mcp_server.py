"""CoreMem MCP server — exposes memory tools to AI agents via stdio.

Tool design notes (AI-first):
- Every tool description states what it returns and when to use it.
- recall output includes message ids so agents can act on them (delete).
- All memory hygiene operations are first-class tools: recall, ingest,
  delete, fetch_session, stats, list_sessions.
"""

from __future__ import annotations


def format_recall_results(results: list) -> str:
    if not results:
        return "No results."
    lines = []
    for r in results:
        content = r.memory.content[:500]
        lines.append(
            f"[{r.memory.role}] (id: {r.memory.id}, score: {r.score:.2f}, "
            f"session: {r.memory.session_id}) {content}"
        )
    return "\n".join(lines)


def format_bundles(bundles: list) -> str:
    if not bundles:
        return "No bundles."
    lines = []
    for b in bundles:
        lines.append(f"## Session {b.session_id} (score: {b.score:.2f})")
        for m in b.messages:
            lines.append(f"  [{m.role}] (id: {m.id}) {m.content[:500]}")
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

    from coremem import get_core

    core = get_core(path=path)
    mcp = MCPServer("coremem")

    @mcp.tool()
    async def recall(
        query: str,
        strategy: str = "episodic",
        limit: int = 5,
        bundles: bool = False,
        session_id: str = "",
        role: str = "",
        ts_after: str = "",
        ts_before: str = "",
        session_cap: int = 1,
    ) -> str:
        """Retrieve memories relevant to a query. Returns ranked messages (or
        session bundles with surrounding context when bundles=True), each with
        its message id so you can act on results.

        Strategies: "episodic" (default, zero LLM calls), "direct" (plain
        hybrid search), "expanded" (1 LLM call for query rephrasing — use when
        precision matters more than cost), "fusion" (2x compute, most session
        diversity).

        Examples:
        - recall(query="what coffee does the user like?")
        - recall(query="when did we deploy?", strategy="episodic", bundles=True)
        - recall(query="project plan", session_id="conv_001", role="user")
        - recall(query="preferences", session_cap=2)  # up to 2 msgs per session
        """
        results = core.recall(
            query, strategy=strategy, limit=limit,
            bundles=bundles,
            session_id=session_id or None,
            role=role or None,
            ts_after=ts_after or None,
            ts_before=ts_before or None,
            session_cap=session_cap,
        )
        if bundles:
            return format_bundles(results)
        return format_recall_results(results)

    @mcp.tool()
    async def ingest(
        role: str, content: str,
        session_id: str = "", user_id: str = "",
        metadata: str = "{}",
    ) -> str:
        """Store one message in memory. Returns the turn_id (needed for the
        compile tool). Use a session_id to group a conversation; the same
        turn_id groups the user message and the assistant reply.

        Example: ingest(role="user", content="I prefer dark roast coffee",
        session_id="conv_001")
        """
        import json as _json

        try:
            meta = _json.loads(metadata) if metadata else {}
        except (TypeError, ValueError):
            meta = {}
        turn_id = core.ingest(
            role, content,
            session_id=session_id or None,
            user_id=user_id,
            metadata=meta,
        )
        return f"turn_id: {turn_id}"

    @mcp.tool()
    async def delete(message_ids: list[str]) -> str:
        """Delete specific memories by message id (ids appear in recall output).
        Use when a stored memory is wrong, superseded, or should not be
        remembered. Returns the number of messages deleted.

        Example: delete(message_ids=["a1b2c3d4e5f6"])
        """
        n = core.delete_messages(message_ids)
        return f"deleted: {n} messages"

    @mcp.tool()
    async def fetch_session(session_id: str, limit: int = 50) -> str:
        """Fetch messages from one session (newest first). Use to read the
        full conversation context of a session found via recall or
        list_sessions.

        Example: fetch_session(session_id="conv_001")
        """
        memories = core.fetch(session_id=session_id, limit=limit)
        if not memories:
            return f"No messages in session {session_id}."
        lines = [f"session {session_id} ({len(memories)} messages, newest first):"]
        for m in memories:
            lines.append(f"[{m.role}] (id: {m.id}) {m.content[:500]}")
        return "\n".join(lines)

    @mcp.tool()
    async def list_sessions(limit: int = 50) -> str:
        """List sessions with message counts, most recent first. Use to see
        what conversations exist in memory.

        Example: list_sessions()
        """
        rows = core.list_sessions()
        return format_sessions(rows[:limit])

    @mcp.tool()
    async def stats() -> str:
        """Return memory statistics (message/session/user counts, last write,
        pending index work). Use for health checks and to decide whether
        memory is initialized.

        Example: stats()
        """
        s = core.stats()
        return (
            f"messages: {s['messages']}, sessions: {s['sessions']}, "
            f"users: {s['users']}, last_ts: {s['last_ts'] or 'n/a'}, "
            f"journal_pending: {s['journal_pending']}"
        )

    @mcp.tool()
    async def compile(turn_id: str) -> str:
        """Compile a turn (from ingest) into daily journal pages. One LLM call
        per turn; use for long-term consolidation, not per-message.

        Example: compile(turn_id="a1b2c3d4e5f6")
        """
        try:
            result = await core.compile_turn(turn_id=turn_id)
        except Exception as exc:
            return f"error: compile failed: {exc}"
        if result is None:
            return "compiled: 0 pages (turn unchanged or unknown)"
        pages = len(result.written_pages) if hasattr(result, "written_pages") else 0
        return f"compiled: {pages} pages"

    @mcp.tool()
    async def rebuild_index() -> str:
        """Rebuild the agent journal search index (navigation files)."""
        result = core.rebuild_index()
        pages = result.get("pages", 0) if isinstance(result, dict) else 0
        return f"rebuilt: {pages} pages"

    mcp.run(transport="stdio")
