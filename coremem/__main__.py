"""CoreMem CLI — memory for AI agents."""
from __future__ import annotations

import argparse
import json
import os
import sys

from coremem import get_core
from coremem.hooks.handler import EVENT_HANDLERS


def _format_recall(results: list) -> str:
    if not results:
        return "No results."
    lines = []
    for r in results:
        content = r.memory.content[:200]
        lines.append(f"[{r.memory.role}] {content} (score: {r.score:.2f}, session: {r.memory.session_id})")
    return "\n".join(lines)


def _format_bundles(bundles: list) -> str:
    if not bundles:
        return "No bundles."
    lines = []
    for b in bundles:
        lines.append(f"## Session {b.session_id} (score: {b.score:.2f})")
        for m in b.messages:
            lines.append(f"  [{m.role}] {m.content[:200]}")
    return "\n".join(lines)


def _format_sessions(rows: list[dict]) -> str:
    if not rows:
        return "No sessions."
    lines = ["session_id | messages | last_ts"]
    lines.append("--- | --- | ---")
    for row in rows:
        lines.append(f"{row['session_id']} | {row['messages']} | {row['last_ts'] or ''}")
    return "\n".join(lines)


def _cmd_recall(args, core):
    results = core.recall(
        args.query,
        strategy=args.strategy,
        limit=args.limit,
        bundles=args.bundles,
        session_id=args.session_id or None,
    )
    if args.bundles:
        print(_format_bundles(results))
    else:
        print(_format_recall(results))


def _cmd_ingest(args, core):
    turn_id = core.ingest(args.role, args.content, session_id=args.session_id or None, user_id=args.user_id)
    print(f"turn_id: {turn_id}")


def _cmd_compile(args, core):
    import asyncio
    if not core._llm_provider:
        print("error: no LLM provider configured (set COREMEM_LLM_MODEL env var)")
        return
    result = asyncio.run(core.compile_turn(turn_id=args.turn_id))
    if result is None:
        print("compiled: 0 pages (nothing to compile)")
        return
    print(f"compiled: {len(result.written_pages)} pages")


def _cmd_rebuild(args, core):
    result = core.rebuild_index()
    pages = result.get("pages", 0)
    print(f"rebuilt: {pages} pages")


def _cmd_sessions(args, core):
    rows = core._db.raw_query(
        "SELECT session_id, COUNT(*) as messages, MAX(ts) as last_ts "
        "FROM messages WHERE session_id IS NOT NULL GROUP BY session_id "
        "ORDER BY last_ts DESC"
    )
    print(_format_sessions(rows))


def _cmd_hook(args, core):
    data = json.loads(sys.stdin.read())
    handler = EVENT_HANDLERS.get(args.event)
    if handler is None:
        print(json.dumps({}))
        return
    result = handler(data, core)
    print(json.dumps(result))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="coremem", description="CoreMem — memory for AI agents")
    parser.add_argument("--path", default=None, help="Memory path (default: COREMEM_PATH env or ~/.coremem/hybrid)")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("mcp", help="Start MCP stdio server")

    p_recall = sub.add_parser("recall", help="Retrieve memories")
    p_recall.add_argument("query", help="Search query")
    p_recall.add_argument("--strategy", default="episodic", choices=["direct", "episodic", "expanded", "fusion"])
    p_recall.add_argument("--limit", type=int, default=5)
    p_recall.add_argument("--bundles", action="store_true")
    p_recall.add_argument("--session-id", default="")

    p_ingest = sub.add_parser("ingest", help="Store a message")
    p_ingest.add_argument("role", choices=["user", "assistant", "system"])
    p_ingest.add_argument("content", help="Message content")
    p_ingest.add_argument("--session-id", default="")
    p_ingest.add_argument("--user-id", default="")

    p_compile = sub.add_parser("compile", help="Compile a turn into journal pages")
    p_compile.add_argument("turn_id", help="Turn ID to compile")

    sub.add_parser("rebuild", help="Rebuild search index")
    sub.add_parser("sessions", help="List sessions")

    p_hook = sub.add_parser("hook", help="Run hook handler (reads JSON from stdin)")
    p_hook.add_argument("event", choices=list(EVENT_HANDLERS.keys()))

    args = parser.parse_args(argv)

    if args.command is None:
        args.command = "mcp"

    if args.command == "mcp":
        from coremem.mcp_server import run_mcp_server
        run_mcp_server(path=args.path)
        return

    core = get_core(path=args.path)

    handlers = {
        "recall": _cmd_recall,
        "ingest": _cmd_ingest,
        "compile": _cmd_compile,
        "rebuild": _cmd_rebuild,
        "sessions": _cmd_sessions,
        "hook": _cmd_hook,
    }
    handler = handlers.get(args.command)
    if handler:
        handler(args, core)


if __name__ == "__main__":
    main()