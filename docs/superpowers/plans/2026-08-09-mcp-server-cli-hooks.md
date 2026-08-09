# MCP Server + CLI + Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an MCP server, CLI, and lifecycle hooks to CoreMem so AI coding agents (Claude Code, Codex, OpenCode) can automatically capture and retrieve conversation memory.

**Architecture:** MCP stdio server exposes 5 tools (recall, ingest, compile, rebuild_index, list_sessions). CLI wraps the same functionality plus `coremem hook <event>` for hook handlers. Hook handlers read JSON from stdin (Claude Code/Codex protocol), call MemoryCore, and output JSON to stdout. Memory path defaults to `~/.coremem/`.

**Tech Stack:** Python 3.11+, MCP SDK 2.0+ (optional dep), argparse, stdlib json/sys

**Spec:** `docs/superpowers/specs/2026-08-09-mcp-server-cli-hooks-design.md`

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `coremem/hooks/__init__.py` | Create | Package init |
| `coremem/hooks/handler.py` | Create | Hook event handlers (user_prompt_submit, stop, pre_compact) |
| `coremem/mcp_server.py` | Create | MCP stdio server with 5 tools |
| `coremem/__main__.py` | Create | CLI entry point (argparse subcommands) |
| `coremem/__init__.py` | Modify | Export `get_core()` helper |
| `pyproject.toml` | Modify | Add `[project.scripts]` + `mcp` extra |
| `integrations/claude-code/hooks.json` | Create | Hook config template |
| `integrations/claude-code/README.md` | Create | Setup instructions |
| `integrations/codex/hooks.json` | Create | Hook config template |
| `integrations/codex/README.md` | Create | Setup instructions |
| `integrations/opencode/opencode.json` | Create | MCP config template |
| `integrations/opencode/README.md` | Create | Setup instructions |
| `tests/test_hooks.py` | Create | Hook handler tests |
| `tests/test_mcp_server.py` | Create | MCP server tool tests |
| `tests/test_cli.py` | Create | CLI subcommand tests |

---

## Task 1: Add `get_core()` helper to `coremem/__init__.py`

A shared helper that creates a MemoryCore from env vars / defaults. Used by MCP server, CLI, and hooks.

**Files:**
- Modify: `coremem/__init__.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_core.py` at the end:

```python
def test_get_core_creates_memorycore_with_defaults(tmp_path):
    import os
    old_path = os.environ.get("COREMEM_PATH")
    os.environ["COREMEM_PATH"] = str(tmp_path / "coremem-test")
    try:
        from coremem import get_core
        core = get_core()
        assert core is not None
        assert hasattr(core, "recall")
        assert hasattr(core, "ingest")
    finally:
        if old_path is None:
            del os.environ["COREMEM_PATH"]
        else:
            os.environ["COREMEM_PATH"] = old_path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python3 -m pytest tests/test_core.py::test_get_core_creates_memorycore_with_defaults -v`
Expected: FAIL with `ImportError: cannot import name 'get_core'`

- [ ] **Step 3: Add `get_core()` to `coremem/__init__.py`**

Add after the `__all__` list at the bottom of `coremem/__init__.py`:

```python
def get_core(path: str | None = None) -> MemoryCore:
    """Create a MemoryCore from path/env/default.

    Resolution: path arg > COREMEM_PATH env > ~/.coremem/hybrid
    """
    import os
    resolved = path or os.environ.get("COREMEM_PATH") or os.path.expanduser("~/.coremem/hybrid")
    model_string = os.environ.get("COREMEM_LLM_MODEL")
    llm_provider = None
    if model_string:
        llm_provider = create_provider(model_string)
    return MemoryCore(path=resolved, llm_provider=llm_provider)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python3 -m pytest tests/test_core.py::test_get_core_creates_memorycore_with_defaults -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 116 passed (115 existing + 1 new)

- [ ] **Step 6: Commit**

```bash
git add coremem/__init__.py tests/test_core.py
git commit -m "feat: add get_core() helper for CLI/MCP/hooks"
```

---

## Task 2: Hook handlers

Hook event handlers that read JSON payloads and call MemoryCore.

**Files:**
- Create: `coremem/hooks/__init__.py`
- Create: `coremem/hooks/handler.py`
- Create: `tests/test_hooks.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hooks.py`:

```python
"""Tests for hook handlers."""
from __future__ import annotations

import json
import os
import tempfile
import shutil
from pathlib import Path

import pytest
from coremem.hooks.handler import (
    _format_recall_results,
    handle_user_prompt_submit,
    handle_stop,
    handle_pre_compact,
)


def _make_core(tmp_path: Path):
    from coremem import get_core
    os.environ["COREMEM_PATH"] = str(tmp_path / "hybrid")
    return get_core()


def test_format_recall_results_empty():
    assert _format_recall_results([]) == ""


def test_format_recall_results_with_hits(tmp_path):
    from coremem import get_core
    os.environ["COREMEM_PATH"] = str(tmp_path / "hybrid")
    core = get_core()
    core.ingest("user", "I love hiking in Yosemite", session_id="s1")
    results = core.recall("hiking", strategy="direct", limit=5)
    formatted = _format_recall_results(results)
    assert "## Relevant memories" in formatted
    assert "[user]" in formatted
    assert "hiking" in formatted
    assert "score:" in formatted


def test_handle_user_prompt_submit_captures_and_retrieves(tmp_path):
    core = _make_core(tmp_path)
    core.ingest("user", "I love hiking in Yosemite", session_id="s1")
    data = {"prompt": "hiking", "session_id": "sess-abc"}
    result = handle_user_prompt_submit(data, core)
    assert "hookSpecificOutput" in result
    assert result["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "## Relevant memories" in result["hookSpecificOutput"]["additionalContext"]


def test_handle_user_prompt_submit_empty_prompt(tmp_path):
    core = _make_core(tmp_path)
    result = handle_user_prompt_submit({"prompt": "", "session_id": "s1"}, core)
    assert result == {}


def test_handle_stop_captures_assistant_message(tmp_path):
    core = _make_core(tmp_path)
    core.ingest("user", "What about hiking?", session_id="s1")
    data = {"last_assistant_message": "Hiking is great!", "session_id": "s1"}
    result = handle_stop(data, core)
    assert result == {}
    results = core.recall("Hiking", strategy="direct", limit=5)
    assert any("Hiking is great" in r.memory.content for r in results)


def test_handle_stop_empty_message(tmp_path):
    core = _make_core(tmp_path)
    result = handle_stop({"last_assistant_message": "", "session_id": "s1"}, core)
    assert result == {}


def test_handle_pre_compact_noop(tmp_path):
    core = _make_core(tmp_path)
    result = handle_pre_compact({"trigger": "auto"}, core)
    assert result == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python3 -m pytest tests/test_hooks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coremem.hooks'`

- [ ] **Step 3: Create `coremem/hooks/__init__.py`**

Create an empty file:

```python
"""Hook handlers for agent lifecycle events."""
```

- [ ] **Step 4: Create `coremem/hooks/handler.py`**

```python
"""Hook event handlers for Claude Code and Codex.

Both platforms share the same stdin JSON / stdout JSON wire protocol.
"""
from __future__ import annotations

from typing import Any

from coremem import MemoryCore


def _format_recall_results(results: list) -> str:
    if not results:
        return ""
    lines = ["## Relevant memories"]
    for r in results:
        role = r.memory.role
        content = r.memory.content[:200]
        score = r.score
        lines.append(f"[{role}] {content} (score: {score:.2f})")
    return "\n".join(lines)


def handle_user_prompt_submit(data: dict[str, Any], core: MemoryCore) -> dict[str, Any]:
    prompt = data.get("prompt", "")
    session_id = data.get("session_id", "")
    if not prompt.strip():
        return {}
    results = core.recall(prompt, strategy="direct", limit=5)
    core.ingest("user", prompt, session_id=session_id)
    context = _format_recall_results(results)
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }


def handle_stop(data: dict[str, Any], core: MemoryCore) -> dict[str, Any]:
    message = data.get("last_assistant_message", "")
    session_id = data.get("session_id", "")
    if not message.strip():
        return {}
    core.ingest("assistant", message, session_id=session_id)
    return {}


def handle_pre_compact(data: dict[str, Any], core: MemoryCore) -> dict[str, Any]:
    return {}


EVENT_HANDLERS = {
    "user_prompt_submit": handle_user_prompt_submit,
    "stop": handle_stop,
    "pre_compact": handle_pre_compact,
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python3 -m pytest tests/test_hooks.py -v`
Expected: 6 passed

- [ ] **Step 6: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 122 passed (116 + 6 new)

- [ ] **Step 7: Commit**

```bash
git add coremem/hooks/ tests/test_hooks.py
git commit -m "feat: add hook handlers for Claude Code and Codex"
```

---

## Task 3: CLI entry point (`__main__.py`)

Argparse-based CLI with subcommands: mcp, recall, ingest, compile, rebuild, sessions, hook.

**Files:**
- Create: `coremem/__main__.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
"""Tests for the CLI."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _run_cli(args: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = {**os.environ}
    if env:
        full_env.update(env)
    return subprocess.run(
        [sys.executable, "-m", "coremem"] + args,
        capture_output=True,
        text=True,
        env=full_env,
        timeout=30,
    )


def test_cli_help():
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "recall" in result.stdout
    assert "ingest" in result.stdout
    assert "mcp" in result.stdout
    assert "hook" in result.stdout


def test_cli_recall(tmp_path):
    env = {"COREMEM_PATH": str(tmp_path / "hybrid")}
    _run_cli(["ingest", "user", "I love hiking in Yosemite", "--session-id", "s1"], env=env)
    result = _run_cli(["recall", "hiking", "--strategy", "direct"], env=env)
    assert result.returncode == 0
    assert "hiking" in result.stdout.lower()


def test_cli_ingest(tmp_path):
    env = {"COREMEM_PATH": str(tmp_path / "hybrid")}
    result = _run_cli(["ingest", "user", "hello world", "--session-id", "s1"], env=env)
    assert result.returncode == 0
    assert "turn_id:" in result.stdout


def test_cli_sessions(tmp_path):
    env = {"COREMEM_PATH": str(tmp_path / "hybrid")}
    _run_cli(["ingest", "user", "hello", "--session-id", "s1"], env=env)
    _run_cli(["ingest", "user", "world", "--session-id", "s2"], env=env)
    result = _run_cli(["sessions"], env=env)
    assert result.returncode == 0
    assert "s1" in result.stdout
    assert "s2" in result.stdout


def test_cli_hook_reads_stdin(tmp_path):
    env = {"COREMEM_PATH": str(tmp_path / "hybrid")}
    hook_input = json.dumps({"prompt": "test prompt", "session_id": "s1"})
    result = subprocess.run(
        [sys.executable, "-m", "coremem", "hook", "user_prompt_submit"],
        input=hook_input,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
        timeout=30,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert "hookSpecificOutput" in output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python3 -m pytest tests/test_cli.py -v`
Expected: FAIL (no `coremem/__main__.py`)

- [ ] **Step 3: Create `coremem/__main__.py`**

```python
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
    summary = asyncio.run(core.compile_turn(turn_id=args.turn_id))
    compiled = summary.get("compiled", [])
    errors = summary.get("errors", [])
    print(f"compiled: {len(compiled)} pages, {len(errors)} errors")


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python3 -m pytest tests/test_cli.py -v`
Expected: 5 passed

- [ ] **Step 5: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 127 passed (122 + 5 new)

- [ ] **Step 6: Commit**

```bash
git add coremem/__main__.py tests/test_cli.py
git commit -m "feat: add CLI with subcommands (recall, ingest, compile, rebuild, sessions, hook)"
```

---

## Task 4: MCP server

MCP stdio server with 5 tools. Uses `mcp` package (optional dependency).

**Files:**
- Create: `coremem/mcp_server.py`
- Create: `tests/test_mcp_server.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_server.py`:

```python
"""Tests for the MCP server."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest


def test_format_recall_results():
    from coremem.mcp_server import format_recall_results
    from coremem import get_core
    os.environ["COREMEM_PATH"] = str(tempfile.mkdtemp() + "/hybrid")
    core = get_core()
    core.ingest("user", "I love hiking in Yosemite", session_id="s1")
    results = core.recall("hiking", strategy="direct", limit=5)
    formatted = format_recall_results(results)
    assert "[user]" in formatted
    assert "hiking" in formatted.lower()
    assert "score:" in formatted


def test_format_bundles():
    from coremem.mcp_server import format_bundles
    from coremem import get_core, SessionBundle
    os.environ["COREMEM_PATH"] = str(tempfile.mkdtemp() + "/hybrid")
    core = get_core()
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python3 -m pytest tests/test_mcp_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'coremem.mcp_server'`

- [ ] **Step 3: Create `coremem/mcp_server.py`**

```python
"""CoreMem MCP server — exposes memory tools to AI agents via stdio."""
from __future__ import annotations

import os
from typing import Any

from coremem import get_core, MemoryCore


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
        if not core._llm_provider:
            return "error: no LLM provider configured (set COREMEM_LLM_MODEL env var)"
        import asyncio
        summary = await core.compile_turn(turn_id=turn_id)
        compiled = summary.get("compiled", [])
        errors = summary.get("errors", [])
        return f"compiled: {len(compiled)} pages, {len(errors)} errors"

    @mcp.tool()
    async def rebuild_index() -> str:
        """Rebuild the agent journal search index."""
        result = core.rebuild_index()
        pages = result.get("pages", 0)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python3 -m pytest tests/test_mcp_server.py -v`
Expected: 3 passed

- [ ] **Step 5: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 130 passed (127 + 3 new)

- [ ] **Step 6: Commit**

```bash
git add coremem/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: add MCP stdio server with 5 tools"
```

---

## Task 5: Update `pyproject.toml`

Add `[project.scripts]` entry point and `mcp` optional dependency.

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add scripts and mcp extra**

In `pyproject.toml`, after the `dependencies` list, add:

```toml
[project.scripts]
coremem = "coremem.__main__:main"
```

And update the optional dependencies:

```toml
[project.optional-dependencies]
observer = ["httpx>=0.25.0"]
mcp = ["mcp>=2.0.0"]
all = ["coremem[observer]", "coremem[mcp]"]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.25.0",
]
```

- [ ] **Step 2: Verify the entry point works**

Run: `uv run python3 -m coremem --help`
Expected: help output with all subcommands

- [ ] **Step 3: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 130 passed

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add coremem CLI entry point and mcp optional dependency"
```

---

## Task 6: Integration configs

Create config templates and README files for Claude Code, Codex, and OpenCode.

**Files:**
- Create: `integrations/claude-code/hooks.json`
- Create: `integrations/claude-code/README.md`
- Create: `integrations/codex/hooks.json`
- Create: `integrations/codex/README.md`
- Create: `integrations/opencode/opencode.json`
- Create: `integrations/opencode/README.md`

- [ ] **Step 1: Create `integrations/claude-code/hooks.json`**

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [{ "type": "command", "command": "coremem hook user_prompt_submit", "timeout": 8 }] }
    ],
    "Stop": [
      { "hooks": [{ "type": "command", "command": "coremem hook stop", "timeout": 30, "async": true }] }
    ],
    "PreCompact": [
      { "hooks": [{ "type": "command", "command": "coremem hook pre_compact", "timeout": 30 }] }
    ]
  },
  "mcpServers": {
    "coremem": { "command": "coremem", "args": ["mcp"] }
  }
}
```

- [ ] **Step 2: Create `integrations/claude-code/README.md`**

```markdown
# CoreMem for Claude Code

## Install

```bash
pip install coremem[mcp]
```

## Setup

Copy the hooks and MCP config into your project's `.claude/settings.json`:

```bash
cat integrations/claude-code/hooks.json >> .claude/settings.json
```

Or merge manually if you have existing config.

## How it works

- **UserPromptSubmit hook**: Captures your prompt, retrieves relevant memories, injects them as context.
- **Stop hook**: Captures the assistant's response after each turn.
- **MCP server**: Exposes `recall`, `ingest`, `compile`, `rebuild_index`, `list_sessions` tools.

## Configuration

- `COREMEM_PATH`: Memory storage path (default: `~/.coremem/hybrid`)
- `COREMEM_LLM_MODEL`: LLM model for journal compilation (e.g. `openai:gpt-4o-mini`)
```

- [ ] **Step 3: Create `integrations/codex/hooks.json`**

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [{ "type": "command", "command": "coremem hook user_prompt_submit", "timeout": 8 }] }
    ],
    "Stop": [
      { "hooks": [{ "type": "command", "command": "coremem hook stop", "timeout": 30, "async": true }] }
    ],
    "PreCompact": [
      { "hooks": [{ "type": "command", "command": "coremem hook pre_compact", "timeout": 30 }] }
    ]
  }
}
```

- [ ] **Step 4: Create `integrations/codex/README.md`**

```markdown
# CoreMem for Codex

## Install

```bash
pip install coremem[mcp]
```

## Setup

Copy hooks to `~/.codex/hooks.json`:

```bash
cp integrations/codex/hooks.json ~/.codex/hooks.json
```

Add MCP server to `~/.codex/config.toml`:

```toml
[mcp_servers.coremem]
command = "coremem"
args = ["mcp"]
```

## How it works

- **UserPromptSubmit hook**: Captures your prompt, retrieves relevant memories, injects them as context.
- **Stop hook**: Captures the assistant's response after each turn.
- **MCP server**: Exposes `recall`, `ingest`, `compile`, `rebuild_index`, `list_sessions` tools.

## Configuration

- `COREMEM_PATH`: Memory storage path (default: `~/.coremem/hybrid`)
- `COREMEM_LLM_MODEL`: LLM model for journal compilation (e.g. `openai:gpt-4o-mini`)
```

- [ ] **Step 5: Create `integrations/opencode/opencode.json`**

```json
{
  "mcp": {
    "coremem": { "type": "local", "command": ["coremem", "mcp"] }
  }
}
```

- [ ] **Step 6: Create `integrations/opencode/README.md`**

```markdown
# CoreMem for OpenCode

## Install

```bash
pip install coremem[mcp]
```

## Setup

Add the MCP server to your `opencode.json`:

```json
{
  "mcp": {
    "coremem": { "type": "local", "command": ["coremem", "mcp"] }
  }
}
```

## How it works

OpenCode gets MCP tools only in v1. The agent must manually call `recall` and
`ingest` tools. Automatic capture via hooks requires a TypeScript plugin (v2).

## Configuration

- `COREMEM_PATH`: Memory storage path (default: `~/.coremem/hybrid`)
- `COREMEM_LLM_MODEL`: LLM model for journal compilation (e.g. `openai:gpt-4o-mini`)
```

- [ ] **Step 7: Commit**

```bash
git add integrations/
git commit -m "feat: add integration configs for Claude Code, Codex, OpenCode"
```

---

## Task 7: Version bump and final verification

- [ ] **Step 1: Bump version in `pyproject.toml`**

Change `version = "0.11.0"` to `version = "0.12.0"`.

- [ ] **Step 2: Bump version in `coremem/__init__.py`**

Change `__version__ = "0.11.0"` to `__version__ = "0.12.0"`.

- [ ] **Step 3: Add CHANGELOG entry**

Add at the top of `CHANGELOG.md`:

```markdown
## [0.12.0] — MCP server, CLI, hooks

### Added
- **MCP server** (`coremem mcp`) — stdio transport, 5 tools: `recall`, `ingest`, `compile`, `rebuild_index`, `list_sessions`
- **CLI** (`coremem`) — subcommands: `recall`, `ingest`, `compile`, `rebuild`, `sessions`, `hook`, `mcp`
- **Hook handlers** for Claude Code and Codex — `UserPromptSubmit` (capture + retrieval injection), `Stop` (capture), `PreCompact` (no-op)
- **`get_core()` helper** — creates MemoryCore from `COREMEM_PATH` env var or `~/.coremem/` default
- **Integration configs** for Claude Code, Codex, and OpenCode in `integrations/`
- **`mcp` optional dependency** — `pip install coremem[mcp]`
- **`COREMEM_LLM_MODEL` env var** — configures LLM provider for `compile` tool

### Changed
- `pyproject.toml` — added `[project.scripts]` entry point, `mcp` extra
```

- [ ] **Step 4: Run full test suite**

Run: `uv run python3 -m pytest tests/ -q`
Expected: 130 passed

- [ ] **Step 5: Verify CLI works**

Run: `uv run python3 -m coremem --help`
Expected: help with all subcommands

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml coremem/__init__.py CHANGELOG.md
git commit -m "feat: bump to 0.12.0, add CHANGELOG entry"
```