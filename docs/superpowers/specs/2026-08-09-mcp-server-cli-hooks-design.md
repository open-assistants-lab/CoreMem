# CoreMem MCP Server + CLI + Hooks Design

**Date:** 2026-08-09
**Status:** Approved

## Motivation

CoreMem needs to integrate with AI coding agents (Claude Code, OpenCode, Codex)
so agents can store and retrieve conversation history. Research shows the
standard pattern is **hybrid**: lifecycle hooks for automatic capture, MCP tools
for retrieval. Pure agent-driven `ingest()` is fragile — agents forget to save.

## Architecture

```
coremem/
├── mcp_server.py           # MCP stdio server (5 tools)
├── __main__.py             # CLI entry point (argparse subcommands)
├── hooks/
│   ├── __init__.py
│   └── handler.py           # Hook handler (reads JSON from stdin, dispatches)
integrations/
├── claude-code/
│   ├── hooks.json           # Hook config template
│   └── README.md            # Setup instructions
├── opencode/
│   ├── opencode.json        # MCP config template (no hooks in v1)
│   └── README.md
├── codex/
│   ├── hooks.json           # Hook config template (same JSON format as Claude Code)
│   └── README.md
```

## MCP Server (5 tools)

Uses the official Python MCP SDK (`mcp` package, 2.0+). Stdio transport.

The MCP server is a **long-lived process** — one MemoryCore instance lives for
the process lifetime. The cross-encoder model loads once and stays in memory.

| Tool | Maps to | Description |
|---|---|---|
| `recall` | `core.recall(query, strategy, limit, bundles, session_id)` | Retrieve memories. Returns formatted text. |
| `ingest` | `core.ingest(role, content, session_id, user_id)` | Store a message. Returns turn_id. |
| `compile` | `asyncio.run(core.compile_turn(turn_id))` | Compile a turn into journal pages. Requires LLM provider. |
| `rebuild_index` | `core.rebuild_index()` | Rebuild agent journal search index. |
| `list_sessions` | SQL on messages table | List distinct session_ids with message counts. |

### Tool signatures

```python
@mcp.tool()
async def recall(query: str, strategy: str = "episodic", limit: int = 5,
                 bundles: bool = False, session_id: str = "") -> str:
    """Retrieve memories by query strategy.

    Strategies: "episodic" (default), "direct", "expanded", "fusion".
    Set bundles=True for session context around each hit.
    """

@mcp.tool()
async def ingest(role: str, content: str, session_id: str = "",
                user_id: str = "") -> str:
    """Store a message in memory. Returns the turn_id."""

@mcp.tool()
async def compile(turn_id: str) -> str:
    """Compile a turn into daily journal pages. Requires LLM provider configured."""

@mcp.tool()
async def rebuild_index() -> str:
    """Rebuild the agent journal search index from stored journal pages."""

@mcp.tool()
async def list_sessions() -> str:
    """List all sessions with message counts and timestamps."""
```

### Return format

All tools return text (MCP `TextContent`).

- `recall` (no bundles): one line per result: `[role] content (score: 0.42, session: s1)`
- `recall` (bundles): one block per bundle: `## Session s1 (score: 0.42)\n[role] content\n...`
- `ingest`: `turn_id: abc123`
- `compile`: `compiled: 3 pages, 0 errors` or `error: no LLM provider configured`
- `rebuild_index`: `rebuilt: 42 pages`
- `list_sessions`: table: `session_id | messages | last_ts`

## CLI

```
coremem                              # Start MCP stdio server (default)
coremem mcp                          # Start MCP stdio server (explicit)
coremem recall <query>               # Quick retrieval (strategy=episodic)
coremem recall <query> --strategy direct --limit 10
coremem recall <query> --bundles
coremem ingest <role> <content>      # Quick storage
coremem ingest user "hello" --session-id s1
coremem compile <turn_id>            # Compile a turn
coremem rebuild                      # Rebuild index
coremem sessions                     # List sessions
coremem hook <event>                 # Run hook handler (reads JSON from stdin)
coremem --path /custom/path mcp      # Override memory path
```

`coremem` with no subcommand starts the MCP server.

## Hooks (Claude Code + Codex)

Claude Code and Codex share the **same hook wire protocol** (stdin JSON, stdout
JSON, exit codes). A single Python handler serves both platforms.

### Hook events

| Event | Type | What it does |
|---|---|---|
| `UserPromptSubmit` | Sync (8s timeout) | **Capture + retrieval injection.** Read user's prompt from stdin `prompt` field, call `ingest("user", prompt)`, then `recall(strategy="direct")`, output results as `additionalContext`. Uses `strategy="direct"` (no cross-encoder) for speed — hook processes are ephemeral and can't afford the ~500MB model load. |
| `Stop` | Async (30s timeout) | **Capture.** Read `last_assistant_message` from stdin, call `ingest("assistant", response, session_id)`. The user prompt was already captured by UserPromptSubmit. |
| `PreCompact` | Sync (30s timeout) | **No-op in v1.** PreCompact stdin does not include `last_assistant_message` (only `trigger` and `custom_instructions`). Capturing would require transcript parsing, which we're avoiding in v1. Stop already captures every assistant response. PreCompact is listed in the config for forward-compatibility — the handler reads stdin and exits 0 without doing anything. |

### Why `strategy="direct"` for hook retrieval

Hook scripts are **ephemeral processes** — each invocation creates a MemoryCore,
does its work, and exits. The cross-encoder model load (~500MB, several seconds)
is too slow for the 8s UserPromptSubmit timeout. `strategy="direct"` (BM25+hybrid)
is fast (SQLite open + query) and still achieves 93.8% session recall on oracle.

The MCP server (long-lived process) uses `strategy="episodic"` by default —
the cross-encoder loads once and stays in memory.

### Capture flow

```
UserPromptSubmit hook fires:
  1. Read JSON from stdin: {session_id, prompt, ...}
  2. Create MemoryCore(path=~/.coremem/hybrid)
  3. core.recall(prompt, strategy="direct", limit=5)
     → recall FIRST (before ingest, so results don't include self-match)
  4. core.ingest("user", prompt, session_id=session_id)
     → auto-generates new turn_id for this user message
  5. Format results as text
  6. Output: {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
             "additionalContext": "## Relevant memories\n[user] ..."}}
  7. Exit 0

(Claude processes the prompt + injected context, generates response)

Stop hook fires:
  1. Read JSON from stdin: {session_id, last_assistant_message, ...}
  2. Create MemoryCore(path=~/.coremem/hybrid)
  3. core.ingest("assistant", last_assistant_message, session_id=session_id)
     → reuses the turn_id from the last user message in this session
  4. Exit 0 (no output needed)
```

### Why no transcript parsing

The hook payload provides everything we need:
- `prompt` (UserPromptSubmit) → the user's message
- `last_assistant_message` (Stop) → the assistant's response
- `session_id` → groups messages into sessions

No need to parse the transcript.jsonl file (which has 7 types of `role:"user"`
impostors and complex content block structures). Each turn is captured as it
happens, in real time.

### SessionStart hook

**Not used in v1.** SessionStart has no query to search with — the user hasn't
typed anything yet. UserPromptSubmit is the primary retrieval injection point
and has the actual query (the user's prompt).

### `coremem hook <event>` CLI command

Hook configs call `coremem hook user_prompt_submit`, `coremem hook stop`, etc.
This reads JSON from stdin, dispatches to the handler, and outputs JSON to stdout.

```python
# coremem/hooks/handler.py
def handle_user_prompt_submit(data: dict, core: MemoryCore) -> dict:
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

def handle_stop(data: dict, core: MemoryCore) -> dict:
    message = data.get("last_assistant_message", "")
    session_id = data.get("session_id", "")
    if not message.strip():
        return {}
    core.ingest("assistant", message, session_id=session_id)
    return {}
```

## Prerequisites

```bash
pip install coremem[mcp]
# or
uv tool install coremem[mcp]
# or
uvx coremem mcp  # run without installing
```

The `coremem` command must be on PATH for both hooks and MCP server configs.
Verify with `coremem --help`.

## LLM Provider for `compile` tool

The `compile` tool requires an LLM provider (for daily journal compilation).
Configure via the `COREMEM_LLM_PROVIDER` env var — one of:
- `openai` (uses `OPENAI_API_KEY`)
- `ollama` (uses local Ollama)
- `ollama-cloud` (uses `OLLAMA_CLOUD_API_KEY`)

If not set, `compile` returns `error: no LLM provider configured`. The other
tools (`recall`, `ingest`, `rebuild_index`, `list_sessions`) work without an
LLM provider.

## Memory Path

- Default: `~/.coremem/` (HybridDB at `~/.coremem/hybrid`)
- Override: `COREMEM_PATH` env var or `--path` CLI flag
- `--path` takes precedence over env var, which takes precedence over default

## Dependencies

```toml
[project.scripts]
coremem = "coremem.__main__:main"

[project.optional-dependencies]
mcp = ["mcp>=2.0.0"]
all = ["coremem[observer]", "coremem[mcp]"]
```

Hooks use only stdlib (json, sys, pathlib) — no extra dependency.

## Integration Configs

### Claude Code (`.claude/settings.json`)

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

### Codex (`~/.codex/hooks.json` + `~/.codex/config.toml`)

hooks.json:
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

config.toml:
```toml
[mcp_servers.coremem]
command = "coremem"
args = ["mcp"]
```

### OpenCode (`opencode.json`)

MCP-only (no hooks in v1 — OpenCode plugin is TypeScript, deferred to v2):

```json
{
  "mcp": {
    "coremem": { "type": "local", "command": ["coremem", "mcp"] }
  }
}
```

## OpenCode (v1: MCP-only)

OpenCode gets MCP tools only in v1. The agent must manually call `recall` and
`ingest`. This is the "fragile" pattern but the best we can do without writing
a TypeScript plugin. The plugin (v2) would use `ctx.session.hook("context")`
for retrieval injection and `tool.execute.after` for capture.

## What's NOT included (YAGNI)

- No HTTP/SSE transport (stdio only)
- No auth (local process, no network exposure)
- No multi-tenancy (one server = one memory store)
- No OpenCode plugin in v1 (TypeScript, deferred)
- No SessionStart hook (no query to search with)
- No transcript parsing (hook payload has everything)
- No resource endpoints or prompt templates (tools only)

## Testing

- MCP server: test tool registration and output formatting
- CLI: test each subcommand with a temp MemoryCore
- Hooks: test handler with fixture JSON payloads (stdin simulation)
- Integration: manual test with Claude Code + Codex configs

## Version

v0.12.0 — adds `mcp` extra, `coremem` CLI entry point, hook handlers.