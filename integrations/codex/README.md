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