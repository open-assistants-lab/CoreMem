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