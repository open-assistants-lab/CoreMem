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