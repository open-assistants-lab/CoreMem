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
