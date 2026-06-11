"""Session-end tool message analyzer.

Reads all role='tool' messages for a session, produces
a structured tool_summary observation. Purely deterministic
analysis — no LLM required.

OSS boundary: No tool catalog, no skill registry, no domain
knowledge. Just message content analysis.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

_ERROR_KEYWORDS = re.compile(
    r"(Error:|error:|failed|not found|could not)", re.IGNORECASE
)


def _classify_error(content: str) -> tuple[bool, str | None]:
    """Heuristic error classification. Returns (is_error, error_type)."""
    if not content:
        return False, None
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    match = _ERROR_KEYWORDS.search(content)
    if match:
        return True, match.group(0).strip().rstrip(":")
    return False, None


def _build_trace(assistant_msgs: list[dict], tool_msgs: list[dict]) -> list[dict]:
    """Pair tool calls with their results using tool_call_id.

    Returns a chronological trace where each entry has:
    - tool_name: str
    - arguments: str (JSON from assistant.tool_calls[i].arguments)
    - result_content: str (from tool message content)
    - success: bool
    - error_type: str | None
    - error_classification: str — "heuristic" or None
    - call_id: str
    - recovery_call_id: str | None (if this was a retry)
    """
    tool_results = {m.get("tool_call_id", ""): m for m in tool_msgs if m.get("tool_call_id")}

    trace: list[dict] = []
    for msg in assistant_msgs:
        for tc in msg.get("tool_calls", []):
            call_id = tc.get("id", "")
            result = tool_results.get(call_id, {})
            content = result.get("content", "")
            is_error, error_type = _classify_error(content)
            trace.append({
                "call_id": call_id,
                "tool_name": tc.get("name", "unknown"),
                "arguments": tc.get("arguments", "{}"),
                "result_content": content,
                "success": not is_error,
                "error_type": error_type,
                "error_classification": "heuristic" if is_error else None,
            })

    # Detect recoveries: error → retry with same tool_name → success
    for i, entry in enumerate(trace):
        if not entry["success"]:
            for j in range(i + 1, min(i + 3, len(trace))):
                if (trace[j]["tool_name"] == entry["tool_name"]
                        and trace[j]["success"]):
                    entry["recovery_call_id"] = trace[j]["call_id"]
                    break

    return trace


def _analyze_deterministic(trace: list[dict]) -> dict:
    """Pure deterministic analysis of tool trace. No LLM."""
    error_by_tool: dict[str, list[str]] = {}
    tool_coverage: set[str] = set()
    recovery_by_tool: dict[str, list[str]] = {}
    sequence_counts: dict[str, int] = {}
    n_errors = 0

    # Build sequences of tool names (length 2 and 3)
    tool_names = [e["tool_name"] for e in trace]
    for seq_len in [2, 3]:
        for i in range(len(tool_names) - seq_len + 1):
            seq = "→".join(tool_names[i:i + seq_len])
            sequence_counts[seq] = sequence_counts.get(seq, 0) + 1

    for entry in trace:
        tool_coverage.add(entry["tool_name"])

        if not entry["success"]:
            n_errors += 1
            err = entry["error_type"] or "unknown"
            error_by_tool.setdefault(entry["tool_name"], []).append(err)

        if entry.get("recovery_call_id"):
            recovery_by_tool.setdefault(entry["tool_name"], []).append(
                "recovered_via_retry"
            )

    return {
        "n_errors": n_errors,
        "error_by_tool": error_by_tool,
        "tool_coverage": sorted(tool_coverage),
        "recovery_by_tool": recovery_by_tool,
        "sequences": dict(sorted(
            sequence_counts.items(), key=lambda x: -x[1]
        )[:10]),
    }


class ToolExtractor:
    """Session-end tool message analyzer.

    Reads all role='tool' messages for a session, produces
    a structured tool_summary observation. Purely deterministic
    analysis by default — no LLM required.
    """

    def __init__(
        self,
        memory: Any,
        session_id: str,
        user_id: str,
        min_tool_messages: int = 5,
        active_skills: list[str] | None = None,
    ):
        self._memory = memory
        self._session_id = session_id
        self._user_id = user_id
        self._min_tool_messages = min_tool_messages
        self._active_skills = active_skills or []

    async def extract(self) -> dict | None:
        """Run tool extraction. Returns the tool_summary dict or None if no tool messages."""
        from coremem.core import _ingest_message as _  # ensure schema exists

        all_msgs = self._memory.db.raw_query(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY ts",
            (self._session_id,),
        )
        if not all_msgs:
            return None

        # Parse metadata JSON for each message
        parsed: list[dict] = []
        for m in all_msgs:
            entry = dict(m)
            meta_raw = entry.get("metadata")
            if meta_raw:
                try:
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            else:
                meta = {}
            entry["parsed_meta"] = meta
            if meta.get("tool_call_id"):
                entry["tool_call_id"] = meta["tool_call_id"]
            if meta.get("name"):
                entry["name"] = meta["name"]
            if meta.get("tool_calls"):
                entry["tool_calls"] = meta["tool_calls"]
            parsed.append(entry)

        assistant_msgs = [m for m in parsed if m.get("role") == "assistant"]
        tool_msgs = [m for m in parsed if m.get("role") == "tool"]

        if not tool_msgs or len(tool_msgs) < self._min_tool_messages:
            return None

        # Pair tool_call_id across assistant.tool_calls and tool results
        trace = _build_trace(assistant_msgs, tool_msgs)

        # Deterministic analysis (no LLM)
        analysis = _analyze_deterministic(trace)

        # Extract user goal from first user message
        user_goal = ""
        for m in all_msgs:
            if m.get("role") == "user":
                user_goal = (m.get("content", "") or "")[:500]
                break

        # Build the tool_summary observation
        observation = {
            "kind": "tool_summary",
            "content": "",
            "observation_ts": datetime.now(UTC).isoformat(),
            "user_id": self._user_id,
            "session_id": self._session_id,
            "importance": 0.7,
            "memory_type": "agent_behavior",
            "metadata": json.dumps({
                "user_goal": user_goal,
                "n_tool_calls": len(trace),
                "n_errors": analysis["n_errors"],
                "agent_behavior": {
                    "error_by_tool": analysis["error_by_tool"],
                    "tool_coverage": analysis["tool_coverage"],
                    "recovery_by_tool": analysis["recovery_by_tool"],
                    "sequences": analysis["sequences"],
                },
                "active_skills": self._active_skills,
                "error_classification": "heuristic",
            }),
        }

        self._memory.insert_observations([observation])
        return observation