"""Phase 5: Semantic dedup and merge for observations."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

_DEDUP_PROMPT = """You are a memory deduplicator. For each new/existing pair, classify:

1. duplicate: both say the SAME thing. Even if wording differs, the fact is identical.
2. refine: new ADDS detail to existing without contradicting (e.g., "works at Anthropic" vs "works at Anthropic as engineer").
3. supersede: new REPLACES existing (time has passed, status changed, e.g., "intern" vs "senior").
4. contradict: new CONFLICTS with existing (e.g., "prefers Mac" vs "switched to Linux").
5. new: DISTINCT fact — no meaningful relationship.

Return ONLY: relationship, and for refine, the merged content string.

Examples:
- New: "User works at Anthropic as a research engineer" vs Old: "User works at Anthropic" -> refine, merged: "User works at Anthropic as a research engineer"
- New: "User self-hosts n8n" vs Old: "User uses n8n cloud" -> supersede
- New: "User switched to Linux" vs Old: "User prefers Mac" -> contradict
- New: "User commutes 45 min each way" vs Old: "User likes audiobooks" -> new
- New: "User enjoys coffee" vs Old: "User likes the coffee scene" -> duplicate
"""

_DEDUP_TOOL = {
    "type": "function",
    "function": {
        "name": "record_relationships",
        "description": "Return dedup classifications",
        "parameters": {
            "type": "object",
            "properties": {
                "pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "new_index": {"type": "integer"},
                            "relationship": {"type": "string"},
                            "merged_content": {"type": "string"},
                            "old_id": {"type": "string"},
                        },
                        "required": ["new_index", "relationship", "old_id"],
                    },
                },
            },
            "required": ["pairs"],
        },
    },
}


def build_dedup_prompt(
    pairs: list[dict[str, Any]], new_obs_list: list[dict[str, Any]],
) -> str:
    """Build prompt listing new/old observation pairs."""
    lines = ["Classify each pair:"]
    for p in pairs:
        ni = p["new_index"]
        new_content = new_obs_list[ni].get("content", "")
        old_content = p["candidate"].get("content", "")
        lines.append(
            f"Pair {ni}: "
            f'New[{ni}]:"{new_content}" '
            f'vs Old({p["candidate"]["id"]}):"{old_content}"'
        )
    return "\n".join(lines)


async def dedup_and_merge(
    provider: Any,
    store: Any,
    new_obs: list[dict[str, Any]],
    batch_size: int = 5,
) -> list[dict[str, Any]]:
    """Run dedup + merge on observations."""
    final: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    # Step 1: Find candidates for each observation
    for i, obs in enumerate(new_obs):
        if obs.get("status") == "archived":
            final.append(obs)
            continue
        user_id = obs.get("user_id")
        candidates = store.get_candidates(obs.get("content", ""), user_id=user_id)
        for candidate in candidates:
            pairs.append({"new_index": i, "candidate": candidate})

    if not pairs:
        for obs in new_obs:
            if obs.get("status") != "archived":
                store.insert_event(obs.get("id", ""), "created")
            final.append(obs)
        return final

    # Step 2: Batch pairs and classify via LLM
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        prompt = build_dedup_prompt(batch, new_obs)
        messages = [
            {"role": "system", "content": _DEDUP_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = await provider.chat_with_tools(messages, [_DEDUP_TOOL])
        classifications = _parse_dedup_response(response)

        # Step 3: Apply each classification with priority ordering
        resolved: dict[int, dict[str, Any]] = {}
        for classification in classifications:
            ni = classification["new_index"]
            ex = resolved.get(ni, {})
            ex_rel = ex.get("relationship", "new")
            new_rel = classification["relationship"]
            priority = {"supersede": 5, "contradict": 4, "refine": 3, "duplicate": 2, "new": 1}
            if priority.get(new_rel, 0) > priority.get(ex_rel, 0):
                resolved[ni] = classification

        for classification in resolved.values():
            ni = classification["new_index"]
            rel = classification["relationship"]
            old_id = classification["old_id"]
            obs = new_obs[ni]

            if rel == "duplicate":
                obs["status"] = "archived"
                obs["superseded_by"] = old_id

            elif rel == "refine":
                merged = classification.get("merged_content", obs.get("content", ""))
                store.update_observation(old_id, {"content": merged})
                old_rows = store._db.query(
                    "observations", where="id = ?", params=(old_id,), limit=1,
                )
                if old_rows:
                    old = old_rows[0]
                    old_src = json.loads(old.get("source_message_ids", "[]"))
                    new_src = json.loads(obs.get("source_message_ids", "[]"))
                    store.update_observation(old_id, {
                        "source_message_ids": json.dumps(old_src + new_src),
                    })
                    old_content = old.get("content", "")
                else:
                    old_content = ""
                store.insert_event(old_id, "merged",
                                   old_value=old_content, new_value=merged)
                obs["status"] = "archived"
                obs["superseded_by"] = old_id

            elif rel == "supersede":
                valid_to = obs.get("observation_ts") or datetime.now(UTC).isoformat()
                store.update_observation(old_id, {
                    "status": "superseded",
                    "valid_to": valid_to,
                })
                store.insert_event(old_id, "superseded",
                                   old_value=old_id, new_value=obs.get("id", ""))
                obs["status"] = "candidate"
                store.insert_event(obs.get("id", ""), "created")

            elif rel == "contradict":
                store.create_conflict(old_id, obs.get("id", ""), "contradiction")
                obs["status"] = "candidate"
                store.insert_event(obs.get("id", ""), "contradicted",
                                   old_value=old_id)

            else:  # new
                store.insert_event(obs.get("id", ""), "created")

    # Collect: archived first, then unprocessed
    for obs in new_obs:
        if obs.get("status") == "archived":
            final.append(obs)
    for obs in new_obs:
        if obs.get("status") != "archived":
            store.insert_event(obs.get("id", ""), "created")
            final.append(obs)

    return final


def _parse_dedup_response(response: Any) -> list[dict[str, Any]]:
    """Parse LLM dedup response."""
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        arguments = tool_calls[0].get("function", {}).get("arguments", "")
        if arguments:
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict) and "pairs" in parsed:
                    return parsed["pairs"]
            except (json.JSONDecodeError, TypeError):
                pass
    return []
