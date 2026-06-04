"""Phase 4: Memory classification and durability filter."""
from __future__ import annotations

import json
from typing import Any

_CLASSIFICATION_PROMPT = """You are a memory classifier. For each observation below, classify:

1. memory_type: profile | preference | project | decision | technical_stack | business_context | people | constraint | workflow | episodic | procedural | sentiment
2. durability: durable | temporary
3. sensitivity: normal | personal | sensitive

CORE RULE: Almost EVERY observation is DURABLE. Only mark as temporary if the fact is truly useless beyond this specific conversation.

DURABLE (99% of observations — these are ALL durable):
- Identity facts: job, degree, name, location, contact info
- Preferences/interests: any like, dislike, or request for recommendations.
  "User asked for meal prep containers" IS durable — reveals interest in meal prep.
  "User asked about warm-up exercises" IS durable — reveals interest in fitness.
  "User is looking for charity events" IS durable — reveals charitable interests.
- Habits/routines: daily activities, recurring patterns
- Past events: festivals, plays, trips, experiences (even one-time events)
- Plans/goals: upcoming travel, purchases, intentions
- Administrative/life-event tasks: name changes, account updates, moving — durable
- Tools/apps: any software, device, service the user uses or considers using
- People/relationships: any named person connected to the user
- Sentiments/opinions: how the user feels about things
- Self-descriptions: struggles, strengths, abilities

TEMPORARY (only these — use sparingly):
- Weather queries: "User asked about the weather forecast for this weekend"
- Hypotheticals with no commitment: "User wondered what if..."

When in doubt, choose DURABLE. Temporary should be the exception, not the rule.

Examples (ALL durable):
- "User works at Anthropic" -> profile, durable, normal, 0.95
- "User asked about meal prep containers" -> preference, durable, normal, 0.85
- "User asked about warm-up exercises" -> preference, durable, normal, 0.85
- "User recently changed their last name from Johnson" -> profile, durable, normal, 0.95
- "User needs to update DMV for name change" -> profile, durable, normal, 0.90
- "User is looking for charity events in LA" -> preference, durable, normal, 0.85
- "User is getting used to a 9-to-5 schedule" -> profile, durable, normal, 0.90
- "User attended a play at community theater" -> episodic, durable, normal, 0.90
- "User is interested in taking acting classes" -> preference, durable, normal, 0.85
- "User struggles with getting into character" -> profile, durable, normal, 0.85
- "User uses Audible for audiobooks" -> technical_stack, durable, normal, 0.90
- "User prefers audiobooks over e-books" -> preference, durable, normal, 0.85
- "User is considering getting a tennis ball machine" -> decision, durable, normal, 0.85

Example of TEMPORARY (the only kind):
- "User asked about the weather forecast for this weekend" -> preference, temporary, normal, 0.90

Return ONLY valid JSON via the observations tool.
"""

_CLASSIFICATION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_classifications",
        "description": "Return classified observations",
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "memory_type": {"type": "string"},
                            "durability": {"type": "string"},
                            "sensitivity": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["index", "memory_type", "durability", "sensitivity", "confidence"],
                    },
                },
            },
            "required": ["observations"],
        },
    },
}


def build_classification_prompt(observations: list[dict[str, Any]]) -> str:
    """Build the user message listing all observations to classify."""
    lines = ["Classify each observation:"]
    for i, obs in enumerate(observations):
        lines.append(f"[{i}] {obs.get('content', '')}")
    return "\n".join(lines)


def parse_classifications(response: Any) -> list[dict[str, Any]]:
    """Parse LLM response into classification dicts."""
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        arguments = tool_calls[0].get("function", {}).get("arguments", "")
        if arguments:
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict) and "observations" in parsed:
                    return parsed["observations"]
            except (json.JSONDecodeError, TypeError):
                pass
    return []


async def classify_observations(
    provider: Any,
    observations: list[dict[str, Any]],
    batch_size: int = 20,
) -> list[dict[str, Any]]:
    """Classify observations in batches, returning enriched dicts."""
    classified: list[dict[str, Any]] = []
    for i in range(0, len(observations), batch_size):
        batch = observations[i : i + batch_size]
        prompt = build_classification_prompt(batch)
        messages = [
            {"role": "system", "content": _CLASSIFICATION_PROMPT},
            {"role": "user", "content": prompt},
        ]
        response = await provider.chat_with_tools(messages, [_CLASSIFICATION_TOOL])
        results = parse_classifications(response)
        for idx, obs in enumerate(batch):
            classification = next((r for r in results if r.get("index") == idx), {})
            obs["memory_type"] = classification.get("memory_type", "")
            obs["durability"] = classification.get("durability", "durable")
            obs["sensitivity"] = classification.get("sensitivity", "normal")
            obs["confidence"] = classification.get("confidence", 0.800)
            if obs["durability"] == "temporary":
                obs["status"] = "archived"
            classified.append(obs)
    return classified
