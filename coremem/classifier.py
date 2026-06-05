"""Phase 4: Memory classification and durability filter."""
from __future__ import annotations

import json
from typing import Any

_CLASSIFICATION_PROMPT = """You are a memory classifier. For each observation below, classify:

1. memory_type: profile | preference | project | decision | technical_stack | business_context | people | constraint | workflow | episodic | procedural | sentiment | stance
2. durability: durable | temporary
3. sensitivity: normal | personal | sensitive

DURABLE — reveals something about the user that would be useful in FUTURE conversations:
- Identity facts: job, degree, name, location, contact info
- Preferences/interests: likes, dislikes, values, causes they care about
- Habits/routines: daily activities, recurring patterns
- Past events: festivals, plays, trips, experiences (even one-time events)
- Plans/goals: upcoming travel, purchases, intentions
- Administrative/life-event tasks: name changes, account updates, moving
- Tools/apps: any software, device, service the user uses or considers
- People/relationships: any named person connected to the user
- Stances/positions: what the user supports, opposes, believes should happen
- Sentiments/opinions: how the user feels about things
- Self-descriptions: struggles, strengths, abilities

TEMPORARY — only useful in THIS conversation, not beyond:
- One-off recommendation requests: "User asked for restaurant recommendations in Hilo"
- Session-specific Q&A: "User wanted to know about fracking regulations in Pennsylvania"
- Task-specific help: "User needed help formatting a spreadsheet"
- Weather/time queries: "User asked about the weather forecast for this weekend"
- Hypotheticals with no commitment: "User wondered what if..."
- When the observation describes what the user asked the assistant, not what the user IS

Decision rule:
- Does this fact help understand the user's identity, preferences, or situation? → DURABLE
- Is this just describing what the user needed help with right now? → TEMPORARY
- Would knowing this next session improve the assistant's responses? → DURABLE
- Is this only relevant to the specific task in this conversation? → TEMPORARY

Examples:
- "User works at Anthropic" -> profile, durable, normal, 0.95
- "User prefers audiobooks over e-books" -> preference, durable, normal, 0.85
- "User recently changed their last name from Johnson to Winters" -> profile, durable, normal, 0.95
- "User attended a play at community theater" -> episodic, durable, normal, 0.90
- "User is concerned about environmental impact of fracking" -> sentiment, durable, normal, 0.85
- "User believes fracking should be completely banned" -> stance, durable, normal, 0.95
- "User thinks long-term consequences of fracking outweigh short-term gains" -> stance, durable, normal, 0.90
- "User does not trust fracking companies to self-monitor" -> stance, durable, normal, 0.85
- "User supports transitioning to renewable energy" -> stance, durable, normal, 0.90
- "User uses Audible for audiobooks" -> technical_stack, durable, normal, 0.90
- "User is looking for charity events in LA" -> preference, durable, normal, 0.85
- "User is getting used to a 9-to-5 schedule" -> profile, durable, normal, 0.90
- "User asked for lunch recommendations in Hilo" -> preference, temporary, normal, 0.80
- "User wanted to know about fracking monitoring requirements" -> preference, temporary, normal, 0.80
- "User needed warm-up exercise recommendations" -> preference, temporary, normal, 0.80
- "User asked for TV recommendations in $800-$1,200 range" -> preference, temporary, normal, 0.80

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
