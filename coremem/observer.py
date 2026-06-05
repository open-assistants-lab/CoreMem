"""Observer — single-pass fact extraction from conversations.

Uses a 3-tier alignment gate (coremem.grounding.align_quote) to
deterministically catch fabricated source_quote values. The model is
prompted via CogCanvas-style system message with 2 few-shot examples
that demonstrate the verbatim-quote contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any, cast

from coremem.grounding import AlignmentTier, align_quote
from coremem.observer_utils import parse_json_array
from coremem.providers import create_provider
from coremem.types import Memory

logger = logging.getLogger("coremem.observer")


# ── Prompt: system message (instructions + 2 few-shot examples) ────────────


OBSERVER_SYSTEM_PROMPT = """You are an observation agent. Extract key facts from a conversation and return them as structured observations via the record_observations tool.

RULES:
- One fact per observation. Be exact with values.
- source_quote: a VERBATIM sub-string of the conversation. Copy-paste exactly — do not rephrase, do not paraphrase, do not change a single character. If you cannot find a verbatim sub-string, you must not return the observation.
- Be liberal in what you consider worth recording. Extract observations from ALL user messages — each typically contains 2-4 distinct facts. Include identity, job, location, preferences, likes, dislikes, hobbies, habits, plans, travel, purchases, past events, tools, apps, relationships, and named entities. Also check for "by the way" asides and incidental mentions of apps, products, playlists, and services.

IMPORTANCE: 0.8-1.0 identity/contact/job/salary.
            0.5-0.7 preferences/habits/projects.
            0.1-0.4 context/trivia.
ENTITIES: list of named entities (people, companies, products, locations).

Few-shot examples:

Example 1:
[2024-01-15T10:00:00] user: I just moved to Seattle last month for a new job at Anthropic. I'll be working as a research engineer on alignment.
[2024-01-15T10:01:00] assistant: Exciting move!

{"id": "obs_1", "content": "User moved to Seattle in January 2024", "referenced_date": "2024-01", "source_quote": "I just moved to Seattle last month", "importance": 0.7, "entities": ["Seattle"]}
{"id": "obs_2", "content": "User works at Anthropic as a research engineer", "referenced_date": "2024-01-15", "source_quote": "I'm a research engineer working on alignment", "importance": 0.9, "entities": ["Anthropic"]}

Example 2:
[2024-02-03T14:30:00] user: My favorite programming language is Rust, though I still use Python for data work. I'm also learning Korean in my free time — been at it about 8 months. I use Duolingo and have a language exchange partner named Ji-hye. I plan to visit Seoul this summer for a tech conference.
[2024-02-03T14:31:00] assistant: Great goals!

{"id": "obs_1", "content": "User's favorite programming language is Rust", "referenced_date": "2024-02-03", "source_quote": "My favorite programming language is Rust", "importance": 0.6, "entities": ["Rust"]}
{"id": "obs_2", "content": "User uses Python for data work", "referenced_date": "2024-02-03", "source_quote": "I still use Python for data work", "importance": 0.5, "entities": ["Python"]}
{"id": "obs_3", "content": "User has been learning Korean for 8 months", "referenced_date": "2024-02-03", "source_quote": "been at it about 8 months", "importance": 0.6, "entities": []}
{"id": "obs_4", "content": "User uses Duolingo for language learning", "referenced_date": "2024-02-03", "source_quote": "I use Duolingo", "importance": 0.4, "entities": ["Duolingo"]}
{"id": "obs_5", "content": "User has a language exchange partner named Ji-hye", "referenced_date": "2024-02-03", "source_quote": "a language exchange partner named Ji-hye", "importance": 0.7, "entities": ["Ji-hye"]}
{"id": "obs_6", "content": "User plans to visit Seoul this summer for a tech conference", "referenced_date": "2024-02-03", "source_quote": "visit Seoul this summer for a tech conference", "importance": 0.7, "entities": ["Seoul"]}
"""

GLEANING_SYSTEM_PROMPT = """You are a review agent. Read the conversation again and find facts we may have missed.

The first pass extracted these facts from the conversation:
{already_extracted}

REVIEW TASKS:
1. Named entities: read every user message looking for named people, places, companies, products, services, apps, locations. We may have missed some.
2. Pronoun references: check for "he", "she", "they", "it" — what do they refer to? These often hide facts.
3. Implicit facts: facts stated indirectly. Example: "I've been doing this for years" → user has years of experience.
4. Buried preferences: check long messages for likes/dislikes mentioned in passing.
5. Plans/past events in passing: "by the way" statements and dependent clauses often contain plans or events.
6. Relationships: connections between people and entities we may have missed.

RULES:
- One fact per observation.
- source_quote: VERBATIM sub-string. Copy-paste exactly — do not rephrase or change a single character.
- Do NOT repeat facts we already extracted.
- Only extract facts from USER messages. Skip assistant content.
- If you cannot find a verbatim sub-string, you must not return the observation.

IMPORTANCE: 0.8-1.0 identity/contact/job/salary.
            0.5-0.7 preferences/habits/projects.
            0.1-0.4 context/trivia.
ENTITIES: list of named entities (people, companies, products, locations).

Few-shot examples:

Example 1 — missed entities and implicit facts:

Already extracted:
- User works at Anthropic as a research engineer
- User moved to Seattle in January 2024

Conversation:
[2024-01-15T10:00:00] user: I just moved to Seattle last month for a new job at Anthropic. I'll be working as a research engineer on alignment.
[2024-01-15T10:01:00] assistant: Exciting move! How are you finding Seattle?
[2024-01-15T10:02:00] user: It's great! The weather is different from my hometown of Austin. I love the coffee scene here and have been exploring hiking trails on weekends. My wife Sarah also moved with me — she's an architect.

{"id": "obs_1", "content": "User previously lived in Austin", "source_quote": "my hometown of Austin", "importance": 0.6, "entities": ["Austin"]}
{"id": "obs_2", "content": "User enjoys Seattle's coffee scene", "source_quote": "I love the coffee scene", "importance": 0.4, "entities": ["Seattle"]}
{"id": "obs_3", "content": "User goes hiking on weekends", "source_quote": "exploring hiking trails on weekends", "importance": 0.5, "entities": []}
{"id": "obs_4", "content": "User's wife Sarah is an architect", "source_quote": "My wife Sarah also moved with me — she's an architect", "importance": 0.9, "entities": ["Sarah"]}

Example 2 — missed tools and plans:

Already extracted:
- User's favorite programming language is Rust
- User uses Python for data work

Conversation:
[2024-02-03T14:30:00] user: My favorite programming language is Rust, though I still use Python for data work. I'm also learning Korean in my free time — been at it about 8 months. I use Duolingo and have a language exchange partner named Ji-hye. My goal is to be conversational by the time I visit Seoul this summer for a tech conference.

{"id": "obs_1", "content": "User has been learning Korean for 8 months", "source_quote": "been at it about 8 months", "importance": 0.6, "entities": []}
{"id": "obs_2", "content": "User uses Duolingo for language learning", "source_quote": "I use Duolingo", "importance": 0.4, "entities": ["Duolingo"]}
{"id": "obs_3", "content": "User has a language exchange partner named Ji-hye", "source_quote": "a language exchange partner named Ji-hye", "importance": 0.7, "entities": ["Ji-hye"]}
{"id": "obs_4", "content": "User plans to visit Seoul this summer for a tech conference", "source_quote": "visit Seoul this summer for a tech conference", "importance": 0.7, "entities": ["Seoul"]}
"""

# ── Phase 1: Labeling Functions (LF) for fact enumeration ─────────

_FILTER_SENTENCES_PROMPT = """You are a sentence classifier. For each sentence, determine if it contains factual information about the USER (a verifiable fact about their identity, job, preferences, actions, plans, habits, tools, or states). Return only fact-bearing sentences.

Skip:
- Greetings, politeness, conversational filler
- Assistant recommendations or opinions  
- Sentences that only ask questions without stating facts
- Pure assistant content

Rules:
- If a sentence contains BOTH a question AND a fact about the user, include it.
- Include "by the way" asides and dependent clauses.
- Each sentence that remains should contain at least one verifiable user fact."""

_LF_ENTITIES_PROMPT = """List every named entity related to the user: people (names, roles), places (cities, countries), companies/organizations, products/tools/apps/services, named creations (playlists, books, projects). Use the list_entities tool."""

_LF_ACTIONS_PROMPT = """List every action, habit, routine, plan, or past event related to the user: what they do, did, will do, or used to do. Include hobbies, commutes, learning activities, event attendance. Use the list_entities tool."""

_LF_PREFERENCES_PROMPT = """List every preference, like, dislike, interest, belief, or opinion expressed by or about the user. Include favorite things, preferred methods, valued qualities. Use the list_entities tool."""

_LF_TEMPORAL_PROMPT = """List every temporal or quantitative fact about the user: dates, durations, numeric values, frequencies, ages, counts. Examples: "commute takes 45 minutes", "learning Korean for 8 months". Use the list_entities tool."""

_LF_SENTIMENT_PROMPT = """List every emotional reaction, affective state, or evaluative judgment expressed by the user: what they found amazing, exciting, frustrating, enjoyable. Look for adjectives like "amazing", "great", "love", "exciting". Use the list_entities tool."""

_LF_POSSESSIONS_PROMPT = """List every item, product, tool, or object the user owns, bought, purchased, uses, or possesses. Include apps, devices, books, equipment, clothing, subscriptions. Examples: "bought new tennis racket from sports store downtown", "uses Audible app", "reads Gone Girl on Kindle", "has a Spotify account". Use the list_entities tool."""

_LF_STANCE_PROMPT = """List every stated position, opinion, stance, belief, value, or judgment expressed by the user. Include:
- Hard positions: anything with "should", "must", "need to", "ban", "oppose", "support", "is not enough", "completely"
- What the user supports or opposes: "fracking should be completely banned", "supports renewable energy", "opposes single-use plastics"
- What the user believes should happen: "companies must be held accountable", "regulation alone is not sufficient", "we should do more"
- The user's values and principles: "long-term consequences outweigh short-term gain", "protecting water is worth the cost"
- Judgments about adequacy: "monitoring measures are insufficient", "current regulations are inadequate", "X is simply not enough"
- Stated motivations and tradeoffs: "X is not worth Y", "we need to do Z to protect A"
- Declarative positions: "clean energy is the only way forward", "fracking has unacceptable risks"
Look for strong language: "completely", "simply not", "must", "should", "never", "always", "need", "worth", "enough", "insufficient", "unacceptable".
Use the list_entities tool."""

_LABELING_FUNCTIONS: list[tuple[str, str]] = [
    ("entities", _LF_ENTITIES_PROMPT),
    ("actions", _LF_ACTIONS_PROMPT),
    ("preferences", _LF_PREFERENCES_PROMPT),
    ("temporal", _LF_TEMPORAL_PROMPT),
    ("sentiment", _LF_SENTIMENT_PROMPT),
    ("possessions", _LF_POSSESSIONS_PROMPT),
    ("stance", _LF_STANCE_PROMPT),
]

# ── Legacy extraction prompts (unused in LF pipeline) ────────────

ENTITY_EXTRACTION_SYSTEM_PROMPT = """You are an extraction agent. List EVERY factual statement about the user in this conversation, expressed as short labels.

Return via the list_entities tool. Include:

NAMED ENTITIES:
- People (names, usernames, roles)
- Places (cities, countries, venues)
- Companies/organizations (employers, schools, brands)
- Products/tools (apps, devices, software, services)
- Named things (playlists, books, movies, projects, animals)
- Dates/events (conferences, festivals, trips)

QUANTITIES:
- Numeric facts: "commute: 45 minutes each way", "learning Korean: 8 months"
- Time/duration facts: "work hours: 9-to-5"

ACTIONS AND HABITS:
- Activities the user does: "plays guitar", "takes notes on audiobooks"
- Routines: "listens to audiobooks during commute", "multitasks with audiobooks"
- Tools they use: "uses Audible app", "reads on Kindle"

PREFERENCES AND STATES:
- Likes/dislikes: "prefers audiobooks over e-books", "enjoys hiking"
- Interests: "into podcasts", "interested in ambient music"
- Abilities: "retains more from audiobooks than e-books"
- Sentiment: "found music festival amazing"

RULES:
- Be EXHAUSTIVE. Every user message contains 2-4 facts. Extract all of them.
- Label each as a short descriptive phrase: "getting back into guitar", "created Spotify playlist Summer Vibes"
- Include from "by the way" asides and dependent clauses.
- Skip assistant-only content.
"""

ENTITY_RELATION_PROMPT = """You are a fact extraction agent. For each item below, describe its relationship to the user using a verbatim source_quote from the conversation.

Items to process:
{entities}

For each item, return one observation with:
- content: a clear statement of the item's relationship to the user
- source_quote: VERBATIM sub-string from the conversation
- importance: 0.8-1.0 for identity/job/contact, 0.5-0.7 for preferences/habits, 0.1-0.4 for context/trivia
- entities: list containing only this item's key entity

RULES:
- One observation per item. If no relationship exists for an item, skip it.
- source_quote must be copy-pasted exactly — do not rephrase.
- If you cannot find a verbatim sub-string, skip that item.
- Some items are attribute-value pairs (e.g. "commute: 45 minutes"). Extract the relationship as a fact about the user and the value.
- IMPORTANT: The content field MUST include the entity/item name verbatim. For entity "The Glass Menagerie", write "User attended The Glass Menagerie", not "User attended a play". For entity "sports store downtown", write "bought racket from sports store downtown", not "bought a new racket".

Examples:

Items: ["Seattle", "Anthropic", "Rust"]
[2024-01-15T10:00:00] user: I just moved to Seattle last month for a new job at Anthropic. My favorite language is Rust.

{"id": "obs_1", "content": "User moved to Seattle in January 2024", "source_quote": "I just moved to Seattle last month", "importance": 0.7, "entities": ["Seattle"]}
{"id": "obs_2", "content": "User works at Anthropic", "source_quote": "a new job at Anthropic", "importance": 0.9, "entities": ["Anthropic"]}
{"id": "obs_3", "content": "User's favorite programming language is Rust", "source_quote": "My favorite language is Rust", "importance": 0.6, "entities": ["Rust"]}

Items: ["Duolingo", "Ji-hye", "Seoul", "commute: 45 minutes each way", "getting back into guitar"]
[2024-02-03T14:30:00] user: I use Duolingo to learn Korean. My exchange partner is named Ji-hye. I plan to visit Seoul this summer. My daily commute takes 45 minutes each way. I've been trying to get back into playing guitar.

{"id": "obs_1", "content": "User uses Duolingo for language learning", "source_quote": "I use Duolingo", "importance": 0.4, "entities": ["Duolingo"]}
{"id": "obs_2", "content": "User has a language exchange partner named Ji-hye", "source_quote": "My exchange partner is named Ji-hye", "importance": 0.7, "entities": ["Ji-hye"]}
{"id": "obs_3", "content": "User plans to visit Seoul this summer", "source_quote": "I plan to visit Seoul this summer", "importance": 0.7, "entities": ["Seoul"]}
{"id": "obs_4", "content": "User's daily commute is 45 minutes each way", "source_quote": "My daily commute takes 45 minutes each way", "importance": 0.6, "entities": ["commute"]}
{"id": "obs_5", "content": "User is getting back into playing the guitar", "source_quote": "I've been trying to get back into playing guitar", "importance": 0.5, "entities": ["guitar"]}
"""

ENTITY_EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "list_entities",
        "description": "List named entities from the conversation",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entity names",
                },
            },
            "required": ["entities"],
        },
    },
}

ENTITY_RELATION_PROMPT = """You are a fact extraction agent. For each entity below, describe its relationship to the user using a verbatim source_quote from the conversation.

Entities to process:
{entities}

For each entity, return one observation with:
- content: a clear statement of the entity's relationship to the user
- source_quote: VERBATIM sub-string from the conversation
- importance: 0.8-1.0 for identity/job, 0.5-0.7 for preferences/habits, 0.1-0.4 for context
- entities: list containing only this entity

RULES:
- One observation per entity. If no relationship exists, skip that entity.
- source_quote must be copy-pasted exactly — do not rephrase.
- If you cannot find a verbatim sub-string, skip that entity.

Examples:

Entities: ["Seattle", "Anthropic", "Rust"]
[2024-01-15T10:00:00] user: I just moved to Seattle last month for a new job at Anthropic. My favorite language is Rust.

{"id": "obs_1", "content": "User moved to Seattle in January 2024", "source_quote": "I just moved to Seattle last month", "importance": 0.7, "entities": ["Seattle"]}
{"id": "obs_2", "content": "User works at Anthropic", "source_quote": "a new job at Anthropic", "importance": 0.9, "entities": ["Anthropic"]}
{"id": "obs_3", "content": "User's favorite programming language is Rust", "source_quote": "My favorite language is Rust", "importance": 0.6, "entities": ["Rust"]}

Entities: ["Duolingo", "Ji-hye", "Seoul"]
[2024-02-03T14:30:00] user: I use Duolingo to learn Korean. My exchange partner is named Ji-hye. I plan to visit Seoul this summer.

{"id": "obs_1", "content": "User uses Duolingo for language learning", "source_quote": "I use Duolingo", "importance": 0.4, "entities": ["Duolingo"]}
{"id": "obs_2", "content": "User has a language exchange partner named Ji-hye", "source_quote": "My exchange partner is named Ji-hye", "importance": 0.7, "entities": ["Ji-hye"]}
{"id": "obs_3", "content": "User plans to visit Seoul this summer", "source_quote": "I plan to visit Seoul this summer", "importance": 0.7, "entities": ["Seoul"]}
"""


# ── Tool schema (no `priority` field) ─────────────────────────────────────


OBSERVATION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_observations",
        "description": "Record observations extracted from the conversation",
        "parameters": {
            "type": "object",
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique ID like obs_01"},
                            "content": {"type": "string", "description": "ONE fact per observation"},
                            "referenced_date": {"type": "string"},
                            "source_quote": {"type": "string", "description": "EXACT sub-string of the conversation (prefix stripped)"},
                            "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "entities": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "id",
                            "content",
                            "referenced_date",
                            "source_quote",
                            "importance",
                            "entities",
                        ],
                    },
                },
            },
            "required": ["observations"],
        },
    },
}


# ── Focus instructions for multi-pass extraction ───────────────

_OBSERVER_FOCUSES: dict[str, str] = {
    "identity": "FOCUS: Extract facts about identity, job, employer, location, education, name, contact, relationships with people and organizations.",
    "preferences": "FOCUS: Extract facts about preferences, likes, dislikes, favorite things, hobbies, habits, opinions, and interests.",
    "plans": "FOCUS: Extract facts about plans, upcoming events, travel, purchases, goals, past events, experiences, tools, apps, and services used.",
}


# ── Observer ───────────────────────────────────────────────────


class Observer:
    """Single-pass fact extraction from conversation messages.

    Makes one chat_with_tools call per invocation. Returns parsed
    observations or [] on parse failure / empty tool_calls.
    """

    def __init__(
        self,
        model: str = "ollama:llama3.2",
        tool_temp: float = 0.1,
    ):
        self._provider = create_provider(model, tool_temp=tool_temp)

    async def run(
        self,
        messages: list[Memory],
        prior_observations: list[dict[str, Any]] | None = None,
        observation_date: str | None = None,
        focus: str | None = None,
        gleaning_context: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        prior = prior_observations or []
        date_str = observation_date or datetime.now(UTC).date().isoformat()

        llm_messages = self._build_messages(
            messages, prior, date_str,
            focus=focus, gleaning_context=gleaning_context,
        )
        response = await self._provider.chat_with_tools(llm_messages, [OBSERVATION_TOOL])
        return self._parse_response(response)

    async def extract_entities(self, messages: list[Memory]) -> list[str]:
        """Phase 1 (legacy): List all named entities from the conversation."""
        date_str = datetime.now(UTC).date().isoformat()
        context_lines: list[str] = []
        for m in messages:
            if m.content and m.ts is not None:
                ts_str = m.ts.isoformat()[:19]
                context_lines.append(f"[{ts_str}] {m.content}")
        conversation = "\n".join(context_lines)

        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"# Conversation\n{conversation}"},
        ]
        response = await self._provider.chat_with_tools(llm_messages, [ENTITY_EXTRACTION_TOOL])
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            arguments = tool_calls[0].get("function", {}).get("arguments", "")
            if arguments:
                parsed = parse_json_array(arguments)
                if parsed and "entities" in parsed[0]:
                    return cast(list[str], parsed[0]["entities"])
                if parsed and isinstance(parsed[0], str):
                    return cast(list[str], parsed)
        return []

    async def filter_sentences(self, messages: list[Memory]) -> str:
        """Filter user messages to fact-bearing sentences only."""
        user_msgs = [m.content for m in messages if m.role == "user" and m.content]
        combined = "\n\n".join(user_msgs)
        if not combined:
            return ""
        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": _FILTER_SENTENCES_PROMPT},
            {"role": "user", "content": f"Conversation:\n{combined}\n\nReturn only fact-bearing sentences (one per line):"},
        ]
        response = await self._provider.chat(llm_messages)
        return getattr(response, "content", "") or ""

    async def run_lf(self, lf_name: str, lf_prompt: str, text: str) -> list[str]:
        """Run a single labeling function on filtered text."""
        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": lf_prompt},
            {"role": "user", "content": f"Text:\n{text}"},
        ]
        response = await self._provider.chat_with_tools(llm_messages, [ENTITY_EXTRACTION_TOOL])
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            arguments = tool_calls[0].get("function", {}).get("arguments", "")
            if arguments:
                parsed = parse_json_array(arguments)
                if parsed and "entities" in parsed[0]:
                    items = cast(list[str], parsed[0]["entities"])
                elif parsed and isinstance(parsed[0], str):
                    items = cast(list[str], parsed)
                else:
                    items = []
                logger.debug("lf_result", {"lf": lf_name, "count": len(items)}, user_id="system")
                return items
        return []

    async def extract_relations(
        self,
        messages: list[Memory],
        entities: list[str],
        prior: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Phase 2: For each entity, describe relationship to user with verbatim quote."""
        if not entities:
            return []

        # Build conversation text
        date_str = datetime.now(UTC).date().isoformat()
        context_lines: list[str] = []
        for m in messages:
            if m.content and m.ts is not None:
                ts_str = m.ts.isoformat()[:19]
                context_lines.append(f"[{ts_str}] {m.content}")
        conversation = "\n".join(context_lines)

        # Substitute entity list into prompt
        entity_list = "\n".join(f"- {e}" for e in entities)
        system_prompt = ENTITY_RELATION_PROMPT.replace("{entities}", entity_list)

        # Prior observations for dedup context
        prior_block = ""
        if prior:
            prior_lines = "\n".join(
                f"- {o.get('content', '')}" for o in prior[:20]
            )
            prior_block = f"# Already extracted (last 20)\n{prior_lines}\n\n"

        llm_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{prior_block}# Conversation\n{conversation}"},
        ]
        response = await self._provider.chat_with_tools(llm_messages, [OBSERVATION_TOOL])
        return self._parse_response(response)

    def _build_messages(
        self,
        messages: list[Memory],
        prior: list[dict[str, Any]],
        date_str: str,
        focus: str | None = None,
        gleaning_context: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """Build the native messages array (system + context + conversation)."""
        system_prompt = OBSERVER_SYSTEM_PROMPT
        context_prior = prior
        if gleaning_context is not None:
            extracted_lines = "\n".join(
                f"- {o.get('content', '')}" for o in gleaning_context
            )
            system_prompt = GLEANING_SYSTEM_PROMPT.replace(
                "{already_extracted}", extracted_lines or "(none)"
            )
            context_prior = []
        context_block = self._build_context_block(context_prior, date_str, focus=focus)
        out: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_block},
        ]
        for m in messages:
            if m.content and m.ts is not None:
                ts_str = m.ts.isoformat()[:19]
                out.append({"role": m.role, "content": f"[{ts_str}] {m.content}"})
        return out

    @staticmethod
    def _build_context_block(
        prior: list[dict[str, Any]],
        date_str: str,
        focus: str | None = None,
    ) -> str:
        """Build the # Already extracted + # Observation date preamble."""
        if prior:
            prior_lines = "\n".join(
                f"- {o.get('content', '')}" for o in prior[:20]
            )
            already = f"# Already extracted (last 20)\n{prior_lines}\n\n"
        else:
            already = "# Already extracted (last 20)\n(none)\n\n"
        ctx = f"{already}# Observation date\n{date_str}"
        if focus:
            instruction = _OBSERVER_FOCUSES.get(focus)
            if instruction:
                ctx = f"{instruction}\n\n{ctx}"
        return ctx

    @staticmethod
    def _parse_response(response: Any) -> list[dict[str, Any]]:
        """Extract observations from a chat_with_tools response.

        Reads payload from tool_calls[0].function.arguments (NOT content).
        Falls back to parsing content if no tool_calls.
        """
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            arguments = tool_calls[0].get("function", {}).get("arguments", "")
            if arguments:
                parsed = parse_json_array(arguments)
                if parsed and "observations" in parsed[0]:
                    observations = cast(list[dict[str, Any]], parsed[0]["observations"])
                elif parsed and isinstance(parsed[0], dict) and "content" in parsed[0]:
                    observations = parsed
                else:
                    return []
            else:
                return []
        else:
            content = getattr(response, "content", "")
            if not content:
                return []
            parsed = parse_json_array(content)
            if parsed and "observations" in parsed[0]:
                observations = cast(list[dict[str, Any]], parsed[0]["observations"])
            elif parsed:
                observations = parsed
            else:
                return []

        for obs in observations:
            obs["importance"] = None
        return observations


# ── ObserverPipeline ───────────────────────────────────────────────────────


class ObserverPipeline:
    """Per-turn fact extraction pipeline with alignment-gated verification.

    Fires after each agent turn but only runs the LLM call when both
    ``token_threshold`` new tokens have accumulated AND ``min_turns``
    have passed since the last run.
    """

    def __init__(
        self,
        core: Any,
        store: Any,
        session_id: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model: str = "ollama:llama3.2",
        token_threshold: int = 8000,
        min_turns: int = 3,
        max_messages: int = 500,
        enable_gleaning: bool = False,
        enable_classification: bool = False,
        enable_dedup: bool = False,
        tool_temp: float = 0.1,
    ):
        self._enable_gleaning = enable_gleaning
        self._enable_classification = enable_classification
        self._enable_dedup = enable_dedup
        self._core = core
        self._store = store
        self._session_id = session_id
        self._user_id = user_id
        self._agent_id = agent_id
        self._metadata = metadata
        self._observer = Observer(model=model, tool_temp=tool_temp)
        self._token_threshold = token_threshold
        self._min_turns = min_turns
        self._max_messages = max_messages

        self._last_observed_id: str | None = None
        self._turns_since_last_run: int = 0
        self._running: bool = False

    async def after_turn(self) -> list[dict[str, Any]] | None:
        """Called after each agent turn. Returns new observations or None."""
        if self._running:
            return None
        self._turns_since_last_run += 1
        try:
            self._running = True
            return await self._maybe_run()
        finally:
            self._running = False

    async def _maybe_run(self) -> list[dict[str, Any]] | None:
        messages = self._core.fetch(
            session_id=self._session_id,
            user_id=self._user_id,
            agent_id=self._agent_id,
            metadata=self._metadata,
            limit=self._max_messages,
        )

        new_messages: list[Memory] = []
        seen_watermark = self._last_observed_id is None
        for m in messages:
            if m.role == "tool":
                continue
            if not seen_watermark:
                if m.id == self._last_observed_id:
                    seen_watermark = True
                continue
            new_messages.append(m)
        if not seen_watermark:
            new_messages = [m for m in messages if m.role != "tool"]
        if not new_messages:
            return None

        # Build full canonical text for threshold check
        full_canonical_lines: list[str] = []
        for m in new_messages:
            if m.content and m.ts is not None:
                ts_str = m.ts.isoformat()[:19]
                full_canonical_lines.append(f"[{ts_str}] {m.content}")
        new_tokens = sum(_estimate_tokens_line(line) for line in full_canonical_lines)
        if new_tokens < self._token_threshold or self._turns_since_last_run < self._min_turns:
            return None

        prior = self._store.get_recent_observations(days=30, limit=50)
        new_obs: list[dict[str, Any]] = []

        # Build canonical text from full conversation
        canonical_lines: list[str] = []
        for m in new_messages:
            if m.content and m.ts is not None:
                ts_str = m.ts.isoformat()[:19]
                canonical_lines.append(f"[{ts_str}] {m.content}")
        canonical_text = "\n".join(canonical_lines)

        # Phase 1: 5-LF parallel enumeration on user messages
        entities: list[str] = []
        try:
            # Build user-only text (strip assistant noise)
            user_msgs = [m.content for m in new_messages if m.role == "user" and m.content]
            filtered = "\n\n".join(user_msgs)
            if filtered:
                # Run 5 labeling functions in parallel
                lf_results = await asyncio.gather(
                    *[self._observer.run_lf(name, prompt, filtered) for name, prompt in _LABELING_FUNCTIONS],
                    return_exceptions=True,
                )
                # Union and dedup entity lists (fuzzy, not exact)
                seen: list[str] = []
                for result in lf_results:
                    if isinstance(result, list):
                        for item in result:
                            key = item.strip()
                            if not key:
                                continue
                            # Check fuzzy similarity against already-seen items
                            if any(_string_similarity(key, s) > 0.75 for s in seen):
                                continue
                            seen.append(key)
                            entities.append(key)
                logger.info(
                    "lf_extraction",
                    {"filtered_chars": len(filtered), "entity_count": len(entities)},
                )
        except Exception as e:
            logger.warning("entity_extraction_error", {"error": str(e)})

        # Phase 2: Extract observations per entity batch
        if entities:
            batch_size = 4
            for i in range(0, len(entities), batch_size):
                    batch = entities[i : i + batch_size]
                    try:
                        batch_obs = await self._observer.extract_relations(
                            new_messages, batch, prior,
                        )
                    except Exception as e:
                        logger.warning("relation_extraction_error", {"error": str(e)})
                        continue

                    for obs in batch_obs:
                        quote = obs.get("source_quote", "").strip()
                        content = obs.get("content", "").strip()
                        if not quote or not content:
                            continue
                        result = align_quote(quote, canonical_text)
                        if result.tier == AlignmentTier.NONE:
                            continue
                        obs["alignment_tier"] = result.tier.value
                        obs["alignment_confidence"] = result.confidence
                        dedup_targets = prior + new_obs
                        if any(content == p.get("content", "") for p in dedup_targets):
                            continue
                        if any(_string_similarity(content, p.get("content", "")) > 0.75 for p in dedup_targets):
                            continue
                        obs["session_id"] = self._session_id
                        obs["user_id"] = self._user_id or ""
                        obs["agent_id"] = self._agent_id or ""
                        obs.pop("id", None)
                        obs["source_message_ids"] = json.dumps(
                            [m.id for m in new_messages if m.id]
                        )
                        new_obs.append(obs)

        # Phase 3: Per-message gleaning — review each user message for missed facts
        if self._enable_gleaning and new_obs:
            # Group into per-user-message pairs
            pairs: list[list[Memory]] = []
            current: list[Memory] = []
            for m in new_messages:
                if m.role == "user" and current:
                    pairs.append(current)
                    current = []
                current.append(m)
            if current:
                pairs.append(current)

            for pair in pairs:
                try:
                    gleaning_obs = await self._observer.run(
                        pair, prior, gleaning_context=new_obs,
                    )
                    # Align each pair's gleaning against its own pair text
                    pair_lines: list[str] = []
                    for m in pair:
                        if m.content and m.ts is not None:
                            pair_lines.append(f"[{m.ts.isoformat()[:19]}] {m.content}")
                    pair_text = "\n".join(pair_lines)

                    for obs in gleaning_obs:
                        quote = obs.get("source_quote", "").strip()
                        content = obs.get("content", "").strip()
                        if not quote or not content:
                            continue
                        result = align_quote(quote, pair_text)
                        if result.tier == AlignmentTier.NONE:
                            continue
                        obs["alignment_tier"] = result.tier.value
                        obs["alignment_confidence"] = result.confidence
                        dedup_targets = prior + new_obs
                        if any(content == p.get("content", "") for p in dedup_targets):
                            continue
                        if any(_string_similarity(content, p.get("content", "")) > 0.75 for p in dedup_targets):
                            continue
                        obs["session_id"] = self._session_id
                        obs["user_id"] = self._user_id or ""
                        obs["agent_id"] = self._agent_id or ""
                        obs.pop("id", None)
                        obs["source_message_ids"] = json.dumps(
                            [m.id for m in pair if m.id]
                        )
                        new_obs.append(obs)
                except Exception as e:
                    logger.warning("gleaning_error", {"error": str(e)})

        # Phase 4: Classification + Durability Filter
        if self._enable_classification and new_obs:
            try:
                from coremem.classifier import classify_observations
                new_obs = await classify_observations(
                    self._observer._provider, new_obs,
                )
            except Exception as e:
                logger.warning("classification_error", {"error": str(e)})

        # Phase 5: Dedup + Merge
        if self._enable_dedup and new_obs:
            try:
                from coremem.dedup import dedup_and_merge
                new_obs = await dedup_and_merge(
                    self._observer._provider, self._store, new_obs,
                )
            except Exception as e:
                logger.warning("dedup_error", {"error": str(e)})

        if new_obs:
            self._store.insert_observations(new_obs)
        if messages:
            self._last_observed_id = messages[0].id
        self._turns_since_last_run = 0
        return new_obs


# ── Helpers ────────────────────────────────────────────────────────────────


def _estimate_tokens_line(text: str) -> int:
    """Rough token count: ~4 chars per token."""
    return max(1, len(text) // 4)


def _string_similarity(a: str, b: str) -> float:
    """Simple string similarity using difflib."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
