"""LLM-backed AgentJournal compiler.

Calls an LLM to generate structured AgentJournal plans from reference turns,
then validates and renders them through the deterministic compiler.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coremem.agent_journal.bundle import (
    ACTIVATIONS,
    EVIDENCE_TYPES,
    MEMORY_KINDS,
    PROFILE_VERSION,
    SCOPES,
    SOURCE_EVIDENCE_TYPES,
    STATUSES,
    TRUST_VALUES,
    AgentJournalBundle,
    AgentJournalError,
)
from coremem.agent_journal.compiler import AgentJournalCompiler, AgentJournalCompileResult
from coremem.providers import LLMProvider, create_provider

SYSTEM_PROMPT = f"""You are a AgentJournal compiler. Your job is to analyze conversation turns and produce structured AgentJournal pages.

Each page has this schema (output as JSON):

```json
{{"pages": [{{...}}]}}
```

Required page fields:
- "operation": "create" or "update"
- "page_id": lowercase with hyphens/underscores, e.g. "fracking-groundwater"
- "title": short single-line title
- "description": single-line description
- "memory_kind": one of {sorted(MEMORY_KINDS)}
- "scope": one of {sorted(SCOPES)}
- "status": one of {sorted(STATUSES)}
- "activation": one of {sorted(ACTIVATIONS)}
- "trust": one of {sorted(TRUST_VALUES)}
- "safe_to_act": boolean
- "summary": 1-3 sentence summary of the key information
- "current_state": non-empty list of claim objects
- "boot_worthy": boolean (only true if activation=startup AND status=active)

Optional page fields:
- "details": list of strings
- "open_questions": list of strings
- "read_next": list of strings

Each claim in current_state has:
- "claim": the factual statement (1-3 sentences, concise)
- "evidence": either a source object or a derived_summary

Source evidence object:
- "evidence_type": one of {sorted(SOURCE_EVIDENCE_TYPES)}
- "source_turn_id": the turn_id from the reference turn
- "source_message_id": the message_id from the reference turn
- "source_quote": exact substring of the message content (no double quotes, no newlines)

Derived summary evidence object:
- "evidence_type": "derived_summary"
- "supporting_sources": list of 2+ source evidence objects

Examples:

Example 1 (single-source claim):
Input:
## turn_0000_user (user)
What is the capital of France?

## turn_0001_assistant (assistant)
The capital of France is Paris.

Output:
{{"pages": [{{"operation": "create", "page_id": "france-capital", "title": "Capital of France", "description": "The capital city of France is Paris", "memory_kind": "project_fact", "scope": "global", "status": "active", "activation": "manual", "trust": "user_authoritative", "safe_to_act": true, "boot_worthy": false, "summary": "The capital of France is Paris.", "current_state": [{{"claim": "The capital of France is Paris.", "evidence": {{"evidence_type": "assistant_action", "source_turn_id": "session_001", "source_message_id": "turn_0001_assistant", "source_quote": "The capital of France is Paris."}}}}]}}]}}

Example 2 (derived_summary from multiple sources):
Input:
## turn_0000_user (user)
Can you tell me about the weather in Tokyo and what to pack?

## turn_0001_assistant (assistant)
Tokyo in March has average highs of 13C and lows of 5C. It is the cherry blossom season. Bring a warm jacket and an umbrella.

## turn_0002_user (user)
Thanks! What about Osaka?

## turn_0003_assistant (assistant)
Osaka in March is similar to Tokyo: 12-15C highs, 4-6C lows. Also cherry blossom season. Pack similarly.

Output:
{{"pages": [{{"operation": "create", "page_id": "japan-march-weather", "title": "Weather in Japan in March", "description": "Weather conditions and packing advice for Tokyo and Osaka in March", "memory_kind": "project_fact", "scope": "global", "status": "active", "activation": "manual", "trust": "user_authoritative", "safe_to_act": true, "boot_worthy": false, "summary": "Tokyo and Osaka in March have similar weather: highs 12-15C, lows 4-6C, cherry blossom season. Pack a warm jacket and umbrella.", "current_state": [{{"claim": "Tokyo in March has average highs of 13C and lows of 5C during cherry blossom season.", "evidence": {{"evidence_type": "assistant_action", "source_turn_id": "session_001", "source_message_id": "turn_0001_assistant", "source_quote": "Tokyo in March has average highs of 13C and lows of 5C."}}}}, {{"claim": "Osaka in March has similar weather to Tokyo with highs of 12-15C and lows of 4-6C.", "evidence": {{"evidence_type": "assistant_action", "source_turn_id": "session_001", "source_message_id": "turn_0003_assistant", "source_quote": "Osaka in March is similar to Tokyo: 12-15C highs, 4-6C lows."}}}}, {{"claim": "Recommended packing for Japan in March includes a warm jacket and umbrella.", "evidence": {{"evidence_type": "derived_summary", "supporting_sources": [{{"evidence_type": "assistant_action", "source_turn_id": "session_001", "source_message_id": "turn_0001_assistant", "source_quote": "Bring a warm jacket and an umbrella."}}, {{"evidence_type": "assistant_action", "source_turn_id": "session_001", "source_message_id": "turn_0003_assistant", "source_quote": "Pack similarly."}}]}}}}]}}]}}

Rules:
1. Each claim must cite evidence from the conversation. Use exact quotes.
2. The quote must be a verbatim substring of the original message content. Do NOT paraphrase or rephrase the quote — copy it character-for-character from the message.
3. Quotes cannot contain double quotes ("), newlines, or carriage returns. If the message contains double quotes, choose a different segment of the same message that does not contain them.
4. user_statement evidence_type requires role=user messages.
5. assistant_action evidence_type requires role=assistant messages.
6. tool_observation evidence_type requires role=tool_result messages.
7. Write concise claims that capture the key information, not verbatim transcripts.
8. Use derived_summary when a claim synthesizes information from multiple messages.
9. Set boot_worthy=true only for critical startup information.
10. The agent_journal_version is "{PROFILE_VERSION}" — do not include it in your output.
11. Preserve key relationship details, dates, names, and events that could be queried later. These details are critical for future retrieval.

Output ONLY valid JSON. No markdown fences, no explanation."""
_RETRY_PROMPT = """The deterministic compiler rejected your plan with these errors:

{errors}

Fix the plan and output corrected JSON only. The most common cause is that your source_quote is not a verbatim substring of the original message content.

CRITICAL: You MUST copy the quote character-for-character from the message. Do NOT change punctuation, capitalization, or formatting. If the message says "Use eggs as a binder**" (with asterisks), your quote must be exactly "Use eggs as a binder**", not "Use eggs as a binder:" or "Use eggs as a binder".

If the error says "cannot contain double quotes", choose a different segment of the SAME message that does not contain double quotes. For example, if the message says: I said "I don't trust them" and then asked about regulations, use "asked about regulations" instead of "I don't trust them".

If the error says "cannot contain newlines", choose a shorter segment that fits on one line.

Pay special attention to:
- Exact quote matching (quotes must be verbatim substrings of the original message)
- No double quotes or newlines in quotes
- Correct evidence_type for each role
- All required fields present"""


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class AgentJournalLLMCompiler:
    """LLM-backed AgentJournal compiler with deterministic validation."""

    def __init__(
        self,
        bundle: AgentJournalBundle,
        model: str = "ollama-cloud:deepseek-v4-flash",
        max_retries: int = 3,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.bundle = bundle
        self.compiler = AgentJournalCompiler(bundle)
        self.provider: LLMProvider = create_provider(model)
        self.max_retries = max_retries
        self._cache_root = Path(cache_dir) if cache_dir else bundle.root / ".llm_cache"

    async def compile_session(
        self,
        turn_id: str,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        timestamp: str | None = None,
        title: str | None = None,
    ) -> AgentJournalCompileResult:
        """Compile a single session into a daily journal section via LLM (with cache).

        Appends a ``## HH:MM - Title`` section to ``daily/YYYY-MM-DD.md``.
        """
        cache_entry = self._check_cache(turn_id, messages)
        if cache_entry is not None:
            return self._apply_section(cache_entry, timestamp=timestamp, title=title)
        turn_content = self._format_turn(turn_id, session_id, messages)
        plan = await self._generate_plan(turn_content, turn_id, session_id, messages)
        self._save_cache(turn_id, messages, plan)
        return self._apply_section(plan, timestamp=timestamp, title=title)

    def _apply_section(
        self, plan: dict[str, Any], *, timestamp: str | None, title: str | None
    ) -> AgentJournalCompileResult:
        """Validate plan and append section to daily page."""
        section = self.compiler.compile_section(plan, timestamp=timestamp or "00:00", title=title or "Conversation")
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        daily_dir = self.bundle.root / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        daily_path = daily_dir / f"{date_str}.md"
        if daily_path.exists():
            existing = daily_path.read_text(encoding="utf-8")
            daily_path.write_text(existing.rstrip() + "\n\n" + section, encoding="utf-8")
        else:
            daily_path.write_text(
                f"---\ndate: {date_str}\nagent_journal_version: 1\n---\n\n"
                f"# {date_str}\n\n{section}",
                encoding="utf-8",
            )
        return AgentJournalCompileResult(written_pages=(daily_path,), boot_pages=())

    def _check_cache(self, turn_id: str, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        path = self._cache_root / f"{turn_id}.json"
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        expected = hashlib.sha256(json.dumps(list(messages), sort_keys=True).encode()).hexdigest()
        if entry.get("_messages_sha256") != expected:
            return None
        return entry["plan"]

    def _save_cache(self, turn_id: str, messages: Sequence[Mapping[str, Any]], plan: dict[str, Any]) -> None:
        self._cache_root.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(json.dumps(list(messages), sort_keys=True).encode()).hexdigest()
        path = self._cache_root / f"{turn_id}.json"
        path.write_text(json.dumps({"_messages_sha256": sha, "plan": plan}, indent=2), encoding="utf-8")

    def clear_cache(self) -> None:
        if self._cache_root.exists():
            import shutil
            shutil.rmtree(self._cache_root)

    def has_cache(self, turn_id: str) -> bool:
        return (self._cache_root / f"{turn_id}.json").exists()

    async def compile_instance(
        self,
        sessions: Sequence[tuple[str, str, Sequence[Mapping[str, Any]]]],
    ) -> AgentJournalCompileResult:
        """Compile all sessions in an instance into pages via a single LLM call."""
        parts: list[str] = []
        for turn_id, session_id, messages in sessions:
            parts.append(self._format_turn(turn_id, session_id, messages))
        all_content = "\n---\n".join(parts)
        plan = await self._generate_instance_plan(all_content, sessions)
        return self.compiler.apply_plan(plan)

    def _format_turn(
        self,
        turn_id: str,
        session_id: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> str:
        lines = [f"Turn ID: {turn_id}", f"Session ID: {session_id}", ""]
        for msg in messages:
            role = msg.get("role", "unknown")
            mid = msg.get("message_id", "?")
            content = msg.get("content", "")
            lines.append(f"## {mid} ({role})")
            lines.append("")
            lines.append(content)
            lines.append("")
        return "\n".join(lines)

    async def _generate_plan(
        self,
        turn_content: str,
        turn_id: str,
        session_id: str,
        conv_messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        user_prompt = f"""Analyze this conversation turn and produce a AgentJournal page plan.

Turn content:
{turn_content}

IMPORTANT citation fields:
- source_turn_id MUST be exactly: "{turn_id}"
- source_message_id MUST be one of the message_ids shown in the turn content (e.g. "{session_id}_turn_0000_user")
- session_id: "{session_id}"

Output a JSON plan with a single page (operation="create"). Use page_id="{session_id}" to match the session."""
        chat_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        msg_map = {m.get("message_id", ""): m.get("content", "") for m in conv_messages}
        for attempt in range(self.max_retries):
            response = await self.provider.chat(chat_messages)
            raw = response.content
            try:
                plan = json.loads(_extract_json(raw))
            except json.JSONDecodeError as exc:
                if attempt == self.max_retries - 1:
                    raise AgentJournalError(
                        f"LLM returned invalid JSON after {self.max_retries} attempts: {exc}"
                    ) from exc
                chat_messages.append({"role": "assistant", "content": raw})
                chat_messages.append({
                    "role": "user",
                    "content": f"Your output was not valid JSON: {exc}. Output ONLY valid JSON.",
                })
                continue
            self._fix_quotes(plan, msg_map)
            try:
                self.compiler._compile_plan(plan)
                return plan
            except AgentJournalError as exc:
                if attempt == self.max_retries - 1:
                    raise AgentJournalError(
                        f"LLM plan rejected after {self.max_retries} attempts: {exc}"
                    ) from exc
                chat_messages.append({"role": "assistant", "content": raw})
                chat_messages.append({
                    "role": "user",
                    "content": _RETRY_PROMPT.format(errors=str(exc)),
                })
        raise AgentJournalError("LLM compiler exhausted retries")

    def _fix_quotes(self, plan: dict[str, Any], msg_map: dict[str, str]) -> None:
        """Post-process LLM plan to fix quotes that aren't exact substrings.

        Scopes quote search to the specific message identified by source_message_id.
        """
        for page in plan.get("pages", []):
            for claim in page.get("current_state", []):
                evidence = claim.get("evidence", {})
                if evidence.get("evidence_type") == "derived_summary":
                    for source in evidence.get("supporting_sources", []):
                        self._fix_source_quote(source, msg_map)
                else:
                    self._fix_source_quote(evidence, msg_map)

    def _fix_source_quote(self, source: dict[str, Any], msg_map: dict[str, str]) -> None:
        quote = source.get("source_quote", "")
        mid = source.get("source_message_id", "")
        content = msg_map.get(mid, "")
        if not content:
            content = "\n".join(msg_map.values())
        if not quote:
            quote = self._auto_extract_quote(content)
            source["source_quote"] = quote
        else:
            fixed = self._find_exact_quote(quote, content)
            if fixed:
                source["source_quote"] = fixed
            else:
                source["source_quote"] = self._auto_extract_quote(content)
        # If the message_id was wrong, find the correct one
        final_quote = source["source_quote"]
        if final_quote and not self._quote_in_message(final_quote, msg_map.get(mid, "")):
            for correct_mid, correct_content in msg_map.items():
                if self._quote_in_message(final_quote, correct_content):
                    source["source_message_id"] = correct_mid
                    break
            # If still not found, the auto-extracted quote might span messages
            # Re-extract from individual messages
            if not self._quote_in_message(final_quote, msg_map.get(source["source_message_id"], "")):
                for correct_mid, correct_content in msg_map.items():
                    re_extracted = self._auto_extract_quote(correct_content)
                    if re_extracted and len(re_extracted) >= 10:
                        source["source_quote"] = re_extracted
                        source["source_message_id"] = correct_mid
                        break

    def _quote_in_message(self, quote: str, content: str) -> bool:
        return quote in content or quote.casefold() in content.casefold()

    def _auto_extract_quote(self, content: str) -> str:
        """Find the first valid quote segment from content (no double quotes, no newlines, >= 10 chars)."""
        banned = {'"', "\n", "\r"}
        best = ""
        current = ""
        for ch in content:
            if ch in banned:
                if len(current) > len(best) and len(current) >= 10:
                    best = current
                current = ""
            else:
                current += ch
        if len(current) > len(best) and len(current) >= 10:
            best = current
        return best[:500] if best else content[:100].replace('"', "'").replace("\n", " ").strip()

    def _find_exact_quote(self, quote: str, content: str) -> str | None:
        """Find an exact substring of content matching the LLM's quote, extended to sentence boundary."""
        banned = {'"', "\n", "\r"}
        if any(ch in banned for ch in quote):
            return self._find_alternative_quote(quote, content)
        if quote in content:
            return self._extend_to_sentence(quote, content)
        idx = content.casefold().find(quote.casefold())
        if idx >= 0:
            return self._extend_to_sentence(content[idx:idx + len(quote)], content)
        return self._find_alternative_quote(quote, content)

    def _find_alternative_quote(self, quote: str, content: str) -> str | None:
        """When the LLM's quote doesn't match verbatim, try stripping markdown and finding the best segment."""
        banned = {'"', "\n", "\r"}
        # Try stripping markdown formatting from both sides
        import re as _re
        def _strip_md(t):
            return _re.sub(r'[*_`#]', '', t)
        stripped_quote = _strip_md(quote)
        stripped_content = _strip_md(content)
        if stripped_quote in stripped_content:
            idx = stripped_content.find(stripped_quote)
            if idx >= 0:
                return self._extend_to_sentence(content[idx:idx + len(stripped_quote)], content)
        # Try word-by-word matching
        words = quote.split()
        for start in range(len(words)):
            for end in range(start + 1, min(start + 101, len(words) + 1)):
                candidate = " ".join(words[start:end])
                if any(ch in banned for ch in candidate):
                    continue
                if len(candidate) < 10:
                    continue
                idx = content.casefold().find(candidate.casefold())
                if idx >= 0:
                    return self._extend_to_sentence(content[idx:idx + len(candidate)], content)
        return None

    def _extend_to_sentence(self, match: str, content: str) -> str:
        """Extend a quote to end at the nearest sentence boundary, but not past newlines."""
        idx = content.find(match)
        if idx < 0:
            return match
        end = idx + len(match)
        remaining = content[end:]
        sent_end = re.search(r"[.!?]\s|$", remaining)
        if sent_end:
            extension = remaining[:sent_end.end()]
            if "\n" not in extension:
                end += sent_end.end()
        return content[idx:end]

    async def _generate_instance_plan(
        self,
        all_content: str,
        sessions: Sequence[tuple[str, str, Sequence[Mapping[str, Any]]]],
    ) -> dict[str, Any]:
        turn_ids = {t for t, _, _ in sessions}
        session_ids = {s for _, s, _ in sessions}
        turn_id_list = sorted(turn_ids)
        session_id_list = sorted(session_ids)
        msg_map: dict[str, str] = {}
        for _, _, msgs in sessions:
            for m in msgs:
                mid = m.get("message_id", "")
                if mid:
                    msg_map[mid] = m.get("content", "")
        user_prompt = f"""Analyze these {len(sessions)} conversation turns and produce a AgentJournal page plan with one page per turn.

Each page must use page_id = the session_id for that turn.

Turn content:
{all_content}

IMPORTANT citation fields:
- source_turn_id MUST be exactly the turn_id shown in the turn content
- source_message_id MUST be one of the message_ids shown in the turn content

Available turn_ids: {turn_id_list}
Available session_ids: {session_id_list}

Output a JSON plan with {len(sessions)} pages (one per turn, operation="create")."""
        chat_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        for attempt in range(self.max_retries):
            response = await self.provider.chat(chat_messages)
            raw = response.content
            try:
                plan = json.loads(_extract_json(raw))
            except json.JSONDecodeError as exc:
                if attempt == self.max_retries - 1:
                    raise AgentJournalError(
                        f"LLM returned invalid JSON after {self.max_retries} attempts: {exc}"
                    ) from exc
                chat_messages.append({"role": "assistant", "content": raw})
                chat_messages.append({
                    "role": "user",
                    "content": f"Your output was not valid JSON: {exc}. Output ONLY valid JSON.",
                })
                continue
            self._fix_quotes(plan, msg_map)
            try:
                self.compiler._compile_plan(plan)
                return plan
            except AgentJournalError as exc:
                if attempt == self.max_retries - 1:
                    raise AgentJournalError(
                        f"LLM plan rejected after {self.max_retries} attempts: {exc}"
                    ) from exc
                chat_messages.append({"role": "assistant", "content": raw})
                chat_messages.append({
                    "role": "user",
                    "content": _RETRY_PROMPT.format(errors=str(exc)),
                })
        raise AgentJournalError("LLM compiler exhausted retries")
