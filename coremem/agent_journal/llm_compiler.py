"""LLM-backed AgentJournal compiler.

Calls an LLM to generate structured AgentJournal plans from source messages,
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

DEFAULT_AGENT_JOURNAL_MODEL = "openai:gpt-4o-mini"

SYSTEM_PROMPT = f"""You are a session summarizer for a memory retrieval system. Your job is to analyze conversation sessions and produce dense, retrieval-optimized summaries that will be used as search indices.

Each session produces a single page with this schema (output as JSON):

```json
{{"pages": [{{...}}]}}
```

Required page fields:
- "operation": "create"
- "page_id": set to the session_id exactly as given (e.g. "session_001")
- "title": short single-line title of the session topic
- "description": single-line description
- "memory_kind": one of {sorted(MEMORY_KINDS)}
- "scope": one of {sorted(SCOPES)}
- "status": "active"
- "activation": "manual"
- "trust": "user_authoritative"
- "safe_to_act": true
- "boot_worthy": false
- "summary": DENSE paragraph (4-6 sentences) capturing the session context, key entities, decisions, and outcomes. This is the PRIMARY search index — write it to match likely future queries. Include specific names, numbers, dates, and technical terms.
- "current_state": list of 1-3 broad summary claim objects (no source_quote required, use derived_summary)

Optional page fields:
- "details": list of strings

Each claim in current_state has:
- "claim": the factual statement (1-2 sentences)
- "evidence": a derived_summary object (always use this — no source_quote needed)

Evidence object (always derived_summary):
- "evidence_type": "derived_summary"
- "supporting_sources": list of 1+ simple source objects with just "evidence_type" and "source_turn_id" (set to the turn_id from the input)

Example:
Input:
Turn ID: turn_abc123
Session ID: s1
## s1_turn_0000_user (user)
I need help setting up CI/CD for our Python project using GitHub Actions.

## s1_turn_0001_assistant (assistant)
Let me help you set up GitHub Actions. First, create a .github/workflows directory and add a ci.yml file...

## s1_turn_0002_user (user)
We use pytest and flake8. Can you include those?

## s1_turn_0003_assistant (assistant)
Sure. Here's a complete workflow that runs pytest with flake8 linting on push and PR to main...

Output:
{{"pages": [{{"operation": "create", "page_id": "s1", "title": "GitHub Actions CI/CD setup for Python project", "description": "Setting up GitHub Actions CI/CD with pytest and flake8 for a Python project", "memory_kind": "project_knowledge", "scope": "project", "status": "active", "activation": "manual", "trust": "user_authoritative", "safe_to_act": true, "boot_worthy": false, "summary": "User requested CI/CD setup for their Python project using GitHub Actions. They use pytest for testing and flake8 for linting. Assistant provided a complete GitHub Actions workflow configuration that runs on push and pull requests to the main branch. The workflow includes Python setup, dependency installation, flake8 linting, and pytest execution.", "current_state": [{{"claim": "The project uses GitHub Actions CI/CD with pytest and flake8 for a Python project, configured to run on push and PR to main.", "evidence": {{"evidence_type": "derived_summary", "supporting_sources": [{{"evidence_type": "assistant_action", "source_turn_id": "turn_abc123"}}]}}}}]}}]}}

Rules:
1. The summary is the most important field — make it dense, specific, and keyword-rich.
2. Include entities, technical terms, names, numbers, and dates that a future query might use.
3. Write in natural prose, not bullet points.
4. Always use "derived_summary" as evidence_type — no source_quote required.
5. source_turn_id must match the turn_id from the input.
6. Output ONLY valid JSON. No markdown fences, no explanation."""
_RETRY_PROMPT = """The deterministic compiler rejected your plan with these errors:

{errors}

Fix the plan and output corrected JSON only. Common issues:
- All required fields must be present (operation, page_id, title, description, etc.)
- Always use "derived_summary" as evidence_type with a "supporting_sources" list
- source_turn_id must match one of the turn_ids from the input
- The JSON must be valid. Pay close attention to the structure."""


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
        model: str = DEFAULT_AGENT_JOURNAL_MODEL,
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
            return self._apply_section(cache_entry, timestamp=timestamp, title=title, messages=messages)
        turn_content = self._format_turn(turn_id, session_id, messages)
        plan = await self._generate_plan(turn_content, turn_id, session_id, messages)
        self._save_cache(turn_id, messages, plan)
        return self._apply_section(plan, timestamp=timestamp, title=title, messages=messages)

    def _apply_section(
        self,
        plan: dict[str, Any],
        *,
        timestamp: str | None,
        title: str | None,
        messages: Sequence[Mapping[str, Any]] | None = None,
    ) -> AgentJournalCompileResult:
        """Validate plan and append section to daily page."""
        date_str: str
        time_str: str
        if timestamp and " " in timestamp:
            date_str, time_str = timestamp.split(" ", 1)
        else:
            date_str = datetime.now(UTC).strftime("%Y-%m-%d")
            time_str = timestamp or "00:00"
        section = self.compiler.compile_section(
            plan,
            timestamp=time_str,
            title=title,
            messages=messages,
        )
        daily_dir = self.bundle.root / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)
        daily_path = daily_dir / f"{date_str}.md"
        if daily_path.exists():
            existing = daily_path.read_text(encoding="utf-8")
            daily_path.write_text(existing.rstrip() + "\n\n" + section, encoding="utf-8")
        else:
            daily_path.write_text(
                f"---\ndate: {date_str}\nagent_journal_version: \"{PROFILE_VERSION}\"\n---\n\n"
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
        all_messages = [msg for _, _, messages in sessions for msg in messages]
        return self.compiler.apply_plan(plan, messages=all_messages)

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
        user_prompt = f"""Analyze this conversation turn and produce a retrieval-optimized session summary.

Turn content:
{turn_content}

IMPORTANT:
- page_id MUST be: "{session_id}"
- source_turn_id MUST be: "{turn_id}"
- Use derived_summary as evidence_type (no source_quote needed)
- The "summary" field is the PRIMARY search index — make it dense, specific, and keyword-rich.

Output a JSON plan with a single page (operation="create")."""
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
        # Sanitize final quote: strip double quotes and newlines that _require_quote would reject
        final = source.get("source_quote", "")
        if final:
            cleaned = final.replace('"', "'").replace("\n", " ").replace("\r", " ").strip()
            if cleaned and len(cleaned) >= 10:
                source["source_quote"] = cleaned[:500]

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
        user_prompt = f"""Analyze these {len(sessions)} conversation sessions and produce retrieval-optimized session summaries — one page per session.

Each page must use page_id = the session_id for that session.

Turn content:
{all_content}

IMPORTANT:
- page_id MUST be the session_id for each session
- source_turn_id MUST match the turn_id shown for each session
- Use derived_summary as evidence_type (no source_quote needed)
- The "summary" field is the PRIMARY search index — make it dense, specific, and keyword-rich.
- Include entities, technical terms, names, numbers, and dates that a future query might use.

Available turn_ids: {turn_id_list}
Available session_ids: {session_id_list}

Output a JSON plan with {len(sessions)} pages (one per session, operation="create")."""
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
