"""Dreaming consolidation: diary study analysis of daily journal pages."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from coremem.agent_journal.bundle import AgentJournalBundle, AgentJournalError
from coremem.providers import create_provider

logger = logging.getLogger(__name__)

_DREAMING_PROMPT = """You are a diary analyst. Read the daily journal entries and produce a structured analysis. Focus on:

- Events: what happened
- Emotions: how the user felt
- Cognitions: what the user thought, how they reasoned
- Behaviors: what the user did, how they interacted
- Context: topics discussed, people mentioned
- Patterns: recurring themes across entries
- Anomalies: deviations from established patterns
- Contradictions: conflicting statements across entries
- Promoted facts: durable facts worth keeping in long-term memory

Output markdown with these sections. Be specific, reference timestamps."""

_REQUIRED_SECTIONS = {"### Events"}
_MAX_DAYS_PER_CALL = 7


def _narrative_text(text: str) -> str:
    """Extract narrative summary from a daily page section.

    Returns text between ## HH:MM - Title and **Claims:** for each section.
    """
    sections: list[str] = []
    pattern = re.compile(r"^## (\d{2}:\d{2}) - (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end]
        claims_idx = section.find("\n**Claims:**")
        if claims_idx >= 0:
            section = section[:claims_idx]
        sections.append(section.strip())
    return "\n\n".join(sections) if sections else text


def _read_cursor(root: Path) -> str | None:
    path = root / ".dreaming_cursor"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def _write_cursor(root: Path, date: str) -> None:
    (root / ".dreaming_cursor").write_text(date + "\n", encoding="utf-8")


def _has_dream_entry(dreams_path: Path, date: str) -> bool:
    if not dreams_path.exists():
        return False
    return f"## {date}" in dreams_path.read_text(encoding="utf-8")


def _validate_output(text: str) -> bool:
    return any(section in text for section in _REQUIRED_SECTIONS)


async def dream(
    bundle: AgentJournalBundle,
    model: str = "ollama-cloud:deepseek-v4-flash",
    max_retries: int = 1,
) -> dict[str, Any]:
    """Run dreaming consolidation on unprocessed daily pages.

    Reads daily pages since last cursor, sends narrative summaries to the LLM,
    appends analysis to DREAMS.md, and promotes facts to MEMORY.md.
    """
    cursor = _read_cursor(bundle.root)
    daily_dir = bundle.root / "daily"
    if not daily_dir.exists():
        return {"processed": 0, "promoted": 0, "errors": []}

    all_dates = sorted(
        p.stem for p in daily_dir.glob("*.md") if p.stem.count("-") == 2
    )
    pending = [d for d in all_dates if cursor is None or d > cursor]
    if not pending:
        return {"processed": 0, "promoted": 0, "errors": []}

    provider = create_provider(model)
    dreams_path = bundle.root / "DREAMS.md"
    mem_path = bundle.root / "MEMORY.md"
    errors: list[str] = []
    promoted_count = 0

    for chunk_start in range(0, len(pending), _MAX_DAYS_PER_CALL):
        chunk = pending[chunk_start:chunk_start + _MAX_DAYS_PER_CALL]
        narratives: list[str] = []
        for date in chunk:
            if _has_dream_entry(dreams_path, date):
                continue
            text = (daily_dir / f"{date}.md").read_text(encoding="utf-8")
            narratives.append(f"## {date}\n\n{_narrative_text(text)}")

        if not narratives:
            continue

        user_prompt = "Analyze these daily journal entries:\n\n" + "\n---\n".join(narratives)
        success = False
        for attempt in range(max_retries + 1):
            try:
                response = await provider.chat([
                    {"role": "system", "content": _DREAMING_PROMPT},
                    {"role": "user", "content": user_prompt},
                ])
                output = str(response.content if hasattr(response, "content") else response)
                if not _validate_output(output):
                    if attempt < max_retries:
                        continue
                    errors.append(f"chunk {chunk[0]}: invalid output")
                    break
                success = True
            except Exception as exc:
                if attempt < max_retries:
                    continue
                errors.append(f"chunk {chunk[0]}: {exc}")
                break

        if not success:
            continue

        dreams_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(dreams_path), "a", encoding="utf-8") as f:
            f.write(output.strip() + "\n\n")

        promoted = re.findall(
            r"^### Promoted Facts\n(.*?)(?=\n### |\Z)", output, re.MULTILINE | re.DOTALL
        )
        if promoted:
            facts = re.findall(r"^- (.+)$", promoted[0], re.MULTILINE)
            with open(str(mem_path), "a", encoding="utf-8") as f:
                for fact in facts:
                    f.write(f"- [{chunk[0]}] {fact}\n")
                    promoted_count += 1

    if pending:
        _write_cursor(bundle.root, pending[-1])

    return {
        "processed": len(pending),
        "promoted": promoted_count,
        "errors": errors,
    }
