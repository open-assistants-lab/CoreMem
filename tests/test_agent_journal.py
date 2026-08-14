"""Tests for AgentJournal bundle primitives."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from coremem.agent_journal import AgentJournalBundle, AgentJournalSearch
from coremem.types import Memory


def _memory(message_id: str, content: str, *, role: str = "user") -> Memory:
    return Memory(
        id=message_id,
        role=role,
        content=content,
        ts=datetime(2026, 6, 21, tzinfo=UTC),
        session_id="s1",
        user_id="u1",
        agent_id="a1",
    )


def _write_page(root, relative: str, body: str) -> None:
    path = root / "daily" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _valid_page() -> str:
    return """---
type: AgentJournal Page
page_id: decisions.memory-architecture
title: Memory Architecture
description: Current decisions about AgentJournal and CoreMem memory architecture.
memory_kind: decision
agent_journal_version: "0.1"
scope: project
status: active
activation: query
trust: user_authoritative
safe_to_act: true
---

# Summary

CoreMem is testing AgentJournal as compiled markdown memory.

# Current State

- The POC starts with direct AgentJournal search before adding observations. [1]

# Citations

[1] turn_001, `msg_001`, `user_statement`:
"message -> wiki (okf) -> search"
"""


def test_bundle_initialization_creates_required_files(tmp_path):
    bundle = AgentJournalBundle(tmp_path / "memorypack")

    bundle.initialize()

    root = tmp_path / "memorypack"
    assert (root / "MEMORY.md").exists()
    assert (root / "index.md").exists()
    assert (root / "log.md").exists()
    assert (root / "SCHEMA.md").exists()


# ── dream() cursor and promotion destination ────────────────────────────────


def _daily_page(date: str) -> str:
    return (
        f"---\ndate: {date}\nagent_journal_version: \"0.1\"\n---\n\n"
        f"# {date}\n\n## 10:00 - Test\n\nSome content for {date}\n"
    )


class _FakeDreamProvider:
    def __init__(self, fail: bool, output: str = "### Events\n- something happened\n"):
        self.fail = fail
        self.output = output

    async def chat(self, messages):
        if self.fail:
            raise RuntimeError("llm down")
        return SimpleNamespace(content=self.output)


def _seed_daily_pages(root, dates):
    daily = root / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    for date in dates:
        (daily / f"{date}.md").write_text(_daily_page(date), encoding="utf-8")


def test_dream_cursor_does_not_advance_past_failed_chunks(monkeypatch, tmp_path):
    from coremem.agent_journal import dreaming

    root = tmp_path / "bundle"
    bundle = AgentJournalBundle(root)
    bundle.initialize()
    _seed_daily_pages(root, ("2026-08-01", "2026-08-02", "2026-08-03"))
    cursor = root / ".dreaming_cursor"

    # First run: LLM fails for every chunk. Cursor must NOT advance.
    monkeypatch.setattr(dreaming, "create_provider", lambda model: _FakeDreamProvider(fail=True))
    result = asyncio.run(dreaming.dream(bundle))
    assert result["errors"]
    assert not cursor.exists(), "cursor must not advance past failed chunks"

    # Second run: LLM recovers. All dates processed, cursor advances to the end.
    monkeypatch.setattr(dreaming, "create_provider", lambda model: _FakeDreamProvider(fail=False))
    result2 = asyncio.run(dreaming.dream(bundle))
    assert result2["errors"] == []
    assert cursor.read_text().strip() == "2026-08-03"


def test_dream_promoted_facts_go_to_dreams_not_memory(monkeypatch, tmp_path):
    from coremem.agent_journal import dreaming

    root = tmp_path / "bundle"
    bundle = AgentJournalBundle(root)
    bundle.initialize()
    _seed_daily_pages(root, ("2026-08-01",))
    output = "### Events\n- something happened\n\n### Promoted Facts\n- user likes coffee\n- user hikes\n"
    monkeypatch.setattr(dreaming, "create_provider", lambda model: _FakeDreamProvider(fail=False, output=output))

    asyncio.run(dreaming.dream(bundle))

    dreams = (root / "DREAMS.md").read_text(encoding="utf-8")
    memory = (root / "MEMORY.md").read_text(encoding="utf-8")
    assert "user likes coffee" in dreams
    assert "user hikes" in dreams
    assert "user likes coffee" not in memory, "MEMORY.md is compiler-owned; dream must not write to it"


def test_rebuild_index_writes_month_index_to_monthly(tmp_path):
    from coremem.agent_journal.rebuild_index import rebuild_index

    root = tmp_path / "bundle"
    bundle = AgentJournalBundle(root)
    bundle.initialize()
    _seed_daily_pages(root, ("2026-08-01", "2026-08-02", "2026-08-15"))
    # Simulate a compiler-written page index at the root
    (root / "index.md").write_text("# AgentJournal Index\n\n- [Page](pages/foo.md)\n", encoding="utf-8")

    result = rebuild_index(root)

    assert result["monthly"] == 1
    month_index = root / "monthly" / "2026-08.md"
    assert month_index.exists()
    # Root index.md must not be clobbered by month navigation
    assert "pages/foo.md" in (root / "index.md").read_text(encoding="utf-8")
    assert (root / "daily").is_dir()
    assert (root / "agent_context" / "manifest.json").exists()


def test_validate_claim_requires_exact_quote(tmp_path):
    bundle = AgentJournalBundle(tmp_path / "memorypack")
    message = _memory("msg_001", "message -> wiki (okf) -> search, as POC?")

    valid = {
        "evidence_type": "user_statement",
        "source_turn_id": "turn_001",
        "source_message_id": "msg_001",
        "source_quote": "message -> wiki (okf) -> search",
    }
    invalid = {**valid, "source_quote": "message -> memorypack -> search"}

    assert bundle.validate_claim(valid, [message]) == []
    assert "source_quote is not an exact substring" in "\n".join(bundle.validate_claim(invalid, [message]))


def test_validate_claim_enforces_evidence_type_role(tmp_path):
    bundle = AgentJournalBundle(tmp_path / "memorypack")
    message = _memory("msg_001", "assistant said this", role="assistant")
    claim = {
        "evidence_type": "user_statement",
        "source_turn_id": "turn_001",
        "source_message_id": "msg_001",
        "source_quote": "assistant said this",
    }

    assert "not supported by role" in "\n".join(bundle.validate_claim(claim, [message]))


def test_validate_derived_summary_uses_supporting_sources(tmp_path):
    bundle = AgentJournalBundle(tmp_path / "memorypack")
    claim = {
        "evidence_type": "derived_summary",
        "supporting_sources": [
            {
                "evidence_type": "user_statement",
                "source_turn_id": "turn_001",
                "source_message_id": "msg_001",
                "source_quote": "message -> wiki (okf) -> search",
            },
            {
                "evidence_type": "user_statement",
                "source_turn_id": "turn_002",
                "source_message_id": "msg_002",
                "source_quote": "hybrid theory works",
            },
        ],
    }
    m1 = _memory("msg_001", "message -> wiki (okf) -> search")
    m2 = _memory("msg_002", "prove that your hybrid theory works")

    assert bundle.validate_claim(claim, [m1, m2]) == []


def test_lint_memory_page_requires_frontmatter_and_summary(tmp_path):
    bundle = AgentJournalBundle(tmp_path / "memorypack")
    bundle.initialize()
    _write_page(tmp_path / "memorypack", "decisions/memory-architecture.md", _valid_page())

    assert bundle.lint() == []

    bad = _valid_page().replace("# Summary", "# Overview")
    _write_page(tmp_path / "memorypack", "decisions/memory-architecture.md", bad)
    assert "must have exactly one # Summary" in "\n".join(bundle.lint())


def test_lint_accepts_daily_journal_sections(tmp_path):
    bundle = AgentJournalBundle(tmp_path / "memorypack")
    bundle.initialize()
    daily = tmp_path / "memorypack" / "daily" / "2026-06-28.md"
    daily.write_text(
        """---
date: 2026-06-28
agent_journal_version: "0.1"
---

# 2026-06-28

## 10:30 - Coffee Chat

The user likes coffee.

**Claims:**
- The user likes coffee. [1]

**Citations:**
[1] msg_001 (user_statement): "I like coffee"
""",
        encoding="utf-8",
    )

    assert bundle.lint() == []


def test_search_ranks_memory_pages_not_references(tmp_path):
    bundle = AgentJournalBundle(tmp_path / "memorypack")
    bundle.initialize()
    _write_page(tmp_path / "memorypack", "decisions/memory-architecture.md", _valid_page())

    hits = AgentJournalSearch(tmp_path / "memorypack").search("compiled markdown memory")

    assert hits
    assert hits[0].path.name == "memory-architecture.md"
