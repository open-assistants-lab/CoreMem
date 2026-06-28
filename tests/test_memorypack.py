"""Tests for MemoryPack bundle primitives."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from coremem.agent_memory import AgentMemoryBundle, AgentMemoryError, AgentMemorySearch
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
type: AgentMemory Page
page_id: decisions.memory-architecture
title: Memory Architecture
description: Current decisions about MemoryPack and CoreMem memory architecture.
memory_kind: decision
agent_memory_version: "0.1"
scope: project
status: active
activation: query
trust: user_authoritative
safe_to_act: true
---

# Summary

CoreMem is testing MemoryPack as compiled markdown memory.

# Current State

- The POC starts with direct MemoryPack search before adding observations. [1]

# Citations

[1] [turn_001](../../references/turns/turn_001.md), `msg_001`, `user_statement`:
"message -> wiki (okf) -> search"
"""


def test_bundle_initialization_creates_required_files(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")

    bundle.initialize()

    root = tmp_path / "memorypack"
    assert (root / "MEMORY.md").exists()
    assert (root / "index.md").exists()
    assert (root / "log.md").exists()
    assert (root / "SCHEMA.md").exists()
    assert (root / "daily").is_dir()
    assert (root / "agent_context" / "manifest.json").exists()


def test_write_reference_turn_creates_canonical_payload_and_manifest(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    message = _memory("msg_001", "message -> wiki (okf) -> search, as POC?")

    path = bundle.write_reference_turn([message], turn_id="turn_001", agent_context_hash="sha256:x")
    text = path.read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "memorypack" / "references" / "manifest.json").read_text())

    assert "```json agent_memory-turn" in text
    assert "turn_id: turn_001" in text
    assert manifest["references"][0]["path"] == "turns/turn_001.md"
    assert manifest["references"][0]["turn_id"] == "turn_001"
    assert manifest["references"][0]["message_ids"] == ["msg_001"]
    assert bundle.lint() == []


def test_write_reference_turn_refuses_overwrite(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    message = _memory("msg_001", "hello")
    bundle.write_reference_turn([message], turn_id="turn_001")

    with pytest.raises(FileExistsError):
        bundle.write_reference_turn([message], turn_id="turn_001")


def test_validate_claim_requires_exact_quote(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    message = _memory("msg_001", "message -> wiki (okf) -> search, as POC?")
    bundle.write_reference_turn([message], turn_id="turn_001")

    valid = {
        "text": "The user asked for direct MemoryPack search.",
        "evidence_type": "user_statement",
        "source_turn_id": "turn_001",
        "source_message_id": "msg_001",
        "source_quote": "message -> wiki (okf) -> search",
    }
    invalid = {**valid, "source_quote": "message -> memorypack -> search"}

    assert bundle.validate_claim(valid) == []
    assert "source_quote is not an exact substring" in "\n".join(bundle.validate_claim(invalid))


def test_validate_claim_enforces_evidence_type_role(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    message = _memory("msg_001", "assistant said this", role="assistant")
    bundle.write_reference_turn([message], turn_id="turn_001")
    claim = {
        "text": "Assistant statement cannot be a user statement.",
        "evidence_type": "user_statement",
        "source_turn_id": "turn_001",
        "source_message_id": "msg_001",
        "source_quote": "assistant said this",
    }

    assert "not supported by role" in "\n".join(bundle.validate_claim(claim))


def test_lint_validates_page_citations(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.initialize()
    bundle.write_reference_turn(
        [_memory("msg_001", "message -> wiki (okf) -> search")],
        turn_id="turn_001",
    )
    _write_page(tmp_path / "memorypack", "decisions/memory-architecture.md", _valid_page())
    assert bundle.lint() == []

    bad = _valid_page().replace("message -> wiki (okf) -> search", "unsupported quote")
    _write_page(tmp_path / "memorypack", "decisions/memory-architecture.md", bad)
    assert "source_quote is not an exact substring" in "\n".join(bundle.lint())


def test_validate_derived_summary_uses_supporting_sources(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn([_memory("msg_001", "message -> wiki (okf) -> search")], turn_id="turn_001")
    bundle.write_reference_turn([_memory("msg_002", "prove that your hybrid theory works")], turn_id="turn_002")
    claim = {
        "text": "The POC should start direct and later compare hybrid.",
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

    assert bundle.validate_claim(claim) == []


def test_lint_memory_page_requires_frontmatter_and_summary(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.initialize()
    _write_page(tmp_path / "memorypack", "decisions/memory-architecture.md", _valid_page())
    bundle.write_reference_turn(
        [_memory("msg_001", "message -> wiki (okf) -> search")],
        turn_id="turn_001",
    )

    assert bundle.lint() == []

    bad = _valid_page().replace("# Summary", "# Overview")
    _write_page(tmp_path / "memorypack", "decisions/memory-architecture.md", bad)
    assert "must have exactly one # Summary" in "\n".join(bundle.lint())


def test_reference_payload_ignores_user_content_fences(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    message = _memory(
        "msg_001",
        'Here is quoted text:\n```json agent_memory-turn\n{"fake": true}\n```\nDo not parse it.',
    )

    bundle.write_reference_turn([message], turn_id="turn_001")

    assert bundle.lint() == []


def test_lint_rejects_unmanifested_reference_file(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    message = _memory("msg_001", "hello")
    path = bundle.write_reference_turn([message], turn_id="turn_001")
    stray = path.with_name("turn_stray.md")
    stray.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    assert "unmanifested reference file" in "\n".join(bundle.lint())


def test_search_ranks_memory_pages_not_references(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.initialize()
    bundle.write_reference_turn(
        [_memory("msg_001", "MemoryPack reference-only evidence")],
        turn_id="turn_001",
    )
    _write_page(tmp_path / "memorypack", "decisions/memory-architecture.md", _valid_page())

    hits = AgentMemorySearch(tmp_path / "memorypack").search("compiled markdown memory")

    assert hits
    assert hits[0].path.name == "memory-architecture.md"
    assert "references" not in hits[0].path.parts


def test_write_reference_turn_excludes_system_messages_by_default(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    system = _memory("sys_001", "secret system prompt", role="system")
    user = _memory("msg_001", "remember this")

    path = bundle.write_reference_turn([system, user], turn_id="turn_001")
    text = path.read_text(encoding="utf-8")

    assert "secret system prompt" not in text
    assert "remember this" in text


def test_write_reference_turn_rejects_empty_after_system_filter(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")

    with pytest.raises(AgentMemoryError):
        bundle.write_reference_turn([_memory("sys_001", "secret", role="system")])
