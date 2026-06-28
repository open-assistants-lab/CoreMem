"""Tests for the deterministic MemoryPack compiler adapter."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from coremem.agent_memory import AgentMemoryBundle, AgentMemoryCompiler, AgentMemoryError
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


def _source(
    *,
    turn_id: str = "turn_001",
    message_id: str = "msg_001",
    evidence_type: str = "user_statement",
    quote: str = "message -> wiki (okf) -> search",
) -> dict[str, str]:
    return {
        "evidence_type": evidence_type,
        "source_turn_id": turn_id,
        "source_message_id": message_id,
        "source_quote": quote,
    }


def _page_plan(
    *,
    page_id: str = "decisions.memory-architecture",
    title: str = "Memory Architecture",
    memory_kind: str = "decision",
    boot_worthy: bool = True,
    evidence: dict | None = None,
) -> dict:
    return {
        "operation": "create",
        "page_id": page_id,
        "title": title,
        "description": "Current decisions about MemoryPack architecture.",
        "memory_kind": memory_kind,
        "scope": "project",
        "status": "active" if boot_worthy else "active",
        "activation": "startup" if boot_worthy else "query",
        "trust": "user_authoritative",
        "safe_to_act": True,
        "summary": "CoreMem is testing MemoryPack as compiled markdown memory.",
        "current_state": [
            {
                "claim": "The POC starts with direct MemoryPack search before observations.",
                "evidence": evidence or _source(),
            }
        ],
        "details": ["The compiler adapter only consumes structured dict plans."],
        "open_questions": ["How much state should be boot-worthy by default?"],
        "read_next": ["Evaluate direct page search against raw references."],
        "boot_worthy": boot_worthy,
    }


def _snapshot_references(bundle: AgentMemoryBundle) -> dict[str, bytes]:
    return {
        path.relative_to(bundle.references_dir).as_posix(): path.read_bytes()
        for path in sorted(bundle.references_dir.rglob("*"))
        if path.is_file()
    }


def test_compiler_creates_valid_page_from_plan(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "message -> wiki (okf) -> search, as POC?")],
        turn_id="turn_001",
    )

    result = AgentMemoryCompiler(bundle).apply_plan({"pages": [_page_plan()]})

    page_path = tmp_path / "memorypack" / "pages" / "decisions" / "memory-architecture.md"
    text = page_path.read_text(encoding="utf-8")
    assert result.written_pages == (page_path,)
    assert "agent_memory_version: \"0.1\"" in text
    assert "# Summary" in text
    assert "# Current State" in text
    assert "# Citations" in text
    assert "[turn_001](../../references/turns/turn_001.md)" in text
    assert bundle.lint() == []


def test_compiler_updates_existing_page_from_plan(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "message -> wiki (okf) -> search, as POC?")],
        turn_id="turn_001",
    )
    compiler = AgentMemoryCompiler(bundle)
    compiler.apply_plan({"pages": [_page_plan()]})
    update_page = {
        **_page_plan(),
        "operation": "update",
        "summary": "CoreMem now has a deterministic compiler adapter.",
    }

    compiler.apply_plan({"pages": [update_page]})

    page_path = bundle.root / "pages" / "decisions" / "memory-architecture.md"
    assert "CoreMem now has a deterministic compiler adapter." in page_path.read_text(
        encoding="utf-8"
    )
    assert bundle.lint() == []


def test_compiler_rejects_invalid_quote(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "message -> wiki (okf) -> search, as POC?")],
        turn_id="turn_001",
    )
    plan = {"pages": [_page_plan(evidence=_source(quote="message -> memorypack -> search"))]}

    with pytest.raises(AgentMemoryError, match="source_quote is not an exact substring"):
        AgentMemoryCompiler(bundle).apply_plan(plan)

    page_path = tmp_path / "memorypack" / "pages" / "decisions" / "memory-architecture.md"
    assert not page_path.exists()


def test_compiler_rejects_role_evidence_mismatch(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "assistant said this", role="assistant")],
        turn_id="turn_001",
    )
    plan = {"pages": [_page_plan(evidence=_source(quote="assistant said this"))]}

    with pytest.raises(AgentMemoryError, match="not supported by role"):
        AgentMemoryCompiler(bundle).apply_plan(plan)

    page_path = tmp_path / "memorypack" / "pages" / "decisions" / "memory-architecture.md"
    assert not page_path.exists()


def test_compiler_supports_derived_summary(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "message -> wiki (okf) -> search, as POC?")],
        turn_id="turn_001",
    )
    bundle.write_reference_turn(
        [_memory("msg_002", "prove that your hybrid theory works")],
        turn_id="turn_002",
    )
    derived = {
        "evidence_type": "derived_summary",
        "supporting_sources": [
            _source(),
            _source(
                turn_id="turn_002",
                message_id="msg_002",
                quote="hybrid theory works",
            ),
        ],
    }

    AgentMemoryCompiler(bundle).apply_plan({"pages": [_page_plan(evidence=derived)]})

    text = (bundle.root / "pages" / "decisions" / "memory-architecture.md").read_text(
        encoding="utf-8"
    )
    assert "[1] [2]" in text
    assert "[turn_002](../../references/turns/turn_002.md)" in text
    assert bundle.lint() == []


def test_compiler_updates_index_log_and_memory(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "message -> wiki (okf) -> search, as POC?")],
        turn_id="turn_001",
    )
    boot_page = _page_plan(title="Boot Worthy Page", boot_worthy=True)
    non_boot_page = _page_plan(
        page_id="workflow.background-note",
        title="Background Note",
        memory_kind="workflow",
        boot_worthy=False,
    )

    result = AgentMemoryCompiler(bundle).apply_plan({
        "pages": [boot_page, non_boot_page],
        "log_message": "structured compiler smoke test",
    })

    index = (bundle.root / "index.md").read_text(encoding="utf-8")
    log = (bundle.root / "log.md").read_text(encoding="utf-8")
    memory = (bundle.root / "MEMORY.md").read_text(encoding="utf-8")
    assert result.boot_pages == ("decisions.memory-architecture",)
    assert "[Boot Worthy Page](pages/decisions/memory-architecture.md)" in index
    assert "[Background Note](pages/workflow/background-note.md)" in index
    assert "structured compiler smoke test" in log
    assert "create [Boot Worthy Page](pages/decisions/memory-architecture.md)" in log
    assert "Boot Worthy Page" in memory
    assert "Background Note" not in memory
    assert "references/turns/" not in memory
    assert bundle.lint() == []


def test_compiler_does_not_mutate_references(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "message -> wiki (okf) -> search, as POC?")],
        turn_id="turn_001",
    )
    before = _snapshot_references(bundle)

    AgentMemoryCompiler(bundle).apply_plan({"pages": [_page_plan()]})

    assert _snapshot_references(bundle) == before


def test_compiler_outputs_lint_clean_bundle(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "message -> wiki (okf) -> search, as POC?")],
        turn_id="turn_001",
    )

    AgentMemoryCompiler(bundle).apply_plan({"pages": [_page_plan()]})

    assert bundle.lint() == []


def test_compiler_rolls_back_on_lint_failure(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    turn_a = [_memory("msg_001", "the sky is blue")]
    turn_b = [_memory("msg_002", "water is wet")]
    bundle.write_reference_turn(turn_a, turn_id="turn_001")
    bundle.write_reference_turn(turn_b, turn_id="turn_002")

    # Create one page successfully, then corrupt it to introduce a lint error.
    page_a = _page_plan(
        page_id="page.a",
        title="Page A",
        evidence=_source(turn_id="turn_001", quote="the sky is blue"),
    )
    compiler = AgentMemoryCompiler(bundle)
    result = compiler.apply_plan({"pages": [page_a]})
    assert result.written_pages

    page_at_path = bundle.root / "pages" / "page" / "a.md"
    corrupted = page_at_path.read_text(encoding="utf-8").replace(
        'agent_memory_version: "0.1"', 'agent_memory_version: "9.9"'
    )
    page_at_path.write_text(corrupted, encoding="utf-8")

    # Snapshot root files before second plan, so we can verify rollback.
    pre_root: dict[str, bytes] = {}
    for name in ("MEMORY.md", "index.md", "log.md"):
        pre_root[name] = (bundle.root / name).read_bytes()

    page_b = _page_plan(
        page_id="page.b",
        title="Page B",
        evidence=_source(
            turn_id="turn_002", message_id="msg_002", quote="water is wet"
        ),
    )
    with pytest.raises(AgentMemoryError, match="compiled bundle failed lint"):
        compiler.apply_plan({"pages": [page_b]})

    # page_b (the create target) was not snapshotted, so it may remain as
    # an orphan — that is the documented partial-atomicity limitation for
    # newly created files.  The root files, however, MUST be restored.
    for name in ("MEMORY.md", "index.md", "log.md"):
        assert (bundle.root / name).read_bytes() == pre_root[name]

    # page_a's corruption is preserved (the compiler never touched it).
    assert 'agent_memory_version: "9.9"' in page_at_path.read_text(encoding="utf-8")


def test_compiler_preserves_extra_frontmatter_on_update(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "message -> wiki (okf) -> search, as POC?")],
        turn_id="turn_001",
    )
    compiler = AgentMemoryCompiler(bundle)
    compiler.apply_plan({"pages": [_page_plan()]})

    # Inject extra frontmatter keys manually.
    page_path = bundle.root / "pages" / "decisions" / "memory-architecture.md"
    text = page_path.read_text(encoding="utf-8")
    _, after = text.split("---\n", 1)
    frontmatter, body = after.split("---\n", 1)
    extra = "x-custom: hello\nx-list:\n  - one\n  - two\n"
    modified = f"---\n{frontmatter}{extra}---\n{body}"
    page_path.write_text(modified, encoding="utf-8")

    update = _page_plan()
    update["summary"] = "Updated summary text here."
    update["operation"] = "update"
    compiler.apply_plan({"pages": [update]})

    text = page_path.read_text(encoding="utf-8")
    assert "x-custom: hello" in text
    assert "  - one" in text
    assert "  - two" in text
    assert "Updated summary text here." in text
    assert bundle.lint() == []


def test_boot_worthy_valid_page_appears_in_memory(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "agent identity is CoreMem v0.10")],
        turn_id="turn_001",
    )
    plan = _page_plan(page_id="agent-identity", title="Agent Identity",
                       boot_worthy=True, evidence=_source(quote="agent identity is CoreMem v0.10"))
    plan["status"] = "active"
    plan["activation"] = "startup"
    result = AgentMemoryCompiler(bundle).apply_plan({"pages": [plan]})
    assert len(result.boot_pages) == 1
    assert result.boot_pages[0] == "agent-identity"
    mem = (tmp_path / "memorypack" / "MEMORY.md").read_text()
    assert "agent-identity" in mem
    assert "# Current Focus" in mem


def test_boot_worthy_rejects_wrong_activation(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "test")],
        turn_id="turn_001",
    )
    plan = _page_plan(page_id="bad-boot", boot_worthy=True, evidence=_source(quote="test"))
    plan["activation"] = "manual"
    plan["status"] = "active"
    with pytest.raises(AgentMemoryError, match="boot_worthy requires activation=startup"):
        AgentMemoryCompiler(bundle).apply_plan({"pages": [plan]})


def test_boot_worthy_rejects_wrong_status(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "test")],
        turn_id="turn_001",
    )
    plan = _page_plan(page_id="bad-boot", boot_worthy=True, evidence=_source(quote="test"))
    plan["activation"] = "startup"
    plan["status"] = "pending_review"
    with pytest.raises(AgentMemoryError, match="boot_worthy requires activation=startup"):
        AgentMemoryCompiler(bundle).apply_plan({"pages": [plan]})


def test_non_boot_page_omitted_from_memory(tmp_path):
    bundle = AgentMemoryBundle(tmp_path / "memorypack")
    bundle.write_reference_turn(
        [_memory("msg_001", "mundane conversation")],
        turn_id="turn_001",
    )
    plan = _page_plan(page_id="mundane-page", boot_worthy=False, evidence=_source(quote="mundane conversation"))
    plan["activation"] = "query"
    result = AgentMemoryCompiler(bundle).apply_plan({"pages": [plan]})
    assert len(result.boot_pages) == 0
    mem = (tmp_path / "memorypack" / "MEMORY.md").read_text()
    assert "mundane-page" not in mem
