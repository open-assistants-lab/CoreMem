"""Deterministic AgentJournal compiler adapter.

The compiler accepts a structured update plan, validates cited claims against
in-memory source messages when available, and writes the derived AgentJournal
files. It intentionally does not call an LLM.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coremem.agent_journal.bundle import (
    ACTIVATIONS,
    MEMORY_KINDS,
    PROFILE_VERSION,
    SCHEMA_VERSION,
    SCOPES,
    SOURCE_EVIDENCE_TYPES,
    STATUSES,
    TRUST_VALUES,
    AgentJournalBundle,
    AgentJournalError,
    _parse_frontmatter,
)

_PLAN_KEYS = {"pages", "log_message"}
_PAGE_REQUIRED_KEYS = {
    "operation",
    "page_id",
    "title",
    "description",
    "memory_kind",
    "scope",
    "status",
    "activation",
    "trust",
    "safe_to_act",
    "summary",
    "current_state",
    "boot_worthy",
}
_PAGE_OPTIONAL_KEYS = {"details", "open_questions", "read_next"}
_PAGE_KEYS = _PAGE_REQUIRED_KEYS | _PAGE_OPTIONAL_KEYS
_STATE_KEYS = {"claim", "evidence"}
_SOURCE_REQUIRED_KEYS = {"evidence_type", "source_turn_id"}
_SOURCE_OPTIONAL_KEYS = {"source_message_id", "source_quote"}
_SOURCE_KEYS = _SOURCE_REQUIRED_KEYS | _SOURCE_OPTIONAL_KEYS
_DERIVED_KEYS = {"evidence_type", "supporting_sources"}
_PAGE_ID_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SAFE_REFERENCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_RENDERED_FRONTMATTER_KEYS = {
    "type",
    "page_id",
    "title",
    "description",
    "memory_kind",
    "agent_journal_version",
    "scope",
    "status",
    "activation",
    "trust",
    "safe_to_act",
    "boot_worthy",
}


@dataclass(frozen=True)
class AgentJournalCompileResult:
    """Result metadata for a deterministic AgentJournal compile."""

    written_pages: tuple[Path, ...]
    boot_pages: tuple[str, ...]


@dataclass(frozen=True)
class _Citation:
    number: int
    source: Mapping[str, Any]


@dataclass(frozen=True)
class _CompiledClaim:
    claim: str
    citation_numbers: tuple[int, ...]


@dataclass(frozen=True)
class _CompiledPage:
    operation: str
    page_id: str
    path: Path
    title: str
    description: str
    memory_kind: str
    scope: str
    status: str
    activation: str
    trust: str
    safe_to_act: bool
    summary: str
    current_state: tuple[_CompiledClaim, ...]
    citations: tuple[_Citation, ...]
    boot_worthy: bool
    details: tuple[str, ...]
    open_questions: tuple[str, ...]
    read_next: tuple[str, ...]


class AgentJournalCompiler:
    """Apply structured AgentJournal update plans without LLM calls."""

    def __init__(self, bundle: AgentJournalBundle) -> None:
        self.bundle = bundle

    def apply_plan(self, plan: Mapping[str, Any], messages: Sequence[Any] | None = None) -> AgentJournalCompileResult:
        """Validate and apply a structured AgentJournal update plan."""
        pages = self._compile_plan(plan, messages)
        self._ensure_output_files()
        snapshots = self._snapshot_targets(pages)

        written: list[Path] = []
        try:
            for page in pages:
                page.path.parent.mkdir(parents=True, exist_ok=True)
                page.path.write_text(self._render_page(page), encoding="utf-8")
                written.append(page.path)

            self._write_index()
            self._append_log(pages, plan.get("log_message"))
            self._write_memory()

            lint_errors = self.bundle.lint()
            if lint_errors:
                raise AgentJournalError("compiled bundle failed lint: " + "; ".join(lint_errors))
        except Exception:
            self._restore_targets(snapshots)
            raise

        return AgentJournalCompileResult(
            written_pages=tuple(written),
            boot_pages=tuple(page.page_id for page in pages if page.boot_worthy),
        )

    def _compile_plan(self, plan: Mapping[str, Any], messages: Sequence[Any] | None = None) -> tuple[_CompiledPage, ...]:
        if not isinstance(plan, Mapping):
            raise AgentJournalError("compiler plan must be an object")
        self._reject_extra_keys(plan, _PLAN_KEYS, "compiler plan")

        raw_pages = plan.get("pages")
        if not isinstance(raw_pages, list) or not raw_pages:
            raise AgentJournalError("compiler plan pages must be a non-empty list")
        log_message = plan.get("log_message")
        if log_message is not None:
            self._require_string(log_message, "compiler plan log_message")

        seen_page_ids: set[str] = set()
        compiled: list[_CompiledPage] = []
        for index, raw_page in enumerate(raw_pages):
            if not isinstance(raw_page, Mapping):
                raise AgentJournalError(f"pages[{index}] must be an object")
            page = self._compile_page(raw_page, f"pages[{index}]", messages)
            if page.page_id in seen_page_ids:
                raise AgentJournalError(f"duplicate page_id in compiler plan: {page.page_id}")
            seen_page_ids.add(page.page_id)
            compiled.append(page)
        return tuple(compiled)

    def _compile_page(self, raw_page: Mapping[str, Any], label: str, messages: Sequence[Any] | None = None) -> _CompiledPage:
        self._reject_extra_keys(raw_page, _PAGE_KEYS, label)
        missing = sorted(key for key in _PAGE_REQUIRED_KEYS if key not in raw_page)
        if missing:
            raise AgentJournalError(f"{label} missing required keys: {', '.join(missing)}")

        operation = self._require_string(raw_page["operation"], f"{label}.operation")
        if operation not in {"create", "update"}:
            raise AgentJournalError(f"{label}.operation must be create or update")

        page_id = self._require_page_id(raw_page["page_id"], f"{label}.page_id")
        path = self._page_path(page_id)
        if operation == "create" and path.exists():
            raise AgentJournalError(f"{label} create would overwrite existing page: {page_id}")
        if operation == "update" and not path.exists():
            raise AgentJournalError(f"{label} update target does not exist: {page_id}")

        title = self._require_frontmatter_string(raw_page["title"], f"{label}.title")
        description = self._require_frontmatter_string(
            raw_page["description"], f"{label}.description"
        )
        memory_kind = self._require_string(raw_page["memory_kind"], f"{label}.memory_kind")
        scope = self._require_string(raw_page["scope"], f"{label}.scope")
        status = self._require_string(raw_page["status"], f"{label}.status")
        activation = self._require_string(raw_page["activation"], f"{label}.activation")
        trust = self._require_string(raw_page["trust"], f"{label}.trust")
        safe_to_act = self._require_bool(raw_page["safe_to_act"], f"{label}.safe_to_act")
        boot_worthy = self._require_bool(raw_page["boot_worthy"], f"{label}.boot_worthy")
        summary = self._require_body_string(raw_page["summary"], f"{label}.summary")

        if memory_kind not in MEMORY_KINDS:
            raise AgentJournalError(f"{label}.memory_kind is invalid")
        if scope not in SCOPES:
            raise AgentJournalError(f"{label}.scope is invalid")
        if status not in STATUSES:
            raise AgentJournalError(f"{label}.status is invalid")
        if activation not in ACTIVATIONS:
            raise AgentJournalError(f"{label}.activation is invalid")
        if trust not in TRUST_VALUES:
            raise AgentJournalError(f"{label}.trust is invalid")
        if boot_worthy and (activation != "startup" or status != "active"):
            raise AgentJournalError(
                f"{label}.boot_worthy requires activation=startup and status=active"
            )

        current_state, citations = self._compile_current_state(raw_page["current_state"], label, messages)
        details = self._optional_string_list(raw_page, "details", label)
        open_questions = self._optional_string_list(raw_page, "open_questions", label)
        read_next = self._optional_string_list(raw_page, "read_next", label)

        return _CompiledPage(
            operation=operation,
            page_id=page_id,
            path=path,
            title=title,
            description=description,
            memory_kind=memory_kind,
            scope=scope,
            status=status,
            activation=activation,
            trust=trust,
            safe_to_act=safe_to_act,
            summary=summary,
            current_state=current_state,
            citations=citations,
            boot_worthy=boot_worthy,
            details=details,
            open_questions=open_questions,
            read_next=read_next,
        )

    def _compile_current_state(
        self, raw_current_state: Any, page_label: str, messages: Sequence[Any] | None = None
    ) -> tuple[tuple[_CompiledClaim, ...], tuple[_Citation, ...]]:
        if not isinstance(raw_current_state, list) or not raw_current_state:
            raise AgentJournalError(f"{page_label}.current_state must be a non-empty list")

        claims: list[_CompiledClaim] = []
        citations: list[_Citation] = []
        for index, raw_claim in enumerate(raw_current_state):
            label = f"{page_label}.current_state[{index}]"
            if not isinstance(raw_claim, Mapping):
                raise AgentJournalError(f"{label} must be an object")
            self._reject_extra_keys(raw_claim, _STATE_KEYS, label)
            missing = sorted(key for key in _STATE_KEYS if key not in raw_claim)
            if missing:
                raise AgentJournalError(f"{label} missing required keys: {', '.join(missing)}")
            claim = self._require_body_string(raw_claim["claim"], f"{label}.claim")
            evidence = self._compile_evidence(raw_claim["evidence"], label)
            validation_claim = dict(evidence)
            validation_claim["text"] = claim
            errors = self.bundle.validate_claim(validation_claim, messages)
            if errors:
                raise AgentJournalError(f"{label} evidence is invalid: " + "; ".join(errors))

            citation_numbers: list[int] = []
            for source in self._citation_sources(evidence):
                citations.append(_Citation(number=len(citations) + 1, source=source))
                citation_numbers.append(citations[-1].number)
            claims.append(_CompiledClaim(claim=claim, citation_numbers=tuple(citation_numbers)))
        return tuple(claims), tuple(citations)

    def _compile_evidence(self, raw_evidence: Any, label: str) -> Mapping[str, Any]:
        if not isinstance(raw_evidence, Mapping):
            raise AgentJournalError(f"{label}.evidence must be an object")

        evidence_type = raw_evidence.get("evidence_type")
        if evidence_type == "derived_summary":
            self._reject_extra_keys(raw_evidence, _DERIVED_KEYS, f"{label}.evidence")
            sources = raw_evidence.get("supporting_sources")
            if not isinstance(sources, list) or len(sources) < 1:
                raise AgentJournalError(
                    f"{label}.evidence.supporting_sources must contain at least one source"
                )
            compiled_sources = [
                self._compile_source(source, f"{label}.evidence.supporting_sources[{index}]")
                for index, source in enumerate(sources)
            ]
            return {"evidence_type": "derived_summary", "supporting_sources": compiled_sources}

        return self._compile_source(raw_evidence, f"{label}.evidence")

    def _compile_source(self, raw_source: Any, label: str) -> Mapping[str, str]:
        if not isinstance(raw_source, Mapping):
            raise AgentJournalError(f"{label} must be an object")
        self._reject_extra_keys(raw_source, _SOURCE_KEYS, label)
        missing = sorted(key for key in _SOURCE_REQUIRED_KEYS if key not in raw_source)
        if missing:
            raise AgentJournalError(f"{label} missing required keys: {', '.join(missing)}")

        evidence_type = self._require_string(raw_source["evidence_type"], f"{label}.evidence_type")
        if evidence_type not in SOURCE_EVIDENCE_TYPES:
            raise AgentJournalError(f"{label}.evidence_type is invalid")

        source = {
            "evidence_type": evidence_type,
            "source_turn_id": self._require_citation_string(
                raw_source["source_turn_id"], f"{label}.source_turn_id"
            ),
        }
        # Optional fields
        if "source_message_id" in raw_source:
            source["source_message_id"] = self._require_citation_string(
                raw_source["source_message_id"], f"{label}.source_message_id"
            )
        else:
            source["source_message_id"] = ""
        if "source_quote" in raw_source:
            source["source_quote"] = self._require_quote(raw_source["source_quote"], f"{label}.source_quote")
        else:
            source["source_quote"] = ""
        return source

    def _citation_sources(self, evidence: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        if evidence.get("evidence_type") == "derived_summary":
            sources = evidence.get("supporting_sources", [])
            return tuple(source for source in sources if isinstance(source, Mapping))
        return (evidence,)

    def _render_page(self, page: _CompiledPage) -> str:
        safe_to_act = "true" if page.safe_to_act else "false"
        boot_worthy = "true" if page.boot_worthy else "false"
        lines = [
            "---",
            "type: AgentJournal Page",
            f"page_id: {page.page_id}",
            f"title: {page.title}",
            f"description: {page.description}",
            f"memory_kind: {page.memory_kind}",
            f"agent_journal_version: \"{PROFILE_VERSION}\"",
            f"scope: {page.scope}",
            f"status: {page.status}",
            f"activation: {page.activation}",
            f"trust: {page.trust}",
            f"safe_to_act: {safe_to_act}",
            f"boot_worthy: {boot_worthy}",
            *self._preserved_frontmatter_lines(page),
            "---",
            "",
            "# Summary",
            "",
            page.summary,
            "",
            "# Current State",
            "",
        ]
        for claim in page.current_state:
            markers = " ".join(f"[{number}]" for number in claim.citation_numbers)
            lines.append(f"- {claim.claim} {markers}".rstrip())
        self._append_optional_section(lines, "Details", page.details)
        self._append_optional_section(lines, "Open Questions", page.open_questions)
        self._append_optional_section(lines, "Read Next", page.read_next)
        lines.extend(["", "# Citations", ""])
        for citation in page.citations:
            source = citation.source
            smid = source.get("source_message_id", "")
            quote = source.get("source_quote", "")
            line = f"[{citation.number}] {source['source_turn_id']}"
            if smid:
                line += f", `{smid}`"
            line += f", `{source['evidence_type']}`:"
            lines.append(line)
            if quote:
                lines.append(f"\"{quote}\"")
        lines.append("")
        return "\n".join(lines)

    def _preserved_frontmatter_lines(self, page: _CompiledPage) -> list[str]:
        if page.operation != "update" or not page.path.exists():
            return []
        frontmatter, _ = _parse_frontmatter(page.path.read_text(encoding="utf-8"))
        lines: list[str] = []
        for key, value in frontmatter.items():
            if key in _RENDERED_FRONTMATTER_KEYS:
                continue
            lines.extend(self._frontmatter_lines(key, value))
        return lines

    def _frontmatter_lines(self, key: str, value: object) -> list[str]:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_-]*$", key):
            return []
        if isinstance(value, list):
            lines = [f"{key}:"]
            lines.extend(f"  - {self._frontmatter_scalar(item)}" for item in value)
            return lines
        return [f"{key}: {self._frontmatter_scalar(value)}"]

    def _frontmatter_scalar(self, value: object) -> str:
        text = str(value)
        if text in {"true", "false"} or re.match(r"^-?\d+(\.\d+)?$", text):
            return text
        if re.search(r"[:#\[\],{}]|^\s|\s$", text):
            return "\"" + text.replace("\\", "\\\\").replace("\"", "\\\"") + "\""
        return text

    def _append_optional_section(
        self, lines: list[str], title: str, items: tuple[str, ...]
    ) -> None:
        if not items:
            return
        lines.extend(["", f"# {title}", ""])
        lines.extend(f"- {item}" for item in items)

    def _write_index(self) -> None:
        entries = self._page_entries()
        lines = ["# AgentJournal Index", ""]
        for entry in entries:
            description = f": {entry['description']}" if entry["description"] else ""
            lines.append(
                f"- [{entry['title']}]({entry['relative_path']}) "
                f"(`{entry['memory_kind']}`, {entry['status']}){description}"
            )
        lines.append("")
        (self.bundle.root / "index.md").write_text("\n".join(lines), encoding="utf-8")

    def _append_log(self, pages: tuple[_CompiledPage, ...], log_message: object) -> None:
        path = self.bundle.root / "log.md"
        existing = path.read_text(encoding="utf-8") if path.exists() else "# AgentJournal Update Log\n"
        header = "# AgentJournal Update Log"
        old_header = "# AgentMemory Update Log"
        if existing.startswith(old_header):
            existing = header + existing[len(old_header):]
        body = existing[len(header):].strip() if existing.startswith(header) else existing.strip()
        timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        lines = [header, "", f"## {timestamp} compiler | {SCHEMA_VERSION}", ""]
        if isinstance(log_message, str) and log_message:
            lines.extend([log_message, ""])
        for page in pages:
            relative = page.path.relative_to(self.bundle.root).as_posix()
            lines.append(f"- {page.operation} [{page.title}]({relative}) (`{page.page_id}`)")
        lines.append("")
        if body:
            lines.extend([body, ""])
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_memory(self) -> None:
        boot_entries = [
            entry
            for entry in self._page_entries()
            if entry["boot_worthy"] and entry["activation"] == "startup" and entry["status"] == "active"
        ]
        lines = ["# AgentJournal", "", "## Current Focus", ""]
        for entry in boot_entries:
            summary = entry["summary"]
            suffix = f": {summary}" if summary else ""
            lines.append(f"- [{entry['title']}]({entry['relative_path']}){suffix}")
        lines.extend(["", "## Read Next", ""])
        for entry in boot_entries:
            lines.append(f"- [{entry['title']}]({entry['relative_path']})")
        lines.append("")
        (self.bundle.root / "MEMORY.md").write_text("\n".join(lines), encoding="utf-8")

    def _page_entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if not self.bundle.pages_dir.exists():
            return entries
        for path in sorted(self.bundle.pages_dir.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            frontmatter, body = _parse_frontmatter(text)
            if frontmatter.get("type") != "AgentJournal Page":
                continue
            page_id = frontmatter.get("page_id")
            if not isinstance(page_id, str):
                continue
            entries.append({
                "page_id": page_id,
                "title": str(frontmatter.get("title") or page_id),
                "description": str(frontmatter.get("description") or ""),
                "memory_kind": str(frontmatter.get("memory_kind") or "unknown"),
                "status": str(frontmatter.get("status") or "unknown"),
                "activation": str(frontmatter.get("activation") or "unknown"),
                "boot_worthy": frontmatter.get("boot_worthy") is True,
                "summary": self._summary(body),
                "relative_path": path.relative_to(self.bundle.root).as_posix(),
            })
        entries.sort(key=lambda entry: str(entry["page_id"]))
        return entries

    def _summary(self, body: str) -> str:
        match = re.search(r"^# Summary\n(.*?)(?=^# |\Z)", body, re.MULTILINE | re.DOTALL)
        if not match:
            return ""
        return " ".join(line.strip() for line in match.group(1).splitlines() if line.strip())

    def _snapshot_targets(self, pages: tuple[_CompiledPage, ...]) -> dict[str, bytes]:
        targets: dict[str, bytes] = {}
        for page in pages:
            path = page.path
            if path.exists():
                targets[str(path)] = path.read_bytes()
        for name in ("MEMORY.md", "index.md", "log.md"):
            path = self.bundle.root / name
            if path.exists():
                targets[str(path)] = path.read_bytes()
        return targets

    def _restore_targets(self, snapshots: dict[str, bytes]) -> None:
        for path_str, content in snapshots.items():
            path = Path(path_str)
            path.write_bytes(content)

    def _ensure_output_files(self) -> None:
        self.bundle.root.mkdir(parents=True, exist_ok=True)
        self.bundle.pages_dir.mkdir(parents=True, exist_ok=True)
        defaults = {
            "MEMORY.md": "# AgentJournal\n\n## Current Focus\n\n## Read Next\n",
            "index.md": "# AgentJournal Index\n\n",
            "log.md": "# AgentJournal Update Log\n\n",
            "SCHEMA.md": f"# AgentJournal Schema\n\nSchema version: `{SCHEMA_VERSION}`\n",
        }
        for relative, content in defaults.items():
            path = self.bundle.root / relative
            if not path.exists():
                path.write_text(content, encoding="utf-8")

    def _page_path(self, page_id: str) -> Path:
        segments = page_id.split(".")
        return self.bundle.pages_dir.joinpath(*segments).with_suffix(".md")

    def _reject_extra_keys(self, value: Mapping[str, Any], allowed: set[str], label: str) -> None:
        extra = sorted(str(key) for key in value.keys() if key not in allowed)
        if extra:
            raise AgentJournalError(f"{label} has unsupported keys: {', '.join(extra)}")

    def _require_page_id(self, value: Any, label: str) -> str:
        page_id = self._require_string(value, label)
        segments = page_id.split(".")
        if any(not _PAGE_ID_SEGMENT_RE.match(segment) for segment in segments):
            raise AgentJournalError(f"{label} is invalid")
        path = self._page_path(page_id).resolve()
        try:
            path.relative_to(self.bundle.pages_dir.resolve())
        except ValueError as exc:
            raise AgentJournalError(f"{label} must stay under pages/") from exc
        return page_id

    def _require_frontmatter_string(self, value: Any, label: str) -> str:
        text = self._require_string(value, label)
        if "\n" in text or "\r" in text:
            raise AgentJournalError(f"{label} must be a single line")
        return text

    def _require_body_string(self, value: Any, label: str) -> str:
        text = self._require_string(value, label)
        if "references/turns/" in text:
            raise AgentJournalError(f"{label} must not contain direct reference links")
        return text

    def _require_citation_string(self, value: Any, label: str) -> str:
        text = self._require_string(value, label)
        if "`" in text or "\n" in text or "\r" in text:
            raise AgentJournalError(f"{label} cannot be rendered as a AgentJournal citation")
        return text

    def _require_quote(self, value: Any, label: str) -> str:
        quote = self._require_string(value, label)
        if '"' in quote or "\n" in quote or "\r" in quote:
            raise AgentJournalError(f"{label} cannot contain double quotes or newlines")
        return quote

    def _require_string(self, value: Any, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise AgentJournalError(f"{label} must be a non-empty string")
        return value.strip()

    def _require_bool(self, value: Any, label: str) -> bool:
        if not isinstance(value, bool):
            raise AgentJournalError(f"{label} must be boolean")
        return value

    def _optional_string_list(
        self, raw_page: Mapping[str, Any], key: str, page_label: str
    ) -> tuple[str, ...]:
        if key not in raw_page:
            return ()
        value = raw_page[key]
        if not isinstance(value, list):
            raise AgentJournalError(f"{page_label}.{key} must be a list")
        return tuple(
            self._require_body_string(item, f"{page_label}.{key}[{index}]")
            for index, item in enumerate(value)
        )

    def compile_section(
        self,
        plan: Mapping[str, Any],
        *,
        timestamp: str,
        title: str | None = None,
        messages: Sequence[Any] | None = None,
    ) -> str:
        """Validate a plan and return a daily journal section string.

        Reuses the same claim validation as apply_plan() but outputs a
        ``## HH:MM - Title`` section instead of writing files.
        """
        pages = self._compile_plan(plan, messages)
        if len(pages) != 1:
            raise AgentJournalError("compile_section requires exactly one page in the plan")
        page = pages[0]
        lines = [f"## {timestamp} - {title or page.title}", ""]
        lines.append(page.summary)
        lines.append("")
        lines.append("**Claims:**")
        for claim in page.current_state:
            markers = " ".join(f"[{number}]" for number in claim.citation_numbers)
            lines.append(f"- {claim.claim} {markers}".rstrip())
        lines.append("")
        lines.append("**Citations:**")
        for citation in page.citations:
            source = citation.source
            lines.append(
                f"[{citation.number}] {source.get('source_message_id', source['source_turn_id'])} "
                f"({source['evidence_type']}): \"{source.get('source_quote', '')}\""
            )
        lines.append("")
        return "\n".join(lines)


def compile_journal_plan(
    bundle: AgentJournalBundle, plan: Mapping[str, Any], messages: Sequence[Any] | None = None
) -> AgentJournalCompileResult:
    """Apply a structured AgentJournal update plan to a bundle."""
    return AgentJournalCompiler(bundle).apply_plan(plan, messages)
