"""AgentJournal bundle primitives.

This module implements the deterministic substrate for the AgentJournal POC:
bundle initialization, immutable reference turns, linting, exact quote
validation, and simple markdown search. It intentionally does not call an LLM.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from coremem.types import Memory

logger = logging.getLogger(__name__)

PROFILE_VERSION = "0.1"
SCHEMA_VERSION = "agent-journal-0.1"
ALLOWED_ROLES = {"system", "developer", "user", "assistant", "tool_call", "tool_result"}
MEMORY_KINDS = {
    "user_profile",
    "preference",
    "project_fact",
    "decision",
    "workflow",
    "active_context",
    "open_question",
    "conflict",
    "daily_note",
    "dream",
}
SCOPES = {"user", "project", "workspace", "global"}
STATUSES = {"active", "superseded", "unresolved", "archived", "pending_review"}
ACTIVATIONS = {"startup", "query", "model_decision", "manual"}
EVIDENCE_TYPES = {"user_statement", "assistant_action", "tool_observation", "derived_summary"}
SOURCE_EVIDENCE_TYPES = {"user_statement", "assistant_action", "tool_observation"}
TRUST_VALUES = {
    "user_authoritative",
    "tool_observed",
    "assistant_derived",
    "untrusted_source",
    "mixed",
}
_SAFE_REFERENCE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class AgentJournalError(ValueError):
    """Raised when a AgentJournal operation cannot be completed safely."""


@dataclass(frozen=True)
class SearchHit:
    """A simple AgentJournal search result."""

    path: Path
    score: float
    session_id: str = ""


def compute_agent_context_hash(context: Mapping[str, Any]) -> str:
    """Compute a stable hash for an agent context manifest payload."""
    payload = json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _frontmatter_list(values: Sequence[str]) -> str:
    return "[" + ", ".join(values) + "]"


def _parse_scalar(value: str) -> object:
    value = value.strip()
    if value == "":
        return ""
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip('"\'') for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"\'')


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    raw = text[4:end]
    body = text[end + 5 :]
    frontmatter: dict[str, object] = {}
    current_key: str | None = None
    for line in raw.splitlines():
        if not line.strip():
            continue
        if line.startswith("  - ") and current_key:
            existing = frontmatter.setdefault(current_key, [])
            if isinstance(existing, list):
                existing.append(line[4:].strip().strip('"\''))
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value == "":
            frontmatter[key] = []
            current_key = key
        else:
            frontmatter[key] = _parse_scalar(value)
            current_key = None
    return frontmatter, body


def _extract_turn_payload(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    parts = text.split("\n# Canonical Turn Payload\n", 1)
    if len(parts) != 2:
        return None, ["reference turn must contain a # Canonical Turn Payload section"]
    matches = re.findall(r"```json agent_journal-turn\n(.*?)\n```", parts[1], re.DOTALL)
    if len(matches) != 1:
        return None, ["reference turn must contain exactly one `json agent_journal-turn` block"]
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        return None, [f"reference turn canonical payload is invalid JSON: {exc}"]
    if not isinstance(payload, dict):
        return None, ["reference turn canonical payload must be a JSON object"]
    return payload, []


def _find_links(text: str) -> list[str]:
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    return [link for link in links if not re.match(r"^[a-z][a-z0-9+.-]*:", link)]


def _extract_citation_claims(text: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r"(?:^[-*]\s+)?(?:\[[^\]]+\]\s+)?"
        r"\[[^\]]+\]\(([^)]+references/turns/[^)]+?\.md)\),\s*"
        r"`([^`]+)`,\s*`([^`]+)`:\s*(?:\n)?\"([^\"]+)\"",
        re.MULTILINE,
    )
    claims: list[dict[str, str]] = []
    for match in pattern.finditer(text):
        link, message_id, evidence_type, quote = match.groups()
        turn_id = Path(link).stem
        claims.append({
            "evidence_type": evidence_type,
            "source_turn_id": turn_id,
            "source_message_id": message_id,
            "source_quote": quote,
        })
    return claims


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return [value]
    return []


class AgentJournalBundle:
    """A local AgentJournal bundle rooted at a directory."""

    def __init__(self, root: str | Path, *, boot_budget_chars: int = 8000, embed_model: str | None = None) -> None:
        self.root = Path(root)
        self.boot_budget_chars = boot_budget_chars
        self._embed_model = embed_model
        self._embedding_index: Any = None

    @property
    def references_dir(self) -> Path:
        return self.root / "references"

    @property
    def turns_dir(self) -> Path:
        return self.references_dir / "turns"

    @property
    def pages_dir(self) -> Path:
        return self.root / "pages"

    @property
    def daily_dir(self) -> Path:
        return self.root / "daily"

    @property
    def manifest_path(self) -> Path:
        return self.references_dir / "manifest.json"

    def initialize(self) -> None:
        """Create an empty AgentJournal bundle if needed."""
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "agent_context").mkdir(parents=True, exist_ok=True)
        (self.root / "schemas").mkdir(parents=True, exist_ok=True)

        self._write_if_missing("MEMORY.md", "# AgentJournal\n\n## Current Focus\n\n## Read Next\n")
        self._write_if_missing("index.md", "# AgentJournal Index\n\n")
        self._write_if_missing("log.md", "# AgentJournal Update Log\n\n")
        self._write_if_missing(
            "SCHEMA.md",
            f"# AgentJournal Schema\n\nSchema version: `{SCHEMA_VERSION}`\n",
        )
        agent_manifest = self.root / "agent_context" / "manifest.json"
        if not agent_manifest.exists():
            agent_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "persist_system_prompts": False,
                        "created_at": _utc_now(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    def write_reference_turn(
        self,
        messages: Sequence[Memory],
        *,
        turn_id: str | None = None,
        session_id: str | None = None,
        agent_context_hash: str = "unavailable",
        metadata: Mapping[str, Any] | None = None,
        include_system: bool = False,
    ) -> Path:
        """Write an immutable reference turn and append it to the manifest."""
        self.initialize()
        kept = [msg for msg in messages if include_system or msg.role not in {"system", "developer"}]
        if not kept:
            raise AgentJournalError("reference turn must contain at least one persisted message")
        ids = [msg.id for msg in kept]
        if len(ids) != len(set(ids)):
            raise AgentJournalError("reference turn message ids must be unique")
        turn_id = turn_id or f"turn_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}_{uuid4().hex[:8]}"
        if not re.match(r"^[A-Za-z0-9_.-]+$", turn_id):
            raise AgentJournalError("turn_id may only contain letters, numbers, dots, underscores, and hyphens")
        session_id = session_id or kept[0].session_id or "default"
        path = self.turns_dir / f"{turn_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)

        sorted_messages = sorted(kept, key=self._message_sort_key)
        payload = {
            "turn_id": turn_id,
            "session_id": session_id,
            "started_at": self._message_ts(sorted_messages[0]),
            "ended_at": self._message_ts(sorted_messages[-1]),
            "agent_context_hash": agent_context_hash,
            "messages": [self._message_payload(msg) for msg in sorted_messages],
            "metadata": dict(metadata or {}),
        }
        content = self._render_reference_turn(
            turn_id=turn_id,
            session_id=session_id,
            message_ids=ids,
            agent_context_hash=agent_context_hash,
            messages=sorted_messages,
            payload=payload,
            sensitive=include_system,
        )
        path.write_text(content, encoding="utf-8")
        self._append_manifest(path, turn_id, ids, agent_context_hash)
        return path

    def validate_claim(self, claim: Mapping[str, Any]) -> list[str]:
        """Validate a proposed claim's evidence against reference turns."""
        errors: list[str] = []
        evidence_type = claim.get("evidence_type")
        if evidence_type not in EVIDENCE_TYPES:
            return ["claim evidence_type is invalid"]
        if evidence_type == "derived_summary":
            for key in ("source_turn_id", "source_message_id", "source_quote"):
                if key in claim:
                    errors.append("derived_summary claims must not use top-level source fields")
                    break
            sources = claim.get("supporting_sources")
            if not isinstance(sources, list) or len(sources) < 2:
                errors.append("derived_summary claims need at least two supporting_sources")
            else:
                for index, source in enumerate(sources):
                    if not isinstance(source, Mapping):
                        errors.append(f"supporting_sources[{index}] must be an object")
                        continue
                    if source.get("evidence_type") not in SOURCE_EVIDENCE_TYPES:
                        errors.append(f"supporting_sources[{index}] evidence_type is invalid")
                    errors.extend(self._validate_source(source, f"supporting_sources[{index}]"))
            return errors
        return self._validate_source(claim, "claim")

    def lint(self) -> list[str]:
        """Return deterministic AgentJournal lint errors."""
        errors: list[str] = []
        for relative in ("MEMORY.md", "index.md", "log.md", "SCHEMA.md"):
            if not (self.root / relative).exists():
                errors.append(f"missing required file: {relative}")
        errors.extend(self._lint_memory_file())
        errors.extend(self._lint_memory_pages())
        errors.extend(self._lint_manifest())
        return errors

    def _write_if_missing(self, relative: str, content: str) -> None:
        path = self.root / relative
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    def _message_ts(self, message: Memory) -> str:
        return message.ts.isoformat() if message.ts else _utc_now()

    def _message_sort_key(self, message: Memory) -> str:
        return message.ts.isoformat() if message.ts else ""

    def _normalize_role(self, role: str) -> str:
        if role == "tool":
            return "tool_result"
        return role

    def _message_payload(self, message: Memory) -> dict[str, Any]:
        role = self._normalize_role(message.role)
        return {
            "message_id": message.id,
            "role": role,
            "created_at": self._message_ts(message),
            "content": message.content,
            "user_id": message.user_id,
            "agent_id": message.agent_id,
            "session_id": message.session_id,
            "metadata": message.metadata,
        }

    def _render_reference_turn(
        self,
        *,
        turn_id: str,
        session_id: str,
        message_ids: Sequence[str],
        agent_context_hash: str,
        messages: Sequence[Memory],
        payload: Mapping[str, Any],
        sensitive: bool,
    ) -> str:
        tags = ["conversation", "source", "memorypack"]
        if sensitive:
            tags.append("sensitive-agent-context")
        lines = [
            "---",
            "type: Conversation Turn Source",
            f"title: {turn_id}",
            f"description: Reference turn {turn_id}.",
            f"resource: coremem://turns/{turn_id}",
            f"tags: {_frontmatter_list(tags)}",
            f"timestamp: {_utc_now()}",
            f"turn_id: {turn_id}",
            f"session_id: {session_id}",
            f"message_ids: {_frontmatter_list(message_ids)}",
            f"agent_context_hash: {agent_context_hash}",
            "---",
            "",
            "# Messages",
            "",
        ]
        for message in messages:
            lines.extend([
                f"## {message.id} {self._normalize_role(message.role)}",
                "",
                message.content,
                "",
            ])
        lines.extend([
            "# Canonical Turn Payload",
            "",
            "```json agent_journal-turn",
            json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
            "```",
            "",
        ])
        return "\n".join(lines)

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"references": []}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise AgentJournalError("references/manifest.json must be an object")
        references = data.setdefault("references", [])
        if not isinstance(references, list):
            raise AgentJournalError("references/manifest.json references must be a list")
        return data

    def _append_manifest(
        self,
        path: Path,
        turn_id: str,
        message_ids: Sequence[str],
        agent_context_hash: str,
    ) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()
        references = manifest["references"]
        if not isinstance(references, list):
            raise AgentJournalError("references/manifest.json references must be a list")
        relative = path.relative_to(self.references_dir).as_posix()
        if any(isinstance(item, dict) and item.get("path") == relative for item in references):
            raise AgentJournalError(f"reference already exists in manifest: {relative}")
        references.append({
            "path": relative,
            "sha256": _sha256_file(path),
            "turn_id": turn_id,
            "message_ids": list(message_ids),
            "agent_context_hash": agent_context_hash,
            "created_at": _utc_now(),
        })
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _validate_source(self, source: Mapping[str, Any], label: str) -> list[str]:
        errors: list[str] = []
        turn_id = source.get("source_turn_id")
        message_id = source.get("source_message_id")
        quote = source.get("source_quote")
        if not isinstance(turn_id, str) or not turn_id:
            errors.append(f"{label} source_turn_id is required")
            return errors
        if not _SAFE_REFERENCE_ID_RE.match(turn_id):
            errors.append(f"{label} source_turn_id is invalid")
            return errors
        if not isinstance(message_id, str) or not message_id:
            errors.append(f"{label} source_message_id is required")
            return errors
        if not _SAFE_REFERENCE_ID_RE.match(message_id):
            errors.append(f"{label} source_message_id is invalid")
            return errors
        if not isinstance(quote, str) or not quote:
            errors.append(f"{label} source_quote is required")
            return errors
        turn_path = self.turns_dir / f"{turn_id}.md"
        if not turn_path.exists():
            errors.append(f"{label} source_turn_id does not resolve: {turn_id}")
            return errors
        manifest_errors = self._lint_manifest_entry(turn_path)
        errors.extend(f"{label} {error}" for error in manifest_errors)
        payload, payload_errors = _extract_turn_payload(turn_path.read_text(encoding="utf-8"))
        errors.extend(f"{label} {error}" for error in payload_errors)
        if payload is None:
            return errors
        messages = payload.get("messages")
        if not isinstance(messages, list):
            errors.append(f"{label} canonical payload messages must be a list")
            return errors
        matched = next((msg for msg in messages if isinstance(msg, dict) and msg.get("message_id") == message_id), None)
        if matched is None:
            errors.append(f"{label} source_message_id does not exist: {message_id}")
            return errors
        content = matched.get("content")
        role = matched.get("role")
        evidence_type = source.get("evidence_type")
        if not self._role_supports_evidence(role, evidence_type):
            errors.append(f"{label} evidence_type {evidence_type!r} is not supported by role {role!r}")
        if not isinstance(content, str):
            errors.append(f"{label} cited message content is not a string")
            return errors
        if quote not in content:
            errors.append(f"{label} source_quote is not an exact substring")
        return errors

    def _role_supports_evidence(self, role: object, evidence_type: object) -> bool:
        if role in {"system", "developer"}:
            return False
        if evidence_type == "user_statement":
            return role == "user"
        if evidence_type == "assistant_action":
            return role in {"assistant", "tool_call"}
        if evidence_type == "tool_observation":
            return role == "tool_result"
        return False

    def _lint_manifest_entry(self, path: Path) -> list[str]:
        try:
            manifest = self._load_manifest()
        except (json.JSONDecodeError, AgentJournalError) as exc:
            return [f"manifest cannot be read: {exc}"]
        relative = path.relative_to(self.references_dir).as_posix()
        references = manifest.get("references", [])
        if not isinstance(references, list):
            return ["manifest references must be a list"]
        matches = [item for item in references if isinstance(item, dict) and item.get("path") == relative]
        if len(matches) != 1:
            return [f"manifest entry missing or duplicated for {relative}"]
        expected = matches[0].get("sha256")
        actual = _sha256_file(path)
        if expected != actual:
            return [f"manifest hash mismatch for {relative}"]
        return []

    def _lint_manifest(self) -> list[str]:
        errors: list[str] = []
        try:
            manifest = self._load_manifest()
        except (json.JSONDecodeError, AgentJournalError) as exc:
            return [f"manifest cannot be read: {exc}"]
        references = manifest.get("references", [])
        if not isinstance(references, list):
            return ["manifest references must be a list"]
        seen: set[str] = set()
        for index, item in enumerate(references):
            if not isinstance(item, dict):
                errors.append(f"manifest references[{index}] must be an object")
                continue
            relative = item.get("path")
            if not isinstance(relative, str) or not relative:
                errors.append(f"manifest references[{index}] path is required")
                continue
            if relative in seen:
                errors.append(f"manifest duplicate reference path: {relative}")
            seen.add(relative)
            path = self.references_dir / relative
            if not path.exists():
                errors.append(f"manifest reference missing file: {relative}")
                continue
            expected = item.get("sha256")
            actual = _sha256_file(path)
            if expected != actual:
                errors.append(f"manifest hash mismatch for {relative}")
            errors.extend(self._lint_reference_manifest_consistency(path, item))
            errors.extend(self._lint_reference_turn(path))
        for path in sorted(self.turns_dir.glob("*.md")):
            relative = path.relative_to(self.references_dir).as_posix()
            if relative not in seen:
                errors.append(f"unmanifested reference file: {relative}")
        return errors

    def _lint_reference_manifest_consistency(self, path: Path, item: Mapping[str, Any]) -> list[str]:
        text = path.read_text(encoding="utf-8")
        frontmatter, _ = _parse_frontmatter(text)
        payload, payload_errors = _extract_turn_payload(text)
        if payload_errors or payload is None:
            return []
        errors: list[str] = []
        if item.get("turn_id") != frontmatter.get("turn_id") or item.get("turn_id") != payload.get("turn_id"):
            errors.append(f"manifest turn_id mismatch for {path.relative_to(self.references_dir)}")
        payload_ids = [
            msg.get("message_id")
            for msg in payload.get("messages", [])
            if isinstance(msg, dict) and isinstance(msg.get("message_id"), str)
        ]
        manifest_ids = item.get("message_ids")
        if isinstance(manifest_ids, list) and manifest_ids != payload_ids:
            errors.append(f"manifest message_ids mismatch for {path.relative_to(self.references_dir)}")
        return errors

    def _lint_reference_turn(self, path: Path) -> list[str]:
        text = path.read_text(encoding="utf-8")
        frontmatter, _ = _parse_frontmatter(text)
        errors = []
        for key in ("type", "turn_id", "session_id", "message_ids", "agent_context_hash"):
            if key not in frontmatter:
                errors.append(f"{path.relative_to(self.root)} missing frontmatter field: {key}")
        payload, payload_errors = _extract_turn_payload(text)
        errors.extend(f"{path.relative_to(self.root)} {error}" for error in payload_errors)
        if payload is None:
            return errors
        messages = payload.get("messages")
        if not isinstance(messages, list):
            errors.append(f"{path.relative_to(self.root)} payload messages must be a list")
            return errors
        ids: list[str] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                errors.append(f"{path.relative_to(self.root)} messages[{index}] must be an object")
                continue
            message_id = message.get("message_id")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(message_id, str) or not message_id:
                errors.append(f"{path.relative_to(self.root)} messages[{index}] message_id is required")
            else:
                ids.append(message_id)
            if role not in ALLOWED_ROLES:
                errors.append(f"{path.relative_to(self.root)} messages[{index}] role is invalid")
            tags = _as_str_list(frontmatter.get("tags", []))
            if role in {"system", "developer"} and "sensitive-agent-context" not in tags:
                errors.append(f"{path.relative_to(self.root)} system/developer message requires sensitive-agent-context tag")
            if not isinstance(content, str):
                errors.append(f"{path.relative_to(self.root)} messages[{index}] content must be a string")
        if len(ids) != len(set(ids)):
            errors.append(f"{path.relative_to(self.root)} message ids must be unique")
        return errors

    def _lint_memory_file(self) -> list[str]:
        path = self.root / "MEMORY.md"
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        errors: list[str] = []
        if len(text) > self.boot_budget_chars:
            errors.append("MEMORY.md exceeds configured boot budget")
        if "references/turns/" in text:
            errors.append("MEMORY.md must not include reference turn links or content")
        errors.extend(self._lint_links(path, text))
        return errors

    def _lint_memory_pages(self) -> list[str]:
        errors: list[str] = []
        search_dirs = [d for d in (self.pages_dir, self.daily_dir) if d.exists()]
        if not search_dirs:
            return errors
        page_ids: set[str] = set()
        for path in sorted(p for d in search_dirs for p in d.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            frontmatter, body = _parse_frontmatter(text)
            relative = path.relative_to(self.root)
            if not frontmatter:
                errors.append(f"{relative} missing frontmatter")
                continue
            version_key = "agent_journal_version" if "agent_journal_version" in frontmatter else "agent_memory_version"
            for key in ("type", "page_id", "memory_kind", version_key):
                if key not in frontmatter:
                    errors.append(f"{relative} missing frontmatter field: {key}")
            if frontmatter.get("type") != "AgentJournal Page":
                errors.append(f"{relative} type must be AgentJournal Page")
            page_id = frontmatter.get("page_id")
            if isinstance(page_id, str):
                if page_id in page_ids:
                    errors.append(f"duplicate page_id: {page_id}")
                page_ids.add(page_id)
            if frontmatter.get("memory_kind") not in MEMORY_KINDS:
                errors.append(f"{relative} memory_kind is invalid")
            if "scope" in frontmatter and frontmatter.get("scope") not in SCOPES:
                errors.append(f"{relative} scope is invalid")
            if "status" in frontmatter and frontmatter.get("status") not in STATUSES:
                errors.append(f"{relative} status is invalid")
            if "activation" in frontmatter and frontmatter.get("activation") not in ACTIVATIONS:
                errors.append(f"{relative} activation is invalid")
            if "trust" in frontmatter and frontmatter.get("trust") not in TRUST_VALUES:
                errors.append(f"{relative} trust is invalid")
            if "safe_to_act" in frontmatter and not isinstance(frontmatter.get("safe_to_act"), bool):
                errors.append(f"{relative} safe_to_act must be boolean")
            actual_version = frontmatter.get("agent_journal_version") or frontmatter.get("agent_memory_version")
            if actual_version != PROFILE_VERSION:
                errors.append(f"{relative} agent_journal_version must be {PROFILE_VERSION}")
            if len(re.findall(r"^# Summary$", body, re.MULTILINE)) != 1:
                errors.append(f"{relative} must have exactly one # Summary section")
            errors.extend(self._lint_page_citations(path, text))
            errors.extend(self._lint_links(path, text))
        errors.extend(self._lint_index_links())
        return errors

    def _lint_page_citations(self, path: Path, text: str) -> list[str]:
        relative = path.relative_to(self.root)
        errors: list[str] = []
        has_current_claims = bool(re.search(r"^# Current State\n.*?^- ", text, re.MULTILINE | re.DOTALL))
        has_citations = "# Citations" in text
        if has_current_claims and not has_citations:
            errors.append(f"{relative} has current claims but no # Citations section")
            return errors
        for claim in _extract_citation_claims(text):
            errors.extend(f"{relative} {error}" for error in self.validate_claim(claim))
        if has_citations and not _extract_citation_claims(text):
            errors.append(f"{relative} has # Citations but no parseable reference citations")
        return errors

    def _lint_index_links(self) -> list[str]:
        path = self.root / "index.md"
        if not path.exists():
            return []
        return self._lint_links(path, path.read_text(encoding="utf-8"))

    def _lint_links(self, path: Path, text: str) -> list[str]:
        errors: list[str] = []
        for link in _find_links(text):
            target = link.split("#", 1)[0]
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(self.root.resolve())
            except ValueError:
                errors.append(f"{path.relative_to(self.root)} link escapes bundle: {link}")
                continue
            if not resolved.exists():
                errors.append(f"{path.relative_to(self.root)} broken link: {link}")
        return errors

    def rebuild_embeddings(self) -> None:
        """Rebuild the embedding index for compiled pages."""
        if self._embed_model is None:
            return
        if self._embedding_index is None:
            from coremem.agent_journal.embeddings import EmbeddingIndex
            self._embedding_index = EmbeddingIndex(self.root, model_name=self._embed_model)
        self._embedding_index.refresh(self.pages_dir)

    @property
    def embedding_index(self):
        return self._embedding_index


_IRREGULAR = {
    "bought": "buy", "brought": "bring", "built": "build", "burnt": "burn",
    "came": "come", "did": "do", "drew": "draw", "drove": "drive",
    "fell": "fall", "flew": "fly", "forgot": "forget", "gave": "give",
    "gone": "go", "grew": "grow", "had": "have", "hid": "hide",
    "knew": "know", "laid": "lay", "led": "lead", "left": "leave",
    "lent": "lend", "lost": "lose", "made": "make", "meant": "mean",
    "met": "meet", "paid": "pay", "ran": "run", "rang": "ring",
    "rose": "rise", "said": "say", "sang": "sing", "sank": "sink",
    "sat": "sit", "slept": "sleep", "spoke": "speak", "spent": "spend",
    "stood": "stand", "stole": "steal", "struck": "strike", "swam": "swim",
    "took": "take", "taught": "teach", "tore": "tear", "told": "tell",
    "thought": "think", "threw": "throw", "understood": "understand",
    "woke": "wake", "wore": "wear", "won": "win", "wrote": "write",
    "got": "get", "ate": "eat", "drank": "drink", "drove": "drive",
    "rode": "ride", "saw": "see", "sent": "send", "shook": "shake",
    "shot": "shoot", "showed": "show", "shut": "shut", "sold": "sell",
    "sought": "seek", "sped": "speed", "spun": "spin", "split": "split",
    "spread": "spread", "stuck": "stick", "stung": "sting", "stank": "stink",
    "strode": "stride", "struck": "strike", "strung": "string",
    "swept": "sweep", "swelled": "swell", "swore": "swear", "swung": "swing",
    "tore": "tear", "threw": "throw", "thrust": "thrust", "trod": "tread",
    "understood": "understand", "undertook": "undertake", "upset": "upset",
    "woke": "wake", "waylaid": "waylay", "wept": "weep", "wound": "wind",
    "withdrew": "withdraw", "withheld": "withhold", "withstood": "withstand",
    "woke": "wake", "won": "win", "wound": "wind", "wring": "wring",
    "wrote": "write",
}


_STOPWORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "back", "be", "because", "been", "being",
    "but", "by", "can", "come", "could", "did", "do", "does", "done",
    "down", "each", "end", "even", "few", "for", "from", "further",
    "get", "go", "got", "had", "has", "have", "her", "here", "hers",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "just", "like", "made", "make", "may", "me", "might", "more",
    "most", "much", "must", "my", "no", "nor", "not", "now", "of",
    "on", "only", "or", "other", "our", "out", "over", "own", "put",
    "round", "said", "same", "see", "shall", "she", "should", "show",
    "side", "since", "so", "some", "such", "take", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this",
    "through", "to", "too", "top", "under", "until", "up", "upon", "us",
    "very", "was", "way", "we", "well", "were", "what", "when",
    "where", "which", "while", "who", "why", "will", "with", "would",
    "yes", "yet", "you", "your",
}


def _stem(word: str) -> str:
    word = word.casefold()
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if base.endswith("nn") or base.endswith("tt") or base.endswith("mm"):
            return base[:-1]
        return base
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if base.endswith("i"):
            return base[:-1] + "y"
        if base.endswith("nn") or base.endswith("tt"):
            return base[:-1]
        return base
    if word.endswith("ly"):
        return word[:-2]
    if word.endswith("er") and len(word) > 4:
        return word[:-2]
    if word.endswith("est") and len(word) > 5:
        return word[:-3]
    return word


def _stem_text(text: str) -> str:
    return " ".join(_stem(w) for w in text.split())


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _fuzzy_tf(term: str, words: list[str], max_dist: int = 1) -> int:
    count = words.count(term)
    if max_dist > 0:
        for w in set(words):
            if w != term and len(w) >= 3 and _levenshtein(term, w) <= max_dist:
                count += words.count(w)
    return count


def _bm25(docs, terms, k1=1.5, b=0.75):
    N = len(docs)
    if N == 0 or not terms:
        return [(label, 0.0) for label, _ in docs]
    avgdl = sum(len(d.split()) for _, d in docs) / N
    df = {t: sum(1 for _, d in docs if t in d) for t in terms}
    idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in terms}
    scored = []
    for label, text in docs:
        words = text.split()
        doclen = len(words)
        score = 0.0
        for t in terms:
            if df[t] == 0:
                continue
            tf = _fuzzy_tf(t, words)
            score += idf[t] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doclen / avgdl))
        scored.append((label, score))
    return scored


class AgentJournalSearch:
    """BM25 file-based AgentJournal search with optional cross-encoder re-ranking.

    Searches daily/ directory by default. Falls back to pages/ for backward compat.
    """

    def __init__(self, root: str | Path, embedding_index=None, reranker=None) -> None:
        self.root = Path(root)
        self.pages_dir = self.root / "pages"
        self.daily_dir = self.root / "daily"
        self._embedding_index = embedding_index
        self._reranker = reranker

    def _search_dir(self) -> Path:
        if self.daily_dir.exists():
            return self.daily_dir
        return self.pages_dir

    def search(self, query: str, *, scope: str | None = None, limit: int = 5) -> list[SearchHit]:
        search_dir = self._search_dir()
        if not search_dir.exists():
            return []
        bm25_limit = limit * 4 if self._reranker is not None else limit
        hits = self._bm25_search(query, scope=scope, limit=bm25_limit)
        if self._reranker is not None and len(hits) > limit:
            try:
                return self._reranker.rerank(query, hits, limit=limit)
            except Exception:
                logger.warning("reranker failed, returning BM25 results", exc_info=True)
        return hits[:limit]

    def _bm25_search(self, query: str, *, scope: str | None = None, limit: int = 5) -> list[SearchHit]:
        terms = [_stem(t) for t in re.findall(r"\w+", query.casefold()) if t not in _STOPWORDS]
        if not terms:
            return []
        search_dir = self._search_dir()
        full_docs: list[tuple[Path, str]] = []
        section_docs: list[tuple[tuple[Path, int], str]] = []
        for path in sorted(search_dir.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            frontmatter, body = _parse_frontmatter(text)
            if scope and frontmatter.get("scope") != scope:
                continue
            weighted = "\n".join([
                str(frontmatter.get("description", "")),
                "\n".join(_as_str_list(frontmatter.get("read_when", []))),
                text,
            ]).casefold()
            full_docs.append((path, _stem_text(weighted)))
            sections = _extract_sections(text)
            for i, sec in enumerate(sections):
                section_docs.append(((path, i), _stem_text(sec.casefold())))
        full_scored = dict(_bm25(full_docs, terms))
        section_scored: dict[Path, float] = {}
        if section_docs:
            for (path, _), score in _bm25(section_docs, terms):
                if path not in section_scored or score > section_scored[path]:
                    section_scored[path] = score
        all_paths = set(full_scored) | set(section_scored)
        hits = []
        for p in all_paths:
            score = max(full_scored.get(p, 0.0), section_scored.get(p, 0.0))
            if score > 0:
                hits.append(SearchHit(path=p, score=score, session_id=p.stem))
        hits.sort(key=lambda hit: (-hit.score, hit.path.as_posix()))
        return hits[:limit]


def _extract_claims(body: str) -> list[str]:
    """Extract individual claim texts from # Current State section.

    Deprecated: use _extract_sections() for daily page format.
    """
    match = re.search(r"^# Current State\n(.*?)(?=^# |\Z)", body, re.MULTILINE | re.DOTALL)
    if not match:
        return []
    claims = []
    for line in match.group(1).splitlines():
        line = line.strip()
        if line.startswith("- "):
            claim = re.sub(r"\s*\[[\d,\s]+\]$", "", line[2:]).strip()
            if claim:
                claims.append(claim)
    return claims


def _extract_sections(text: str) -> list[str]:
    """Extract timestamped sections from a daily page.

    Each section is delimited by ## HH:MM - Title. The **Citations:** block
    is excluded from each section (noisy for BM25 scoring).
    """
    sections: list[str] = []
    pattern = re.compile(r"^## (\d{2}:\d{2}) - (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return sections
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end]
        cit_idx = section.find("\n**Citations:**")
        if cit_idx >= 0:
            section = section[:cit_idx]
        sections.append(section.strip())
    return sections
    for hit in hits:
        sid = hit.session_id
        if sid not in grouped or hit.score > grouped[sid][0]:
            grouped[sid] = (hit.score, hit.path)
    top = sorted(grouped.items(), key=lambda x: -x[1][0])[:limit]
    return [SearchHit(path=p, score=s, session_id=sid) for sid, (s, p) in top]
