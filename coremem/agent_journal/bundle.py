"""AgentJournal bundle primitives.

This module implements the deterministic substrate for the AgentJournal:
bundle initialization, linting, exact quote validation, and simple
markdown search. It intentionally does not call an LLM.
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


def _find_links(text: str) -> list[str]:
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    return [link for link in links if not re.match(r"^[a-z][a-z0-9+.-]*:", link)]


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
    def pages_dir(self) -> Path:
        return self.root / "pages"

    @property
    def daily_dir(self) -> Path:
        return self.root / "daily"

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

    def validate_claim(self, claim: Mapping[str, Any], messages: Sequence[Any] | None = None) -> list[str]:
        """Validate a proposed claim's evidence against in-memory messages."""
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
            if not isinstance(sources, list) or len(sources) < 1:
                errors.append("derived_summary claims need at least one supporting_source")
            else:
                for index, source in enumerate(sources):
                    if not isinstance(source, Mapping):
                        errors.append(f"supporting_sources[{index}] must be an object")
                        continue
                    if source.get("evidence_type") not in SOURCE_EVIDENCE_TYPES:
                        errors.append(f"supporting_sources[{index}] evidence_type is invalid")
                    errors.extend(self._validate_source(source, f"supporting_sources[{index}]", None))
            return errors
        return self._validate_source(claim, "claim", messages)

    def lint(self) -> list[str]:
        """Return deterministic AgentJournal lint errors."""
        errors: list[str] = []
        for relative in ("MEMORY.md", "index.md", "log.md", "SCHEMA.md"):
            if not (self.root / relative).exists():
                errors.append(f"missing required file: {relative}")
        errors.extend(self._lint_memory_file())
        errors.extend(self._lint_memory_pages())
        return errors

    def _write_if_missing(self, relative: str, content: str) -> None:
        path = self.root / relative
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    def _validate_source(self, source: Mapping[str, Any], label: str, messages: Sequence[Any] | None = None) -> list[str]:
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
            if not isinstance(quote, str) or not quote:
                return errors
        if message_id and not _SAFE_REFERENCE_ID_RE.match(message_id):
            errors.append(f"{label} source_message_id is invalid")
            return errors
        if not isinstance(quote, str) or not quote:
            return errors
        if messages is not None and message_id:
            matched = next((msg for msg in messages if self._message_value(msg, "message_id", "id") == message_id), None)
            if matched is None:
                errors.append(f"{label} source_message_id does not exist: {message_id}")
                return errors
            role = self._message_value(matched, "role")
            content = self._message_value(matched, "content")
            role = "tool_result" if role == "tool" else role
            evidence_type = source.get("evidence_type")
            if not self._role_supports_evidence(role, evidence_type):
                errors.append(f"{label} evidence_type {evidence_type!r} is not supported by role {role!r}")
            if not isinstance(content, str):
                errors.append(f"{label} cited message content is not a string")
                return errors
            if quote not in content:
                errors.append(f"{label} source_quote is not an exact substring")
        return errors

    def _message_value(self, message: Any, *keys: str) -> Any:
        if isinstance(message, Mapping):
            for key in keys:
                if key in message:
                    return message[key]
            return None
        for key in keys:
            if hasattr(message, key):
                return getattr(message, key)
        return None

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

    def _lint_memory_file(self) -> list[str]:
        path = self.root / "MEMORY.md"
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8")
        errors: list[str] = []
        if len(text) > self.boot_budget_chars:
            errors.append("MEMORY.md exceeds configured boot budget")
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
            if self._is_daily_journal_file(path, frontmatter):
                errors.extend(self._lint_daily_journal(path, frontmatter, body))
                errors.extend(self._lint_links(path, text))
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
            errors.extend(self._lint_links(path, text))
        errors.extend(self._lint_index_links())
        return errors

    def _is_daily_journal_file(self, path: Path, frontmatter: Mapping[str, object]) -> bool:
        try:
            path.relative_to(self.daily_dir)
        except ValueError:
            return False
        return "date" in frontmatter and "page_id" not in frontmatter

    def _lint_daily_journal(self, path: Path, frontmatter: Mapping[str, object], body: str) -> list[str]:
        relative = path.relative_to(self.root)
        errors: list[str] = []
        date = frontmatter.get("date")
        if not isinstance(date, str) or not date:
            errors.append(f"{relative} date is required")
        version = frontmatter.get("agent_journal_version")
        if version != PROFILE_VERSION:
            errors.append(f"{relative} agent_journal_version must be {PROFILE_VERSION}")
        if isinstance(date, str) and f"# {date}" not in body:
            errors.append(f"{relative} must have # {date} heading")
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
