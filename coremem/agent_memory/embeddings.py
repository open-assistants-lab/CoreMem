"""Embedding index for MemoryPack pages using sentence-transformers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_EMBED_DIR = ".embeddings"
_PAGES_NPY = "pages.npy"
_PAGES_IDS = "pages_ids.json"
_PAGES_HASHES = "pages_hashes.json"
_QUERIES_NPY = "queries.npy"
_QUERIES_HASHES = "queries_hashes.json"
_BATCH_SIZE = 32


class EmbeddingIndex:
    """Lazy-loaded embedding index for MemoryPack pages.

    Stores embeddings as a numpy array on disk under ``.embeddings/``.
    Change detection via sha256 of page content.
    Query embedding cache is in-memory only.
    """

    def __init__(self, root: str | Path, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.root = Path(root)
        self.model_name = model_name
        self._model: Any = None
        self._embeddings: np.ndarray | None = None
        self._page_ids: list[str] = []
        self._page_hashes: dict[str, str] = {}
        self._query_cache: dict[str, np.ndarray] = {}

    @property
    def _dir(self) -> Path:
        return self.root / _EMBED_DIR

    @property
    def _npy_path(self) -> Path:
        return self._dir / _PAGES_NPY

    @property
    def _ids_path(self) -> Path:
        return self._dir / _PAGES_IDS

    @property
    def _hashes_path(self) -> Path:
        return self._dir / _PAGES_HASHES

    @property
    def _queries_npy_path(self) -> Path:
        return self._dir / _QUERIES_NPY

    @property
    def _queries_hashes_path(self) -> Path:
        return self._dir / _QUERIES_HASHES

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _truncate(self, text: str) -> str:
        model = self._load_model()
        tokens = model.tokenizer.encode(text, truncation=True, max_length=model.max_seq_length)
        return model.tokenizer.decode(tokens, skip_special_tokens=True)

    def _page_text(self, path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        frontmatter, body = _parse_frontmatter(text)
        summary = _summary(body)
        description = str(frontmatter.get("description", ""))
        return f"{description}\n{summary}\n{body}"

    def _page_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def build(self, pages_dir: Path) -> None:
        """Embed all pages and write to disk."""
        paths = sorted(pages_dir.rglob("*.md"))
        if not paths:
            self._embeddings = np.empty((0, 384), dtype=np.float32)
            self._page_ids = []
            self._page_hashes = {}
            self._save()
            return
        texts = [self._truncate(self._page_text(p)) for p in paths]
        model = self._load_model()
        self._embeddings = model.encode(texts, batch_size=_BATCH_SIZE, convert_to_numpy=True)
        self._page_ids = [p.stem for p in paths]
        self._page_hashes = {p.stem: self._page_hash(p) for p in paths}
        self._save()

    def refresh(self, pages_dir: Path) -> None:
        """Re-embed only changed pages, remove deleted pages."""
        if not self._dir.exists():
            self.build(pages_dir)
            return
        self._load()
        current_paths = {p.stem: p for p in pages_dir.rglob("*.md")}
        current_ids = set(current_paths.keys())
        stored_ids = set(self._page_ids)

        deleted = stored_ids - current_ids
        added_or_changed = {
            pid for pid in current_ids
            if pid not in stored_ids or self._page_hash(current_paths[pid]) != self._page_hashes.get(pid)
        }

        if not deleted and not added_or_changed:
            return

        if deleted:
            keep = [i for i, pid in enumerate(self._page_ids) if pid not in deleted]
            self._embeddings = self._embeddings[keep]
            self._page_ids = [self._page_ids[i] for i in keep]
            self._page_hashes = {pid: h for pid, h in self._page_hashes.items() if pid not in deleted}

        if added_or_changed:
            new_paths = [current_paths[pid] for pid in sorted(added_or_changed)]
            texts = [self._truncate(self._page_text(p)) for p in new_paths]
            model = self._load_model()
            new_emb = model.encode(texts, batch_size=_BATCH_SIZE, convert_to_numpy=True)
            new_ids = [p.stem for p in new_paths]
            new_hashes = {p.stem: self._page_hash(p) for p in new_paths}
            self._embeddings = np.vstack([self._embeddings, new_emb]) if self._embeddings.size else new_emb
            self._page_ids.extend(new_ids)
            self._page_hashes.update(new_hashes)

        self._save()

    def search(self, query: str, limit: int = 5) -> list[tuple[str, float]]:
        """Embed query, cosine sim against all pages, return top-k (page_id, score)."""
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            self._load()
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            return []
        q_emb = self._embed_query(query)
        sims = np.dot(self._embeddings, q_emb)
        top_indices = np.argsort(-sims)[:limit]
        return [(self._page_ids[i], float(sims[i])) for i in top_indices if sims[i] > 0]

    def _embed_query(self, query: str) -> np.ndarray:
        q_hash = hashlib.sha256(query.encode()).hexdigest()
        if q_hash in self._query_cache:
            return self._query_cache[q_hash]
        model = self._load_model()
        emb = model.encode([query], convert_to_numpy=True)[0]
        self._query_cache[q_hash] = emb
        return emb

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self._npy_path), self._embeddings)
        self._ids_path.write_text(json.dumps(self._page_ids, indent=2), encoding="utf-8")
        self._hashes_path.write_text(json.dumps(self._page_hashes, indent=2, sort_keys=True), encoding="utf-8")

    def _load(self) -> None:
        if not self._npy_path.exists():
            self._embeddings = np.empty((0, 384), dtype=np.float32)
            self._page_ids = []
            self._page_hashes = {}
            return
        self._embeddings = np.load(str(self._npy_path))
        self._page_ids = json.loads(self._ids_path.read_text(encoding="utf-8"))
        self._page_hashes = json.loads(self._hashes_path.read_text(encoding="utf-8"))


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
                existing.append(line[4:].strip().strip("\"'"))
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
        return [item.strip().strip("\"'") for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def _summary(body: str) -> str:
    match = re.search(r"^# Summary\n(.*?)(?=^# |\Z)", body, re.MULTILINE | re.DOTALL)
    return match.group(1) if match else ""
