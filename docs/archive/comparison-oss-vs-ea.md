# CoreMem: OSS vs EA Comparison

2026-05-31

## Verdict

**OSS CoreMem is the canonical, more advanced version.** EA's `src/coremem/` is an older snapshot that was never updated after extraction. All new features (metadata filters, search_enhanced, reranking, export/delete, etc.) exist only in OSS.

---

## Differences

### OSS has — EA does not

| What | Details |
|------|---------|
| `Memory.metadata` dict | Arbitrary key-value pairs for filtering. EA still has old `workspace_id` field. |
| `SearchQuery.filters` dict | Flat equality filters. EA still has old `wing`/`room` fields. |
| `search_enhanced()` | Multi-query expansion + cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`). |
| `export()` / `export_all()` | Paginated export with filters. |
| `delete()` / `count()` / `clear()` | Lifecycle management on `MemoryCore`. |
| `StoreBackend.list()` / `delete()` | Abstract methods on the backend ABC. |
| `warmup()` | Pre-downloads cross-encoder model. |
| `Memory.embedding` param on ingest | Pre-computed embedding support. |
| `__future__ import annotations` | Best practice in `chroma.py`, `base.py`. |
| `_CHROMA_INTERNAL_META_KEYS` | ChromaDB metadata key filtering. |
| `query.py` module | Multi-query expansion (regex + optional LLM). |
| `rerank.py` module | Cross-encoder reranking pipeline. |
| Tests (`tests/`) | 5 test files: core, heuristics, layers, backend chroma, backend hybrid. |
| Benchmarks (`benchmarks/`) | LongMemEval adapter + eval script. |
| `pyproject.toml` with `[build-system]` | Hatchling build, classifiers, authors, keywords. |
| Ruff + mypy config | Linter and type checker settings. |
| `hybriddb>=0.3.0` | EA pins `0.1.0`. |
| `dev` extras | `pytest`, `pytest-asyncio`. |
| Duplicate `ingest_batch` removed | EA still has the double definition in `chroma.py`. |

### EA has — OSS does not

Only one meaningful difference in `hybrid.py`:

| File | What | Detail |
|------|------|--------|
| `hybrid.py` | `_ensure_tables()` schema | `"id": "TEXT PRIMARY KEY"` + `"metadata": "TEXT"` instead of no id column + `"metadata": "JSON"` |
| `hybrid.py` | `ingest_batch()` | Client-side UUID generation (`str(uuid.uuid4())[:12]`) + explicit `"id": mid` pass to HybridDB |
| `hybrid.py` | Import | `try: src.sdk.hybrid_db → hybriddb` dual fallback |

This is the **schema change** referenced as the last change in EA's CoreMem. It needs to be ported to OSS.

---

## Sync Checklist

- [ ] Port `hybrid.py` schema (`"id": "TEXT PRIMARY KEY"`, `"metadata": "TEXT"`) to OSS
- [ ] Port client-side UUID generation to OSS `ingest_batch()`
- [ ] Keep everything else as-is (OSS is the source of truth)

---

## Why EA's CoreMem Exists

EA's `src/coremem/` is a vendored copy used by the EA agent system. It was extracted as the OSS repo but never received updates after extraction. EA has since made one schema change (text UUID keys) that needs to flow back to OSS.
