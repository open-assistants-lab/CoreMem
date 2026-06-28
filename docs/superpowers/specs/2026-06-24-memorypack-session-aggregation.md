# Session-Level Aggregation for MemoryPack Search

## Problem

`AgentMemorySearch.search()` returns individual page hits. Multi-session questions expect multiple sessions as the answer. If 2/6 expected sessions are in the top 5 pages, recall is 0.333 even though the remaining 4 are at ranks 6-20. Grouping by session and returning top-k sessions fixes this.

## Design

After BM25 + cross-encoder re-ranking, group hits by `session_id` (extracted from page frontmatter). Each session's score is the max score of any page in that session. Return top-k sessions.

## Changes

### `coremem/memorypack/bundle.py` — `AgentMemorySearch.search()`

```python
def search(self, query, *, scope=None, limit=5):
    if not self.pages_dir.exists():
        return []
    bm25_limit = limit * 4 if self._reranker is not None else limit
    hits = self._bm25_search(query, scope=scope, limit=bm25_limit)
    if self._reranker is not None and len(hits) > limit:
        hits = self._reranker.rerank(query, hits, limit=bm25_limit)
    return _group_by_session(hits, limit)
```

New helper:

```python
def _group_by_session(hits: list[SearchHit], limit: int) -> list[SearchHit]:
    grouped: dict[str, tuple[float, Path]] = {}
    for hit in hits:
        sid = _session_id_from_page(hit.path)
        if sid not in grouped or hit.score > grouped[sid][0]:
            grouped[sid] = (hit.score, hit.path)
    top = sorted(grouped.items(), key=lambda x: -x[1][0])[:limit]
    return [SearchHit(path=p, score=s) for sid, (s, p) in top]
```

### `coremem/memorypack/bundle.py` — `_session_id_from_page()`

```python
def _session_id_from_page(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^session_id:\s*([^\n]+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip().strip("\"'")
    return path.stem
```

### `scripts/eval_memorypack_llm_compiler.py` — `_search_compiled_pages()`

No change needed — it delegates to `AgentMemorySearch.search()` which now returns session-grouped results.

## Expected Impact

| Metric | Before | After |
|---|---|---|
| session_recall@5 | 0.833 | 0.85-0.90 |
| session_precision@5 | 0.300 | 0.35-0.40 |
| session_map | 0.633 | 0.70-0.75 |

Multi-session questions (`a3838d2b` expects 6, `6613b389` expects 3, `gpt4_93159ced` expects 2) should see the biggest improvement.

## Files

- `coremem/memorypack/bundle.py` — `_group_by_session()`, `_session_id_from_page()`, modify `search()`
