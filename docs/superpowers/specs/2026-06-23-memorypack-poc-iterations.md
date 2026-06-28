# MemoryPack POC Iterations

2026-06-23

## 1. Goal

Close the remaining gaps in the MemoryPack POC: replace TF-IDF with BM25,
stabilize the LLM compiler (caching, few-shot, resume), test boot memory,
and investigate honest abstention detection.

## 2. BM25 Search

### Purpose

Replace raw TF-IDF with BM25 for all four search paths:

- `AgentMemorySearch.search()` (bundle.py)
- `_search_compiled_pages()` (compiler eval, LLM compiler eval) — two copies
- `_search_reference_messages()` (baseline eval)

### Formula

```
BM25(q, d) = sum over terms t in q of:
    IDF(t) * (tf(t,d) * (k1 + 1)) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

Parameters: k1 = 1.5, b = 0.75.

### Implementation

Add `_bm25()` helper:

```
def _bm25(docs: list[tuple[Any, str]], terms: list[str], k1=1.5, b=0.75) -> list[tuple[Any, float]]:
    N = len(docs)
    avgdl = sum(len(d.split()) for _, d in docs) / max(N, 1)
    df = {t: sum(1 for _, d in docs if t in d) for t in terms}
    idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in terms}
    scored = []
    for label, text in docs:
        tf = {t: text.split().count(t) for t in terms}
        doclen = len(text.split())
        score = 0.0
        for t in terms:
            if df[t] == 0: continue
            score += idf[t] * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * doclen / avgdl))
        scored.append((label, score))
    return scored
```

~20 lines. Goes in each file that needs it (or shared module if warranted).

### Files

- `coremem/memorypack/bundle.py`: `AgentMemorySearch.search()`
- `scripts/eval_memorypack_longmemeval.py`: `_search_reference_messages()`
- `scripts/eval_memorypack_compiler.py`: `_search_compiled_pages()`
- `scripts/eval_memorypack_llm_compiler.py`: `_search_compiled_pages()`

### Results

All four configurations improved:

| Metric | 8-inst baseline | 8-inst heuristic | 20-inst baseline | 20-inst heuristic |
|---|---|---|---|---|
| **TF-IDF recall@5** | 0.542 | 0.583 | 0.553 | 0.482 |
| **BM25 recall@5** | **0.708** | **0.833** | **0.746** | **0.579** |
| **TF-IDF MRR** | 0.542 | 0.604 | 0.614 | 0.447 |
| **BM25 MRR** | **0.792** | **0.854** | **0.886** | **0.553** |

## 3. LLM Compiler Caching + --resume

### Purpose

Make LLM compiler eval deterministic and idempotent. Same session always
produces the same compiled page, so metrics change only when compiler code
changes. Also enables `--resume` for timeout recovery.

### Design

Cache key: `sha256(canonical_messages_json)` — the JSON payload of the
messages array only, NOT the full file content (which includes a changing
`timestamp` frontmatter field). Messages are immutable once written.

Cache value: the validated plan dict. Stored as JSON on disk under
`bundle.root / ".llm_cache" / {turn_id}.json`.

Flow:
1. Before calling the LLM, check `.llm_cache/{turn_id}.json`
2. If exists, validate the cache key hash still matches (integrity check),
   then feed plan to deterministic compiler
3. If not, call LLM, validate, save cache entry, then render

`--resume` flag: when set, skip sessions that already have cache entries.
Check `.llm_cache/` before creating each task, print skipped/total summary.

Cache invalidation: delete `.llm_cache/` entirely when compiler code changes
(system prompt, quote fixer, retry logic).

### Files

- `coremem/memorypack/llm_compiler.py`
- `scripts/eval_memorypack_llm_compiler.py`

## 4. Few-Shot Examples in System Prompt

### Purpose

Improve LLM output quality and reduce retries by showing the model concrete
examples of correct MemoryPack pages.

### Design

Add 2-3 example page plans to the system prompt between the schema
description and the rules list. Each example shows:

- Input: 2-3 conversation messages
- Output: complete page plan (frontmatter fields + summary + claims with evidence)

Demonstrate:
1. Single-source claim (one message, one exact quote)
2. Multi-source derived_summary claim (2+ supporting sources)
3. Edge case: message with quotes/newlines (correct quote extraction)

### Files

- `coremem/memorypack/llm_compiler.py`: `SYSTEM_PROMPT` constant

## 5. Boot Memory Test

### Purpose

Verify the deterministic compiler correctly enforces the boot_worthy guard:
`boot_worthy=true` requires `activation=startup` + `status=active`, and pages
with `boot_worthy=true` appear in MEMORY.md.

This tests the compiler's enforcement, not the LLM's judgment. The LLM's
ability to set boot_worthy correctly is a separate eval concern.

### Design

Use a hand-crafted plan (not an LLM call) fed directly to the deterministic
compiler:

Test 1 (valid boot):
- Plan with `boot_worthy=true`, `activation=startup`, `status=active`
- Apply plan
- Assert page written
- Assert page appears in MEMORY.md

Test 2 (invalid boot — wrong activation):
- Plan with `boot_worthy=true`, `activation=manual`, `status=active`
- Apply plan → expect AgentMemoryError

Test 3 (invalid boot — wrong status):
- Plan with `boot_worthy=true`, `activation=startup`, `status=draft`
- Apply plan → expect AgentMemoryError

Test 4 (non-boot page does not appear in MEMORY.md):
- Plan with `boot_worthy=false`, `activation=manual`, `status=active`
- Apply plan
- Assert page NOT in MEMORY.md

### Files

- `tests/test_memorypack_compiler.py` (4 new tests)

## 6. Abstention Detection

### Status

After BM25, the abstention question `c8090214_abs` (the only abstention
question across both datasets) naturally returns empty retrieval with 0
false positive rate. No threshold hack needed. The current
`abstention_expected` skip-search guard is kept as a safety net but is no
longer necessary for this dataset.

### Monitoring

If new abstention questions are added, verify they also return empty
retrieval with BM25. If not, add a score threshold (top_score < 0.1 → empty).

## 7. Heuristic Compiler Removal

### Purpose

The heuristic compiler (`generate_session_plan` in
`scripts/eval_memorypack_compiler.py`) served its purpose as a regression
test fixture for the deterministic compiler. With BM25 in place and the LLM
compiler working, it creates confusion (different numbers, different code
path). Remove it.

### Removal

- Delete `scripts/eval_memorypack_compiler.py`
- Delete `tests/test_memorypack_eval_compiler.py`
- Remove any references to the heuristic compiler from other scripts/docs
- The deterministic compiler tests (`test_memorypack_compiler.py`) remain
  as they test the compiler itself, not the heuristic plan generator