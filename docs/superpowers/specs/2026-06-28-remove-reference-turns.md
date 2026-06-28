# Remove References/Turns from AgentJournal

2026-06-28

## 1. Problem

The `references/turns/` directory and `references/manifest.json` are never
written by production code — they're only created by eval scripts and tests.
But the deterministic compiler's `validate_claim()` tries to read from them,
which means `compile_turn()` would fail on any MemoryCore that hasn't had
reference turns pre-written.

This creates two issues:
- **Dead code burden**: ~400 lines of bundle.py, the entire `references/` dir,
  manifest logic, and reference-turn linting are never exercised in real use.
- **Broken developer experience**: the first real `compile_turn()` call hits
  `validate_claim()` → `FileNotFoundError` (via `turn_path.read_text()`).

## 2. Solution

### 2A. Remove `references/turns/` + `references/manifest.json`

Delete the on-disk reference turn system entirely:

- **`references/` directory** — no longer created or expected by `initialize()`
  or `lint()`
- **`references/turns/<id>.md`** — immutable reference turn files removed
- **`references/manifest.json`** — SHA256-verified manifest removed
- **`write_reference_turn()`** — public API removed (was unused by production)
- **`_render_reference_turn()`** — unused outside `write_reference_turn()`
- **`_append_manifest()`** — unused outside `write_reference_turn()`
- **`_load_manifest()`** — unused outside manifest methods
- **All manifest/reference linting** — `_lint_manifest()`,
  `_lint_manifest_entry()`, `_lint_reference_turn()`,
  `_lint_reference_manifest_consistency()` removed
- **Properties** — `references_dir`, `turns_dir`, `manifest_path` removed
- **`_extract_turn_payload()`** — canonical payload parser removed
- **`_normalize_role()`, `_message_ts()`, `_message_sort_key()`,
  `_message_payload()`** — only used by `write_reference_turn()`, removed

### 2B. Replace `validate_claim()` with in-memory validation

Currently `validate_claim()` reads a reference turn file from disk:

```python
turn_path = self.turns_dir / f"{turn_id}.md"
if not turn_path.exists():
    errors.append(...)
    return errors
payload, _ = _extract_turn_payload(turn_path.read_text())
matched = next(msg for msg in payload["messages"]
               if msg["message_id"] == message_id)
if quote not in matched["content"]:
    errors.append("source_quote is not an exact substring")
```

Replace with in-memory validation against messages passed from the caller:

```python
def validate_claim(
    claim: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],  # NEW: messages from the current turn
) -> list[str]:
    # Same checks for evidence_type, required fields
    # BUT check against `messages` instead of reading a file
    source_turn_id = claim.get("source_turn_id")
    source_message_id = claim.get("source_message_id")
    source_quote = claim.get("source_quote")
    # 1. Validate message_id exists in provided messages
    # 2. Validate role matches evidence_type
    # 3. Validate quote is exact substring of content
    # (No file I/O)
```

The `AgentJournalCompiler` is always called with the turn's messages —
pass them to `validate_claim()` instead of reading from disk.

### 2C. Simplify `compile_section()` citations

The `compile_section()` method already renders citations as:

```
**Citations:**
[1] msg_abc123 (user_statement): "quoted text"
```

No file links. The `_citation_link()` method (which generated relative links
to `references/turns/<id>.md`) is removed. Citations are just message IDs.

### 2D. Simplify lint

Remove:
- `_lint_page_citations()` — validates citations against reference turn
  files on disk. No longer needed since citations are just message IDs.
- `_lint_memory_file()` check for `"references/turns/"` in MEMORY.md content
  (no longer relevant)
- `_extract_citation_claims()` — regex parser for old citation link format

Keep:
- `_lint_memory_pages()` for frontmatter validation
- `_lint_memory_file()` for boot budget and link validation
- `_lint_links()` for link integrity
- `_lint_index_links()` for index link integrity

### 2E. Remove `_lint_page_citations()` dependencies

The methods that solely exist for `_lint_page_citations()`:
- `validate_claim()` — kept but simplified (see 2B) since it's also used by
  the compiler
- `_extract_citation_claims()` — removed (only used by lint)
- `_lint_page_citations()` — removed

Keep (used by `validate_claim()` or frontmatter lint):
- `_role_supports_evidence()` — used by `validate_claim()`
- `EVIDENCE_TYPES`, `SOURCE_EVIDENCE_TYPES` — used by `validate_claim()`
- `TRUST_VALUES`, `MEMORY_KINDS`, `SCOPES`, `STATUSES`, `ACTIVATIONS` — used
  by frontmatter lint

## 3. Files Changed

### 3.1 `coremem/agent_journal/bundle.py`

Remove:
- `write_reference_turn()` method
- `_render_reference_turn()`, `_append_manifest()`, `_load_manifest()`
- `_normalize_role()`, `_message_ts()`, `_message_sort_key()`, `_message_payload()`
- `_validate_source()`
- `_lint_manifest()`, `_lint_manifest_entry()`, `_lint_reference_turn()`,
  `_lint_reference_manifest_consistency()`
- `_lint_page_citations()`, `_extract_citation_claims()`
- `_lint_memory_file()` check for `"references/turns/"` string
- `references_dir`, `turns_dir`, `manifest_path` properties
- `_extract_turn_payload()`

Simplify:
- `initialize()` — no longer creates references directory structure
- `lint()` — remove `_lint_manifest()` and `_lint_page_citations()` calls
- `validate_claim()` — accept `messages` parameter, validate in-memory
  instead of reading from disk

### 3.2 `coremem/agent_journal/compiler.py`

Remove:
- `_compile_evidence()` — remove `validate_claim()` call within it
  (the rest of the method — citation extraction and numbering — stays)
- `_citation_link()` — remove (generated relative links to turn files)
- `_render_page()` — remove citation link generation (`_citation_link` call)

Simplify:
- `_compile_current_state()` — pass messages through to `_compile_evidence()`
  so `validate_claim()` can validate against in-memory messages instead of disk

### 3.3 `coremem/agent_journal/__init__.py`

- Remove `validate_claim` from exports if no longer used (it becomes a
  private method)
- Keep `AgentJournalCompiler`, `AgentJournalCompileResult`,
  `compile_journal_plan`

### 3.4 `coremem/core.py`

- No changes needed (core.py doesn't call any references/turns APIs directly)

### 3.5 `coremem/agent_journal/llm_compiler.py`

- No changes needed (calls `compiler.compile_section()` which handles evidence)

### 3.6 Eval scripts

- Remove `write_reference_turn()` calls — eval scripts no longer write
  reference turns
- Update to use the new `validate_claim()` signature if they call it
- `scripts/eval_agent_journal_internal.py`
- `scripts/eval_agent_journal_longmemeval.py`
- `scripts/eval_agent_journal_llm_compiler.py`
- `scripts/save_stage4_output.py`

### 3.7 Tests

- Remove `test_write_reference_turn_*` tests
- Remove `test_validate_claim_*` tests (or rewrite for in-memory)
- Remove citation-related tests (`test_lint_validates_page_citations`)
- Update any tests that create reference turn files
- `tests/test_agent_journal.py`
- `tests/test_agent_journal_compiler.py`

## 4. Migration Order

1. Update `validate_claim()` to accept `messages` and validate in memory
2. Remove all reference turn / manifest / citation-lint methods from bundle.py
3. Remove `_compile_evidence()` validate_claim call from compiler.py
4. Remove `_citation_link()` and related from compiler.py
5. Remove `references/` creation from `initialize()`
6. Update `lint()` to remove manifest and citation checks
7. Update eval scripts to not write reference turns
8. Update tests
9. Run tests

## 5. Verification

- All tests pass
- `compile_turn()` works without any reference turn files on disk
- `compile_turn()` still validates evidence quotes (against in-memory messages)
- `lint()` doesn't flag missing `references/` or manifest
- No references to `references/`, `turns/`, or `manifest.json` in
  `coremem/agent_journal/`

## 6. Journal directory after removal

```
agent_journal/
├── daily/              ← compiled daily journal pages
├── pages/              ← compiled memory pages (deterministic compiler)
├── MEMORY.md           ← boot-worthy context
├── index.md            ← navigation index
├── log.md              ← change log
└── SCHEMA.md           ← schema version
```
