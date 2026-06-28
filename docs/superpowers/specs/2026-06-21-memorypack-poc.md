# MemoryPack POC

2026-06-21

## 1. Goal

Build the smallest useful proof-of-concept for a CoreMem memory pipeline that
uses `MemoryPack`, a markdown-native memory bundle format inspired by LLM Wiki,
OKF, Agent Skills, OpenClaw, Hermes, and CoreMem's grounded-observation model:

```text
conversation turn/message
  -> immutable reference source
  -> LLM-maintained MemoryPack pages
  -> MemoryPack-first search/retrieval
```

This POC intentionally skips the existing observation layer at first. The goal is
to test whether compiled markdown memory can beat raw message retrieval on
long-horizon recall, token efficiency, citation quality, and human
inspectability.

If the direct MemoryPack pipeline works, the next experiment is the hybrid
version:

```text
conversation turn/message
  -> source-grounded observations
  -> LLM-maintained MemoryPack pages
  -> MemoryPack-first search/retrieval
```

## 2. Thesis

CoreMem currently stores and searches messages, observations, and reflections.
That is strong for retrieval, but the agent still has to synthesize scattered
facts at query time.

The LLM Wiki pattern moves synthesis to ingest time. Every new source updates a
persistent, interlinked wiki. Later queries read the compiled state instead of
rediscovering it from raw messages.

MemoryPack gives this compiled memory layer a memory-native markdown format:

- Markdown files with YAML frontmatter.
- Required `type` and `memory_kind` fields for memory pages.
- `description` and `read_when` fields for progressive disclosure.
- `index.md` for progressive disclosure.
- `MEMORY.md` as a compact boot index.
- `log.md` for chronological history.
- Markdown links for relationships.
- `# Citations` for exact-quote evidence.

## 3. References Adopted

### LLM Wiki

Adopt these mechanics from LLM Wiki:

- Raw sources are immutable.
- Compiled memory is a persistent, compounding artifact.
- The LLM owns memory maintenance: summaries, entity pages, cross-links,
  contradictions, stale claims, and filing good answers back.
- The schema file tells the agent how to maintain memory.
- Queries search/read compiled memory first, then fall back to raw sources.
- `index.md` is content-oriented navigation.
- `log.md` is chronological, append-only history.
- Lint passes periodically check contradictions, orphan pages, missing links,
  missing concepts, and stale claims.

### OKF

Adopt these ideas, but do not treat raw OKF as the complete memory format:

- A knowledge bundle is a directory tree of markdown files.
- Memory pages are documents with YAML frontmatter.
- Every memory page has a non-empty `type` field.
- `index.md` and `log.md` are reserved filenames.
- Unknown frontmatter fields are allowed and preserved.
- Links are ordinary markdown links.
- Broken links are tolerated by consumers but flagged by lint.
- Claims sourced from reference turns should be listed under `# Citations`.

MemoryPack is a memory-native profile, not raw OKF. It adds reserved files,
reference-turn semantics, exact quote validation, memory lifecycle fields, and
agent-context safety rules that OKF does not define.

### Agent Skills

Adopt the format lessons, not the skill catalog:

- Frontmatter `description` is the routing surface.
- Compact top-level files enable progressive disclosure.
- Detailed evidence and tooling live in referenced files.
- Validators and scripts should enforce structure instead of relying only on
  prose instructions.

### OpenClaw and Hermes

Adopt these memory-system lessons:

- A compact boot memory file is useful at session start.
- Daily/session notes and compiled durable memory should be separate.
- Memory should be action-sensitive: source, trust, status, and safe-to-act
  semantics matter.
- Prompt context should stay stable during a session unless explicitly refreshed.

## 4. Non-Goals

- Do not replace `MemoryCore`, `HybridDB`, observations, or reflections.
- Do not require Obsidian or any external wiki tool.
- Do not build vector search in the first POC.
- Do not compile every message into a new memory page.
- Do not persist full system/developer prompts by default.
- Do not let user/tool content override the memory compiler's instructions.
- Do not load the full MemoryPack into the prompt by default.

## 5. Directory Layout

The POC uses a configurable local directory, defaulting to a path inside the
CoreMem store. Example:

```text
memorypack/
  MEMORY.md
  index.md
  log.md
  SCHEMA.md
  agent_context/
    manifest.json
  pages/
    users/
      default.md
      preferences.md
    projects/
      coremem.md
    decisions/
      memory-architecture.md
    open-questions.md
  references/
    manifest.json
    turns/
      turn_20260621_000001.md
  schemas/
    memorypack-page.schema.json
```

`memorypack/` is the MemoryPack bundle root.

`MEMORY.md` is the compact boot index. It should be small enough to load at
session start and should point to deeper pages rather than duplicating them.

`index.md` is the page catalog and routing surface for progressive disclosure.

`log.md` is append-only chronological memory history.

`SCHEMA.md` is the MemoryPack compiler schema: it tells the memory compiler how
to route, write, cite, link, and lint pages. It is read-only to ordinary ingest.

`pages/` contains compiled durable memory pages. Every `.md` file under
`pages/` must be a MemoryPack page with YAML frontmatter and a non-empty `type`
and `memory_kind`.

`references/` contains immutable provenance sources. Reference files are
human-readable markdown with a canonical JSON payload. Normal retrieval treats
them as fallback evidence rather than compiled memory.

Every `.md` file under `references/turns/` must have frontmatter with `type`,
`turn_id`, `session_id`, `message_ids`, and `agent_context_hash`, plus a valid
canonical JSON payload.

`schemas/` contains machine-readable JSON schemas for validators.

`agent_context/manifest.json` records the instruction context that produced
memory updates without persisting full system/developer prompt text.

## 6. Reference Source Unit

The preferred source unit is a conversation turn, not a single isolated message.
A turn can include:

- User message.
- Assistant response.
- Tool calls.
- Tool outputs.
- Timestamps.
- Session/workspace metadata.
- Agent context manifest hash.

For early tests, a single message can be wrapped as a one-message turn.

### Reference Turn Example

Reference turn files are both human-readable MemoryPack reference documents and
machine-readable source records. The validator reads the canonical JSON payload,
not the surrounding prose. This avoids ambiguous parsing when tool output
contains markdown headings, code fences, repeated strings, or text that looks
like citations.

````markdown
---
type: Conversation Turn Source
title: Turn 20260621 000001
description: User proposes direct message-to-MemoryPack POC.
resource: coremem://turns/turn_20260621_000001
tags: [conversation, source, memory-poc]
timestamp: 2026-06-21T00:00:00Z
turn_id: turn_20260621_000001
session_id: session_001
message_ids: [msg_001]
agent_context_hash: sha256:abc123
---

# Messages

## msg_001 user

message -> wiki (okf) -> search, as POC? to be even simplier?

# Canonical Turn Payload

```json agent_memory-turn
{
  "turn_id": "turn_20260621_000001",
  "session_id": "session_001",
  "started_at": "2026-06-21T00:00:00Z",
  "ended_at": "2026-06-21T00:00:05Z",
  "agent_context_hash": "sha256:abc123",
  "messages": [
    {
      "message_id": "msg_001",
      "role": "user",
      "created_at": "2026-06-21T00:00:00Z",
      "content": "message -> wiki (okf) -> search, as POC? to be even simplier?"
    }
  ],
  "metadata": {
    "repo": "CoreMem",
    "workspace_kind": "git-worktree"
  }
}
```
````

Canonical payload rules:

- The payload must be valid JSON inside exactly one fenced code block whose info
  string is exactly `json agent_memory-turn`.
- `messages[].message_id` values must be unique within the turn.
- `messages[].role` must be one of `system`, `developer`, `user`, `assistant`,
  `tool_call`, or `tool_result`.
- `messages[].content` is the only text used for `source_quote` substring
  validation.
- Tool metadata such as `tool_name`, `tool_input`, and `tool_call_id` may be
  included as additional JSON fields.
- Full `system` and `developer` prompt messages are excluded by default. If a
  local-only debug mode persists them, the reference must be tagged
  `sensitive-agent-context` and excluded from normal search.
- The human-readable `# Messages` section is for review only and must not be
  used as the validation source of truth.

Reference turn files are reference documents inside the MemoryPack bundle. They
may be searched as fallback evidence, but normal retrieval reads compiled memory
pages first.

### Reference Manifest

`references/manifest.json` records every immutable reference file:

This non-markdown manifest is a POC tooling sidecar. MemoryPack readers can
ignore it when they only need human-readable memory, but validators use it to
detect accidental reference mutation.

```json
{
  "references": [
    {
      "path": "turns/turn_20260621_000001.md",
      "sha256": "sha256:abc123",
      "turn_id": "turn_20260621_000001",
      "message_ids": ["msg_001"],
      "agent_context_hash": "sha256:abc123",
      "created_at": "2026-06-21T00:00:00Z"
    }
  ]
}
```

Rules:

- New reference files are append-only.
- Existing reference files must not be rewritten by ingest.
- Lint recomputes each file hash and compares it to the manifest.
- The manifest itself is append-only except for adding new references.

POC trust model:

- The manifest protects against accidental mutation and ordinary compiler bugs.
- It does not protect against an adversary or bug that rewrites both a reference
  file and the manifest consistently.
- Stronger tamper evidence should use git commits, git object IDs, an external
  append-only log, or a hash chain stored outside the writable MemoryPack
  directory.
- The POC should record git commit/worktree metadata when available, but should
  not require git for local experiments.

Optional git anchoring:

- If `memorypack/` lives inside a git repository, record the current branch,
  commit, and worktree path in `references/manifest.json` when available.
- After committing a MemoryPack update, the git commit becomes the tamper-evident
  boundary for that version of the bundle.
- The POC must not pretend manifest hashes alone provide adversarial integrity.

## 7. Agent System Prompt Handling

The agent's system/developer prompt matters because it defines the operating
policy for the memory compiler. It must not be treated as ordinary user memory.

### Role Semantics

| Role | Memory meaning | Can support durable user/project claims? |
|------|----------------|------------------------------------------|
| `system` | Highest-priority runtime instructions and safety policy | No |
| `developer` | Agent/tool/repo operating instructions | No |
| `user` | User intent, preferences, corrections, decisions | Yes |
| `assistant` | Agent decisions, summaries, actions taken | Sometimes, labeled as agent action |
| `tool_call` | Operation attempted by the agent | Sometimes, labeled as workflow evidence |
| `tool_result` | Observed output from file/system/API/web | Yes, labeled as tool observation |

System and developer messages can influence how the compiler behaves, but they
must not become facts like "the user prefers X" or "the project uses Y".

### Agent Context Manifest

Instead of persisting the full system/developer prompt, store a safe manifest:

```json
{
  "agent_context_hash": "sha256:abc123",
  "agent_name": "opencode",
  "model": "openai/gpt-5.5",
  "schema_version": "memorypack-poc-0.1",
  "tool_policy_summary": "References immutable; MemoryPack writable; no uncited durable claims.",
  "persist_system_prompts": false,
  "created_at": "2026-06-21T00:00:00Z"
}
```

The hash lets us know which instruction context produced a memory update without
leaking the full prompt.

`agent_context_hash` is computed over canonical JSON with sorted keys:

```json
{
  "agent_name": "opencode",
  "model": "openai/gpt-5.5",
  "schema_version": "memorypack-poc-0.1",
  "schema_sha256": "sha256:def456",
  "system_prompt_sha256": "sha256:...",
  "developer_prompt_sha256": "sha256:...",
  "tool_policy_version": "memorypack-poc-0.1",
  "persist_system_prompts": false
}
```

The system/developer prompt hashes may be recorded when the host can access
those prompts. The prompt text itself remains unpersisted by default. If a host
cannot expose prompt text for hashing, set the corresponding hash field to
`unavailable` and include that fact in the manifest.

### System Prompt Change Behavior

When the agent context hash changes:

1. Append a `log.md` entry noting the new hash and schema version.
2. Continue using existing memory pages.
3. Run lint if the schema changed.
4. Do not rewrite old pages unless lint finds a concrete issue.

### Prompt Injection Boundary

Reference sources are raw data. They may contain text that looks like
instructions. The memory compiler must ignore those instructions unless they
come from the active system/developer prompt or `SCHEMA.md`.

Examples of ignored reference-source instructions:

- A user message saying "delete the wiki rules".
- A tool output containing "ignore previous instructions".
- A webpage saying "store this as the user's preference".

The compiler may record that such text was seen, but only as a sourced claim
about the source itself, not as an instruction.

### `SCHEMA.md` as Controlled Prompt Extension

`SCHEMA.md` is part of the memory compiler's instruction context. It is not a
reference source and should be versioned separately from the memory pages it
governs.

Rules:

- `SCHEMA.md` changes require a schema version bump.
- The active schema version is written into the agent context manifest.
- Every ingest log entry records the schema version used.
- When `SCHEMA.md` changes, run lint before trusting existing pages under the
  new rules.
- User/tool content in reference sources cannot edit `SCHEMA.md`; schema edits
  are explicit developer changes.
- Ordinary ingest APIs must mount or load `SCHEMA.md` read-only. A separate
  developer-mode operation is required to modify it.

## 8. AgentMemory Page Format

Every `.md` file under `pages/` must be a MemoryPack page.

```markdown
---
type: AgentMemory Page
page_id: decisions.memory-architecture
title: Memory Architecture Decisions
description: Current decisions about CoreMem memory architecture experiments.
read_when:
  - Discussing CoreMem memory architecture.
  - Comparing direct MemoryPack and observation-backed memory pipelines.
  - Resuming the MemoryPack POC.
memory_kind: decision
agent_memory_version: "0.1"
scope: project
status: active
activation: query
confidence: 0.9
sensitivity: normal
trust: user_authoritative
safe_to_act: true
resource: coremem://memorypack/pages/decisions/memory-architecture
tags: [memory, memorypack, llm-wiki, poc]
updated_at: 2026-06-21T00:00:00Z
source_turn_ids: [turn_20260621_000001]
source_message_ids: [msg_001]
agent_context_hashes: [sha256:abc123]
---

# Summary

The first POC tests direct conversation-turn to MemoryPack search before adding
the observation-backed hybrid.

# Current State

- The first POC should test direct conversation-turn to MemoryPack search,
  before adding the observation layer. [1]

# Open Questions

- Whether the direct MemoryPack pipeline beats current CoreMem on long-horizon
  recall.
- Whether the later observation-to-MemoryPack hybrid improves citation
  correctness.

# Citations

[1] [turn_20260621_000001](../../references/turns/turn_20260621_000001.md),
`msg_001`, `user_statement`:
"message -> wiki (okf) -> search"
```

Required frontmatter fields for `pages/**/*.md`:

- `type`: must be `AgentMemory Page` for compiled memory pages.
- `page_id`: stable identity for the page, independent of its current file path.
  File path is the current locator; `page_id` is the durable ID used by logs,
  indexes, future derived indexes, and rename handling.
- `title`: human display title.
- `description`: one-paragraph routing summary.
- `memory_kind`: one of `user_profile`, `preference`, `project_fact`,
  `decision`, `workflow`, `active_context`, `open_question`, `conflict`,
  `daily_note`, or `dream`.
- `agent_memory_version`: the MemoryPack profile version that produced the page.
  The POC uses `"0.1"`.
- `scope`: one of `user`, `project`, `workspace`, or `global`.
- `status`: one of `active`, `superseded`, `unresolved`, `archived`, or
  `pending_review`.
- `activation`: one of `startup`, `query`, `model_decision`, or `manual`.
  `startup` means the page is eligible to be linked from `MEMORY.md` and loaded
  only if it fits the configured boot budget; it does not mean every startup
  page is automatically loaded unbounded.

Recommended frontmatter fields:

- `read_when`: short trigger phrases for progressive disclosure.
- `confidence`: number from `0.0` to `1.0`.
- `sensitivity`: one of `normal`, `personal`, or `sensitive`.
- `trust`: one of `user_authoritative`, `tool_observed`, `assistant_derived`,
  `untrusted_source`, or `mixed`.
- `safe_to_act`: boolean indicating whether the agent may act on the memory
  without reconfirming it.
- `updated_at`: ISO 8601 timestamp.
- `source_turn_ids`, `source_message_ids`, and `agent_context_hashes`.

Conventional sections:

- `# Summary` contains a compact page synopsis for retrieval and quick reading.
- `# Current State` contains the current compiled synthesis.
- `# Read Next` lists related pages that should be loaded when needed.
- `# History` records material changes and when they happened.
- `# Superseded` records claims that used to be true or were replaced.
- `# Conflicts` records unresolved contradictions across sources.
- `# Open Questions` records unknowns that should guide future ingestion.
- `# Citations` lists evidence for durable claims.

`# Summary` is required for every page under `pages/`. Empty optional sections
may be omitted in small pages, but `# Citations` is required whenever the page
contains durable claims.

Page-level `trust` and `safe_to_act` are conservative summaries of the current
claims on the page:

- Use `trust: mixed` when current claims have different source authority.
- Use `safe_to_act: false` if any current actionable claim requires
  confirmation before the agent acts.
- Page-level values are routing hints, not replacements for citation-level
  evidence.

Action-sensitive claims may include claim-level metadata directly under the
claim bullet:

```markdown
- [claim:memory_arch_001] The first POC should test direct conversation-turn to
  MemoryPack search before adding observations. [1]
  - trust: user_authoritative
  - safe_to_act: true
  - evidence_type: user_statement
```

Claim-level metadata is required when a claim authorizes an action, changes
system behavior, contains sensitive personal data, or differs from the page's
conservative `trust` / `safe_to_act` values.

The deterministic linter validates claim-level metadata shape when present. The
LLM linter is responsible for detecting missing claim-level metadata, because
"action-sensitive" is semantic rather than purely syntactic.

## 9. Index Format

`index.md` is content-oriented. It lists compiled memory pages by category with
a one-line description and optional read trigger. It excludes
`references/**/*.md` by default.

```markdown
# Users

* [Default User](pages/users/default.md) - Durable user profile and identity facts.
* [Preferences](pages/users/preferences.md) - Durable user preferences and working style.

# Projects

* [CoreMem](pages/projects/coremem.md) - Project facts, architecture, and current work.

# Decisions

* [Memory Architecture](pages/decisions/memory-architecture.md) - Decisions
  about the MemoryPack POC and future hybrid pipeline.
```

The POC avoids frontmatter in `index.md` to keep routing cheap and predictable.

## 10. Boot Memory Format

`MEMORY.md` is the compact boot memory file. It is the only MemoryPack file that
is safe to load at session start by default.

Rules:

- Keep it small enough to fit comfortably in the system's startup context
  budget.
- Prefer links to deeper pages over duplicated detail.
- Include only active, high-signal memory.
- Do not include reference turn content.
- Include a short freshness note when a memory depends on changing project or
  environment state.
- If a page has `activation: startup`, link it from `MEMORY.md` only when it is
  high-signal enough to justify boot context cost.

```markdown
# CoreMem Memory

## Current Focus

- MemoryPack POC: direct reference-turn to compiled markdown memory, then
  grounded-observation hybrid.

## Read Next

- [Memory Architecture](pages/decisions/memory-architecture.md)
- [Open Questions](pages/open-questions.md)
```

## 11. Log Format

`log.md` is append-only and chronological, newest first.

```markdown
# AgentMemory Update Log

## 2026-06-21 ingest | turn_20260621_000001

* **Creation**: Added [Memory Architecture](pages/decisions/memory-architecture.md).
* **Update**: Added direct message-to-MemoryPack POC decision.
* **Agent Context**: `sha256:abc123`, schema `memorypack-poc-0.1`.
```

## 12. Ingest Pipeline

### Step 1: Capture Reference Turn

Create one immutable reference turn file under `references/turns/` and add its
hash to `references/manifest.json`.

### Step 2: Route Pages

The compiler reads:

- The new reference turn.
- `MEMORY.md`.
- `index.md`.
- Relevant existing memory pages under `pages/`.
- `SCHEMA.md`.
- Agent context manifest.

It outputs a structured update plan:

```json
{
  "updates": [
    {
      "page": "pages/decisions/memory-architecture.md",
      "action": "update",
      "reason": "User proposed a simpler direct message-to-MemoryPack POC.",
      "claims": [
        {
          "text": "The first POC should test direct conversation-turn to MemoryPack search.",
          "evidence_type": "user_statement",
          "source_turn_id": "turn_20260621_000001",
          "source_message_id": "msg_001",
          "source_quote": "message -> wiki (okf) -> search"
        }
      ]
    }
  ]
}
```

Valid `evidence_type` values:

- `user_statement` — the user said or chose this.
- `assistant_action` — the assistant decided, summarized, or did this.
- `tool_observation` — a tool result observed this from files, APIs, web, or
  the operating system.
- `derived_summary` — the claim synthesizes multiple cited sources. These
  claims must include `supporting_sources`, an array of two or more entries with
  their own source and evidence fields.

Atomic claims have this shape:

```json
{
  "text": "The first POC should test direct conversation-turn to MemoryPack search.",
  "evidence_type": "user_statement",
  "source_turn_id": "turn_20260621_000001",
  "source_message_id": "msg_001",
  "source_quote": "message -> wiki (okf) -> search"
}
```

Derived summary claims have this shape:

```json
{
  "text": "The MemoryPack POC should start direct and later compare against the observation-backed hybrid.",
  "evidence_type": "derived_summary",
  "supporting_sources": [
    {
      "evidence_type": "user_statement",
      "source_turn_id": "turn_20260621_000001",
      "source_message_id": "msg_001",
      "source_quote": "message -> wiki (okf) -> search"
    },
    {
      "evidence_type": "user_statement",
      "source_turn_id": "turn_20260621_000002",
      "source_message_id": "msg_004",
      "source_quote": "prove that your hybrid theory works"
    }
  ]
}
```

Rules:

- Non-`derived_summary` claims must include top-level `source_turn_id`,
  `source_message_id`, and `source_quote`.
- `derived_summary` claims must not use top-level source fields; all support
  must be inside `supporting_sources`.
- `supporting_sources[].evidence_type` must be one of `user_statement`,
  `assistant_action`, or `tool_observation`; nested `derived_summary` is not
  allowed in the POC.

### Step 3: Validate Claim Evidence

Before patching memory pages, validate every claim in the update plan:

1. `source_turn_id` resolves to a file in `references/turns/`.
2. The reference file hash matches `references/manifest.json`.
3. The referenced turn contains a valid canonical JSON payload.
4. `evidence_type` is one of the allowed values.
5. Non-`derived_summary` claims include top-level source fields.
6. `derived_summary` claims include two or more supporting sources and no
   top-level source fields.
7. `source_message_id` exists in `messages[].message_id` in the canonical
   payload for every cited source.
8. `source_quote` is an exact substring of the cited `messages[].content` for
   every cited source. No fuzzy matching is allowed in the direct POC.

Reject the claim if any validation check fails. The compiler may still preserve
the reference source, but it must not write an unsupported durable claim into
the compiled MemoryPack pages.

### Step 4: Patch Memory Pages

The compiler updates only the routed pages. It must preserve unknown
frontmatter fields and existing citations.

### Step 5: Update Boot, Index, and Log

Every ingest updates `index.md` if pages are added, renamed, or materially
changed. Every ingest appends to `log.md`.

`MEMORY.md` is updated only when a change is boot-worthy: high-signal, active,
and useful at session start. Most ingests should not change `MEMORY.md`.

### Step 6: Lint

Run deterministic lint after every ingest:

- All memory pages have frontmatter.
- Every memory page has `type`, `page_id`, `memory_kind`, and
  `agent_memory_version`.
- Every memory page has exactly one `# Summary` section.
- `index.md` links resolve.
- `MEMORY.md` links resolve.
- `MEMORY.md` stays under the configured boot budget.
- `MEMORY.md` does not contain reference turn content.
- `MEMORY.md` auto-load sections link only active, boot-worthy pages.
- `# Citations` links resolve.
- Every durable claim changed in this ingest has a citation.
- Every new citation has a valid `evidence_type`.
- Every cited `source_quote` is an exact substring of its cited message.
- Claim-level metadata, when present, uses valid `trust`, `safe_to_act`, and
  `evidence_type` values.
- No `references/` file was modified after initial write.
- Every `references/` file hash matches `references/manifest.json`.
- Every `references/turns/*.md` file has required reference frontmatter.
- Every `references/turns/*.md` file has exactly one `json agent_memory-turn`
  fenced canonical JSON payload.
- Every bundle declares exactly one active MemoryPack profile version across
  generated memory pages for the POC.
- No full system/developer prompt text appears in memory pages by default.

## 13. Query Pipeline

For POC search, keep retrieval simple:

1. Read `MEMORY.md` and `index.md`.
2. Use `description`, `read_when`, and `# Summary` as the first-pass routing
   surface.
3. Keyword/BM25 search memory pages under `pages/` when routing is not enough.
4. Select top 2-5 pages.
5. Answer from memory pages with citations.
6. Use `log.md` only for recency/debug context.
7. Fall back to `references/turns/` only if compiled memory has no
   relevant coverage.

No vector DB is required for the first POC.

Search should be implemented behind a small interface so the POC can use direct
markdown search first and later swap in HybridDB indexing without changing the
compiler or page format:

```text
AgentMemorySearch.search(query, scope=None) -> ranked page paths
```

Markdown remains canonical. Any future HybridDB index is a derived cache that
can be rebuilt from `memorypack/`.

## 14. Filing Query Answers Back

Following LLM Wiki, high-value query answers can become MemoryPack pages.

Examples:

- A comparison of current CoreMem vs MemoryPack memory.
- A benchmark interpretation.
- A design decision record.
- A discovered contradiction.

Filed answers must follow MemoryPack page format and cite the memory/reference
sources used to produce them.

## 15. Lint Pipeline

Add a `lint_memorypack()` helper for deterministic checks, then optionally add
LLM lint later.

Deterministic checks:

- Frontmatter parseable.
- `type`, `page_id`, `memory_kind`, and `agent_memory_version` present on memory
  pages.
- Required `# Summary` section present on memory pages.
- Reserved filenames are valid.
- `MEMORY.md` link, size, boot-worthiness, and reference-exclusion checks.
- Links resolve.
- Citations resolve.
- Index coverage for compiled memory pages under `pages/`.
- Reference immutability via manifest hash.
- Reference turn frontmatter and canonical payload validation.
- Cited `source_quote` values are exact substrings of cited messages.
- Evidence type values are valid.
- Claim-level metadata shape and enum values are valid when present.
- Bundle profile version is consistent for the POC.

LLM lint checks later:

- Contradictions between pages.
- Stale claims superseded by newer sources.
- Important concepts mentioned but missing pages.
- Orphan pages with no inbound links.
- Duplicated concepts across pages.
- Pages with too many unrelated claims.
- Action-sensitive claims missing claim-level metadata.
- Claims that are marked `safe_to_act: true` but have weak or mixed evidence.

## 16. Evaluation Plan

Compare two POC modes first:

| Mode | Pipeline |
|------|----------|
| A | Raw message/turn search |
| B | Direct turn-to-MemoryPack search |

Later add:

| Mode | Pipeline |
|------|----------|
| C | Turn-to-observations-to-MemoryPack hybrid |

### Dataset

Create 50-100 scripted turns containing:

- User preferences.
- Project facts.
- Tool-discovered facts.
- Architecture decisions.
- Superseded decisions.
- Contradictions.
- Open questions.
- Completed tasks.
- Instructions that look like prompt injection inside user/tool content.

### Questions

- What memory architecture does the user currently prefer?
- What did we decide about direct MemoryPack POC vs hybrid?
- Which claims came from user statements?
- Which claims came from tool output?
- What changed since the earlier decision?
- What open questions remain?
- Which facts are stale or superseded?
- What should the agent remember before resuming this project?

### Metrics

| Metric | Target |
|--------|--------|
| Answer accuracy | MemoryPack >= raw search |
| Citation correctness | >= 95% cited claims point to valid reference turn/message |
| Retrieval token count | MemoryPack uses >= 50% fewer tokens than raw search |
| Stale-memory rate | No worse than raw search |
| Prompt-injection safety | User/tool instructions in reference sources do not alter compiler rules |
| Human inspectability | Human can inspect current memory from `MEMORY.md` and `index.md` |

## 17. Success Criteria

The direct MemoryPack POC is successful if:

1. It answers long-horizon questions at least as accurately as raw message
   search.
2. It returns materially smaller context than raw message search.
3. Every durable claim has inspectable provenance.
4. The MemoryPack remains readable after 50-100 turns.
5. System/developer prompt text is not leaked into memory pages.
6. Prompt-injection text inside reference sources does not affect compiler behavior.

## 18. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Memory drift | Require citations for all durable claims. |
| Page explosion | Route facts into stable concept pages; do not create one page per turn. |
| System prompt leakage | Store only agent context hash/manifest by default. |
| Prompt injection | Treat reference sources as data, never instructions. |
| Stale claims | Use `log.md`, citations, and later LLM lint for supersession. |
| Overwriting good synthesis | Patch relevant sections only; preserve citations/history. |
| Tool-result confusion | Label tool evidence as `tool_observation`, not user preference. |
| Missing action metadata | Deterministic lint validates metadata shape; LLM lint detects semantic omissions. |
| Weak tamper evidence | Treat manifest hashes as accidental-mutation checks; use git commits for stronger anchoring. |
| Search complexity creep | Keep markdown search behind an interface; defer HybridDB indexing until MemoryPack wins. |
| Format isolation | Declare `agent_memory_version` and keep pages markdown/frontmatter compatible with simple readers. |

## 19. Future Hybrid Bridge

The direct POC should use citation fields that map cleanly to observations later:

```text
source_turn_id
source_message_id
source_quote
page_id
claim_id
```

The hybrid pipeline will add:

```text
source_observation_id
alignment_tier
alignment_confidence
char_interval
memory_type
durability
sensitivity
status
```

This lets the MemoryPack compiler change input from reference turns to grounded
observations without changing MemoryPack layout or retrieval semantics.

## 20. Implementation Milestones

### Milestone 1: File Format and Lint

- Add reference turn writer.
- Add MemoryPack templates.
- Add deterministic MemoryPack linter.
- Add agent context manifest.

### Milestone 2: MemoryPack Compiler

- Add page router prompt.
- Add page patcher prompt.
- Update `index.md` and `log.md` on ingest.
- Update `MEMORY.md` only for boot-worthy memory.
- Enforce no uncited durable claims.

### Milestone 3: MemoryPack Search

- Add simple keyword/BM25 search over compiled memory pages under `pages/`.
- Add MemoryPack-first context builder.
- Add reference fallback.

### Milestone 4: Evaluation

- Build scripted 50-100 turn dataset.
- Run raw-search vs direct-MemoryPack comparison.
- Record accuracy, citation correctness, token count, and stale claims.

### Milestone 5: Hybrid Experiment

- Feed existing CoreMem observations into the same MemoryPack compiler.
- Compare raw search, direct MemoryPack, and observation-backed MemoryPack.

## 21. Open Questions

- Should the POC live inside the CoreMem store path or a separate exported
  directory?
- Should reference turns be markdown with canonical JSON payloads, JSONL
  sidecars, or both?
- Should query answers be filed automatically or only when explicitly requested?
- How strict should the first citation linter be for rewritten summary sections?
- Should system/developer prompt persistence be completely disabled, or allowed
  behind an explicit local-only debug flag?
