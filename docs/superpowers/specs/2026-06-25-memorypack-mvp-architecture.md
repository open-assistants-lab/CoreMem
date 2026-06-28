# MemoryPack MVP Architecture

## Data Flow

```
Agent conversation
       │
       ▼
  Raw messages (turn)
       │
       ├──► HybridDB (structured search: by role, date, session)
       │
       └──► LLM compiler ──► Compiled page (MD file)
                                │
                                ├── pages/turn_id.md
                                │   (structured claims + exact quotes)
                                │
                                ├── daily/2026-06-25.md
                                │   (synthesized from today's turn pages)
                                │
                                ├── weekly/2026-W26.md
                                │   (synthesized from daily pages)
                                │
                                └── monthly/2026-06.md
                                    (synthesized from weekly pages)
```

## How the Agent Uses It

**1. Search** — Agent calls `search_memory("fracking regulations")` → BM25 + cross-encoder finds top-5 compiled pages → injected into context. Fast (~400ms), deterministic, cheap.

**2. Browse** — Agent reads `daily/2026-06-25.md` to see "what happened today." Links to turn pages for details. No search needed.

**3. Read** — Agent opens `pages/fracking-groundwater.md` directly. Sees summary + claims + citations. Human-readable, LLM-readable.

**4. Reflect** — The daily/weekly/monthly pages include a "Patterns" section synthesized from the level below. Reflections are a byproduct of summarization, not a separate pipeline.

## File Structure

```
memorypack/
  pages/              ← Compiled turn pages (per-turn)
    fracking-groundwater.md
    bachelorette-party.md
    charity-events.md
  daily/              ← Daily summaries
    2026-06-25.md
    2026-06-26.md
  weekly/             ← Weekly summaries
    2026-W26.md
  monthly/            ← Monthly summaries
    2026-06.md
  index.md            ← Table of contents
  MEMORY.md           ← Boot memory (critical pages)
```

## Key Principle

Everything is an MD file. The agent reads MD files directly or searches them. No database for compiled pages. Raw messages go in HybridDB for structured search (by role, date, session). Compiled pages are always MD files — human-readable, LLM-readable, git-trackable.
