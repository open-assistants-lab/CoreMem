# CoreMem: Observer & Reflector Pipeline

2026-05-31

## Context

CoreMem is a zero-LLM memory system for AI agents. It stores and retrieves
conversation messages with semantic search, cross-encoder reranking, and
exact-match filtering. What's missing: **automatic enrichment** — extracting
facts from conversations and discovering patterns across them.

Current state: these exist only inside EA (`src/sdk/tools_core/observation.py`,
`src/sdk/middleware_observation.py`, `src/storage/memory.py` — ~600 lines total).
They are tightly coupled to EA's workspace system, provider factory, and
middleware hooks. This spec extracts them into CoreMem as a first-class feature.

## Summary

Add `coremem[observer]` extra providing:

| Module | Contents |
|--------|----------|
| `coremem/providers.py` | Lightweight provider factory (`create_provider("openai:gpt-4o")` → `.chat()`) |
| `coremem/observer.py` | `Observer` + `ObserverPipeline` (per-turn fact extraction) |
| `coremem/reflector.py` | `Reflector` + `ReflectorPipeline` (scheduled pattern discovery) |
| `coremem/memory_store.py` | `MemoryStore` — observations + reflections tables (extracted from EA) |

CoreMem owns scheduling, cursor tracking, token counting, dedup, quality gates.
EA's `ObservationMiddleware` becomes a ~10-line adapter mapping workspace → session_id.

## Provider System

### Design

CoreMem bundles its own provider factory using `httpx`. No SDK dependencies
(no `openai`, no `anthropic`). Model strings follow the format
`provider_prefix:model_name`:

```
openai:gpt-4o                                → https://api.openai.com/v1/chat/completions
ollama:llama3.2                              → http://localhost:11434/v1/chat/completions
anthropic:claude-sonnet-4-20250514           → https://api.anthropic.com/v1/messages
gemini:gemini-2.5-flash                      → https://generativelanguage.googleapis.com/v1beta
```

### API Key Resolution

Keys come from standard environment variables:

| Prefix | Env var | Endpoint URL |
|--------|---------|-------------|
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| `anthropic` | `ANTHROPIC_API_KEY` | `https://api.anthropic.com/v1` |
| `gemini` | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com/v1beta` |
| `ollama` | (none) | `http://localhost:11434/v1` |
| `ollama-cloud` | `OLLAMA_API_KEY` | `https://api.ollama.com` |

### Factory API

```python
from coremem.providers import create_provider

# Returns object with async .chat(messages) → ChatResponse
provider = create_provider("openai:gpt-4o-mini")

# CoreMem Protocol (stays — escape hatch for custom providers)
class LLMProvider(Protocol):
    def chat(self, messages: list[Any]) -> Any: ...
```

CoreMem never stores or touches API keys — it reads them at call time from
`os.environ`. Users with custom key sources inject their own provider object
that implements the `LLMProvider` Protocol.

### Why CoreMem's provider is simpler than EA's

EA's provider factory (`src/sdk/providers/factory.py`, 207 lines) is complex for
good reason — it powers a universal agent runtime. CoreMem only needs chat completion
with JSON-parsable responses. Different problem, different scale:

| | EA | CoreMem |
|---|---|---|
| **Goal** | Universal agent — any model, any provider | Chat completion — prompt in, JSON out |
| **API formats** | 4 (OpenAI, Anthropic, Gemini native, OllamaCloud native) | 2 (OpenAI-compatible, Anthropic) |
| **Protocols** | Streaming (17 event types), tool calling, token counting, reasoning blocks | `.chat(messages)` → `.content` |
| **Provider classes** | 4 heavyweight classes (280-380 lines each) | Lightweight inline adapters |
| **Model discovery** | models.dev registry (4172+ models, 110+ providers) | Not needed — user specifies model string |
| **Key storage** | Env vars + per-user settings.json + `provider_keys` dict | Env vars only |
| **SDK deps** | Package dependencies for every provider | Zero SDK deps — httpx only |

EA learned from experience that hardcoding 20 models meant every new provider
required a code change. The models.dev registry was the fix. But CoreMem's
Observer/Reflector doesn't need model discovery — the user explicitly chooses
one model for perception (Observer) and optionally one for reasoning (Reflector).

The 4 hardcoded provider names in CoreMem (`openai`, `anthropic`, `gemini`, `ollama`)
are not a model list — they're *API format adapters*. Every provider on earth
speaks either OpenAI-compatible (`POST /v1/chat/completions`) or Anthropic
(`POST /v1/messages`). Two adapter classes cover everything. A provider prefix
not in the known list falls through to the OpenAI-compatible adapter with
`{PREFIX}_API_KEY` and `{PREFIX}_BASE_URL` env vars — no code change needed.

If a new provider appears tomorrow that speaks OpenAI-compatible (which is nearly
everyone — Groq, Together, DeepSeek, OpenRouter, Fireworks, Mistral, Perplexity,
xAI, Cohere, Anyscale, Cerebras, etc.), it works with zero code changes.

## MemoryStore

### Schema

```sql
CREATE TABLE observations (
    id               TEXT PRIMARY KEY,
    content          LONGTEXT,          -- embedded for semantic search
    priority         TEXT,              -- 🔴 high / 🟡 medium / 🟢 low
    observation_ts   TEXT,              -- ISO timestamp of observation
    referenced_date  TEXT               -- date mentioned in content
);

CREATE TABLE reflections (
    id                     TEXT PRIMARY KEY,
    content                LONGTEXT,     -- embedded for semantic search
    domain                 TEXT,         -- preference, career, lifestyle, ...
    linked_observation_ids TEXT,         -- JSON array of supporting observation IDs
    score                  REAL          -- decays over time
);
```

### Scoping

Observations and reflections are implicitly scoped by the session they were
produced from. `ObserverPipeline` passes `session_id` to `MemoryCore.fetch()`
(line 192 of the algorithm), which returns only that session's messages. The
resulting observations belong to that session by construction — no explicit
`session_id` column needed on the observations/reflections tables.

If a future `MemoryStore.list_observations(session_id=...)` filter is needed,
add a `session_id TEXT` column to both tables. Not required for v1 — each
MemoryStore instance is isolated per session directory.

### API

```python
store = MemoryStore(path="./memory")

# Observations
store.insert_observations([{...}, ...])
store.get_observations(ts_after="2026-05-01", limit=50)
store.get_recent_observations(days=30, limit=50)

# Reflections
store.insert_reflections([{...}, ...])
store.get_reflections(limit=10)
store.apply_decay()       # reduces scores for reflections older than N days
```

Extracted from EA's `src/storage/memory.py` with `user_id` and `workspace_id`
removed — CoreMem doesn't know about users or workspaces. Those become metadata
keys if needed.

## ObserverPipeline

### Purpose

Extracts facts from conversation messages as discrete observations. Fires when
a configurable token threshold of *new* (unobserved) messages accumulates.

### API

```python
from coremem import MemoryCore, MemoryStore
from coremem.backends.hybrid import HybridBackend
from coremem.observer import ObserverPipeline

core = MemoryCore(backend=HybridBackend(path="./conversation"))
store = MemoryStore(path="./memory")

pipeline = ObserverPipeline(
    core=core,                          # reads messages via core.fetch()
    store=store,                        # writes observations
    session_id="main_conversation",     # cursor tracked per session
    model="ollama:llama3.2",           # cheap model, perception only
    token_threshold=8000,               # fire after 8K new tokens (default)
    min_turns=3,                        # at least 3 turns between runs (default)
    max_messages=500,                   # max messages to fetch (default)
)

# After each agent turn
await pipeline.after_turn()
```

### Internal State

```python
class ObserverPipeline:
    _last_observed_message_id: str | None     # watermark cursor
    _turns_since_last_run: int                # min_turns throttle
    _running: bool                            # prevents concurrent runs
```

### Algorithm

```
after_turn():
  1. Increment _turns_since_last_run
   2. Fetch recent messages: core.fetch(ts_after=_last_observed_ts_for_query, limit=max_messages)
   3. Filter: skip role="tool" messages (tool outputs never contain user facts)
   4. Count tokens in *new* messages only (since watermark)
   5. If tokens < token_threshold OR turns < min_turns: return
   6. Fetch prior observations: store.get_recent_observations(days=30, limit=50)
   7. Call Observer.run(conversation, prior_observations)
   8. Post-hoc dedup: skip new observations with >85% string similarity to existing
   9. Insert remaining: store.insert_observations(...)
   10. Update _last_observed_message_id from newest message processed
```

### Observer LLM Call

```python
class Observer:
    def __init__(self, model: str):
        self._provider = create_provider(model)

    async def run(
        self,
        conversation: list[dict],           # [{"role": "user", "content": "..."}, ...]
        prior_observations: list[dict],     # for dedup context
    ) -> list[dict]:
        prompt = self._build_prompt(conversation, prior_observations)
        response = await self._provider.chat([
            {"role": "system", "content": "You are an Observer. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ])
        return _parse_json_array(response.content)
```

Prompt uses the existing `OBSERVER_PROMPT` from EA with restructuring: conversation
text only appears once (not duplicated as both `conversation` and `previous_context`
template variables).

## ReflectorPipeline

### Purpose

Synthesizes patterns, relationships, and predictions from observations. Runs on
a configurable timer — not per-turn.

### API

```python
from coremem.reflector import ReflectorPipeline

reflector = ReflectorPipeline(
    store=store,                            # reads observations, writes reflections
    model="openai:gpt-4o",                 # strong model for reasoning
    interval_hours=24,                      # how often to run (default)
    min_observations=10,                    # skip if fewer observations (default)
)

# After each agent turn — no-op unless interval has elapsed
await reflector.maybe_run()

# Or run explicitly (bypasses timer)
await reflector.run_now()
```

### Internal State

```python
class ReflectorPipeline:
    _last_run_ts: float                     # time.time() of last run
    _last_run_observation_id: str | None    # cursor — only new observations since last run
```

### Algorithm

```
maybe_run():
  1. If now - _last_run_ts < interval_hours * 3600: return
  2. Fetch observations since _last_run_observation_id
  3. If count < min_observations: return
  4. Fetch prior reflections: store.get_reflections(limit=10)
   5. Priority sampling: if observations > 200, include all 🔴+🟡, sample 🟢 by recency
  6. Call Reflector.run(observations, prior_reflections)
  7. Quality gate: skip reflections with >90% cosine similarity to existing
  8. Insert remaining: store.insert_reflections(...)
  9. Update _last_run_ts and _last_run_observation_id
```

### Reflector LLM Call

```python
class Reflector:
    def __init__(self, model: str):
        self._provider = create_provider(model)

    async def run(
        self,
        observations: list[dict],
        prior_reflections: list[dict],
    ) -> list[dict]:
        prompt = self._build_prompt(observations, prior_reflections)
        response = await self._provider.chat([
            {"role": "system", "content": "You discover patterns and meaning from observations."},
            {"role": "user", "content": prompt},
        ])
        return _parse_json_array(response.content)
```

Uses the existing `REFLECTOR_PROMPT` from EA with incremental context — only
sends observations since the last run, not the full corpus.

### Quality Gate & Embedding

The quality gate (step 7) requires embedding reflection text for cosine similarity
comparison. CoreMem already depends on `sentence-transformers` — it reuses the same
embedding model used for semantic search. No new dependency.

```python
from coremem.embedding import embed

def _is_redundant(new_text, existing_reflections, threshold=0.9):
    new_emb = embed(new_text)
    for r in existing_reflections:
        stored_emb = r.get("embedding")
        if stored_emb and cosine_sim(new_emb, stored_emb) > threshold:
            return True
    return False
```

### Message Format

Observer and Reflector LLM calls use plain dicts for messages — not EA's `Message`
class (which lives in the SDK). The provider factory accepts `list[dict[str, Any]]`:

```python
response = await self._provider.chat([
    {"role": "system", "content": "You are an Observer. Return only valid JSON."},
    {"role": "user", "content": prompt},
])
```

This keeps CoreMem's provider layer framework-agnostic — no dependency on EA's SDK types.

## Separation of Concerns

```
ObserverPipeline          ReflectorPipeline
     │                          │
     │ reads messages           │
     ▼                          │
  MemoryCore.fetch()            │
     │                          │ reads observations
     │ writes observations      ▼
     └──────────────────→  MemoryStore.observations  ←─────────┐
                                │ reads/writes reflections     │
                                ▼                              │
                           MemoryStore.reflections ────────────┘
```

- **Observer** bridges raw messages → structured observations
- **Reflector** bridges observations → synthesized reflections
- **Neither depends on the other** — they only share `MemoryStore` tables
- **Either can be used independently** — user may only want Observer, or only Reflector

## EA Integration

### Before (current)

~600 lines of EA code: `observation.py` (187 lines), `middleware_observation.py`
(164 lines), `memory.py` (243 lines — observation/reflection parts).

### After (target)

~20 lines of EA code — thin middleware adapters:

```python
from coremem.observer import ObserverPipeline
from coremem.reflector import ReflectorPipeline
from src.sdk.middleware import Middleware

class ObservationMiddleware(Middleware):
    def __init__(self, workspace_id="personal", observer_model=None, reflector_model=None):
        self._observer = ObserverPipeline(
            core=_get_core(workspace_id),
            store=_get_store(workspace_id),
            session_id=workspace_id,       # workspace → session_id mapping
            model=observer_model or "ollama:llama3.2",
        )
        self._reflector = ReflectorPipeline(
            store=_get_store(workspace_id),
            model=reflector_model or "openai:gpt-4o",
        )

    async def after_agent(self, state):
        await self._observer.after_turn()
        await self._reflector.maybe_run()
        return None
```

EA maps `workspace_id` → `session_id` on the way in. CoreMem never hears the
word "workspace." Similarly, EA's `user_id` becomes `session_id` metadata in
CoreMem messages/observations/reflections — a user identifier stored as a
first-class column on records in CoreMem, not a path prefix.

`_get_core(workspace_id)` and `_get_store(workspace_id)` are thin helpers that
create/reuse `MemoryCore` and `MemoryStore` instances keyed to the workspace's
conversation and memory directories on disk. Same pattern as `get_message_store()` today.

## Configuration

### Environment

```
OBSERVER_MODEL=ollama:llama3.2        # optional override
REFLECTOR_MODEL=openai:gpt-4o         # optional override
OPENAI_API_KEY=sk-...                 # if using openai: prefix
ANTHROPIC_API_KEY=sk-ant-...          # if using anthropic: prefix
GEMINI_API_KEY=...                    # if using gemini: prefix
```

### Programmatic

```python
# Minimal — Observer only, local Ollama
pipeline = ObserverPipeline(core=core, store=store, session_id="main")
# Uses default model "ollama:llama3.2", default threshold 8000

# Full — both pipelines, cloud models
observer = ObserverPipeline(
    core=core, store=store, session_id="main",
    model="openai:gpt-4o-mini", token_threshold=4000, min_turns=2,
)
reflector = ReflectorPipeline(
    store=store,
    model="anthropic:claude-sonnet-4-20250514",
    interval_hours=12, min_observations=5,
)
```

## Migration Path

1. Add `coremem/providers.py`, `coremem/memory_store.py` to CoreMem
2. Add `coremem/observer.py`, `coremem/reflector.py`
3. Add `[observer]` extra to CoreMem's `pyproject.toml`: `httpx` (only new dependency)
4. Keep EA's observation middleware as-is until CoreMem release
5. After release: refactor EA's `ObservationMiddleware` to delegate to CoreMem pipes
6. Delete `src/sdk/tools_core/observation.py`, `src/sdk/middleware_observation.py`
7. Delete `src/storage/memory.py` — the entire file moves to `coremem/memory_store.py`
   (its only purpose is observations + reflections tables)

## Non-goals

- No streaming support for Observer/Reflector calls (one-shot prompts)
- No multi-agent observation (one Observer per session)
- No cross-session reflection (Reflector operates within one session's observations)
- No configurable MMR lambda for diversity (Observer/Reflector prompts are single-turn JSON extraction)

### Why summarization stays in EA

Summarization is a different category of operation — it is **destructive** (replaces
messages with a summary system prompt), operates **inside the request loop** (before
every LLM call), and manages the **agent's context window** — all agent runtime
concerns. Observer and Reflector are **additive** background enrichment that write
to separate tables.

| | Observer | Reflector | Summarization |
|---|---|---|---|
| **Operation** | Creates observations | Creates reflections | Replaces messages |
| **Trigger** | 8K new tokens since cursor | Every 24h | >50K total tokens before every LLM call |
| **Frequency** | ~every 3 turns | ~once daily | Potentially any turn |
| **Cursor** | `last_observed_message_id` | `last_run_observation_id` | Token budget per request |
| **Domain** | Memory enrichment | Memory enrichment | Agent runtime context window |

CoreMem is a storage/retrieval library. It creates and queries data. It should not
decide what to delete — that's EA's responsibility as the agent runtime.

## Dependencies

| Package | CoreMem base | With `[observer]` |
|---------|-------------|-------------------|
| `chromadb` | ✅ | ✅ |
| `numpy` | ✅ | ✅ |
| `sentence-transformers` | ✅ | ✅ |
| `httpx` | — | ✅ (new) |
| `hybriddb` | `[hybrid]` only | `[hybrid]` only |
| `openai` SDK | — | — |
| `anthropic` SDK | — | — |
| `google-genai` SDK | — | — |

Zero SDK dependencies. `httpx` is already a transitive dependency of most AI
libraries and is < 200KB. No new heavy dependencies.
