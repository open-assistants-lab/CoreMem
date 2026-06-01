# Observer & Reflector Optimization

2026-05-31

## Context

The memory system has a two-tier pipeline:
- **Observer** — extracts facts from conversations as discrete observations. Fires when 8K+ unobserved tokens accumulate (per workspace).
- **Reflector** — discovers patterns across observations, synthesizes reflections. Fires every 24 hours.

Both run background LLM calls. Neither currently has model configurability, delta tracking, or dedup. This spec captures optimization recommendations.

## Observer Recommendations

### 1. Track observed-message watermark (critical)

Currently sends all 500 messages every Observer run, even if 90% were already observed in prior runs. Store a `last_observed_message_id` in the middleware, pass only messages after that cursor.

```
BEFORE: 500 messages × 100 chars = ~50K prompt chars per run
AFTER:  ~20 new messages × 100 chars = ~2K prompt chars per run
```

Prior observations are still passed as "KNOWN OBSERVATIONS" context for dedup — the watermark only reduces the conversation input, not the dedup context.

### 2. Delta-based firing, not total-token

`_count_unobserved_tokens()` currently counts all 500 recent messages regardless of whether they were already observed. After implementing #1 (watermark), count only the delta since last observation. Result: fires predictably after ~8K *new* tokens regardless of total conversation length.

### 3. User-configurable Observer model

Allow the user to choose their Observer LLM provider and model. The Observer extracts facts from structured text — it doesn't need the same model tier as the main agent. A cheap model (`llama3.2`, `gemini-flash`, `gpt-4o-mini`) is more than sufficient and avoids burning expensive tokens on a perception-only task.

```python
ObservationMiddleware(
    user_id=...,
    workspace_id=...,
    observer_model="ollama:llama3.2",     # or None → use main agent model
)
```

### 4. Filter tool messages from Observer input

Tool outputs are the largest messages and never contain user facts. Skip `role="tool"` entirely. Cuts prompt size another 40-60%.

```
raw_messages = [
    {"role": m.role, "content": m.content}
    for m in messages
    if m.content.strip() and m.role != "tool"  # skip tool outputs
]
```

### 5. Post-hoc dedup before insert

Observer returns 20 observations, 15 are near-duplicates of existing ones. Run cheap string similarity check before inserting:

```python
from difflib import SequenceMatcher

for obs in new_observations:
    for existing in known_observations:
        ratio = SequenceMatcher(None, obs["content"].lower(), existing["content"].lower()).ratio()
        if ratio > 0.85:
            break  # skip near-duplicate
    else:
        insert(obs)
```

### 6. Observer prompt restructuring

With the watermark (#1), the conversation shouldn't appear twice in the prompt. Currently:

```
OBSERVER_PROMPT.format(conversation=..., previous_context=...)
# previous_context = "KNOWN OBS: ...\n\nNEW CONVERSATION:\n{conversation}"
# → conversation appears in both template vars
```

Fix: remove the duplicate `conversation` template variable, structure as:

```
CONTEXT (prior 10 observations): ...
NEW CONVERSATION (since last run): ...
```

## Reflector Recommendations

### 7. Use a *better* model, not a cheaper one

Reflector does pattern discovery (contradiction detection, multi-observation synthesis, predictions). It benefits from strong reasoning — same tier as the main agent. Observer is the one where cheap models shine.

```python
ObservationMiddleware(
    user_id=...,
    workspace_id=...,
    reflector_model="ollama:minimax-m2.5",  # or None → use main agent model
)
```

### 8. Incremental, not full corpus

Currently dumps ALL observations every Reflector run. After a few weeks that's thousands of records. Instead: only send observations created since the last successful reflection run.

```
BEFORE: 2000 observations × 100 bytes = ~200KB prompt
AFTER:  ~50 observations (7 days of delta) × 100 bytes = ~5KB prompt
```

Prior reflections already carry the historical synthesis — they act as compressed context. New observations only need to be compared against prior reflections, not against the raw observation corpus.

### 9. Sampling by priority for context overflow

If the incremental window still has 200+ observations, sample by priority:

```python
def sample_observations(obs: list[dict], max_tokens: int = 20000) -> list[dict]:
    high = [o for o in obs if o.get("priority") == "🔴"]
    medium = [o for o in obs if o.get("priority") == "🟡"]
    low = [o for o in obs if o.get("priority") == "🟢"]

    result = high + medium  # all high/medium
    remaining = max_tokens - sum(len(o["content"]) for o in result)
    for o in sorted(low, key=lambda o: o.get("observation_ts", ""), reverse=True):
        if remaining <= 0:
            break
        result.append(o)
        remaining -= len(o["content"])
    return result
```

Low-priority 🟢 observations contribute little to pattern discovery — they're context/trivia. Always include all 🔴 and 🟡.

### 10. Configurable Reflector schedule

`REFLECTOR_INTERVAL_SECONDS` hardcoded at 24h. Pass as constructor param:

```python
ObservationMiddleware(
    user_id=...,
    workspace_id=...,
    reflector_interval_seconds=12 * 3600,  # 12h for power users
)
```

### 11. Decouple decay from Reflector

`_fire_reflector()` calls `apply_decay()` as a side-effect. Decay should be either:
- A separate hourly timer, or
- Lazy: compute from timestamps at query time (`SELECT * FROM reflections WHERE ts + decay_days > NOW()`)

Reflector's job is synthesis, not mutation.

### 12. Reflector quality gate

If a new reflection's embedding cosine similarity vs existing reflections is > 0.9, skip it. Avoids "rediscovering the same pattern" on consecutive runs.

```python
from numpy import dot
from numpy.linalg import norm

def _is_redundant(new_text: str, existing_reflections: list[dict], threshold: float = 0.9) -> bool:
    new_emb = embed(new_text)
    for existing in existing_reflections:
        existing_emb = existing.get("embedding")
        if existing_emb and dot(new_emb, existing_emb) / (norm(new_emb) * norm(existing_emb)) > threshold:
            return True
    return False
```

## Nice-to-haves (lower priority)

### 13. Link observations to source messages

Add `source_message_ids: list[str]` on observations. Enables: "What conversation produced this observation?" Currently impossible to trace an observation back to the source.

### 14. Observation priority auto-upgrade

If a fact appears in 3+ observations (e.g., user keeps mentioning their dog's name), auto-upgrade to 🔴 priority. Reflector then prioritizes it for pattern discovery.

### 15. Merge existing hooks into `ObservationMiddleware`

`MemoryMiddleware` (in `middleware_memory.py`) is currently commented out in runner.py. If uncommented, it auto-injects memory context into the conversation — which overlaps with `ObservationMiddleware`. Merge into one class or reconcile the division of responsibility.

## Implementation priority

| # | Item | Impact | Effort |
|---|------|--------|--------|
| 1 | Observer watermark | High | Low |
| 4 | Filter tool messages | High | Low |
| 2 | Delta-based firing | Medium | Low |
| 7 | Reflector model config | Medium | Low |
| 3 | Observer model config | Medium | Low |
| 6 | Prompt restructuring | Medium | Low |
| 8 | Reflector incremental | High | Medium |
| 5 | Post-hoc dedup | Medium | Medium |
| 9 | Priority sampling | Medium | Medium |
| 10 | Configurable schedule | Medium | Low |
| 11 | Decouple decay | Medium | Medium |
| 12 | Quality gate | Medium | Medium |
| 13 | Source message links | Low | Medium |
| 14 | Priority auto-upgrade | Low | Medium |
| 15 | Merge memory middlewares | Low | Medium |
