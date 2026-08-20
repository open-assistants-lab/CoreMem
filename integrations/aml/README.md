# CoreMem × Agent Memory Leaderboard (AML)

**AML-specific integration.** This directory exists only to expose CoreMem
through the [Agent Memory Leaderboard](https://agentmemories.ai/) evaluation
contract. It is not part of the `coremem` package.

## What this is

AML evaluates memory systems through two operations:

- **Add** — the platform writes conversation history in chunks
- **Search** — the platform asks for evidence relevant to a question

The platform owns answer generation (`gpt-4o-mini`), judging, and scoring.
CoreMem implements the memory side: `integrations/aml/server.py` maps the
AML contract onto `core.ingest()` / `core.recall()`.

## Contract implemented

Verified against the live page https://agentmemories.ai/api-guide (rendered
via browser automation).

| Endpoint | Request | Response |
|---|---|---|
| `POST /v1/memories/add` | `{request_id, user_id, session_id, messages: [{role, content, timestamp?}]}` | `{success: true, request_id, user_id, session_id}` — HTTP 200 only after messages are stored and searchable |
| `POST /v1/memories/search` | `{query, user_id, top_k, options?}` | `{data: [{id, content, score?, created_at?}]}` ordered most-relevant first; `[]` when nothing relevant |
| `GET /health` | — | any 2xx = healthy (unauthenticated) |

Contract details honored:

- **Isolation boundary** — `user_id` is stored exactly and used identically
  for Add and Search (the docs require "the identical isolation boundary").
  `session_id` maps to CoreMem's `session_id` (grouping only).
- **Timestamps** — message `timestamp` (Unix ms) is converted to a UTC
  datetime, so temporal questions keep their ordering. `created_at` in
  search results is an ISO timestamp.
- **Relevance gate** — CoreMem's recall always returns top-k; the adapter
  drops results below `AML_MIN_SCORE` (default `0.0`, applied to the
  cross-encoder logit) so the contract's "empty array when nothing relevant"
  holds. Verified: relevant pairs score ~+7, irrelevant ~−11. Note: for
  non-episodic strategies the gate falls back to the result score, which has
  no absolute meaning — use the default `episodic` strategy.
- **Concurrency** — the platform runs 64 Add workers concurrently; all
  writes are serialized through a lock (HybridDB's SQLite + ChromaDB are not
  safe for concurrent writes). Verified: 32 concurrent Add requests hang
  without the lock, complete cleanly with it.
- **Zero-LLM compliance** — AML requires any model used during Add/Search to
  be `gpt-4o-mini`. CoreMem uses **no model** in either operation (the
  default `episodic` strategy is fully local: BM25 + embeddings + a local
  cross-encoder). The submission is deterministic and exactly reproducible.
- **Retrieval pipeline (validated on LongMemEval-S, 500 questions)** — the
  default `episodic` strategy includes:
  - query decomposition (temporal: from/to, since/when, ago-event cues;
    +0.037 session recall on the 133 temporal-reasoning questions)
  - preference-question routing through a per-variant top-40 union
    (+0.033 session recall on the 30 preference questions)
  - hybrid search → RRF fusion → cross-encoder rerank (MiniLM-L-6) → MMR
    session diversity (0.950 session recall@5 on S)

## Run locally

```bash
pip install -e . fastapi uvicorn
COREMEM_PATH=/tmp/aml-memory uv run uvicorn aml_server:app --port 8000
```

Smoke it:

```bash
curl -X POST localhost:8000/v1/memories/add -H 'Content-Type: application/json' -d '{
  "request_id": "r1", "user_id": "u1", "session_id": "c1",
  "messages": [{"role": "user", "content": "I love hiking in Yosemite"}]}'
curl -X POST localhost:8000/v1/memories/search -H 'Content-Type: application/json' -d '{
  "query": "hiking Yosemite", "user_id": "u1", "top_k": 5}'
```

## Submit (academic route)

1. Make this repository public.
2. The platform builds the image from `integrations/aml/Dockerfile`
   (`pip install .` from the submitted tree — the evaluated code is exactly
   the submission) and starts the documented entrypoint.
3. No leaderboard key is issued for the academic route; the platform runs
   the smoke suite (Top K 90) and then the formal `full` evaluation.

> **Form values**: `integrations/aml/SUBMISSION_NOTES.md` contains the exact
> field values and the paste-ready submission notes for the evaluation
> request form (Academic leaderboard → Submit GitHub code for platform
> deployment).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `COREMEM_PATH` | `/data/memory` | Memory storage path |
| `AML_STRATEGY` | `episodic` | Recall strategy (`episodic`, `direct`, `expanded`, `fusion`) |
| `AML_TOP_K` | `100` | Default `top_k` when the request omits it |
| `AML_MIN_SCORE` | `-10.0` | Relevance floor: logits below this are "no relevant memory" (measured: the irrelevant cluster sits at −10 to −12; evidence spans −8 to +7 across query types). Preference queries skip the gate entirely |
| `AML_API_KEY` | *(unset)* | If set, require it via `Authorization: Bearer` or `X-Api-Key` |

## Notes

- The first Search downloads the cross-encoder (~500 MB) if not cached; the
  server warms up at startup so the first benchmark question is not slow.
- The platform requires public HTTP(S) endpoints for hosted submissions;
  the academic route deploys the container itself.
- Data sent by the platform is evaluation-only and deleted within 30 days.
