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

| Endpoint | Request | Response |
|---|---|---|
| `POST /v1/memories/add` | `{request_id, scope, conversation_id, messages: [{role, content, ts?}]}` | `{request_id, status: "ok"}` — sent only after messages are stored and searchable |
| `POST /v1/memories/search` | `{query, scope, options?, top_k?}` | `{results: [{id, text, ts}]}` ordered most-relevant first; `[]` when nothing relevant |
| `GET /health` | — | `{status: "ok"}` |

Contract details honored:

- **Scope isolation** — `scope` is stored exactly and used to isolate searches
  (stored in message metadata, filtered on every recall). `conversation_id`
  maps to CoreMem's `session_id` (grouping only).
- **Timestamps** — Unix milliseconds are converted to UTC datetimes, so
  temporal questions keep their ordering.
- **Relevance gate** — CoreMem's recall always returns top-k; the adapter
  drops results below `AML_MIN_SCORE` (default `0.0`, applied to the
  cross-encoder logit) so the contract's "empty array when nothing relevant"
  holds. Verified: relevant pairs score ~+7, irrelevant ~−11.
- **Zero-LLM compliance** — AML requires any model used during Add/Search to
  be `gpt-4o-mini`. CoreMem uses **no model** in either operation (the
  default `episodic` strategy is fully local: BM25 + embeddings + a local
  cross-encoder). The submission is deterministic and exactly reproducible.

## Run locally

```bash
pip install -e . fastapi uvicorn
COREMEM_PATH=/tmp/aml-memory uv run uvicorn aml_server:app --port 8000
```

Smoke it:

```bash
curl -X POST localhost:8000/v1/memories/add -H 'Content-Type: application/json' -d '{
  "request_id": "r1", "scope": "s1", "conversation_id": "c1",
  "messages": [{"role": "user", "content": "I love hiking in Yosemite"}]}'
curl -X POST localhost:8000/v1/memories/search -H 'Content-Type: application/json' -d '{
  "query": "hiking Yosemite", "scope": "s1", "top_k": 5}'
```

## Submit (academic route)

1. Make this repository public.
2. The platform builds the image from `integrations/aml/Dockerfile`
   (`pip install .` from the submitted tree — the evaluated code is exactly
   the submission) and starts the documented entrypoint.
3. No leaderboard key is issued for the academic route; the platform runs
   the smoke suite (Top K 90) and then the formal `full` evaluation.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `COREMEM_PATH` | `/data/memory` | Memory storage path |
| `AML_STRATEGY` | `episodic` | Recall strategy (`episodic`, `direct`, `expanded`, `fusion`) |
| `AML_TOP_K` | `100` | Default `top_k` when the request omits it |
| `AML_MIN_SCORE` | `0.0` | Relevance gate (cross-encoder logit) |
| `AML_API_KEY` | *(unset)* | If set, require it via `Authorization: Bearer` or `X-Api-Key` |

## Notes

- The first Search downloads the cross-encoder (~500 MB) if not cached; the
  server warms up at startup so the first benchmark question is not slow.
- The platform requires public HTTP(S) endpoints for hosted submissions;
  the academic route deploys the container itself.
- Data sent by the platform is evaluation-only and deleted within 30 days.
