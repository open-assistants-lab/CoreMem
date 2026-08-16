"""AML (Agent Memory Leaderboard) Add/Search adapter for CoreMem.

AML-specific: this module is NOT part of the coremem package. It exists only
to expose CoreMem through the AML evaluation contract
(https://agentmemories.ai/api-guide) for the academic submission route
(public GitHub repository + Docker entrypoint).

Contract implemented:
  POST /v1/memories/add     — write conversation chunks
  POST /v1/memories/search  — retrieve evidence for a question
  GET  /health              — liveness (optional per contract)

Key contract points:
  - Retrieval isolation is by ``scope`` (stored exactly, echoed in Search).
  - ``conversation_id`` identifies the source conversation (grouping only).
  - Add must respond only after messages are fully stored and searchable.
  - Search returns an ordered array of {id, text, ts}, empty when nothing
    is relevant. The platform preserves the order for its answer pipeline.
  - Any model used during Add/Search must be gpt-4o-mini. CoreMem is
    zero-LLM: no model is used, so the submission is deterministic and
    fully reproducible.

Environment:
  COREMEM_PATH   memory storage path (default /data/memory)
  AML_STRATEGY   recall strategy: episodic (default), direct, expanded, fusion
  AML_TOP_K      default top_k when the request omits it (default 100)
  AML_MIN_SCORE  relevance gate: results below this score are dropped so the
                 contract's "empty array when nothing relevant" holds.
                 Applied to the cross-encoder logit when available (episodic),
                 else to the result score. Default 0.0 (logit > 0 = relevant).
  AML_API_KEY    if set, require it via Bearer or X-Api-Key (default: no auth)
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from coremem import MemoryCore

logger = logging.getLogger(__name__)

MEMORY_PATH = os.environ.get("COREMEM_PATH", "/data/memory")
AML_STRATEGY = os.environ.get("AML_STRATEGY", "episodic")
AML_TOP_K = int(os.environ.get("AML_TOP_K", "100"))
AML_MIN_SCORE = float(os.environ.get("AML_MIN_SCORE", "0.0"))
AML_API_KEY = os.environ.get("AML_API_KEY", "")
SCOPE_METADATA_KEY = "aml_scope"

core = MemoryCore(path=MEMORY_PATH)

# The AML platform runs Add/Search workers concurrently (64 Add workers by
# default). HybridDB's SQLite + ChromaDB are not safe for concurrent writes,
# so all ingest() calls are serialized through this lock.
_add_lock = threading.Lock()


# ── Startup warmup ─────────────────────────────────────────────────────────


def _warmup() -> None:
    """Pre-load embedding + cross-encoder models so the first Search is fast.

    The first episodic recall downloads the cross-encoder (~500 MB) if not
    cached; doing it at startup avoids a slow first benchmark question.
    """
    try:
        core.recall("warmup", strategy=AML_STRATEGY, limit=1)
    except Exception:
        logger.warning("AML warmup failed; models will load lazily on first Search", exc_info=True)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _warmup()
    yield


app = FastAPI(title="CoreMem AML Adapter", lifespan=lifespan)


# ── Request/response models ────────────────────────────────────────────────


class AddMessage(BaseModel):
    role: str
    content: str
    ts: int | None = None  # Unix milliseconds


class AddRequest(BaseModel):
    request_id: str
    scope: str
    conversation_id: str
    messages: list[AddMessage]


class AddResponse(BaseModel):
    request_id: str
    status: str = "ok"


class SearchRequest(BaseModel):
    query: str
    scope: str
    options: list[str] | None = None  # answer choices; not used for retrieval
    top_k: int | None = None


class SearchResult(BaseModel):
    id: str
    text: str
    ts: int | None = None  # Unix milliseconds


class SearchResponse(BaseModel):
    results: list[SearchResult]


# ── Auth ──────────────────────────────────────────────────────────────────


def _check_auth(request: Request) -> None:
    if not AML_API_KEY:
        return
    if request.headers.get("Authorization") == f"Bearer {AML_API_KEY}":
        return
    if request.headers.get("X-Api-Key") == AML_API_KEY:
        return
    raise HTTPException(status_code=401, detail="invalid API key")


# ── Endpoints ─────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/memories/add", response_model=AddResponse)
def add(payload: AddRequest, request: Request) -> AddResponse:
    """Store a chunk of conversation messages under the request's scope."""
    _check_auth(request)
    with _add_lock:
        for msg in payload.messages:
            ts = datetime.fromtimestamp(msg.ts / 1000, tz=UTC) if msg.ts else None
            core.ingest(
                msg.role,
                msg.content,
                session_id=payload.conversation_id,
                ts=ts,
                metadata={SCOPE_METADATA_KEY: payload.scope},
            )
    # ingest() is synchronous: messages are committed and searchable
    # before this response is returned (contract requirement).
    return AddResponse(request_id=payload.request_id)


@app.post("/v1/memories/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    """Return evidence for a question, ordered most relevant first."""
    _check_auth(request)
    top_k = payload.top_k or AML_TOP_K
    results = core.recall(
        payload.query,
        strategy=AML_STRATEGY,
        limit=top_k,
        metadata={SCOPE_METADATA_KEY: payload.scope},
    )
    # Relevance gate: recall always returns top-k, but the contract requires
    # an empty array when nothing relevant exists. The cross-encoder logit
    # (set on results by the episodic strategy) separates relevant from
    # irrelevant cleanly; fall back to the result score for other strategies.
    gated = [
        r for r in results
        if (getattr(r, "_ce_score", None) if getattr(r, "_ce_score", None) is not None else r.score)
        >= AML_MIN_SCORE
    ]
    return SearchResponse(results=[
        SearchResult(
            id=r.memory.id,
            text=r.memory.content,
            ts=int(r.memory.ts.timestamp() * 1000) if r.memory.ts else None,
        )
        for r in gated
    ])
