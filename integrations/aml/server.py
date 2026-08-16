"""AML (Agent Memory Leaderboard) Add/Search adapter for CoreMem.

AML-specific: this module is NOT part of the coremem package. It exists only
to expose CoreMem through the AML evaluation contract for the academic
submission route (public GitHub repository + Docker entrypoint).

Contract implemented (verified against https://agentmemories.ai/api-guide):
  POST /v1/memories/add     — write conversation chunks
  POST /v1/memories/search  — retrieve evidence for a question
  GET  /health              — liveness (unauthenticated; any 2xx = healthy)

Key contract points:
  - Retrieval isolation is by ``user_id`` (store exactly, use identically
    for Add and Search — "the identical isolation boundary").
  - ``session_id`` identifies the source conversation (grouping only).
  - Add must return HTTP 200 only after messages are fully stored and
    searchable; echo ``request_id``, ``user_id``, ``session_id`` with
    ``success: true``.
  - Search returns a ``data`` array ordered most-relevant first, empty when
    nothing is relevant. Result fields: ``id`` (stable), ``content`` (memory
    text), optional ``score`` (higher = more relevant) and ``created_at``.
  - ``top_k`` is required in Search requests.
  - Any model used during Add/Search must be gpt-4o-mini. CoreMem is
    zero-LLM: no model is used, so the submission is deterministic and
    fully reproducible.

Environment:
  COREMEM_PATH   memory storage path (default /data/memory)
  AML_STRATEGY   recall strategy: episodic (default), direct, expanded, fusion
  AML_TOP_K      default top_k if the request omits it (contract says
                 required; this is a safety net, default 100)
  AML_MIN_SCORE  relevance gate: results below this score are dropped so the
                 contract's "empty array when nothing relevant" holds.
                 Applied to the cross-encoder logit when available (episodic),
                 else to the result score. Default 0.0 (logit > 0 = relevant).
  AML_API_KEY    if set, require it via Bearer or X-Api-Key (default: no
                 auth; "none" is limited to public smoke per the docs)
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
    timestamp: int | None = None  # Unix milliseconds


class AddRequest(BaseModel):
    request_id: str
    user_id: str  # retrieval isolation boundary
    session_id: str  # source conversation, grouping only
    messages: list[AddMessage]


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str
    user_id: str  # must exactly match the Add value
    top_k: int
    options: list[str] | None = None  # answer choices; not retrieval cues


class SearchResult(BaseModel):
    id: str
    content: str
    score: float | None = None  # higher = more relevant
    created_at: str | None = None  # ISO timestamp


class SearchResponse(BaseModel):
    data: list[SearchResult]


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
    """Store a chunk of conversation messages under the user's scope."""
    _check_auth(request)
    with _add_lock:
        for msg in payload.messages:
            ts = datetime.fromtimestamp(msg.timestamp / 1000, tz=UTC) if msg.timestamp else None
            core.ingest(
                msg.role,
                msg.content,
                session_id=payload.session_id,
                user_id=payload.user_id,
                ts=ts,
            )
    # ingest() is synchronous: messages are committed and searchable
    # before this response is returned (contract requirement).
    return AddResponse(
        request_id=payload.request_id,
        user_id=payload.user_id,
        session_id=payload.session_id,
    )


@app.post("/v1/memories/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    """Return evidence for a question, ordered most relevant first."""
    _check_auth(request)
    results = core.recall(
        payload.query,
        strategy=AML_STRATEGY,
        limit=payload.top_k,
        user_id=payload.user_id,
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
    return SearchResponse(data=[
        SearchResult(
            id=r.memory.id,
            content=r.memory.content,
            score=(
                getattr(r, "_ce_score", None)
                if getattr(r, "_ce_score", None) is not None
                else r.score
            ),
            created_at=(
                r.memory.ts.astimezone(UTC).isoformat().replace("+00:00", "Z")
                if r.memory.ts
                else None
            ),
        )
        for r in gated
    ])
