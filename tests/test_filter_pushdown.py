"""Tests for hybriddb 0.7.0 metadata pre-filtering pushdown (0.16.0).

Filters (role/session_id/user_id/agent_id) are pushed into the Chroma scan
before the vector search; the Python post-filter remains the authority.
docs/versioned-memory-design.md §2 spirit — engine-level filtering.
"""

from __future__ import annotations

import shutil
import tempfile

import pytest

from coremem import MemoryCore


def _make_core(**kwargs) -> MemoryCore:
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d, **kwargs)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


def _noise(core: MemoryCore, n: int = 60, session: str = "noise") -> None:
    msgs = [
        {"id": f"n{i}", "role": "user",
         "content": f"cooking recipe number {i}: boil pasta, add garlic, stir the sauce.",
         "session_id": session, "ts": "2024-01-01T00:00:00+00:00"}
        for i in range(n)
    ]
    core.ingest_many(msgs)


def test_role_filter_returns_only_matching_role():
    core = _make_core()
    try:
        core.ingest_many([
            {"id": "u1", "role": "user", "content": "I love gardening and planted tomatoes.",
             "session_id": "s1", "ts": "2024-01-01T00:00:00+00:00"},
            {"id": "a1", "role": "assistant", "content": "Tomatoes need sun and water weekly.",
             "session_id": "s1", "ts": "2024-01-01T00:01:00+00:00"},
        ])
        hits = core.recall("tomatoes", role="user", limit=5)
        assert hits
        assert all(h.memory.role == "user" for h in hits)
    finally:
        core._test_cleanup()


def test_session_filter_finds_message_beyond_overfetch_window():
    """The starvation case the pushdown fixes: 150 exact-phrase matches in a
    noise session push the target out of the unfiltered over-fetch window
    (top-100), so the old unfiltered-then-postfilter path returned EMPTY.
    With the Chroma pre-filter, the vector scan only sees session-matching
    rows and the target is the top hit."""
    core = _make_core()
    try:
        core.ingest_many([
            {"id": f"c{i}", "role": "user",
             "content": f"coffee note {i}: flat white oat milk coffee order at the cafe was great.",
             "session_id": "noise", "ts": "2024-01-02T00:00:00+00:00"}
            for i in range(150)
        ])
        core.ingest_many([
            {"id": "target", "role": "user",
             "content": "My usual order is an oat-milk flat white.",
             "session_id": "target", "ts": "2024-01-01T00:00:00+00:00"},
        ])
        hits = core.recall("flat white oat milk coffee order", session_id="target", limit=5)
        assert hits, "session-filtered recall must not starve"
        assert all(h.memory.session_id == "target" for h in hits)
        assert any(h.memory.id == "target" for h in hits)
    finally:
        core._test_cleanup()


def test_combined_role_and_session_pushdown():
    core = _make_core()
    try:
        core.ingest_many([
            {"id": "u_t", "role": "user", "content": "planning the Kyoto trip itinerary.",
             "session_id": "trips", "ts": "2024-01-01T00:00:00+00:00"},
            {"id": "a_t", "role": "assistant", "content": "Kyoto trip: temples and food markets.",
             "session_id": "trips", "ts": "2024-01-01T00:01:00+00:00"},
            {"id": "u_o", "role": "user", "content": "planning the Kyoto trip but in another session",
             "session_id": "other", "ts": "2024-01-01T00:00:00+00:00"},
        ])
        hits = core.recall("kyoto trip planning", session_id="trips", role="user", limit=5)
        assert hits
        assert all(h.memory.session_id == "trips" and h.memory.role == "user" for h in hits)
    finally:
        core._test_cleanup()


def test_empty_string_filters_are_no_filter():
    core = _make_core()
    try:
        core.ingest_many([
            {"id": "x1", "role": "user", "content": "note about the aquarium visit.",
             "session_id": "s1", "ts": "2024-01-01T00:00:00+00:00"},
        ])
        # empty strings normalize to no-filter (same as None)
        hits = core.recall("aquarium", session_id="", role="", limit=5)
        assert any(h.memory.id == "x1" for h in hits)
    finally:
        core._test_cleanup()


def test_metadata_and_ts_filters_still_work_python_side():
    core = _make_core()
    try:
        core.ingest_many([
            {"id": "k1", "role": "user", "content": "the warranty expires next March.",
             "session_id": "s1", "metadata": {"topic": "warranty"},
             "ts": "2024-01-01T00:00:00+00:00"},
            {"id": "k2", "role": "user", "content": "the warranty for the laptop is separate.",
             "session_id": "s1", "metadata": {"topic": "hardware"},
             "ts": "2024-01-01T00:00:00+00:00"},
        ])
        hits = core.recall("warranty", metadata={"topic": "warranty"}, limit=5)
        assert any(h.memory.id == "k1" for h in hits)
        assert all((h.memory.metadata or {}).get("topic") != "hardware" for h in hits)
    finally:
        core._test_cleanup()