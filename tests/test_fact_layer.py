"""Tests for the fact layer — facts versioned table + add_fact with
ontology-cardinality supersession (docs/superpowers/plans/2026-09-07-fact-layer.md Task 2)."""

from __future__ import annotations

import shutil
import tempfile

import pytest

from coremem import MemoryCore


def _make_core(**kwargs) -> MemoryCore:
    d = tempfile.mkdtemp()
    core = MemoryCore(path=d, author="test", **kwargs)
    core._test_cleanup = lambda: shutil.rmtree(d, ignore_errors=True)
    return core


def test_add_fact_records_on_chain_and_lists_current():
    core = _make_core(versioned=True)
    try:
        fid = core.add_fact("user", "dog", "Max", source_message_id="a1")
        facts = core.list_facts(entity="user", attribute="dog")
        assert len(facts) == 1 and facts[0]["value"] == "Max"
        assert facts[0]["valid_to"] == "" and facts[0]["source_message_id"] == "a1"
        assert core.verify_memory_chain()["facts"]["valid"]
    finally:
        core._test_cleanup()


def test_single_cardinality_supersedes_multi_appends():
    core = _make_core(versioned=True)
    try:
        core.add_fact("user", "employer", "Acme")
        core.add_fact("user", "employer", "Globex")
        current = core.list_facts(entity="user", attribute="employer")
        assert len(current) == 1 and current[0]["value"] == "Globex"
        core.add_fact("user", "child", "Emma")
        core.add_fact("user", "child", "Leo")
        kids = core.list_facts(entity="user", attribute="child")
        assert {k["value"] for k in kids} == {"Emma", "Leo"}
        assert core.verify_memory_chain()["facts"]["valid"]
    finally:
        core._test_cleanup()


def test_add_fact_requires_versioned_store():
    core = _make_core(versioned=False)
    try:
        with pytest.raises(RuntimeError, match="versioned"):
            core.add_fact("user", "dog", "Max")
    finally:
        core._test_cleanup()

def test_list_facts_filters_and_hides_expired():
    core = _make_core(versioned=True)
    try:
        core.add_fact("user", "employer", "Acme")
        core.add_fact("user", "employer", "Globex")   # supersedes Acme
        core.add_fact("user", "pet", "Max")
        values = {f["value"] for f in core.list_facts(entity="user")}
        assert "Globex" in values and "Acme" not in values and "Max" in values
        expired = core.list_facts(entity="user", include_expired=True)
        assert any(f["value"] == "Acme" and f["valid_to"] for f in expired)
        assert all(f["user_id"] == "" for f in expired)
    finally:
        core._test_cleanup()


def test_get_fact_returns_row_or_none():
    core = _make_core(versioned=True)
    try:
        fid = core.add_fact("user", "dog", "Max")
        assert core.get_fact(fid)["value"] == "Max"
        assert core.get_fact("nonexistent") is None
    finally:
        core._test_cleanup()


def test_fact_history_shows_chain_events():
    core = _make_core(versioned=True)
    try:
        fid = core.add_fact("user", "dog", "Max")
        events = core.fact_history(fid)
        assert len(events) >= 1
        assert all(e.get("op") in ("insert", "add", "update", "delete") for e in events)
    finally:
        core._test_cleanup()
