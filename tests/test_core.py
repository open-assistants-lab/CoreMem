"""Tests for MemoryCore with ChromaBackend."""

from coremem.backends.chroma import ChromaBackend
from coremem.core import MemoryCore
from coremem.types import Memory


def test_core_ingest_and_search(chroma_tmp_path, sample_messages):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest_many(sample_messages)
    assert core.count() == len(sample_messages)

    results = core.search("model kits", limit=10)
    assert len(results) > 0
    assert any("model" in r.memory.content.lower() for r in results)


def test_core_ingest_single(chroma_tmp_path):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    mid = core.ingest("user", "Hello world")
    assert mid
    assert core.count() == 1


def test_core_wake_up(chroma_tmp_path, sample_messages):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest_many(sample_messages)

    ctx = core.wake_up(user_id="alice")
    assert "[L0: Identity]" in ctx
    assert "alice" in ctx
    assert "[L1: Essential]" in ctx


def test_core_deep_search_context(chroma_tmp_path, sample_messages):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest_many(sample_messages)

    ctx = core.deep_search_context("model kits")
    assert ctx is not None
    assert "model kits" in ctx


def test_core_search_result_boosts_heuristics(chroma_tmp_path):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest("user", "I love building model kits. Finished a Revell F-15 Eagle.")
    core.ingest("user", "My favorite food is pizza.")

    results = core.search("how many model kits", limit=5)
    assert len(results) > 0

    model_result = next((r for r in results if "model" in r.memory.content.lower()), None)
    assert model_result is not None
    assert model_result.score > 0.0


def test_core_clear(chroma_tmp_path, sample_messages):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest_many(sample_messages)
    assert core.count() == len(sample_messages)
    core.clear()
    assert core.count() == 0


def test_model_kits_counting_scenario(chroma_tmp_path):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest("user", "I recently finished a Revell F-15 Eagle model kit")
    core.ingest("user", "Started a Tamiya 1/48 scale Spitfire Mk.V")
    core.ingest("user", "Just bought a 1/72 scale B-29 bomber")
    core.ingest("user", "Also got a 1/24 scale '69 Camaro kit")
    core.ingest("user", "My current project is a Tiger I tank")

    results = core.search("How many model kits have I worked on or bought?", limit=10)
    assert len(results) > 0

    kit_results = [r for r in results if "kit" in r.memory.content.lower()
                   or "spitfire" in r.memory.content.lower()
                   or "f-15" in r.memory.content.lower()
                   or "model" in r.memory.content.lower()]
    assert len(kit_results) >= 1


def test_knowledge_update_temporal(chroma_tmp_path):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest("user", "I just ran a 5K in 30:00")
    core.ingest("user", "New personal best: 25:50 at the charity run")
    core.ingest("user", "My name is Alice")

    results = core.search("What is my current 5K time?", limit=5)
    assert len(results) > 0
    assert any("25:50" in r.memory.content for r in results)


# ── fetch / fetch_all / store ──


def test_fetch_all_returns_all_messages(chroma_tmp_path, sample_messages):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest_many(sample_messages)

    results = core.fetch_all()
    assert len(results) == len(sample_messages)
    for r in results:
        assert isinstance(r, Memory)


def test_fetch_with_limit(chroma_tmp_path, sample_messages):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest_many(sample_messages)

    page1 = core.fetch(limit=2)
    assert len(page1) == 2

    page2 = core.fetch(limit=2, offset=2)
    assert len(page2) == 2

    assert page1[0].content != page2[0].content


def test_fetch_filters_by_role(chroma_tmp_path, sample_messages):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest_many(sample_messages)

    user_msgs = core.fetch(role="user")
    assert len(user_msgs) == 4
    assert all(m.role == "user" for m in user_msgs)

    assistant_msgs = core.fetch(role="assistant")
    assert len(assistant_msgs) == 2
    assert all(m.role == "assistant" for m in assistant_msgs)


def test_fetch_role_bad_name_returns_empty(chroma_tmp_path, sample_messages):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest_many(sample_messages)

    results = core.fetch(role="system")
    assert results == []


def test_fetch_metadata_filter(chroma_tmp_path):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest("user", "Doc A", metadata={"tag": "important", "project": "ea"})
    core.ingest("user", "Doc B", metadata={"tag": "normal", "project": "ea"})
    core.ingest("user", "Doc C", metadata={"tag": "important", "project": "oss"})

    tagged = core.fetch(metadata={"tag": "important"})
    assert len(tagged) == 2
    assert all(m.metadata.get("tag") == "important" for m in tagged)

    project = core.fetch(metadata={"project": "oss"})
    assert len(project) == 1
    assert project[0].content == "Doc C"


def test_store_and_fetch_roundtrip(chroma_tmp_path):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))

    memories = [
        Memory(id="m1", role="user", content="Store test 1", session_id="s1"),
        Memory(id="m2", role="assistant", content="Store test 2", session_id="s1"),
        Memory(id="m3", role="user", content="Store test 3", session_id="s2"),
    ]
    ids = core.store(memories)
    assert len(ids) == 3
    assert core.count() == 3

    fetched = core.fetch(session_id="s1")
    assert len(fetched) == 2
    assert {m.content for m in fetched} == {"Store test 1", "Store test 2"}


def test_store_empty_list(chroma_tmp_path):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    ids = core.store([])
    assert ids == []
    assert core.count() == 0


def test_fetch_all_paginates_internally(chroma_tmp_path):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    for i in range(50):
        core.ingest("user", f"Message number {i}")

    results = core.fetch_all()
    assert len(results) == 50
    assert isinstance(results[0], Memory)


def test_fetch_with_session_filter(chroma_tmp_path):
    core = MemoryCore(backend=ChromaBackend(path=chroma_tmp_path))
    core.ingest("user", "Session A message 1", session_id="A")
    core.ingest("user", "Session A message 2", session_id="A")
    core.ingest("user", "Session B message 1", session_id="B")

    a_msgs = core.fetch(session_id="A")
    assert len(a_msgs) == 2
    assert all(m.session_id == "A" for m in a_msgs)

    b_msgs = core.fetch(session_id="B")
    assert len(b_msgs) == 1
    assert b_msgs[0].session_id == "B"
