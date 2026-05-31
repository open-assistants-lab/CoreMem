"""Tests for ChromaBackend."""

from coremem.backends.chroma import ChromaBackend
from coremem.types import Memory, SearchQuery


def test_ingest_and_count(chroma_tmp_path):
    be = ChromaBackend(path=chroma_tmp_path)
    be.ingest(Memory(id="m1", content="Hello world", role="user"))
    be.ingest(Memory(id="m2", content="I like pizza", role="user"))
    assert be.count() == 2


def test_search_returns_results(chroma_tmp_path):
    be = ChromaBackend(path=chroma_tmp_path)
    be.ingest(Memory(id="m1", content="I love building model kits", role="user"))
    be.ingest(Memory(id="m2", content="My favorite food is pizza", role="user"))

    results = be.search(SearchQuery(text="model kits", limit=5))
    assert len(results) > 0
    assert any("model" in r.memory.content.lower() for r in results)


def test_search_filters_by_metadata(chroma_tmp_path):
    be = ChromaBackend(path=chroma_tmp_path)
    be.ingest(Memory(id="m1", content="I like chess", role="user", metadata={"topic": "hobbies"}))
    be.ingest(Memory(id="m2", content="I enjoy painting", role="user", metadata={"topic": "arts"}))

    results = be.search(SearchQuery(text="hobby", limit=5, metadata={"topic": "hobbies"}))
    assert len(results) > 0
    assert all("chess" in r.memory.content.lower() for r in results)

    results_all = be.search(SearchQuery(text="hobby", limit=5))
    assert len(results_all) >= 2


def test_get_recent(chroma_tmp_path):
    be = ChromaBackend(path=chroma_tmp_path)
    for i in range(5):
        be.ingest(Memory(id=f"m{i}", content=f"Memory {i}", role="user"))
    recent = be.get_recent(limit=3)
    assert len(recent) <= 3


def test_clear(chroma_tmp_path):
    be = ChromaBackend(path=chroma_tmp_path)
    be.ingest(Memory(id="m1", content="test", role="user"))
    assert be.count() == 1
    be.clear()
    assert be.count() == 0
