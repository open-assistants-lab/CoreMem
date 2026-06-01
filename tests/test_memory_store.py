"""Tests for MemoryStore — schema migration for alignment columns."""

import tempfile

import pytest

from coremem.memory_store import MemoryStore


@pytest.fixture
def tmp_store():
    d = tempfile.mkdtemp()
    store = MemoryStore(path=d)
    yield store
    import shutil
    shutil.rmtree(d, ignore_errors=True)


def _list_columns(store, table: str) -> set[str]:
    """Helper: list column names in a table via PRAGMA."""
    rows = store._db.raw_query(f"PRAGMA table_info({table})")
    return {r["name"] for r in rows}


class TestMigration:
    def test_new_store_has_alignment_columns(self, tmp_store):
        columns = _list_columns(tmp_store, "observations")
        assert "alignment_tier" in columns
        assert "alignment_confidence" in columns

    def test_insert_observations_writes_alignment_fields(self, tmp_store):
        tmp_store.insert_observations([{
            "content": "test",
            "source_quote": "test",
            "importance": 0.5,
            "alignment_tier": "exact",
            "alignment_confidence": 1.0,
        }])
        obs = tmp_store.get_observations()
        assert len(obs) == 1
        assert obs[0]["alignment_tier"] == "exact"
        assert obs[0]["alignment_confidence"] == 1.0

    def test_migration_is_idempotent(self):
        d = tempfile.mkdtemp()
        MemoryStore(path=d)
        store2 = MemoryStore(path=d)
        columns = {r["name"] for r in store2._db.raw_query("PRAGMA table_info(observations)")}
        assert "alignment_tier" in columns
        assert "alignment_confidence" in columns
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    def test_existing_observations_get_default_alignment(self):
        d = tempfile.mkdtemp()
        store1 = MemoryStore(path=d)
        store1.insert_observations([{
            "content": "old fact",
            "source_quote": "old quote",
            "importance": 0.5,
        }])
        store2 = MemoryStore(path=d)
        obs = store2.get_observations()
        assert len(obs) == 1
        assert obs[0]["content"] == "old fact"
        import shutil
        shutil.rmtree(d, ignore_errors=True)
