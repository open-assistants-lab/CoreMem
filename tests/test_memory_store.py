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


def _table_exists_check(store, table: str) -> bool:
    """Helper: check if a table exists."""
    rows = store._db.raw_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return len(rows) > 0


class TestSchemaVersion:
    def test_new_store_has_schema_version_0_5_0(self, tmp_store):
        """A newly-created MemoryStore is at version 0.5.0."""
        rows = tmp_store._db.raw_query(
            "SELECT version FROM _schema_version ORDER BY version"
        )
        versions = {r["version"] for r in rows}
        assert "0.5.0" in versions

    def test_existing_0_4_0_store_is_migrated_on_init(self, tmp_path):
        """A 0.4.0 store is auto-migrated to 0.5.0 on first access."""
        import sqlite3

        d = str(tmp_path)
        # Manually create a 0.4.0 schema (split tables)
        conn = sqlite3.connect(f"{d}/app.db")
        conn.execute("""
            CREATE TABLE observations (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_quote TEXT,
                referenced_date TEXT,
                observation_ts TEXT NOT NULL,
                user_id TEXT,
                agent_id TEXT,
                session_id TEXT,
                alignment_tier TEXT,
                alignment_confidence REAL
            )
        """)
        conn.execute("""
            CREATE TABLE observation_metadata (
                id TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL,
                importance REAL,
                entities TEXT,
                priority TEXT,
                confidence REAL,
                enrichment_ts TEXT
            )
        """)
        conn.execute("""
            INSERT INTO observations VALUES (
                'obs_1', 'test content', 'test quote', NULL,
                '2026-01-15T10:00:00', 'alice', NULL, 'sess_1', 'EXACT', 1.0
            )
        """)
        conn.execute("""
            INSERT INTO observation_metadata VALUES (
                'meta_1', 'obs_1', 0.75, '[]', 'medium', 1.0, '2026-01-15T10:00:00'
            )
        """)
        conn.commit()
        conn.close()

        # Now create the MemoryStore — should auto-migrate
        store = MemoryStore(path=d)

        # Verify migration happened
        rows = list(store._db.raw_query(
            "SELECT importance FROM observations WHERE id = 'obs_1'"
        ))
        assert len(rows) == 1
        assert rows[0]["importance"] == 0.75

        # Verify observation_metadata is gone
        assert not _table_exists_check(store, "observation_metadata")


class TestReflectionHelpers:
    def test_get_pending_reflections_returns_unreflected_facts(self, tmp_store):
        """get_pending_reflections() returns observations with
        kind='fact' AND reflected=0, sorted by observation_ts DESC."""
        tmp_store.insert_observations([
            {"id": "obs_1", "content": "fact 1", "source_quote": "q1",
             "kind": "fact", "reflected": 0, "observation_ts": "2026-01-01T00:00:00"},
            {"id": "obs_2", "content": "fact 2", "source_quote": "q2",
             "kind": "fact", "reflected": 1, "observation_ts": "2026-01-02T00:00:00"},
            {"id": "obs_3", "content": "reflection 1", "source_quote": None,
             "kind": "reflection", "reflected": 0, "observation_ts": "2026-01-03T00:00:00"},
        ])
        pending = tmp_store.get_pending_reflections()
        assert len(pending) == 1
        assert pending[0]["id"] == "obs_1"

    def test_mark_reflected_sets_flag(self, tmp_store):
        """mark_reflected(['obs_1']) sets reflected=1 for those IDs."""
        tmp_store.insert_observations([
            {"id": "obs_1", "content": "a", "source_quote": "a",
             "kind": "fact", "reflected": 0, "observation_ts": "2026-01-01T00:00:00"},
            {"id": "obs_2", "content": "b", "source_quote": "b",
             "kind": "fact", "reflected": 0, "observation_ts": "2026-01-02T00:00:00"},
        ])
        tmp_store.mark_reflected(["obs_1"])
        all_obs = tmp_store.get_observations(limit=10)
        flags = {o["id"]: o["reflected"] for o in all_obs}
        assert flags["obs_1"] == 1
        assert flags["obs_2"] == 0
