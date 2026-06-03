"""Tests for the 0.4.0 -> 0.5.0 schema migration.

The 0.4.0 schema has TWO tables (observations + observation_metadata).
The 0.5.0 schema has ONE table (observations) with all columns from
both, plus new ones (kind, source_fact_ids, reflected, importance,
entities).

The migration:
1. Creates 0.4.0 schema (manually, for testing)
2. Inserts representative data
3. Runs the migration
4. Verifies the 0.5.0 schema
5. Verifies all user data is preserved
"""

from __future__ import annotations

import json
import sqlite3


def _create_0_4_0_schema(db_path: str) -> None:
    """Create the 0.4.0 schema and insert test data.

    This recreates what MemoryStore(path=...) would have done in 0.4.0.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
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
    cur.execute("""
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
    cur.execute("""
        INSERT INTO observations VALUES (
            'obs_1', 'User lives in Seattle', 'I live in Seattle',
            '2026-01', '2026-01-15T10:00:00', 'alice', NULL, 'sess_1',
            'EXACT', 1.0
        )
    """)
    cur.execute("""
        INSERT INTO observations VALUES (
            'obs_2', 'User prefers Rust', 'I prefer Rust over Python',
            '2026-02', '2026-02-10T14:00:00', 'alice', NULL, 'sess_2',
            'EXACT', 1.0
        )
    """)
    cur.execute("""
        INSERT INTO observation_metadata VALUES (
            'meta_1', 'obs_1', 0.85, '["Seattle"]', 'medium', 1.0, '2026-01-15T10:00:00'
        )
    """)
    cur.execute("""
        INSERT INTO observation_metadata VALUES (
            'meta_2', 'obs_2', 0.60, '["Rust", "Python"]', 'medium', 1.0, '2026-02-10T14:00:00'
        )
    """)
    conn.commit()
    conn.close()


def _column_names(db_path: str, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    names = {r[1] for r in rows}
    conn.close()
    return names


def _table_exists(db_path: str, table: str) -> bool:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    conn.close()
    return row is not None


class TestMigration:
    def test_adds_required_columns(self, tmp_path):
        """0.4.0 -> 0.5.0: observations gets +kind, +source_fact_ids,
        +importance, +entities, +reflected."""
        from coremem.migrations.v0_4_to_v0_5 import migrate

        db_path = str(tmp_path / "test.db")
        _create_0_4_0_schema(db_path)
        migrate(db_path)

        cols = _column_names(db_path, "observations")
        assert "kind" in cols
        assert "source_fact_ids" in cols
        assert "importance" in cols
        assert "entities" in cols
        assert "reflected" in cols

    def test_drops_observation_metadata_table(self, tmp_path):
        """The 0.4.0 observation_metadata table is dropped post-migration."""
        from coremem.migrations.v0_4_to_v0_5 import migrate

        db_path = str(tmp_path / "test.db")
        _create_0_4_0_schema(db_path)
        migrate(db_path)

        assert not _table_exists(db_path, "observation_metadata")

    def test_copies_importance_to_observations(self, tmp_path):
        """observation_metadata.importance values are copied to observations.importance."""
        from coremem.migrations.v0_4_to_v0_5 import migrate

        db_path = str(tmp_path / "test.db")
        _create_0_4_0_schema(db_path)
        migrate(db_path)

        conn = sqlite3.connect(db_path)
        rows = dict(conn.execute(
            "SELECT id, importance FROM observations"
        ).fetchall())
        conn.close()

        assert rows["obs_1"] == 0.85
        assert rows["obs_2"] == 0.60

    def test_copies_entities_to_observations(self, tmp_path):
        """observation_metadata.entities values are copied to observations.entities."""
        from coremem.migrations.v0_4_to_v0_5 import migrate

        db_path = str(tmp_path / "test.db")
        _create_0_4_0_schema(db_path)
        migrate(db_path)

        conn = sqlite3.connect(db_path)
        rows = dict(conn.execute(
            "SELECT id, entities FROM observations"
        ).fetchall())
        conn.close()

        assert json.loads(rows["obs_1"]) == ["Seattle"]
        assert json.loads(rows["obs_2"]) == ["Rust", "Python"]

    def test_orphan_observation_does_not_break_migration(self, tmp_path):
        """An observation with no matching observation_metadata row
        (e.g., from a crash between the two inserts in the 0.4.0
        Observer) should not break the migration. The entities column
        should keep its default '[]', and importance should be NULL."""
        from coremem.migrations.v0_4_to_v0_5 import migrate

        db_path = str(tmp_path / "test.db")

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("""
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
        cur.execute("""
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
        cur.execute("""
            INSERT INTO observations VALUES (
                'obs_1', 'User lives in Seattle', 'I live in Seattle',
                '2026-01', '2026-01-15T10:00:00', 'alice', NULL, 'sess_1',
                'EXACT', 1.0
            )
        """)
        cur.execute("""
            INSERT INTO observations VALUES (
                'obs_orphan', 'User is learning Rust', 'I am learning Rust',
                '2026-02', '2026-02-10T14:00:00', 'alice', NULL, 'sess_2',
                'EXACT', 1.0
            )
        """)
        cur.execute("""
            INSERT INTO observation_metadata VALUES (
                'meta_1', 'obs_1', 0.85, '["Seattle"]', 'medium', 1.0, '2026-01-15T10:00:00'
            )
        """)
        conn.commit()
        conn.close()

        migrate(db_path)

        conn = sqlite3.connect(db_path)
        rows = {
            r[0]: {"entities": r[1], "importance": r[2]}
            for r in conn.execute(
                "SELECT id, entities, importance FROM observations"
            ).fetchall()
        }
        conn.close()

        assert len(rows) == 2
        assert json.loads(rows["obs_1"]["entities"]) == ["Seattle"]
        assert rows["obs_1"]["importance"] == 0.85
        assert json.loads(rows["obs_orphan"]["entities"]) == []
        assert rows["obs_orphan"]["importance"] is None

    def test_preserves_user_data(self, tmp_path):
        """All user-data columns from 0.4.0 observations are preserved."""
        from coremem.migrations.v0_4_to_v0_5 import migrate

        db_path = str(tmp_path / "test.db")
        _create_0_4_0_schema(db_path)
        migrate(db_path)

        conn = sqlite3.connect(db_path)
        rows = list(conn.execute(
            "SELECT id, content, source_quote, user_id, alignment_tier, alignment_confidence "
            "FROM observations ORDER BY id"
        ).fetchall())
        conn.close()

        assert rows[0] == ("obs_1", "User lives in Seattle", "I live in Seattle", "alice", "EXACT", 1.0)
        assert rows[1] == ("obs_2", "User prefers Rust", "I prefer Rust over Python", "alice", "EXACT", 1.0)

    def test_creates_indexes(self, tmp_path):
        """All 5 indexes exist post-migration."""
        from coremem.migrations.v0_4_to_v0_5 import migrate

        db_path = str(tmp_path / "test.db")
        _create_0_4_0_schema(db_path)
        migrate(db_path)

        conn = sqlite3.connect(db_path)
        indexes = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()}
        conn.close()

        assert "idx_observations_kind" in indexes
        assert "idx_observations_user" in indexes
        assert "idx_observations_session" in indexes
        assert "idx_observations_reflected" in indexes
        assert "idx_observations_importance" in indexes

    def test_is_idempotent(self, tmp_path):
        """Running the migration twice does not error or duplicate data."""
        from coremem.migrations.v0_4_to_v0_5 import migrate

        db_path = str(tmp_path / "test.db")
        _create_0_4_0_schema(db_path)
        migrate(db_path)
        migrate(db_path)

        conn = sqlite3.connect(db_path)
        rows = list(conn.execute("SELECT id FROM observations").fetchall())
        conn.close()

        assert len(rows) == 2
