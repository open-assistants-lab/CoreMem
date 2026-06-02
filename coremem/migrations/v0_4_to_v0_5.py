"""Schema migration from CoreMem 0.4.0 to 0.5.0.

Collapses observations + observation_metadata into a single
observations table. See docs/superpowers/specs/2026-06-02-observer-revision-design.md
section "Schema restructure" for the full rationale.

The migration is idempotent: running it twice on a 0.5.0 schema is
a no-op. This is achieved by checking the _schema_version table.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger("coremem.migrations.v0_4_to_v0_5")

TARGET_VERSION = "0.5.0"


def _is_already_migrated(conn: sqlite3.Connection) -> bool:
    """Check if the schema is already at 0.5.0."""
    try:
        row = conn.execute(
            "SELECT version FROM _schema_version WHERE version = ?",
            (TARGET_VERSION,),
        ).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        return False


def _has_0_4_0_schema(conn: sqlite3.Connection) -> bool:
    """Check if the database has the 0.4.0 split schema."""
    try:
        conn.execute("SELECT 1 FROM observation_metadata LIMIT 1")
        return True
    except sqlite3.OperationalError:
        return False


def migrate(db_path: str) -> None:
    """Run the 0.4.0 -> 0.5.0 schema migration.

    Steps:
    1. Open DB
    2. Check _schema_version; if at 0.5.0, return (idempotent)
    3. Check for 0.4.0 split schema; if not present, raise
    4. ADD COLUMN x5 to observations
    5. UPDATE observations SET importance, entities from observation_metadata
    6. DROP TABLE observation_metadata
    7. CREATE INDEX x5
    8. UPDATE _schema_version
    """
    conn = sqlite3.connect(db_path)
    try:
        if _is_already_migrated(conn):
            logger.info("Database already at version 0.5.0, skipping")
            return

        if not _has_0_4_0_schema(conn):
            raise RuntimeError(
                "Database does not have the 0.4.0 split schema. "
                "Cannot run 0.4.0 -> 0.5.0 migration."
            )

        logger.info("Starting 0.4.0 -> 0.5.0 schema migration")
        cur = conn.cursor()

        cur.execute("BEGIN")

        cur.execute(
            "ALTER TABLE observations ADD COLUMN kind TEXT NOT NULL DEFAULT 'fact'"
        )
        cur.execute(
            "ALTER TABLE observations ADD COLUMN source_fact_ids TEXT NOT NULL DEFAULT '[]'"
        )
        cur.execute(
            "ALTER TABLE observations ADD COLUMN importance REAL"
        )
        cur.execute(
            "ALTER TABLE observations ADD COLUMN entities TEXT NOT NULL DEFAULT '[]'"
        )
        cur.execute(
            "ALTER TABLE observations ADD COLUMN reflected INTEGER NOT NULL DEFAULT 0"
        )

        cur.execute("""
            UPDATE observations
            SET importance = (
                SELECT importance FROM observation_metadata
                WHERE observation_metadata.observation_id = observations.id
            ),
            entities = (
                SELECT entities FROM observation_metadata
                WHERE observation_metadata.observation_id = observations.id
            )
        """)

        cur.execute("DROP TABLE observation_metadata")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS _schema_version (
                version TEXT PRIMARY KEY,
                migrated_at TEXT NOT NULL
            )
        """)
        cur.execute(
            "INSERT INTO _schema_version (version, migrated_at) VALUES (?, datetime('now'))",
            (TARGET_VERSION,),
        )

        cur.execute("CREATE INDEX idx_observations_kind ON observations(kind)")
        cur.execute("CREATE INDEX idx_observations_user ON observations(user_id)")
        cur.execute("CREATE INDEX idx_observations_session ON observations(session_id)")
        cur.execute("CREATE INDEX idx_observations_reflected ON observations(reflected)")
        cur.execute("CREATE INDEX idx_observations_importance ON observations(importance)")

        cur.execute("COMMIT")
        logger.info("0.4.0 -> 0.5.0 migration complete")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
