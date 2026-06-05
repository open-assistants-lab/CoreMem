"""Migrate observations table from v0.5 to v0.6 schema.

Adds: memory_type, durability, sensitivity, status, confidence,
      valid_from, valid_to, superseded_by, source_message_ids, embedding.
Creates: observation_events, observation_conflicts, reflections tables.
"""
from __future__ import annotations

from typing import Any

from coremem.core import _OBSERVATION_EVENTS_SCHEMA, _OBSERVATION_CONFLICTS_SCHEMA, _REFLECTIONS_SCHEMA


def migrate(db: Any, db_path: str) -> None:
    """Add new columns and create new tables."""
    new_columns = [
        ("source_message_ids", "TEXT DEFAULT '[]'"),
        ("confidence", "REAL DEFAULT 0.800"),
        ("memory_type", "TEXT"),
        ("durability", "TEXT DEFAULT 'durable'"),
        ("sensitivity", "TEXT DEFAULT 'normal'"),
        ("status", "TEXT DEFAULT 'candidate'"),
        ("valid_from", "TEXT"),
        ("valid_to", "TEXT"),
        ("superseded_by", "TEXT"),
        ("embedding", "TEXT"),
    ]
    for col_name, col_def in new_columns:
        try:
            db.raw_query(
                f"ALTER TABLE observations ADD COLUMN {col_name} {col_def}"
            )
        except Exception:
            pass  # Column already exists

    if "observation_events" not in db.list_tables():
        db.create_table("observation_events", _OBSERVATION_EVENTS_SCHEMA)
        db.raw_query(
            "CREATE INDEX IF NOT EXISTS idx_observation_events_obs "
            "ON observation_events(observation_id)"
        )

    if "observation_conflicts" not in db.list_tables():
        db.create_table("observation_conflicts", _OBSERVATION_CONFLICTS_SCHEMA)
        db.raw_query(
            "CREATE INDEX IF NOT EXISTS idx_observation_conflicts_status "
            "ON observation_conflicts(resolution_status)"
        )

    if "reflections" not in db.list_tables():
        db.create_table("reflections", _REFLECTIONS_SCHEMA)
