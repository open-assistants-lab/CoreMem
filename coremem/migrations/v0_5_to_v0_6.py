"""Migrate observations table from v0.5 to v0.6 schema.

Adds: memory_type, durability, sensitivity, status, confidence,
      valid_from, valid_to, superseded_by, source_message_ids, embedding.
Creates: memory_events, memory_conflicts tables.
"""
from __future__ import annotations

from typing import Any


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

    # Create new tables from memory_store schemas
    from coremem.memory_store import _MEMORY_EVENTS_SCHEMA, _MEMORY_CONFLICTS_SCHEMA

    if "memory_events" not in db.list_tables():
        db.create_table("memory_events", _MEMORY_EVENTS_SCHEMA)
        db.raw_query(
            "CREATE INDEX IF NOT EXISTS idx_memory_events_memory "
            "ON memory_events(memory_id)"
        )

    if "memory_conflicts" not in db.list_tables():
        db.create_table("memory_conflicts", _MEMORY_CONFLICTS_SCHEMA)
        db.raw_query(
            "CREATE INDEX IF NOT EXISTS idx_memory_conflicts_status "
            "ON memory_conflicts(resolution_status)"
        )
