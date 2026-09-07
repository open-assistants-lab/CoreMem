# coremem/ontology.py
"""Fact-layer attribute ontology — a small closed vocabulary, not a knowledge graph.

Cardinality is an ontological property you cannot learn from data:
single-valued attributes supersede on update; multi-valued ones append.
Declared once in ontology.toml; the fact layer enforces it deterministically.
Traversal stays out (falsified twice — docs/retrieval-experiments.md).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttributeDef:
    name: str
    cardinality: str = "multi"  # "single" | "multi"
    value_type: str = "text"    # "text" | "number" | "date"


# Starter vocabulary for common personal-memory attributes. Extend via
# ontology.toml; unknown attributes default to multi/text (safe: append only).
STARTER_ATTRIBUTES: dict[str, dict] = {
    "employer": {"cardinality": "single"},
    "job_title": {"cardinality": "single"},
    "home_city": {"cardinality": "single"},
    "partner": {"cardinality": "single"},
    "birthday": {"cardinality": "single", "value_type": "date"},
    "child": {"cardinality": "multi"},
    "pet": {"cardinality": "multi"},
    "preference": {"cardinality": "multi"},
    "allergy": {"cardinality": "multi"},
}


class FactOntology:
    def __init__(self, config: dict | None = None, config_path: str | Path | None = None):
        # Merge order: starter set <- user TOML <- user dict.
        # TOML shape: [attribute.name] with cardinality/value_type keys.
        loaded: dict = {}
        if config_path is not None:
            with open(config_path, "rb") as fh:
                loaded = tomllib.load(fh).get("attribute", {})
        elif config:
            loaded = config.get("attribute", config)
        merged = {**STARTER_ATTRIBUTES, **(loaded or {})}
        self._attributes: dict[str, AttributeDef] = {}
        for name, spec in merged.items():
            spec = spec if isinstance(spec, dict) else {}
            self._attributes[name] = AttributeDef(
                name=name,
                cardinality=spec.get("cardinality", "multi"),
                value_type=spec.get("value_type", "text"),
            )

    def attribute(self, name: str) -> AttributeDef:
        return self._attributes.get(name, AttributeDef(name=name))