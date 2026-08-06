"""coremem — Zero-LLM memory for AI agents.

Usage:
    from coremem import MemoryCore

    core = MemoryCore(path="./memory")
    results = core.recall("How many model kits?")
    core.ingest("user", "I built a Spitfire model kit")
"""

from coremem.core import MemoryCore
from coremem.heuristics import SearchHeuristics
from coremem.query import decompose_queries, expand_queries
from coremem.rerank import rerank
from coremem.types import Memory, SearchResult, SessionBundle
from coremem.providers import create_provider

__version__ = "0.10.0"

__all__ = [
    "MemoryCore", "Memory", "SearchResult", "SessionBundle",
    "SearchHeuristics", "decompose_queries", "expand_queries", "rerank",
    "create_provider",
]
