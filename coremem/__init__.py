"""coremem — Zero-LLM memory for AI agents.

Usage:
    from coremem import MemoryCore

    core = MemoryCore(path="./memory")
    results = core.search_messages("How many model kits?")
    core.ingest("user", "I built a Spitfire model kit")
"""

from coremem.core import MemoryCore
from coremem.heuristics import SearchHeuristics, _mmr_diversify
from coremem.query import decompose_queries, expand_queries
from coremem.rerank import rerank
from coremem.types import Memory, SearchQuery, SearchResult, SessionBundle
from coremem.providers import create_provider

__version__ = "0.10.0"

__all__ = [
    "MemoryCore", "Memory", "SearchResult", "SearchQuery", "SessionBundle",
    "SearchHeuristics", "decompose_queries", "expand_queries", "rerank", "_mmr_diversify",
    "create_provider",
]
