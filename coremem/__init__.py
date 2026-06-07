"""coremem — Zero-LLM memory for AI agents.

Usage:
    from coremem import MemoryCore

    core = MemoryCore(path="./memory")
    results = core.search("How many model kits?")
    core.ingest("user", "I built a Spitfire model kit")
"""

from coremem.core import MemoryCore
from coremem.heuristics import SearchHeuristics, _mmr_diversify
from coremem.query import expand_queries
from coremem.rerank import rerank
from coremem.types import Memory, SearchQuery, SearchResult

__version__ = "0.7.1"

__all__ = [
    "MemoryCore", "Memory", "SearchResult", "SearchQuery",
    "SearchHeuristics", "expand_queries", "rerank", "_mmr_diversify",
]

try:
    from coremem.observer import Observer, ObserverPipeline
    from coremem.providers import create_provider
    from coremem.reflector import Reflector, ReflectorPipeline
    __all__.extend([
        "Observer", "ObserverPipeline", "Reflector", "ReflectorPipeline",
        "create_provider",
    ])
except ImportError:
    pass