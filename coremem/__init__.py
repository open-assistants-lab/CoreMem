"""coremem — Zero-LLM memory for AI agents.

Default backend (0.5.0+): HybridBackend (HybridDB — SQLite+FTS5+ChromaDB)
Legacy: ChromaBackend (pure ChromaDB, emits DeprecationWarning)

Usage:
    from coremem import MemoryCore
    from coremem.backends.hybrid import HybridBackend

    core = MemoryCore(backend=HybridBackend(path="./memory"))
    results = core.search("How many model kits?")
    context = core.wakeup(user_id="alice")
    core.ingest("user", "I built a Spitfire model kit")
"""

from coremem.core import MemoryCore
from coremem.heuristics import SearchHeuristics, _mmr_diversify
from coremem.query import expand_queries
from coremem.rerank import rerank
from coremem.types import Memory, SearchQuery, SearchResult

__version__ = "0.3.0"

__all__ = [
    "MemoryCore", "Memory", "SearchResult", "SearchQuery",
    "SearchHeuristics", "expand_queries", "rerank", "_mmr_diversify",
]

# Conditionally export observer/reflector (requires httpx)
try:
    from coremem.memory_store import MemoryStore  # noqa: F401
    from coremem.observer import Observer, ObserverPipeline  # noqa: F401
    from coremem.providers import create_provider  # noqa: F401
    from coremem.reflector import Reflector, ReflectorPipeline  # noqa: F401
    __all__.extend([
        "Observer", "ObserverPipeline", "Reflector", "ReflectorPipeline",
        "MemoryStore", "create_provider",
    ])
except ImportError:
    pass
