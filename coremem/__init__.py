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

__version__ = "0.12.3"

__all__ = [
    "MemoryCore", "Memory", "SearchResult", "SessionBundle",
    "SearchHeuristics", "decompose_queries", "expand_queries", "rerank",
    "create_provider", "get_core",
]


def get_core(path: str | None = None) -> MemoryCore:
    """Create a MemoryCore from path/env/default.

    Resolution: path arg > COREMEM_PATH env > ~/.coremem/hybrid
    """
    import os
    resolved = path or os.environ.get("COREMEM_PATH") or os.path.expanduser("~/.coremem/hybrid")
    model_string = os.environ.get("COREMEM_LLM_MODEL")
    llm_provider = None
    kwargs: dict[str, str] = {}
    if model_string:
        llm_provider = create_provider(model_string)
        kwargs["agent_journal_model"] = model_string
    return MemoryCore(path=resolved, llm_provider=llm_provider, **kwargs)
