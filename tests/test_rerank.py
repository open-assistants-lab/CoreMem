"""Tests for cross-encoder reranking."""

from __future__ import annotations

import sys
import threading
import types


def test_get_cross_encoder_loads_once_under_concurrency(monkeypatch):
    """Concurrent first calls must load the model exactly once.

    Without a lock around the lazy init, N threads each see _cross_encoder
    is None and each load the ~500 MB model (wasteful, can OOM).
    """
    rerank_mod = sys.modules["coremem.rerank"]

    import time

    instantiated: list[int] = []
    instantiated_lock = threading.Lock()

    class _FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            # Widen the race window so un-locked code reliably lets every
            # thread pass the `is None` check before the first finishes.
            time.sleep(0.05)
            with instantiated_lock:
                instantiated.append(1)

        def predict(self, pairs, **kwargs):
            return [1.0] * len(pairs)

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    monkeypatch.setattr(rerank_mod, "_cross_encoder", None)

    results: list[object] = []

    def worker() -> None:
        results.append(rerank_mod.get_cross_encoder())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(instantiated) == 1, f"model loaded {len(instantiated)} times"
    assert len(results) == 8
    assert all(result is not None for result in results)
