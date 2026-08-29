#!/usr/bin/env python3
"""Benchmark versioned vs non-versioned CoreMem stores (docs/versioned-memory-design.md §7).

The perf gate for flipping `versioned=True` to the default. Measures, on
CoreMem's real paths (not raw hybriddb CRUD), versioned=False vs True:

  - ingest_many throughput (batched path, N messages)
  - recall latency after ingest
  - storage after ingest
  - rollback_memory cost (append-heavy: remove 1k ingested rows)
  - verify_memory_chain runtime at chain length N

Usage:
  uv run python3 scripts/bench_versioned_memory.py [--rows 10000]
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from coremem import MemoryCore


def _messages(n: int, start: int = 0) -> list[dict]:
    msgs = []
    for i in range(start, start + n):
        msgs.append({
            "id": f"m{i}",
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"memory entry {i}: the user fixed the garden fence and later "
                       f"planted tomatoes near the shed. entry number {i}.",
            "session_id": f"s{i // 10}",
            "ts": (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i)).isoformat(),
        })
    return msgs


def _dir_size_mb(path: str | Path) -> float:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / (1024 * 1024)


def _build(versioned: bool) -> tuple[MemoryCore, str]:
    d = tempfile.mkdtemp()
    core = MemoryCore(path=str(Path(d) / "m"), versioned=versioned)
    return core, d


def bench_ingest(rows: int, rounds: int = 3) -> dict:
    arms = {}
    for versioned in (False, True):
        times = []
        for _ in range(rounds):
            core, d = _build(versioned)
            msgs = _messages(rows)
            t0 = time.perf_counter()
            for start in range(0, len(msgs), 500):
                core.ingest_many(msgs[start:start + 500])
            dt = time.perf_counter() - t0
            times.append(dt)
            if versioned:
                assert core.verify_memory_chain()["messages"]["valid"]
            core.close()
            shutil.rmtree(d, ignore_errors=True)
        med = statistics.median(times)
        arms["versioned" if versioned else "unversioned"] = {
            "median_s": round(med, 2), "rows_per_s": round(rows / med),
        }
    overhead = arms["versioned"]["median_s"] / arms["unversioned"]["median_s"] - 1
    return {"arms": arms, "overhead_pct": round(overhead * 100, 1)}


def bench_recall_and_storage(rows: int) -> dict:
    arms = {}
    for versioned in (False, True):
        core, d = _build(versioned)
        for start in range(0, rows, 500):
            core.ingest_many(_messages(500, start=start))
        core.recall("garden fence tomatoes shed", strategy="episodic", limit=5)  # warmup (CE load)
        t0 = time.perf_counter()
        runs = 10
        for _ in range(runs):
            core.recall("garden fence tomatoes shed", strategy="episodic", limit=5)
        recall_ms = (time.perf_counter() - t0) / runs * 1000
        arms["versioned" if versioned else "unversioned"] = {
            "recall_ms": round(recall_ms, 1),
            "storage_mb": round(_dir_size_mb(d), 1),
        }
        core.close()
        shutil.rmtree(d, ignore_errors=True)
    ratio = arms["versioned"]["storage_mb"] / max(arms["unversioned"]["storage_mb"], 1e-9)
    return {"arms": arms, "storage_ratio_vs_unversioned": round(ratio, 2)}


def bench_rollback_and_verify(rows: int) -> dict:
    """Versioned arm only: rollback cost (remove the last 1k ingested rows)
    and verify_chain runtime at chain length ~rows."""
    core, d = _build(versioned=True)
    bulk = _messages(rows)
    for start in range(0, len(bulk), 500):
        core.ingest_many(bulk[start:start + 500])
    core.checkpoint_memory("pre-extra")
    extra = _messages(1000, start=rows)
    t_ing = time.perf_counter()
    for start in range(0, len(extra), 500):
        core.ingest_many(extra[start:start + 500])
    ingest_1k_s = time.perf_counter() - t_ing
    count_before = core.count()
    t0 = time.perf_counter()
    result = core.rollback_memory("pre-extra")
    rollback_s = time.perf_counter() - t0
    assert core.count() == rows, f"rollback restored {core.count()} != {rows}"
    t0 = time.perf_counter()
    chain = core.verify_memory_chain()
    verify_s = time.perf_counter() - t0
    assert chain["messages"]["valid"]
    out = {
        "rollback_1k_rows_s": round(rollback_s, 2),
        "ingest_1k_rows_s_for_reference": round(ingest_1k_s, 2),
        "verify_chain_s": round(verify_s, 2),
        "chain_len": chain["messages"]["checked"],
        "restored_count": count_before - rows,
    }
    core.close()
    shutil.rmtree(d, ignore_errors=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    print(f"ingest benchmark ({args.rows} rows, median of 3)...")
    ingest = bench_ingest(args.rows)
    print("recall + storage benchmark...")
    recall = bench_recall_and_storage(args.rows)
    print("rollback + verify benchmark...")
    rb = bench_rollback_and_verify(args.rows)
    result = {"ingest": ingest, "recall_storage": recall, "rollback_verify": rb}
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())