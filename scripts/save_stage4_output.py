#!/usr/bin/env python3
"""Compile LongMemEval sessions with LLM compiler and save pages to disk."""

import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from coremem.agent_journal import AgentJournalBundle, AgentJournalLLMCompiler
from eval_memorypack_longmemeval import load_longmemeval_instances, prepare_instances, build_reference_bundle


async def main():
    data_path = Path("data/longmemeval_8_remaining_subset.json")
    out_dir = Path("stage4_output")
    out_dir.mkdir(exist_ok=True)

    raw = load_longmemeval_instances(data_path, limit=1)
    prepared, truth = prepare_instances(raw)

    tmpdir = Path(tempfile.mkdtemp())
    bundle = build_reference_bundle(tmpdir, prepared)

    compiler = AgentJournalLLMCompiler(bundle, model="ollama-cloud:deepseek-v4-flash")
    sem = asyncio.Semaphore(8)

    async def compile_one(session):
        text = (bundle.turns_dir / f"{session.turn_id}.md").read_text()
        match = re.search(r"```json agent_journal-turn\n(.*?)\n```", text, re.DOTALL)
        payload = json.loads(match.group(1))
        msgs = payload["messages"]
        async with sem:
            result = await compiler.compile_session(
                turn_id=session.turn_id,
                session_id=session.session_id,
                messages=msgs,
            )
            for p in result.written_pages:
                dest = out_dir / p.name
                dest.write_text(p.read_text())
                print(f"  saved {dest.name}")

    tasks = [compile_one(s) for s in prepared[0].sessions]
    await asyncio.gather(*tasks)

    print(f"\nDone. {len(list(out_dir.glob('*.md')))} pages in {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
