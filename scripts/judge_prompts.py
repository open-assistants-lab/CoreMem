#!/usr/bin/env python3
"""LLM judge for Observer/Reflector prompt quality.

Uses a separate LLM (the "judge") to evaluate how well the Observer and
Reflector prompts extract meaningful information from fixed test conversations.

Runs each pipeline, collects output, then asks the judge to rate quality
on 5 dimensions (1-10 scale). Composite score = weighted average.

Usage:
  python scripts/judge_prompts.py --provider ollama:gemma4:e4b --judge ollama:gemma4:e4b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from typing import Any

from coremem import MemoryCore, MemoryStore
from coremem.backends.hybrid import HybridBackend
from coremem.observer_utils import chat_messages, parse_json_array
from coremem.providers import create_provider

# Same test conversations as verify_prompts.py
TEST_CONVERSATIONS: list[list[dict[str, str]]] = [
    [
        {"role": "user", "content": "I just moved to San Francisco last week from Chicago. I'm a software engineer at DataCraft."},
        {"role": "assistant", "content": "Welcome to SF!"},
        {"role": "user", "content": "I lead the backend team of 5 engineers. Salary $180K but Google offered $350K base + $200K equity."},
        {"role": "assistant", "content": "That's a big step up!"},
        {"role": "user", "content": "I teach 3 junior devs on weekends through CodePath. Value mentorship over money."},
        {"role": "assistant", "content": "Those are great values."},
    ],
    [
        {"role": "user", "content": "I've been hiking Marin every weekend. Dipsea Trail is my favorite. Run marathons — PR 3:42 at Chicago 2024."},
        {"role": "assistant", "content": "Impressive! How do you train?"},
        {"role": "user", "content": "40 miles a week. Adopted a golden retriever Max last month, 8 months old. Got a Garmin Fenix 7X."},
        {"role": "assistant", "content": "Max must love SF! Good gear choice."},
        {"role": "user", "content": "He can do 3 miles now. Battery lasts 2 weeks — way better than my old Apple Watch."},
        {"role": "assistant", "content": "Happy trails!"},
    ],
    [
        {"role": "user", "content": "Daily standup at 9am. We use Jira, Python, FastAPI on AWS ECS. Migrating to Kubernetes next quarter."},
        {"role": "assistant", "content": "Who's leading the migration?"},
        {"role": "user", "content": "I am — my third K8s migration. Did one in Chicago too. Still own a condo there, renting for $2,800/month."},
        {"role": "assistant", "content": "Nice passive income!"},
        {"role": "user", "content": "Condo is in Lincoln Park. My tenant is a Northwestern med student. Never missed a payment in 2 years."},
        {"role": "assistant", "content": "Sounds like a great setup."},
    ],
]

JUDGE_OBSERVER_PROMPT = """You are a strict LLM judge. Evaluate the following Observer output.

The Observer extracted facts from a conversation. Rate the output on these dimensions (1-10):

1. COMPLETENESS: Did it capture ALL key facts? (names, numbers, locations, roles, preferences)
2. PRECISION: Are facts EXACT (not paraphrased)? No hallucinated values.
3. PRIORITY: Are priorities correct? 🔴 for precise values (names/$/numbers), 🟡 for preferences, 🟢 for context.
4. NO TRIVIA: Did it skip greetings, meta-commentary, filler?
5. CONCISENESS: One fact per observation. No combined facts. Concise wording.

Output:
- Score for each dimension (1-10)
- Brief justification (one sentence each)
- Final composite score (average)

Conversation:
{conversation}

Observer Output:
{observations}

Return ONLY a JSON object with keys: completeness, precision, priority, no_trivia, conciseness, justification, composite. No markdown wrapping."""

JUDGE_REFLECTOR_PROMPT = """You are a strict LLM judge. Evaluate the following Reflector output.

The Reflector discovered patterns from observations. Rate the output on these dimensions (1-10):

1. INSIGHT: Are these real patterns, not just restated facts?
2. DOMAIN: Are domain labels accurate?
3. DIVERSITY: Do reflections cover different domains (not all "career")?
4. DEPTH: Do they go beyond surface ("lives in SF") to meaning ("values outdoor access over commute time")?
5. CONTRADICTIONS: Are contradictions noted where appropriate?

Output:
- Score for each dimension (1-10)
- Brief justification (one sentence each)
- Final composite score (average)

Observations (context):
{observations}

Reflector Output:
{reflections}

Return ONLY a JSON object with keys: insight, domain, diversity, depth, contradictions, justification, composite. No markdown wrapping."""


async def evaluate_prompts(provider: str, judge_provider: str) -> dict[str, Any]:
    from coremem.observer import ObserverPipeline
    from coremem.reflector import ReflectorPipeline

    d1 = tempfile.mkdtemp()
    d2 = tempfile.mkdtemp()

    try:
        core = MemoryCore(backend=HybridBackend(path=d1))
        store = MemoryStore(path=d2)
        judge = create_provider(judge_provider)

        # ── Observer evaluation ──────────────────────
        all_observations: list[dict] = []
        for si, conv in enumerate(TEST_CONVERSATIONS):
            sid = f"judge_session_{si}"
            for msg in conv:
                core.ingest(msg["role"], msg["content"], session_id=sid)

            pipeline = ObserverPipeline(
                core=core, store=store, session_id=sid,
                model=provider, token_threshold=1, min_turns=1,
            )
            raw = await pipeline.after_turn()
            if raw:
                all_observations.extend(raw)

        # Judge the observer output
        obs_text = "\n".join(
            f"- [{o.get('priority', '?')}] {o.get('content', '')}"
            for o in all_observations
        )
        conv_text = "\n".join(
            f"[{m['role']}] {m['content']}"
            for conv in TEST_CONVERSATIONS for m in conv
        )
        obs_prompt = JUDGE_OBSERVER_PROMPT.format(
            conversation=conv_text, observations=obs_text or "(none)"
        )
        obs_response = await judge.chat(chat_messages("", obs_prompt))
        obs_result = parse_json_array(obs_response.content)
        observer_score = obs_result[0] if obs_result else {"composite": 0}

        # ── Reflector evaluation ─────────────────────
        obs_in_store = store.get_observations()
        reflector_score = {"composite": 0}

        if len(obs_in_store) >= 5:
            reflector_pipeline = ReflectorPipeline(
                store=store, model=provider, min_observations=5,
            )
            reflections = await reflector_pipeline.run_now()
            if reflections:
                ref_text = "\n".join(
                    f"- [{r.get('domain', '?')}] {r.get('content', '')}"
                    for r in reflections
                )
                ref_prompt = JUDGE_REFLECTOR_PROMPT.format(
                    observations=obs_text, reflections=ref_text,
                )
                ref_response = await judge.chat(chat_messages("", ref_prompt))
                ref_result = parse_json_array(ref_response.content)
                reflector_score = ref_result[0] if ref_result else {"composite": 0}

        # ── Composite ──────────────────────────────────
        obs_comp = float(observer_score.get("composite", 0))
        ref_comp = float(reflector_score.get("composite", 0))
        final = round(0.6 * obs_comp + 0.4 * ref_comp, 2)

        return {
            "observer": observer_score,
            "reflector": reflector_score,
            "composite": final,
            "observation_count": len(all_observations),
        }

    finally:
        import shutil
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="ollama:gemma4:e4b", help="Model for Observer/Reflector")
    parser.add_argument("--judge", default="ollama:gemma4:e4b", help="Model for judging")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = await evaluate_prompts(args.provider, args.judge)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"OBSERVER: {json.dumps(result['observer'], indent=2)}")
        print(f"REFLECTOR: {json.dumps(result['reflector'], indent=2)}")
        print(f"COMPOSITE: {result['composite']}/10 ({result['observation_count']} obs)")
        print(f"PROMPT_SCORE: {result['composite']}")


if __name__ == "__main__":
    asyncio.run(main())
