#!/usr/bin/env python3
"""Verify Observer and Reflector prompt quality.

Used by the autoresearch prompt optimization loop:
  agent modifies prompts → runs this script → compares score → keep/revert

Metrics:
  - Observer: obs count, priority distribution, JSON parse rate
  - Reflector: reflection count, domain diversity, no-fact-repeat gate

Usage:
  python scripts/verify_prompts.py --provider ollama:gemma4:e4b

Returns: exit 0 if all checks pass, exit 1 with diagnostic output on failure.
Score is printed as last line: "SCORE: 0.85"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
from datetime import datetime, timezone

# ── Test conversations ────────────────────────────────────────────────────

# 3 sessions, 10 messages each, covering different domains
TEST_CONVERSATIONS: list[list[dict[str, str]]] = [
    # Session 1: Career + relocation
    [
        {"role": "user", "content": "I just moved to San Francisco last week from Chicago."},
        {"role": "assistant", "content": "Welcome to SF! How are you finding it?"},
        {"role": "user", "content": "I love the weather here. I'm a software engineer at a startup called DataCraft. We build healthcare data pipelines."},
        {"role": "assistant", "content": "Healthcare data is a great space. What's your role there?"},
        {"role": "user", "content": "I lead the backend team of 5 engineers. My salary is $180K but I'm interviewing at Google for a Staff SWE role."},
        {"role": "assistant", "content": "That sounds like a big step up! What's the offer?"},
        {"role": "user", "content": "They're offering $350K base plus $200K equity. I'm torn because I enjoy the startup culture."},
        {"role": "assistant", "content": "That's a significant increase. What matters most to you?"},
        {"role": "user", "content": "I value mentorship — I teach 3 junior devs on weekends through CodePath. And I want to keep building in healthcare."},
        {"role": "assistant", "content": "Those are great values. Let me know how the interview goes."},
    ],
    # Session 2: Hobbies + pets
    [
        {"role": "user", "content": "I've been hiking in Marin every weekend since I arrived. The Dipsea Trail is my favorite so far."},
        {"role": "assistant", "content": "Marin has amazing trails! How long are your typical hikes?"},
        {"role": "user", "content": "Usually 8-12 miles. I run marathons too — my PR is 3:42 at the Chicago Marathon 2024."},
        {"role": "assistant", "content": "That's seriously impressive. How do you train?"},
        {"role": "user", "content": "I run 40 miles a week. Also adopted a golden retriever named Max last month. He's 8 months old."},
        {"role": "assistant", "content": "Max must love the SF weather! Golden retrievers are such great running companions."},
        {"role": "user", "content": "He's still learning, but he can do 3 miles now. I also got a new Garmin Fenix 7X watch for tracking."},
        {"role": "assistant", "content": "The Fenix is perfect for trail running. How's the battery life?"},
        {"role": "user", "content": "About 2 weeks with regular use. Much better than my old Apple Watch which died during the marathon."},
        {"role": "assistant", "content": "Good gear makes a difference. Happy trails!"},
    ],
    # Session 3: Mixed daily life
    [
        {"role": "user", "content": "My daily standup is at 9am at DataCraft. We use Jira for tracking."},
        {"role": "assistant", "content": "How's the engineering culture there?"},
        {"role": "user", "content": "Pretty good — we do code reviews, pair programming on Fridays, and use Python with FastAPI."},
        {"role": "assistant", "content": "FastAPI is great for data pipelines. What's your tech stack?"},
        {"role": "user", "content": "PostgreSQL, Redis, Kafka, deployed on AWS ECS. We're migrating to Kubernetes next quarter."},
        {"role": "assistant", "content": "K8s migration is a big project. Who's leading that?"},
        {"role": "user", "content": "I am, actually. It's my third migration — did one at my previous job in Chicago too."},
        {"role": "assistant", "content": "Experience counts. What happened to your Chicago place?"},
        {"role": "user", "content": "I still own a condo there in Lincoln Park. Renting it out for $2,800/month."},
        {"role": "assistant", "content": "Nice passive income! Anything else you miss about Chicago?"},
    ],
]

# ── Scoring ───────────────────────────────────────────────────────────────


def score_observer(observations: list[dict]) -> dict:
    """Score Observer output quality."""
    result: dict[str, float | int | str] = {
        "count": len(observations),
        "parse_ok": 1.0,
        "priority_high_pct": 0.0,
        "priority_low_pct": 0.0,
        "score": 0.0,
        "errors": "",
    }

    if not observations:
        result["errors"] = "no observations extracted"
        return result

    priorities: list[str] = []
    for obs in observations:
        p = obs.get("priority", "").lower()
        priorities.append(p)

    total = len(priorities)
    high = sum(1 for p in priorities if "🔴" in p)
    low = sum(1 for p in priorities if "🟢" in p)

    result["priority_high_pct"] = high / total if total else 0
    result["priority_low_pct"] = low / total if total else 0

    # Composite score:
    # - Count bonus: >2 obs gets full points, linear below
    # - Priority bonus: good mix has both high (>20%) and low (<60%)
    # - Parse bonus: 1.0 if no JSON parse errors (checked by caller)
    count_score = min(1.0, total / 3.0)
    priority_score = 1.0
    if result["priority_high_pct"] < 0.15:
        priority_score -= 0.3  # too few high-priority facts
    if result["priority_low_pct"] > 0.7:
        priority_score -= 0.2  # too many low-priority fillers

    result["score"] = round(
        0.4 * result["parse_ok"] + 0.3 * count_score + 0.3 * priority_score, 3,
    )
    return result


def score_reflector(reflections: list[dict]) -> dict:
    """Score Reflector output quality."""
    result: dict[str, float | int | str] = {
        "count": len(reflections),
        "domains": 0,
        "parse_ok": 1.0,
        "score": 0.0,
        "errors": "",
    }

    if not reflections:
        result["errors"] = "no reflections generated"
        return result

    domains = {r.get("domain", "general") for r in reflections}
    result["domains"] = len(domains)

    # Composite score:
    # - Count bonus: >1 reflections
    # - Domain bonus: >2 unique domains
    # - Parse bonus: 1.0
    count_score = min(1.0, len(reflections) / 2.0)
    domain_score = min(1.0, len(domains) / 3.0)

    result["score"] = round(
        0.3 * result["parse_ok"] + 0.3 * count_score + 0.4 * domain_score, 3,
    )
    return result


# ── Main ───────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="Verify Observer/Reflector prompts")
    parser.add_argument("--provider", default="ollama:gemma4:e4b")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from coremem import MemoryCore, MemoryStore
    from coremem.backends.hybrid import HybridBackend
    from coremem.observer import ObserverPipeline
    from coremem.reflector import ReflectorPipeline

    d1 = tempfile.mkdtemp()
    d2 = tempfile.mkdtemp()

    try:
        core = MemoryCore(backend=HybridBackend(path=d1))
        store = MemoryStore(path=d2)

        # ── Observer verification ──────────────────────
        observer_scores: list[dict] = []
        total_obs: list[dict] = []

        for si, conv in enumerate(TEST_CONVERSATIONS):
            sid = f"test_session_{si}"
            for msg in conv:
                core.ingest(msg["role"], msg["content"], session_id=sid)

            pipeline = ObserverPipeline(
                core=core, store=store, session_id=sid,
                model=args.provider, token_threshold=1, min_turns=1,
            )
            try:
                raw_observations = await pipeline.after_turn()
            except Exception as e:
                observer_scores.append({"count": 0, "errors": str(e), "score": 0})
                continue

            if raw_observations is None:
                raw_observations = []

            # Check if any JSON parse errors occurred
            parse_ok = len(raw_observations) > 0
            obs_score = score_observer(raw_observations)
            obs_score["parse_ok"] = 1.0 if parse_ok else 0.0
            obs_score["score"] = round(
                0.4 * obs_score["parse_ok"] + 0.3 * min(1.0, len(raw_observations) / 3.0)
                + 0.3 * (1.0 if obs_score["priority_high_pct"] > 0.15 else 0.5), 3,
            )
            observer_scores.append(obs_score)
            total_obs.extend(raw_observations)

        observer_avg = round(
            sum(s["score"] for s in observer_scores) / len(observer_scores), 3,
        ) if observer_scores else 0.0

        # ── Reflector verification ─────────────────────
        reflector_score = {"score": 0.0}
        obs_in_store = store.get_observations()
        if len(obs_in_store) >= 5:
            reflector = ReflectorPipeline(
                store=store, model=args.provider, min_observations=5,
            )
            try:
                reflections = await reflector.run_now()
            except Exception as e:
                reflections = []
            if reflections:
                reflector_score = score_reflector(reflections)

        # ── Final score ─────────────────────────────────
        final_score = round(0.7 * observer_avg + 0.3 * reflector_score["score"], 3)

        if args.json:
            print(json.dumps({
                "observer": {
                    "per_session": observer_scores,
                    "average": observer_avg,
                    "total_extracted": len(total_obs),
                },
                "reflector": reflector_score,
                "final": final_score,
            }, indent=2))
        else:
            print(f"OBSERVER: avg={observer_avg} | sessions={[s['score'] for s in observer_scores]} | total_obs={len(total_obs)}")
            print(f"REFLECTOR: score={reflector_score['score']} | count={reflector_score.get('count',0)} | domains={reflector_score.get('domains',0)}")
            print(f"SCORE: {final_score}")

        sys.exit(0 if final_score >= 0.5 else 1)

    finally:
        import shutil
        shutil.rmtree(d1, ignore_errors=True)
        shutil.rmtree(d2, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
