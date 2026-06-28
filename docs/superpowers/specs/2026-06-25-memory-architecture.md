# Memory Architecture: Human-Inspired Memory for Agents

## How Human Memory Works

```
Sensory → Working Memory (context window) → Long-term Memory
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    ▼                           ▼                           ▼
            Episodic Memory              Semantic Memory            Procedural Memory
        (what happened when)           (what is true)            (how to do things)
        • Specific events              • Facts & concepts         • Skills & routines
        • Autobiographical             • General knowledge        • Procedures
        • "I discussed fracking        • "User prefers           • "User wants concise
           on June 10th"                 concise answers"           answers with bullet
                                                                     points for technical
                                                                     questions"
```

**Key insight:** Human memory consolidates over time. An episodic memory (specific conversation) → repeated → becomes semantic (fact about the user) → repeated → becomes procedural (how the user wants things done). Different types of memory need different storage and retrieval.

## How Agent Memory Should Work

```
Agent Conversation
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│                    WORKING MEMORY (context window)           │
│  Current conversation + top-5 retrieved memories             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼ (background consolidation)
┌─────────────────────────────────────────────────────────────┐
│                    LONG-TERM MEMORY                          │
│                                                              │
│  ┌─────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   EPISODIC      │  │  SEMANTIC    │  │  PROCEDURAL      │  │
│  │  (what happened)│  │  (what is    │  │  (how to do)     │  │
│  │                 │  │   true)      │  │                  │  │
│  │  pages/         │  │  facts/      │  │  workflows/      │  │
│  │  daily/         │  │  concepts/   │  │  preferences/    │  │
│  │  weekly/        │  │  beliefs/    │  │  decisions/      │  │
│  │  monthly/       │  │              │  │                  │  │
│  └─────────────────┘  └──────────────┘  └─────────────────┘  │
│                                                              │
│  Consolidation: Episodic → Semantic → Procedural             │
│  (driven by repetition, importance, recency)                 │
└─────────────────────────────────────────────────────────────┘
```

## Proposed Memory Format

### Episodic Memory (what happened)

```markdown
---
type: episode
id: turn_2026-06-25_001
created: 2026-06-25T10:30:00Z
importance: 0.7
tags: [fracking, environment, marcellus-shale]
---

# Fracking and Groundwater: Debate on Ban

User asked about fracking's impact on groundwater in Marcellus Shale.
Assistant presented conflicting research, monitoring measures, and accountability mechanisms.
User advocated for a complete ban.

## Claims

- Fracking's effect on groundwater is uncertain (conflicting research)
- EPA requires pre/post-drilling monitoring
- Pennsylvania requires baseline testing
- User distrusts self-monitoring by fracking companies
- User advocates for a ban

## Citations

[1] User: "How has fracking affected the groundwater quality..."
[2] Assistant: "There is conflicting research on the effect..."
```

### Semantic Memory (what is true)

```markdown
---
type: fact
id: fact_user_stance_fracking
created: 2026-06-25T10:30:00Z
updated: 2026-06-25T10:30:00Z
confidence: 0.8
sources: [turn_2026-06-25_001, turn_2026-06-20_003]
tags: [user-stance, fracking, environment]
---

# User Stance on Fracking

The user strongly opposes fracking and advocates for a complete ban.
They distrust industry self-monitoring and want government enforcement.

## Evidence

- Turn 001: "I don't trust the fracking companies to properly monitor"
- Turn 003: "we need to completely ban fracking to protect our water sources"
```

### Procedural Memory (how to do things)

```markdown
---
type: procedure
id: pref_communication_style
created: 2026-06-20T10:00:00Z
updated: 2026-06-25T10:30:00Z
confidence: 0.9
sources: [turn_2026-06-25_001, turn_2026-06-20_003, turn_2026-06-15_007]
tags: [preference, communication]
---

# Communication Preferences

The user prefers:
- Concise answers with bullet points for technical questions
- Concrete examples over abstract explanations
- Follow-up questions that dig deeper into the topic
```

## Consolidation Pipeline

```
Episodic (turn) ──► Semantic (fact) ──► Procedural (preference)
     │                   │                      │
     │ Repeated          │ Repeated              │
     │ across 3+ turns   │ across 5+ facts       │
     ▼                   ▼                      ▼
  Daily summary      Confidence ↑           Confidence ↑
```

The consolidation runs as a background process:
1. After every N turns, scan episodic memory for repeated patterns
2. If a pattern appears 3+ times, promote to semantic memory
3. If a semantic fact is reinforced 5+ times, promote to procedural memory
4. Decay: memories not accessed for 30 days lose confidence, eventually archived

## Retrieval

The agent retrieves from all three stores simultaneously:
- **Episodic**: "what happened when I asked about fracking?"
- **Semantic**: "what does the user believe about fracking?"
- **Procedural**: "how does the user want me to answer?"

Results are merged by relevance, with procedural memories weighted highest
(they apply to all future conversations).
