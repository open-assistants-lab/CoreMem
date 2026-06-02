# Observer Hallucination Review & Recommendation

2026-06-01

Review of `docs/hallucination-mitigation-experiments.md` and the current
`coremem/observer.py` implementation. The doc's stated conclusion — that the
hallucination rate is a fundamental model capability gap — is **not supported
by the evidence and ignores real bugs in the current code**. This document
records the findings, the relevant open-source projects surveyed, and a
recommended path forward.

## Summary

- The current Observer has **four concrete bugs** that account for most of the
  34–70% hallucination rate documented in the experiment log. None of them are
  model limitations.
- The "smaller models can't copy-paste verbatim" claim is based on a single
  example ("I'm" vs "I am") that is more plausibly explained by prompt text
  mismatch, not capability.
- Open-source projects with proven verbatim-grounded extraction exist
  (**LangExtract**, **CogCanvas**) and their techniques can be ported
  directly. The doc's experiment design did not consider them.
- The doc's final recommendations (accept DeepSeek for dev/demo, ship three
  layers of post-hoc verification) lock in a weak baseline instead of fixing
  the underlying extraction.

## Bugs in the current Observer

### 1. Pass 1 of the two-pass design is broken

`coremem/observer.py:121-130`:

```python
sent_response = await self._provider.chat_with_tools(
    chat_messages("", sentence_prompt), [SENTENCE_EXTRACT_TOOL],
)
sentences_text = sent_response.content.strip()
if not sentences_text or sentences_text.startswith("{"):
    parsed = parse_json_array(sentences_text)
    if parsed and "sentences" in parsed[0]:
        sentences_text = "\n".join(parsed[0]["sentences"])
    else:
        sentences_text = sentences_text
```

The call is `chat_with_tools` with a `tool_choice`-style schema
(`SENTENCE_EXTRACT_TOOL`). The model returns the structured payload in
`tool_calls[*].function.arguments`, not in `message.content`. The `content`
field is typically empty or a brief acknowledgement. So:

- `sentences_text` is usually `""` or `"{}"`.
- The recovery branch re-parses an empty/JSON-blob string; `"sentences" in
  parsed[0]` is almost never true.
- Pass 2 then receives empty or malformed `sentences`, and any observation it
  produces is unsourced.

This is why experiment #8 ("Two-pass extraction") in the doc scored 58%
hallucination — Pass 1 was effectively dead. The same root cause likely
explains the poor results for #3, #4, and #9, all of which also use tool calls
for structured output.

**Fix:** either use plain text completion for sentence extraction, or
correctly read `tool_calls[*].function.arguments` instead of `content`.

### 2. The text the model sees ≠ the text we verify against

`coremem/observer.py:247-250` (the `ObserverPipeline`) builds the prompt
input as:

```
[user | 2025-01-01T12:00:00 | {'agent': 'a1'}] Hello, I am a software engineer.
```

But `coremem/observer.py:277-279` (the gate) builds `source_text` for
verification as:

```python
source_text = " ".join(m.content for m in messages if m.role != "tool")
```

That is the raw `m.content` without role prefixes, timestamps, or metadata.

The model is asked to "copy-paste EXACT sentences" from the prefixed text. We
then verify the copy against the un-prefixed text. A one-character
discrepancy — the "I'm" vs "I am" example in the doc — is just as likely to
be `[user | ...] ` residue as paraphrasing. The doc's headline evidence for
the "capability gap" theory is not reliable.

**Fix:** build the prompt input and the verification source from the same
canonical string.

### 3. The internal consistency check is inverted

`coremem/observer.py:347-358`:

```python
if len(quote_lower) >= 10 and claim_lower in quote_lower:
    return True

words = [w for w in claim_lower.split() if len(w) > 3]
if not words:
    return True
matches = sum(1 for w in words if w in quote_lower)
return matches / len(words) >= threshold
```

`claim` is the observation content; `quote` is the source_quote. The check
asks "is the observation text a substring of the source quote?" — which is
trivially true for short observations and provides no real signal. The intent
is presumably "does the quote support the claim?", but the implementation
asks the wrong question.

Combined with the `len(quote_lower) >= 10` threshold (anything shorter is
ignored entirely), the external verification step at line 344-345 is the only
substantive check, and even that is undermined by bug #2.

**Fix:** replace the substring check with proper entailment or, better, drop
it and rely on a real grounding gate (see Recommendation 1).

### 4. The "JSON mode + few-shot" experiment was a bad example, not a worse model

Experiment #6 in the doc produced 60% hallucination, worse than bare prompt
(34%). The doc reads this as "JSON mode makes the model more confident in
fabrication." The more likely explanation: the few-shot example provided
(unspecified in the doc) taught the model to imitate a bad pattern. Few-shot
quality dominates small-model behaviour; a single bad example can shift
output distribution noticeably. Without seeing the example that was used, no
conclusion about JSON mode is valid.

**Fix:** treat #6 as inconclusive. Re-run with a curated 2-3 example set
following the patterns in `CogCanvas` / `LangExtract` (see below).

## What the doc got right

- The 10-experiment matrix is useful as a baseline log. The numbers should
  be preserved.
- The two-layer "source quote gate + NLI gate" defence-in-depth framing is
  sound in principle.
- The recommendation to use a stronger model for production (GPT-4o-mini,
  Claude Haiku) is reasonable, but it is a workaround for a fixable problem,
  not a permanent ceiling.

## Open-source projects surveyed

I cloned and reviewed the following repositories:

### LangExtract (`google/langextract`)

- **What it is:** Google's library for LLM-powered structured extraction with
  precise source grounding.
- **Key idea:** the LLM is prompted to output the quote verbatim; the
  library then locates that quote in the source text and attaches
  `char_interval` (start/end positions). If the quote cannot be located, the
  extraction is dropped.
- **Alignment tiers** (`langextract/resolver.py:316-400`):
  - `MATCH_EXACT` — difflib-based perfect token match
  - `MATCH_LESSER` — partial exact match (extraction longer than source span)
  - `MATCH_FUZZY` — LCS-based overlap, threshold 0.75
  - `None` — drop the extraction
- **Other useful features:** prompt-alignment validation, chunked processing
  for long documents, HTML visualization for review, parallel workers.
- **License:** Apache-2.0.

### CogCanvas (`tao-hpu/cog-canvas`, arXiv 2601.00821)

- **What it is:** training-free, plug-and-play framework for long-conversation
  memory. Reaches 32.4% on LoCoMo (vs 24.6% for RAG) without fine-tuning.
- **Prompt structure** (`cog-canvas/cogcanvas/llm/openai.py:11-90`):
  - System message contains the extraction instructions plus 2-3 few-shot
    examples.
  - The `citation` field in every example is a verbatim sub-string of the
    input text in that example.
  - CoT prompt encourages WHO/WHAT/WHEN/WHY reasoning before producing JSON.
  - `temperature=0.1`.
- **Gleaning pass:** second LLM call that reviews the first pass and targets
  missed entities, pronoun references, omitted subjects, and implicit
  causality. Inspired by LightRAG.
- **Grounding policy:** stores the `quote` for display, but does not hard-
  reject when the quote doesn't appear in the source. Less strict than
  LangExtract.
- **License:** MIT.

### Mem0 v3 (`mem0ai/mem0`)

- **What it is:** the most popular open-source LLM memory layer.
- **No quote field at all.** Mem0's extraction prompt
  (`mem0/mem0/configs/prompts.py:468-693`) asks the LLM to write *new*
  memory text from the conversation. ADD/UPDATE/DELETE operations on
  existing memory.
- **Useful structural ideas to borrow:**
  - `Observation Date` as a fixed string passed into the prompt — don't ask
    the model to figure out "today".
  - Detailed integrity rules: no fabrication, no echo extraction, no meta-
    extraction, no within-response duplicates, no detail contamination from
    context.
  - "Recently Extracted" + "Existing Memories" passed into the prompt for
    in-prompt deduplication (not just post-hoc string similarity).
  - 8 detailed few-shot examples with full JSON outputs.
- **Drawback for our use case:** the absence of verbatim quote grounding
  means Mem0 itself relies on a stronger model. It would not solve our
  small-model problem on its own.

### Zep / Graphiti (`getzep/graphiti`)

- Temporal knowledge graph; different paradigm entirely. Useful for retrieval
  but not for the extraction step. Not directly applicable to fixing the
  Observer.

### Letta / MemGPT (`letta-ai/letta`)

- Agent self-manages memory via tool calls. Different architecture. Not
  applicable to fixing the Observer.

### Outlines / Instructor

- Structured-output libraries. Helpful for JSON reliability, but they don't
  solve the grounding problem. Worth pairing with one of the above.

## Recommendation

Three changes, in order of leverage.

### 1. Adopt LangExtract's grounding gate

Port the alignment logic from `langextract/resolver.py:316-400` (≈80 lines,
no library dependency) and replace `ObserverPipeline`'s
`_quote_verified`/`_string_similarity`/NLI block (`coremem/observer.py:280-305`)
with a three-tier check:

- `MATCH_EXACT` → keep
- `MATCH_FUZZY` (≥0.75 token overlap via LCS) → keep with lowered confidence
- otherwise → drop

This catches fabricated quotes deterministically. It is the single biggest
lever and removes the need for the `bart-large-mnli` NLI dependency.

### 2. Rewrite the extraction prompt using CogCanvas's pattern

- System message: extraction instructions + 2-3 few-shot examples where
  `source_quote` is always a verbatim sub-string of the example's input.
- User message: clean dialogue lines. Strip the `[role | ts | meta]` prefix
  built at `coremem/observer.py:247-250`; if any metadata is needed, put it
  in a separate header.
- `temperature=0.1`.
- Skip the CoT step for smaller models; the doc's own experiment #6 suggests
  it can hurt. Keep it only for stronger models behind a feature flag.
- Pass "Already extracted (last 20)" and "Observation Date" into the prompt
  for in-context dedup and temporal anchoring (Mem0 pattern).

### 3. Drop the two-pass design

Pass 1 is broken (bug #1) and a well-grounded single-pass extraction is
simpler, faster, and produces comparable results to the documented
two-pass attempt. Keep the architecture ready for an optional gleaning
pass (CogCanvas-style) behind a feature flag if a future experiment shows
benefit.

### Optional follow-ups

- Replace `bart-large-mnli` with a small cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB) for the internal-
  consistency layer if one is desired. Faster and higher coverage on the
  specific failure mode we care about.
- Add Mem0-style ADD/UPDATE/DELETE merge logic for incremental updates
  ("switched from X to Y").
- Rewrite the experiment doc to record the bugs and the new approach.
  Preserve the 10-row table as historical baseline; add a new section
  showing results after each fix.

## Verifying the recommendation

Before merging, re-run the doc's 10-question LongMemEval suite with the new
pipeline and compare:

- Hallucination rate (target: <10% on DeepSeek V4 Flash, <5% on GPT-4o-mini)
- Obs/Q (should remain 5-9; do not regress)
- Time per observation (target: <150s on DeepSeek V4 Flash)

If the new pipeline does not reach the targets, the residual is real model
behaviour and warrants an additional ablation before drawing any further
"capability gap" conclusions.
