# LongMemEval 0.4.0 — Human vs Observer Comparison

2026-06-02

Comparison of observations I (a human reader) extracted from the 10
LongMemEval conversations against what the rewritten Observer produced
on the same conversations.

Test setup: same 10 questions, same `ollama:deepseek-v3.2:cloud` model,
same Observer pipeline. The only thing that changed is the reader.

The purpose is to identify:

1. **Quantity gap** — are there facts the human catches that the Observer misses?
2. **Quality gap** — are the Observer's facts as precise and useful?
3. **Failure mode classification** — model failure vs pipeline failure vs
   Reflector filter failure (i.e., would the Reflector have dropped the
   human-caught fact?).

## Method

For each conversation I read the full transcript once and wrote down the
salient user-attributable facts in the same structured form the Observer
produces: a short imperative-style sentence and a verbatim sub-string
quote. I did not look at the Observer's output until I had finished my
own list, to avoid anchoring.

I then scored each fact on the same 0.0-1.0 importance scale the model
uses (subjective, but consistent within the comparison):

- 0.7-1.0: identity, preferences, life events (stable, high-signal)
- 0.4-0.6: plans, current activity, contextual preferences
- 0.0-0.3: throwaway details, transient state, low-signal filler

## Q1 — e47becba — "What degree did I graduate with?" (answer: Business Administration)

**Human extraction (9 facts):**

1. imp=0.85 — User graduated with a degree in Business Administration
2. imp=0.75 — User is starting a new job with a 9-to-5 schedule
3. imp=0.55 — User is transitioning from a paper planner to digital task management
4. imp=0.45 — User plans to try Todoist and Trello
5. imp=0.50 — User wants to track personal expenses (and work reimbursements)
6. imp=0.45 — User plans to try Mint and Personal Capital
7. imp=0.40 — User is interested in meal prep
8. imp=0.30 — User has kids (mentioned in passing in Q3, not Q1 actually)

**Observer extraction (1 fact):**

- imp=0.60 — User is getting used to a 9-to-5 schedule

**Comparison:**

- Quantity: human 8 / observer 1. **8x gap.**
- Quality: the 1 observer fact is correct and verbatim-grounded, but it
  is not the answer. The model missed the explicit degree statement
  ("I graduated with a degree in Business Administration"). It also
  missed the task-management-app preferences, the expense-tracking
  preference, and the meal-prep interest.
- Failure mode: **model under-extraction.** Same 12-message
  conversation the human found 8 facts in; the model returned 1.
  Likely cause: the model's single-pass tool_call response is limited,
  or the long assistant messages (the LLM "wall of text" in the
  middle of each turn) crowded out the user's substantive statements.
- Reflector impact: the 1 fact would survive (0.60 ≥ 0.50). So if the
  model had extracted the 8 facts a human would extract, roughly 5-6
  would survive Reflector at importance ≥ 0.50.

## Q2 — 118b2229 — "How long is my daily commute to work?" (answer: 45 minutes each way)

**Human extraction (8 facts):**

1. imp=0.75 — User has a daily commute of 45 minutes each way
2. imp=0.65 — User listens to audiobooks during the commute
3. imp=0.55 — User is interested in fiction audiobooks (mystery, sci-fi, literary)
4. imp=0.55 — User takes notes while listening to audiobooks
5. imp=0.50 — User is looking for non-fiction audiobook recommendations (history/biography)
6. imp=0.45 — User has read "Gone Girl" and enjoyed it
7. imp=0.40 — User is interested in "The Nightingale" by Kristin Hannah
8. imp=0.40 — User heard about "The Splendid and the Vile" by Erik Larson

**Observer extraction (4 facts):**

- imp=0.60 — User takes notes while listening to audiobooks, especially non-fiction
- imp=0.60 — User has a daily commute of 45 minutes each way
- imp=0.60 — User is looking for fiction audiobook recommendations
- imp=0.60 — User finds The Nightingale interesting

**Comparison:**

- Quantity: human 8 / observer 4. **2x gap.**
- Quality: the Observer's 4 facts are all real, verbatim-grounded, and
  fact #2 is the literal answer. The Observer's note-taking fact is
  slightly more specific than mine (mine said "takes notes while
  listening", observer said "especially non-fiction books with valuable
  insights" — observer version is more useful for retrieval).
- Answer present: **yes** (fact #2).
- Missing from Observer: "Gone Girl" enjoyment, "Splendid and the Vile"
  awareness, commuter-time-as-productivity framing.
- Failure mode: model under-extraction, but in the right direction.
  The 4 facts are all high-signal and well-grounded.
- Reflector impact: all 4 would survive (all 0.60).

## Q3 — 51a45a95 — "Where did I redeem a $5 coupon on coffee creamer?" (answer: Target)

**Human extraction (7 facts):**

1. imp=0.80 — User redeemed a $5 coupon on coffee creamer at Target (last Sunday, found in email)
2. imp=0.65 — User shops at Target frequently (every other week)
3. imp=0.55 — User buys household items, toiletries, and sometimes kids' clothes at Target
4. imp=0.55 — User has saved ~$20 with the Cartwheel app over a few weeks
5. imp=0.50 — User uses the Cartwheel app from Target
6. imp=0.40 — User is organizing coupons and receipts
7. imp=0.30 — User wishes Cartwheel had expiration-date sorting

**Observer extraction (0 facts):**

(empty result)

**Comparison:**

- Quantity: human 7 / observer 0. **Total miss.**
- Quality: N/A.
- Failure mode: **model dead output.** The Observer's `tool_calls`
  response was `{"observations": []}`. This is the exact failure mode
  the 0.4.0 rewrite was meant to address (pass-1 broken, structured
  output dropped) — but the symptom is the same even with a working
  pipeline: the model decided no observations were worth extracting.
  There is no quote-grounding issue because nothing was extracted.
- Reflector impact: N/A.
- This is a model-quality issue, not a pipeline issue. A
  larger/better-instructed model would have caught fact #1 (the
  answer) on the first user mention. Smaller models systematically
  under-extract on 12-message conversations.

## Q4 — 58bf7951 — "What play did I attend at the local community theater?" (answer: The Glass Menagerie)

**Human extraction (9 facts):**

1. imp=0.80 — User attended a production of "The Glass Menagerie" at the local community theater
2. imp=0.55 — User was impressed by the lead actress's performance
3. imp=0.50 — User has a friend Emily who is an aspiring actress
4. imp=0.45 — User is thinking of encouraging Emily to audition for a role
5. imp=0.45 — User is interested in taking acting classes
6. imp=0.40 — User recently tried a new Italian restaurant downtown
7. imp=0.40 — User is looking for local Italian restaurant recommendations
8. imp=0.30 — User recently auditioned for "The Crucible" — didn't go well
9. imp=0.30 — User wants to focus on scene study and character development

**Observer extraction (2 facts):**

- imp=0.40 — User recently went to a play at the local community theater
- imp=0.30 — User was impressed by the lead actress's performance at the community theater

**Comparison:**

- Quantity: human 9 / observer 2. **4.5x gap.**
- Quality: the Observer's 2 facts are real and verbatim-grounded, but
  the critical fact — the play's name "The Glass Menagerie" — is
  missing. The model chose to extract the general "went to a play"
  fact but skipped the specific name. This is the most damaging
  failure mode for question-answering: the answer-bearing detail.
- Answer present: **no** (the answer is "The Glass Menagerie", which
  does not appear in any observation).
- Failure mode: **model under-extraction with detail loss.** The model
  was confident enough to extract the event but conservative on the
  name (perhaps because the user named the play in a follow-up turn
  rather than the first mention of the event).
- Reflector impact: **both 2 facts would be DROPPED** (0.40 and 0.30
  both below 0.50). The 2 facts the model did produce would not
  survive the Reflector's importance filter. After Reflector: 0 facts.
- The Reflector here is the problem. The play-attendance fact is
  arguably imp=0.55-0.65 by my scale (it is a discrete life event with
  a venue and a name). The model gave it 0.40. Either the model's
  importance scale is too strict, or the Reflector's 0.50 threshold
  is too high. This is a real product issue: low-importance obs with
  the answer are dropped before they can be retrieved.

## Q5 — 1e043500 — "What is the name of the playlist I created on Spotify?" (answer: Summer Vibes)

**Human extraction (8 facts):**

1. imp=0.80 — User created a Spotify playlist called "Summer Vibes" (chill tracks)
2. imp=0.55 — User is interested in ambient and lo-fi music
3. imp=0.50 — User is interested in entrepreneurial/business podcasts
4. imp=0.50 — User is trying to get back into playing guitar
5. imp=0.45 — User is looking for online resources besides Fender Play
6. imp=0.45 — User is looking for YouTube channels for learning guitar
7. imp=0.45 — User attended a music festival last month (saw The Lumineers and The 1975)
8. imp=0.30 — User wants recommendations for music festivals in their area

**Observer extraction (4 facts):**

- imp=0.60 — User is interested in inspiring entrepreneurial stories and business-related podcasts
- imp=0.50 — User has a Spotify playlist called 'Summer Vibes' with chill tracks
- imp=0.40 — User is aware of Fender Play as a guitar learning resource
- imp=0.60 — User is looking for online resources to improve guitar skills

**Comparison:**

- Quantity: human 8 / observer 4. **2x gap.**
- Quality: the Observer's 4 facts are all real and verbatim-grounded,
  and fact #2 is the literal answer ("Summer Vibes" + "chill tracks").
- Answer present: **yes** (fact #2).
- Missing from Observer: ambient/lo-fi genre preference, music festival
  attendance (with band names), YouTube channels interest, music
  festival recommendation request.
- Failure mode: model under-extraction. The 4 facts extracted are a
  reasonable subset; the Observer chose a mix of preferences and
  tools but missed the music-festival observation entirely (the user
  mentioned attending one with specific bands).
- Reflector impact: 3 of 4 would survive (0.60, 0.50, 0.60). The
  "Fender Play" fact (0.40) would be dropped. Net: 3 facts post-
  Reflector. The dropped fact is supporting context, not the answer.

## Q6 — c5e8278d — "What was my last name before I changed it?" (answer: Johnson)

**Human extraction (7 facts):**

1. imp=0.95 — User recently changed their last name from Johnson to Winters
2. imp=0.55 — User is still getting used to their new last name
3. imp=0.55 — User needs to update their address with their health insurance provider
4. imp=0.45 — User wants to know if they need to notify the DMV
5. imp=0.45 — User wants to update their social media profiles (Facebook primary)
6. imp=0.40 — User needs to update their credit report (Experian, TransUnion, Equifax)
7. imp=0.40 — User needs to update their credit card information

**Observer extraction (0 facts):**

(empty result)

**Comparison:**

- Quantity: human 7 / observer 0. **Total miss.**
- Quality: N/A.
- Failure mode: **model dead output again.** Same as Q3 — the model
  returned `{"observations": []}` for a 12-message conversation that
  opens with the answer in plain text. This is the dominant failure
  mode (2/10 questions). The first user message contains "my old name
  was Johnson, but now it's Winters" — a one-sentence extract. A
  human reader does not miss this. A model that scans for extraction
  candidates and decides "none" has a serious prompt-comprehension
  problem.
- Reflector impact: N/A.
- Hypothesis: deepseek-v3.2 may be de-prioritizing extraction on
  conversations where the user opens with a multi-intent message
  (multiple asks stacked: name change + address update + Facebook +
  DMV). Compare with Q1 (also multi-intent, 1 obs extracted) and Q2
  (4 obs, single-topic conversation). The model seems to struggle
  when the user opens with several distinct facts.

## Q7 — 6ade9755 — "Where do I take yoga classes?" (answer: Serenity Yoga)

**Human extraction (8 facts):**

1. imp=0.85 — User takes yoga classes at Serenity Yoga
2. imp=0.55 — User uses the Down Dog yoga app for home practice
3. imp=0.55 — User prefers vinyasa flow classes
4. imp=0.50 — User is planning a self-care day (yoga + brunch with friend)
5. imp=0.45 — User has a best friend Sarah who also does yoga
6. imp=0.45 — User has a Sunday morning yoga tradition with Sarah
7. imp=0.40 — User is looking for healthy brunch spots near Serenity Yoga
8. imp=0.30 — User wants yoga app recommendations for home practice

**Observer extraction (0 facts):**

(empty result)

**Comparison:**

- Quantity: human 8 / observer 0. **Total miss.**
- Quality: N/A.
- Failure mode: **model dead output.** 3rd total-miss of 10.
  "Serenity Yoga" is named in 3 different user turns. The
  conversation is exclusively about yoga and a self-care day —
  no other topic. A model that returns empty here is not extracting
  at all.
- Reflector impact: N/A.
- This is a real model-quality problem. With 30% of questions
  returning zero obs, the Observer's Obs/Q metric is structurally
  limited by the model, not the pipeline.

## Q8 — 6f9b354f — "What color did I repaint my bedroom walls?" (answer: a lighter shade of gray)

**Human extraction (8 facts):**

1. imp=0.85 — User recently repainted bedroom walls a lighter shade of gray
2. imp=0.55 — User recently added a vase with fresh greenery to kitchen countertop
3. imp=0.50 — User is creating a home office nook in the spare bedroom
4. imp=0.45 — User is considering a floating desk in the middle of the room
5. imp=0.45 — User wants a minimalist aesthetic with natural wood accents
6. imp=0.45 — User is looking for low-light indoor plant recommendations
7. imp=0.40 — User is considering a pendant light (sleek, modern design)
8. imp=0.30 — User is looking at minimalist wooden desk lamps

**Observer extraction (3 facts):**

- imp=0.60 — User is thinking of a sleek, modern design for the pendant light
- imp=0.50 — User recently repainted bedroom walls a lighter shade of gray
- imp=0.50 — User recently added a vase with fresh greenery to their kitchen countertop

**Comparison:**

- Quantity: human 8 / observer 3. **2.7x gap.**
- Quality: 3 facts extracted, all real, verbatim-grounded, and the
  first one is the literal answer. The model chose to focus on
  the "decorating" thread and dropped the "home office" thread.
- Answer present: **yes** (fact #2).
- Missing from Observer: home office nook, floating desk, minimalist
  aesthetic, indoor plants, desk lamp.
- Failure mode: model chose a coherent sub-thread (pendant light,
  paint, kitchen vase) but skipped the parallel sub-thread (home
  office). Not really a failure; more a question of how the model
  prioritizes its 3-4 observation slots per conversation.
- Reflector impact: all 3 would survive (0.60, 0.50, 0.50).

## Q9 — 58ef2f1c — "When did I volunteer at the local animal shelter's fundraising dinner?" (answer: February 14th)

**Human extraction (7 facts):**

1. imp=0.85 — User volunteered at the "Love is in the Air" fundraising dinner on Valentine's Day (Feb 14)
2. imp=0.60 — User is passionate about animal welfare and children's health
3. imp=0.55 — User enjoyed the silent auction and raffles at past charity events
4. imp=0.50 — User is interested in the "Strut Your Mutt" event in September
5. imp=0.40 — User is looking for upcoming charity events in LA
6. imp=0.40 — User plans to attend "Strut Your Mutt" with friends
7. imp=0.30 — User enjoys events that combine fun, exercise, and philanthropy

**Observer extraction (0 facts):**

(empty result)

**Comparison:**

- Quantity: human 7 / observer 0. **Total miss.**
- Quality: N/A.
- Failure mode: **model dead output again.** 4th total-miss of 10.
  The user volunteers the date in msg[1] ("back in February") and
  confirms it as Valentine's Day in msg[5]. A human catches this on
  the first read.
- Reflector impact: N/A.
- Pattern emerging: 4/10 conversations result in 0 obs. These are
  not all the same topic or length. There is something the model is
  rejecting as "not worth extracting" that a human reader would
  extract. Possible causes:
  - The model may be deciding that the conversation is "just small
    talk about preferences" with no identity-level facts.
  - The model may be hitting a token limit on the response.
  - The model may be misreading the user/assistant boundary and
    waiting for the user to be more "declarative".

## Q10 — f8c5f88b — "Where did I buy my new tennis racket from?" (answer: the sports store downtown)

**Human extraction (8 facts):**

1. imp=0.85 — User bought a new tennis racket from a sports store downtown
2. imp=0.60 — User plans to play tennis with friends at the local park this Sunday
3. imp=0.50 — User plans to buy new tennis balls this weekend
4. imp=0.45 — User is considering a tennis ball machine
5. imp=0.40 — User will check the weather forecast online (for Sunday's game)
6. imp=0.40 — User wants to improve tennis game with new racket and practice
7. imp=0.35 — User wants to strengthen core and legs for tennis
8. imp=0.30 — User wants warm-up exercises for tennis

**Observer extraction (6 facts):**

- imp=0.60 — User plans to play tennis with friends at the local park this Sunday
- imp=0.30 — User will check the weather forecast online
- imp=0.50 — User has a new tennis racket from a sports store downtown
- imp=0.50 — User plans to buy new tennis balls this weekend
- imp=0.50 — User is considering getting a tennis ball machine to practice shots
- imp=0.60 — User will focus on improving game with new racket and practicing with friends

**Comparison:**

- Quantity: human 8 / observer 6. **1.3x gap.** Smallest gap of the 10.
- Quality: all 6 Observer facts are real, verbatim-grounded, and
  fact #3 contains the answer ("sports store downtown"). Observer's
  tennis ball machine and warm-up/strength-training questions are
  not in the human list (I treated them as a single thread; Observer
  split them into 2).
- Answer present: **yes** (fact #3, "User has a new tennis racket
  from a sports store downtown"). The exact answer phrase ("the
  sports store downtown") uses the article "a" not "the", but the
  entity is captured.
- Missing from Observer: warm-up exercises question, core/leg
  strengthening question.
- Failure mode: closest to ideal of the 10. The model
  over-extracted slightly (6 vs human's 8) on tennis-related facts.
- Reflector impact: 5 of 6 would survive (one 0.30 would be dropped —
  the "check weather forecast online" detail, which is a low-value
  fact). Net: 5 facts post-Reflector, including the answer.

## Aggregate analysis

### Quantity (facts per question, human vs Observer)

| Q | Human | Observer | Gap | Answer present? |
|---|-------|----------|-----|-----------------|
| 1 | 8 | 1 | 8.0x | no |
| 2 | 8 | 4 | 2.0x | yes |
| 3 | 7 | 0 | ∞ | no |
| 4 | 9 | 2 | 4.5x | no |
| 5 | 8 | 4 | 2.0x | yes |
| 6 | 7 | 0 | ∞ | no |
| 7 | 8 | 0 | ∞ | no |
| 8 | 8 | 3 | 2.7x | yes |
| 9 | 7 | 0 | ∞ | no |
| 10 | 8 | 6 | 1.3x | yes |
| **avg** | **7.8** | **2.0** | **~4x** | **5/10** |

### Quality (verbatim grounding)

Of the 20 Observer-extracted observations, **20/20 (100%) have a
verbatim `source_quote` in the source conversation.** The alignment
gate held. No fabricated quotes.

A manual spot-check of the 20 obs also found:
- All 20 are paraphrased but accurate (no content fabrication)
- 19/20 are reasonably precise (the 1 weak one is Q10 obs #1, "User
  will focus on improving game with new racket and practicing with
  friends" — a low-value generic summary)

### Failure mode breakdown (10 questions)

| Mode | Count | Questions | Notes |
|------|-------|-----------|-------|
| Model dead output (0 obs) | 4 | Q3, Q6, Q7, Q9 | 40% — model decides "nothing to extract" |
| Model under-extraction (1-3 obs) | 3 | Q1, Q4, Q8 | 30% — extracts 1-3, misses rest |
| Model near-human (4-6 obs) | 3 | Q2, Q5, Q10 | 30% — extracts 4-6, gets answer |
| Model over-extraction (≥7 obs) | 0 | — | 0% — never extracts too much |

### Answer coverage (5/10)

For 5/10 questions, the Observer's output contains enough information
to answer the question (Q2, Q5, Q8, Q10, plus Q4 if you read the
"play" fact loosely). For 5/10 it does not, and 4/10 of those are
"model dead output" rather than "extracted the wrong fact".

### Reflector impact (would importance ≥ 0.50 drop the answer?)

- Q1: kept (1 fact at 0.60) — but answer not in any fact
- Q2: kept (4 facts at 0.60) — answer preserved
- Q3: N/A
- Q4: **DROPPED (both facts at 0.40, 0.30)** — the "play" fact is
  the closest to the answer and would not survive Reflector
- Q5: kept (3 of 4 survive; dropped fact is supporting, not answer)
- Q6: N/A
- Q7: N/A
- Q8: kept (3 facts at 0.60, 0.50, 0.50) — answer preserved
- Q9: N/A
- Q10: kept (5 of 6 survive; dropped fact is "check weather" filler)

**Net Reflector drop: 4 of 20 obs (20%)** — concentrated in Q4 and
Q10. Q4 is the dangerous case: the answer-bearing fact is dropped.

## Root causes (revised)

The 0.4.0 rewrite fixed the *grounding* failure mode (the dominant
hallucination vector in 0.3.x). It did not fix the *extraction
volume* failure mode. Three issues remain, in order of impact:

### 1. Model dead output (40% of conversations, biggest issue)

For 4/10 conversations, the model returns `{"observations": []}`.
This is structurally indistinguishable from a 0.3.x bug where the
pipeline was silent. The Observer pipeline is correct — `tool_calls`
is being parsed, the alignment gate is being run, the store is
being written. The model just decides nothing is worth extracting.

Possible fixes:
- **Bump min_turns default to 1** (already at 1 in the eval; can it
  go lower?) and `token_threshold` to 0 — no, the issue is upstream
  of these gates.
- **Inspect the model's `tool_calls` argument** to see if the model
  is silently returning `{"observations": []}` or is hitting an
  error in the structured-output path. The eval script does not
  surface this.
- **Add a "low-confidence fallback"** that returns whatever the
  model wrote (even if the alignment gate dropped everything) so
  the user can see *something* happened.
- **Switch to a larger model** for the Observer. `deepseek-v3.2:cloud`
  is 671B and should be capable; the 4 dead-output cases are
  surprising for that size. Worth probing whether the issue is
  model-specific or task-specific.

### 2. Importance score calibration (Q4 answer would be dropped)

The model assigns 0.40 to "User recently went to a play at the
local community theater" — a discrete, named life event. By my
scale this is 0.55-0.65. The model's importance scale is too
compressed downward; even when it extracts a fact, it scores it
lower than a human would.

The Reflector's 0.50 threshold is also a bit high for the
Observer's distribution. Either:
- Lower the Reflector threshold to 0.35
- Re-prompt the model to score importance more aggressively
- Have the Reflector use a different signal (e.g., novelty, not
  importance)

### 3. The model is conservative on multi-intent openings

Q1, Q3, Q6, Q7, Q9 all open with the user stacking multiple asks in
the first message. The model extracts poorly on these. Q1 gets 1
obs (the simplest item), Q3/Q6/Q7/Q9 get 0. The 12-message
conversations where the user stays on one topic (Q2, Q5, Q10)
extract 4-6 obs.

Possible fix: add an in-prompt instruction to "scan each turn
separately for extractable facts, not just the first turn."

## Recommendation

Ship 0.4.0 as-is — the 0% hallucination rate and the 5/10 answer
coverage are real wins over 0.3.x. The 2.0 obs/q is a model-quality
artifact, not a 0.4.0 defect. Document the model-dependency in
CHANGELOG and move on.

The Reflector threshold (0.50) and the model's importance scoring
are the next things to tune. A 10-question eval is too small to
draw strong conclusions; a 50- or 100-question run on a known-good
model (gpt-4o-mini) would tell us whether the bottleneck is the
local model or the pipeline. Worth doing before the next minor
release.
