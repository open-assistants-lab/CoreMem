"""Answer-prompt helpers for consuming recall bundles (RAG reading strategies).

CoreMem delivers retrieval results — SessionBundles with evidence-first
ordering. The *reading strategy* (how an LLM turns a bundle into an answer)
is the consumer's choice. This module ships the reading prompt that was
validated on LongMemEval-S (500 questions, paired A/B, same contexts):

- Direct prompting (baseline): 0.674 accuracy
- Chain-of-Note (CoN, extract-then-reason): 0.692 accuracy (+0.018,
  17 wins / 8 losses, McNemar p=0.108)

Guidance (from the full-S experiment, docs/retrieval-experiments.md):
- Enable CoN for synthesis-heavy questions (multi-session counting,
  temporal arithmetic) — that is where its gains concentrate.
- Disable when abstention fidelity matters: CoN's extract-then-answer
  framing nudged one false-premise answer toward hallucination in the
  full-S run (abstention accuracy 0.900 -> 0.867).
- Cost: CoN answers carry ~50% more output tokens (2.1s vs 1.4s mean
  with a 120B reader).

Zero-LLM by itself — these are prompt builders, not API calls.
"""

from __future__ import annotations

from coremem.types import SessionBundle

_CON_PROMPT = (
    "Answer the question using only the memory context. Work step by step: first go "
    "through each memory item and write a brief note of any information relevant to "
    "the question, then reason over your notes to reach the final answer. Resolve "
    "comparisons, counts, and date differences when the context supports them. If the "
    "context is insufficient, say that the information is insufficient. End your reply "
    "with a final line of the exact form: Final answer: <concise answer>\n\n"
    "Question:\n{question}\n\nMemory context:\n{context}"
)

_DIRECT_PROMPT = (
    "Answer the question using only the memory context. Resolve comparisons, counts, "
    "and date differences when the context supports them. If the context is insufficient, "
    "say that the information is insufficient. Give a concise direct answer.\n\n"
    "Question:\n{question}\n\nMemory context:\n{context}"
)

_FINAL_ANSWER_PREFIX = "final answer:"


def build_context(bundles: list[SessionBundle]) -> str:
    """Serialize bundles into the evidence-first context format.

    Anchors (the retrieved messages that pulled each session in) lead;
    remaining messages follow chronologically. This matches the bundle
    formatting validated in the S answer eval.
    """
    from datetime import UTC, datetime

    parts: list[str] = []
    for bundle in bundles:
        anchor_ids = set(bundle.anchor_ids or [])
        ordered = sorted(
            bundle.messages,
            key=lambda m: (
                m.id not in anchor_ids,
                m.ts or datetime.min.replace(tzinfo=UTC),
            ),
        )
        message_parts = []
        for message in ordered:
            date = message.ts.date().isoformat() if message.ts else "unknown"
            message_parts.append(f"[{date} {message.role}] {message.content}")
        parts.append(
            f"[session={bundle.session_id} complete={str(bundle.complete).lower()}]\n"
            + "\n".join(message_parts)
        )
    return "\n\n".join(parts)


def build_answer_prompt(
    question: str,
    context: str,
    *,
    chain_of_note: bool = False,
) -> str:
    """Build a reading prompt for an LLM over recall context.

    ``chain_of_note=True`` uses the extract-then-reason reading strategy
    validated at +0.018 accuracy on LongMemEval-S (docs/retrieval-experiments.md).
    Prefer it for synthesis-heavy questions; prefer the direct prompt when
    abstention fidelity or token cost dominates.
    """
    template = _CON_PROMPT if chain_of_note else _DIRECT_PROMPT
    return template.format(question=question, context=context)


def extract_final_answer(response: str) -> str:
    """Pull the final answer out of a Chain-of-Note response.

    CoN replies end with ``Final answer: <answer>``; returns the text after
    the last such marker, or the stripped response when the marker is absent
    (so direct-prompt responses pass through unchanged).
    """
    lowered = response.casefold()
    idx = lowered.rfind(_FINAL_ANSWER_PREFIX)
    if idx == -1:
        return response.strip()
    return response[idx + len(_FINAL_ANSWER_PREFIX):].strip()