"""3-tier alignment gate — port of langextract/resolver.py:316-400.

Pure function, stdlib only. Checks whether a quote appears as a
sub-string of a source text, with three confidence tiers:
EXACT (token-list match after whitespace + case normalization),
FUZZY (character-level SequenceMatcher.ratio() on whitespace+case-
normalized strings, >= 0.75 threshold — chosen over token-level
because LLM source_quote values drift by single characters or
punctuation, e.g. "engineer." vs "engineer" or "I'm" vs "I am",
which would be token-mismatches but should still pass),
NONE (drop the observation).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum


class AlignmentTier(Enum):
    EXACT = "exact"
    FUZZY = "fuzzy"
    NONE = "none"


@dataclass
class AlignmentResult:
    tier: AlignmentTier
    confidence: float
    char_interval: tuple[int, int] | None


_FUZZY_THRESHOLD = 0.75


def align_quote(quote: str, source: str) -> AlignmentResult:
    """Check whether ``quote`` appears as a sub-string of ``source``.

    Tokenizes both on whitespace and lowercases for case-insensitive
    matching. Returns EXACT if the quote's tokens form a contiguous
    window in the source, FUZZY if SequenceMatcher.ratio() on the
    character sequences is >= 0.75, otherwise NONE.

    ``char_interval`` semantics:
    - EXACT: points to the matched substring in ``source``.
    - FUZZY: points to the longest common substring (LCS) between
      ``source`` and ``quote`` (computed on case+whitespace-normalized
      strings; the returned indices are valid for the original
      ``source`` for ASCII text).
    - NONE: ``None``.
    """
    if not quote or not source:
        return AlignmentResult(AlignmentTier.NONE, 0.0, None)

    quote_lower = quote.lower().strip()
    source_lower = source.lower()

    quote_tokens = quote_lower.split()
    source_tokens = source_lower.split()

    if not quote_tokens:
        return AlignmentResult(AlignmentTier.NONE, 0.0, None)
    if len(quote_tokens) > len(source_tokens):
        return AlignmentResult(AlignmentTier.NONE, 0.0, None)

    # EXACT: contiguous window of source tokens matches all quote tokens
    for i in range(len(source_tokens) - len(quote_tokens) + 1):
        window = source_tokens[i : i + len(quote_tokens)]
        if window == quote_tokens:
            char_interval = _char_span_for_token_range(
                source, i, i + len(quote_tokens)
            )
            return AlignmentResult(AlignmentTier.EXACT, 1.0, char_interval)

    # FUZZY: char-level SequenceMatcher on lowered strings (LCS-based).
    # Char-level (not token-level) so single-character / punctuation
    # drift in the LLM's source_quote (e.g. "engineer." vs "engineer")
    # still passes the gate.
    sm = SequenceMatcher(None, source_lower, quote_lower)
    ratio = sm.ratio()
    if ratio >= _FUZZY_THRESHOLD:
        block = sm.find_longest_match(0, len(source_lower), 0, len(quote_lower))
        char_interval = (block.a, block.a + block.size)
        return AlignmentResult(AlignmentTier.FUZZY, ratio, char_interval)

    return AlignmentResult(AlignmentTier.NONE, 0.0, None)


def _char_span_for_token_range(
    source: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    """Map a token-index range to (char_start, char_end) in source."""
    char_start = -1
    char_end = -1
    token_idx = 0
    i = 0
    while i < len(source):
        # Skip leading whitespace
        while i < len(source) and source[i].isspace():
            i += 1
        if i >= len(source):
            break
        # Read token
        token_start = i
        while i < len(source) and not source[i].isspace():
            i += 1
        token_end = i
        if token_idx == start:
            char_start = token_start
        if token_idx == end - 1:
            char_end = token_end
            break
        token_idx += 1
    if char_start < 0 or char_end < 0:
        return (0, 0)
    return (char_start, char_end)
