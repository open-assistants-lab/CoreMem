"""3-tier alignment gate — port of langextract/resolver.py:316-400.

Pure function, stdlib only. Checks whether a quote appears as a
sub-string of a source text, with three confidence tiers:
EXACT (difflib-perfect token match), FUZZY (LCS ratio >= 0.75),
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
            char_interval = _token_window_to_char_span(
                source, source_tokens, i, i + len(quote_tokens)
            )
            return AlignmentResult(AlignmentTier.EXACT, 1.0, char_interval)

    # FUZZY: SequenceMatcher ratio on characters (LCS-based)
    sm = SequenceMatcher(None, source, quote)
    ratio = sm.ratio()
    if ratio >= _FUZZY_THRESHOLD:
        block = sm.find_longest_match(0, len(source), 0, len(quote))
        char_interval = (block.a, block.a + block.size)
        return AlignmentResult(AlignmentTier.FUZZY, ratio, char_interval)

    return AlignmentResult(AlignmentTier.NONE, 0.0, None)


def _token_window_to_char_span(
    source: str,
    source_tokens: list[str],
    start: int,
    end: int,
) -> tuple[int, int]:
    """Map a token-index range in source_tokens to (char_start, char_end) in source."""
    if start >= len(source_tokens):
        return (len(source), len(source))
    # Walk through source to find char offsets
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
