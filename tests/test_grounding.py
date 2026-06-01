"""Tests for the 3-tier alignment gate (LangExtract port)."""


from coremem.grounding import (
    AlignmentResult,
    AlignmentTier,
    align_quote,
)

# ── Tier enum ──────────────────────────────────────────────────────────────


class TestAlignmentTier:
    def test_values(self):
        assert AlignmentTier.EXACT.value == "exact"
        assert AlignmentTier.FUZZY.value == "fuzzy"
        assert AlignmentTier.NONE.value == "none"


class TestAlignmentResult:
    def test_exact_construction(self):
        r = AlignmentResult(AlignmentTier.EXACT, 1.0, (10, 20))
        assert r.tier == AlignmentTier.EXACT
        assert r.confidence == 1.0
        assert r.char_interval == (10, 20)

    def test_none_construction(self):
        r = AlignmentResult(AlignmentTier.NONE, 0.0, None)
        assert r.tier == AlignmentTier.NONE
        assert r.char_interval is None


# ── align_quote: EXACT tier ────────────────────────────────────────────────


class TestAlignExact:
    def test_perfect_match_middle(self):
        r = align_quote("Hello world", "Say Hello world there")
        assert r.tier == AlignmentTier.EXACT
        assert r.confidence == 1.0
        assert r.char_interval is not None
        # The matched span should start at "Hello" and end after "world"
        start, end = r.char_interval
        assert "Say Hello world there"[start:end] == "Hello world"

    def test_perfect_match_at_start(self):
        r = align_quote("Hello", "Hello world")
        assert r.tier == AlignmentTier.EXACT

    def test_perfect_match_at_end(self):
        r = align_quote("world", "Hello world")
        assert r.tier == AlignmentTier.EXACT

    def test_whitespace_only_diff(self):
        # Multiple internal spaces collapse on tokenize
        r = align_quote("Hello   world", "Hello world")
        assert r.tier == AlignmentTier.EXACT

    def test_case_mismatch(self):
        r = align_quote("HELLO", "hello")
        assert r.tier == AlignmentTier.EXACT


# ── align_quote: FUZZY tier ────────────────────────────────────────────────


class TestAlignFuzzy:
    def test_single_char_drift(self):
        # "I'm" vs "I am" — ratio >= 0.75
        r = align_quote("I'm a software engineer", "I am a software engineer")
        assert r.tier == AlignmentTier.FUZZY
        assert r.confidence >= 0.75
        assert r.char_interval is not None

    def test_trailing_punctuation(self):
        r = align_quote("engineer.", "engineer")
        assert r.tier == AlignmentTier.FUZZY
        assert r.confidence >= 0.75

    def test_fuzzy_char_interval_is_substring(self):
        # Trailing-punctuation drift — FUZZY char_interval should be a
        # non-empty sub-string of source (the longest common substring).
        source = "engineer"
        r = align_quote("engineer.", source)
        assert r.tier == AlignmentTier.FUZZY
        assert r.char_interval is not None
        start, end = r.char_interval
        matched = source[start:end]
        assert len(matched) > 0
        assert matched in source

    def test_fuzzy_char_interval_in_middle(self):
        # Inserted/clipped drift in the middle of source — FUZZY tier,
        # char_interval points to the LCS within source.
        source = "I am a software engineer"
        r = align_quote("I am a software engineer.", source)
        assert r.tier == AlignmentTier.FUZZY
        assert r.char_interval is not None
        start, end = r.char_interval
        matched = source[start:end]
        assert len(matched) > 0
        assert matched in source


# ── align_quote: NONE tier ─────────────────────────────────────────────────


class TestAlignNone:
    def test_fabricated_quote(self):
        r = align_quote("lives on Mars", "lives in Denver")
        assert r.tier == AlignmentTier.NONE
        assert r.confidence < 0.75
        assert r.char_interval is None

    def test_below_threshold_overlap(self):
        r = align_quote("Hello cruel world", "Hello world")
        assert r.tier == AlignmentTier.NONE

    def test_empty_quote(self):
        r = align_quote("", "Hello world")
        assert r.tier == AlignmentTier.NONE
        assert r.char_interval is None

    def test_empty_source(self):
        r = align_quote("Hello", "")
        assert r.tier == AlignmentTier.NONE

    def test_quote_longer_than_source(self):
        r = align_quote("Hello world foo", "Hello")
        assert r.tier == AlignmentTier.NONE

    def test_whitespace_only_quote(self):
        r = align_quote("   ", "Hello world")
        # Whitespace tokenizes to empty tokens → NONE
        assert r.tier == AlignmentTier.NONE


# ── align_quote: char_interval accuracy ────────────────────────────────────


class TestCharInterval:
    def test_exact_span_points_to_matched_text(self):
        source = "Alice said Hello world to Bob"
        r = align_quote("Hello world", source)
        assert r.tier == AlignmentTier.EXACT
        start, end = r.char_interval
        assert source[start:end] == "Hello world"
