"""Tier A tests for the pinyin diacritic→numeric normalizer.

The normalizer feeds espeak's ``cmn-latn-pinyin`` voice, which expects
numeric-tone pinyin. These tests pin the input shapes the parser must
accept (diacritic, numeric, mixed, neutral-defaulted).
"""

from __future__ import annotations

from vocal_ipa.pinyin import parse_pinyin_text, strip_tone

# -- parse_pinyin_text ---------------------------------------------------


class TestParsePinyinText:
    def test_numeric_form_passes_through(self):
        assert parse_pinyin_text("ni3 hao3") == ["ni3", "hao3"]

    def test_diacritic_form_normalizes_to_numeric(self):
        assert parse_pinyin_text("nǐ hǎo") == ["ni3", "hao3"]

    def test_mixed_diacritic_and_numeric(self):
        assert parse_pinyin_text("nǐ hao3") == ["ni3", "hao3"]

    def test_bare_syllable_defaults_to_neutral_tone_5(self):
        assert parse_pinyin_text("de") == ["de5"]
        assert parse_pinyin_text("de ne ma") == ["de5", "ne5", "ma5"]

    def test_all_four_tones_on_ma_numeric(self):
        # Pinyin's classic minimal-pair example.
        assert parse_pinyin_text("ma1 ma2 ma3 ma4 ma") == [
            "ma1",
            "ma2",
            "ma3",
            "ma4",
            "ma5",
        ]

    def test_all_four_tones_on_ma_diacritic(self):
        assert parse_pinyin_text("mā má mǎ mà ma") == [
            "ma1",
            "ma2",
            "ma3",
            "ma4",
            "ma5",
        ]

    def test_uppercase_lowercased(self):
        assert parse_pinyin_text("NǏ HǍO") == ["ni3", "hao3"]

    def test_v_substitute_for_umlaut_after_l_initial(self):
        assert parse_pinyin_text("lv4") == ["lü4"]
        assert parse_pinyin_text("lǜ") == ["lü4"]

    def test_v_substitute_for_umlaut_after_n_initial(self):
        assert parse_pinyin_text("nv3") == ["nü3"]

    def test_v_not_normalized_after_other_initials(self):
        # "v" doesn't otherwise occur in pinyin; defensive: don't munge it.
        assert parse_pinyin_text("xv4") == ["xv4"]

    def test_empty_string(self):
        assert parse_pinyin_text("") == []

    def test_whitespace_only(self):
        assert parse_pinyin_text("   \t\n  ") == []

    def test_punctuation_treated_as_boundary(self):
        assert parse_pinyin_text("nǐ hǎo, qǐng wèn?") == [
            "ni3",
            "hao3",
            "qing3",
            "wen4",
        ]

    def test_apostrophe_is_syllable_separator(self):
        # Pinyin's syllable disambiguator (e.g. Xī'ān vs xiān).
        assert parse_pinyin_text("xī'ān") == ["xi1", "an1"]

    def test_hyphen_is_syllable_separator(self):
        assert parse_pinyin_text("shàng-bān") == ["shang4", "ban1"]

    def test_combining_diacritic_normalized_to_precomposed(self):
        # Base vowel + combining caron (U+030C) rather than precomposed ǐ/ǎ.
        text = "nǐ hǎo"
        assert parse_pinyin_text(text) == ["ni3", "hao3"]


# -- strip_tone ----------------------------------------------------------


class TestStripTone:
    def test_strips_trailing_digit(self):
        assert strip_tone("ni3") == "ni"
        assert strip_tone("hao3") == "hao"
        assert strip_tone("ma5") == "ma"

    def test_preserves_already_toneless(self):
        assert strip_tone("ni") == "ni"

    def test_handles_empty(self):
        assert strip_tone("") == ""

    def test_only_strips_1_through_5(self):
        # Defensive: don't strip random trailing digits that aren't pinyin tones.
        assert strip_tone("ni7") == "ni7"
        assert strip_tone("ni0") == "ni0"
