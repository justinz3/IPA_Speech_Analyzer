"""Tier A smoke tests for orthography → IPA conversion + locale resolution.

These tests hit the real espeak-ng binary (it's a system requirement, not a
slow dependency) and verify the (lang, dialect) → espeak-code map plus the
public alias surface. Codes ("es-es", "es-419", "fr-fr") are canonical;
human names ("castilian", "latam", "parisian") are aliases.
"""

from __future__ import annotations

import pytest

from vocal_ipa.reference import Locale, resolve_locale, text_to_ipa

# -- resolve_locale -----------------------------------------------------------


def test_resolve_locale_bare_es_uses_default_dialect():
    loc = resolve_locale("es")
    assert loc == Locale(lang="es", dialect="es-es", espeak="es")


def test_resolve_locale_bare_fr_uses_default_dialect():
    loc = resolve_locale("fr")
    assert loc == Locale(lang="fr", dialect="fr-fr", espeak="fr-fr")


def test_resolve_locale_composite_alias_pins_dialect():
    assert resolve_locale("es-419").dialect == "es-419"
    assert resolve_locale("es-es").dialect == "es-es"


def test_resolve_locale_dialect_arg_accepts_canonical_code():
    loc = resolve_locale("es", dialect="es-419")
    assert loc == Locale(lang="es", dialect="es-419", espeak="es-419")


def test_resolve_locale_dialect_arg_accepts_human_alias():
    # 'latam' is an alias that resolves to canonical code 'es-419'.
    loc = resolve_locale("es", dialect="latam")
    assert loc.dialect == "es-419"


def test_resolve_locale_castilian_alias_resolves_to_es_es():
    loc = resolve_locale("es", dialect="castilian")
    assert loc.dialect == "es-es"
    assert loc.espeak == "es"


def test_resolve_locale_composite_and_dialect_agreeing_is_fine():
    # Code form.
    loc = resolve_locale("es-419", dialect="es-419")
    assert loc.dialect == "es-419"
    # Alias form (latam → es-419, agrees with es-419 lang code).
    loc = resolve_locale("es-419", dialect="latam")
    assert loc.dialect == "es-419"


def test_resolve_locale_composite_and_dialect_conflicting_raises():
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_locale("es-419", dialect="castilian")


def test_resolve_locale_unsupported_language_raises():
    # 'ja' (Japanese) is reserved for Phase 5b; not yet wired up.
    with pytest.raises(ValueError, match="Unsupported language"):
        resolve_locale("ja")


def test_resolve_locale_unsupported_dialect_raises():
    with pytest.raises(ValueError, match="Unsupported dialect"):
        resolve_locale("es", dialect="venusian")


# -- text_to_ipa --------------------------------------------------------------


def test_spanish_basic():
    assert text_to_ipa("hola", lang="es") == "ola"


def test_spanish_default_is_castilian_with_theta():
    # Castilian distinguishes /θ/ on `z` and soft `c`; bare lang="es" must
    # still produce that to preserve Phase 2 fixture goldens.
    assert "θ" in text_to_ipa("manzana", lang="es")


def test_spanish_latam_dialect_drops_theta():
    out = text_to_ipa("manzana", lang="es", dialect="es-419")
    assert "θ" not in out
    assert "s" in out


def test_spanish_latam_alias_works():
    # 'latam' alias should give the same output as the canonical 'es-419' code.
    via_alias = text_to_ipa("manzana", lang="es", dialect="latam")
    via_code = text_to_ipa("manzana", lang="es", dialect="es-419")
    assert via_alias == via_code


def test_spanish_composite_lang_alias_matches_dialect_arg():
    via_lang_alias = text_to_ipa("manzana", lang="es-419")
    via_dialect = text_to_ipa("manzana", lang="es", dialect="es-419")
    assert via_lang_alias == via_dialect


def test_french_basic_routes_through_fr_fr_espeak_code():
    # `lang="fr"` must be mapped to espeak's "fr-fr" internally; bare "fr"
    # would raise. The output is the canonical phonemizer realization for
    # "lune" — front rounded /y/, no length marks, no stress.
    assert text_to_ipa("lune", lang="fr") == "lyn"


def test_french_nasal_vowels_emitted_as_combining_diacritics():
    # Each nasal is the base vowel + U+0303 (combining tilde). Whitespace
    # between words is preserved (tokenization happens downstream).
    out = text_to_ipa("vin blanc", lang="fr")
    assert "ɛ̃" in out
    assert "ɑ̃" in out


def test_unknown_language_raises_unsupported():
    # Now that resolve_locale gates language, the error fires before
    # phonemizer is touched.
    with pytest.raises(ValueError, match="Unsupported language"):
        text_to_ipa("hello", lang="xx-xx")


# -- Mandarin (Phase 5a) ------------------------------------------------------


def test_mandarin_resolve_locale_cmn():
    loc = resolve_locale("cmn")
    assert loc == Locale(lang="cmn", dialect="cmn-cn", espeak="cmn-latn-pinyin")


def test_mandarin_resolve_locale_zh_alias():
    # ISO 639-1 macro code 'zh' is accepted as an alias for cmn.
    loc = resolve_locale("zh")
    assert loc.lang == "cmn"
    assert loc.dialect == "cmn-cn"


def test_mandarin_hanzi_input_produces_ipa():
    # 你好 → routes through pypinyin → numeric pinyin → espeak cmn-latn-pinyin.
    out = text_to_ipa("你好", lang="cmn")
    # Espeak Mandarin emits the close-front vowel (with tone digit) for 'ni'
    # and a uvular fricative + diphthong+tone for 'hao'. The exact tone
    # digits are espeak's internal encoding (not pinyin tone numbers).
    assert "i" in out
    assert "χ" in out


def test_mandarin_diacritic_input_matches_hanzi():
    # The three input forms — Hanzi, diacritic pinyin, numeric pinyin —
    # must converge on identical IPA for the same utterance. This is the
    # invariant that makes pinyin support useful in the first place.
    via_hanzi = text_to_ipa("你好", lang="cmn")
    via_diacritic = text_to_ipa("nǐ hǎo", lang="cmn")
    via_numeric = text_to_ipa("ni3 hao3", lang="cmn")
    assert via_hanzi == via_diacritic == via_numeric


def test_mandarin_zh_alias_routes_to_same_path():
    via_cmn = text_to_ipa("ni3 hao3", lang="cmn")
    via_zh = text_to_ipa("ni3 hao3", lang="zh")
    assert via_cmn == via_zh


def test_mandarin_zhongguo_uses_retroflex_initial():
    # 中国 should produce espeak's retroflex affricate `ts.` (zh initial)
    # plus the velar stop `k` (g initial).
    out = text_to_ipa("中国", lang="cmn")
    assert "ts." in out
    assert "k" in out


def test_mandarin_neutral_tone_via_pypinyin_with_five():
    # pypinyin is configured with neutral_tone_with_five=True so all
    # syllables get an explicit tone digit (matches the parser's output).
    # 妈妈 — first syllable has tone 1, second is dictionary tone 1 too;
    # use a sentence where the parser-vs-pypinyin tone defaulting matters.
    out_hanzi = text_to_ipa("妈妈", lang="cmn")
    out_pinyin = text_to_ipa("ma1 ma1", lang="cmn")
    assert out_hanzi == out_pinyin


def test_spanish_path_unchanged_by_mandarin_dispatch():
    # Make sure the cmn dispatch in text_to_ipa() doesn't accidentally
    # affect non-cmn locales.
    assert text_to_ipa("hola", lang="es") == "ola"
    assert "θ" in text_to_ipa("manzana", lang="es")
