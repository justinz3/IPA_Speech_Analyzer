"""Tier A smoke tests for orthography → IPA conversion + locale resolution.

These tests hit the real espeak-ng binary (it's a system requirement, not a
slow dependency) and verify the (lang, dialect) → espeak-code map plus the
public alias surface.
"""

from __future__ import annotations

import pytest

from vocal_ipa.reference import Locale, resolve_locale, text_to_ipa

# -- resolve_locale -----------------------------------------------------------


def test_resolve_locale_bare_es_uses_default_dialect():
    loc = resolve_locale("es")
    assert loc == Locale(lang="es", dialect="castilian", espeak="es")


def test_resolve_locale_bare_fr_uses_default_dialect():
    loc = resolve_locale("fr")
    assert loc == Locale(lang="fr", dialect="parisian", espeak="fr-fr")


def test_resolve_locale_composite_alias_pins_dialect():
    assert resolve_locale("es-419").dialect == "latam"
    assert resolve_locale("es-es").dialect == "castilian"


def test_resolve_locale_dialect_arg_overrides_bare_lang():
    loc = resolve_locale("es", dialect="latam")
    assert loc == Locale(lang="es", dialect="latam", espeak="es-419")


def test_resolve_locale_composite_and_dialect_agreeing_is_fine():
    loc = resolve_locale("es-419", dialect="latam")
    assert loc.dialect == "latam"


def test_resolve_locale_composite_and_dialect_conflicting_raises():
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_locale("es-419", dialect="castilian")


def test_resolve_locale_unsupported_language_raises():
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
    out = text_to_ipa("manzana", lang="es", dialect="latam")
    assert "θ" not in out
    assert "s" in out


def test_spanish_composite_lang_alias_matches_dialect_arg():
    via_alias = text_to_ipa("manzana", lang="es-419")
    via_dialect = text_to_ipa("manzana", lang="es", dialect="latam")
    assert via_alias == via_dialect


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
