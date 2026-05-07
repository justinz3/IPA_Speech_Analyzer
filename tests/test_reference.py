"""Tier A smoke tests for orthography → IPA conversion.

These tests hit the real espeak-ng binary (it's a system requirement, not a
slow dependency) and verify that each public language code maps to the right
phonemizer backend code. Catches accidental drops of the public→espeak map.
"""

from __future__ import annotations

import pytest

from vocal_ipa.reference import text_to_ipa


def test_spanish_basic():
    assert text_to_ipa("hola", lang="es") == "ola"


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


def test_unknown_language_propagates_phonemizer_error():
    # The text_to_ipa map is a thin wrapper, not a gate; only pipeline/score
    # enforce SUPPORTED_LANGUAGES. An unknown code passed straight through
    # should still fail loudly at the phonemizer boundary.
    with pytest.raises(RuntimeError, match="not supported"):
        text_to_ipa("hello", lang="xx-xx")
