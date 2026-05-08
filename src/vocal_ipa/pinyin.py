"""Pinyin notation normalizer: diacritic / numeric / mixed → numeric form.

Used by reference.py's Mandarin path. Espeak's ``cmn-latn-pinyin`` voice
expects pinyin with numeric tone digits ("ni3 hao3"); users typing
diacritic pinyin ("nǐ hǎo") or unmarked neutral-tone syllables ("de")
need normalization first. Pure Python, no external deps.
"""

from __future__ import annotations

import re
import unicodedata

# Diacritic-tone marks → (base vowel, tone number). Covers the five vowels
# that pinyin marks (a, e, i, o, u) plus ü.
_DIACRITIC_TO_TONE: dict[str, tuple[str, int]] = {
    "ā": ("a", 1),
    "á": ("a", 2),
    "ǎ": ("a", 3),
    "à": ("a", 4),
    "ē": ("e", 1),
    "é": ("e", 2),
    "ě": ("e", 3),
    "è": ("e", 4),
    "ī": ("i", 1),
    "í": ("i", 2),
    "ǐ": ("i", 3),
    "ì": ("i", 4),
    "ō": ("o", 1),
    "ó": ("o", 2),
    "ǒ": ("o", 3),
    "ò": ("o", 4),
    "ū": ("u", 1),
    "ú": ("u", 2),
    "ǔ": ("u", 3),
    "ù": ("u", 4),
    "ǖ": ("ü", 1),
    "ǘ": ("ü", 2),
    "ǚ": ("ü", 3),
    "ǜ": ("ü", 4),
}

# Whitespace + common punctuation + apostrophe + hyphen all serve as syllable
# boundaries. Apostrophe is the standard pinyin syllable disambiguator
# (Xī'ān ≠ xiān); hyphens show up in compound spellings.
_BOUNDARY_RE = re.compile(r"[\s.,!?;:\"'\-]+")

# A syllable already in numeric form: any prefix followed by a tone digit.
_NUMERIC_TONE_RE = re.compile(r"^(.+?)([1-5])$")


def parse_pinyin_text(text: str) -> list[str]:
    """Normalize pinyin text into a list of numeric-tone syllable strings.

    Accepts diacritic ("nǐ hǎo"), numeric ("ni3 hao3"), or mixed input.
    Bare syllables with no tone mark default to neutral tone 5 ("de" →
    "de5"). Output is always lowercase with explicit tone digits.
    """
    normalized = unicodedata.normalize("NFC", text.lower())
    return [_parse_one_syllable(s) for s in _BOUNDARY_RE.split(normalized) if s]


def strip_tone(syllable: str) -> str:
    """Remove the trailing tone digit from a numeric-form syllable."""
    if syllable and syllable[-1] in "12345":
        return syllable[:-1]
    return syllable


def _parse_one_syllable(syllable: str) -> str:
    m = _NUMERIC_TONE_RE.match(syllable)
    if m:
        return f"{_normalize_v_to_umlaut(m.group(1))}{m.group(2)}"

    out_chars: list[str] = []
    tone = 5
    for ch in syllable:
        base, t = _DIACRITIC_TO_TONE.get(ch, (ch, None))
        out_chars.append(base)
        if t is not None:
            tone = t
    return f"{_normalize_v_to_umlaut(''.join(out_chars))}{tone}"


def _normalize_v_to_umlaut(syllable: str) -> str:
    """``lv`` / ``nv`` are common typed substitutes for ``lü`` / ``nü``."""
    if syllable.startswith(("l", "n")) and "v" in syllable:
        return syllable.replace("v", "ü")
    return syllable
