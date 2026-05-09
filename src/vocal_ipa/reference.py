"""Orthographic text -> reference IPA via espeak-ng.

System requirement: `espeak-ng` binary on PATH (`sudo apt install espeak-ng`).
The python `phonemizer` package wraps the binary; without it, calls raise
RuntimeError at first use (not at import time).

Locale model: a (lang, dialect) pair. `lang` is the canonical ISO-639-1/-3
code ("es", "fr", "cmn"); `dialect` is a canonical *locale code* ("es-es",
"es-419", "fr-fr", "cmn-cn") — codes are the canonical identifiers, with
human names like "castilian" / "latam" / "parisian" accepted as aliases.
Internally each (lang, dialect) maps to an espeak voice code, which mostly
matches the dialect code but isn't required to (espeak's codes are
inconsistent — bare "es" works but bare "fr" doesn't, and Mandarin's
default `cmn` voice falls back to English mid-utterance, so we use
`cmn-latn-pinyin` and feed it numeric pinyin instead of Hanzi).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from phonemizer import phonemize

from .pinyin import parse_pinyin_text


@dataclass(frozen=True)
class Locale:
    lang: str  # canonical: "es" or "fr"
    dialect: str  # canonical short name: "castilian", "latam", "parisian", ...
    espeak: str  # espeak language code: "es", "es-419", "fr-fr", "fr-ca", ...


# Canonical (lang, dialect) -> espeak code. Dialects are *codes*
# ("es-es", "es-419", "fr-fr"), not human names — see _DIALECT_ALIASES
# below for human-name → code resolution.
#
# Dialect support is reference-source-limited: espeak-ng's IPA output is
# *identical* across French regional voices (fr-fr, fr-be, fr-ch, fr-ca) —
# only the synthesized speech timbre differs, not the phonemic rules. And
# `es-mx` produces the same IPA as `es-419`. So the only dialect distinction
# that actually changes reference IPA is Spanish Castilian (es-es, /θ/ on
# z, soft c) vs Latin American (es-419, /s/ instead). Real Quebec/Belgian
# dialect handling would need a different reference source than espeak.
_DIALECT_MAP: dict[tuple[str, str], str] = {
    ("es", "es-es"): "es",
    ("es", "es-419"): "es-419",
    ("fr", "fr-fr"): "fr-fr",
    # Mandarin uses espeak's pinyin-input voice; the default `cmn` voice
    # falls back to English IPA mid-utterance and is unusable.
    ("cmn", "cmn-cn"): "cmn-latn-pinyin",
    # Japanese bypasses espeak entirely (espeak's `ja` voice falls back
    # to English mid-utterance); the value is a label only — text_to_ipa
    # routes through pyopenjtalk before the espeak field is consulted.
    ("ja", "ja-jp"): "ja",
}

DEFAULT_DIALECT: dict[str, str] = {
    "es": "es-es",
    "fr": "fr-fr",
    "cmn": "cmn-cn",
    "ja": "ja-jp",
}

# Human-friendly aliases that resolve to canonical dialect codes.
_DIALECT_ALIASES: dict[str, str] = {
    "castilian": "es-es",
    "latam": "es-419",
    "es-latam": "es-419",
    "parisian": "fr-fr",
}

# Codes accepted on `lang`. Bare codes ("es", "fr") are unspecified-dialect
# and let an explicit `dialect` arg override silently. Composite codes
# ("es-419", "es-es", ...) pin a dialect; passing a conflicting `dialect`
# arg alongside one is an error.
_BARE_LANGS = frozenset({"es", "fr", "cmn", "ja"})

_LANG_ALIASES: dict[str, tuple[str, str]] = {
    "es": ("es", "es-es"),
    "es-es": ("es", "es-es"),
    "es-419": ("es", "es-419"),
    "es-latam": ("es", "es-419"),
    "fr": ("fr", "fr-fr"),
    "fr-fr": ("fr", "fr-fr"),
    "cmn": ("cmn", "cmn-cn"),
    "cmn-cn": ("cmn", "cmn-cn"),
    "zh": ("cmn", "cmn-cn"),  # ISO 639-1 macro code, common alias
    "ja": ("ja", "ja-jp"),
    "ja-jp": ("ja", "ja-jp"),
}


def supported_languages() -> frozenset[str]:
    """Canonical languages — used by callers that want to gate on language."""
    return frozenset(DEFAULT_DIALECT)


def resolve_locale(lang: str, dialect: str | None = None) -> Locale:
    """Single normalization point for (lang, dialect) inputs.

    - `lang` may be a canonical code ("es", "fr") or any alias from
      _LANG_ALIASES (e.g., "es-419", "es-latam").
    - `dialect`, if given, may be a canonical code ("es-419", "fr-fr") or
      a human alias from _DIALECT_ALIASES ("castilian", "latam"). Both
      forms normalize to the same canonical code.
    - If `dialect` is None, uses the language's default (via the lang alias).
    - Composite lang aliases that disagree with an explicit dialect raise.
    """
    if lang not in _LANG_ALIASES:
        sup = sorted(_LANG_ALIASES)
        raise ValueError(
            f"Unsupported language {lang!r}; supported: {sup}. See pronunciation_app_roadmap.md."
        )
    canonical_lang, alias_dialect = _LANG_ALIASES[lang]

    if dialect is None:
        chosen = alias_dialect
    else:
        # Normalize human-name aliases to canonical codes before validating.
        normalized = _DIALECT_ALIASES.get(dialect, dialect)
        if (canonical_lang, normalized) not in _DIALECT_MAP:
            valid_codes = sorted(d for (lc, d) in _DIALECT_MAP if lc == canonical_lang)
            valid_aliases = sorted(a for a, c in _DIALECT_ALIASES.items() if c in valid_codes)
            raise ValueError(
                f"Unsupported dialect {dialect!r} for language {canonical_lang!r}; "
                f"valid codes: {valid_codes}; aliases: {valid_aliases}."
            )
        if lang not in _BARE_LANGS and normalized != alias_dialect:
            raise ValueError(
                f"Conflicting language code and dialect: lang={lang!r} implies "
                f"dialect={alias_dialect!r}, but dialect={dialect!r} was given."
            )
        chosen = normalized

    return Locale(
        lang=canonical_lang,
        dialect=chosen,
        espeak=_DIALECT_MAP[(canonical_lang, chosen)],
    )


def text_to_ipa(text: str, lang: str = "es", dialect: str | None = None) -> str:
    """Convert orthographic text to a reference IPA string.

    Output is whitespace-separated, with stress and word boundaries stripped
    so it lines up with what the wav2vec2 phoneme model emits. The reference
    is itself an approximation (espeak rules are imperfect, dialect varies),
    so use it as a soft oracle for PER, not absolute ground truth.

    Mandarin (lang="cmn"/"zh") routes through espeak's `cmn-latn-pinyin`
    voice via pypinyin or the diacritic-pinyin parser. Japanese (lang="ja")
    routes through pyopenjtalk + a per-phoneme IPA map (espeak's `ja` voice
    is broken — falls back to English IPA mid-utterance).
    """
    locale = resolve_locale(lang, dialect)
    if locale.lang == "cmn":
        return _mandarin_text_to_ipa(text, locale.espeak)
    if locale.lang == "ja":
        return _japanese_text_to_ipa(text)
    out = phonemize(
        text,
        language=locale.espeak,
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
    )
    return " ".join(out.split())


# Hanzi range: CJK Unified Ideographs (covers the bulk of modern Chinese).
# Doesn't include the Extension blocks; rare/historical characters won't
# trigger Hanzi mode and would land in the pinyin parser as gibberish.
_HANZI_RE = re.compile(r"[一-鿿]")


def _has_hanzi(text: str) -> bool:
    return bool(_HANZI_RE.search(text))


def _mandarin_text_to_ipa(text: str, espeak_voice: str) -> str:
    """Hanzi or pinyin text → IPA via espeak's pinyin-input voice.

    Hanzi input flows through pypinyin → numeric-tone pinyin string. Pinyin
    input flows through the diacritic→numeric normalizer in `pinyin.py`.
    Both end up as space-separated pinyin syllables fed to espeak's
    `cmn-latn-pinyin` voice, which emits IPA in the convention the
    wav2vec2 phoneme model was trained on.
    """
    if _has_hanzi(text):
        # pypinyin is a Chinese-specific dep; only imported when actually
        # needed so non-Mandarin paths don't pay the import cost.
        from pypinyin import Style, lazy_pinyin

        syllables = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    else:
        syllables = parse_pinyin_text(text)
    pinyin_str = " ".join(syllables)
    out = phonemize(
        pinyin_str,
        language=espeak_voice,
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
    )
    return " ".join(out.split())


# pyopenjtalk emits its own romaji-style phoneme set; we map each phoneme to
# the closest IPA token available in the wav2vec2 model vocab. Some surface
# distinctions are dropped: gʲ collapses to ɡ (no gʲ in vocab), devoiced
# vowels (capital I/U) collapse to plain forms, and the geminate marker `cl`
# is replaced by the glottal stop ʔ (best available in vocab; not phonetically
# accurate but at least represents a closure event the model can recognize).
_PYOPENJTALK_TO_IPA: dict[str, str] = {
    # Vowels
    "a": "a",
    "i": "i",
    "u": "u",
    "e": "e",
    "o": "o",
    "I": "i",
    "U": "u",  # devoiced — collapse to plain (not in vocab)
    # Stops
    "k": "k",
    "ky": "kʲ",
    "g": "ɡ",
    "gy": "ɡ",  # vocab has no gʲ, drop palatalization
    "p": "p",
    "py": "pʲ",
    "b": "b",
    "by": "bʲ",
    "t": "t",
    "d": "d",
    # Affricates
    "ts": "ts",
    "ch": "tɕ",
    "j": "dʑ",
    # Fricatives
    "s": "s",
    "sh": "ɕ",
    "z": "z",
    "h": "h",
    "hy": "ç",
    "f": "ɸ",
    # Nasals
    "n": "n",
    "ny": "ɲ",
    "m": "m",
    "my": "mʲ",
    # Liquids — Japanese r is alveolar tap; rʲ (palatalized) is in vocab,
    # ɾʲ is not, so palatalized "ry" maps to rʲ for vocab compatibility.
    "r": "ɾ",
    "ry": "rʲ",
    # Glides
    "w": "w",
    "y": "j",
    # Special
    "N": "ɴ",  # moraic nasal (-ん)
    "cl": "ʔ",  # geminate marker (best vocab approximation)
    "pau": "",  # pause — drop entirely
}

# Vowels eligible for length-marker collapse (consecutive same vowels →
# Vː tokens that the model emits as single units).
_LENGTH_BASE_VOWELS = frozenset({"a", "i", "u", "e", "o"})


def _japanese_text_to_ipa(text: str) -> str:
    """Kanji/kana text → IPA via pyopenjtalk + per-phoneme map.

    pyopenjtalk handles all three Japanese scripts internally and emits
    space-separated phonemes in its own romaji-style notation. We map each
    phoneme to a model-vocab IPA token, then collapse consecutive same
    vowels into a single Vː token (which the wav2vec2 vocab represents as a
    single entry — `aː`, `iː`, `uː`, `eː`, `oː` are all in vocab).
    """
    import pyopenjtalk

    raw_phonemes = pyopenjtalk.g2p(text).split()
    ipa_tokens: list[str] = []
    for phoneme in raw_phonemes:
        ipa = _PYOPENJTALK_TO_IPA.get(phoneme)
        if ipa is None or ipa == "":
            continue
        # Collapse consecutive same-vowel pairs into Vː (e.g. "o o" → "oː").
        if ipa in _LENGTH_BASE_VOWELS and ipa_tokens and ipa_tokens[-1] == ipa:
            ipa_tokens[-1] = f"{ipa}ː"
            continue
        ipa_tokens.append(ipa)
    return " ".join(ipa_tokens)
