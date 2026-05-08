"""Orthographic text -> reference IPA via espeak-ng.

System requirement: `espeak-ng` binary on PATH (`sudo apt install espeak-ng`).
The python `phonemizer` package wraps the binary; without it, calls raise
RuntimeError at first use (not at import time).

Locale model: a (lang, dialect) pair. `lang` is the project's canonical
ISO-639-1 code ("es", "fr"); `dialect` is a project-canonical short name
("castilian", "latam", "parisian", "quebec", ...). Internally these resolve
to an espeak language code, since espeak's codes are inconsistent — bare
"es" works for Spanish but bare "fr" doesn't, and dialect codes vary in
shape ("es-419", "es-mx", "fr-ca").
"""

from __future__ import annotations

from dataclasses import dataclass

from phonemizer import phonemize


@dataclass(frozen=True)
class Locale:
    lang: str       # canonical: "es" or "fr"
    dialect: str    # canonical short name: "castilian", "latam", "parisian", ...
    espeak: str     # espeak language code: "es", "es-419", "fr-fr", "fr-ca", ...


# Canonical (lang, dialect) -> espeak code.
#
# Dialect support is reference-source-limited: espeak-ng's IPA output is
# *identical* across French regional voices (fr-fr, fr-be, fr-ch, fr-ca) —
# only the synthesized speech timbre differs, not the phonemic rules. And
# `es-mx` produces the same IPA as `es-419`. So the only dialect distinction
# that actually changes reference IPA is Spanish Castilian (/θ/ on z, soft c)
# vs Latin American (/s/ instead). Real Quebec/Belgian dialect handling
# would need a different reference source than espeak.
_DIALECT_MAP: dict[tuple[str, str], str] = {
    ("es", "castilian"): "es",
    ("es", "latam"):     "es-419",
    ("fr", "parisian"):  "fr-fr",
}

DEFAULT_DIALECT: dict[str, str] = {"es": "castilian", "fr": "parisian"}

# Codes accepted on `lang`. Bare codes ("es", "fr") are unspecified-dialect
# and let an explicit `dialect` arg override silently. Composite codes
# ("es-419", "es-es", ...) pin a dialect; passing a conflicting `dialect`
# arg alongside one is an error.
_BARE_LANGS = frozenset({"es", "fr"})

_LANG_ALIASES: dict[str, tuple[str, str]] = {
    "es":       ("es", "castilian"),
    "es-es":    ("es", "castilian"),
    "es-419":   ("es", "latam"),
    "es-latam": ("es", "latam"),
    "fr":       ("fr", "parisian"),
    "fr-fr":    ("fr", "parisian"),
}


def supported_languages() -> frozenset[str]:
    """Canonical languages — used by callers that want to gate on language."""
    return frozenset(DEFAULT_DIALECT)


def resolve_locale(lang: str, dialect: str | None = None) -> Locale:
    """Single normalization point for (lang, dialect) inputs.

    - `lang` may be a canonical code ("es", "fr") or any alias from
      _LANG_ALIASES (e.g., "es-419", "fr-ca").
    - `dialect`, if given, must be a canonical short name and match the lang.
    - If `dialect` is None, uses the language's default (via the alias).
    - Composite lang aliases that disagree with an explicit dialect raise.
    """
    if lang not in _LANG_ALIASES:
        sup = sorted(_LANG_ALIASES)
        raise ValueError(
            f"Unsupported language {lang!r}; supported: {sup}. "
            "See pronunciation_app_roadmap.md."
        )
    canonical_lang, alias_dialect = _LANG_ALIASES[lang]

    if dialect is None:
        chosen = alias_dialect
    else:
        if (canonical_lang, dialect) not in _DIALECT_MAP:
            valid = sorted(d for (lc, d) in _DIALECT_MAP if lc == canonical_lang)
            raise ValueError(
                f"Unsupported dialect {dialect!r} for language {canonical_lang!r}; "
                f"valid dialects: {valid}."
            )
        if lang not in _BARE_LANGS and dialect != alias_dialect:
            raise ValueError(
                f"Conflicting language code and dialect: lang={lang!r} implies "
                f"dialect={alias_dialect!r}, but dialect={dialect!r} was given."
            )
        chosen = dialect

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
    """
    locale = resolve_locale(lang, dialect)
    out = phonemize(
        text,
        language=locale.espeak,
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
    )
    return " ".join(out.split())
