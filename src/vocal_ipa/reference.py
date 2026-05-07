"""Orthographic text -> reference IPA via espeak-ng.

System requirement: `espeak-ng` binary on PATH (`sudo apt install espeak-ng`).
The python `phonemizer` package wraps the binary; without it, calls raise
RuntimeError at first use (not at import time).
"""

from __future__ import annotations

from phonemizer import phonemize


def text_to_ipa(text: str, lang: str = "es") -> str:
    """Convert orthographic text to a reference IPA string.

    Output is whitespace-separated, with stress and word boundaries stripped
    so it lines up with what the wav2vec2 phoneme model emits. The reference
    is itself an approximation (espeak rules are imperfect, dialect varies),
    so use it as a soft oracle for PER, not absolute ground truth.
    """
    out = phonemize(
        text,
        language=lang,
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
    )
    return " ".join(out.split())
