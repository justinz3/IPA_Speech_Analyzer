"""Phoneme-error-rate metric for IPA strings."""

from __future__ import annotations

from jiwer import cer


def phoneme_error_rate(hypothesis: str, reference: str) -> float:
    """Levenshtein distance between two IPA strings, normalized by reference length.

    Whitespace is stripped from both inputs before comparison: the wav2vec2
    model emits one phoneme per space-separated token while phonemizer emits
    only word-boundary spaces, and we want to compare phoneme content rather
    than tokenization.

    Implemented as character error rate (`jiwer.cer`). Most espeak IPA tokens
    are single Unicode codepoints, so CER ≈ PER for our purposes; multi-
    codepoint phonemes (e.g., affricates with combining ties) are rare enough
    in Spanish that the approximation is fine for soft-oracle checks.
    """
    hyp = "".join(hypothesis.split())
    ref = "".join(reference.split())
    return float(cer(ref, hyp))
