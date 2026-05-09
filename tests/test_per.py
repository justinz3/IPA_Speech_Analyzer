"""Tier C: Phoneme-Error-Rate tests against phonemizer-derived references.

Marked `slow` because each test loads the wav2vec2 model. Validates *approximate
correctness* — the threshold is deliberately loose because the espeak-derived
reference is itself an approximation (dialect, allophones, espeak rule gaps).

If snapshot tests break but PER stays under threshold -> benign drift, regen
goldens. If PER also degrades -> real regression.

Requires `espeak-ng` on PATH for the phonemizer reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vocal_ipa.metrics import phoneme_error_rate
from vocal_ipa.pipeline import transcribe
from vocal_ipa.reference import text_to_ipa

FIXTURES = Path(__file__).parent / "data" / "fixtures"
MANIFEST = FIXTURES / "manifest.json"

# Loose thresholds: espeak references and the model's espeak-trained output
# disagree more than you'd think due to dialect coverage and allophone choices.
# Per-language because Mandarin's PER baseline is much higher than es/fr —
# espeak's cmn-latn-pinyin tone digits don't 1:1 match the model's tone-bearing
# vowel tokens (tones 1+4 collapse onto `5`, tone 3 emits as vowel-quality
# shift instead of a digit), which inflates segmental PER. Phase 6's prosody
# work should tighten this once tone scoring moves out of the segmental path.
PER_THRESHOLDS = {
    "es": 0.35,
    "fr": 0.35,
    "cmn": 0.70,
    # Japanese is loose — the wav2vec2-lv-60-espeak-cv-ft model emits
    # English-like phonemes (aɪ, ə, e̞ instead of Japanese vowels) when
    # fed Japanese audio. Espeak's `ja` voice is also broken, so the
    # training labels were likely degraded for ja. Slow tests stay green
    # at this threshold; better Japanese accuracy would need a Japanese-
    # fine-tuned wav2vec2 checkpoint (model swap).
    "ja": 0.85,
}


def _fixture_params():
    if not MANIFEST.exists():
        return []
    return [
        pytest.param(entry, id=entry["id"])
        for entry in json.loads(MANIFEST.read_text(encoding="utf-8"))
    ]


@pytest.mark.slow
@pytest.mark.parametrize("entry", _fixture_params())
def test_per_under_threshold(entry: dict) -> None:
    clip_id = entry["id"]
    lang = entry["language"]
    audio = FIXTURES / f"{clip_id}.flac"
    transcript = (FIXTURES / f"{clip_id}.txt").read_text(encoding="utf-8").strip()

    reference = text_to_ipa(transcript, lang=lang)
    hypothesis = transcribe(audio, lang=lang).ipa

    rate = phoneme_error_rate(hypothesis, reference)
    threshold = PER_THRESHOLDS[lang]
    assert rate < threshold, (
        f"\n  clip:       {clip_id} ({lang})"
        f"\n  PER:        {rate:.3f} (threshold {threshold})"
        f"\n  reference:  {reference!r}"
        f"\n  hypothesis: {hypothesis!r}"
    )
