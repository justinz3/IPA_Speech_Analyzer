"""Tier C: behavioral tests for per-phoneme scoring.

Slow (loads the wav2vec2 model). Validates *approximate correctness* —
the scoring path should mostly agree with the audio when the reference
matches it, and disagree loudly when the reference doesn't.

If snapshot tests break but these stay green → benign drift, regenerate.
If these break → real model/alignment regression to investigate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vocal_ipa.score import score

FIXTURES = Path(__file__).parent / "data" / "fixtures"
MANIFEST = FIXTURES / "manifest.json"

# Per-language thresholds. es/fr LibriVox clips land at 0.07-0.24 with the
# threshold loosened to 0.30 to absorb fixture-by-fixture variation without
# turning into a regression target. Mandarin (ST-CMDS) lands at 0.30-0.63 due
# to espeak/model tone-encoding mismatch (see test_per.py) — set to 0.65 so
# the worst observed fixture still passes with a small drift margin. Phase 6
# prosody work should tighten cmn once tone scoring moves out of segmental.
SELF_CONSISTENCY_PER_THRESHOLDS = {
    "es": 0.30,
    "fr": 0.30,
    "cmn": 0.65,
    # Japanese is very loose — see test_per.py for the model-vs-reference
    # mismatch explanation. JSUT fixtures land 0.44-0.82; threshold gives
    # a small margin above the worst observed fixture.
    "ja": 0.85,
}
CROSS_PAIR_PER_FLOOR = 0.50


def _all_entries() -> list[dict]:
    if not MANIFEST.exists():
        return []
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _fixture_params():
    return [pytest.param(entry, id=entry["id"]) for entry in _all_entries()]


def _transcript(clip_id: str) -> str:
    return (FIXTURES / f"{clip_id}.txt").read_text(encoding="utf-8").strip()


@pytest.mark.slow
@pytest.mark.parametrize("entry", _fixture_params())
def test_score_self_consistency(entry: dict) -> None:
    """Audio against its own transcript should land under the PER threshold."""
    clip_id = entry["id"]
    lang = entry["language"]
    audio_path = FIXTURES / f"{clip_id}.flac"
    transcript = _transcript(clip_id)
    result = score(audio_path, transcript, lang=lang)
    threshold = SELF_CONSISTENCY_PER_THRESHOLDS[lang]
    assert result.per < threshold, f"{clip_id} ({lang}): PER {result.per:.3f} >= {threshold}"


@pytest.mark.slow
def test_score_cross_pair_detects_mismatch() -> None:
    """Audio against a *different* clip's transcript should produce many errors.

    Restricted to same-language pairs — a cross-language pair would conflate
    'wrong reference' with 'wrong-language phoneme inventory.' We use Spanish
    as the original validated path; if no Spanish pair exists, fall back to
    the first language with at least two fixtures.
    """
    by_lang: dict[str, list[str]] = {}
    for entry in _all_entries():
        by_lang.setdefault(entry["language"], []).append(entry["id"])
    candidates = ["es"] + [lang for lang in by_lang if lang != "es"]
    pair_lang = next((lang for lang in candidates if len(by_lang.get(lang, [])) >= 2), None)
    if pair_lang is None:
        pytest.skip("Need 2+ fixtures of the same language for a cross-pair test")

    ids = by_lang[pair_lang]
    audio_path = FIXTURES / f"{ids[0]}.flac"
    wrong_transcript = _transcript(ids[1])
    result = score(audio_path, wrong_transcript, lang=pair_lang)
    assert result.per > CROSS_PAIR_PER_FLOOR, (
        f"Mismatched {pair_lang} audio/transcript pair only produced PER {result.per:.3f}; "
        f"expected > {CROSS_PAIR_PER_FLOOR} (scoring should clearly detect a wrong reference)"
    )
