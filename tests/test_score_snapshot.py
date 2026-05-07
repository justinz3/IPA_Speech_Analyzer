"""Tier B: snapshot tests for per-phoneme scoring against frozen goldens.

Marked `slow` because each test loads the wav2vec2 model. Catches
*alignment regressions* — any change in torchaudio.forced_align /
merge_tokens / our tokenization that shifts spans or per-span argmax
trips this. When the diff is intentional, regenerate via:

    uv run python scripts/regen_score_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vocal_ipa.score import score

FIXTURES = Path(__file__).parent / "data" / "fixtures"
MANIFEST = FIXTURES / "manifest.json"


def _fixture_params():
    if not MANIFEST.exists():
        return []
    return [
        pytest.param(entry, id=entry["id"])
        for entry in json.loads(MANIFEST.read_text(encoding="utf-8"))
    ]


@pytest.mark.slow
@pytest.mark.parametrize("entry", _fixture_params())
def test_score_snapshot_matches_golden(entry: dict) -> None:
    clip_id = entry["id"]
    lang = entry["language"]
    audio_path = FIXTURES / f"{clip_id}.flac"
    transcript_path = FIXTURES / f"{clip_id}.txt"
    golden_path = FIXTURES / f"{clip_id}.golden.score.json"
    if not golden_path.exists():
        pytest.skip(f"No golden for {clip_id}; run scripts/regen_score_goldens.py")

    transcript = transcript_path.read_text(encoding="utf-8").strip()
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    result = score(audio_path, transcript, lang=lang)
    actual = [
        {
            "expected": p.expected,
            "produced": p.produced,
            "start_s": round(p.start_s, 2),
            "end_s": round(p.end_s, 2),
            "ok": p.ok,
        }
        for p in result.phonemes
    ]

    assert actual == golden, (
        f"\n  clip: {clip_id} ({lang})"
        f"\n  Snapshot drift detected. If intentional, regenerate via:"
        f"\n    uv run python scripts/regen_score_goldens.py"
    )
