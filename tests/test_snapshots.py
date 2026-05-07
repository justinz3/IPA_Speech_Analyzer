"""Tier B: snapshot tests against frozen IPA goldens.

Marked `slow` because each test loads the wav2vec2 model (~1GB on first run).
Catches *regressions* — any change in transformers / torch / postprocess that
shifts model output trips this. When the diff is intentional, regenerate via:

    uv run python scripts/regen_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vocal_ipa.pipeline import transcribe

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
def test_snapshot_matches_golden(entry: dict) -> None:
    clip_id = entry["id"]
    lang = entry["language"]
    audio_path = FIXTURES / f"{clip_id}.flac"
    golden_path = FIXTURES / f"{clip_id}.golden.ipa"
    if not golden_path.exists():
        pytest.skip(f"No golden for {clip_id}; run scripts/regen_goldens.py")

    expected = golden_path.read_text(encoding="utf-8").strip()
    result = transcribe(audio_path, lang=lang)
    assert result.ipa == expected, (
        f"\n  clip:     {clip_id} ({lang})"
        f"\n  expected: {expected!r}"
        f"\n  got:      {result.ipa!r}"
        f"\n  Regenerate via: uv run python scripts/regen_goldens.py"
    )


def test_manifest_exists_with_fixtures() -> None:
    """Sanity check that fixtures were fetched. Not slow."""
    if not MANIFEST.exists():
        pytest.skip("Fixtures not fetched; run scripts/fetch_fixtures.py")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert len(manifest) >= 1
    for entry in manifest:
        assert (FIXTURES / entry["audio"]).exists()
        assert (FIXTURES / entry["transcript"]).exists()
