"""Regenerate golden IPA snapshots for snapshot tests.

Run after fetching fixtures, OR when intentionally accepting model output drift
(e.g., after upgrading transformers / torch). Writes <id>.golden.ipa next to
each fixture.

    uv run python scripts/regen_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

from vocal_ipa.pipeline import transcribe

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "fixtures"


def main() -> None:
    manifest_path = FIXTURE_DIR / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"No manifest at {manifest_path}. Run scripts/fetch_fixtures.py first.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest:
        clip_id = entry["id"]
        # Pre-Phase-3 manifests didn't carry language; default to es for backcompat.
        lang = entry.get("language", "es")
        audio_path = FIXTURE_DIR / entry["audio"]
        result = transcribe(audio_path, lang=lang)
        out_path = FIXTURE_DIR / f"{clip_id}.golden.ipa"
        out_path.write_text(result.ipa + "\n", encoding="utf-8")
        print(f"{clip_id} ({lang}): {result.ipa}")


if __name__ == "__main__":
    main()
