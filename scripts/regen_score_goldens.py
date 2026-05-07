"""Regenerate score-snapshot goldens for Phase 2 snapshot tests.

Writes <id>.golden.score.json next to each fixture. Each entry in the
list captures expected/produced/timing/ok per phoneme; raw float scores
are intentionally omitted (they drift with floating-point ops and would
make the snapshot brittle without catching anything useful).

Run after fetching fixtures or when intentionally accepting alignment
drift (transformers/torchaudio upgrade):

    uv run python scripts/regen_score_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

from vocal_ipa.score import score

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "fixtures"


def _round(x: float) -> float:
    # 20 ms frame stride — 2 decimals is the natural granularity.
    return round(x, 2)


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
        transcript_path = FIXTURE_DIR / entry["transcript"]
        transcript = transcript_path.read_text(encoding="utf-8").strip()

        result = score(audio_path, transcript, lang=lang)
        snapshot = [
            {
                "expected": p.expected,
                "produced": p.produced,
                "start_s": _round(p.start_s),
                "end_s": _round(p.end_s),
                "ok": p.ok,
            }
            for p in result.phonemes
        ]
        out_path = FIXTURE_DIR / f"{clip_id}.golden.score.json"
        out_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        wrong = sum(1 for p in result.phonemes if not p.ok)
        print(
            f"{clip_id} ({lang}): PER {result.per:.3f} "
            f"({wrong}/{len(result.phonemes)} wrong) -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
