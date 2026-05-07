"""Fetch short clips from MLS test split for use as test fixtures.

LibriVox-derived (CC-BY 4.0); zero training-set leakage with the wav2vec2 model
(which was fine-tuned on Common Voice, not LibriVox).

Run after `uv sync --extra fixtures`:

    uv run python scripts/fetch_fixtures.py --language es
    uv run python scripts/fetch_fixtures.py --language fr -n 10

Idempotent: skips IDs already present in the manifest. Appends new entries to
manifest.json rather than overwriting, so adding a second language doesn't
clobber the first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

# `datasets` is in the [fixtures] extra; importing it lazily inside main() keeps
# `--help` usable without the extra installed.

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "fixtures"
MANIFEST_PATH = OUT_DIR / "manifest.json"
DEFAULT_N_CLIPS = 5
# MLS clips are pre-segmented audiobook chapters; durations cluster around
# 10-15s. Anything within this window is a clean choice for snapshot/PER tests.
MIN_SECONDS = 5.0
MAX_SECONDS = 15.0
TARGET_SR = 16_000

# Public lang code → (MLS dataset config, fixture id prefix).
_LANG_CONFIG = {
    "es": ("spanish", "es"),
    "fr": ("french", "fr"),
}


def _to_16k_mono(samples: np.ndarray, sr: int) -> np.ndarray:
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32, copy=False)
    if sr == TARGET_SR:
        return samples
    import torch
    import torchaudio.functional as F

    tensor = torch.from_numpy(samples).unsqueeze(0)
    return F.resample(tensor, sr, TARGET_SR).squeeze(0).numpy()


def _load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _save_manifest(entries: list[dict]) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--language",
        choices=sorted(_LANG_CONFIG),
        default="es",
        help="Language to fetch clips for (default: es).",
    )
    parser.add_argument(
        "-n",
        "--num-clips",
        type=int,
        default=DEFAULT_N_CLIPS,
        help=f"How many new clips to fetch (default: {DEFAULT_N_CLIPS}).",
    )
    args = parser.parse_args()

    config_name, prefix = _LANG_CONFIG[args.language]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()
    existing_ids = {entry["id"] for entry in manifest}
    next_idx = 1 + max(
        (
            int(entry["id"].split("_")[1])
            for entry in manifest
            if entry["id"].startswith(f"{prefix}_")
        ),
        default=0,
    )
    target_count = args.num_clips
    added: list[dict] = []

    print(
        f"Streaming facebook/multilingual_librispeech ({config_name}, test) — "
        f"target: {target_count} new {args.language} clip(s)..."
    )
    from datasets import load_dataset

    ds = load_dataset(
        "facebook/multilingual_librispeech",
        config_name,
        split="test",
        streaming=True,
    )

    for sample in ds:
        if len(added) >= target_count:
            break
        audio = sample["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        if duration < MIN_SECONDS or duration > MAX_SECONDS:
            continue

        clip_id = f"{prefix}_{next_idx:03d}"
        if clip_id in existing_ids:
            next_idx += 1
            continue
        next_idx += 1

        flac_path = OUT_DIR / f"{clip_id}.flac"
        txt_path = OUT_DIR / f"{clip_id}.txt"

        samples = _to_16k_mono(np.asarray(audio["array"]), audio["sampling_rate"])
        sf.write(flac_path, samples, TARGET_SR)
        txt_path.write_text(sample["transcript"].strip() + "\n", encoding="utf-8")

        entry = {
            "id": clip_id,
            "audio": flac_path.name,
            "transcript": txt_path.name,
            "speaker_id": str(sample.get("speaker_id", "")),
            "duration_seconds": round(duration, 2),
            "language": args.language,
            "source_dataset": "facebook/multilingual_librispeech",
            "config": config_name,
            "split": "test",
            "license": "CC-BY 4.0 (LibriVox public-domain audio)",
        }
        added.append(entry)
        print(f"  kept {clip_id}: {duration:.2f}s — {sample['transcript'][:70]!r}")

    if not added:
        print("Nothing new to add (manifest already has the requested count).")
        return

    manifest.extend(added)
    _save_manifest(manifest)
    print(f"\nAppended {len(added)} fixture(s) to {MANIFEST_PATH}")
    print(
        "Next: uv run python scripts/regen_goldens.py && "
        "uv run python scripts/regen_score_goldens.py"
    )


if __name__ == "__main__":
    main()
