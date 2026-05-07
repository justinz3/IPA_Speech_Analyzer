"""Fetch 5 short Spanish clips from MLS test split for use as test fixtures.

LibriVox-derived (CC-BY 4.0); zero training-set leakage with the wav2vec2 model
(which was fine-tuned on Common Voice, not LibriVox).

Run after `uv sync --extra fixtures`:

    uv run python scripts/fetch_fixtures.py

Idempotent: overwrites existing fixtures and manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "fixtures"
N_CLIPS = 5
# MLS clips are pre-segmented audiobook chapters; durations cluster around
# 10-15s. Anything within this window is a clean choice for snapshot/PER tests.
MIN_SECONDS = 5.0
MAX_SECONDS = 15.0
TARGET_SR = 16_000


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Streaming facebook/multilingual_librispeech (Spanish, test)...")
    ds = load_dataset(
        "facebook/multilingual_librispeech",
        "spanish",
        split="test",
        streaming=True,
    )

    manifest: list[dict] = []
    for sample in ds:
        if len(manifest) >= N_CLIPS:
            break
        audio = sample["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        if duration < MIN_SECONDS or duration > MAX_SECONDS:
            continue

        idx = len(manifest) + 1
        clip_id = f"es_{idx:03d}"
        flac_path = OUT_DIR / f"{clip_id}.flac"
        txt_path = OUT_DIR / f"{clip_id}.txt"

        samples = _to_16k_mono(np.asarray(audio["array"]), audio["sampling_rate"])
        sf.write(flac_path, samples, TARGET_SR)
        txt_path.write_text(sample["transcript"].strip() + "\n", encoding="utf-8")

        manifest.append(
            {
                "id": clip_id,
                "audio": flac_path.name,
                "transcript": txt_path.name,
                "speaker_id": str(sample.get("speaker_id", "")),
                "duration_seconds": round(duration, 2),
                "source_dataset": "facebook/multilingual_librispeech",
                "config": "spanish",
                "split": "test",
                "license": "CC-BY 4.0 (LibriVox public-domain audio)",
            }
        )
        print(f"  kept {clip_id}: {duration:.2f}s — {sample['transcript'][:70]!r}")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nSaved {len(manifest)} fixtures to {OUT_DIR}")
    print("Next: uv run python scripts/regen_goldens.py")


if __name__ == "__main__":
    main()
