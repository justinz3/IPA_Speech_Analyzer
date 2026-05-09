"""Fetch short clips for use as test fixtures.

Spanish + French come from MLS (LibriVox-derived, CC-BY 4.0; zero training-set
leakage with the wav2vec2 model, which was fine-tuned on Common Voice).
Mandarin uses ST-CMDS (urarik/free_st_chinese_mandarin_corpus, OpenSLR-38;
CC-0 / public domain). MLS doesn't ship Mandarin and Common Voice's Mandarin
config is gated, so ST-CMDS is the practical choice. **Caveat**: ST-CMDS may
overlap with parts of the model's CommonVoice fine-tune set, so cmn PER
numbers will look better than realistic — flagged in the manifest.

Run after `uv sync --extra fixtures`:

    uv run python scripts/fetch_fixtures.py --language es
    uv run python scripts/fetch_fixtures.py --language fr -n 10
    uv run python scripts/fetch_fixtures.py --language cmn -n 5

Idempotent: skips IDs already present in the manifest. Appends new entries to
manifest.json rather than overwriting, so adding a second language doesn't
clobber the first.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# `datasets` is in the [fixtures] extra; importing it lazily inside main() keeps
# `--help` usable without the extra installed.

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "fixtures"
MANIFEST_PATH = OUT_DIR / "manifest.json"
DEFAULT_N_CLIPS = 5
TARGET_SR = 16_000


@dataclass(frozen=True)
class LangSource:
    dataset: str
    config: str | None
    split: str
    transcript_field: str
    license_note: str
    prefix: str
    min_seconds: float
    max_seconds: float


# Per-lang dataset coordinates. MLS clips are pre-segmented audiobook
# chapters in the 10-15s range; ST-CMDS Mandarin clips are shorter
# (2-6s typical), so cmn uses a wider window.
_LANG_SOURCES: dict[str, LangSource] = {
    "es": LangSource(
        dataset="facebook/multilingual_librispeech",
        config="spanish",
        split="test",
        transcript_field="transcript",
        license_note="CC-BY 4.0 (LibriVox public-domain audio)",
        prefix="es",
        min_seconds=5.0,
        max_seconds=15.0,
    ),
    "fr": LangSource(
        dataset="facebook/multilingual_librispeech",
        config="french",
        split="test",
        transcript_field="transcript",
        license_note="CC-BY 4.0 (LibriVox public-domain audio)",
        prefix="fr",
        min_seconds=5.0,
        max_seconds=15.0,
    ),
    "cmn": LangSource(
        # ST-CMDS Mandarin via OpenSLR-38 (free, ungated). Note: may overlap
        # with the wav2vec2 CommonVoice fine-tune set — Mandarin PER numbers
        # are biased optimistic. Phase 6's prosody work won't rely on these
        # for tone scoring (different problem).
        dataset="urarik/free_st_chinese_mandarin_corpus",
        config=None,
        split="test",
        transcript_field="sentence",
        license_note="OpenSLR-38 ST-CMDS (CC-0 / public domain). May overlap with model fine-tune set.",
        prefix="cmn",
        min_seconds=3.0,
        max_seconds=10.0,
    ),
    "ja": LangSource(
        # JSUT basic5000 — single female speaker, studio-quality 48 kHz.
        # CC-BY-SA 4.0. Not in the wav2vec2 fine-tune mix, so PER numbers
        # are not artificially low.
        dataset="japanese-asr/ja_asr.jsut_basic5000",
        config=None,
        split="test",
        transcript_field="transcription",
        license_note="JSUT basic5000 (CC-BY-SA 4.0). Single speaker, studio-quality.",
        prefix="ja",
        min_seconds=3.0,
        max_seconds=10.0,
    ),
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


def _extract_audio(audio_field) -> tuple[np.ndarray, int]:
    """Pull samples + sample-rate from a `datasets` audio field, supporting
    both the legacy dict shape (`{"array": ..., "sampling_rate": ...}`) and
    the newer torchcodec AudioDecoder objects."""
    if isinstance(audio_field, dict):
        return np.asarray(audio_field["array"]), audio_field["sampling_rate"]
    # torchcodec AudioDecoder: lazy decode via get_all_samples().
    decoded = audio_field.get_all_samples()
    samples = decoded.data.squeeze().cpu().numpy()
    sr = audio_field.metadata.sample_rate
    return np.asarray(samples), int(sr)


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
        choices=sorted(_LANG_SOURCES),
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

    src = _LANG_SOURCES[args.language]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()
    existing_ids = {entry["id"] for entry in manifest}
    next_idx = 1 + max(
        (
            int(entry["id"].split("_")[1])
            for entry in manifest
            if entry["id"].startswith(f"{src.prefix}_")
        ),
        default=0,
    )
    target_count = args.num_clips
    added: list[dict] = []

    config_label = f", {src.config}" if src.config else ""
    print(
        f"Streaming {src.dataset}{config_label} ({src.split}) — "
        f"target: {target_count} new {args.language} clip(s)..."
    )
    from datasets import load_dataset

    load_kwargs = {"split": src.split, "streaming": True}
    if src.config is not None:
        ds = load_dataset(src.dataset, src.config, **load_kwargs)
    else:
        ds = load_dataset(src.dataset, **load_kwargs)

    for sample in ds:
        if len(added) >= target_count:
            break
        samples_arr, sr = _extract_audio(sample["audio"])
        duration = len(samples_arr) / sr
        if duration < src.min_seconds or duration > src.max_seconds:
            continue

        clip_id = f"{src.prefix}_{next_idx:03d}"
        if clip_id in existing_ids:
            next_idx += 1
            continue
        next_idx += 1

        flac_path = OUT_DIR / f"{clip_id}.flac"
        txt_path = OUT_DIR / f"{clip_id}.txt"

        resampled = _to_16k_mono(samples_arr, sr)
        sf.write(flac_path, resampled, TARGET_SR)
        transcript_text = str(sample[src.transcript_field]).strip()
        txt_path.write_text(transcript_text + "\n", encoding="utf-8")

        entry = {
            "id": clip_id,
            "audio": flac_path.name,
            "transcript": txt_path.name,
            "speaker_id": str(sample.get("speaker_id", "") or sample.get("client_id", "")),
            "duration_seconds": round(duration, 2),
            "language": args.language,
            "source_dataset": src.dataset,
            "config": src.config or "",
            "split": src.split,
            "license": src.license_note,
        }
        added.append(entry)
        print(f"  kept {clip_id}: {duration:.2f}s — {transcript_text[:70]!r}")

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
