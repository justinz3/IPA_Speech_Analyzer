"""Audio loading and resampling for the IPA pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16_000


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file as float32 mono samples.

    Returns (samples, sample_rate). Stereo input is downmixed by averaging.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32, copy=False)
    return data, sr


def ensure_16k(samples: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz if needed; no-op if already at the target rate."""
    if sr == TARGET_SR:
        return samples
    import torch
    import torchaudio.functional as F

    tensor = torch.from_numpy(samples).unsqueeze(0)
    out = F.resample(tensor, sr, TARGET_SR)
    return out.squeeze(0).numpy()
