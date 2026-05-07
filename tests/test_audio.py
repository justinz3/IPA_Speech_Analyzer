"""Code-correctness tests for vocal_ipa.audio (no model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vocal_ipa.audio import TARGET_SR, ensure_16k, load_audio


def test_load_audio_returns_float32_mono(sine_wav_16k: Path) -> None:
    samples, sr = load_audio(sine_wav_16k)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert sr == TARGET_SR
    assert len(samples) == TARGET_SR  # 1 second


def test_load_audio_downmixes_stereo(stereo_sine_wav_16k: Path) -> None:
    samples, sr = load_audio(stereo_sine_wav_16k)
    assert samples.ndim == 1
    assert sr == TARGET_SR
    assert samples.dtype == np.float32


def test_ensure_16k_is_noop_at_target_rate() -> None:
    samples = np.zeros(TARGET_SR, dtype=np.float32)
    out = ensure_16k(samples, TARGET_SR)
    assert out is samples or np.array_equal(out, samples)


def test_ensure_16k_resamples_22k_to_16k(sine_wav_22k: Path) -> None:
    samples, sr = load_audio(sine_wav_22k)
    assert sr == 22_050
    out = ensure_16k(samples, sr)
    expected_len = round(len(samples) * TARGET_SR / sr)
    # torchaudio's resampler may differ by a few samples vs the naive ratio.
    assert abs(len(out) - expected_len) <= 4
    assert out.dtype == np.float32
