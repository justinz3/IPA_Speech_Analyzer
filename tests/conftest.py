"""Shared pytest fixtures: synthesized audio for code-correctness tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _sine(duration_s: float, sr: int, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def sine_wav_16k(tmp_path: Path) -> Path:
    sr = 16_000
    samples = _sine(1.0, sr)
    path = tmp_path / "sine_16k.wav"
    sf.write(path, samples, sr)
    return path


@pytest.fixture
def sine_wav_22k(tmp_path: Path) -> Path:
    sr = 22_050
    samples = _sine(1.0, sr)
    path = tmp_path / "sine_22k.wav"
    sf.write(path, samples, sr)
    return path


@pytest.fixture
def stereo_sine_wav_16k(tmp_path: Path) -> Path:
    sr = 16_000
    left = _sine(1.0, sr, freq=440.0)
    right = _sine(1.0, sr, freq=660.0)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "stereo_sine_16k.wav"
    sf.write(path, stereo, sr)
    return path
