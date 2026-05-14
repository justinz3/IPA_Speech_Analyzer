"""Tier A tests for prosody.py — no model, no real audio.

All tests use synthetic pitch/RMS arrays or simple sine waves so they
run in < 1 s with no network access.
"""

from __future__ import annotations

import numpy as np

from vocal_ipa.prosody import (
    HOP_LENGTH,
    SR,
    ProsodicScore,
    extract_pitch,
    extract_rms,
    frame_slice,
    intonation_type,
    score_accent,
    score_intonation,
    score_stress,
    score_tone,
    token_tone,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic audio
# ---------------------------------------------------------------------------

def _sine(freq: float = 200.0, duration_s: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _flat_f0(value: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (f0, voiced) arrays: all frames voiced at `value` Hz."""
    f0 = np.full(n, value, dtype=np.float32)
    voiced = np.ones(n, dtype=bool)
    return f0, voiced


def _rising_f0(start: float, end: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    f0 = np.linspace(start, end, n, dtype=np.float32)
    voiced = np.ones(n, dtype=bool)
    return f0, voiced


def _dip_f0(high: float, low: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """F0 that falls then rises — canonical tone-3 contour."""
    mid = n // 2
    falling = np.linspace(high, low, mid, dtype=np.float32)
    rising = np.linspace(low, high, n - mid, dtype=np.float32)
    f0 = np.concatenate([falling, rising])
    voiced = np.ones(n, dtype=bool)
    return f0, voiced


def _falling_f0(start: float, end: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    f0 = np.linspace(start, end, n, dtype=np.float32)
    voiced = np.ones(n, dtype=bool)
    return f0, voiced


def _rms_array(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# extract_pitch / extract_rms / frame_slice
# ---------------------------------------------------------------------------

def test_extract_pitch_returns_correct_shape() -> None:
    samples = _sine(200.0, 0.5)
    f0, voiced = extract_pitch(samples)
    expected_frames = 1 + len(samples) // HOP_LENGTH
    assert abs(len(f0) - expected_frames) <= 2
    assert f0.shape == voiced.shape


def test_extract_pitch_sine_is_mostly_voiced() -> None:
    samples = _sine(200.0, 0.5)
    _, voiced = extract_pitch(samples)
    assert voiced.mean() > 0.5


def test_extract_rms_returns_positive_values() -> None:
    samples = _sine(440.0, 0.3)
    rms = extract_rms(samples)
    assert rms.ndim == 1
    assert (rms >= 0).all()
    assert rms.mean() > 0.0


def test_extract_rms_silence_is_near_zero() -> None:
    samples = np.zeros(SR, dtype=np.float32)
    rms = extract_rms(samples)
    assert rms.max() < 1e-6


def test_frame_slice_basic() -> None:
    arr = np.arange(100, dtype=np.float32)
    hop_s = HOP_LENGTH / SR  # 0.01 s
    sl = frame_slice(arr, 0.0, 0.10, hop_s)
    assert len(sl) == 10


def test_frame_slice_clamps_to_array_bounds() -> None:
    arr = np.arange(5, dtype=np.float32)
    hop_s = HOP_LENGTH / SR
    sl = frame_slice(arr, 0.0, 1.0, hop_s)  # 1 s → 100 frames, but array only has 5
    assert len(sl) == 5


def test_frame_slice_empty_when_start_equals_end() -> None:
    arr = np.arange(20, dtype=np.float32)
    hop_s = HOP_LENGTH / SR
    sl = frame_slice(arr, 0.10, 0.10, hop_s)
    assert len(sl) == 0


# ---------------------------------------------------------------------------
# token_tone
# ---------------------------------------------------------------------------

def test_token_tone_extracts_digit() -> None:
    assert token_tone("ɑ1") == 1
    assert token_tone("iɛ2") == 2
    assert token_tone("onɡ5") == 5


def test_token_tone_none_for_no_digit() -> None:
    assert token_tone("n") is None
    assert token_tone("ts.") is None
    assert token_tone("k") is None


def test_token_tone_none_for_empty() -> None:
    assert token_tone("") is None


# ---------------------------------------------------------------------------
# score_tone (Mandarin)
# ---------------------------------------------------------------------------

def _tone_score(espeak_digit: int, f0: np.ndarray, voiced: np.ndarray) -> ProsodicScore:
    return score_tone(f0, voiced, start_s=0.0, end_s=len(f0) * HOP_LENGTH / SR, espeak_digit=espeak_digit)


def test_score_tone_neutral_not_scored() -> None:
    f0, voiced = _flat_f0(150.0, 20)
    result = _tone_score(5, f0, voiced)
    assert result.ok is None
    assert "neutral" in result.label


def test_score_tone_rising_passes_on_rising_contour() -> None:
    f0, voiced = _rising_f0(100.0, 250.0, 30)
    result = _tone_score(1, f0, voiced)  # digit 1 = pinyin tone 2 (rising)
    assert result.ok is True
    assert "✓" in result.label


def test_score_tone_rising_fails_on_flat_contour() -> None:
    f0, voiced = _flat_f0(150.0, 30)
    result = _tone_score(1, f0, voiced)
    assert result.ok is False
    assert "✗" in result.label


def test_score_tone_dip_passes_on_dip_contour() -> None:
    f0, voiced = _dip_f0(180.0, 100.0, 30)
    result = _tone_score(2, f0, voiced)  # digit 2 = pinyin tone 3 (dip)
    assert result.ok is True


def test_score_tone_dip_fails_on_flat() -> None:
    f0, voiced = _flat_f0(150.0, 30)
    result = _tone_score(2, f0, voiced)
    assert result.ok is False


def test_score_tone_flat_fall_passes_on_falling() -> None:
    f0, voiced = _falling_f0(220.0, 80.0, 30)
    result = _tone_score(4, f0, voiced)  # digit 4 → flat or falling
    assert result.ok is True


def test_score_tone_flat_fall_passes_on_flat() -> None:
    f0, voiced = _flat_f0(150.0, 30)
    result = _tone_score(4, f0, voiced)
    assert result.ok is True


def test_score_tone_insufficient_voiced_returns_none() -> None:
    f0 = np.array([150.0, 0.0], dtype=np.float32)
    voiced = np.array([True, False])
    result = score_tone(f0, voiced, start_s=0.0, end_s=len(f0) * HOP_LENGTH / SR, espeak_digit=1)
    assert result.ok is None


def test_score_tone_f0_mean_populated() -> None:
    f0, voiced = _flat_f0(180.0, 20)
    result = _tone_score(4, f0, voiced)
    assert result.f0_mean is not None
    assert abs(result.f0_mean - 180.0) < 1.0


# ---------------------------------------------------------------------------
# score_accent (Japanese)
# ---------------------------------------------------------------------------

def _accent_score(expected_high: bool, f0: np.ndarray, voiced: np.ndarray, median: float) -> ProsodicScore:
    return score_accent(f0, voiced, start_s=0.0, end_s=len(f0) * HOP_LENGTH / SR,
                        utterance_f0_median=median, expected_high=expected_high)


def test_score_accent_high_passes_above_median() -> None:
    f0, voiced = _flat_f0(200.0, 20)
    result = _accent_score(expected_high=True, f0=f0, voiced=voiced, median=150.0)
    assert result.ok is True
    assert "H" in result.label and "✓" in result.label


def test_score_accent_high_fails_below_median() -> None:
    f0, voiced = _flat_f0(100.0, 20)
    result = _accent_score(expected_high=True, f0=f0, voiced=voiced, median=150.0)
    assert result.ok is False


def test_score_accent_low_passes_below_median() -> None:
    f0, voiced = _flat_f0(100.0, 20)
    result = _accent_score(expected_high=False, f0=f0, voiced=voiced, median=150.0)
    assert result.ok is True


def test_score_accent_low_fails_above_median() -> None:
    f0, voiced = _flat_f0(200.0, 20)
    result = _accent_score(expected_high=False, f0=f0, voiced=voiced, median=150.0)
    assert result.ok is False


def test_score_accent_unvoiced_returns_none() -> None:
    f0 = np.zeros(5, dtype=np.float32)
    voiced = np.zeros(5, dtype=bool)
    result = score_accent(f0, voiced, 0.0, 0.05, utterance_f0_median=150.0, expected_high=True)
    assert result.ok is None


# ---------------------------------------------------------------------------
# score_stress (Spanish)
# ---------------------------------------------------------------------------

def _stress_score(rms_values: list[float], utt_mean: float, expected_stressed: bool) -> ProsodicScore:
    rms = _rms_array(rms_values)
    return score_stress(rms, start_s=0.0, end_s=len(rms) * HOP_LENGTH / SR,
                        utterance_rms_mean=utt_mean, expected_stressed=expected_stressed)


def test_score_stress_stressed_passes_when_high_rms() -> None:
    # span RMS = 0.2, utterance mean = 0.1 → ~6 dB above mean
    result = _stress_score([0.2] * 10, utt_mean=0.1, expected_stressed=True)
    assert result.ok is True
    assert "stressed" in result.label and "✓" in result.label


def test_score_stress_stressed_fails_when_low_rms() -> None:
    # span RMS ≈ utterance mean → not stressed
    result = _stress_score([0.1] * 10, utt_mean=0.1, expected_stressed=True)
    assert result.ok is False


def test_score_stress_unstressed_passes_when_low_rms() -> None:
    result = _stress_score([0.1] * 10, utt_mean=0.1, expected_stressed=False)
    assert result.ok is True


def test_score_stress_unstressed_fails_when_high_rms() -> None:
    result = _stress_score([0.2] * 10, utt_mean=0.1, expected_stressed=False)
    assert result.ok is False


def test_score_stress_rms_db_populated() -> None:
    result = _stress_score([0.2] * 10, utt_mean=0.1, expected_stressed=True)
    assert result.rms_db is not None
    assert result.rms_db > 0.0


def test_score_stress_no_frames_returns_none() -> None:
    rms = np.array([], dtype=np.float32)
    result = score_stress(rms, start_s=0.5, end_s=0.5, utterance_rms_mean=0.1, expected_stressed=True)
    assert result.ok is None


# ---------------------------------------------------------------------------
# score_intonation (French)
# ---------------------------------------------------------------------------

def _make_full_f0(tail_slope: str, n_total: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Build a full-utterance f0/voiced pair with a rising or falling tail."""
    flat_val = 150.0
    flat_part = np.full(int(n_total * 0.8), flat_val, dtype=np.float32)
    tail_n = n_total - len(flat_part)
    if tail_slope == "rising":
        tail = np.linspace(flat_val, flat_val * 2.0, tail_n, dtype=np.float32)
    else:
        tail = np.linspace(flat_val, flat_val * 0.5, tail_n, dtype=np.float32)
    f0 = np.concatenate([flat_part, tail])
    voiced = np.ones(n_total, dtype=bool)
    return f0, voiced


def test_score_intonation_interrogative_passes_rising_tail() -> None:
    f0, voiced = _make_full_f0("rising")
    result = score_intonation(f0, voiced, expected="interrogative")
    assert result.ok is True
    assert "✓" in result.label


def test_score_intonation_interrogative_fails_falling_tail() -> None:
    f0, voiced = _make_full_f0("falling")
    result = score_intonation(f0, voiced, expected="interrogative")
    assert result.ok is False


def test_score_intonation_declarative_passes_falling_tail() -> None:
    f0, voiced = _make_full_f0("falling")
    result = score_intonation(f0, voiced, expected="declarative")
    assert result.ok is True


def test_score_intonation_declarative_fails_rising_tail() -> None:
    f0, voiced = _make_full_f0("rising")
    result = score_intonation(f0, voiced, expected="declarative")
    assert result.ok is False


def test_score_intonation_insufficient_voiced_returns_none() -> None:
    f0 = np.full(10, 150.0, dtype=np.float32)
    voiced = np.ones(10, dtype=bool)
    result = score_intonation(f0, voiced, expected="declarative")
    assert result.ok is None  # < MIN_VOICED * 5 = 15 frames


def test_score_intonation_f0_mean_populated() -> None:
    f0, voiced = _make_full_f0("falling")
    result = score_intonation(f0, voiced, expected="declarative")
    assert result.f0_mean is not None


# ---------------------------------------------------------------------------
# intonation_type helper
# ---------------------------------------------------------------------------

def test_intonation_type_question_mark() -> None:
    assert intonation_type("Comment allez-vous?") == "interrogative"
    assert intonation_type("Comment allez-vous? ") == "interrogative"


def test_intonation_type_declarative() -> None:
    assert intonation_type("Je parle français.") == "declarative"
    assert intonation_type("Bonjour") == "declarative"
