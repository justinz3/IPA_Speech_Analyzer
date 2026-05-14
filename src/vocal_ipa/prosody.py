"""Prosody feature extraction and per-language scoring.

Operates on raw 16 kHz float32 audio samples; entirely independent of the
wav2vec2 segmental model. Pitch is extracted via librosa.pyin (probabilistic
YIN with voiced/unvoiced detection). Intensity is numpy RMS.

Hop size is 160 samples → 10 ms per frame at 16 kHz. The wav2vec2 forced-
alignment runs at 20 ms (50 Hz); prosody frames are therefore 2× finer,
giving sub-phoneme pitch resolution without needing to up-sample alignment.

Shaky-baseline caveat (cmn / ja): the AlignedPhoneme spans fed to these
scorers come from forced alignment over the wav2vec2 segmental model, which
is degraded for Mandarin and near-unusable for Japanese (broken espeak-ja
training labels). Prosody results for those languages are approximate until a
language-specific model checkpoint replaces the current multilingual one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

SR = 16_000
HOP_LENGTH = 160          # 10 ms @ 16 kHz
FRAME_RATE_HZ = SR / HOP_LENGTH  # 100 Hz

# Minimum number of voiced frames needed to attempt a score.
_MIN_VOICED = 3

# Tone-contour thresholds (normalised slope = Hz/s / median_f0).
_SLOPE_THRESH = 0.8       # |normalised slope| above this → "moving"
_DIP_THRESH = 0.35        # midpoint must be this far below endpoints for tone-3 dip

# Stress: RMS boost (dB) required for a vowel to be called stressed.
_STRESS_DB_THRESH = 2.0

# Intonation: tail is the last 20 % of voiced frames; slope threshold (normalised).
_INTONATION_SLOPE_THRESH = 0.5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProsodicScore:
    """Result of a single prosody scoring operation on one span (or utterance)."""

    ok: bool | None     # True/False = pass/fail; None = not enough voiced frames
    label: str          # e.g. "tone 2 (rising) ✓", "stressed ✗", "declarative ✓"
    f0_mean: float | None   # mean F0 in Hz over voiced frames; None if unvoiced
    rms_db: float | None    # RMS of span relative to utterance mean (dB); None if not computed


# ---------------------------------------------------------------------------
# Core extractors
# ---------------------------------------------------------------------------

def extract_pitch(
    samples: np.ndarray,
    sr: int = SR,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0, voiced) arrays, one entry per hop frame.

    f0 is in Hz (0.0 where unvoiced). voiced is a boolean mask.
    Uses librosa.pyin with a speech-appropriate F0 range.
    """
    import librosa  # lazy import — librosa is a large dep

    f0, voiced_flag, _ = librosa.pyin(
        samples.astype(np.float32),
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz — below any speech F0
        fmax=librosa.note_to_hz("C7"),   # ~2093 Hz — well above falsetto
        sr=sr,
        hop_length=HOP_LENGTH,
        fill_na=0.0,
    )
    return f0.astype(np.float32), voiced_flag.astype(bool)


def extract_rms(
    samples: np.ndarray,
    frame_length: int = 512,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Return per-hop RMS energy array (linear scale, float32)."""
    import librosa

    rms = librosa.feature.rms(
        y=samples.astype(np.float32),
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return rms[0].astype(np.float32)  # shape (n_frames,)


def frame_slice(
    feature: np.ndarray,
    start_s: float,
    end_s: float,
    hop_s: float = HOP_LENGTH / SR,
) -> np.ndarray:
    """Slice a per-hop feature array to the half-open interval [start_s, end_s)."""
    i = max(0, int(start_s / hop_s))
    j = max(i, int(end_s / hop_s))
    return feature[i:j]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _voiced_f0(f0_slice: np.ndarray, voiced_slice: np.ndarray) -> np.ndarray:
    """Return only voiced F0 samples from the slice pair."""
    if len(f0_slice) == 0 or len(voiced_slice) == 0:
        return np.empty(0, dtype=np.float32)
    n = min(len(f0_slice), len(voiced_slice))
    return f0_slice[:n][voiced_slice[:n]]


def _normalised_slope(f0_voiced: np.ndarray) -> float:
    """Linear regression slope divided by median F0 — dimensionless."""
    if len(f0_voiced) < 2:
        return 0.0
    x = np.arange(len(f0_voiced), dtype=np.float32)
    slope = float(np.polyfit(x, f0_voiced, 1)[0])
    median = float(np.median(f0_voiced))
    if median < 1.0:
        return 0.0
    # Convert to Hz/frame then to Hz/s, then normalise.
    return (slope * FRAME_RATE_HZ) / median


def _rms_to_db(rms: float, ref: float) -> float:
    """Convert RMS ratio to dB (20·log10(rms/ref)). Returns 0 if ref ≈ 0."""
    if ref < 1e-9:
        return 0.0
    return 20.0 * float(np.log10(max(rms, 1e-9) / ref))


# ---------------------------------------------------------------------------
# Tone digit extraction (Mandarin)
# ---------------------------------------------------------------------------

_TONE_DIGIT_RE = re.compile(r"([1-5])$")


def token_tone(token: str) -> int | None:
    """Extract trailing tone digit from a Mandarin IPA token.

    Returns 1–5 if present, None otherwise (non-tonal token or consonant).
    The digit reflects espeak's encoding: espeak tone 1 = pinyin tone 2,
    espeak tone 2 = pinyin tone 3, espeak tone 5 = pinyin tone 1 or 4.
    """
    m = _TONE_DIGIT_RE.search(token)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Per-language scorers
# ---------------------------------------------------------------------------

def score_tone(
    f0: np.ndarray,
    voiced: np.ndarray,
    start_s: float,
    end_s: float,
    espeak_digit: int,
) -> ProsodicScore:
    """Score Mandarin tone contour for a vowel span.

    espeak_digit is the digit suffix on the IPA token (1–5), using espeak's
    encoding convention (not standard pinyin tone numbers):
      digit 1 → pinyin tone 2 (rising)
      digit 2 → pinyin tone 3 (falling-rising dip)
      digit 5 → pinyin tone 1 or 4 (flat-high or falling — both accepted)
    Tone 5 (neutral) tokens are not scored (returns ok=None).
    """
    hop_s = HOP_LENGTH / SR
    f0_sl = frame_slice(f0, start_s, end_s, hop_s)
    v_sl = frame_slice(voiced, start_s, end_s, hop_s)
    vf = _voiced_f0(f0_sl, v_sl)
    f0_mean = float(np.mean(vf)) if len(vf) > 0 else None

    if espeak_digit == 5:
        # Neutral tone — don't penalise.
        label = "neutral (not scored)"
        return ProsodicScore(ok=None, label=label, f0_mean=f0_mean, rms_db=None)

    if len(vf) < _MIN_VOICED:
        tone_names = {1: "rising", 2: "dip", 3: "flat/high", 4: "flat/falling"}
        name = tone_names.get(espeak_digit, f"tone {espeak_digit}")
        return ProsodicScore(ok=None, label=f"tone {espeak_digit} ({name}) — unvoiced", f0_mean=None, rms_db=None)

    slope = _normalised_slope(vf)

    if espeak_digit == 1:
        # Pinyin tone 2: rising.
        ok = slope > _SLOPE_THRESH
        direction = "↑" if ok else f"↑ expected, got slope {slope:+.2f}"
        label = f"tone 2/rising {'✓' if ok else '✗'} ({direction})"

    elif espeak_digit == 2:
        # Pinyin tone 3: falling-rising dip. Midpoint should be below endpoints.
        if len(vf) >= 3:
            mid = float(np.mean(vf[len(vf) // 3: 2 * len(vf) // 3]))
            endpoints = (float(vf[0]) + float(vf[-1])) / 2.0
            dip = (endpoints - mid) / max(endpoints, 1.0)
            ok = dip > _DIP_THRESH
        else:
            ok = False
            dip = 0.0
        label = f"tone 3/dip {'✓' if ok else '✗'}"

    else:
        # espeak_digit 3 or 4 → flat/high or falling — accept either.
        # (espeak emits digit 3 for some unusual combinations; treat like 4.)
        falling = slope < -_SLOPE_THRESH
        flat = abs(slope) <= _SLOPE_THRESH
        ok = falling or flat
        direction = "↓" if falling else ("→" if flat else f"↑ unexpected, slope {slope:+.2f}")
        label = f"tone 1/4 (flat/fall) {'✓' if ok else '✗'} ({direction})"

    return ProsodicScore(ok=ok, label=label, f0_mean=f0_mean, rms_db=None)


def score_accent(
    f0: np.ndarray,
    voiced: np.ndarray,
    start_s: float,
    end_s: float,
    utterance_f0_median: float,
    expected_high: bool,
) -> ProsodicScore:
    """Score Japanese pitch accent mora (H=high, L=low).

    A mora is considered high-pitched if its mean F0 exceeds the utterance
    median. Low is the complement.
    """
    hop_s = HOP_LENGTH / SR
    f0_sl = frame_slice(f0, start_s, end_s, hop_s)
    v_sl = frame_slice(voiced, start_s, end_s, hop_s)
    vf = _voiced_f0(f0_sl, v_sl)

    if len(vf) < _MIN_VOICED or utterance_f0_median < 1.0:
        return ProsodicScore(
            ok=None,
            label=f"{'H' if expected_high else 'L'} — unvoiced",
            f0_mean=None,
            rms_db=None,
        )

    f0_mean = float(np.mean(vf))
    detected_high = f0_mean > utterance_f0_median
    ok = detected_high == expected_high
    expected_str = "H" if expected_high else "L"
    detected_str = "H" if detected_high else "L"
    label = f"accent {expected_str} {'✓' if ok else f'✗ (got {detected_str})'}"
    return ProsodicScore(ok=ok, label=label, f0_mean=f0_mean, rms_db=None)


def score_stress(
    rms: np.ndarray,
    start_s: float,
    end_s: float,
    utterance_rms_mean: float,
    expected_stressed: bool,
) -> ProsodicScore:
    """Score Spanish lexical stress on a vowel span.

    A vowel is considered stressed if its mean RMS is at least _STRESS_DB_THRESH
    dB above the utterance mean. Unstressed should be at or below the mean.
    """
    hop_s = HOP_LENGTH / SR
    rms_sl = frame_slice(rms, start_s, end_s, hop_s)

    if len(rms_sl) == 0:
        return ProsodicScore(
            ok=None,
            label=f"{'stressed' if expected_stressed else 'unstressed'} — no frames",
            f0_mean=None,
            rms_db=None,
        )

    span_rms = float(np.mean(rms_sl))
    db = _rms_to_db(span_rms, utterance_rms_mean)

    detected_stressed = db >= _STRESS_DB_THRESH
    ok = detected_stressed == expected_stressed
    exp_str = "stressed" if expected_stressed else "unstressed"
    label = f"{exp_str} {'✓' if ok else '✗'} ({db:+.1f} dB vs mean)"
    return ProsodicScore(ok=ok, label=label, f0_mean=None, rms_db=db)


def score_intonation(
    f0: np.ndarray,
    voiced: np.ndarray,
    expected: str,
) -> ProsodicScore:
    """Score French utterance-level intonation contour.

    expected: 'interrogative' (expect rising tail) or 'declarative' (expect falling/flat).
    Analyses the last 20 % of voiced frames in the utterance.
    """
    vf_indices = np.where(voiced)[0]
    if len(vf_indices) < _MIN_VOICED * 5:
        return ProsodicScore(ok=None, label=f"{expected} — insufficient voiced frames", f0_mean=None, rms_db=None)

    tail_start = vf_indices[int(len(vf_indices) * 0.80)]
    tail_f0 = f0[tail_start:][voiced[tail_start:]]

    if len(tail_f0) < _MIN_VOICED:
        return ProsodicScore(ok=None, label=f"{expected} — unvoiced tail", f0_mean=None, rms_db=None)

    f0_mean = float(np.mean(tail_f0))
    slope = _normalised_slope(tail_f0)

    if expected == "interrogative":
        ok = slope > _INTONATION_SLOPE_THRESH
        direction = "↑" if slope > 0 else "↓"
        label = f"interrogative (rising) {'✓' if ok else f'✗ (tail {direction}, slope {slope:+.2f})'}"
    else:
        ok = slope < _INTONATION_SLOPE_THRESH
        direction = "↓" if slope < 0 else "→"
        label = f"declarative (fall/flat) {'✓' if ok else f'✗ (tail ↑, slope {slope:+.2f})'}"

    return ProsodicScore(ok=ok, label=label, f0_mean=f0_mean, rms_db=None)


def intonation_type(reference_text: str) -> str:
    """Infer French utterance contour type from punctuation.

    Returns 'interrogative' if the text ends with '?', else 'declarative'.
    """
    return "interrogative" if reference_text.rstrip().endswith("?") else "declarative"
