"""Per-phoneme scoring built on forced alignment.

Given audio + a known reference sentence, score each phoneme of the
reference: did the model produce the expected phoneme at that position?
The CLI (`pronounce <audio> --reference ...`) and Gradio UI both call
``score()`` and render the result.

Phase 6 extends the result with per-phoneme prosody scoring (pitch contour
for Mandarin tones, pitch accent observation for Japanese, RMS stress for
Spanish, utterance-end slope for French). Prosody runs from the raw audio
samples returned by _run_model(); it is a parallel path that does not affect
the segmental PER calculation.
"""

from __future__ import annotations

import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .align import forced_align, reference_to_token_ids
from .coaching import MissReference, lookup_miss
from .model import DEFAULT_MODEL, load, resolve_device
from .pipeline import Transcription, _run_model
from .prosody import (
    HOP_LENGTH,
    SR,
    ProsodicScore,
    _voiced_f0,
    extract_pitch,
    extract_rms,
    frame_slice,
    intonation_type,
    score_intonation,
    score_stress,
    score_tone,
    token_tone,
)
from .reference import resolve_locale, text_to_ipa

_BLANK_SURFACE = "∅"

# IPA vowel characters the model emits — used for stress and accent span detection.
_VOWELS = frozenset("aeiouɛʊɔæəɐɑɪ")


@dataclass
class ScoredPhoneme:
    expected: str  # phoneme from the reference (post-tokenization)
    produced: str  # most-frequent non-blank argmax over the span; "∅" if span is all blank
    start_s: float
    end_s: float
    score: float  # mean per-frame log-prob of the expected token over the span
    ok: bool  # produced == expected
    miss_reference: MissReference | None = (
        None  # populated for misses; None for ok or out-of-inventory
    )
    prosody: ProsodicScore | None = None  # populated for prosody-scored spans


@dataclass
class ScoreResult:
    phonemes: list[ScoredPhoneme]
    per: float  # 1 - (correct / total)
    reference_ipa: str  # raw phonemizer output (pre-tokenization), for debugging
    transcription: Transcription  # the underlying free-transcribe result
    language: str  # canonical lang of the score (mirrors transcription.language)
    dialect: str  # canonical dialect code ("es-es", "es-419", "fr-fr")
    dropped_reference_count: int = 0  # tokens dropped during reference tokenization
    miss_references: list[MissReference] = field(default_factory=list)
    # Deduped by (expected, produced) preserving first occurrence; one entry
    # per unique miss across the utterance. Empty if every phoneme is ok or
    # if no miss has an inventory entry.
    prosody_score: float | None = None
    # Fraction of prosodic spans where ok=True; None if language has no prosody
    # scoring or no scorable spans were found. For cmn/ja this is approximate
    # (span timings come from the degraded segmental model — see README).

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict already recurses into nested dataclasses; nothing else to do.
        return d


def score(
    audio_path: str | Path,
    reference_text: str,
    lang: str = "es",
    dialect: str | None = None,
    device: str = "auto",
    model_id: str = DEFAULT_MODEL,
) -> ScoreResult:
    """Score audio against a known reference sentence.

    Pipeline: orthography → IPA via phonemizer → tokenize for the model →
    run model on audio → forced-align reference against log-probs →
    per-span argmax-vs-expected + per-span prosody scoring.
    """
    if not reference_text.strip():
        raise ValueError("reference_text must be non-empty")

    locale = resolve_locale(lang, dialect)
    ref_ipa = text_to_ipa(reference_text, lang=locale.lang, dialect=locale.dialect)

    dev = resolve_device(device)
    processor, _ = load(model_id, dev)  # LRU-cached; _run_model reuses the same load

    target_ids, surface_kept, is_stressed = reference_to_token_ids(ref_ipa, processor)
    dropped = _count_input_chars(ref_ipa) - sum(len(t) for t in surface_kept)
    if not target_ids:
        raise ValueError(
            f"Reference text {reference_text!r} produced no alignable phonemes "
            f"(phonemizer output: {ref_ipa!r})"
        )

    transcription, log_probs, blank_id, samples = _run_model(
        audio_path, locale.lang, locale.dialect, device, model_id
    )
    spans = forced_align(log_probs, target_ids, blank_id)

    id_to_token = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    scored = [
        _score_span(surface, span, log_probs, blank_id, id_to_token)
        for surface, span in zip(surface_kept, spans, strict=True)
    ]
    correct = sum(1 for p in scored if p.ok)
    per = 1.0 - (correct / len(scored)) if scored else 0.0

    miss_references = _attach_miss_references(scored, locale.lang)
    try:
        prosody_score = _attach_prosody(scored, is_stressed, samples, locale.lang, reference_text)
    except Exception:
        warnings.warn(
            "Prosody scoring failed (segmental scoring unaffected):\n"
            + traceback.format_exc(),
            stacklevel=2,
        )
        prosody_score = None

    return ScoreResult(
        phonemes=scored,
        per=per,
        reference_ipa=ref_ipa,
        transcription=transcription,
        language=locale.lang,
        dialect=locale.dialect,
        dropped_reference_count=max(dropped, 0),
        miss_references=miss_references,
        prosody_score=prosody_score,
    )


# ---------------------------------------------------------------------------
# Prosody dispatch
# ---------------------------------------------------------------------------

def _attach_prosody(
    scored: list[ScoredPhoneme],
    is_stressed: list[bool],
    samples: np.ndarray,
    lang: str,
    reference_text: str,
) -> float | None:
    """Run prosody scoring in-place on scored phonemes; return aggregate score."""
    if not scored:
        return None
    if lang == "cmn":
        return _prosody_cmn(scored, samples)
    if lang == "ja":
        return _prosody_ja(scored, samples)
    if lang == "es":
        return _prosody_es(scored, is_stressed, samples)
    if lang == "fr":
        return _prosody_fr(scored, samples, reference_text)
    return None


def _prosody_cmn(scored: list[ScoredPhoneme], samples: np.ndarray) -> float | None:
    """Mandarin: score tone contour for each tone-bearing vowel span.

    Shaky-baseline note: span timings come from the degraded wav2vec2 segmental
    model (palatal/retroflex confusion, ü-vowel collapse). Results are approximate.
    """
    f0, voiced = extract_pitch(samples)
    results: list[bool] = []
    for sp in scored:
        digit = token_tone(sp.expected)
        if digit is None:
            continue
        ps = score_tone(f0, voiced, sp.start_s, sp.end_s, espeak_digit=digit)
        sp.prosody = ps
        if ps.ok is not None:
            results.append(ps.ok)
    return (sum(results) / len(results)) if results else None


def _prosody_ja(scored: list[ScoredPhoneme], samples: np.ndarray) -> float | None:
    """Japanese: record F0 per vowel span for display.

    Ground-truth pitch accent labels require parsing pyopenjtalk's full-context
    label output, which is complex and would need the original text passed down
    the call stack. For Phase 6 v1, we record f0_mean per vowel without a
    pass/fail verdict (ok=None). A future commit can add accent ground truth once
    the label parsing is implemented.

    Shaky-baseline note: span timings come from the degraded segmental model
    (English-like output). F0 observations are still meaningful but positional
    accuracy is low.
    """
    f0, voiced = extract_pitch(samples)
    voiced_f0 = f0[voiced]
    hop_s = HOP_LENGTH / SR
    for sp in scored:
        if not any(c in _VOWELS for c in sp.expected):
            continue
        f0_sl = frame_slice(f0, sp.start_s, sp.end_s, hop_s)
        v_sl = frame_slice(voiced, sp.start_s, sp.end_s, hop_s)
        vf = _voiced_f0(f0_sl, v_sl)
        f0_mean = float(np.mean(vf)) if len(vf) > 0 else None
        sp.prosody = ProsodicScore(
            ok=None,
            label="F0 observed (no accent ground truth yet)",
            f0_mean=f0_mean,
            rms_db=None,
        )
    _ = voiced_f0  # referenced to keep the extraction consistent; used for future ok= logic
    return None  # no scalar without accent ground truth


def _prosody_es(
    scored: list[ScoredPhoneme],
    is_stressed: list[bool],
    samples: np.ndarray,
) -> float | None:
    """Spanish: score lexical stress on vowel spans via RMS intensity."""
    rms = extract_rms(samples)
    utterance_rms_mean = float(np.mean(rms)) if len(rms) > 0 else 1e-9
    results: list[bool] = []
    for sp, stressed in zip(scored, is_stressed, strict=True):
        if not any(c in _VOWELS for c in sp.expected):
            continue
        ps = score_stress(rms, sp.start_s, sp.end_s, utterance_rms_mean, stressed)
        sp.prosody = ps
        if ps.ok is not None:
            results.append(ps.ok)
    return (sum(results) / len(results)) if results else None


def _prosody_fr(
    scored: list[ScoredPhoneme],
    samples: np.ndarray,
    reference_text: str,
) -> float | None:
    """French: score utterance-level intonation (rising vs falling tail)."""
    f0, voiced = extract_pitch(samples)
    expected = intonation_type(reference_text)
    ps = score_intonation(f0, voiced, expected)
    if scored:
        scored[-1].prosody = ps
    if ps.ok is not None:
        return 1.0 if ps.ok else 0.0
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _attach_miss_references(scored: list[ScoredPhoneme], lang: str) -> list[MissReference]:
    """Look up coaching info for each miss; mutate scored in place; return
    a deduped per-utterance list keyed on (expected, produced).
    """
    seen: dict[tuple[str, str], MissReference] = {}
    for p in scored:
        if p.ok:
            continue
        ref = lookup_miss(lang, p.expected, p.produced)
        if ref is None:
            continue
        p.miss_reference = ref
        key = (p.expected, p.produced)
        if key not in seen:
            seen[key] = ref
    return list(seen.values())


def _score_span(
    surface: str,
    span,
    log_probs: torch.Tensor,
    blank_id: int,
    id_to_token: dict[int, str],
) -> ScoredPhoneme:
    """Argmax-decode within the span, ignoring blank frames; compare to expected."""
    frames = log_probs[span.start_frame : span.end_frame]
    frame_argmax = frames.argmax(dim=-1)
    non_blank = frame_argmax[frame_argmax != blank_id]
    if non_blank.numel() == 0:
        produced_id = blank_id
    else:
        produced_id = int(torch.mode(non_blank).values.item())
    produced = _BLANK_SURFACE if produced_id == blank_id else id_to_token.get(produced_id, "?")
    return ScoredPhoneme(
        expected=surface,
        produced=produced,
        start_s=span.start_seconds,
        end_s=span.end_seconds,
        score=span.score,
        ok=(produced == surface),
    )


def _count_input_chars(ref_ipa: str) -> int:
    """Count non-whitespace, non-stress, non-length characters in the phonemizer output.

    Used only for the dropped-token tally; align.reference_to_token_ids does the
    real work and we just want to know how much it discarded.
    """
    skip = "ˈˌːˑ"
    return sum(1 for c in ref_ipa if not c.isspace() and c not in skip)
