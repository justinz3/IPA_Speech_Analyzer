"""Per-phoneme scoring built on forced alignment.

Given audio + a known reference sentence, score each phoneme of the
reference: did the model produce the expected phoneme at that position?
The CLI (`pronounce <audio> --reference ...`) and Gradio UI both call
``score()`` and render the result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .align import forced_align, reference_to_token_ids
from .model import DEFAULT_MODEL, load, resolve_device
from .pipeline import Transcription, _run_model
from .reference import resolve_locale, text_to_ipa

_BLANK_SURFACE = "∅"


@dataclass
class ScoredPhoneme:
    expected: str  # phoneme from the reference (post-tokenization)
    produced: str  # most-frequent non-blank argmax over the span; "∅" if span is all blank
    start_s: float
    end_s: float
    score: float  # mean per-frame log-prob of the expected token over the span
    ok: bool  # produced == expected


@dataclass
class ScoreResult:
    phonemes: list[ScoredPhoneme]
    per: float  # 1 - (correct / total)
    reference_ipa: str  # raw phonemizer output (pre-tokenization), for debugging
    transcription: Transcription  # the underlying free-transcribe result
    language: str  # canonical lang of the score (mirrors transcription.language)
    dialect: str  # canonical dialect ("castilian", "latam", ...)
    dropped_reference_count: int = 0  # tokens dropped during reference tokenization

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
    per-span argmax-vs-expected.
    """
    if not reference_text.strip():
        raise ValueError("reference_text must be non-empty")

    locale = resolve_locale(lang, dialect)
    ref_ipa = text_to_ipa(reference_text, lang=locale.lang, dialect=locale.dialect)

    dev = resolve_device(device)
    processor, _ = load(model_id, dev)  # LRU-cached; _run_model reuses the same load

    target_ids, surface_kept = reference_to_token_ids(ref_ipa, processor)
    dropped = _count_input_chars(ref_ipa) - sum(len(t) for t in surface_kept)
    if not target_ids:
        raise ValueError(
            f"Reference text {reference_text!r} produced no alignable phonemes "
            f"(phonemizer output: {ref_ipa!r})"
        )

    transcription, log_probs, blank_id = _run_model(
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

    return ScoreResult(
        phonemes=scored,
        per=per,
        reference_ipa=ref_ipa,
        transcription=transcription,
        language=locale.lang,
        dialect=locale.dialect,
        dropped_reference_count=max(dropped, 0),
    )


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
