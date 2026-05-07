"""Forced-alignment primitive for Phase 2 scoring.

Given the model's CTC log-probs over T frames and a sequence of reference
phoneme token-ids, return per-phoneme time spans via Viterbi alignment.
Wraps torchaudio.functional.forced_align + merge_tokens; the rest is
reference-text tokenization that bypasses the wav2vec2 tokenizer's
built-in (English-by-default) phonemizer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torchaudio.functional as TAF

# wav2vec2 base: 16 kHz audio, conv stack downsamples by 320 → 50 Hz frames.
# Hardcoded because the project uses one model. Revisit if a different
# feature extractor enters the pipeline (likely Phase 5+).
FRAME_RATE_HZ = 50.0
SECONDS_PER_FRAME = 1.0 / FRAME_RATE_HZ

_STRIP_MARKS = str.maketrans("", "", "ˈˌːˑ")
_SPECIAL_TOKEN_PREFIX = "<"


@dataclass
class AlignedPhoneme:
    token: str  # surface IPA phoneme as it appears in the vocab
    token_id: int  # vocab id
    start_frame: int  # inclusive
    end_frame: int  # exclusive
    score: float  # mean per-frame log-prob over the span

    @property
    def start_seconds(self) -> float:
        return self.start_frame * SECONDS_PER_FRAME

    @property
    def end_seconds(self) -> float:
        return self.end_frame * SECONDS_PER_FRAME


def reference_to_token_ids(
    ref_ipa: str,
    processor,
) -> tuple[list[int], list[str]]:
    """Tokenize phonemizer-style IPA into model-vocab IDs.

    Strips stress (ˈ ˌ) and length (ː ˑ) marks, then walks the string with
    greedy max-munch against the model vocab so multi-codepoint phonemes
    like ``tʃ``, ``oʊ`` match before their single-char prefixes.

    Tokens not in the vocab are dropped with a one-shot warning. Returns
    ``(token_ids, surface_phonemes_kept)`` so callers can reconstruct the
    expected-phoneme sequence.
    """
    cleaned = ref_ipa.translate(_STRIP_MARKS)
    vocab: dict[str, int] = processor.tokenizer.get_vocab()
    candidates = sorted(
        (t for t in vocab if t and not t.startswith(_SPECIAL_TOKEN_PREFIX)),
        key=len,
        reverse=True,
    )

    ids: list[int] = []
    kept: list[str] = []
    dropped: list[str] = []
    i = 0
    n = len(cleaned)
    while i < n:
        ch = cleaned[i]
        if ch.isspace():
            i += 1
            continue
        match = None
        for tok in candidates:
            if cleaned.startswith(tok, i):
                match = tok
                break
        if match is None:
            dropped.append(ch)
            i += 1
            continue
        ids.append(vocab[match])
        kept.append(match)
        i += len(match)

    if dropped:
        sample = "".join(dropped[:10])
        warnings.warn(
            f"Dropped {len(dropped)} reference character(s) not in model vocab: {sample!r}",
            stacklevel=2,
        )
    return ids, kept


def forced_align(
    log_probs: torch.Tensor,
    target_ids: list[int],
    blank_id: int,
) -> list[AlignedPhoneme]:
    """Viterbi-align ``target_ids`` against ``log_probs`` and return per-token spans.

    ``log_probs`` is shape ``(T, V)`` — already log-softmaxed and on CPU.
    Targets must be non-empty and contain no blank ids (CTC inserts blanks
    internally). Returns one ``AlignedPhoneme`` per target id, in order.
    """
    if not target_ids:
        raise ValueError("target_ids must be non-empty")
    if blank_id in target_ids:
        raise ValueError("target_ids must not contain the blank id")

    log_probs_b = log_probs.unsqueeze(0)
    targets_b = torch.tensor([target_ids], dtype=torch.int32)
    paths, scores = TAF.forced_align(log_probs_b, targets_b, blank=blank_id)
    spans = TAF.merge_tokens(paths[0], scores[0], blank=blank_id)

    if len(spans) != len(target_ids):
        # forced_align should always produce one span per target token; if
        # this ever fires it means the alignment failed silently and we'd
        # rather raise than silently mis-pair phonemes with spans.
        raise RuntimeError(
            f"forced_align returned {len(spans)} spans for {len(target_ids)} targets"
        )

    return [
        AlignedPhoneme(
            token="",  # filled in by caller from surface_phonemes_kept
            token_id=int(span.token),
            start_frame=int(span.start),
            end_frame=int(span.end),
            score=float(span.score),
        )
        for span in spans
    ]
