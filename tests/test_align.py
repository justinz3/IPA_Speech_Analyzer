"""Tier A tests for the forced-alignment primitive.

No real model loaded — we use a synthetic vocab and handcrafted log_probs
so the right Viterbi path is unambiguous.
"""

from __future__ import annotations

import math

import pytest
import torch

from vocal_ipa.align import (
    SECONDS_PER_FRAME,
    AlignedPhoneme,
    forced_align,
    reference_to_token_ids,
)


class _FakeTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab


class _FakeProcessor:
    def __init__(self, vocab: dict[str, int]):
        self.tokenizer = _FakeTokenizer(vocab)


# Mini vocab modelling the wav2vec2 layout: 0 is blank, special tokens use <>.
_VOCAB = {
    "<pad>": 0,
    "<unk>": 1,
    "k": 2,
    "a": 3,
    "s": 4,
    "o": 5,
    "ʊ": 6,
    "oʊ": 7,  # multi-codepoint token must beat "o" + "ʊ" via max-munch
    "tʃ": 8,
    "t": 9,
    "ʃ": 10,
}


def _processor() -> _FakeProcessor:
    return _FakeProcessor(_VOCAB)


# -- reference_to_token_ids ---------------------------------------------------


def test_reference_to_token_ids_simple_word():
    ids, kept = reference_to_token_ids("kasa", _processor())
    assert kept == ["k", "a", "s", "a"]
    assert ids == [_VOCAB["k"], _VOCAB["a"], _VOCAB["s"], _VOCAB["a"]]


def test_reference_to_token_ids_strips_stress_and_length():
    ids, kept = reference_to_token_ids("ˈkaːsa", _processor())
    assert kept == ["k", "a", "s", "a"]
    assert ids == [_VOCAB["k"], _VOCAB["a"], _VOCAB["s"], _VOCAB["a"]]


def test_reference_to_token_ids_drops_word_boundary_spaces():
    ids, _ = reference_to_token_ids("ka sa", _processor())
    assert ids == [_VOCAB["k"], _VOCAB["a"], _VOCAB["s"], _VOCAB["a"]]


def test_reference_to_token_ids_greedy_max_munch():
    # "oʊ" is in the vocab; max-munch must prefer it over "o" + "ʊ".
    ids, kept = reference_to_token_ids("koʊ", _processor())
    assert kept == ["k", "oʊ"]
    assert ids == [_VOCAB["k"], _VOCAB["oʊ"]]


def test_reference_to_token_ids_greedy_match_does_not_overshoot():
    # "tʃ" is in the vocab but "ta" is not; we must fall through to "t" + "a".
    _, kept = reference_to_token_ids("ta", _processor())
    assert kept == ["t", "a"]


def test_reference_to_token_ids_warns_on_unknown_char():
    with pytest.warns(UserWarning, match="not in model vocab"):
        _, kept = reference_to_token_ids("ka@sa", _processor())
    assert kept == ["k", "a", "s", "a"]
    assert "@" not in kept


# -- forced_align -------------------------------------------------------------


def _make_log_probs(
    plan: list[int], vocab_size: int = 11, confidence: float = 50.0
) -> torch.Tensor:
    """Build a (T, V) log-prob tensor where each frame puts almost all mass on plan[t]."""
    T = len(plan)
    logits = torch.full((T, vocab_size), -confidence, dtype=torch.float32)
    for t, target in enumerate(plan):
        logits[t, target] = 0.0
    return torch.log_softmax(logits, dim=-1)


def test_forced_align_basic_two_phonemes():
    # Frames: a a ∅ ∅ s s
    log_probs = _make_log_probs([_VOCAB["a"], _VOCAB["a"], 0, 0, _VOCAB["s"], _VOCAB["s"]])
    spans = forced_align(log_probs, [_VOCAB["a"], _VOCAB["s"]], blank_id=0)
    assert len(spans) == 2
    a_span, s_span = spans
    assert (a_span.token_id, a_span.start_frame, a_span.end_frame) == (_VOCAB["a"], 0, 2)
    assert (s_span.token_id, s_span.start_frame, s_span.end_frame) == (_VOCAB["s"], 4, 6)
    # high-confidence frames → log-prob close to 0
    assert a_span.score > -1e-3
    assert s_span.score > -1e-3


def test_forced_align_repeated_phoneme_separated_by_blank():
    # "a∅a" — CTC requires the blank between repeated targets.
    log_probs = _make_log_probs([_VOCAB["a"], 0, _VOCAB["a"]])
    spans = forced_align(log_probs, [_VOCAB["a"], _VOCAB["a"]], blank_id=0)
    assert len(spans) == 2
    assert spans[0].start_frame == 0 and spans[0].end_frame == 1
    assert spans[1].start_frame == 2 and spans[1].end_frame == 3


def test_forced_align_seconds_helpers():
    log_probs = _make_log_probs([_VOCAB["k"]] * 5)
    spans = forced_align(log_probs, [_VOCAB["k"]], blank_id=0)
    assert spans[0].start_seconds == pytest.approx(0.0)
    assert spans[0].end_seconds == pytest.approx(5 * SECONDS_PER_FRAME)


def test_forced_align_rejects_empty_targets():
    log_probs = _make_log_probs([_VOCAB["k"]])
    with pytest.raises(ValueError, match="non-empty"):
        forced_align(log_probs, [], blank_id=0)


def test_forced_align_rejects_blank_in_targets():
    log_probs = _make_log_probs([_VOCAB["k"]])
    with pytest.raises(ValueError, match="blank"):
        forced_align(log_probs, [0, _VOCAB["k"]], blank_id=0)


def test_aligned_phoneme_dataclass_shape():
    # Documents the dataclass surface so callers know the score units (log-prob).
    p = AlignedPhoneme(token="a", token_id=3, start_frame=0, end_frame=2, score=-0.05)
    assert p.start_seconds == pytest.approx(0.0)
    assert p.end_seconds == pytest.approx(2 * SECONDS_PER_FRAME)
    assert math.isclose(p.score, -0.05)
