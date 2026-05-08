"""Tier A tests for per-phoneme scoring.

Mocks _run_model + text_to_ipa so the pipeline runs end-to-end against
canned data; the real model is never loaded.
"""

from __future__ import annotations

import pytest
import torch

from vocal_ipa import score as score_module
from vocal_ipa.align import AlignedPhoneme
from vocal_ipa.pipeline import Transcription
from vocal_ipa.score import ScoredPhoneme, ScoreResult, _score_span, score

# -- argument validation ------------------------------------------------------


def test_score_rejects_unsupported_lang(tmp_path):
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")
    with pytest.raises(ValueError, match="Unsupported language"):
        score(audio, "irrelevant", lang="ja")


def test_score_rejects_empty_reference(tmp_path):
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")
    with pytest.raises(ValueError, match="non-empty"):
        score(audio, "   ", lang="es")


# -- _score_span: argmax-over-span semantics ---------------------------------


def _flat_log_probs(plan: list[int], vocab_size: int = 5) -> torch.Tensor:
    logits = torch.full((len(plan), vocab_size), -50.0, dtype=torch.float32)
    for t, target in enumerate(plan):
        logits[t, target] = 0.0
    return torch.log_softmax(logits, dim=-1)


def test_score_span_all_blank_produces_blank_marker():
    log_probs = _flat_log_probs([0, 0, 0])  # blank everywhere
    span = AlignedPhoneme(token="a", token_id=2, start_frame=0, end_frame=3, score=-0.0)
    out = _score_span("a", span, log_probs, blank_id=0, id_to_token={2: "a"})
    assert out.produced == "∅"
    assert out.ok is False


def test_score_span_argmax_ignores_blank_frames():
    # 4 frames: blank, a, blank, a — model produced 'a' even though half the
    # frames argmax to blank. Mode-over-non-blank should pick 'a'.
    log_probs = _flat_log_probs([0, 2, 0, 2])
    span = AlignedPhoneme(token="a", token_id=2, start_frame=0, end_frame=4, score=-0.1)
    out = _score_span("a", span, log_probs, blank_id=0, id_to_token={2: "a"})
    assert out.produced == "a"
    assert out.ok is True


def test_score_span_marks_mismatch_as_not_ok():
    log_probs = _flat_log_probs([3, 3, 3])  # produced phoneme 3, expected 2
    span = AlignedPhoneme(token="a", token_id=2, start_frame=0, end_frame=3, score=-1.0)
    out = _score_span("a", span, log_probs, blank_id=0, id_to_token={2: "a", 3: "u"})
    assert out.expected == "a"
    assert out.produced == "u"
    assert out.ok is False


# -- end-to-end score() with monkey-patched collaborators --------------------


class _StubTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab
        self.pad_token_id = vocab["<pad>"]

    def get_vocab(self):
        return self._vocab


class _StubProcessor:
    def __init__(self, vocab):
        self.tokenizer = _StubTokenizer(vocab)


def _patch_pipeline(monkeypatch, *, log_probs, target_ids, transcription):
    """Wire up score()'s collaborators with deterministic returns."""
    monkeypatch.setattr(
        score_module, "text_to_ipa", lambda text, lang="es", dialect=None: "kasa"
    )
    monkeypatch.setattr(score_module, "resolve_device", lambda dev: "cpu")
    vocab = {"<pad>": 0, "k": 1, "a": 2, "s": 3, "u": 4}
    proc = _StubProcessor(vocab)
    monkeypatch.setattr(score_module, "load", lambda model_id, dev: (proc, object()))
    monkeypatch.setattr(
        score_module,
        "reference_to_token_ids",
        lambda ref, p: (target_ids, ["k", "a", "s", "a"]),
    )
    monkeypatch.setattr(
        score_module,
        "_run_model",
        lambda *args, **kwargs: (transcription, log_probs, 0),
    )


def _stub_transcription() -> Transcription:
    return Transcription(
        ipa="k a s a",
        raw_phonemes="k a s a",
        language="es",
        dialect="castilian",
        model="stub",
        audio_seconds=0.16,
        model_load_seconds=0.0,
        inference_seconds=0.0,
    )


def test_score_end_to_end_all_correct(monkeypatch, tmp_path):
    # 8 frames, target [k, a, s, a]: model emits each cleanly with blanks between
    # adjacent identical tokens — but here all targets differ from neighbors so
    # forced_align won't insert blanks of its own accord.
    log_probs = _flat_log_probs([1, 1, 2, 2, 3, 3, 2, 2])  # k k a a s s a a
    _patch_pipeline(
        monkeypatch,
        log_probs=log_probs,
        target_ids=[1, 2, 3, 2],
        transcription=_stub_transcription(),
    )
    audio = tmp_path / "ignored.wav"
    audio.write_bytes(b"")

    result = score(audio, "casa", lang="es")
    assert isinstance(result, ScoreResult)
    assert [p.expected for p in result.phonemes] == ["k", "a", "s", "a"]
    assert [p.produced for p in result.phonemes] == ["k", "a", "s", "a"]
    assert all(p.ok for p in result.phonemes)
    assert result.per == pytest.approx(0.0)


def test_score_end_to_end_one_wrong(monkeypatch, tmp_path):
    # Same as above but the third position emits 'u' instead of 's'.
    log_probs = _flat_log_probs([1, 1, 2, 2, 4, 4, 2, 2])  # k k a a u u a a
    _patch_pipeline(
        monkeypatch,
        log_probs=log_probs,
        target_ids=[1, 2, 3, 2],
        transcription=_stub_transcription(),
    )
    audio = tmp_path / "ignored.wav"
    audio.write_bytes(b"")

    result = score(audio, "casa", lang="es")
    flags = [p.ok for p in result.phonemes]
    produced = [p.produced for p in result.phonemes]
    assert flags == [True, True, False, True]
    assert produced[2] == "u"
    assert result.per == pytest.approx(0.25)


def test_score_result_to_dict_is_jsonable(monkeypatch, tmp_path):
    import json

    log_probs = _flat_log_probs([1, 1, 2, 2, 3, 3, 2, 2])
    _patch_pipeline(
        monkeypatch,
        log_probs=log_probs,
        target_ids=[1, 2, 3, 2],
        transcription=_stub_transcription(),
    )
    audio = tmp_path / "ignored.wav"
    audio.write_bytes(b"")

    result = score(audio, "casa", lang="es")
    d = result.to_dict()
    # Round-trip through json to assert no datetime/Path/Tensor leaks
    json.dumps(d)
    assert d["per"] == 0.0
    assert isinstance(d["phonemes"], list)
    assert d["phonemes"][0]["expected"] == "k"
    assert d["transcription"]["ipa"] == "k a s a"


def test_scored_phoneme_is_dataclass():
    p = ScoredPhoneme(expected="a", produced="a", start_s=0.0, end_s=0.04, score=-0.1, ok=True)
    assert p.ok is True
