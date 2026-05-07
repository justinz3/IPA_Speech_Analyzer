"""Code-correctness tests for vocal_ipa.pipeline (no model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vocal_ipa.pipeline import Transcription, postprocess, transcribe


def _trans(**overrides) -> Transcription:
    base = dict(
        ipa="o l a",
        raw_phonemes="  o  l  a  ",
        language="es",
        model="facebook/wav2vec2-lv-60-espeak-cv-ft",
        audio_seconds=1.0,
        model_load_seconds=0.1,
        inference_seconds=0.05,
    )
    base.update(overrides)
    return Transcription(**base)


def test_transcribe_rejects_unsupported_language(sine_wav_16k: Path) -> None:
    with pytest.raises(ValueError, match="Spanish only"):
        transcribe(sine_wav_16k, lang="fr")


def test_to_dict_excludes_raw_by_default() -> None:
    d = _trans().to_dict()
    assert "raw_phonemes" not in d
    assert d["ipa"] == "o l a"
    assert d["language"] == "es"


def test_to_dict_includes_raw_when_requested() -> None:
    d = _trans().to_dict(include_raw=True)
    assert d["raw_phonemes"] == "  o  l  a  "


def test_postprocess_collapses_whitespace() -> None:
    assert postprocess("  o  l  a  ") == "o l a"
    assert postprocess("o\tl\na") == "o l a"
    assert postprocess("ola") == "ola"
