"""CLI surface tests. Model is mocked so these stay in Tier A."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from vocal_ipa.cli import app
from vocal_ipa.pipeline import Transcription

runner = CliRunner()


def _fake_transcription() -> Transcription:
    return Transcription(
        ipa="o l a",
        raw_phonemes="  o   l   a  ",
        language="es",
        model="facebook/wav2vec2-lv-60-espeak-cv-ft",
        audio_seconds=1.0,
        model_load_seconds=0.1,
        inference_seconds=0.05,
    )


def test_help_shows_usage() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Transcribe AUDIO into IPA" in result.stdout
    assert "--lang" in result.stdout
    assert "--format" in result.stdout
    assert "--raw" in result.stdout


def test_unsupported_language_exits_nonzero(sine_wav_16k: Path) -> None:
    result = runner.invoke(app, [str(sine_wav_16k), "--lang", "fr"])
    assert result.exit_code != 0


def test_missing_audio_exits_nonzero() -> None:
    result = runner.invoke(app, ["/no/such/file.wav"])
    assert result.exit_code != 0


def test_text_format_emits_ipa(sine_wav_16k: Path) -> None:
    with patch("vocal_ipa.cli.transcribe", return_value=_fake_transcription()):
        result = runner.invoke(app, [str(sine_wav_16k), "--lang", "es"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "o l a"


def test_raw_flag_emits_raw_phonemes(sine_wav_16k: Path) -> None:
    with patch("vocal_ipa.cli.transcribe", return_value=_fake_transcription()):
        result = runner.invoke(app, [str(sine_wav_16k), "--lang", "es", "--raw"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "o   l   a"  # whitespace preserved in raw


def test_json_format_emits_valid_json(sine_wav_16k: Path) -> None:
    with patch("vocal_ipa.cli.transcribe", return_value=_fake_transcription()):
        result = runner.invoke(app, [str(sine_wav_16k), "--lang", "es", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ipa"] == "o l a"
    assert payload["language"] == "es"
    assert payload["audio_seconds"] == 1.0
    assert "raw_phonemes" not in payload  # excluded unless --raw


def test_json_format_with_raw_includes_raw_phonemes(sine_wav_16k: Path) -> None:
    with patch("vocal_ipa.cli.transcribe", return_value=_fake_transcription()):
        result = runner.invoke(
            app, [str(sine_wav_16k), "--lang", "es", "--format", "json", "--raw"]
        )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["raw_phonemes"] == "  o   l   a  "
