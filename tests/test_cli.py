"""CLI surface tests (no model invoked)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from vocal_ipa.cli import app

runner = CliRunner()


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
