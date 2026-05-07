"""CLI surface tests. Model is mocked so these stay in Tier A."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from vocal_ipa.cli import app
from vocal_ipa.pipeline import Transcription
from vocal_ipa.score import ScoredPhoneme, ScoreResult

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
    result = runner.invoke(app, [str(sine_wav_16k), "--lang", "ja"])
    assert result.exit_code != 0


def test_french_lang_is_accepted(sine_wav_16k: Path) -> None:
    fake = Transcription(
        ipa="b ɔ̃ ʒ u ʁ",
        raw_phonemes="b ɔ̃ ʒ u ʁ",
        language="fr",
        model="facebook/wav2vec2-lv-60-espeak-cv-ft",
        audio_seconds=1.0,
        model_load_seconds=0.1,
        inference_seconds=0.05,
    )
    with patch("vocal_ipa.cli.transcribe", return_value=fake):
        result = runner.invoke(app, [str(sine_wav_16k), "--lang", "fr"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "b ɔ̃ ʒ u ʁ"


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


# -- --reference scoring path -------------------------------------------------


def _fake_score_result(per: float = 0.25) -> ScoreResult:
    return ScoreResult(
        phonemes=[
            ScoredPhoneme(
                expected="k", produced="k", start_s=0.04, end_s=0.10, score=-0.1, ok=True
            ),
            ScoredPhoneme(
                expected="a", produced="a", start_s=0.10, end_s=0.18, score=-0.2, ok=True
            ),
            ScoredPhoneme(
                expected="s", produced="s", start_s=0.18, end_s=0.22, score=-0.3, ok=True
            ),
            ScoredPhoneme(
                expected="a", produced="o", start_s=0.22, end_s=0.30, score=-1.5, ok=False
            ),
        ],
        per=per,
        reference_ipa="kasa",
        transcription=_fake_transcription(),
    )


def test_reference_text_format_renders_table(sine_wav_16k: Path) -> None:
    with patch("vocal_ipa.cli.score", return_value=_fake_score_result()):
        result = runner.invoke(app, [str(sine_wav_16k), "--reference", "casa"])
    assert result.exit_code == 0
    assert "expected" in result.stdout
    assert "produced" in result.stdout
    assert "PER:" in result.stdout
    # Ok and not-ok rows both present
    assert "✓" in result.stdout
    assert "✗" in result.stdout
    assert "1/4 phonemes wrong" in result.stdout


def test_reference_json_format_emits_score_result(sine_wav_16k: Path) -> None:
    with patch("vocal_ipa.cli.score", return_value=_fake_score_result(per=0.25)):
        result = runner.invoke(app, [str(sine_wav_16k), "--reference", "casa", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["per"] == 0.25
    assert len(payload["phonemes"]) == 4
    assert payload["phonemes"][3]["ok"] is False
    assert payload["transcription"]["ipa"] == "o l a"


def test_reference_with_raw_is_rejected(sine_wav_16k: Path) -> None:
    result = runner.invoke(app, [str(sine_wav_16k), "--reference", "casa", "--raw"])
    assert result.exit_code != 0
    assert "raw" in result.stderr.lower() or "raw" in result.stdout.lower()


def test_reference_with_unsupported_lang_exits_nonzero(sine_wav_16k: Path) -> None:
    # Lang enum restricts to {es, fr}; ja is not in the enum, so typer rejects.
    result = runner.invoke(app, [str(sine_wav_16k), "--reference", "今日は", "--lang", "ja"])
    assert result.exit_code != 0
