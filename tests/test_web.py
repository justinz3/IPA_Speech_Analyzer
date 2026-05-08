"""Smoke test for the Gradio app. Skipped when [web] extra isn't installed."""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")


def test_build_app_constructs_without_error() -> None:
    from vocal_ipa.web import build_app

    app = build_app()
    assert isinstance(app, gr.Blocks)


def test_run_with_empty_audio_returns_helpful_message() -> None:
    from vocal_ipa.web import _run

    ipa, raw, msg, scored = _run(None, "")
    assert ipa == ""
    assert raw == ""
    assert "audio" in msg.lower()
    assert scored == ""


def test_run_with_reference_renders_scored_html(monkeypatch, tmp_path) -> None:
    from vocal_ipa import web as web_module
    from vocal_ipa.pipeline import Transcription
    from vocal_ipa.score import ScoredPhoneme, ScoreResult

    def fake_score(audio_path, reference_text, lang="es"):
        return ScoreResult(
            phonemes=[
                ScoredPhoneme(
                    expected="k", produced="k", start_s=0.0, end_s=0.04, score=-0.1, ok=True
                ),
                ScoredPhoneme(
                    expected="a", produced="o", start_s=0.04, end_s=0.10, score=-1.5, ok=False
                ),
            ],
            per=0.5,
            reference_ipa="ka",
            transcription=Transcription(
                ipa="k o",
                raw_phonemes="k o",
                language="es",
                dialect="es-es",
                model="stub",
                audio_seconds=0.1,
                model_load_seconds=0.0,
                inference_seconds=0.0,
            ),
            language="es",
            dialect="es-es",
        )

    monkeypatch.setattr(web_module, "score", fake_score)

    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")
    ipa, raw, timing, scored = _run_helper(audio, reference="ka")

    assert ipa == "k o"
    assert raw == "k o"
    assert "0.10s" in timing
    # Highlight markup includes both ok and miss spans.
    assert 'class="phoneme ok"' in scored
    assert 'class="phoneme miss"' in scored
    assert 'data-start="0.04"' in scored
    assert "1/2 phonemes wrong" in scored


def _run_helper(audio_path, reference: str, lang: str = "es"):
    from vocal_ipa.web import _run

    return _run(str(audio_path), reference, lang)


def test_run_threads_lang_into_score(monkeypatch, tmp_path) -> None:
    """Selecting French in the radio must propagate to score()."""
    from vocal_ipa import web as web_module
    from vocal_ipa.pipeline import Transcription
    from vocal_ipa.score import ScoredPhoneme, ScoreResult

    captured = {}

    def fake_score(audio_path, reference_text, lang="es"):
        captured["lang"] = lang
        return ScoreResult(
            phonemes=[
                ScoredPhoneme(
                    expected="b", produced="b", start_s=0.0, end_s=0.02, score=-0.1, ok=True
                ),
            ],
            per=0.0,
            reference_ipa="bɔ̃ʒuʁ",
            transcription=Transcription(
                ipa="b",
                raw_phonemes="b",
                language="fr",
                dialect="fr-fr",
                model="stub",
                audio_seconds=0.02,
                model_load_seconds=0.0,
                inference_seconds=0.0,
            ),
            language="fr",
            dialect="fr-fr",
        )

    monkeypatch.setattr(web_module, "score", fake_score)

    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")
    _run_helper(audio, reference="bonjour", lang="fr")
    assert captured["lang"] == "fr"
