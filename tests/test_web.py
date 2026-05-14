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

    ipa, raw, msg, scored = _run(None, "", "es", None)
    assert ipa == ""
    assert raw == ""
    assert "audio" in msg.lower()
    assert scored == ""


def test_run_with_reference_renders_scored_html(monkeypatch, tmp_path) -> None:
    from vocal_ipa import web as web_module
    from vocal_ipa.pipeline import Transcription
    from vocal_ipa.score import ScoredPhoneme, ScoreResult

    def fake_score(audio_path, reference_text, lang="es", dialect=None):
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
    # IPA tokens in the score table should be wrapped in tooltip spans.
    assert 'class="ipa-tip"' in scored
    assert 'data-tip=' in scored


def _run_helper(audio_path, reference: str, lang: str = "es", dialect: str | None = None):
    from vocal_ipa.web import _run

    return _run(str(audio_path), reference, lang, dialect)


def test_run_threads_lang_into_score(monkeypatch, tmp_path) -> None:
    """Selecting French in the radio must propagate to score()."""
    from vocal_ipa import web as web_module
    from vocal_ipa.pipeline import Transcription
    from vocal_ipa.score import ScoredPhoneme, ScoreResult

    captured = {}

    def fake_score(audio_path, reference_text, lang="es", dialect=None):
        captured["lang"] = lang
        captured["dialect"] = dialect
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
    assert captured["dialect"] is None  # web sends None when "Default" picked


def test_run_threads_cmn_into_score(monkeypatch, tmp_path) -> None:
    """Selecting Mandarin in the radio must propagate to score()."""
    from vocal_ipa import web as web_module
    from vocal_ipa.pipeline import Transcription
    from vocal_ipa.score import ScoredPhoneme, ScoreResult

    captured = {}

    def fake_score(audio_path, reference_text, lang="es", dialect=None):
        captured["lang"] = lang
        captured["dialect"] = dialect
        return ScoreResult(
            phonemes=[
                ScoredPhoneme(
                    expected="n", produced="n", start_s=0.0, end_s=0.02, score=-0.1, ok=True
                ),
            ],
            per=0.0,
            reference_ipa="ni2 χɑu2",
            transcription=Transcription(
                ipa="n",
                raw_phonemes="n",
                language="cmn",
                dialect="cmn-cn",
                model="stub",
                audio_seconds=0.02,
                model_load_seconds=0.0,
                inference_seconds=0.0,
            ),
            language="cmn",
            dialect="cmn-cn",
        )

    monkeypatch.setattr(web_module, "score", fake_score)

    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")
    _run_helper(audio, reference="你好", lang="cmn")
    assert captured["lang"] == "cmn"
    assert captured["dialect"] is None


def test_dialect_choices_includes_cmn() -> None:
    """The web UI's dialect map must offer at least one cmn entry so the
    dropdown isn't empty when Mandarin is selected."""
    from vocal_ipa.web import _DIALECT_CHOICES

    assert "cmn" in _DIALECT_CHOICES
    cmn_choices = _DIALECT_CHOICES["cmn"]
    assert len(cmn_choices) >= 1
    # Default option (None value) is always present.
    assert any(value is None for _label, value in cmn_choices)


def test_dialect_choices_includes_ja() -> None:
    from vocal_ipa.web import _DIALECT_CHOICES

    assert "ja" in _DIALECT_CHOICES
    ja_choices = _DIALECT_CHOICES["ja"]
    assert len(ja_choices) >= 1
    assert any(value is None for _label, value in ja_choices)


def test_run_threads_ja_into_score(monkeypatch, tmp_path) -> None:
    """Selecting Japanese in the radio must propagate to score()."""
    from vocal_ipa import web as web_module
    from vocal_ipa.pipeline import Transcription
    from vocal_ipa.score import ScoredPhoneme, ScoreResult

    captured = {}

    def fake_score(audio_path, reference_text, lang="es", dialect=None):
        captured["lang"] = lang
        captured["dialect"] = dialect
        return ScoreResult(
            phonemes=[
                ScoredPhoneme(
                    expected="k", produced="k", start_s=0.0, end_s=0.02, score=-0.1, ok=True
                ),
            ],
            per=0.0,
            reference_ipa="k o ɴ n i tɕ i w a",
            transcription=Transcription(
                ipa="k",
                raw_phonemes="k",
                language="ja",
                dialect="ja-jp",
                model="stub",
                audio_seconds=0.02,
                model_load_seconds=0.0,
                inference_seconds=0.0,
            ),
            language="ja",
            dialect="ja-jp",
        )

    monkeypatch.setattr(web_module, "score", fake_score)
    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")
    _run_helper(audio, reference="こんにちは", lang="ja")
    assert captured["lang"] == "ja"
    assert captured["dialect"] is None


def test_run_threads_dialect_into_score(monkeypatch, tmp_path) -> None:
    """Selecting an explicit dialect must propagate to score()."""
    from vocal_ipa import web as web_module
    from vocal_ipa.pipeline import Transcription
    from vocal_ipa.score import ScoredPhoneme, ScoreResult

    captured = {}

    def fake_score(audio_path, reference_text, lang="es", dialect=None):
        captured["dialect"] = dialect
        return ScoreResult(
            phonemes=[
                ScoredPhoneme(
                    expected="m", produced="m", start_s=0.0, end_s=0.02, score=-0.1, ok=True
                ),
            ],
            per=0.0,
            reference_ipa="mansana",
            transcription=Transcription(
                ipa="m",
                raw_phonemes="m",
                language="es",
                dialect="es-419",
                model="stub",
                audio_seconds=0.02,
                model_load_seconds=0.0,
                inference_seconds=0.0,
            ),
            language="es",
            dialect="es-419",
        )

    monkeypatch.setattr(web_module, "score", fake_score)

    audio = tmp_path / "fake.wav"
    audio.write_bytes(b"")
    _run_helper(audio, reference="manzana", lang="es", dialect="es-419")
    assert captured["dialect"] == "es-419"


def test_build_app_lays_out_lang_and_dialect_controls() -> None:
    """The app must expose both Language and Dialect widgets."""
    from vocal_ipa.web import build_app

    app = build_app()
    # Walk the components to find labels — Gradio Blocks expose .blocks dict.
    labels = []
    for component in app.blocks.values():
        label = getattr(component, "label", None)
        if isinstance(label, str):
            labels.append(label)
    assert "Language" in labels
    assert "Dialect" in labels


def test_render_scored_html_includes_miss_cards_when_present() -> None:
    from vocal_ipa.coaching import MissReference, Phoneme, Tip
    from vocal_ipa.pipeline import Transcription
    from vocal_ipa.score import ScoredPhoneme, ScoreResult
    from vocal_ipa.web import _render_scored_html

    miss = MissReference(
        expected=Phoneme(token="β", name="voiced bilabial fricative"),
        produced=Phoneme(token="v", name="voiced labiodental fricative"),
        tip=Tip(
            lang="es",
            expected="β",
            produced="v",
            title="Don't use English [v]",
            tip="Lips touch lightly without contact between lower lip and upper teeth.",
        ),
    )
    result = ScoreResult(
        phonemes=[
            ScoredPhoneme(
                expected="β", produced="v", start_s=0.0, end_s=0.04, score=-1.5, ok=False
            ),
        ],
        per=1.0,
        reference_ipa="β",
        transcription=Transcription(
            ipa="v",
            raw_phonemes="v",
            language="es",
            dialect="es-es",
            model="stub",
            audio_seconds=0.04,
            model_load_seconds=0.0,
            inference_seconds=0.0,
        ),
        language="es",
        dialect="es-es",
        miss_references=[miss],
    )

    html = _render_scored_html(result)
    assert 'class="misses-heading"' in html
    assert "Misses (1 unique):" in html
    assert 'class="miss-card"' in html
    # Both sides present.
    assert "Expected" in html
    assert "You said" in html
    assert "voiced bilabial fricative" in html
    assert "voiced labiodental fricative" in html
    # Tip rendered.
    assert "Don&#x27;t use English [v]" in html or "Don't use English [v]" in html
    # No image/audio elements (skeleton inventory has null fields).
    assert "<img" not in html
    assert "<audio" not in html


def test_render_scored_html_omits_misses_block_when_empty() -> None:
    from vocal_ipa.pipeline import Transcription
    from vocal_ipa.score import ScoredPhoneme, ScoreResult
    from vocal_ipa.web import _render_scored_html

    result = ScoreResult(
        phonemes=[
            ScoredPhoneme(expected="a", produced="a", start_s=0.0, end_s=0.04, score=-0.1, ok=True),
        ],
        per=0.0,
        reference_ipa="a",
        transcription=Transcription(
            ipa="a",
            raw_phonemes="a",
            language="es",
            dialect="es-es",
            model="stub",
            audio_seconds=0.04,
            model_load_seconds=0.0,
            inference_seconds=0.0,
        ),
        language="es",
        dialect="es-es",
        miss_references=[],
    )
    html = _render_scored_html(result)
    # The CSS rules with these classes are always present (in <style>);
    # check for the actual elements opening tags instead.
    assert 'class="misses-heading"' not in html
    assert 'class="miss-card"' not in html


def test_render_library_html_contains_vowel_cards() -> None:
    from html import escape

    from vocal_ipa.web import _render_library_html

    html = _render_library_html()
    # Core vowels should appear as ph-token entries.
    for token in ("a", "e", "i", "u", "y", "ø", "ɛ", "ɔ̃"):
        assert escape(token) in html or token in html, f"missing token {token!r} in library"
    # Audio players and SVG images should appear for vowels.
    assert "<audio" in html
    assert "_pos.svg" in html
    # IPA chart image at top.
    assert "ipa_vowel_chart.png" in html
    # Consonant section should also appear.
    assert "β" in html
    assert "ʁ" in html


def test_render_miss_side_emits_image_when_phoneme_has_image() -> None:
    from vocal_ipa.coaching import MissReference, Phoneme
    from vocal_ipa.web import _render_miss_card

    ref = MissReference(
        expected=Phoneme(
            token="β",
            name="voiced bilabial fricative",
            image="phonemes/voiced_bilabial_fricative.png",
            audio="phonemes/voiced_bilabial_fricative.ogg",
        ),
        produced=None,
        tip=None,
    )
    html = _render_miss_card(ref)
    # Image and audio elements should reference the package data path
    # via Gradio's /gradio_api/file= scheme.
    assert "<img" in html
    assert "voiced_bilabial_fricative.png" in html
    assert "/gradio_api/file=" in html
    assert "<audio" in html
    assert "voiced_bilabial_fricative.ogg" in html
    # No produced side because produced is None.
    assert "You said" not in html
