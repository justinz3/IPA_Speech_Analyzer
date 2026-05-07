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

    ipa, raw, msg = _run(None)
    assert ipa == ""
    assert raw == ""
    assert "audio" in msg.lower()
