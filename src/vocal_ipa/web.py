"""Gradio web UI: in-browser microphone recording -> IPA.

Reuses the same `transcribe()` function as the CLI; no logic duplication.
Install with the optional [web] extra:

    uv tool install vocal-ipa-trainer[web]
    pronounce-web

Or run from a clone:

    uv sync --extra web
    uv run pronounce-web
"""

from __future__ import annotations

import gradio as gr

from .pipeline import transcribe

DESCRIPTION = """\
# Audio → IPA

Record yourself speaking Spanish (or upload a file), then click **Transcribe**
to see the International Phonetic Alphabet representation of what you said.

*Phase 1: Spanish only. The model loads on first use (~1 GB download), then
subsequent transcriptions take well under a second.*
"""


def _run(audio_path: str | None) -> tuple[str, str, str]:
    """Returns (ipa, raw_phonemes, timing_summary)."""
    if not audio_path:
        return ("", "", "Record or upload audio first.")
    result = transcribe(audio_path, lang="es")
    timing = (
        f"audio {result.audio_seconds:.2f}s · "
        f"model load {result.model_load_seconds:.2f}s · "
        f"inference {result.inference_seconds:.2f}s"
    )
    return result.ipa, result.raw_phonemes, timing


def build_app() -> gr.Blocks:
    with gr.Blocks(title="vocal-ipa-trainer") as app:
        gr.Markdown(DESCRIPTION)
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
        btn = gr.Button("Transcribe", variant="primary")
        ipa = gr.Textbox(label="IPA", lines=2, show_copy_button=True)
        raw = gr.Textbox(label="Raw model output", lines=2, show_copy_button=True)
        timing = gr.Markdown()
        btn.click(_run, inputs=audio, outputs=[ipa, raw, timing])
    return app


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
