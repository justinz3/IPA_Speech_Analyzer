"""Gradio web UI: in-browser microphone recording -> IPA.

Reuses the same `transcribe()` / `score()` functions as the CLI; no logic
duplication. Install with the optional [web] extra:

    uv tool install vocal-ipa-trainer[web]
    pronounce-web

Or run from a clone:

    uv sync --extra web
    uv run pronounce-web
"""

from __future__ import annotations

from html import escape

import gradio as gr

from .pipeline import transcribe
from .score import ScoreResult, score

DESCRIPTION = """\
# Audio → IPA

Record yourself speaking Spanish (or upload a file). Leave **Reference text**
empty for free transcription, or paste the sentence you meant to read to get
per-phoneme scoring against the reference IPA.

*Phase 1: Spanish only. The model loads on first use (~1 GB download), then
subsequent runs take well under a second.*
"""

_SCORED_STYLES = """
<style>
.scored-line { font-size: 1.4em; line-height: 2.2em; margin: 0.5em 0; }
.scored-line .phoneme { padding: 2px 4px; border-radius: 4px; margin-right: 2px; }
/* Force both bg + fg so the highlight is readable on dark themes. */
.scored-line .phoneme.ok { background: #d4edda; color: #155724; }
.scored-line .phoneme.miss { background: #f8d7da; color: #721c24; text-decoration: line-through; }
.scored-summary { margin: 0.5em 0; font-weight: 600; }
.scored-table { border-collapse: collapse; font-size: 0.95em; }
.scored-table th, .scored-table td { padding: 4px 10px; border-bottom: 1px solid currentColor; text-align: left; }
/* No row tinting — ✓/✗ already conveys ok/miss, and a tinted bg without an
   explicit fg breaks contrast on dark themes (Gradio's text turns white on
   dark, so light-pink bg + white text ≈ white-on-white). */
</style>
"""


def _run(audio_path: str | None, reference_text: str) -> tuple[str, str, str, str]:
    """Returns (ipa, raw_phonemes, timing_summary, scored_html).

    Empty reference → existing free-transcribe path (scored_html is blank).
    Non-empty reference → score path (textboxes hold the underlying free
    transcription so users can compare).
    """
    if not audio_path:
        return ("", "", "Record or upload audio first.", "")

    if not reference_text.strip():
        result = transcribe(audio_path, lang="es")
        timing = _format_timing(
            audio_s=result.audio_seconds,
            load_s=result.model_load_seconds,
            inference_s=result.inference_seconds,
        )
        return result.ipa, result.raw_phonemes, timing, ""

    score_result = score(audio_path, reference_text, lang="es")
    txn = score_result.transcription
    timing = _format_timing(
        audio_s=txn.audio_seconds,
        load_s=txn.model_load_seconds,
        inference_s=txn.inference_seconds,
    )
    return txn.ipa, txn.raw_phonemes, timing, _render_scored_html(score_result)


def _format_timing(*, audio_s: float, load_s: float, inference_s: float) -> str:
    return f"audio {audio_s:.2f}s · model load {load_s:.2f}s · inference {inference_s:.2f}s"


def _render_scored_html(result: ScoreResult) -> str:
    wrong = sum(1 for p in result.phonemes if not p.ok)
    total = len(result.phonemes)
    spans = []
    rows = []
    for p in result.phonemes:
        cls = "ok" if p.ok else "miss"
        title = "ok" if p.ok else f"produced: {p.produced}"
        spans.append(
            f'<span class="phoneme {cls}" '
            f'data-start="{p.start_s:.2f}" data-end="{p.end_s:.2f}" '
            f'title="{escape(title)}">{escape(p.expected)}</span>'
        )
        rows.append(
            f"<tr><td>{escape(p.expected)}</td><td>{escape(p.produced)}</td>"
            f"<td>{p.start_s:.2f}</td><td>{p.end_s:.2f}</td>"
            f"<td>{'✓' if p.ok else '✗'}</td></tr>"
        )
    table = (
        '<table class="scored-table"><thead><tr>'
        "<th>expected</th><th>produced</th><th>start</th><th>end</th><th>ok</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    summary = (
        f'<div class="scored-summary">PER {result.per:.3f} ({wrong}/{total} phonemes wrong)</div>'
    )
    body = '<div class="scored-line">' + "".join(spans) + "</div>" + summary + table
    return _SCORED_STYLES + body


def build_app() -> gr.Blocks:
    with gr.Blocks(title="vocal-ipa-trainer") as app:
        gr.Markdown(DESCRIPTION)
        reference = gr.Textbox(
            label="Reference text (optional)",
            placeholder="que pase un buen día",
            lines=1,
        )
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio")
        btn = gr.Button("Transcribe / Score", variant="primary")
        scored = gr.HTML(label="Scoring", visible=True)
        ipa = gr.Textbox(label="IPA", lines=2, show_copy_button=True)
        raw = gr.Textbox(label="Raw model output", lines=2, show_copy_button=True)
        timing = gr.Markdown()
        btn.click(_run, inputs=[audio, reference], outputs=[ipa, raw, timing, scored])
    return app


def main() -> None:
    build_app().launch()


if __name__ == "__main__":
    main()
