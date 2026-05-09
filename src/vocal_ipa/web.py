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
from importlib.resources import files

import gradio as gr

from .coaching import MissReference
from .pipeline import transcribe
from .score import ScoreResult, score

DESCRIPTION = """\
# Audio → IPA

Pick a language (and optional dialect), record yourself speaking (or upload
a file). Leave **Reference text** empty for free transcription, or paste
the sentence you meant to read to get per-phoneme scoring against the
reference IPA.

*Spanish, French, Mandarin (Hanzi or pinyin), and Japanese (Kanji + kana).
The model loads on first use (~1 GB download), then subsequent runs take
well under a second.*
"""

_SCORED_STYLES = """
<style>
.scored-line { font-size: 1.4em; line-height: 2.2em; margin: 0.5em 0; }
.scored-line .phoneme { padding: 2px 4px; border-radius: 4px; margin-right: 2px; }
/* Force both bg + fg so the highlight is readable on dark themes. */
.scored-line .phoneme.ok { background: #d4edda; color: #155724; }
.scored-line .phoneme.miss { background: #f8d7da; color: #721c24; text-decoration: line-through; }
.scored-summary { margin: 0.5em 0; font-weight: 600; }
.scored-locale { font-size: 0.9em; opacity: 0.75; margin-bottom: 0.4em; }
.scored-table { border-collapse: collapse; font-size: 0.95em; }
.scored-table th, .scored-table td { padding: 4px 10px; border-bottom: 1px solid currentColor; text-align: left; }
/* No row tinting — ✓/✗ already conveys ok/miss, and a tinted bg without an
   explicit fg breaks contrast on dark themes (Gradio's text turns white on
   dark, so light-pink bg + white text ≈ white-on-white). */

/* Miss comparison cards (Phase 4b coaching panel). */
.misses-heading { margin: 1em 0 0.5em; font-weight: 600; }
.miss-card { display: flex; flex-wrap: wrap; gap: 1em; margin: 0.7em 0;
             padding: 0.8em; border: 1px solid currentColor; border-radius: 6px;
             align-items: flex-start; }
.miss-side { flex: 1 1 220px; min-width: 200px; }
.miss-label { display: block; font-size: 0.85em; opacity: 0.7; }
.miss-token { font-size: 1.6em; margin-right: 0.4em; }
.miss-name { font-style: italic; }
.miss-card img { display: block; max-width: 100%; height: auto; margin: 0.4em 0; }
.miss-card audio { display: block; width: 100%; margin: 0.4em 0; }
.miss-tip { flex: 1 1 100%; padding-top: 0.4em; border-top: 1px solid currentColor;
            font-size: 0.95em; }
.miss-tip strong { display: block; margin-bottom: 0.2em; }
.miss-tip p { margin: 0; white-space: pre-wrap; }
</style>
"""

_DATA_DIR = files("vocal_ipa") / "data"

# Per-language dialect options as (label, value) tuples for gr.Dropdown.
# `None` = use the language's default. Spanish has a real dialect axis;
# French currently only has fr-fr at the IPA level (regional voices share
# rules in espeak), so the dropdown shows just the default for fr. Mandarin
# uses the cmn-latn-pinyin voice; the cmn-cn dialect is the only option.
_DIALECT_CHOICES: dict[str, list[tuple[str, str | None]]] = {
    "es": [
        ("Default (es-es Castilian)", None),
        ("es-es (Castilian)", "es-es"),
        ("es-419 (Latin American)", "es-419"),
    ],
    "fr": [
        ("Default (fr-fr Parisian)", None),
        ("fr-fr (Parisian)", "fr-fr"),
    ],
    "cmn": [
        ("Default (cmn-cn Mandarin)", None),
        ("cmn-cn (Mandarin)", "cmn-cn"),
    ],
    "ja": [
        ("Default (ja-jp)", None),
        ("ja-jp", "ja-jp"),
    ],
}


def _run(
    audio_path: str | None,
    reference_text: str,
    lang: str = "es",
    dialect: str | None = None,
) -> tuple[str, str, str, str]:
    """Returns (ipa, raw_phonemes, timing_summary, scored_html).

    Empty reference → existing free-transcribe path (scored_html is blank).
    Non-empty reference → score path (textboxes hold the underlying free
    transcription so users can compare).
    """
    if not audio_path:
        return ("", "", "Record or upload audio first.", "")

    if not reference_text.strip():
        result = transcribe(audio_path, lang=lang, dialect=dialect)
        timing = _format_timing(
            audio_s=result.audio_seconds,
            load_s=result.model_load_seconds,
            inference_s=result.inference_seconds,
        )
        return result.ipa, result.raw_phonemes, timing, ""

    score_result = score(audio_path, reference_text, lang=lang, dialect=dialect)
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
    locale_line = (
        f'<div class="scored-locale">language: {escape(result.language)} '
        f"({escape(result.dialect)})</div>"
    )
    summary = (
        f'<div class="scored-summary">PER {result.per:.3f} ({wrong}/{total} phonemes wrong)</div>'
    )
    misses_html = _render_misses_html(result.miss_references)
    body = (
        locale_line
        + '<div class="scored-line">'
        + "".join(spans)
        + "</div>"
        + summary
        + table
        + misses_html
    )
    return _SCORED_STYLES + body


def _render_misses_html(miss_refs: list[MissReference]) -> str:
    if not miss_refs:
        return ""
    cards = []
    for ref in miss_refs:
        cards.append(_render_miss_card(ref))
    return f'<div class="misses-heading">Misses ({len(miss_refs)} unique):</div>' + "".join(cards)


def _render_miss_card(ref: MissReference) -> str:
    parts = ['<div class="miss-card">']
    parts.append(_render_miss_side("Expected", ref.expected))
    if ref.produced is not None:
        parts.append(_render_miss_side("You said", ref.produced))
    if ref.tip is not None:
        parts.append(
            '<div class="miss-tip">'
            f"<strong>{escape(ref.tip.title)}</strong>"
            f"<p>{escape(ref.tip.tip.rstrip())}</p>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _render_miss_side(label: str, phoneme) -> str:
    parts = [
        '<div class="miss-side">',
        f'<span class="miss-label">{escape(label)}</span>',
        f'<span class="miss-token">{escape(phoneme.token)}</span>',
        f'<span class="miss-name">{escape(phoneme.name)}</span>',
    ]
    image_src = _media_url(phoneme.image)
    if image_src:
        parts.append(f'<img src="{image_src}" alt="{escape(phoneme.name)} diagram">')
    audio_src = _media_url(phoneme.audio)
    if audio_src:
        parts.append(f'<audio controls preload="none" src="{audio_src}"></audio>')
    if phoneme.video:
        parts.append(
            f'<a href="{escape(phoneme.video)}" target="_blank" rel="noreferrer">video</a>'
        )
    parts.append("</div>")
    return "".join(parts)


def _media_url(rel_path: str | None) -> str | None:
    """Map a phoneme.yaml relative path (e.g. "phonemes/foo.png") to a URL
    Gradio will serve via its allowed_paths-backed file route."""
    if not rel_path:
        return None
    abs_path = str(_DATA_DIR / rel_path)
    return f"/gradio_api/file={abs_path}"


def _on_lang_change(new_lang: str) -> gr.Dropdown:
    """Refresh the dialect dropdown's choices when the language radio changes."""
    return gr.Dropdown(choices=_DIALECT_CHOICES[new_lang], value=None)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="vocal-ipa-trainer") as app:
        gr.Markdown(DESCRIPTION)
        with gr.Row():
            lang = gr.Radio(choices=["es", "fr", "cmn", "ja"], value="es", label="Language")
            dialect = gr.Dropdown(
                choices=_DIALECT_CHOICES["es"],
                value=None,
                label="Dialect",
                allow_custom_value=False,
            )
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
        lang.change(_on_lang_change, inputs=[lang], outputs=[dialect])
        btn.click(
            _run,
            inputs=[audio, reference, lang, dialect],
            outputs=[ipa, raw, timing, scored],
        )
    return app


def main() -> None:
    # Allow Gradio to serve phoneme reference media (images, audio) shipped
    # in the package's data/ directory. Required for the miss-comparison
    # cards once 4b-11 populates image/audio fields.
    build_app().launch(allowed_paths=[str(_DATA_DIR)])


if __name__ == "__main__":
    main()
