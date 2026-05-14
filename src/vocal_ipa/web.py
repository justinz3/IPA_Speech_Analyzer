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

from .coaching import MissReference, load_phonemes, load_phrases
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

/* IPA token tooltips — shown on hover over any token in the score table. */
.ipa-tip {
  position: relative;
  cursor: help;
  text-decoration: underline dotted currentColor;
}
.ipa-tip[data-tip]::after {
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 4px);
  left: 50%;
  transform: translateX(-50%);
  background: #222;
  color: #f8f8f8;
  padding: 5px 9px;
  border-radius: 5px;
  white-space: pre-wrap;
  max-width: 280px;
  font-size: 0.82em;
  line-height: 1.35;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.12s ease;
  z-index: 50;
}
.ipa-tip[data-tip]:hover::after { opacity: 1; }
</style>
"""

_DATA_DIR = files("vocal_ipa") / "data"

# Per-language dialect options as (label, value) tuples for gr.Dropdown.
# `None` = use the language's default. Spanish has a real dialect axis;
# French currently only has fr-fr at the IPA level (regional voices share
# rules in espeak), so the dropdown shows just the default for fr. Mandarin
# uses the cmn-latn-pinyin voice; the cmn-cn dialect is the only option.
_DIALECT_CHOICES: dict[str, list[tuple[str, str | None]]] = {
    "en": [
        ("Default (en-us American)", None),
        ("en-us (American)", "en-us"),
        ("en-gb (British RP)", "en-gb"),
    ],
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


def _ipa_tip(token: str, lang: str, inv: dict) -> str:
    """Wrap an IPA token in a CSS tooltip span if it is in the phoneme inventory."""
    phoneme = inv.get(token)
    if phoneme is None:
        return escape(token)
    tip = phoneme.name
    note = phoneme.notes.get(lang, "")
    if note:
        tip += "\n" + note[:100]
    return f'<span class="ipa-tip" data-tip="{escape(tip)}">{escape(token)}</span>'


def _render_scored_html(result: ScoreResult) -> str:
    wrong = sum(1 for p in result.phonemes if not p.ok)
    total = len(result.phonemes)
    has_prosody = any(p.prosody is not None for p in result.phonemes)
    inv = load_phonemes()
    lang = result.language
    spans = []
    rows = []
    for p in result.phonemes:
        cls = "ok" if p.ok else "miss"
        # Build tooltip title for the scored-line span: phoneme name + prosody if any.
        phoneme = inv.get(p.expected)
        tip_parts = [phoneme.name if phoneme else p.expected]
        if not p.ok:
            tip_parts.append(f"produced: {p.produced}")
        if p.prosody is not None:
            tip_parts.append(p.prosody.label)
        spans.append(
            f'<span class="phoneme {cls}" '
            f'data-start="{p.start_s:.2f}" data-end="{p.end_s:.2f}" '
            f'title="{escape(" | ".join(tip_parts))}">{escape(p.expected)}</span>'
        )
        row = (
            f"<tr><td>{_ipa_tip(p.expected, lang, inv)}</td>"
            f"<td>{_ipa_tip(p.produced, lang, inv)}</td>"
            f"<td>{p.start_s:.2f}</td><td>{p.end_s:.2f}</td>"
            f"<td>{'✓' if p.ok else '✗'}</td>"
        )
        if has_prosody:
            prosody_str = escape(p.prosody.label) if p.prosody is not None else ""
            row += f"<td>{prosody_str}</td>"
        row += "</tr>"
        rows.append(row)

    header_cells = "<th>expected</th><th>produced</th><th>start</th><th>end</th><th>ok</th>"
    if has_prosody:
        header_cells += "<th>prosody</th>"
    table = (
        f'<table class="scored-table"><thead><tr>{header_cells}'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )
    locale_line = (
        f'<div class="scored-locale">language: {escape(result.language)} '
        f"({escape(result.dialect)})</div>"
    )
    per_line = f"PER {result.per:.3f} ({wrong}/{total} phonemes wrong)"
    if result.prosody_score is not None:
        per_line += f" · Prosody {int(result.prosody_score * 100)}% correct"
    summary = f'<div class="scored-summary">{per_line}</div>'
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


def _phoneme_media_html(phoneme, alt_suffix: str = "") -> list[str]:
    """Return <img> and <audio> elements for a phoneme's media fields, if present."""
    parts = []
    image_src = _media_url(phoneme.image)
    if image_src:
        parts.append(f'<img src="{image_src}" alt="{escape(phoneme.name)}{alt_suffix}">')
    audio_src = _media_url(phoneme.audio)
    if audio_src:
        parts.append(f'<audio controls preload="none" src="{audio_src}"></audio>')
    return parts


def _render_miss_side(label: str, phoneme) -> str:
    parts = [
        '<div class="miss-side">',
        f'<span class="miss-label">{escape(label)}</span>',
        f'<span class="miss-token">{escape(phoneme.token)}</span>',
        f'<span class="miss-name">{escape(phoneme.name)}</span>',
    ]
    parts.extend(_phoneme_media_html(phoneme, alt_suffix=" diagram"))
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


_LIBRARY_STYLES = """
<style>
.ph-chart { max-width: 420px; margin: 0.5em 0 1.5em; }
.ph-chart img { max-width: 100%; height: auto; border-radius: 4px; }
.ph-group-heading { font-size: 1.1em; font-weight: 700; margin: 1.4em 0 0.6em;
                    border-bottom: 2px solid currentColor; padding-bottom: 0.2em; }
.ph-grid { display: flex; flex-wrap: wrap; gap: 0.8em; }
.ph-card { display: flex; flex-direction: column; align-items: center; padding: 0.8em;
           border: 1px solid currentColor; border-radius: 8px; width: 175px; }
.ph-token { font-size: 2.2em; font-family: serif; line-height: 1.2; }
.ph-name { font-size: 0.76em; text-align: center; opacity: 0.7; margin-bottom: 0.4em; }
.ph-card img { max-width: 120px; height: auto; margin: 0.3em 0; }
.ph-card audio { width: 155px; margin: 0.3em 0; }
.ph-notes { font-size: 0.73em; margin-top: 0.4em; width: 100%; }
.ph-note { margin: 0.2em 0; }
.ph-note-lang { font-weight: 700; }
</style>
"""

_LIBRARY_GROUPS = [
    ("Oral vowels", ["a", "e", "i", "o", "u", "y", "ø", "œ", "ɛ", "ɔ", "ə"]),
    ("Nasal vowels", ["ɑ̃", "ɛ̃", "ɔ̃", "œ̃"]),
    ("Consonants", ["β", "ɾ", "r", "x", "θ", "ð", "ɣ", "ʁ", "ɲ", "ʃ", "ʒ", "tʃ"]),
]


def _render_library_html() -> str:
    inv = load_phonemes()
    parts = [_LIBRARY_STYLES]
    chart_url = _media_url("phonemes/ipa_vowel_chart.png")
    if chart_url:
        parts.append(
            f'<div class="ph-chart"><img src="{chart_url}" alt="IPA vowel chart"></div>'
        )
    for group_name, tokens in _LIBRARY_GROUPS:
        parts.append(f'<div class="ph-group-heading">{escape(group_name)}</div>')
        parts.append('<div class="ph-grid">')
        for token in tokens:
            phoneme = inv.get(token)
            if phoneme is None:
                continue
            parts.append('<div class="ph-card">')
            parts.append(f'<div class="ph-token">{escape(token)}</div>')
            parts.append(f'<div class="ph-name">{escape(phoneme.name)}</div>')
            parts.extend(_phoneme_media_html(phoneme))
            notes_html = [
                f'<div class="ph-note"><span class="ph-note-lang">{lc}:</span> {escape(note)}</div>'
                for lc in ("es", "fr")
                if (note := phoneme.notes.get(lc, ""))
            ]
            if notes_html:
                parts.append('<div class="ph-notes">' + "".join(notes_html) + "</div>")
            parts.append("</div>")
        parts.append("</div>")
    return "".join(parts)


def _phrase_targets_str(targets: tuple[str, ...]) -> str:
    return " · ".join(targets) if targets else ""


def _build_phrases_tab(tabs: gr.Tabs, reference: gr.Textbox) -> None:
    """Render the Phrases tab as Gradio Button cards grouped by language and category."""
    all_phrases = load_phrases()
    lang_names = {"en": "English", "es": "Spanish", "fr": "French", "cmn": "Mandarin", "ja": "Japanese"}
    for lang_code in ("en", "es", "fr", "cmn", "ja"):
        phrases = all_phrases.get(lang_code, [])
        if not phrases:
            continue
        with gr.Accordion(lang_names.get(lang_code, lang_code), open=(lang_code == "en")):
            by_cat: dict[str, list] = {}
            for ph in phrases:
                by_cat.setdefault(ph.category, []).append(ph)
            for cat in _CATEGORY_ORDER:
                cat_phrases = by_cat.get(cat, [])
                if not cat_phrases:
                    continue
                gr.Markdown(f"**{_CATEGORY_LABELS.get(cat, cat)}**")
                for ph in cat_phrases:
                    label = ph.text
                    if ph.note:
                        label += f"  —  {ph.note}"
                    if ph.targets:
                        label += f"  [{_phrase_targets_str(ph.targets)}]"
                    text = ph.text
                    gr.Button(label, size="sm").click(
                        lambda _tabs, t=text: (t, gr.Tabs(selected=0)),
                        inputs=[tabs],
                        outputs=[reference, tabs],
                    )


_CATEGORY_LABELS = {
    "beginner": "Beginner",
    "pangram": "Pangrams",
    "targeted": "Targeted",
    "tongue-twister": "Tongue-twisters",
}
_CATEGORY_ORDER = ["beginner", "pangram", "targeted", "tongue-twister"]


def _default_phrase(lang: str) -> str:
    phrases = load_phrases().get(lang, [])
    return phrases[0].text if phrases else ""


def _on_lang_change(new_lang: str) -> tuple[gr.Dropdown, str]:
    """Refresh dialect dropdown and pre-fill reference with a default phrase."""
    return gr.Dropdown(choices=_DIALECT_CHOICES[new_lang], value=None), _default_phrase(new_lang)


def _random_phrase(lang: str) -> str:
    import random
    phrases = load_phrases().get(lang, [])
    return random.choice(phrases).text if phrases else ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="vocal-ipa-trainer") as app:
        gr.Markdown(DESCRIPTION)
        with gr.Tabs() as tabs:
            with gr.Tab("Scorer"):
                with gr.Row():
                    lang = gr.Radio(
                        choices=["en", "es", "fr", "cmn", "ja"], value="en", label="Language"
                    )
                    dialect = gr.Dropdown(
                        choices=_DIALECT_CHOICES["en"],
                        value=None,
                        label="Dialect",
                        allow_custom_value=False,
                    )
                with gr.Row():
                    reference = gr.Textbox(
                        label="Reference text (optional)",
                        value=_default_phrase("en"),
                        lines=1,
                        scale=8,
                    )
                    random_btn = gr.Button("🎲", scale=1, min_width=48)
                    browse_btn = gr.Button("Browse →", scale=1, min_width=80)
                audio = gr.Audio(
                    sources=["microphone", "upload"], type="filepath", label="Audio"
                )
                btn = gr.Button("Transcribe / Score", variant="primary")
                scored = gr.HTML(label="Scoring", visible=True)
                ipa = gr.Textbox(label="IPA", lines=2, show_copy_button=True)
                raw = gr.Textbox(label="Raw model output", lines=2, show_copy_button=True)
                timing = gr.Markdown()
                lang.change(_on_lang_change, inputs=[lang], outputs=[dialect, reference])
                random_btn.click(_random_phrase, inputs=[lang], outputs=[reference])
                browse_btn.click(lambda: gr.Tabs(selected=2), outputs=[tabs])
                btn.click(
                    _run,
                    inputs=[audio, reference, lang, dialect],
                    outputs=[ipa, raw, timing, scored],
                )
            with gr.Tab("Phoneme Library"):
                gr.HTML(_render_library_html())
            with gr.Tab("Phrases"):
                _build_phrases_tab(tabs, reference)
    return app


def main() -> None:
    # Allow Gradio to serve phoneme reference media (images, audio) shipped
    # in the package's data/ directory. Required for the miss-comparison
    # cards once 4b-11 populates image/audio fields.
    build_app().launch(allowed_paths=[str(_DATA_DIR)])


if __name__ == "__main__":
    main()
