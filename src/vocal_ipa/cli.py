"""Pronounce CLI: audio file -> IPA, with optional per-phoneme scoring."""

from __future__ import annotations

import json
import sys
from enum import StrEnum
from pathlib import Path

import typer

from .model import DEFAULT_MODEL
from .pipeline import transcribe
from .reference import resolve_locale
from .score import ScoreResult, score

app = typer.Typer(
    add_completion=False,
    help="Audio -> IPA pronunciation transcription. Spanish, French, and Mandarin.",
    no_args_is_help=True,
)


class Lang(StrEnum):
    """Language code accepted on `--lang`.

    Bare codes (es, fr, cmn) use the language's default dialect. Composite
    codes (es-es, es-419, es-latam, fr-fr, cmn-cn) pin a dialect that must
    agree with `--dialect` if both are given. `zh` is a common alias for
    `cmn` (ISO 639-1 macro code).
    """

    es = "es"
    es_es = "es-es"
    es_419 = "es-419"
    es_latam = "es-latam"
    fr = "fr"
    fr_fr = "fr-fr"
    cmn = "cmn"
    cmn_cn = "cmn-cn"
    zh = "zh"


class Fmt(StrEnum):
    text = "text"
    json = "json"


class Device(StrEnum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"


def _model_is_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return True  # assume yes; we'll see download progress anyway
    return try_to_load_from_cache(model_id, "config.json") is not None


@app.command()
def main(
    audio: Path = typer.Argument(
        ..., exists=True, readable=True, help="Path to a WAV/FLAC/OGG audio file."
    ),
    lang: Lang = typer.Option(
        Lang.es,
        "--lang",
        help=(
            "Language or composite locale code "
            "(es, es-es, es-419, es-latam, fr, fr-fr, cmn, cmn-cn, zh)."
        ),
    ),
    dialect: str = typer.Option(
        None,
        "--dialect",
        help=(
            "Dialect override; codes (es-es, es-419, fr-fr) or aliases "
            "(castilian, latam, parisian). Optional; defaults match --lang."
        ),
    ),
    fmt: Fmt = typer.Option(Fmt.text, "--format", help="Output format."),
    device: Device = typer.Option(Device.auto, "--device", help="Inference device."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Hugging Face model id."),
    raw: bool = typer.Option(
        False, "--raw", help="Emit pre-postprocessing model labels instead of cleaned IPA."
    ),
    reference: str = typer.Option(
        None,
        "--reference",
        help=(
            "Known sentence the AUDIO is meant to read. Triggers per-phoneme "
            "scoring against the reference IPA instead of free transcription."
        ),
    ),
) -> None:
    """Transcribe AUDIO into IPA, or score it against --reference text."""
    if raw and reference is not None:
        raise typer.BadParameter(
            "--raw cannot combine with --reference (raw is for free transcription)"
        )

    try:
        locale = resolve_locale(lang.value, dialect)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    if sys.stderr.isatty() and not _model_is_cached(model):
        typer.echo(
            f"Loading {model} (~1GB on first run; cached to ~/.cache/huggingface/)...",
            err=True,
        )

    if reference is None:
        result = transcribe(
            audio, lang=locale.lang, dialect=locale.dialect, device=device.value, model_id=model
        )
        if fmt is Fmt.text:
            typer.echo(result.raw_phonemes if raw else result.ipa)
        else:
            typer.echo(json.dumps(result.to_dict(include_raw=raw), ensure_ascii=False))
        return

    score_result = score(
        audio,
        reference,
        lang=locale.lang,
        dialect=locale.dialect,
        device=device.value,
        model_id=model,
    )
    if fmt is Fmt.text:
        typer.echo(_render_score_table(score_result))
    else:
        typer.echo(json.dumps(score_result.to_dict(), ensure_ascii=False))


def _render_score_table(result: ScoreResult) -> str:
    lines = [f"language: {result.language} ({result.dialect})", ""]
    lines.append(f"{'expected':<10}{'produced':<10}{'start':<8}{'end':<8}{'ok'}")
    for p in result.phonemes:
        mark = "✓" if p.ok else "✗"
        lines.append(f"{p.expected:<10}{p.produced:<10}{p.start_s:<8.2f}{p.end_s:<8.2f}{mark}")
    wrong = sum(1 for p in result.phonemes if not p.ok)
    total = len(result.phonemes)
    lines.append("")
    lines.append(f"PER: {result.per:.3f}  ({wrong}/{total} phonemes wrong)")
    if result.dropped_reference_count:
        lines.append(
            f"(dropped {result.dropped_reference_count} reference char(s) not in model vocab)"
        )
    misses_block = _render_misses(result)
    if misses_block:
        lines.append("")
        lines.append(misses_block)
    return "\n".join(lines)


def _render_misses(result: ScoreResult) -> str:
    """Render the per-unique-miss comparison block. Empty string if no
    miss has a coaching MissReference attached."""
    if not result.miss_references:
        return ""
    blocks = [f"Misses ({len(result.miss_references)} unique):", ""]
    for ref in result.miss_references:
        blocks.append(_render_one_miss(ref))
    return "\n".join(blocks)


def _render_one_miss(ref) -> str:
    lines = []
    lines.append(f"  expected {ref.expected.token}  {ref.expected.name}")
    for label, value in (("image", ref.expected.image), ("audio", ref.expected.audio)):
        if value:
            lines.append(f"            {label}: {value}")
    if ref.produced is not None:
        lines.append(f"  produced {ref.produced.token}  {ref.produced.name}")
        for label, value in (("image", ref.produced.image), ("audio", ref.produced.audio)):
            if value:
                lines.append(f"            {label}: {value}")
    if ref.tip is not None:
        lines.append(f"  tip: {ref.tip.title}")
        for tip_line in ref.tip.tip.rstrip().splitlines():
            lines.append(f"       {tip_line}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    app()
