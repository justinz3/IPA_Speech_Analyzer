"""Pronounce CLI: audio file -> IPA."""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path

import typer

from .model import DEFAULT_MODEL
from .pipeline import transcribe

app = typer.Typer(
    add_completion=False,
    help="Audio -> IPA pronunciation transcription. Phase 1: Spanish only.",
    no_args_is_help=True,
)


class Lang(str, Enum):
    es = "es"


class Fmt(str, Enum):
    text = "text"
    json = "json"


class Device(str, Enum):
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
    lang: Lang = typer.Option(Lang.es, "--lang", help="Language code (Phase 1: only 'es')."),
    fmt: Fmt = typer.Option(Fmt.text, "--format", help="Output format."),
    device: Device = typer.Option(Device.auto, "--device", help="Inference device."),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Hugging Face model id."),
    raw: bool = typer.Option(
        False, "--raw", help="Emit pre-postprocessing model labels instead of cleaned IPA."
    ),
) -> None:
    """Transcribe AUDIO into IPA."""
    if sys.stderr.isatty() and not _model_is_cached(model):
        typer.echo(
            f"Loading {model} (~1GB on first run; cached to ~/.cache/huggingface/)...",
            err=True,
        )

    result = transcribe(audio, lang=lang.value, device=device.value, model_id=model)

    if fmt is Fmt.text:
        typer.echo(result.raw_phonemes if raw else result.ipa)
    else:
        typer.echo(json.dumps(result.to_dict(include_raw=raw), ensure_ascii=False))


if __name__ == "__main__":
    app()
