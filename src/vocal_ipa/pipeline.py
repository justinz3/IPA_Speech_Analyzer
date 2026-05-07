"""End-to-end audio -> IPA pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .audio import TARGET_SR, ensure_16k, load_audio
from .model import DEFAULT_MODEL, load, resolve_device

SUPPORTED_LANGUAGES = frozenset({"es", "fr"})


@dataclass
class Transcription:
    ipa: str
    raw_phonemes: str
    language: str
    model: str
    audio_seconds: float
    model_load_seconds: float
    inference_seconds: float

    def to_dict(self, include_raw: bool = False) -> dict:
        d = asdict(self)
        if not include_raw:
            d.pop("raw_phonemes")
        return d


def transcribe(
    audio_path: str | Path,
    lang: str = "es",
    device: str = "auto",
    model_id: str = DEFAULT_MODEL,
) -> Transcription:
    transcription, _, _ = _run_model(audio_path, lang, device, model_id)
    return transcription


def _run_model(
    audio_path: str | Path,
    lang: str = "es",
    device: str = "auto",
    model_id: str = DEFAULT_MODEL,
) -> tuple[Transcription, torch.Tensor, int]:
    """Run the model once; return Transcription plus CTC log-probs and blank id.

    transcribe() discards the extras; score() (Phase 2) uses them for forced
    alignment, so both paths share a single forward pass.
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language {lang!r}; supported: {sorted(SUPPORTED_LANGUAGES)}. "
            "See pronunciation_app_roadmap.md."
        )

    samples, sr = load_audio(Path(audio_path))
    samples = ensure_16k(samples, sr)
    audio_seconds = len(samples) / TARGET_SR

    dev = resolve_device(device)
    t0 = time.perf_counter()
    processor, model = load(model_id, dev)
    model_load_seconds = time.perf_counter() - t0

    inputs = processor(samples, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    t1 = time.perf_counter()
    with torch.inference_mode():
        logits = model(**inputs).logits
    pred_ids = logits.argmax(dim=-1)
    raw = processor.batch_decode(pred_ids)[0]
    log_probs = torch.log_softmax(logits[0], dim=-1).cpu()
    inference_seconds = time.perf_counter() - t1

    transcription = Transcription(
        ipa=postprocess(raw),
        raw_phonemes=raw,
        language=lang,
        model=model_id,
        audio_seconds=audio_seconds,
        model_load_seconds=model_load_seconds,
        inference_seconds=inference_seconds,
    )
    return transcription, log_probs, processor.tokenizer.pad_token_id


def postprocess(raw: str) -> str:
    """Phase 1: collapse whitespace only.

    The espeak-ng IPA inventory and the wav2vec2 model's emitted labels do not
    match exactly (roadmap risk). Real mapping work happens after observing
    failure modes on real Spanish audio (Phase 1 close).
    """
    return " ".join(raw.split())
