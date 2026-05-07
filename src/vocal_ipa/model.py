"""Wav2Vec2 phoneme model loader."""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC

DEFAULT_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def load(model_id: str = DEFAULT_MODEL, device: str = "cpu") -> tuple[AutoProcessor, Wav2Vec2ForCTC]:
    """Load processor + model. Cached per (model_id, device) so repeated calls
    in the same process (CLI re-invocations from tests, web app) don't re-download."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
    return processor, model
