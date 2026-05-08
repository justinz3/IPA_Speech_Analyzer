"""Coaching: per-phoneme reference media + per-error-pair override tips.

Two YAML data files drive the coaching surface:

- ``data/phonemes.yaml`` is the foundation: one entry per IPA token in the
  es+fr inventory, each with ``name`` (e.g., "voiced bilabial fricative")
  plus optional ``image`` / ``audio`` / ``video`` paths or URLs and per-
  language ``notes``. Mechanically populated; covers every miss the model
  can produce on supported languages.

- ``data/correction_overrides.yaml`` is the additive layer: a sparse list
  of (lang, expected, produced) entries with a ``title`` + ``tip`` prose.
  Hand-curated for the high-value error pairs surfaced in the failure-modes
  docs. Empty overrides are fine — the comparison view (expected vs
  produced reference media) still works.

For every miss in a ScoreResult, ``lookup_miss(lang, expected, produced)``
returns a ``MissReference`` carrying both phoneme entries plus the optional
override Tip. Returns ``None`` when the expected token isn't in the
inventory yet (graceful degradation; a separate test asserts inventory
coverage).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from importlib.resources import files

import yaml


@dataclass(frozen=True)
class Phoneme:
    token: str
    name: str
    image: str | None = None
    audio: str | None = None
    video: str | None = None
    notes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Tip:
    lang: str
    expected: str
    produced: str
    title: str
    tip: str


@dataclass(frozen=True)
class MissReference:
    expected: Phoneme            # always populated when the function returns non-None
    produced: Phoneme | None     # None for "∅" or out-of-inventory tokens
    tip: Tip | None              # populated only when an override matches


def _read_data(filename: str) -> str:
    return (files("vocal_ipa") / "data" / filename).read_text(encoding="utf-8")


@functools.cache
def load_phonemes() -> dict[str, Phoneme]:
    raw = yaml.safe_load(_read_data("phonemes.yaml")) or {}
    return {
        token: Phoneme(
            token=token,
            name=entry["name"],
            image=entry.get("image"),
            audio=entry.get("audio"),
            video=entry.get("video"),
            notes=dict(entry.get("notes") or {}),
        )
        for token, entry in raw.items()
    }


@functools.cache
def load_overrides() -> list[Tip]:
    raw = yaml.safe_load(_read_data("correction_overrides.yaml")) or []
    return [
        Tip(
            lang=entry["lang"],
            expected=entry["expected"],
            produced=entry["produced"],
            title=entry["title"],
            tip=entry["tip"],
        )
        for entry in raw
    ]


def lookup_miss(lang: str, expected: str, produced: str) -> MissReference | None:
    """Return reference media (and optional tip) for a (expected, produced) miss.

    Returns ``None`` when the expected token isn't in the inventory yet.
    Callers should treat None as "no coaching available, render the miss
    in the score table as usual."
    """
    phonemes = load_phonemes()
    expected_phoneme = phonemes.get(expected)
    if expected_phoneme is None:
        return None
    produced_phoneme = phonemes.get(produced)  # None if "∅" or out of inventory
    return MissReference(
        expected=expected_phoneme,
        produced=produced_phoneme,
        tip=_find_tip(lang, expected, produced),
    )


def _find_tip(lang: str, expected: str, produced: str) -> Tip | None:
    for tip in load_overrides():
        if tip.lang == lang and tip.expected == expected and tip.produced == produced:
            return tip
    return None
