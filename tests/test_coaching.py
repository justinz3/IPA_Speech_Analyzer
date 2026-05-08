"""Tier A tests for the coaching library: phoneme inventory + override lookup.

Reads the real YAML files shipped in the package; no mocks. The skeleton
content is small enough that these tests can hold concrete assertions against
named entries without becoming brittle.
"""

from __future__ import annotations

from vocal_ipa.coaching import (
    MissReference,
    Phoneme,
    Tip,
    load_overrides,
    load_phonemes,
    lookup_miss,
)

# -- loaders ------------------------------------------------------------------


def test_load_phonemes_returns_typed_dict() -> None:
    inv = load_phonemes()
    assert len(inv) >= 1
    for token, phoneme in inv.items():
        assert isinstance(phoneme, Phoneme)
        assert phoneme.token == token
        assert phoneme.name  # non-empty
        # Optional fields are either None or strings; notes is a dict.
        assert phoneme.image is None or isinstance(phoneme.image, str)
        assert phoneme.audio is None or isinstance(phoneme.audio, str)
        assert isinstance(phoneme.notes, dict)


def test_load_phonemes_contains_skeleton_entries() -> None:
    inv = load_phonemes()
    # The 4b-5 skeleton must include β and y so the override schema has
    # a known expected token to point at.
    assert "β" in inv
    assert inv["β"].name == "voiced bilabial fricative"
    assert "y" in inv
    assert "rounded" in inv["y"].name


def test_load_overrides_returns_typed_list() -> None:
    overrides = load_overrides()
    for entry in overrides:
        assert isinstance(entry, Tip)
        assert entry.lang in {"es", "fr"}
        assert entry.title and entry.tip


# -- lookup_miss --------------------------------------------------------------


def test_lookup_miss_returns_none_for_unknown_expected() -> None:
    # No phoneme entry for fake token "xx" → graceful None.
    assert lookup_miss("es", "xx", "yy") is None


def test_lookup_miss_with_known_expected_returns_miss_reference() -> None:
    result = lookup_miss("es", "β", "v")
    assert isinstance(result, MissReference)
    assert result.expected.token == "β"
    assert result.produced is not None
    assert result.produced.token == "v"


def test_lookup_miss_with_blank_produced_returns_none_produced() -> None:
    # "∅" is the BLANK_SURFACE; never has an inventory entry.
    result = lookup_miss("es", "β", "∅")
    assert result is not None
    assert result.expected.token == "β"
    assert result.produced is None


def test_lookup_miss_finds_matching_override_tip() -> None:
    result = lookup_miss("es", "β", "v")
    assert result is not None
    assert result.tip is not None
    assert "v" in result.tip.title  # "Don't use English [v]"


def test_lookup_miss_returns_no_tip_when_none_matches() -> None:
    # β is in the inventory, but no override exists for (es, β, ∅).
    result = lookup_miss("es", "β", "∅")
    assert result is not None
    assert result.tip is None


def test_lookup_miss_french_override_works() -> None:
    result = lookup_miss("fr", "y", "i")
    assert result is not None
    assert result.tip is not None
    assert "round" in result.tip.tip.lower()
