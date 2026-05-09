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


def test_load_phonemes_contains_mandarin_inventory() -> None:
    inv = load_phonemes()
    # Phase 5a adds Mandarin segmental tokens. Sample a few across categories
    # so a missed YAML edit fails fast.
    for token in ("ts.", "ts.h", "tɕh", "ɕ", "ʐ", "χ", "ŋ"):
        assert token in inv, f"missing Mandarin consonant: {token!r}"
    for token in ("ɑ1", "ɑ2", "ɑ5", "iɛ1", "i.5", "onɡ5", "ərɜ"):
        assert token in inv, f"missing Mandarin vowel/tone token: {token!r}"
    # Notes must include Mandarin commentary.
    assert "cmn" in inv["ts."].notes


def test_load_phonemes_contains_japanese_inventory() -> None:
    inv = load_phonemes()
    # Phase 5b adds Japanese segmental tokens (after pyopenjtalk → IPA mapping
    # and consecutive-same-vowel collapse).
    for token in ("kʲ", "pʲ", "bʲ", "mʲ", "rʲ"):
        assert token in inv, f"missing Japanese palatalized: {token!r}"
    for token in ("dʑ", "ɕ", "ç", "ɸ", "ɴ", "ʔ"):
        assert token in inv, f"missing Japanese consonant: {token!r}"
    for token in ("aː", "iː", "uː"):
        assert token in inv, f"missing Japanese long vowel: {token!r}"
    assert "ja" in inv["ɴ"].notes


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


# -- inventory coverage ------------------------------------------------------


def test_phoneme_inventory_covers_observed_fixture_tokens() -> None:
    """Every IPA token observed across the fixture transcribe goldens must
    have a Phoneme entry. Catches inventory gaps at test time before they
    become silent runtime nulls. Covers every language present in the
    manifest (es, fr, cmn as of Phase 5a)."""
    import json
    from pathlib import Path

    fixtures = Path(__file__).parent / "data" / "fixtures"
    manifest_path = fixtures / "manifest.json"
    if not manifest_path.exists():
        return

    inv = load_phonemes()
    observed: set[str] = set()
    for entry in json.loads(manifest_path.read_text()):
        golden = fixtures / f"{entry['id']}.golden.ipa"
        if golden.exists():
            observed.update(golden.read_text().split())

    missing = sorted(observed - set(inv))
    assert not missing, (
        f"Phoneme inventory is missing {len(missing)} token(s) observed in "
        f"transcribe goldens: {missing}. Add entries to "
        f"src/vocal_ipa/data/phonemes.yaml."
    )
