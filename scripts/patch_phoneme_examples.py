"""Add `examples:` field to phonemes.yaml entries from Wikipedia IPA pages.

Run with: uv run python scripts/patch_phoneme_examples.py
"""
from __future__ import annotations

import re
from pathlib import Path

YAML_PATH = Path(__file__).parent.parent / "src/vocal_ipa/data/phonemes.yaml"

# token -> {lang_code: "example words"}
# Sourced from Wikipedia Help:IPA/* pages for en/es/fr/cmn/ja.
EXAMPLES: dict[str, dict[str, str]] = {
    # ── Oral monophthongs ──────────────────────────────────────────────────
    "a": {
        "es": "p**a**so, c**a**sa",
        "fr": "p**a**tte, l**à**, femme",
        "ja": "**あ**る (aru)",
    },
    "e": {
        "es": "p**e**so, m**e**sa",
        "fr": "cl**é**, ch**e**z, all**e**r",
        "ja": "**え**き (eki)",
    },
    "i": {
        "es": "p**i**so, s**i**, **y**",
        "fr": "s**i**, **î**le, rég**ie**",
        "en": "fl**ee**ce, s**ee**, b**e**",
        "ja": "**い**る (iru)",
    },
    "o": {
        "es": "p**o**so, c**o**sa",
        "fr": "s**au**t, bure**au**, ch**o**se",
        "ja": "**お**に (oni)",
    },
    "u": {
        "es": "p**u**so, l**u**na",
        "fr": "c**ou**p, r**ou**e",
        "en": "g**oo**se, bl**ue**",
        "ja": "**う**なぎ (unagi)",
    },
    "y": {
        "fr": "t**u**, s**û**r, r**ue**",
    },
    "ø": {
        "fr": "p**eu**, c**eu**x, qu**eue**",
    },
    "œ": {
        "fr": "s**œu**r, p**eu**r, j**eu**ne",
    },
    "ɛ": {
        "en": "dr**e**ss, b**e**d, s**ai**d, fr**ie**nd",
        "fr": "b**ai**e, f**ai**te, cr**è**me, m**è**re",
    },
    "ɔ": {
        "fr": "s**o**rt, p**o**mme, h**o**mme",
    },
    "ə": {
        "fr": "r**e**poser, j**e**, p**e**tit",
        "en": "comm**a**, ab**o**ut, th**e**",
    },
    "ɑ": {
        "fr": "p**â**te, gl**as**",
    },
    # ── English vowels ─────────────────────────────────────────────────────
    "ɪ": {
        "en": "k**i**t, b**i**g, s**i**ng",
    },
    "æ": {
        "en": "tr**a**p, b**a**g, s**a**ng",
    },
    "ʌ": {
        "en": "str**u**t, s**u**ng, bl**oo**d",
    },
    "ʊ": {
        "en": "f**oo**t, h**oo**k, p**u**t",
    },
    "ɜː": {
        "en": "n**ur**se, b**ir**d, h**ear**d",
    },
    "ɑː": {
        "en": "p**al**m, br**a**, f**a**ther",
    },
    "eɪ": {
        "en": "f**a**ce, v**a**gue, d**a**y",
    },
    "oʊ": {
        "en": "g**oa**t, g**o**, h**o**me",
    },
    "aɪ": {
        "en": "pr**i**ce, p**ie**, fl**y**",
    },
    "aʊ": {
        "en": "m**ou**th, h**ow**, n**ow**",
    },
    "ɔɪ": {
        "en": "ch**oi**ce, b**oy**, c**oi**n",
    },
    "iː": {
        "en": "fl**ee**ce, s**ee**, b**e**",
    },
    "uː": {
        "en": "g**oo**se, bl**ue**, n**ew**",
    },
    "ɔː": {
        "en": "th**ough**t, c**augh**t, l**aw**",
    },
    "ɒ": {
        "en": "l**o**t, b**o**ther, c**o**t",
    },
    # ── Nasal vowels ───────────────────────────────────────────────────────
    "ɑ̃": {
        "fr": "s**an**s, ch**am**p, v**en**t, t**em**ps",
    },
    "ɛ̃": {
        "fr": "v**in**, imp**ai**r, p**ain**, pl**ein**",
    },
    "ɔ̃": {
        "fr": "s**on**, n**om**, b**on**",
    },
    "œ̃": {
        "fr": "**un**, parf**um**",
    },
    # ── Consonants ─────────────────────────────────────────────────────────
    "β": {
        "es": "be**b**é (intervocalic), fút**b**ol",
    },
    "ɾ": {
        "es": "ca**r**o, pe**r**o, pe**r**a",
        "ja": "**ろ**く (roku), そ**ら** (sora)",
    },
    "r": {
        "es": "**r**ío, ca**rr**o, hon**r**a",
    },
    "x": {
        "es": "**j**arra, **g**ente, Mé**x**ico",
        "cmn": "**火** (huǒ), **好** (hǎo)",
    },
    "θ": {
        "es": "c**e**ra, ca**z**a (Castilian)",
        "en": "**th**igh, pa**th**, **th**ing",
    },
    "ð": {
        "es": "de**d**o (intervocalic), ca**d**a, ar**d**e",
        "en": "**th**y, brea**th**e, fa**th**er",
    },
    "ɣ": {
        "es": "gal**g**o (intervocalic), si**g**no, á**g**uila",
    },
    "ʁ": {
        "fr": "**r**ega**r**der, **r**ue, nôt**r**e",
    },
    "ɲ": {
        "es": "**ñ**u, ca**ñ**a, có**ny**uge",
        "fr": "ga**gn**er, champa**gn**e",
        "ja": "**に**わ (niwa), こん**にゃ**く (konnyaku)",
    },
    "ʃ": {
        "es": "**sh**ow (loanword), Frei**x**enet",
        "fr": "**ch**ance, t**ch**èque",
        "en": "**sh**y, ca**sh**, emo**ti**on",
    },
    "ʒ": {
        "fr": "**j**amais, vi**s**age",
        "en": "plea**s**ure, bei**ge**, mea**s**ure",
    },
    "tʃ": {
        "es": "**ch**ico, mu**ch**o",
        "fr": "t**ch**èque",
        "en": "**Ch**ina, cat**ch**, **ch**urch",
    },
    "ŋ": {
        "es": "te**ng**o",
        "fr": "campi**ng**, fu**nk**",
        "en": "sa**ng**, si**nk**, si**ng**er",
        "cmn": "**江** (jiāng), **明** (míng)",
        "ja": "り**ん**ご (ringo), な**ん**きょく (nankyoku)",
    },
    "ɥ": {
        "fr": "**h**uit, p**ui**ts",
        "cmn": "**月** (yuè)",
    },
    # ── Shared consonants with multi-language examples ─────────────────────
    "n": {
        "es": "**n**i, ca**n**a, si**n**",
        "fr": "**n**ous, bo**nn**e",
        "en": "**n**igh, ca**n**",
        "ja": "**な**っとう (natto)",
    },
    "m": {
        "es": "**m**eta, ca**m**a",
        "fr": "**m**ême",
        "en": "**m**y, ca**m**",
        "ja": "**み**かん (mikan)",
    },
    "l": {
        "es": "**l**una, ha**l**a",
        "fr": "**l**aisser, seu**l**",
        "en": "**l**ie, ga**l**",
    },
    "s": {
        "es": "**s**aco, ca**s**a, e**s**tá",
        "fr": "**s**ans, ça, a**ss**ez",
        "en": "**s**igh, ma**ss**",
        "ja": "**す**る (suru)",
    },
    "f": {
        "es": "**f**aro",
        "fr": "**f**aire, vi**f**",
        "en": "**f**ind, lea**f**",
    },
    "k": {
        "es": "**c**aso, **qu**e, **k**ilo",
        "fr": "co**r**ps, ave**c**, **qu**and",
        "en": "**k**ind, s**k**y, cra**ck**",
        "cmn": "**干** (gān)",
        "ja": "**く**る (kuru)",
    },
    "p": {
        "es": "**p**ato, lu**p**a",
        "fr": "**p**ère, grou**p**e",
        "en": "**p**ie, s**p**y, ca**p**",
        "cmn": "**帮** (bāng)",
        "ja": "**パ**ン (pan)",
    },
    "t": {
        "es": "**t**amiz",
        "fr": "**t**out, **th**é",
        "en": "**t**ie, s**t**y, ca**t**",
        "cmn": "**端** (duān)",
        "ja": "**た**べる (taberu)",
    },
    "j": {
        "es": "**Vi**ena, re**y**",
        "fr": "fie**f**, pa**y**er, fi**ll**e",
        "ja": "**や**くしゃ (yakusha), **ゆ**ず (yuzu)",
    },
    "w": {
        "es": "H**u**ila, a**u**to, **w**eb",
        "fr": "**ou**i, lo**i**",
        "ja": "**わ**さび (wasabi)",
    },
    "ɕ": {
        "cmn": "**晓** (xiǎo)",
        "ja": "**し**た (shita), いっ**しょ**う (isshou)",
    },
    "ʐ": {
        "cmn": "**日** (rì)",
    },
    "ʂ": {
        "cmn": "**矢** (shǐ), **时** (shí)",
    },
    "ʈʂ": {
        "cmn": "**之** (zhī), **中** (zhōng)",
    },
    "χ": {
        "cmn": "**好** (hǎo — variant of x)",
    },
    # Mandarin aspirated stops/affricates
    "ts.": {
        "cmn": "**子** (zǐ), **字** (zì)",
    },
    "ts.h": {
        "cmn": "**此** (cǐ), **次** (cì)",
    },
    "tɕh": {
        "cmn": "**去** (qù), **请** (qǐng)",
    },
    # Japanese-specific consonants
    "ç": {
        "ja": "**ひ**と (hito), **ひょ**う (hyou)",
    },
    "ɸ": {
        "ja": "**ふ**じ (fuji)",
    },
    "ɴ": {
        "ja": "に**ほ**ん (nihon — final N)",
    },
    "dʑ": {
        "ja": "**じょ**じょ (jojo), かん**じゃ** (kanja)",
    },
    "kʲ": {
        "ja": "**きょ**うかい (kyoukai)",
    },
    "pʲ": {
        "ja": "はっ**ぴょ**う (happyou)",
    },
    "bʲ": {
        "ja": "**びょ**うき (byouki)",
    },
    "ɾʲ": {
        "ja": "**りょ**うり (ryouri)",
    },
    "mʲ": {
        "ja": "**みゃ**く (myaku)",
    },
}


def _build_examples_block(examples: dict[str, str]) -> list[str]:
    lines = ["  examples:"]
    for lang in sorted(examples):
        val = examples[lang].replace('"', '\\"')
        lines.append(f'    {lang}: "{val}"')
    return lines


def main() -> None:
    content = YAML_PATH.read_text()
    lines = content.splitlines()

    # Find all top-level keys: lines where col-0 is non-whitespace, non-comment,
    # followed by ':'
    TOP = re.compile(r'^(\S[^:]*?):\s*$|^(\S[^:]*?):(?=\s)')
    entries: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        m = TOP.match(line)
        if m and not line.startswith("#"):
            key = (m.group(1) or m.group(2)).strip()
            entries.append((idx, key))

    # Process in reverse so insertions don't shift earlier indices.
    result = list(lines)
    for i in range(len(entries) - 1, -1, -1):
        start, key = entries[i]
        if key not in EXAMPLES:
            continue

        # Already has examples? Skip.
        block_end = entries[i + 1][0] if i + 1 < len(entries) else len(result)
        block = result[start:block_end]
        if any(l.startswith("  examples:") for l in block):
            continue

        # Find last non-blank line in this block
        insert_after = start
        for j in range(block_end - 1, start, -1):
            if result[j].strip():
                insert_after = j
                break

        ex_lines = _build_examples_block(EXAMPLES[key])
        for k, ex_line in enumerate(ex_lines):
            result.insert(insert_after + 1 + k, ex_line)

    YAML_PATH.write_text("\n".join(result) + "\n")
    print(f"Patched {sum(1 for _, k in entries if k in EXAMPLES)} entries.")


if __name__ == "__main__":
    main()
