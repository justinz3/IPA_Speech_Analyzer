# Spanish — observed failure modes

Notes from dogfooding `pronounce` against `--lang es` audio. Distinguishes:

- **Your audio** — real L2 patterns the model is faithfully capturing.
- **Reference dialect** — Castilian vs Latin American mismatch.
- **Model / representation artifact** — quirks that don't reflect either
  your audio or the reference.

When in doubt: assume your audio. The model is mostly faithful, and L2
patterns *look* like errors because they really are diverging from native
phonetics.

## L2 audio patterns (your pronunciation)

- **`β → v` in intervocalic /b/, /v/ contexts** (`favor`, `vamos`).
  Reaching for the labiodental English `[v]`; Spanish wants the bilabial
  fricative `[β]`. Single most actionable thing to drill.
- **`ɾ → r` (single tap → trill).** Over-trilling intervocalic `r` (`por`,
  `quisiera`). The Spanish single `r` is a flap — same articulation as
  the American English "tt" in "butter". Trills are reserved for `rr`
  and word-initial `r`.
- **`tʃ → dʒ` in intervocalic positions** (`leche`). Voicing the "ch"
  between vowels. Spanish keeps `ch` voiceless everywhere.
- **`x → h` in `j` / soft-`g` words** (`jardín`, `mujer`). Substituting
  English /h/ for the back-of-throat Spanish /x/. Should feel like
  clearing your throat.
- **Diphthong splitting in `ie` / `ue`** (`quisiera` → `ki-si-e-ra`
  instead of `ki-sje-ra`). Reading vowel pairs as separate syllables
  rather than glide-fusing.
- **Vowel diphthongization** (`o → oɪ`, `a → aɪ`). English vowels glide;
  Spanish vowels stay pure.
- **`ʒ` / `ʃ` for "ll".** Landing in the postalveolar zone when reaching
  for Spanish "ll". Dialectally there *is* a Río de la Plata realization
  with `/ʒ ʃ/`, but for a US English speaker this is much more likely
  to be your audio than the model picking up a dialect bias.

## Reference dialect mismatch

- **`θ → s` (or `ts`) for `z` and soft `c`** (`manzana`, `cinco`).
  Phonemizer defaults to Castilian Spanish (espeak `es` / `es-es`), which
  uses `/θ/` (English "th" in "thin"). Latin American Spanish uses `/s/`.
  If your target is Latin American, the reference will mark your audio
  "wrong" on every `z` and soft `c`. **Phase 4 should expose dialect
  selection** — same pattern would apply for English regional accents
  if/when English enters the pipeline.

## Reference orthography

- **Missing accents change phonemic interpretation.** Typing `dia` instead
  of `día` makes phonemizer treat the `i` as a non-syllabic glide
  (`dja` instead of `dia`), and the model's vocab tokenizes that as
  the 2-char token `ja`. Use accents in your reference text — the
  difference is load-bearing.

## Model / representation artifacts

See the README's "Known model failure modes" section for the canonical
list (length marks, intervocalic lenition flicker, `/t/` drop in clusters,
coda `/l/` drop, hesitation captured as repeated vowels). Nothing new in
Phase 2/3 dogfooding beyond those — Spanish-side scoring is essentially
just surfacing your audio against the Castilian reference.

## Phrase set used for dogfooding

```
Buenos días, mucho gusto.
¿Cómo estás? Me llamo Justin.
Quisiera un café con leche, por favor.
El niño come manzanas en el jardín de su mamá.
```
