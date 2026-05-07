# French — observed failure modes

Notes from dogfooding `pronounce` against `--lang fr` audio. Same
attribution categories as `spanish_failure_modes.md`.

The PER on French audio is roughly **4× higher than on Spanish** for the
same speaker — English speakers are missing entire French phonemes
outright. This is exactly the gap the project is meant to close.

## L2 audio patterns (your pronunciation)

### #1: Front rounded vowels `/y/`, `/ø/`, `/œ/`

The single biggest issue. English has no front rounded vowels at all, so
the tongue defaults to the unrounded equivalents.

- `y → i` in `tu`, `huit`: not rounding lips on the `/i/` sound.
- `ø → eː` in `deux`, `ø → ʉ` in `bleues`: partial rounding, wrong
  position. (`/ʉ/` is *central* rounded; `/ø/` is *front* rounded.)
- `œ → ø` in `fleurs`: right vowel family, wrong openness.

**Practice cue:** hold your lips in a strict whistle/kiss shape, then
say "ee" → that's `/y/`, or "ay" → that's `/ø/`. The lip shape carries.

### #2: Uvular `/ʁ/`

- `ʁ → ∅` in `aujourd'hui`, `fleurs`. Substituting English alveolar /r/
  for French uvular /ʁ/. The two articulations are so different the
  model often hears nothing where the /ʁ/ should be. Same back-of-throat
  zone as German "ach" or Spanish `/x/`.

### #3: Nasal vowels

- `ɑ̃ → ɑ` in `comment`: dropping nasalization entirely.
- `ɛ̃ → ɑ̃` in `pains`: substituting an easier-for-English back nasal
  for the harder front nasal. `/ɛ̃/` has no English analogue; `/ɑ̃/`
  is closer to "haunt" without the /t/ and so feels safer.

### #4: Affrication before front rounded vowels

- `t → tʃ` in `tu`, `d → dʒ` in `deux`. Hitting /t/+/y/ or /d/+/ø/
  English-style produces an affricate (think "tube" → "choob"). Brace
  tongue tip forward when you see `tu` / `du` / `tu`-style clusters.

### Lower-priority patterns

- **Schwa drift** `ə → e` (in `je`, `petits`). English schwa is
  centralized; French schwa is slightly more rounded/back. Subtle.
- **Liaison missing** `z → ∅` in `vous-aujourd'hui`. Reasonable for a
  learner. Phonemizer applies liaisons inconsistently anyway, so this
  is a soft signal at best.
- **Vowel diphthongization** `ɛ → eɪ` in `plaît`. English vowels glide;
  French vowels stay pure. Same anglicization as Spanish.

## Model / representation artifacts

- **`yi → o` for `huit`-style words.** The wav2vec2 vocab represents the
  `huit` glide as a 2-char token `yi` (since `/ɥ/` isn't in the vocab and
  phonemizer outputs `yit` rather than `ɥit`). When the audio is
  not-quite-rounded, the model can reach for an unrelated single token
  (`o`) instead of partial credit. This makes `huit` and similar `/ɥi/`
  words score harshly even when the audio is in the right neighborhood.
- **Boundary slop** at word junctures (e.g., `u → əl` between `vous` and
  `plaît`). Forced alignment can produce weird per-frame argmax over an
  unstable boundary span; the produced phoneme is misleading.

## Reference dialect mismatch

The Phase 3 plumbing routes `lang="fr"` to phonemizer's `fr-fr` (Parisian
French). Quebec French (espeak `fr-ca` if/when phonemizer accepts it),
Belgian, and Swiss varieties differ in nasal vowel inventory, schwa
behavior, and prosody. Same Phase 4 dialect-selection feature called out
in the Spanish doc applies here.

## Phrase set used for dogfooding

```
Bonjour, je m'appelle Justin.
Comment allez-vous aujourd'hui ?
Tu as deux fleurs bleues.
Je voudrais huit petits pains, s'il vous plaît.
```
