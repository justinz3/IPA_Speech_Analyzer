---
title: vocal-ipa-trainer
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Audio to IPA pronunciation feedback with per-phoneme scoring. Spanish, French, Mandarin, Japanese.
---

# vocal-ipa-trainer

Audio in, IPA out. A pronunciation feedback CLI for language learners.

> **Status:** Phase 7 — Spanish, French, Mandarin, and Japanese. Segmental scoring, prosody scoring, IPA tooltips, and vowel audio in the web UI. Experimental. See [`pronunciation_app_roadmap.md`](pronunciation_app_roadmap.md) for the long arc.

## What it does

Reads a recording of you speaking and prints what your audio sounds like in
the International Phonetic Alphabet. With `--reference`, it goes one step
further: forced-aligns your audio against the reference IPA and tells you
which phonemes the model heard a different sound at.

Free transcription:

```
$ pronounce tests/data/fixtures/es_001.flac --lang es
k o m o t o ð a s l a x p e n a s a n k e t a n t a s s o n u n a s o l a p e n a u n a s o l a ɲ i n f i n i t a ...
```

Per-phoneme scoring against a known sentence:

```
$ pronounce tests/data/fixtures/es_001.flac \
            --reference "$(cat tests/data/fixtures/es_001.txt)"
expected  produced  start   end     ok
k         k         0.54    0.56    ✓
o         o         0.58    0.60    ✓
m         m         0.66    0.70    ✓
o         o         0.70    0.72    ✓
t         t         0.90    0.92    ✓
o         o         0.96    0.98    ✓
ð         ð         1.08    1.10    ✓
a         a         1.12    1.14    ✓
s         s         1.18    1.20    ✓
l         l         1.26    1.28    ✓
a         a         1.28    1.30    ✓
s         x         1.34    1.36    ✗
...
PER: 0.073  (8/109 phonemes wrong)
```

Behind the scenes: a wav2vec2 phoneme recognizer
([`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft))
trained with espeak-ng phoneme labels.

When a miss matches a curated error pattern, the score table appends a
"Misses" block with the expected vs produced phoneme names plus a
targeted tip:

```
Misses (1 unique):

  expected β  voiced bilabial fricative
  produced v  voiced labiodental fricative
  tip: Don't use English [v]
       Lips touch lightly without contact between lower lip and upper teeth.
       Same articulation as a Spanish `b` between vowels — bilabial, not
       labiodental.
```

The Gradio UI renders the same as side-by-side comparison cards (with
slots for articulator diagrams and audio samples once those land).

## Install

Recommended (single-binary install via [uv](https://docs.astral.sh/uv/)):

```
uv tool install vocal-ipa-trainer
```

Plain pip works too:

```
pip install vocal-ipa-trainer
```

**First run downloads ~1 GB** of model weights to `~/.cache/huggingface/hub/`.
Cached after that.

### System requirements

- Python ≥ 3.11 (< 3.13)
- `libsndfile` for audio I/O — `sudo apt install libsndfile1` on Debian/Ubuntu
- `espeak-ng` for the reference-IPA / PER paths — `sudo apt install espeak-ng`
- Optional CUDA for GPU inference (CPU works fine; ~0.5s per second of audio)

## Usage

```
pronounce <audio> [--lang es] [--dialect CODE] [--format text|json]
                  [--device auto|cpu|cuda] [--model HF_ID] [--raw]
                  [--reference TEXT]
```

| Flag          | Default                                | Notes |
|---------------|----------------------------------------|-------|
| `--lang`      | `es`                                   | Language or composite locale: `es`, `es-es`, `es-419`, `es-latam`, `fr`, `fr-fr`, `cmn`, `cmn-cn`, `zh`, `ja`, `ja-jp`. |
| `--dialect`   | (use `--lang` default)                 | Reference IPA dialect: codes (`es-es`, `es-419`, `fr-fr`) or aliases (`castilian`, `latam`, `parisian`). See **Dialects** below. |
| `--format`    | `text`                                 | `text` is one IPA line (free transcribe) or a per-phoneme table (scoring). `json` includes timing/score fields. |
| `--device`    | `auto`                                 | `auto` picks CUDA if available else CPU. |
| `--model`     | `facebook/wav2vec2-lv-60-espeak-cv-ft` | Override at your own risk. |
| `--raw`       | off                                    | Emit the model's raw labels before whitespace cleanup. Free-transcribe only. |
| `--reference` | unset                                  | When set, score audio against this sentence (forced alignment + per-phoneme errors). Cannot combine with `--raw`. |

`pronounce --help` prints the full surface.

### Languages

| Code | Aliases | Reference IPA source | Notes |
|------|---------|----------------------|-------|
| `es` | `es-es`, `es-419`, `es-latam` | espeak-ng | Two dialects: Castilian (`es-es`) and Latin American (`es-419`). |
| `fr` | `fr-fr` | espeak-ng | One dialect at the IPA level (`fr-fr`). |
| `cmn` | `cmn-cn`, `zh` | espeak `cmn-latn-pinyin` via pypinyin | Accepts Hanzi (e.g., `你好`) or pinyin (diacritic `nǐ hǎo` or numeric `ni3 hao3`). Tone digits embedded in vowel tokens (`ɑ1`, `iɛ2`); tones register as wrong-token misses in the segmental scorer. Full pitch-contour tone scoring lands in Phase 6. |
| `ja` | `ja-jp` | pyopenjtalk | Accepts any mix of Kanji/hiragana/katakana. Consecutive same-vowel pairs collapsed to Vː (long vowel). **Caveat:** the wav2vec2 model emits English-like phonemes for Japanese audio due to degraded training labels (espeak-ja was broken). PER thresholds are loose; a Japanese-finetuned checkpoint would improve accuracy. |

### Dialects

The reference IPA depends on which dialect you target. The most concrete
case: in Castilian Spanish (`es-es`), `manzana` and `cinco` use `/θ/` for
`z` and soft `c`; in Latin American Spanish (`es-419`) they use `/s/`. If
you say "manSana" but the reference is Castilian, every `z` will get
flagged. Set `--dialect` to match your target accent.

| Code      | Alias       | Espeak code | Notes |
|-----------|-------------|-------------|-------|
| `es-es`   | `castilian` | `es`        | Default for `--lang es`. /θ/ on `z` and soft `c`. |
| `es-419`  | `latam`     | `es-419`    | Latin American Spanish. /s/ instead of /θ/. |
| `fr-fr`   | `parisian`  | `fr-fr`     | Default for `--lang fr`. The only French dialect espeak distinguishes at the IPA level. |

Composite codes on `--lang` work too: `pronounce ... --lang es-419 --reference manzana`
is the same as `pronounce ... --lang es --dialect latam`.

**Why no Quebec / Belgian / Mexican entries?** Probing espeak-ng directly
shows the IPA output is identical across `fr-fr`, `fr-be`, `fr-ch`, `fr-ca`
(only the synthesized speech timbre differs, not the phonemic rules), and
`es-mx` produces the same IPA as `es-419`. Real Quebec/Belgian dialect
handling needs a different reference source than espeak; tracked as future
work.

### IPA tooltips

Every IPA token in the score table is wrapped in a hover tooltip showing its
phoneme name and a language-specific note. Hover over any `expected` or
`produced` cell to see e.g. "close front rounded vowel / French `u` in `tu`".
Tokens not in the inventory (model artifacts, `∅`) are shown as plain text.

When a miss-comparison card appears for a vowel, an inline audio player lets
you hear the reference pronunciation (15 es+fr vowels curated from Wikimedia
Commons CC-BY-SA 3.0; see `src/vocal_ipa/data/phonemes/LICENSES.md`).

### Prosody scoring

When `--reference` is used, a parallel prosody scoring path runs alongside the
segmental scorer. It operates directly on the raw audio signal (pitch via
`librosa.pyin`, intensity via numpy RMS) and is completely independent of the
wav2vec2 model.

| Language | What is scored | Signal used |
|----------|---------------|-------------|
| `es` | Lexical stress — is the stressed syllable louder than the mean? | RMS intensity per vowel span |
| `fr` | Utterance intonation — rising (interrogative `?`) or falling (declarative) | Pitch slope of utterance tail |
| `cmn` | Tone contour per syllable — flat, rising, dip, or falling | Pitch contour shape over vowel span |
| `ja` | F0 mean per vowel (no pass/fail yet — accent ground truth pending) | Pitch over vowel span |

The prosody column appears next to each phoneme in the score table, and a
`Prosody: N%` summary line appears below the PER line.

**Shaky-baseline caveat for Mandarin and Japanese:** prosody span timings come
from the forced-alignment step, which itself depends on the wav2vec2 segmental
model. That model is degraded for Mandarin (palatal/retroflex confusion,
ü-vowel collapse) and near-unusable for Japanese (English-like output from
broken training labels). Prosody *values* are real, but the span boundaries may
be slightly mis-positioned. For Spanish and French the segmental model is
reliable and prosody results are trustworthy.

## How it works

1. Decode your audio to float32 mono via `soundfile`; resample to 16 kHz with
   `torchaudio` if needed.
2. Feed it through a wav2vec2 phoneme model (`Wav2Vec2ForCTC`) under
   `torch.inference_mode`.
3. Argmax-decode the per-frame logits and run them through the model's
   `Wav2Vec2PhonemeCTCTokenizer` to get an IPA string.
4. Light post-processing: collapse whitespace. The roadmap calls out a known
   espeak↔IPA inventory mismatch; that's tracked and will be addressed when
   real failure modes surface (Phase 1 close).

### Scoring (`--reference`)

5. Run `phonemizer` over the reference text to get its IPA.
6. Greedy-tokenize the reference IPA into the model's vocabulary, bypassing
   `Wav2Vec2PhonemeCTCTokenizer`'s built-in (English-by-default) phonemizer
   so Spanish phonemes don't get silently anglicized into `oʊ`/`ɑː`/`iː`.
7. Forced-align the reference token sequence against the CTC log-probs via
   `torchaudio.functional.forced_align` (Viterbi). This yields per-phoneme
   start/end frame indices with 20 ms granularity (50 Hz frame rate).
8. For each aligned span, take the most-frequent non-blank argmax across the
   span's frames as the *produced* phoneme; compare to *expected*. The
   headline PER is the count of mismatches divided by the total reference
   phonemes — no insertion/deletion noise because alignment forces the
   reference shape.

## Known limitations

- **L2 speech is noisy.** The model was fine-tuned on Common Voice native
  speakers; non-native pronunciation will mis-recognize more often.
- **espeak ↔ IPA inventory mismatch.** What espeak emits for Spanish or
  French IPA may not perfectly match what the wav2vec2 model emits. Tracked.
- **Single-word audio is unreliable.** Forced alignment needs enough context
  to find sensible spans; sentence-level prompts are the practical minimum.
- **Stress is not scored.** The wav2vec2 model's vocabulary doesn't contain
  espeak's `ˈ`/`ˌ` stress marks (it was trained on stress-stripped labels),
  so a learner stressing the wrong syllable in `papel` won't be caught by
  `--reference` scoring. Stress is a pitch + duration problem and shares
  machinery with tonal/pitch-accent scoring; both land later in the
  roadmap.
- **French liaison is partial.** Phonemizer applies some liaisons
  (`les amis → lez ami`) and skips others (`comment allez-vous → kɔmɑ̃ alevu`,
  no /t/ liaison). Don't expect uniform liaison treatment in the reference.
- **French schwas are kept.** Phonemizer surfaces all `/ə/` in the formal
  pronunciation (`je ne sais pas → ʒə nə sɛ pa`); casual French drops them.
  Dropping schwas in your audio will register as a mismatch — that's
  honest feedback, not a bug.
- **French /ɥ/ is not emitted.** Phonemizer outputs `lui → lyi`, `huit → yit`
  rather than using the labio-palatal glide. The model vocab also lacks `ɥ`,
  so this is consistent end-to-end.
- **French nasal merger not applied.** Phonemizer keeps `/ɛ̃/` and `/œ̃/`
  distinct (`vin → vɛ̃`, `un → œ̃`). Many Parisian speakers merge them; if
  your audio merges them, the model may flag the difference.

## Known model failure modes

Quirks of `wav2vec2-lv-60-espeak-cv-ft` observed on real Spanish input
(both the LibriVox fixtures and informal recordings). These are things to
know when reading the IPA, not bugs to file:

- **Spurious vowel-length marks** (`oː`, `eː`). Spanish vowel length isn't
  phonemic; the model emits length marks anyway, mostly on stressed or held
  vowels. Treat as decoration.
- **Intervocalic lenition is inconsistent.** Spanish lenites `/b d g/` to
  `/β ð ɣ/` between vowels. The model gets this right most of the time, but
  you'll occasionally see a surface stop where it should be a fricative
  (and vice versa). Don't treat a single character as definitive.
- **/t/ can drop in clusters.** `detrás /deˈtɾas/` has come back as
  `d e r a s`, and `Estados /esˈta.ðos/` close to `e s a o s`. The /t/
  weakens in `/tɾ/` and `/st/` contexts.
- **Coda /l/ can disappear.** `los → o s` is common in fast speech; the
  model drops `/l/` in syllable-final positions before another consonant.
- **Hesitation gets transcribed.** Filler like "uhh" comes out as repeated
  vowels (`o o o`). The model doesn't distinguish speech from non-speech.

For your own pronunciation feedback: many things that look "wrong" in the
IPA are actually the model faithfully capturing your audio. Common ones:

- A glide on what should be a pure Spanish vowel (`oɪ`, `aɪ`) — Spanish
  vowels don't diphthongize the way English ones do.
- `v` (English labiodental) where Spanish has `β` (bilabial fricative).
- `ʒ` or `ʃ` for "ll" — Spanish dialects realize "ll" as `/ʝ/`, `/j/`,
  `/ʎ/`, or in Río de la Plata `/ʒ ʃ/`. If you're an English speaker
  reaching for "ll", you may land in the postalveolar zone by accident.

When in doubt: if the surface "error" matches a known L2 pattern, suspect
the audio before suspecting the model.

For per-language patterns observed during dogfooding, see
[`spanish_failure_modes.md`](spanish_failure_modes.md) and
[`french_failure_modes.md`](french_failure_modes.md). Each organizes
misses by attribution (your audio vs reference dialect vs model
artifact) — useful when you're trying to figure out whether a "wrong"
phoneme is something to drill or something to ignore.

### Mandarin failure modes

The reference IPA is generated via espeak's `cmn-latn-pinyin` voice, which
encodes tones differently from standard pinyin numbering:

- **Tone encoding is not 1:1 with pinyin numbers.** Tones 1 and 4 both
  map to the digit `5` in the vowel token (e.g., `ɑ5`). Tone 3 maps to
  the digit `2` (e.g., `yɛ2`) — not tone 2. Tone 2 produces a vowel
  quality shift without a digit. This means two syllables differing only
  by tone 1 vs tone 4 look identical in the reference, and tone 3 looks
  like tone 2. Phase 6's prosody scorer uses these espeak digits directly
  to score pitch contour shape, so it inherits the same encoding quirks.
- **Palatal/retroflex confusion** (`ɕ` vs `s.`, `tɕ` vs `ts.`). The
  palatal consonants (`x → ɕ`, `j/q → tɕ`) and the retroflex sibilants
  (`sh → s.`, `zh → ts.`) are acoustically close; the model frequently
  substitutes one for the other. If your reference is `ɕ` and the model
  produces `s.`, it's a model accuracy issue, not a pronunciation error.
- **`ü`-vowel collapse to `i`.** Syllables with the `ü` medial — `xue`,
  `jue`, `que`, `lü`, `nü` etc. — use the vowel sequence `yɛ`/`y` in
  the reference. The model often produces `i` or `i.` instead, because
  `ü` (close front rounded) is underrepresented in the multilingual
  training distribution and sits close to `i` in acoustic space.
- **Final nasals frequently dropped.** Coda `-n` and `-ng` (reference
  tokens `n` and `ŋ`) are often produced as `∅`. If the last phoneme
  of a syllable is missing, suspect coda nasal deletion before
  suspecting your pronunciation.
- **ST-CMDS fixture PER is optimistically low.** The Mandarin slow-test
  fixtures come from ST-CMDS (OpenSLR-38), which may overlap with the
  wav2vec2 fine-tune set. Real-world Mandarin PER on unseen audio will
  be higher than the fixture numbers suggest.

### Japanese failure modes

The reference IPA is generated via pyopenjtalk (espeak's `ja` voice is
broken — it falls back to English IPA mid-utterance). The wav2vec2 model
also has degraded Japanese accuracy because espeak-ja was broken when the
model's training labels were generated.

- **Model emits English-like phonemes.** The training labels for Japanese
  audio were produced by a broken espeak-ja, so they contain English-like
  IPA tokens (`aɪ`, `e̞`, `tʃ`, `ə`). The model learned these distorted
  labels, so it emits English-adjacent phonemes even on clean Japanese
  speech. PER thresholds are set to 0.85 to absorb this; treat Japanese
  scoring as approximate.
- **`u` → `i` substitution** (and `u` → `ɯ`-adjacent errors). Japanese
  `u` is phonetically an unrounded back vowel `ɯ`, but we map it to `u`
  in the reference (the only back-close token in the model vocab).
  The model frequently produces `i` instead — especially in devoiced
  contexts (e.g., after unvoiced consonants like `k`, `s`, `t`) or after
  the palatal glide `j` (e.g., `ゆ → j u` often registers as `j i`).
  If you see `u → i` in the score table, it is almost always a model
  artifact, not a pronunciation error.
- **Pitch accent is not scored.** Japanese minimal pairs distinguished
  only by pitch accent (e.g., 橋 vs 箸, both `h a sh i`) score
  identically. Accent scoring lands in Phase 6.
- **Kanji shared with Chinese (e.g., 雪) reads as Japanese.** pyopenjtalk
  applies Japanese readings, so 雪 → `yuki` (ゆき), not `xuě`. If you
  paste Hanzi into `--lang ja`, you get the Japanese on'yomi/kun'yomi
  reading, which may not match your intention. Use `--lang cmn` for
  Mandarin input.

## Hosted demo (Hugging Face Spaces)

Deploy `pronounce-web` as a free hosted demo on [Hugging Face Spaces](https://huggingface.co/spaces):

1. Create a new Space — SDK: **Gradio**.
2. Add this repo as a remote and push:

   ```
   git remote add space https://huggingface.co/spaces/<your-user>/vocal-ipa-trainer
   git push space main
   ```

   Spaces reads its config from the YAML frontmatter at the top of this README
   (`sdk: gradio`, `app_file: app.py`). Python deps come from
   [`requirements.txt`](requirements.txt); apt packages (libsndfile1,
   espeak-ng) come from [`packages.txt`](packages.txt). The model is pulled
   from Hugging Face on first request and cached.

3. Open the Space URL — it shares the same UI as `pronounce-web` locally.

`requirements.txt` mirrors `pyproject.toml`'s base + `[web]` deps. When
upgrading dependencies, update both.

## Roadmap

Full plan: [`pronunciation_app_roadmap.md`](pronunciation_app_roadmap.md).

- **Phase 1:** Spanish-only audio → IPA CLI. ✓
- **Phase 1.5:** Gradio web UI with mic recording. ✓
- **Phase 2:** Forced alignment + per-phoneme scoring (Spanish, phoneme identity only — stress not scored). ✓
- **Phase 3:** Add French. Same scoring; no new ML — phonemizer's `fr-fr` and the multilingual model handle it. ✓
- **Phase 4:** Dialect selection (es-es / es-419 / fr-fr) + curated coaching lookup (per-phoneme reference + override tips for common error pairs). Plumbing complete; phoneme media curation is a follow-up content commit. ✓

## Development

```
git clone <repo>
cd vocal_ipa_trainer
uv sync --extra dev
uv run pytest -m "not slow"     # fast tests, no model load
uv run pronounce path/to/audio.wav --lang es
```

Slow tests (snapshot + PER + score behavior across 15 fixtures, ~170 s,
downloads model on first run):

```
uv run pytest -m slow
```

Regenerating test fixtures (rare — only when intentionally updating the
LibriVox-derived fixture set or accepting model output drift):

```
uv sync --extra fixtures
uv run python scripts/fetch_fixtures.py --language es        # default 5 Spanish clips from MLS
uv run python scripts/fetch_fixtures.py --language fr -n 10  # 10 French clips
uv run python scripts/fetch_fixtures.py --language cmn -n 5  # 5 Mandarin clips (ST-CMDS)
uv run python scripts/fetch_fixtures.py --language ja -n 5   # 5 Japanese clips (JSUT)
uv run python scripts/regen_goldens.py         # frozen IPA snapshots (free transcribe)
uv run python scripts/regen_score_goldens.py   # frozen per-phoneme score snapshots
```

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgments

- [`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)
  — the multilingual phoneme recognizer this whole project rides on.
  ([Xu et al., 2021 — *Simple and Effective Zero-shot Cross-lingual Phoneme Recognition*](https://arxiv.org/abs/2109.11680))
- [`facebook/multilingual_librispeech`](https://huggingface.co/datasets/facebook/multilingual_librispeech)
  — LibriVox-derived test fixtures (CC-BY 4.0).
- [`phonemizer`](https://github.com/bootphon/phonemizer) + [`espeak-ng`](https://github.com/espeak-ng/espeak-ng) — reference IPA generation.
