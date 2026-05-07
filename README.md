---
title: vocal-ipa-trainer
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Audio to IPA pronunciation feedback. Spanish only (Phase 1).
---

# vocal-ipa-trainer

Audio in, IPA out. A pronunciation feedback CLI for language learners.

> **Status:** Phase 1 — Spanish only, experimental. See [`pronunciation_app_roadmap.md`](pronunciation_app_roadmap.md) for the long arc.

## What it does

Reads a recording of you speaking and prints what your audio sounds like in
the International Phonetic Alphabet. The intended use is comparing your
pronunciation against the reference IPA for the sentence you meant to say —
that comparison itself is Phase 2 work; Phase 1 just gives you the IPA.

```
$ pronounce tests/data/fixtures/es_001.flac --lang es
k o m o t o ð a s l a x p e n a s a n k e t a n t a s s o n u n a s o l a p e n a u n a s o l a ɲ i n f i n i t a ...
```

Behind the scenes: a wav2vec2 phoneme recognizer
([`facebook/wav2vec2-lv-60-espeak-cv-ft`](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft))
trained with espeak-ng phoneme labels.

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
pronounce <audio> [--lang es] [--format text|json] [--device auto|cpu|cuda]
                  [--model HF_ID] [--raw]
```

| Flag        | Default                                | Notes |
|-------------|----------------------------------------|-------|
| `--lang`    | `es`                                   | Phase 1 only accepts Spanish. |
| `--format`  | `text`                                 | `text` is one IPA line, pipe-friendly. `json` includes timing fields. |
| `--device`  | `auto`                                 | `auto` picks CUDA if available else CPU. |
| `--model`   | `facebook/wav2vec2-lv-60-espeak-cv-ft` | Override at your own risk. |
| `--raw`     | off                                    | Emit the model's raw labels before whitespace cleanup. |

`pronounce --help` prints the full surface.

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

## Known limitations

- **L2 speech is noisy.** The model was fine-tuned on Common Voice native
  speakers; non-native pronunciation will mis-recognize more often.
- **espeak ↔ IPA inventory mismatch.** What espeak emits for Spanish IPA may
  not perfectly match what the wav2vec2 model emits. Tracked.
- **Single-word audio is unreliable.** Forced alignment (Phase 2) will need
  sentence-level prompts as the minimum unit. Already true for Phase 1 too.
- **No per-segment timestamps.** The model exposes 50 Hz CTC frames internally
  but Phase 1 doesn't expose timestamps; that's Phase 2 alignment work.

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

- **Phase 1 (here):** Spanish-only audio → IPA CLI.
- **Phase 1.5:** Gradio web UI with mic recording.
- **Phase 2:** Forced alignment + per-phoneme scoring (still Spanish).
- **Phase 3:** French — the actual target.
- **Phase 4:** Curated correction lookup (phoneme → articulatory diagram + reference video).

## Development

```
git clone <repo>
cd vocal_ipa_trainer
uv sync --extra dev
uv run pytest -m "not slow"     # fast tests, no model load
uv run pronounce path/to/audio.wav --lang es
```

Slow tests (snapshot + PER, ~35s, downloads model on first run):

```
uv run pytest -m slow
```

Regenerating test fixtures (rare — only when intentionally updating the
LibriVox-derived fixture set or accepting model output drift):

```
uv sync --extra fixtures
uv run python scripts/fetch_fixtures.py     # 5 short Spanish clips from MLS
uv run python scripts/regen_goldens.py      # frozen IPA snapshots
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
