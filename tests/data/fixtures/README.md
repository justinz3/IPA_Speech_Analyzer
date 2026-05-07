# Test fixtures

Five short Spanish clips used by snapshot tests (`tests/test_snapshots.py`)
and PER tests (`tests/test_per.py`).

## Source

All clips are streamed from
[`facebook/multilingual_librispeech`](https://huggingface.co/datasets/facebook/multilingual_librispeech),
Spanish config, **test** split. The underlying audio is from LibriVox
(public-domain audiobook recordings). The dataset is distributed under
**CC-BY 4.0**.

LibriVox-derived audio has zero training-set leakage with the wav2vec2
checkpoint we use (`facebook/wav2vec2-lv-60-espeak-cv-ft`), which was
fine-tuned on Common Voice — a separate corpus.

## Files

For each clip `es_NNN`:

| File                  | What                                                  |
|-----------------------|-------------------------------------------------------|
| `es_NNN.flac`         | 16 kHz mono audio (downsampled if necessary)          |
| `es_NNN.txt`          | Orthographic transcript (UTF-8, newline-terminated)   |
| `es_NNN.golden.ipa`   | Frozen IPA output of the current pipeline (snapshot)  |

`manifest.json` lists every clip with speaker id, duration, source, and license.

## Regenerating

Audio + transcripts:

    uv sync --extra fixtures
    uv run python scripts/fetch_fixtures.py

IPA goldens (run after fetching, or when intentionally accepting drift):

    uv run python scripts/regen_goldens.py
