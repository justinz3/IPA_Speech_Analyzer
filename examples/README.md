# Examples

Quickest way to try `pronounce` without recording your own audio: use one of
the bundled test fixtures.

```
uv run pronounce tests/data/fixtures/es_001.flac --lang es
uv run pronounce tests/data/fixtures/es_001.flac --lang es --format json
```

## Recording your own sample

Record a 3–10 second clip of yourself speaking Spanish. Audio just needs to be
something `soundfile` can read — WAV, FLAC, OGG. Anything works:

- **GUI:** Audacity, GarageBand, Voice Memos.
- **CLI:** `arecord -f cd -d 6 sample.wav` (Linux), `rec sample.wav` (sox).
- **ffmpeg:** `ffmpeg -f alsa -i default -t 6 sample.wav`

The CLI handles any sample rate (resampled to 16 kHz on load). Mono or
stereo both work; stereo is downmixed by averaging.

Once Phase 1.5 ships (Gradio web UI), in-browser microphone recording will
replace this whole section.

## Reading the output

Spaces in the IPA output separate individual phonemes — this is what the
wav2vec2 model emits, not standard IPA prose. For example,
`o l a` is `/ola/` ("hola" with the silent `h`), not three words.
