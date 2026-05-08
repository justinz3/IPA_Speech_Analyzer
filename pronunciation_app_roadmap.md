# Pronunciation Feedback Tool — Project Roadmap

Kickoff brief for a fresh project directory. Copy to the new project root and start work from here.

## What we're building

A personal pronunciation coach for language learners. User reads a known sentence aloud → tool aligns audio against the reference pronunciation → flags per-phoneme errors → surfaces curated articulatory references (diagrams, videos) explaining how to fix each one.

## Who it's for

Justin is user #1. Studies four languages (Mandarin, Japanese, French, Spanish). The acute personal need is **French**: Duolingo flags pronunciation as bad without saying *what* is wrong. Mandarin and Spanish are already fine; Japanese pitch accent is a curiosity, not an active gap.

## Why this shape

- Real ML/infra meat: forced alignment, phoneme-level scoring, retrieval over a curated reference library.
- Publicizable from day one — strangers can `pip install` and run with a local audio file. No personal account coupling.
- Differentiated from existing tools:
  - Duolingo grades pass/fail, doesn't explain.
  - Phonetics learning apps explain phonemes but don't grade *your* audio.
  - This sits in the gap.
- Justin will dogfood it for years across multiple languages.

## Key technical decisions (already made — don't relitigate without reason)

### Forced alignment, not free transcription
User reads a *known* sentence. Tool aligns audio to that text and scores per-phoneme deviation. This sidesteps Whisper's well-known failure mode of "correcting" non-native pronunciation toward the nearest native-sounding word — which is exactly what kills pronunciation scoring.

Free transcription ("what does my audio sound like in IPA?") is still useful as a standalone Phase 1 feature, but it is **not** the basis of the scoring loop.

### Phoneme-level recognizer, not Whisper
Candidate model: `facebook/wav2vec2-lv-60-espeak-cv-ft` — multilingual phoneme recognition using the espeak phonetic alphabet. Verify it's still SOTA at build time; the field moves fast.

### Reference IPA generation
Use `phonemizer` (Python) backed by `espeak-ng`. No need to train a G2P model — espeak-ng has decent G2P for all four target languages.

### Retrieval, not generation, for corrections
Phoneme mismatch → look up a curated articulatory diagram + linked phonetics video (YouTube). Don't generate articulatory videos — that's research-grade and the human-made references are higher quality anyway. The interesting ML problem is matching the user's specific error pattern to the right reference, not synthesizing new media.

## Spanish as the calibration testbed

Build v0/v1 against **Spanish**, not French — Justin's Spanish is decent and Spanish phonology is regular, so he can sanity-check whether the tool is producing correct output before trusting it on French (where he doesn't know what's wrong, so can't tell a tool bug from a real pronunciation error).

Spanish proves the pipeline. French is what the pipeline is for.

## French phonology challenges to plan for

The whole point of the tool is surfacing these specifically — Phase 3 (when French enters the pipeline) needs to handle them well:

- Front rounded vowels: /y/ (lune), /ø/ (peu), /œ/ (peur)
- Nasal vowels: /ɑ̃/, /ɛ̃/, /ɔ̃/
- Liaison (silent final consonants pronounced before vowel-initial words)
- Schwa drops (/ə/ deletion in casual speech)

Phoneme inventory mismatch is real here — espeak-ng's French phoneme set should cover these but verify against IPA references early.

## Phased plan

### Phase 1 — Free-transcription baseline (week 1)
- Audio in → IPA out, Spanish only
- wav2vec2 phoneme model + simple post-processing
- CLI, GitHub repo with a README
- Standalone-useful: "show me what I sound like in IPA"

### Phase 2 — Forced alignment + per-phoneme scoring (weeks 2–3)
- User reads a known prompt sentence
- Align audio to reference IPA, score per-phoneme deviation
- Still Spanish — verify the scoring matches Justin's intuition before adding French
- Output: which phonemes were missed, what was produced instead

### Phase 3 — French (week 4+)
- Add French to the pipeline ✓ (2026-05-07)
- Real dogfooding starts here
- The phoneme challenges above (front rounded vowels, nasal vowels, liaison, schwas) all surfaced cleanly in phonemizer's `fr-fr` output and the model vocab; no per-phoneme remediation work was needed. Self-consistency PERs on 10 LibriVox French fixtures range 0.075-0.170 — comparable to or better than Spanish.

### Phase 4 — Dialect selection + coaching lookup ✓ (2026-05-07)
- **Dialect selection** (`--dialect` flag, composite codes on `--lang`, Gradio dropdown). Spanish gets Castilian (`es-es`) vs LatAm (`es-419`); French has only `fr-fr` at the IPA level (espeak's regional voices share phonemic rules — `fr-be`/`fr-ch`/`fr-ca` produce identical IPA, just different synth timbre). Real Quebec/Belgian dialect handling needs a non-espeak reference source.
- **Coaching lookup**: per-phoneme reference media (image/audio/video) is the foundation, with hand-written override tips for common error pairs as an additive layer. ~52 phoneme entries cover the es+fr inventory; 13 override tips lifted from the failure-modes docs (β/v, ɾ/r, x/h, tʃ/dʒ for es; y/i, ø/eː, ʁ/∅, t→tʃ for fr).
- **Phoneme media is deferred.** The plumbing is wired (CLI prints image/audio paths, Gradio renders `<img>`/`<audio>` via `allowed_paths`), but no PNG/OGG files ship in v1. Curation from Wikimedia Commons (CC-BY-SA / public domain) is a follow-up content commit — no engineering blocker.

### Phase 5 — Add another language
- Mandarin (tonal scoring is a *new* ML problem — pitch contour extraction + DTW vs canonical tone shapes) or Japanese (pitch accent — also a pitch contour problem, simpler)
- Pick by motivation

### Phase 6 — Optional polish
- Web UI / containerize / deploy
- IaC (CDK) tier only if still motivated and resume-shaped value still feels missing

### Phase 7 — Read-aloud companion (exploratory, "borderline another project")
Hook the phonetic scorer to OCR'd pages so the user reads a real book and gets pronunciation feedback as they go. Phonetic scoring is solved by Phase 2–3; the *new* work is **position tracking** — knowing where in the OCR'd text the reader currently is so divergences can be flagged in context.
- Cheap version (today, no code): OCR a page elsewhere, paste into the Reference field, record, score. Already works.
- v1: in-app OCR (Apple Vision / Google Vision / Tesseract) → reference text → batched scoring. One page at a time.
- v2: continuous audio + advancing position pointer. 5–10 s lag is acceptable — not strict real-time. Streaming CTC over chunked audio with overlap windows; alignment converges → pointer advances.
- **Manga is its own research project.** Chinese/Japanese manga reads non-linearly (panel order, speech bubbles, vertical / RTL text); sequential position tracking won't work without upstream panel+bubble detection. Standard L→R book pages are the v1 target; manga is explicitly out of scope until v3+.
- **Watch for scope creep.** This is borderline its own project — the only reason it lives here is that it consumes the phonetic scorer's output. Don't let v2 work pull focus from Phase 3 (French) or Phase 4 (curated corrections), which unlock the project's core value.

Portfolio value is unlocked by end of Phase 4. Phases 5–7 are bonus.

## First concrete tasks (Phase 1)

1. Set up Python project, GitHub repo, README
2. Pick and load the phoneme recognition model (verify the candidate above is still appropriate)
3. Write a CLI: `pronounce <audio.wav> --lang es` → prints IPA transcription
4. Test against Justin's own Spanish recordings — does the IPA output look right to a fluent speaker?
5. Document the model's failure modes you find

Don't start building scoring until Phase 1 produces sane IPA on Spanish audio.

## Risks / things to watch

- **L2 speech recognition is noisy.** Even phoneme-level models trained on native speech will mis-recognize learner audio. Calibration on Spanish is meant to surface this; budget time for it.
- **espeak-ng phoneme inventories vary by language.** What espeak emits for Spanish IPA may not match what the wav2vec2 model emits. Phoneme mapping/normalization between the two will probably eat a day.
- **Forced alignment quality on short utterances.** Single-word recordings may not give the alignment model enough context. Consider sentence-level prompts as the minimum unit.
- **Scope creep into TTS / generation.** Resist. Retrieval is the v1 answer for corrections.

## Hosting / deployment (deferred — Phase 6)

Keep everything local for Phases 1–5. Hosting decisions live in `side_project_ideas.md` in the job_applications directory and don't need to be made until Phase 6.
