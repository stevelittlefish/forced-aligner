# How the alignment works (and where it breaks)

## The job

Forced alignment: you already have the words; you want their timing. This is a
fundamentally easier and more reliable problem than transcription, because the
model is not allowed to invent, reorder, or drop words — only to decide *where*
in time each known word lands. That constraint is what makes it work even on
slurred or sustained singing.

## The pipeline, step by step

1. **Load and resample.** Decode the vocal stem, resample to 16 kHz mono — what
   wav2vec2 expects.

2. **Trim to the sung region.** Voice-activity detection finds where singing
   actually starts and ends. Aligning against long leading silence or an
   instrumental intro pushes the model to smear the first word across the gap.

3. **One segment, whole text.** The known lyrics go in as a single block
   spanning the sung region — not line by line. The aligner sees all the words
   and all the audio together and places them monotonically.

4. **wav2vec2 emits frame → character probabilities.** The acoustic model
   produces, for every ~20 ms frame, a probability distribution over its
   character vocabulary (letters + a CTC "blank"). This is the only neural step;
   it knows nothing about the lyrics.

5. **CTC forced alignment (Viterbi).** Given the known character sequence and
   the per-frame probabilities, a trellis search finds the single most likely
   *monotonic* path that emits exactly those characters in order. Monotonic =
   time only moves forward, so words cannot be reordered. This is the "forced"
   part.

6. **Characters → words → lines.** Character timings are grouped into word
   start/end times. Words are then walked back onto the original input lines (we
   know how many word-tokens each line had), giving line-level start/end for the
   karaoke display.

## Where the model comes from

WhisperX maintains a map from language code to a default wav2vec2 model on
Hugging Face (Meta's `wav2vec2` family and community fine-tunes — English,
German, French, Japanese, and so on). `load_align_model(language_code, device)`
downloads the right one on first use and caches it to disk. There is a
**different model per language**; a language with no mapped model can't be
aligned (the service returns `422`).

Models load on demand, with English preloaded by default before readiness.
Only one language model stays resident; switching languages unloads the previous
one. `/v1/info` reports the loaded language. Downloaded weights remain cached.

## Where it breaks — and why the spike matters

wav2vec2 was trained on **speech**, not singing. Known failure shapes:

- **Sustained notes / held vowels.** The model sees the same character for a
  long stretch; the word-end timestamp smears late. Crooner and ballad material
  is exactly this.
- **Harmonies and backing vocals.** Multiple simultaneous voices confuse the
  frame probabilities; the aligner may lock onto the wrong line.
- **Screamed / distorted vocals.** Far outside the training distribution;
  character probabilities get mushy and timing degrades.
- **Poor enunciation and ad-libs.** Words in the audio that aren't in the text
  (or vice versa) break the monotonic assumption locally and cause drift around
  that point.

None of these are reasons not to ship — they are reasons the Milestone 0 spike
deliberately tests crooner, nu-metal, harmony-heavy and screamed material, not
just clean pop. The service's job is to be **honest** about failure: unresolved
words come back with `null` timestamps (see [spec.md](spec.md)) so the caller
can fall back to line-level timing or flag the song, rather than trusting a
confident-looking wrong number.

## Alternatives we deliberately did not start with

- **torchaudio `forced_align`** — same wav2vec2 + CTC idea, leaner code path,
  GPU-friendly. A drop-in to compare against later.
- **Qwen3-ForcedAligner-0.6B** — one multilingual model instead of a
  per-language wav2vec2 zoo, handles CJK in a single pass. Worth benchmarking
  once the WhisperX baseline exists.

We start with WhisperX because it is the most battle-tested on this exact
task. See nightingale's `app-core/analyzer/align.py` for a production
implementation of the same approach.
