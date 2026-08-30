# forced-aligner

A small LAN HTTP service that answers one question:

> **Given a vocal audio clip and the exact words sung in it, when is each word
> sung?**

Text + audio in, word-level timestamps out. It does **forced alignment**: it
forces known lyrics onto audio and finds the timing. It does *not* transcribe —
it never guesses the words, it only times the ones you give it. Forcing known
text is far more robust than transcribing singing, which is why this is a
separate, focused service.

Built for [the sing thing](https://github.com/stevelittlefish/the_sing_thing)
(LAN karaoke), but it is generic: nothing here is karaoke-specific. It's one
more GPU service on the AI server, alongside Demucs, Whisper and the rest, called
over HTTP.

## What it does, in one breath

1. Load a vocal stem (WAV) and the known lyric text.
2. A **wav2vec2 CTC acoustic model** (per language, from Hugging Face) emits a
   character-probability distribution per ~20 ms audio frame.
3. A CTC/Viterbi pass finds the single best *monotonic* alignment of the known
   characters to those frames.
4. Character timings group into words; words map back onto the input lines.
5. Return line- and word-level start/end times.

The alignment engine is [WhisperX](https://github.com/m-bain/whisperX) — note
**Whisper itself does no aligning here**; WhisperX's alignment step is a wav2vec2
model plus the CTC trellis. See [docs/spec.md](docs/spec.md) for the details and
[docs/alignment.md](docs/alignment.md) for how the algorithm works.

## Where the model comes from

Nothing to train or manage. WhisperX ships a language-code → wav2vec2-model map
and **downloads the right model from Hugging Face on first use**, then caches it
on disk. That download is the only time this service touches the internet; after
that it runs fully local on the LAN.

## API

One real endpoint. Multipart, mirroring how Demucs is called on this LAN.

```
POST /align       multipart: `audio` (WAV) + `params` JSON {text, language?}
                  -> { language, lines:[ {start,end,text, words:[{text,start,end}]} ] }
GET  /health      -> { status, device, models_loaded }
GET  /models      -> loaded/available alignment models
```

Full contract in [docs/spec.md](docs/spec.md).

## Quick start (dev)

Needs Python 3.11+ and, for real speed, a CUDA GPU. CPU works for smoke tests.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp config.example.toml config.toml     # edit device/port if needed
./run.sh                               # uvicorn on the configured port
```

Smoke test:

```sh
curl -s http://localhost:8830/health | jq
curl -s -F 'audio=@vocals.wav' \
     -F 'params={"text":"Hello darkness my old friend","language":"en"}' \
     http://localhost:8830/align | jq
```

## Deploy

One container on the AI server. See the [Dockerfile](Dockerfile) (CUDA base) and
[docs/deploy.md](docs/deploy.md). Configuration is a `config.toml` — **no
environment variables**.
