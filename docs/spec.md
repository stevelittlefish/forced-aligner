# API specification

One service, multipart in / JSON out, matching the LAN convention (Demucs is
called the same way). Base URL is `http://<host>:<port>` from `config.toml`.

## `POST /align`

Align known lyric text to a vocal audio clip.

**Request** — `multipart/form-data`:

| part | type | required | notes |
|---|---|---|---|
| `audio` | file (WAV) | yes | The vocal stem. Mono or stereo; any sample rate (resampled internally to 16 kHz for the model). |
| `params` | JSON string | yes | See below. |

`params` fields:

| field | type | required | notes |
|---|---|---|---|
| `text` | string | yes | The lyrics. **Newlines are line breaks** — the response preserves this line structure. Blank lines are ignored. |
| `language` | string | no | ISO code (`en`, `de`, `ja`, …). Selects the wav2vec2 model. If omitted, falls back to `default_language` from config. See the auto-detect note below. |

**Response** — `200`, JSON:

```json
{
  "language": "en",
  "duration": 184.2,
  "lines": [
    {
      "start": 12.42,
      "end": 16.81,
      "text": "Hello darkness my old friend",
      "words": [
        {"text": "Hello",    "start": 12.42, "end": 12.91},
        {"text": "darkness",  "start": 13.05, "end": 13.74},
        {"text": "my",        "start": 14.18, "end": 14.39},
        {"text": "old",       "start": 14.43, "end": 14.79},
        {"text": "friend",    "start": 14.82, "end": 16.81}
      ]
    }
  ]
}
```

This is deliberately the same shape as the sing thing's internal lyric
representation, so the Go side stores it almost as-is.

### Unresolved timing

A word the aligner could not place gets `"start": null, "end": null` rather than
a guessed value. This happens on sustained notes, screams, ad-libs not in the
text, and mismatched lyrics. A line whose words are all null gets null
start/end. **Nulls are the honest signal that a caller should fall back to
line-level timing or flag the song** — never invent a timestamp to avoid one.

### Errors

| status | when | body |
|---|---|---|
| `400` | missing/unparseable `params`, empty `text`, no `audio` | `{"error": "..."}` |
| `415` | `audio` not decodable as audio | `{"error": "..."}` |
| `422` | unsupported `language` (no wav2vec2 model maps to it) | `{"error": "...", "supported": [...]}` |
| `500` | model load or alignment failure | `{"error": "..."}` |

### The auto-detect caveat

Language detection normally comes from a *transcription* pass. This service does
not transcribe, so it cannot truly detect language from audio alone. If
`language` is omitted it uses `default_language`. The caller (the sing thing)
usually knows the language from track metadata or the LRCLIB result, so it should
send it. Documented so nobody expects magic here.

## `GET /health`

```json
{ "status": "ok", "device": "cuda", "models_loaded": ["en", "de"] }
```

`status` is `ok` once the app is up. `models_loaded` is the languages whose
wav2vec2 model is currently resident in memory (loaded lazily on first use).

## `GET /models`

```json
{
  "device": "cuda",
  "loaded": ["en"],
  "default_language": "en"
}
```

Introspection for debugging which models are warm. Loading is lazy and
per-language; the first request for a new language pays the load cost (seconds).
