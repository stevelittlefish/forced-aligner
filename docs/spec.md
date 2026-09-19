# ASS backend contract

`POST /v1/align` accepts multipart:

- `audio`: audio bytes supported by ffmpeg (typically the vocal WAV).
- `params`: a JSON object with nonempty `text` and optional nonempty `language`
  (defaults to the configured language, `en` in the image).

After validating and saving the upload, return HTTP 202:

```json
{"job_id":"<32 hex characters>","state":"queued","artifacts":[]}
```

One worker transitions each job through `queued` → `running` → `succeeded` or
`failed`. Poll `GET /v1/jobs/{job_id}`. On success the response includes:

```json
{"job_id":"<id>","state":"succeeded","artifacts":[{"name":"alignment.json","kind":"metadata","content_type":"application/json","bytes":1234}]}
```

The byte count is the actual file size. Download it from
`GET /v1/jobs/{job_id}/result/alignment.json`. The document contains `language`,
`duration` (seconds), and `lines`, each with `text`, `start`, `end`, and `words`.
Words contain `text`, `start`, and `end`; unresolved timings remain JSON `null`.

Malformed parameters or empty audio return 400; missing multipart fields return
422; oversized audio returns 413; a full queue returns 503. An accepted job that
cannot decode audio, load a language model, or complete inference becomes
`failed` with an `error` string and empty `artifacts`. Unknown jobs/artifacts
return 404; downloading before success returns 409.

`GET /health` is ready after configured preloads finish. A preload failure
aborts startup. `GET /v1/info` reports the loaded language, default language,
`max_loaded_languages: 1`, `capabilities: ["align"]`, `eviction: "stop"`, and
`vram: {cuda, device, allocated_mb, reserved_mb, peak_mb}`. Memory values are
MiB; peak is PyTorch's process-lifetime peak allocated memory, not total driver
memory. CPU reports zeros without importing torch.

If `server.auth_token` is set, submission, polling and artifact download require
its Bearer token. Leave it empty for ASS: ASS does not forward backend auth.
Health and info are unauthenticated LAN diagnostics.

The synchronous `/align` and `/models` routes are removed. No park/unpark routes
are provided; configure stop eviction. Run one uvicorn process (no `--workers`):
the worker queue and concurrency limit belong to that process.
