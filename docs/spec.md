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
`max_loaded_languages: 1`, `capabilities: ["align"]`, `eviction: "park"`,
`parked: <bool>` (true while the model is parked in CPU RAM), and
`vram: {cuda, device, allocated_mb, reserved_mb, peak_mb}`. Memory values are
MiB; peak is PyTorch's process-lifetime peak allocated memory, not total driver
memory. CPU reports zeros without importing torch.

## Park / unpark (ASS eviction)

`POST /park` moves the resident model to CPU RAM and frees the CUDA cache back to
the driver, so ASS can hand the GPU to another backend without making us
cold-start later. `POST /unpark` moves it back onto the GPU. Both return a small
JSON summary (`{parked|unparked, device, languages}`) and are unauthenticated
node-local control calls, like `/health` and `/v1/info`. Off CUDA they are
honest no-ops (`parked: false`). The process and the loaded weights stay alive
across a park, so unpark is a PCIe copy, not a model reload.

ASS only parks a backend with no in-flight jobs, so park never races a running
align. As a belt-and-braces guard, a job that somehow arrives while parked
brings the model back on-device before inference rather than running on CPU.

If `server.auth_token` is set, submission, polling and artifact download require
its Bearer token. Leave it empty for ASS: ASS does not forward backend auth.
Health and info are unauthenticated LAN diagnostics.

## Synchronous standalone operation

`POST /align` takes the same multipart `audio` and `params`, but waits and returns
HTTP 200 with the timing document directly. Duration is rounded to three decimal
places, as in the original endpoint. No job ID or result artifact is retained.
It shares the single inference worker, queue bound, upload limit and optional
Bearer authentication with `/v1/align`; HTTP health and polling stay responsive.

Unsupported languages return 422 with `error` and `supported` language codes;
audio decode errors return 415; inference/load failures return 500. Other
validation and queue errors match the async route. If a request is cancelled,
its queued/running work finishes and cleans up its temporary files and slot.
Clients must allow enough timeout for both queueing and inference.

Use `/align` when running the service standalone. Through ASS, use its async job
API so ASS can hold the GPU lease and harvest results before eviction. Do not
send direct standalone requests to an ASS-managed container.

`/models` is replaced by `/v1/info`. `/park` and `/unpark` are provided, so ASS
can use `evict = "park"` (or still `stop`, its choice). Run one uvicorn process
(no `--workers`): the worker queue and concurrency limit belong to that process.
