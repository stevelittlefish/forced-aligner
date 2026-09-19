# forced-aligner

An ASS backend that aligns known lyrics to a vocal audio clip using WhisperX's
wav2vec2 alignment engine. It returns word and line timings, not a transcription.
Unresolved words keep `null` timestamps so callers can flag them for correction.

## ASS API

- `GET /health` — readiness; configured preloads finish before serving.
- `GET /v1/info` — model, loaded language, capabilities, and VRAM telemetry.
- `POST /v1/align` — multipart `audio` and `params` JSON (`text`, optional
  `language`); returns HTTP 202 with `{job_id, state: "queued", artifacts: []}`.
- `GET /v1/jobs/{id}` — queued/running/succeeded/failed, error and artifacts.
- `GET /v1/jobs/{id}/result/alignment.json` — the completed timing document.

Standalone clients can use synchronous `POST /align` with the same multipart
fields and receive the timing JSON directly (HTTP 200). It waits its turn on the
same serial worker; allow enough client timeout for queueing and inference.
Scratch files are removed when it finishes, even if the caller stops waiting.
`/models` remains replaced by `/v1/info`. ASS clients
should submit to ASS at `/v1/aligner/jobs`, poll `/v1/jobs/{id}`, and download
`/v1/jobs/{id}/result/alignment.json`. See [the contract](docs/spec.md).

## Residency and jobs

One serial inference worker keeps HTTP responsive and prevents overlapping GPU
jobs. Only one language model is resident: switching languages unloads the old
model before loading the new one. Disk caches retain downloaded weights.

ASS uses `evict = "stop"`; there are no park/unpark endpoints. ASS harvests the
JSON before removing the container. Jobs and artifacts live in the container's
scratch directory, with uploads removed after completion or failure. A process
restart marks interrupted jobs failed; completed results survive while their
scratch directory exists. There is no result expiry that could race harvesting.

## Development

Python 3.11+. The HTTP tests mock alignment and need no GPU or torch:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
ruff check . && python -m compileall app && pytest -q
```

For actual alignment, install `requirements.txt`, copy `config.example.toml` to
`config.toml`, and adjust the device and writable cache/job paths. `./run.sh`
reads the host/port from that TOML file. Models download on first use.

See [deployment](docs/deploy.md) for image release and ASS configuration, and
[alignment notes](docs/alignment.md) for the algorithm's limits with singing.
