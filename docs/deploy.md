# Deploying on the AI server

One container, GPU-backed. Same neighbourhood as Demucs and Whisper.

## torch and CUDA

The one fiddly part. `torch` must match the host's CUDA. The Dockerfile installs
the `cu121` wheel against a `cuda:12.1.1` base — **change both together** if the
host driver differs. `whisperx` and its wav2vec2 stack sit on top of that torch.

Verify inside the container:

```sh
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

`False` means the torch wheel and the base image (or the host driver) disagree.

## Config

No environment variables — mount a `config.toml`:

```sh
cp config.example.toml config.toml   # then edit device/port/auth
```

Set `[align].device = "cuda"` and `compute_type = "float16"` on the server.

## Model cache

Models download from Hugging Face on first use (per language) and are cached.
Persist that cache across restarts or every restart re-downloads:

- set `[align].model_cache_dir = "/models"` in config, and
- mount a volume at `/models`.

## Run

```sh
docker build -t forced-aligner .
docker run --gpus all \
  -p 8830:8830 \
  -v "$PWD/config.toml:/app/config.toml:ro" \
  -v aligner-models:/models \
  forced-aligner
```

Then from another LAN host:

```sh
curl -s http://<host>:8830/health | jq
```

The first `/align` for a given language is slow (model download + load); after
that the model is warm. `/models` shows what's loaded.

## Placing it in SERVERS.md

Once it's up, add it to the sing thing's `SERVERS.md` with its real endpoint and
the `/align` contract, so the karaoke app knows where to find it. The client
shape is the same submit-multipart pattern used for Demucs.
