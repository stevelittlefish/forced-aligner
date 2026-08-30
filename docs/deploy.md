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

The base image is `nvidia/cuda:...ubuntu22.04`. Ubuntu 22.04 ships Python 3.10,
so the Dockerfile pulls **3.11 from the deadsnakes PPA** (we need 3.11+ for
stdlib `tomllib`) and points `python`/`pip` at it — don't "simplify" that back to
`apt-get install python3.11`, which fails on jammy.

Host state lives under `/srv/forced-aligner` — the config bind-mount and the
model cache both come from there. Create it once and drop the config in:

```sh
sudo mkdir -p /srv/forced-aligner/models
sudo cp config.example.toml /srv/forced-aligner/config.toml   # then edit it
```

Easiest is compose (GPU reservation, mounts and port are all wired in it):

```sh
docker compose up -d --build
docker compose logs -f                # first /align downloads the model (slow)
```

Or the raw equivalent:

```sh
docker build -t forced-aligner .
docker run --gpus '"device=0"' \
  -p 8830:8830 \
  -v /srv/forced-aligner/config.toml:/app/config.toml:ro \
  -v /srv/forced-aligner/models:/models \
  forced-aligner
```

The image `EXPOSE`s and the published port is **8830**; the app binds whatever
`[server].port` says in the mounted `config.toml`. Keep them equal (or change
both) or the container will listen on a port nothing is mapped to. The image has
a `HEALTHCHECK` hitting `/health`, so `docker ps` shows healthy/unhealthy once
it's up.

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
