# Deploy through ASS

The image defaults to CUDA on port 8830, English preloading, one resident
language model, and stop eviction. Application settings are TOML; the image
sets HF_HOME, TORCH_HOME and HF_TOKEN_PATH only for third-party cache wiring.
No custom config mount is required. Override `/app/config.toml` if needed.

## Build and release

```sh
docker build -t ghcr.io/stevelittlefish/forced-aligner:latest .
# Or publish through the tag-triggered GitHub Actions container workflow:
./make_release.sh v0.2.0 "ASS job API and bounded language residency"
```

The workflow builds linux/amd64 and publishes `latest` plus version tags to
GHCR. Make the GHCR package accessible to the server before pulling. The
Dockerfile retains the existing CUDA 12.1 / Python 3.11 base and dependency
installation; building and real GPU inference must be verified on the server.

## ASS registration

The ASS repository's `ass.toml` contains `[services.aligner]` with `verb = "align"`,
port 8830, and the image above. It mounts `/srv/ass/cache:/cache`:

- WhisperX downloads: `/cache/aligner/models`.
- HF cache: `/cache/aligner/huggingface`.
- torch cache: `/cache/aligner/torch`.
- Shared HF token file: `/cache/hf-token`, if needed.

The scratch results live at `/app/outputs` without a host mount. ASS downloads
all artifacts before removing the container. Do not run the standalone compose
service alongside ASS; ASS owns the container and GPU scheduling.

Initial **unmeasured** reservations are 12,000 MiB pinned VRAM and 6,000 MiB RAM,
with stop eviction (zero parked reservation). The VRAM estimate allows headroom
over a historical roughly 9.9 GB reserved-memory observation on a ten-minute
track. It is not a guarantee for arbitrarily long audio or every language.
The RAM estimate covers model loading, decoded audio and alignment scratch;
measure and adjust both on representative tracks. Only one language is cached.

## Server acceptance

1. Release/build the image, then run ASS's `./pull-services.sh` on the GPU host.
2. Submit a short vocal WAV with known lyrics through ASS:

   ```sh
   curl -s -F 'audio=@vocals.wav' \
     -F 'params={"text":"Hello world","language":"en"}' \
     http://localhost:2645/v1/aligner/jobs
   ```

3. Poll `/v1/jobs/{id}`, download `/v1/jobs/{id}/result/alignment.json`, and check
   the timings, language, duration and unresolved words.
4. Run a long track and switch language. Inspect `/v1/backends/aligner/info`
   and measure driver VRAM and process RAM to replace the estimated reservations.
5. Force eviction by running another large backend. The old alignment result
   must still download from ASS. Run alignment again and confirm caches prevent
   another weight download.

For standalone debugging, `docker compose up --build` uses the same cache
mount. Remove it before letting ASS manage the service. The image's uvicorn CMD
binds port 8830; changing ports requires overriding CMD as well as ASS config.
