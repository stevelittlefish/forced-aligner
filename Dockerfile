# CUDA runtime base so torch finds the GPU. Match the tag to the host's driver
# and keep it in step with the torch wheel index below (both cu121 here).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Python 3.11 is NOT in Ubuntu 22.04's default repos (jammy ships 3.10), and we
# need 3.11+ for stdlib tomllib — so pull it from deadsnakes. ffmpeg is
# whisperx's audio decoder; curl is only for the container HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Give 3.11 pip and make `python`/`pip` unambiguously point at it, so nothing
# accidentally runs against the base image's 3.10.
RUN python3.11 -m ensurepip --upgrade \
    && python3.11 -m pip install --upgrade pip \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Install the CUDA torch build first, on its own index, so the wheel matches
# cu121 rather than whatever whisperx would pull from PyPI.
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Configuration defaults make this image directly runnable by ASS. A TOML
# bind mount can override them; the library cache env is infrastructure only.
COPY app/ ./app/
COPY config.example.toml ./config.toml
# expandable_segments makes torch's caching allocator hand freed VRAM back to
# the driver instead of hoarding fragmented reserved segments. A long song
# spikes wav2vec2's attention into several GB of transient allocation; without
# this the reserved pool stays fat after the job and empty_cache() can't fully
# release it, so the shared GPU's co-tenants (Demucs, Stable Audio) never see
# the memory come back. Same class of env as the cache paths below: library
# wiring, not app config.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/cache/aligner/huggingface \
    TORCH_HOME=/cache/aligner/torch \
    HF_TOKEN_PATH=/cache/hf-token
RUN mkdir -p /cache/aligner /app/outputs

EXPOSE 8830
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8830/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8830"]
