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

COPY app/ ./app/
COPY config.example.toml .

# HF cache lives here; mount a volume at /models and point
# [align].model_cache_dir at it so models download once (see docs/deploy.md).
ENV HF_HOME=/models
RUN mkdir -p /models

# config.toml is provided at run time (mount or copy in). The app reads host/port
# from it, but the process must bind the same port this image exposes: 8830. Keep
# [server].port = 8830 in the mounted config, or override the CMD.
EXPOSE 8830

# Fails the container's health status if the app isn't serving. --start-period
# gives uvicorn + first model import time to come up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8830/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8830"]
