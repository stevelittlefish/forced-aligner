# CUDA runtime base so torch finds the GPU. Match the tag to the host's driver.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# whisperx wants ffmpeg for audio decoding; python3.11 for stdlib tomllib.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CUDA torch build first, then the rest, so the wheel matches cu121.
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY config.example.toml .

# config.toml is provided at run time (mount or copy). Persist the HF cache to a
# volume so models download once — see docs/deploy.md.
EXPOSE 8830
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8830"]
