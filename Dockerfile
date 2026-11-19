# AZZANI Image Enhancements — Production GPU / Headless Server Container (MIT License)
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    AZZANI_CACHE_DIR=/models/cache \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY azzani_enhance ./azzani_enhance

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision \
    && /opt/venv/bin/pip install --no-deps -e . \
    && /opt/venv/bin/pip install "spandrel>=0.4.0" "opencv-python-headless>=4.8.0" "pillow>=10.0.0" "numpy>=1.26.0" "scipy>=1.11.0" "safetensors>=0.4.0" "rich>=13.0.0" "huggingface-hub>=0.23.0"

RUN mkdir -p /data/inbox /data/output /models/hf /models/cache
VOLUME ["/data/inbox", "/data/output", "/models"]

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/healthz || exit 1

ENTRYPOINT ["azzani-enhance"]
CMD ["--server", "--host", "0.0.0.0", "--port", "8765", "--models-dir", "/models"]
