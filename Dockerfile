# ---------------------------------------------------------------------------
# Malaria detection API - CPU-only image tuned for low-spec hardware
# ---------------------------------------------------------------------------
# python:3.12-slim keeps the base small; PyTorch ships CPU-only wheels because
# the CUDA build would add ~2 GB this machine (i5-7200U, no GPU) cannot use.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Limit math libraries to the 2 physical cores available to the container.
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TORCH_NUM_THREADS=2 \
    # Keep the Hugging Face cache inside the image so it works offline.
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Minimal system packages:
#  - libgomp1 : OpenMP runtime required by PyTorch
#  - curl     : used by the HEALTHCHECK below
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies. The CPU wheel index is primary so torch/torchvision
# come from download.pytorch.org; everything else falls back to PyPI.
COPY requirements.txt .
RUN pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt

# Application code
COPY app ./app
COPY train ./train

# Pre-fetch the pretrained weights at build time so the container starts fast
# and does not need network access at runtime.
RUN python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='Svetozar1993/LocalMedScan-malaria-mobilenetv2', filename='model.pth')"

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
