# Technology Stack

Every technology used in this project, why it was chosen, and how it fits the
low-spec target machine (Intel i5-7200U, 2 cores, 8 GB RAM, no GPU).

## Runtime & serving

| Technology | Version | Role | Why this choice |
|---|---|---|---|
| **Python** | 3.12 | Language runtime | torch 2.4.1 publishes wheels for 3.12; the host's 3.14 has none yet |
| **FastAPI** | 0.115 | HTTP API framework | Async file uploads, automatic OpenAPI/Swagger docs, Pydantic validation for free |
| **Uvicorn** | 0.30 | ASGI server | Standard FastAPI server; `[standard]` extra adds fast HTTP parsing |
| **Pydantic** | 2.x | Response schemas | Type-safe responses; rejects malformed model output early |
| **python-multipart** | 0.0.9 | Form parsing | Required by FastAPI to accept multipart image uploads |

## Machine learning

| Technology | Version | Role | Why this choice |
|---|---|---|---|
| **PyTorch** | 2.4.1 **CPU build** | Model inference + training | The CPU wheel index avoids ~2 GB of unusable CUDA libraries |
| **torchvision** | 0.19.1 | MobileNetV2 + transforms | Provides the exact architecture the checkpoint was trained with |
| **MobileNetV2** | — | Model backbone | ~2.2M params, ~9 MB, depthwise-separable convolutions → fast on 2 cores |
| **5-view TTA** | — | Accuracy booster | Rotation/flip averaging; cells are rotationally invariant |

`transformers` was deliberately **not** used: the chosen checkpoint is a native
torchvision state dict, so pulling in the full transformers library would add
weight with no benefit.

## Data

| Technology | Version | Role | Why this choice |
|---|---|---|---|
| **Hugging Face `datasets`** | 2.21 | Dataset download & caching | One-line load of a parquet-backed image dataset; caches locally |
| **`huggingface_hub`** | 0.24 | Model weight download | Official client; used both at image build time and at runtime |
| **Pillow** | 10.4 | Image decoding | Robust handling of arbitrary uploaded image formats |
| **NumPy** | 1.26 | Array interop | Tensor ↔ metric conversions |
| **scikit-learn** | 1.5 | Evaluation metrics | `accuracy_score`, `precision/recall/f1_score`, `confusion_matrix` |

### Datasets

| Dataset | Size | Splits | Used as |
|---|---|---|---|
| `flame623/Malaria-classification` | 150 MB | 6,020 train / 6,900 test | **Default** — fast to download and iterate on |
| `dpdl-benchmark/malaria` | 331 MB | 27,558 train | Optional full NIH set via `DATASET_ID` |

## Packaging & operations

| Technology | Role | Why this choice |
|---|---|---|
| **Docker** | Containerisation | Reproducible CPU-only runtime; no host pollution |
| **Docker Compose** | Orchestration | Declarative CPU/memory caps sized to the host |
| **`python:3.12-slim`** | Base image | Small, glibc-based, compatible with the CPU wheels |
| **libgomp1** | System lib | OpenMP runtime that PyTorch links against |
| **`HEALTHCHECK`** (curl) | Container health | Lets Docker/compose report readiness |
| **Non-root user** | Security | Container runs as `appuser` (uid 1000), not root |

## Testing

| Technology | Version | Role |
|---|---|---|
| **pytest** | 8.3 | Test runner |
| **Starlette TestClient** | (via FastAPI) | In-process HTTP calls without a live server |
| **httpx** | 0.27 | Underlying transport for TestClient |

## CPU-specific engineering decisions

These exist purely because of the 2-core / 3.78 GiB Docker limits:

1. **CPU-only PyTorch wheels** — removes ~2 GB of CUDA payload.
2. **Thread caps** — `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`,
   `TORCH_NUM_THREADS=2` and `torch.set_num_threads(2)` prevent oversubscription
   (4 threads on 2 cores causes context-switch thrashing).
3. **Batch size 8** — keeps peak memory well under the container limit.
4. **Subset training with early phase control** — defaults to 2,000 images and
   2 epochs so a run finishes in minutes.
5. **Frozen backbone first (progressive unfreezing)** — the classifier head is
   only 2,562 parameters, so the `head` phase trains in seconds.
6. **`num_workers=0`** — process-spawn overhead exceeds the benefit on 2 cores.
7. **Model pre-fetched at image build time** — the container needs no network at
   startup and boots fast.
