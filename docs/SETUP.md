# Setup Guide

Complete instructions to run, train and test the malaria detection project.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker v20+ (Docker Desktop on Windows) | For the containerised route |
| Python 3.12 + pip | Only for the local (non-Docker) route |
| ~2 GB free disk | Image (~1 GB) + dataset cache (~150 MB) |
| Internet (first run only) | To download model weights and dataset |

On this machine Docker is configured with **2 CPUs and 3.78 GiB memory**; the
compose file caps the container at 2 CPUs / 3 GB to match.

---

## 1. Run with Docker (recommended)

From the project root:

```bash
cd malaria-detect
docker compose up --build
```

First build downloads the CPU PyTorch wheels (~200 MB) and pre-fetches the model
weights, so it takes several minutes. Subsequent starts are fast.

Once running:

- Web UI → <http://localhost:8000/>
- Interactive API docs → <http://localhost:8000/docs>
- Health check → <http://localhost:8000/health>

Stop with `Ctrl+C`, or `docker compose down`.

### Plain Docker (without Compose)

```bash
docker build -t malaria-detect:latest .
docker run --rm -p 8000:8000 --cpus=2 --memory=3g malaria-detect:latest
```

---

## 2. Run locally (no Docker)

```bash
cd malaria-detect
python -m venv .venv
# Windows (Git Bash)
.venv/Scripts/python.exe -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -r requirements.txt
# Linux/macOS
# .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
#   --extra-index-url https://pypi.org/simple -r requirements.txt

.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> The CPU wheel index matters: it installs `torch==2.4.1+cpu` instead of the
> ~2 GB CUDA build this machine cannot use.

---

## 3. Call the API

### Health

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}
```

### Model info

```bash
curl http://localhost:8000/model-info
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/blood_cell.png"
```

Example response:

```json
{
  "label": "Parasitized",
  "confidence": 0.9987,
  "probabilities": { "Parasitized": 0.9987, "Uninfected": 0.0013 },
  "inference_ms": 312.4,
  "tta": true
}
```

Or simply open <http://localhost:8000/> and drag an image in.

---

## 4. Download data, fine-tune and evaluate

All scripts work both as `python train/<script>.py` and `python -m train.<script>`.

### 4a. Download and cache the dataset

```bash
.venv/Scripts/python.exe train/download_data.py
```

To use the full NIH set instead of the small default:

```bash
DATASET_ID=dpdl-benchmark/malaria .venv/Scripts/python.exe train/download_data.py
```

### 4b. Fine-tune

Quick run (head only, 2,000 images, 2 epochs — minutes on this CPU):

```bash
.venv/Scripts/python.exe train/fine_tune.py
```

Full progressive run (head → partial → full — slow, best done overnight):

```bash
.venv/Scripts/python.exe train/fine_tune.py --phase progressive --epochs 3 --max-train-samples 6000
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--phase` | `head` | `head`, `partial`, `full` or `progressive` |
| `--epochs` | `2` | Epochs per phase |
| `--batch-size` | `8` | Lower if you hit memory limits |
| `--max-train-samples` | `2000` | Subset size (shuffled, seed 42) |
| `--max-eval-samples` | `500` | Eval subset size |
| `--lr` | per-phase | Override the learning rate |
| `--output` | `models/malaria_mobilenetv2.pth` | Where to save the state dict |
| `--threads` | `2` | PyTorch CPU threads |

Output: `models/malaria_mobilenetv2.pth`. The API automatically prefers this
file over the Hub download on the next start.

### 4c. Evaluate

```bash
.venv/Scripts/python.exe train/evaluate.py --max-samples 500
```

Writes `reports/evaluation.json` and prints accuracy, precision, recall, F1 for
`Parasitized`, and a confusion matrix. Omit `--max-samples` for the full split.

### 4d. Serve your fine-tuned model

```bash
MODEL_PATH=models/malaria_mobilenetv2.pth .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
```

With Docker, the `./models` directory is bind-mounted to `/app/models`, so set
`MODEL_PATH=/app/models/malaria_mobilenetv2.pth` (see `docker-compose.yml`).

---

## 5. Run the tests

```bash
.venv/Scripts/python.exe -m pytest -v
```

Six tests cover health, model info, the prediction contract, bad-input handling
and the UI route. Prediction tests auto-skip if the weights cannot be fetched
(e.g. offline).

---

## Configuration reference

All settings are environment variables — no code changes needed.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_REPO` | `Svetozar1993/LocalMedScan-malaria-mobilenetv2` | Hub repo for weights |
| `MODEL_FILENAME` | `model.pth` | File within the repo |
| `MODEL_PATH` | `models/malaria_mobilenetv2.pth` | Local checkpoint (takes precedence) |
| `USE_TTA` | `true` | Enable 5-view TTA |
| `TORCH_NUM_THREADS` | `2` | PyTorch CPU threads |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | `2` | Math library threads |
| `DATASET_ID` | `flame623/Malaria-classification` | Training dataset |
| `DATASET_CACHE` | `.cache/datasets` | Dataset cache location |
| `HF_HOME` | image: `/app/.cache/huggingface` | Hugging Face cache |
| `LOG_LEVEL` | `INFO` | API log verbosity |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker compose` build runs out of memory | Lower the compose `mem_limit`, or build with `docker build --memory=3g` |
| `/predict` returns 503 | Weights failed to load — check container logs; ensure network on first run |
| `/predict` returns 400 | The uploaded file is not a decodable image |
| Fine-tune is very slow | Reduce `--max-train-samples`, keep `--phase head`, reduce `--epochs` |
| WSL/Windows symlink warning from `huggingface_hub` | Harmless; set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence it |
| Port 8000 already in use | Change the host port in `docker-compose.yml` (`"8080:8000"`) |
