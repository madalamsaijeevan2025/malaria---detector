# 🩸 Malaria Detection — Dockerised Lightweight ML Service

A CPU-friendly malaria parasite detector: a **MobileNetV2** classifier (9 MB,
2.2M parameters) served through **FastAPI** with a drag-and-drop web UI, fully
containerised and fine-tunable on a low-spec machine.

> ⚠️ **Research/educational demo only — not a medical device.** Do not use for
> clinical diagnosis.

---

## What it does

Upload a Giemsa-stained thin blood-smear image and get a binary
classification — **Parasitized** or **Uninfected** — with calibrated
probabilities and inference timing. Works on **single-cell crops** and on
**whole microscope fields**: full-field images are analysed automatically
(per-cell segmentation or a patch grid) and report a positivity estimate.

```bash
curl -X POST http://localhost:8000/predict -F "file=@cell.png"
```
```json
{
  "label": "Parasitized",
  "confidence": 0.9987,
  "probabilities": { "Parasitized": 0.9987, "Uninfected": 0.0013 },
  "inference_ms": 312.4,
  "tta": true
}
```

Whole-field example (dense smear → patch-grid mode):
```json
{
  "label": "Parasitized",
  "confidence": 0.98,
  "probabilities": { "Parasitized": 0.931, "Uninfected": 0.069 },
  "mode": "field-patches",
  "patches_analyzed": 16,
  "positive_patches": 16,
  "field_positivity": 1.0
}
```

## Quickstart

```bash
cd malaria-detect
docker compose up --build
```

Then open <http://localhost:8000/>.

Without Docker? See **[docs/SETUP.md](docs/SETUP.md)**.

## Model at a glance

| | |
|---|---|
| Architecture | MobileNetV2 (torchvision) |
| Parameters / size | 2,226,434 / ~9 MB |
| Source | `Svetozar1993/LocalMedScan-malaria-mobilenetv2` (MIT) |
| Input | RGB 224×224 |
| Classes | `Parasitized`, `Uninfected` |
| Measured accuracy | 0.974 (500-image subset, with 5-view TTA) |
| Runtime | CPU only, 2 threads |

## Dataset

| Dataset | Size | Role |
|---|---|---|
| `flame623/Malaria-classification` | 150 MB (6,020 train / 6,900 test) | Default |
| `dpdl-benchmark/malaria` | 331 MB (27,558 images) | Full NIH set, optional |

## Project layout

```
malaria-detect/
├── app/
│   ├── main.py            FastAPI routes + lifespan
│   ├── model.py           MobileNetV2 loader, TTA, predict()
│   ├── cells.py           Full-field analysis (segmentation + patch grid)
│   ├── schemas.py         Pydantic response models
│   └── static/index.html  Web UI
├── train/
│   ├── dataset.py         Dataset loading, label normalisation, transforms
│   ├── download_data.py   Fetch + cache the dataset
│   ├── fine_tune.py       Progressive-unfreezing training (CPU-tuned)
│   └── evaluate.py        Metrics + confusion matrix → JSON report
├── tests/
│   ├── test_api.py        API endpoint tests
│   └── test_cells.py      Field-analysis tests
├── docs/
│   ├── PROGRESS.md        Build log with measured results
│   ├── SETUP.md           Full setup & usage guide
│   ├── MODEL.md           Model details, metrics, limitations
│   └── TECH_STACK.md      Every technology and why
├── Dockerfile             CPU-only image, threads capped to 2
├── docker-compose.yml     2 CPUs / 3 GB limits
└── requirements.txt
```

## Common commands

```bash
docker compose up --build              # run the service
python train/download_data.py          # fetch the dataset
python train/fine_tune.py              # quick head-only fine-tune
python train/evaluate.py --max-samples 500
python -m pytest -v                    # run tests
```

## Hardware note

Built for an **Intel i5-7200U (2 cores), 8 GB RAM, no GPU**. Every design choice
— CPU-only torch wheels, 2-thread caps, batch size 8, frozen-backbone-first
training, subset defaults — exists to run comfortably inside Docker's 3.78 GiB
memory limit. Details in [docs/TECH_STACK.md](docs/TECH_STACK.md).

## Documentation

- **[docs/SETUP.md](docs/SETUP.md)** — install, run, train, test, troubleshoot
- **[docs/MODEL.md](docs/MODEL.md)** — architecture, provenance, metrics, limits
- **[docs/TECH_STACK.md](docs/TECH_STACK.md)** — technologies and rationale
- **[docs/PROGRESS.md](docs/PROGRESS.md)** — build log and verification results
- **[docs/EC2-DEPLOYMENT.md](docs/EC2-DEPLOYMENT.md)** — deploying to AWS EC2, every command explained
- **[docs/PRESENTATION.md](docs/PRESENTATION.md)** — slide outline, talk track and demo script
