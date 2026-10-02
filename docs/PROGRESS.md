# Progress Log

Chronological record of how this project was built, including the reasoning
behind each decision and the hardware constraints that shaped it.

---

## 0. Hardware assessment (baseline)

Measured on the target machine before any code was written:

| Component | Value | Consequence |
|---|---|---|
| CPU | Intel Core i5-7200U, 2 cores / 4 threads @ 2.5 GHz | CPU-only inference and training |
| GPU | Intel HD 620 (integrated) | No CUDA; PyTorch CPU wheels only |
| RAM | 8 GB total, **3.78 GiB visible to Docker** | Keep resident model < 500 MB |
| Disk free | ~82 GB on `C:` | One small dataset is affordable |
| Docker | v29.7.2 (Docker Desktop / WSL2) | 4 CPUs, 3.78 GiB memory allocated |
| Python | 3.12.10 | Chosen interpreter (torch 2.4.1 has no 3.14 wheels) |

**Conclusion:** the design had to be inference-first with light fine-tuning on a
subset, not full-scale training.

---

## 1. Technology selection

- Chose **MobileNetV2** over ViT-base because it is ~24× smaller (14 MB vs
  330 MB) and dramatically faster on 2 CPU cores.
- Chose plain **PyTorch + torchvision** over `transformers` for the model, so the
  container does not carry a dispatch layer it does not need.
- Chose **FastAPI** for async image uploads and automatic OpenAPI docs.
- Verified every model and dataset id against the live Hugging Face Hub API
  before writing code (no guessed identifiers).

## 2. Scaffolding

Created `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`,
`.gitignore`. The Dockerfile installs PyTorch from the **CPU wheel index** to
avoid ~2 GB of unusable CUDA libraries, and caps `OMP/MKL/TORCH` thread counts
to 2 to match the physical core count.

## 3. Inference layer — `app/model.py`

- Loads MobileNetV2 with a 2-class head from a `model.pth` state dict.
- Weights resolution order: **local fine-tuned checkpoint → Hugging Face Hub**,
  so retraining immediately changes what the API serves.
- Added **5-view TTA** (4 rotations + horizontal flip) averaged at the softmax
  level; blood-cell morphology is rotationally invariant, so this is safe.
- `weights_only=True` when unpickling for safety.
- Process-wide singleton so the model loads once.

## 4. API + UI — `app/main.py`, `app/static/index.html`

- Endpoints: `GET /`, `GET /health`, `GET /model-info`, `POST /predict`.
- Startup is resilient: if the model fails to load, `/health` still answers and
  `/predict` returns 503 instead of crashing the container.
- Upload guards: empty-file, 10 MB size cap, and non-image rejection (400).
- Single-file HTML UI with drag-and-drop, a probability breakdown and a timing
  readout; includes an explicit "not a medical device" disclaimer.

## 5. Data pipeline — `train/dataset.py`, `train/download_data.py`

- Default dataset `flame623/Malaria-classification` (150 MB).
- Label normalisation maps the dataset's `ClassLabel` names onto the canonical
  order `(Parasitized, Uninfected)`; falls back to identity for integer-only
  datasets such as `dpdl-benchmark/malaria`.
- **Bug found and fixed during testing:** the test split is stored grouped by
  class, so a naive `select(range(n))` grabbed only one class. Subsampling now
  shuffles with a fixed seed (`_subsample`, seed=42) to stay representative and
  reproducible.

## 6. Fine-tuning — `train/fine_tune.py`

- Progressive unfreezing: `head` (classifier only) → `partial` (last 4 blocks)
  → `full`, each at a lower learning rate, mirroring the original authors' recipe.
- CPU-friendly defaults: 2000 train images, batch 8, 2 epochs, `num_workers=0`.
- Saves a plain state dict so it drops straight into `app/model.py`.

## 7. Evaluation — `train/evaluate.py`

- Accuracy + precision/recall/F1 for the *Parasitized* (positive) class, plus a
  confusion matrix, written to `reports/evaluation.json`.

## 8. Verification run (actual results on this machine)

Environment: Python 3.12.10, torch 2.4.1+cpu, torchvision 0.19.1+cpu.

| Check | Result |
|---|---|
| `py_compile` on all modules | pass |
| `pytest` (6 tests) | **6 passed** in ~44 s |
| `train/download_data.py` | 6,020 train (3,010 / 3,010) · 6,900 test (1,887 / 5,013) |
| `train/evaluate.py --max-samples 500` | **accuracy 0.974**, precision 0.966, recall 0.926, F1 0.945 |
| `train/fine_tune.py` smoke run (64 imgs, 1 epoch, head) | pass, checkpoint written |
| Local-checkpoint precedence | confirmed (`weights_source = local-checkpoint`) |

Raw baseline report: [`reports/evaluation.json`](../reports/evaluation.json).

## 9. Docker build & container verification

`docker build -t malaria-detect:latest .` succeeded. Running with
`--cpus=2 --memory=3g`, the service was exercised end to end:

| Check | Result |
|---|---|
| Image build | **success** |
| Image size | 2.19 GB (dominated by the PyTorch/`datasets` dependency stack, not the 9 MB model) |
| `GET /health` | `{"status":"ok","model_loaded":true}` |
| `GET /model-info` | `MobileNetV2`, 2,226,434 params, `device=cpu`, `torch_num_threads=2` |
| `POST /predict` (valid image) | `200` — valid JSON with probabilities summing to 1.0 |
| `POST /predict` (non-image) | `400` as designed |
| `GET /` (web UI) | `200` |
| Docker `HEALTHCHECK` | `healthy` |
| Memory while serving | **322 MiB / 3 GiB (10.5%)** |
| Latency (5-view TTA, 2 cores) | ~2.4 s cold, ~1.4 s warm |

The memory footprint is comfortable: the container uses roughly a tenth of its
allotted memory, so there is ample headroom for a locally fine-tuned checkpoint.

## 11. Full-field support — domain-mismatch fix

**Symptom:** the two MIT malaria-diagnosis sample images (900×600 whole
smear fields) were both classified *Uninfected*.

**Diagnosis:** single-cell dataset images classified correctly
(P=0.94–0.95), but the full fields returned the majority class at coin-flip
confidence (image 2: 0.53). Root cause: the model was trained on
~100–150 px single-cell crops, and a 900×600 field is far outside that
distribution. A first per-cell attempt (fixed saturation threshold) also
failed: real smear backgrounds are tinted, so everything merged into clumps
(mask coverage 39–54%, "cells" were multi-cell blobs clustered in one
corner).

**Fix (`app/cells.py`, numpy + Pillow only):**

- `analyze_field` — background-adaptive segmentation: per-image median
  background, Euclidean colour distance, Otsu binarisation, median +
  close morphology, connected components (plain flood fill, no scipy)
  with area/aspect filters. Works for *sparse* fields with separated
  cells → `field-cells` mode with a parasitemia estimate.
- `grid_patches` — overlapping 224×224 patch grid (25% overlap, ≤16
  patches, evenly subsampled) for *dense* fields where cells touch →
  `field-patches` mode. Validated on synthetic fields built from known
  dataset cells: uninfected field → 0/24 patches positive (max P=0.26);
  parasitized field → 23/24 positive (mean P=0.92). No out-of-domain
  bias toward "Parasitized".

**Result on the MIT samples:**

| Image | Mode | Result |
|---|---|---|
| MIT-Malaria-Diagnosis-01.jpg | field-patches | Parasitized, conf 0.78, 16/16 patches, mean P=0.646 |
| MIT-Malaria-Diagnosis-02.jpg | field-patches | Parasitized, conf 0.98, 16/16 patches, mean P=0.931 |

`predict()` now routes by image content (dense → patches, sparse → cells,
small → single-cell TTA). `schemas.py` and the web UI report the analysis
mode and per-unit statistics (cells/patches analyzed, positive units,
positivity estimate). Tests extended to 13 (segmentation, patch grid,
field-mode API).

## 12. Presentation & deployment docs

- `docs/EC2-DEPLOYMENT.md` — full EC2 launch walkthrough with
  every command explained (package → scp → install Docker →
  swap → `docker compose up --build -d` → security group →
  verify → teardown), a troubleshooting table and the
  "terminate the instance" billing warning.
- `docs/PRESENTATION.md` — 10-slide outline with a talk track,
  the domain-mismatch story as the centrepiece, a 3–4 min live
  demo script and Q&A prep.
- Both linked from the README documentation list.

## 13. Open items / next steps

- [ ] Run the full three-phase fine-tune (`--phase progressive`) overnight for a
      larger accuracy gain.
- [ ] Optionally evaluate against the full 27,558-image NIH set
      (`DATASET_ID=dpdl-benchmark/malaria`) for a publishable number.
- [ ] Optional slim-down: split the image into a runtime stage without the
      training-only packages (`datasets`, `pandas`, `pyarrow`) to cut ~1 GB.
- [ ] Optional: disable TTA (`USE_TTA=false`) to roughly halve inference latency
      if throughput matters more than the last fraction of a percent of accuracy.
- [ ] Calibrate `CELL_POSITIVE_THRESHOLD` / `DENSE_FIELD_COVERAGE` against
      labelled full-field data (e.g. the MIT dataset) for a better
      sensitivity/specificity trade-off than the current any-positive rule.
- [ ] Optional: add watershed separation to lift `field-cells` mode into
      denser smears (currently the patch grid covers that case).
