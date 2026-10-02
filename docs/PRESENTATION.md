# Presentation Guide

How to present the malaria-detection project: slide outline, what to say,
a live-demo script, and answers to likely questions. Assumes ~10 minutes
plus demo.

## Slide outline (10 slides, ~8 min talking)

### 1. Title
**Lightweight Malaria Detection — a Dockerised ML service**
Your name, date, one line: "MobileNetV2 + FastAPI, built to run on a
2-core laptop and deployed to AWS EC2."

### 2. The problem
- Malaria kills ~600,000 people a year (WHO), mostly children under 5
- Diagnosis today = a technician counting cells under a microscope:
  slow, subjective, scarce in endemic regions
- Goal: a **lightweight, automated second reader** that runs on cheap
  hardware — no GPU, no cloud dependency

### 3. The constraints (what shaped every decision)
| Constraint | Design consequence |
|---|---|
| Intel i5-7200U, 2 cores / 4 threads | CPU-only PyTorch wheels, thread caps |
| 8 GB RAM (3.78 GiB in Docker) | Model must stay < 500 MB resident |
| No GPU | MobileNetV2, not a ViT (24× smaller than ViT-base) |

Say: *"I measured the machine first, then chose the stack to fit it."*

### 4. Tech stack
- **MobileNetV2** (2.2 M params, ~9 MB) — depthwise separable
  convolutions, made for mobile/CPU
- **PyTorch / torchvision** — no `transformers` dispatch layer needed
- **FastAPI** — async uploads, automatic OpenAPI docs
- **Docker** — identical environment on laptop and EC2
- **Hugging Face Hub** — pretrained weights, one-line download

### 5. Model & dataset
- Checkpoint: `Svetozar1993/LocalMedScan-malaria-mobilenetv2` (MIT),
  97.97% reported accuracy with 5-view TTA
- Dataset: `flame623/Malaria-classification` (6,020 train / 6,900 test)
- Our measured eval (500-image subset): **accuracy 0.974,
  precision 0.966, recall 0.926, F1 0.945**
- Point out recall > precision is the right trade-off: a missed
  infection is worse than a false alarm

### 6. System architecture
```
Browser (drag & drop) ──▶ FastAPI /predict ──▶ MobileNetV2 (CPU)
                              │
        train/ : download → fine-tune (progressive unfreezing) → evaluate
```
Endpoints: `GET /`, `/health`, `/model-info`, `POST /predict`
Upload guards: 10 MB cap, image-only, empty-file rejection

### 7. The interesting engineering problem ★ (best slide)
- **Symptom:** whole microscope fields were classified "Uninfected"
- **Root cause:** the model was trained on single-cell crops
  (~100–150 px); a 900×600 field is outside its training
  distribution → it sees mush and defaults to the majority class
- **Fix (`app/cells.py`):** every upload is analysed first
  - sparse field → adaptive segmentation (median background + Otsu) →
    classify each cell → parasitemia estimate
  - dense field → overlapping 224×224 patch grid → majority vote
  - small image → single-cell path with 5-view TTA
- **Result:** both MIT sample fields now correctly Parasitized
  (16/16 patches positive, mean P = 0.65 and 0.93)

### 8. Live demo (see script below)

### 9. Deployment
- Dockerised: 2.19 GB image, 322 MiB resident, healthy in Docker
- Deployed to EC2 t3.medium: scp → install Docker → swap file →
  `docker compose up --build -d` → open port 8000
- Full command-by-command walkthrough in `docs/EC2-DEPLOYMENT.md`

### 10. Limitations & future work
- Not a medical device — research/educational demo only
- Binary output; no species ID or parasitemia quantification
- Thresholds (`CELL_POSITIVE_THRESHOLD`, `DENSE_FIELD_COVERAGE`)
  could be calibrated on labelled full-field data
- Optional: watershed separation for denser smears, HTTPS reverse
  proxy on EC2, full NIH 27,558-image evaluation

## Live demo script (3–4 min)

Have ready: the web UI tab, the two MIT images, a terminal.

1. Open `http://<EC2_PUBLIC_IP>:8000/` — show the drag-and-drop UI
2. Upload a **single-cell crop** (from the dataset) → shows fast
   inference (~0.3–1.4 s), TTA on, probabilities
3. Upload **MIT-Malaria-Diagnosis-02.jpg** → Parasitized,
   mode "full-field (patch grid)", 16/16 patches, positivity 100%
4. Upload **MIT-Malaria-Diagnosis-01.jpg** → Parasitized, weaker
   signal (P≈0.65) — point out the API reports *how sure* it is
5. (Optional) `curl` the `/model-info` endpoint to show the OpenAPI
   contract

## Key lines to memorise

- "I measured the hardware before choosing the model — that's why
  MobileNetV2 and not a transformer."
- "The model wasn't wrong on the field images — it was being asked a
  question outside its training distribution. We analyse the image
  first and route it to the mode it was trained for."
- "Recall over precision: in screening, a false negative costs more
  than a false positive."
- "Everything is containerised, so the same image runs on my laptop
  and on EC2."

## Q&A prep

| Question | Answer |
|---|---|
| Why not a ViT or ResNet? | ViT-base is 24× larger and slow on CPU; MobileNetV2's depthwise convs are built for this. Accuracy on this task is already 0.97. |
| Why patch voting for dense fields? | Cells touch, so colour segmentation can't separate them; overlapping patches put a few cells per crop — close to the training distribution. |
| Is it production-ready? | No — research demo. No auth, no species ID, no clinician validation. Thresholds would need calibration on labelled field data. |
| How does TTA help? | Giemsa cells have no canonical orientation; averaging 5 rotation/flip views adds a small accuracy gain for 5× latency. |
| Why Docker? | Reproducibility: same Python/torch versions everywhere; resource caps (2 CPU, 3 GB) enforced by compose. |
| What did you learn? | Domain mismatch is a real failure mode; preprocessing/analysis matters as much as the model; hardware-aware design. |
