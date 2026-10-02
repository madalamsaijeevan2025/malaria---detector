# Model Details

## Summary

| Property | Value |
|---|---|
| Task | Binary image classification (blood-smear microscopy) |
| Classes | `Parasitized`, `Uninfected` |
| Architecture | MobileNetV2 (inverted residual blocks, depthwise separable convolutions) |
| Parameters | 2,226,434 (measured) |
| Weights file | ~9 MB (`model.pth`) |
| Input | RGB 224 × 224 |
| Output | 2 logits → softmax probabilities |
| Framework | PyTorch / torchvision |
| Runtime | CPU only |
| License | MIT |

## Provenance

| Field | Value |
|---|---|
| Source | Hugging Face Hub |
| Repository | `Svetozar1993/LocalMedScan-malaria-mobilenetv2` |
| Part of | LocalMedScan project (Svetozar Technologies) |
| Pretrained base | MobileNetV2, ImageNet weights |
| Training recipe | 3-phase progressive unfreezing |
| Reported accuracy | 97.97% (with 5-view TTA) on the NIH test split |

The checkpoint is a plain `torch.save` state dict and is loaded directly into
`torchvision.models.mobilenet_v2` with `classifier[1]` replaced by a
`Linear(1280, 2)` head.

## Class mapping

The checkpoint uses the following index order, which this project treats as
canonical everywhere:

| Index | Label |
|---|---|
| 0 | `Parasitized` |
| 1 | `Uninfected` |

`Parasitized` is therefore the positive class in all precision/recall/F1
reporting — it is the medically important one (a missed infection is worse than
a false alarm).

## Preprocessing

Inference uses deterministic resizing and ImageNet normalisation, matching the
model card's usage example:

```
Resize(224, 224) → ToTensor() → Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
```

Training additionally applies `RandomResizedCrop(0.7–1.0)`, `RandomRotation(90°)`,
`RandomHorizontalFlip` and mild `ColorJitter`. A full 90° rotation range is safe
because Giemsa-stained cells have no canonical orientation, and it was part of
the original training pipeline.

## Test-time augmentation (TTA)

Five views are evaluated and their softmax distributions averaged:

1. original
2. rotate 90°
3. rotate 180°
4. rotate 270°
5. horizontal flip

TTA costs ~5× inference time but typically adds a small accuracy gain. It is
enabled by default (`USE_TTA=true`) and can be disabled to roughly halve latency.

## Full-field (whole-smear) images

The checkpoint was trained on tightly cropped **single-cell** images
(~100–150 px, one cell per image). A full microscope field (hundreds of
cells) is far outside that distribution: fed directly, the network
downscales the whole field into mush and falls back to the majority class
("Uninfected", 73% of the test split). `app/cells.py` therefore analyses
every upload and routes it to the mode that fits:

| Mode | Trigger | How it works |
|---|---|---|
| `single-cell` | small image, or no stain structure found | whole-image classification with TTA (above) |
| `field-cells` | ≥ `FIELD_MIN_CELLS` separated cells | background-adaptive segmentation finds individual cells; each crop is classified and the votes aggregated |
| `field-patches` | stain coverage ≥ `DENSE_FIELD_COVERAGE` | a grid of overlapping 224×224 patches is classified and the votes aggregated |

Both field modes use a **high-sensitivity screening rule**: the field is
called *Parasitized* when **any** cell/patch tests positive
(`CELL_POSITIVE_THRESHOLD`, default 0.5), because a missed infection is
worse than a false alarm. The response also reports the fraction of positive
units (`parasitemia_estimate` / `field_positivity`) so borderline cases stay
interpretable.

The segmentation adapts to the background per image (median colour +
Otsu binarisation of the colour-distance map) because real smear backgrounds
are often tinted pale pink/lavender, not pure white — a fixed saturation
threshold merges everything into clumps. Only numpy and Pillow are used, so
the Docker image gains no dependency.

Measured on the MIT malaria-diagnosis sample fields (900×600, dense):

| Image | Mode | Patches positive | Mean P(Parasitized) | Verdict |
|---|---|---|---|---|
| MIT-Malaria-Diagnosis-01.jpg | field-patches | 16/16 | 0.646 | Parasitized (conf. 0.78) |
| MIT-Malaria-Diagnosis-02.jpg | field-patches | 16/16 | 0.931 | Parasitized (conf. 0.98) |

The patch approach was validated on synthetic fields built from known
dataset cells: an uninfected field gave 0/24 patches positive (max P=0.26)
and a parasitized field 23/24 positive (mean P=0.92) — the aggregation has
no out-of-domain bias toward "Parasitized".

Latency on the i5-7200U: ~1.4–1.7 s per full field (16 patches in one
batched forward pass).

Knobs (all env-overridable): `FIELD_MIN_CELLS`, `FIELD_MAX_CELLS`,
`DENSE_FIELD_COVERAGE`, `FIELD_PATCH_OVERLAP`, `FIELD_MAX_PATCHES`,
`CELL_POSITIVE_THRESHOLD`.

## Measured performance (this machine)

Command: `python train/evaluate.py --max-samples 500`
Weights: pretrained checkpoint from the Hub (no additional fine-tuning).

| Metric | Value |
|---|---|
| Accuracy | **0.974** |
| Precision (Parasitized) | 0.966 |
| Recall (Parasitized) | 0.926 |
| F1 (Parasitized) | 0.945 |
| Confusion matrix (rows = true, cols = pred, order `[Parasitized, Uninfected]`) | `[[112, 9], [4, 375]]` |

Interpretation: of 500 randomly sampled test cells, the model missed 9
parasitized cells (false negatives) and raised 4 false alarms.

> Results come from a 500-image random subset, not the full test split, so they
> carry a margin of error. Run without `--max-samples` for the full number (slow
> on 2 CPU cores).

## Limitations & ethical notes

- **Not a medical device.** This is a research/educational demo. It must not be
  used for clinical diagnosis or treatment decisions.
- Trained on Giemsa-stained thin blood smears; performance on thick smears,
  other stains, or different microscopes is unknown.
- Binary output only — it does **not** identify *Plasmodium* species or
  parasitaemia level, which matter clinically.
- The dataset is a single curated source; real-world domain shift is likely.
- Class balance differs between splits (train is balanced, test is 27%
  parasitized), so accuracy alone is misleading — always read recall/F1.

## Retraining

See `docs/SETUP.md` §4. Any checkpoint produced by `train/fine_tune.py` is picked
up automatically by the API because weight resolution prefers a local file over
the Hub download.
