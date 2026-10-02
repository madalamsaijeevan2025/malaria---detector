"""Model loading and inference for malaria parasite detection.

Backbone
--------
MobileNetV2 (~3.4M parameters, ~14 MB) fine-tuned for binary classification of
Giemsa-stained thin blood-smear cell images into ``Parasitized`` / ``Uninfected``.

Weights
-------
By default the weights are pulled from the Hugging Face Hub repo
``Svetozar1993/LocalMedScan-malaria-mobilenetv2`` (MIT licensed, 97.97% reported
test accuracy on the NIH malaria dataset). If a locally fine-tuned checkpoint
exists at ``MODEL_PATH`` it takes precedence, so training with
``train/fine_tune.py`` immediately improves what the API serves.

CPU notes
---------
The target machine is a 2-core i5-7200U, so this module caps the PyTorch thread
pool and keeps everything on CPU. A 5-view test-time augmentation (TTA) is
enabled by default because it buys accuracy at a small latency cost that is
still acceptable for a single-image request.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from app.cells import (
    CellCrop,
    FieldSegmentation,
    analyze_field,
    grid_patches,
)

logger = logging.getLogger(__name__)

# --- Configuration (env-overridable) ---------------------------------------
MODEL_REPO = os.getenv("MODEL_REPO", "Svetozar1993/LocalMedScan-malaria-mobilenetv2")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model.pth")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "malaria_mobilenetv2.pth"))
NUM_CLASSES = 2
LABELS: List[str] = ["Parasitized", "Uninfected"]
INPUT_SIZE = 224
USE_TTA = os.getenv("USE_TTA", "true").lower() in {"1", "true", "yes"}
NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "2"))

# Full-field (whole-smear) support.  The model was trained on
# single-cell crops, so a full field is never classified as one
# image.  Instead the field is analysed and routed to the mode
# that fits:
#   * dense field (stain coverage >= DENSE_FIELD_COVERAGE) ->
#     a grid of overlapping patches, classified and aggregated;
#   * sparse field (>= FIELD_MIN_CELLS separated cells) ->
#     cell-by-cell classification with a parasitemia estimate;
#   * otherwise the image is treated as a single-cell crop.
FIELD_MIN_CELLS = int(os.getenv("FIELD_MIN_CELLS", "3"))
FIELD_MAX_CELLS = int(os.getenv("FIELD_MAX_CELLS", "10"))
DENSE_FIELD_COVERAGE = float(os.getenv("DENSE_FIELD_COVERAGE", "0.12"))
FIELD_PATCH_SIZE = INPUT_SIZE
FIELD_PATCH_OVERLAP = float(os.getenv("FIELD_PATCH_OVERLAP", "0.25"))
FIELD_MAX_PATCHES = int(os.getenv("FIELD_MAX_PATCHES", "16"))
CELL_POSITIVE_THRESHOLD = float(os.getenv("CELL_POSITIVE_THRESHOLD", "0.5"))

# ImageNet normalisation stats - the preprocessing used when the pretrained
# model was fine-tuned (see the model card's usage snippet).
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _build_transform() -> transforms.Compose:
    """Deterministic preprocessing used for both inference and evaluation."""
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )


def build_network() -> nn.Module:
    """MobileNetV2 with a fresh 2-class classifier head."""
    net = models.mobilenet_v2(weights=None)
    net.classifier[1] = nn.Linear(net.last_channel, NUM_CLASSES)
    return net


def resolve_weights() -> tuple[str, str]:
    """Return ``(path, source)`` for the weights that should be loaded.

    A locally fine-tuned checkpoint wins over the Hub download so retraining
    has an immediate effect on the served model.
    """
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH, "local-checkpoint"
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    return path, "huggingface-hub"


def _tta_views(image: Image.Image) -> List[Image.Image]:
    """Five rotation/flip views that exploit blood-cell rotational invariance."""
    return [
        image,
        image.transpose(Image.ROTATE_90),
        image.transpose(Image.ROTATE_180),
        image.transpose(Image.ROTATE_270),
        image.transpose(Image.FLIP_LEFT_RIGHT),
    ]


@dataclass
class MalariaClassifier:
    """Thin wrapper around the network that exposes a clean ``predict`` API."""

    model: nn.Module
    transform: transforms.Compose = field(default_factory=_build_transform)
    labels: List[str] = field(default_factory=lambda: list(LABELS))
    source: str = "unknown"
    tta: bool = USE_TTA

    @torch.inference_mode()
    def predict(
        self, image: Image.Image, use_tta: Optional[bool] = None
    ) -> Dict[str, object]:
        """Classify a PIL image in the mode that fits its content.

        The image is analysed first (see :mod:`app.cells`):

        * dense full fields (high stain coverage) -> a grid of
          overlapping patches, each classified and aggregated
          (``mode="field-patches"``);
        * sparse fields with separated cells -> cell-by-cell
          classification with a parasitemia estimate
          (``mode="field-cells"``);
        * everything else (single-cell crops) -> whole-image
          classification with TTA, as before.

        Returns a dict with the predicted ``label``, its ``confidence``,
        the full ``probabilities`` map, ``inference_ms``, the ``mode``
        tag and, for field images, per-unit statistics.
        """
        started = time.perf_counter()
        segmentation: FieldSegmentation = analyze_field(
            image, max_cells=FIELD_MAX_CELLS
        )
        # Patch/cell analysis only makes sense for images larger
        # than a single training crop.
        field_sized = max(image.width, image.height) > FIELD_PATCH_SIZE

        if field_sized and segmentation.coverage >= DENSE_FIELD_COVERAGE:
            return self._predict_patches(image, started)
        if field_sized and len(segmentation.crops) >= FIELD_MIN_CELLS:
            return self._predict_field(segmentation.crops, started)
        return self._predict_single(image, use_tta, started)

    def _predict_single(
        self,
        image: Image.Image,
        use_tta: Optional[bool],
        started: float,
    ) -> Dict[str, object]:
        """Whole-image classification (single-cell crops)."""
        apply_tta = self.tta if use_tta is None else use_tta
        views = _tta_views(image) if apply_tta else [image]
        batch = torch.stack([self.transform(view.convert("RGB")) for view in views])

        logits = self.model(batch)
        # Average the softmax distributions across the augmented views.
        probs = torch.softmax(logits, dim=1).mean(dim=0)

        index = int(torch.argmax(probs).item())
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return {
            "label": self.labels[index],
            "confidence": float(probs[index].item()),
            "probabilities": {
                label: float(probs[i].item()) for i, label in enumerate(self.labels)
            },
            "inference_ms": round(elapsed_ms, 2),
            "tta": apply_tta,
            "mode": "single-cell",
        }

    def _predict_field(
        self, crops: List[CellCrop], started: float
    ) -> Dict[str, object]:
        """Cell-by-cell classification of a sparse full field.

        Each cell crop is one forward pass (batched into a single
        call).  The field is called Parasitized when *any* cell tests
        positive -- a high-sensitivity screening rule -- and the
        parasitemia estimate (fraction of positive cells) is reported
        alongside so borderline cases stay interpretable.
        """
        batch = torch.stack([self.transform(crop.image) for crop in crops])
        cell_probs = torch.softmax(self.model(batch), dim=1)
        p_par = cell_probs[:, 0]

        positives = p_par >= CELL_POSITIVE_THRESHOLD
        n_cells = len(crops)
        n_positive = int(positives.sum().item())
        mean_probs = cell_probs.mean(dim=0)

        if n_positive > 0:
            label = self.labels[0]  # Parasitized
            confidence = float(p_par.max().item())
        else:
            label = self.labels[1]  # Uninfected
            confidence = float(cell_probs[:, 1].max().item())

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                label: float(mean_probs[i].item())
                for i, label in enumerate(self.labels)
            },
            "inference_ms": round(elapsed_ms, 2),
            "tta": False,
            "mode": "field-cells",
            "cells_analyzed": n_cells,
            "parasitized_cells": n_positive,
            "parasitemia_estimate": round(n_positive / max(n_cells, 1), 4),
        }

    def _predict_patches(
        self, image: Image.Image, started: float
    ) -> Dict[str, object]:
        """Patch-based classification of a dense full field.

        Dense fields cannot be split into single cells by colour
        thresholding (the cells touch), so the field is covered by
        overlapping patches that each contain a few cells.  Every
        patch is classified in one batched forward pass; the field
        is called Parasitized when *any* patch tests positive
        (high-sensitivity screening), with the field-level
        positivity fraction reported alongside.
        """
        patches = grid_patches(
            image,
            patch_size=FIELD_PATCH_SIZE,
            overlap=FIELD_PATCH_OVERLAP,
            max_patches=FIELD_MAX_PATCHES,
        )
        batch = torch.stack([self.transform(p) for p in patches])
        patch_probs = torch.softmax(self.model(batch), dim=1)
        p_par = patch_probs[:, 0]

        positives = p_par >= CELL_POSITIVE_THRESHOLD
        n_patches = len(patches)
        n_positive = int(positives.sum().item())
        mean_probs = patch_probs.mean(dim=0)

        if n_positive > 0:
            label = self.labels[0]  # Parasitized
            confidence = float(p_par.max().item())
        else:
            label = self.labels[1]  # Uninfected
            confidence = float(patch_probs[:, 1].max().item())

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "label": label,
            "confidence": confidence,
            "probabilities": {
                label: float(mean_probs[i].item())
                for i, label in enumerate(self.labels)
            },
            "inference_ms": round(elapsed_ms, 2),
            "tta": False,
            "mode": "field-patches",
            "patches_analyzed": n_patches,
            "positive_patches": n_positive,
            "field_positivity": round(n_positive / max(n_patches, 1), 4),
        }


def load_model() -> MalariaClassifier:
    """Build the network, load the best available weights, and set eval mode."""
    torch.set_num_threads(NUM_THREADS)

    weights_path, source = resolve_weights()
    logger.info("Loading malaria weights from %s (%s)", weights_path, source)

    net = build_network()
    # weights_only=True keeps the loader safe: only tensors are unpickled.
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    return MalariaClassifier(model=net, source=source)


# Module-level singleton so the (relatively slow) load happens once.
_model: Optional[MalariaClassifier] = None


def get_model() -> MalariaClassifier:
    """Return the process-wide classifier, loading it on first use."""
    global _model
    if _model is None:
        _model = load_model()
    return _model


def model_info() -> Dict[str, object]:
    """Metadata about the configured model, for the ``/model-info`` endpoint."""
    model = get_model()
    params = sum(p.numel() for p in model.model.parameters())
    return {
        "architecture": "MobileNetV2",
        "num_classes": NUM_CLASSES,
        "labels": model.labels,
        "parameters": params,
        "input_size": INPUT_SIZE,
        "weights_source": model.source,
        "weights_path": MODEL_PATH if model.source == "local-checkpoint" else MODEL_REPO,
        "tta_enabled": model.tta,
        "field_mode": {
            "min_cells": FIELD_MIN_CELLS,
            "max_cells": FIELD_MAX_CELLS,
            "dense_coverage": DENSE_FIELD_COVERAGE,
            "patch_size": FIELD_PATCH_SIZE,
            "patch_overlap": FIELD_PATCH_OVERLAP,
            "max_patches": FIELD_MAX_PATCHES,
            "cell_positive_threshold": CELL_POSITIVE_THRESHOLD,
        },
        "device": "cpu",
        "torch_num_threads": NUM_THREADS,
    }
