"""Field analysis for full microscope-field images.

The MobileNetV2 model was trained on tightly cropped *single-cell*
images (~100-150 px, one cell per image).  Full-field smears (e.g.
the MIT malaria-diagnosis samples) contain hundreds of cells, which
is far outside that training distribution: the network downsamples
the whole field into mush and falls back to its majority class
("Uninfected").

This module puts field images back into the model's training
domain.  It analyses the image and provides:

* ``analyze_field`` -- background-adaptive segmentation that finds
  individual stained cells (works well for *sparse* fields), and
* ``grid_patches`` -- a sliding grid of model-sized patches (the
  robust fallback for *dense* fields, where cells touch and cannot
  be separated by colour thresholding alone).

Only numpy and Pillow are used, so the Docker image gains no new
dependency.

Segmentation notes
------------------
A fixed saturation threshold fails on real smears because the
background itself is often tinted (pale pink/lavender), not pure
white.  Instead the background colour is estimated per image (the
per-channel median), every pixel's Euclidean distance from it is
computed, and the distance map is binarised with Otsu's method --
adapting to the stain, the camera white balance and the lighting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# Longest side used for analysis (bounds memory and latency);
# bounding boxes are scaled back to the original resolution.
ANALYSIS_MAX_SIDE = 1600
# Downsample factor for the connected-component labelling pass.
LABEL_DOWNSAMPLE = 4
# Morphological cleanup: median removes speckle, Max+Min is a
# cheap close that fills the hollow centre of each cell.
# Kept small (3) so touching cells merge as little as possible.
CLOSE_SIZE = 3
# Component filters (area as a fraction of the analysis image).
MIN_AREA_FRAC = 3e-4
MAX_AREA_FRAC = 5e-2
MAX_ASPECT = 2.5
# Margin added around each detected cell before cropping (fraction
# of the cell's bounding box), so the cell rim is fully included.
CROP_MARGIN = 0.25


@dataclass
class CellCrop:
    """One detected cell: the crop and where it came from."""

    image: Image.Image
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1 in original px
    area: int


@dataclass
class FieldSegmentation:
    """Result of analysing a whole microscope field."""

    crops: List[CellCrop] = field(default_factory=list)
    # Fraction of pixels that differ from the estimated background.
    # High coverage == densely packed field (cells touching).
    coverage: float = 0.0
    # The adaptive distance threshold Otsu's method picked.
    threshold: float = 0.0


def _rgb_to_float(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32)


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu's method on a float distance map (0 .. ~442)."""
    hist, edges = np.histogram(values, bins=256, range=(0.0, 442.0))
    total = values.size
    if total == 0 or hist.sum() == 0:
        return 0.0
    sum_all = (np.arange(256) * hist).sum()
    sum_b, w_b, max_var, best = 0.0, 0.0, 0.0, 0.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var, best = var, edges[t]
    return float(best)


def _label_components(
    mask: Image.Image,
) -> List[Tuple[int, int, int, int, int]]:
    """Connected components on a downscaled mask.

    Returns ``(x0, y0, x1, y1, area)`` boxes in analysis-image
    pixels.  A plain stack-based flood fill is used so no scipy
    dependency is needed; the mask is downscaled first to keep the
    Python-level loop fast.
    """
    small = mask.resize(
        (max(1, mask.width // LABEL_DOWNSAMPLE),
         max(1, mask.height // LABEL_DOWNSAMPLE)),
        Image.BILINEAR,
    )
    arr = np.asarray(small) > 127
    h, w = arr.shape
    seen = np.zeros_like(arr, dtype=bool)
    boxes: List[Tuple[int, int, int, int, int]] = []

    for y in range(h):
        for x in range(w):
            if not arr[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            y0 = y1 = y
            x0 = x1 = x
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                if cy < y0:
                    y0 = cy
                elif cy > y1:
                    y1 = cy
                if cx < x0:
                    x0 = cx
                elif cx > x1:
                    x1 = cx
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and arr[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            boxes.append((x0, y0, x1, y1, area))
    return boxes


def analyze_field(
    image: Image.Image, max_cells: int = 10
) -> FieldSegmentation:
    """Segment stained cells in a full microscope field.

    Returns a :class:`FieldSegmentation` with up to ``max_cells``
    cell crops (largest first) plus the field's stain coverage.
    A whole-image (single-cell) input yields ~1 crop and high
    coverage; an empty field yields neither.
    """
    rgb = image.convert("RGB")
    scale = min(1.0, ANALYSIS_MAX_SIDE / max(rgb.width, rgb.height))
    if scale < 1.0:
        seg_img = rgb.resize(
            (int(rgb.width * scale), int(rgb.height * scale)),
            Image.BILINEAR,
        )
    else:
        seg_img = rgb

    arr = np.asarray(seg_img, dtype=np.uint8)
    # Background = the dominant colour (per-channel median).
    bg = np.median(arr.reshape(-1, 3), axis=0)
    dist = np.sqrt(
        ((_rgb_to_float(arr) - bg) ** 2).sum(axis=2)
    )
    threshold = _otsu_threshold(dist)

    mask = Image.fromarray((dist > threshold).astype(np.uint8) * 255, mode="L")
    mask = mask.filter(ImageFilter.MedianFilter(size=3))
    mask = mask.filter(ImageFilter.MaxFilter(size=CLOSE_SIZE))
    mask = mask.filter(ImageFilter.MinFilter(size=CLOSE_SIZE))

    mask_arr = np.asarray(mask) > 127
    coverage = float(mask_arr.mean())

    inv_scale = 1.0 / scale
    seg_area = float(seg_img.width * seg_img.height)

    kept: List[Tuple[int, int, int, int, int]] = []
    for x0, y0, x1, y1, area in _label_components(mask):
        area_full = area * LABEL_DOWNSAMPLE ** 2
        frac = area_full / seg_area
        if not (MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC):
            continue
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if max(w, h) / max(min(w, h), 1) > MAX_ASPECT:
            continue
        kept.append((x0, y0, x1, y1, area_full))

    kept.sort(key=lambda b: b[4], reverse=True)

    crops: List[CellCrop] = []
    for x0, y0, x1, y1, area in kept[:max_cells]:
        # Scale the box back to the original resolution, add a
        # margin, and make a square-ish crop clamped to the image.
        X0, Y0, X1, Y1 = (int(round(v * inv_scale)) for v in (x0, y0, x1, y1))
        cw, ch = X1 - X0, Y1 - Y0
        cx, cy = (X0 + X1) // 2, (Y0 + Y1) // 2
        half = max(cw, ch) // 2 + int(max(cw, ch) * CROP_MARGIN)
        left = max(0, cx - half)
        upper = max(0, cy - half)
        right = min(rgb.width, cx + half)
        lower = min(rgb.height, cy + half)
        crops.append(
            CellCrop(
                image=rgb.crop((left, upper, right, lower)),
                bbox=(left, upper, right, lower),
                area=int(area),
            )
        )

    logger.debug(
        "field analysis: coverage=%.3f threshold=%.1f cells=%d",
        coverage, threshold, len(crops),
    )
    return FieldSegmentation(crops=crops, coverage=coverage, threshold=threshold)


# Backwards-compatible alias: the crops are what most callers want.
def segment_cells(image: Image.Image, max_cells: int = 10) -> List[CellCrop]:
    """Return only the cell crops (see :func:`analyze_field`)."""
    return analyze_field(image, max_cells=max_cells).crops


def grid_patches(
    image: Image.Image,
    patch_size: int = 224,
    overlap: float = 0.25,
    max_patches: int = 16,
) -> List[Image.Image]:
    """Slide a grid of ``patch_size`` crops over a full field.

    Dense fields cannot be split into single cells by colour
    thresholding alone (the cells touch), so the field is instead
    covered by overlapping patches that each contain a few cells --
    close to what the single-cell model saw during training.
    Patches are subsampled evenly when the grid exceeds
    ``max_patches``.
    """
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w <= patch_size and h <= patch_size:
        return [rgb]

    step = max(1, int(patch_size * (1.0 - overlap)))
    xs = list(range(0, max(1, w - patch_size + 1), step))
    ys = list(range(0, max(1, h - patch_size + 1), step))
    if xs[-1] != max(0, w - patch_size):
        xs.append(max(0, w - patch_size))
    if ys[-1] != max(0, h - patch_size):
        ys.append(max(0, h - patch_size))

    patches = [
        rgb.crop((x, y, min(w, x + patch_size), min(h, y + patch_size)))
        for y in ys
        for x in xs
    ]

    if len(patches) > max_patches:
        # Evenly subsample so the whole field stays represented.
        idx = np.linspace(0, len(patches) - 1, max_patches).round().astype(int)
        patches = [patches[int(i)] for i in sorted(set(idx))]

    return patches
