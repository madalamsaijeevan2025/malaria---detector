"""Tests for full-field analysis and patch-based inference.

Segmentation tests use synthetic smear fields (stained blobs on
a white background), so they run without model weights.  The
prediction tests skip automatically when the weights cannot be
fetched (offline CI), mirroring tests/test_api.py.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from app.cells import analyze_field, grid_patches


def _field_image(width: int = 600, height: int = 400, n_blobs: int = 8):
    """Synthetic microscope field: separated stained blobs on white."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # A magenta-purple resembling Giemsa-stained cells.
    for i in range(n_blobs):
        x = 50 + (i * 67) % (width - 120)
        y = 40 + (i * 91) % (height - 120)
        draw.ellipse([x, y, x + 60, y + 60], fill=(150, 60, 160))
    return img


def test_analyze_field_finds_blobs():
    seg = analyze_field(_field_image())
    assert len(seg.crops) >= 3
    for crop in seg.crops:
        assert crop.image.width > 0 and crop.image.height > 0
        assert crop.area > 0
        x0, y0, x1, y1 = crop.bbox
        assert 0 <= x0 < x1 <= 600
        assert 0 <= y0 < y1 <= 400
    assert seg.coverage > 0.0


def test_analyze_field_empty_on_white():
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    seg = analyze_field(img)
    assert seg.crops == []
    assert seg.coverage < 0.01


def test_analyze_field_touching_blobs_is_dense_not_cells():
    # Overlapping blobs merge into one huge component: high
    # coverage, no individual cells -- the "dense field" signal.
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i in range(6):
        x = 40 + i * 55
        y = 100 + (i % 2) * 30
        draw.ellipse([x, y, x + 110, y + 110], fill=(150, 60, 160))
    seg = analyze_field(img)
    assert seg.coverage >= 0.1
    assert len(seg.crops) <= 1


def test_analyze_field_caps_max_cells():
    seg = analyze_field(_field_image(n_blobs=20), max_cells=4)
    assert 1 <= len(seg.crops) <= 4


def test_grid_patches_covers_image():
    patches = grid_patches(_field_image(), patch_size=224, overlap=0.25)
    assert 1 <= len(patches) <= 16
    for patch in patches:
        assert patch.size == (224, 224)


def test_grid_patches_small_image_yields_one_patch():
    # Smaller than the patch size: the whole image is the
    # single patch (the model transform resizes it to 224).
    patches = grid_patches(Image.new("RGB", (100, 100), (200, 200, 200)))
    assert len(patches) == 1
    assert patches[0].size == (100, 100)


def _png_bytes(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _model_ready() -> bool:
    try:
        from app.model import get_model

        get_model()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_ready(), reason="model weights unavailable offline")
def test_predict_field_mode():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("field.png", _png_bytes(_field_image()), "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    # Separated blobs on a pale field -> per-cell mode.
    assert body["mode"] == "field-cells"
    assert body["cells_analyzed"] >= 3
    assert body["parasitized_cells"] is not None
    assert body["parasitemia_estimate"] is not None
    assert 0.0 <= body["parasitemia_estimate"] <= 1.0
    assert body["label"] in ("Parasitized", "Uninfected")
