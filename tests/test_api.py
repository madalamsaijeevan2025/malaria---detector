"""API tests for the malaria detection service.

The prediction tests are skipped automatically when the model weights cannot be
fetched (e.g. offline CI), so the health/info tests stay green everywhere.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _model_ready(client: TestClient) -> bool:
    return client.get("/health").json()["model_loaded"]


def _png_bytes(size: tuple[int, int] = (224, 224), color=(200, 40, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


def test_model_info(client):
    if not _model_ready(client):
        pytest.skip("model weights unavailable offline")
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["architecture"] == "MobileNetV2"
    assert body["labels"] == ["Parasitized", "Uninfected"]
    assert body["parameters"] > 0


def test_predict_returns_valid_structure(client):
    if not _model_ready(client):
        pytest.skip("model weights unavailable offline")

    response = client.post(
        "/predict", files={"file": ("cell.png", _png_bytes(), "image/png")}
    )
    assert response.status_code == 200

    body = response.json()
    assert body["label"] in ("Parasitized", "Uninfected")
    assert set(body["probabilities"]) == {"Parasitized", "Uninfected"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-3
    assert body["inference_ms"] >= 0


def test_predict_rejects_non_image(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 400


def test_predict_rejects_empty_file(client):
    response = client.post(
        "/predict", files={"file": ("empty.png", b"", "image/png")}
    )
    assert response.status_code == 400


def test_index_serves_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Malaria Detection" in response.text
