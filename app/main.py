"""FastAPI service exposing malaria detection over HTTP.

Endpoints
---------
GET  /            - minimal web UI (drag & drop an image)
GET  /health      - liveness probe
GET  /model-info  - architecture, labels and runtime details
POST /predict     - multipart image upload -> classification result
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app import model as model_module
from app.schemas import HealthResponse, ModelInfoResponse, PredictionResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("malaria.api")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Eagerly load the model so the first request is not slow.

    A failure here is logged but not fatal: ``/health`` stays up so an
    orchestrator can report the container as running while ``/predict``
    returns 503 until the weights become available.
    """
    try:
        model_module.get_model()
        logger.info("Model ready.")
    except Exception as exc:  # noqa: BLE001 - startup must never crash the app
        logger.warning("Model unavailable at startup: %s", exc)
    yield


app = FastAPI(
    title="Malaria Detection API",
    description=(
        "Lightweight MobileNetV2 classifier for Parasitized vs Uninfected "
        "blood-smear cell images."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page demo UI."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe that also reports whether the model finished loading."""
    return HealthResponse(status="ok", model_loaded=model_module._model is not None)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Describe the model currently being served."""
    try:
        return ModelInfoResponse(**model_module.model_info())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Model not loaded: {exc}") from exc


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Classify an uploaded blood-smear cell image."""
    raw = await file.read()

    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail="Uploaded file is not a valid image."
        ) from exc

    try:
        classifier = model_module.get_model()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Model not loaded: {exc}") from exc

    result = classifier.predict(image.convert("RGB"))
    return PredictionResponse(**result)
