"""Pydantic response models for the malaria detection API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    architecture: str
    num_classes: int
    labels: List[str]
    parameters: int
    input_size: int
    weights_source: str
    weights_path: str
    tta_enabled: bool
    field_mode: Optional[dict] = None
    device: str
    torch_num_threads: int


class PredictionResponse(BaseModel):
    label: str = Field(..., examples=["Parasitized"])
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]
    inference_ms: float
    tta: bool
    # Full-field (whole-smear) results.  Only populated when the
    # image was analysed per-cell ("field-cells") or per-patch
    # ("field-patches") instead of as a whole crop.
    mode: str = "single-cell"
    cells_analyzed: Optional[int] = None
    parasitized_cells: Optional[int] = None
    parasitemia_estimate: Optional[float] = None
    patches_analyzed: Optional[int] = None
    positive_patches: Optional[int] = None
    field_positivity: Optional[float] = None
