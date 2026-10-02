"""Dataset helpers shared by the download, fine-tune and evaluate scripts.

Default dataset: ``flame623/Malaria-classification`` (150 MB, ~6k train /
~6.9k test) with explicit ``parasitized`` / ``uninfected`` labels. Override
with the ``DATASET_ID`` environment variable, e.g.::

    DATASET_ID=dpdl-benchmark/malaria python train/fine_tune.py

for the full 27,558-image NIH set (integer labels only).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from torchvision import transforms

DATASET_ID = os.getenv("DATASET_ID", "flame623/Malaria-classification")
CACHE_DIR = os.getenv("DATASET_CACHE", os.path.join(".cache", "datasets"))

# Canonical class order used everywhere (API, training, evaluation).
CLASS_NAMES: Tuple[str, str] = ("Parasitized", "Uninfected")
INPUT_SIZE = 224
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Training augmentation. Blood cells are rotationally invariant, so a full
# 90-degree rotation range is safe and was used by the original model authors.
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ]
)


def load_splits(cache_dir: Optional[str] = None) -> Dict[str, object]:
    """Download (or load from cache) the dataset and return its splits."""
    return load_dataset(DATASET_ID, cache_dir=cache_dir or CACHE_DIR)


def resolve_label_map(split) -> Dict[int, int]:
    """Map the dataset's label ids onto our canonical 0=Parasitized/1=Uninfected.

    Uses the declared ``ClassLabel`` names when present; falls back to identity
    for datasets that only expose integer labels (e.g. ``dpdl-benchmark/malaria``).
    """
    canonical = {name.lower(): i for i, name in enumerate(CLASS_NAMES)}
    feature = split.features.get("label")
    names: Optional[List[str]] = getattr(feature, "names", None)

    if not names:
        return {i: i for i in range(len(CLASS_NAMES))}

    mapping: Dict[int, int] = {}
    for idx, name in enumerate(names):
        key = name.lower()
        if key not in canonical:
            raise ValueError(
                f"Unexpected class label '{name}'. Expected one of {CLASS_NAMES}."
            )
        mapping[idx] = canonical[key]
    return mapping


class MalariaDataset(torch.utils.data.Dataset):
    """Torch wrapper around a Hugging Face split."""

    def __init__(
        self,
        hf_split,
        transform: transforms.Compose,
        label_map: Dict[int, int],
    ) -> None:
        self.hf_split = hf_split
        self.transform = transform
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, index: int):
        row = self.hf_split[index]
        image = row["image"].convert("RGB")
        label = self.label_map[int(row["label"])]
        return self.transform(image), label


def _subsample(split, n: Optional[int], seed: int = 42):
    """Deterministically take a *representative* random subset of a split.

    The csv/parquet test split stores images grouped by class, so slicing the
    first ``n`` rows would return a single class. Shuffling with a fixed seed
    keeps subsets reproducible while preserving the class mix.
    """
    if not n or n >= len(split):
        return split
    return split.shuffle(seed=seed).select(range(n))


def build_datasets(
    train_max: Optional[int] = None,
    eval_max: Optional[int] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42,
):
    """Return ``(train_dataset, eval_dataset)`` ready for a DataLoader."""
    splits = load_splits(cache_dir)
    train_split = splits["train"]
    eval_split = splits["test"] if "test" in splits else splits["validation"]

    label_map = resolve_label_map(train_split)
    train_split = _subsample(train_split, train_max, seed)
    eval_split = _subsample(eval_split, eval_max, seed)

    train_ds = MalariaDataset(train_split, TRAIN_TRANSFORM, label_map)
    eval_ds = MalariaDataset(eval_split, EVAL_TRANSFORM, label_map)
    return train_ds, eval_ds
