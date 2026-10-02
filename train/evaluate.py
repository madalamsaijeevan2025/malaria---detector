"""Evaluate the malaria classifier on the held-out test split.

Reports accuracy, precision, recall and F1 (for the *Parasitized* class, the
medically important one) plus a confusion matrix, and writes a JSON report to
``reports/evaluation.json`` so results can be cited in the docs.

Usage::

    python train/evaluate.py
    python train/evaluate.py --checkpoint models/malaria_mobilenetv2.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow both `python train/evaluate.py` and `python -m train.evaluate`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.model import build_network, resolve_weights
from train.dataset import CLASS_NAMES, build_datasets

REPORT_PATH = os.path.join("reports", "evaluation.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the malaria classifier.")
    parser.add_argument("--checkpoint", default=None, help="Path to a .pth state dict")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None, help="Subsample the test set")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", default=REPORT_PATH)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    if args.checkpoint and os.path.exists(args.checkpoint):
        weights_path, source = args.checkpoint, "local-checkpoint"
    else:
        weights_path, source = resolve_weights()
    print(f"Evaluating weights: {weights_path} ({source})")

    _, eval_ds = build_datasets(train_max=None, eval_max=args.max_samples)
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    net = build_network()
    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    net.eval().to(device)

    all_preds, all_labels = [], []
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="evaluating"):
            preds = net(images.to(device)).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # Canonical order is 0=Parasitized, 1=Uninfected; treat Parasitized as the
    # positive class for precision/recall/F1.
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=0, zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=0, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist()

    report = {
        "weights": weights_path,
        "weights_source": source,
        "num_samples": len(all_labels),
        "labels": list(CLASS_NAMES),
        "accuracy": round(float(accuracy), 4),
        "precision_parasitized": round(float(precision), 4),
        "recall_parasitized": round(float(recall), 4),
        "f1_parasitized": round(float(f1), 4),
        "confusion_matrix": cm,
    }

    print("\n=== Evaluation ===")
    for key in ("accuracy", "precision_parasitized", "recall_parasitized", "f1_parasitized"):
        print(f"  {key:<24} {report[key]}")
    print(f"  confusion matrix (rows=true, cols=pred, order={CLASS_NAMES}): {cm}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
