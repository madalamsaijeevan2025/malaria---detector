"""Download and cache the malaria dataset from the Hugging Face Hub.

Usage::

    python train/download_data.py
    DATASET_ID=dpdl-benchmark/malaria python train/download_data.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

# Allow both `python train/download_data.py` and `python -m train.download_data`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.dataset import CLASS_NAMES, DATASET_ID, CACHE_DIR, load_splits, resolve_label_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the malaria dataset.")
    parser.add_argument("--dataset", default=DATASET_ID, help="Hugging Face dataset id")
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Where to cache the data")
    args = parser.parse_args()

    print(f"Downloading '{args.dataset}' into '{args.cache_dir}' ...")
    splits = load_splits(args.cache_dir)

    for name, split in splits.items():
        label_map = resolve_label_map(split)
        counts = Counter(label_map[int(label)] for label in split["label"])
        pretty = {CLASS_NAMES[i]: counts.get(i, 0) for i in range(len(CLASS_NAMES))}
        print(f"  {name:<10} {len(split):>7} images  {pretty}")

    print("Done. Data is cached and ready for train/fine_tune.py.")


if __name__ == "__main__":
    main()
