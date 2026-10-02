"""Fine-tune MobileNetV2 on the malaria dataset, tuned for a weak CPU.

Strategy (progressive unfreezing, as used by the original model authors):

1. ``head``      - freeze the backbone, train only the classifier.
2. ``partial``   - unfreeze the last inverted-residual blocks at a lower LR.
3. ``full``      - unfreeze everything at a very low LR.

On a 2-core i5-7200U the defaults (2000 training images, 2 epochs, head phase)
finish in minutes. Pass ``--phase progressive`` for the full three-stage run.

Usage::

    python train/fine_tune.py
    python train/fine_tune.py --phase progressive --epochs 3 --max-train-samples 6000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow both `python train/fine_tune.py` and `python -m train.fine_tune`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.model import build_network, resolve_weights
from train.dataset import build_datasets

DEFAULT_OUTPUT = os.path.join("models", "malaria_mobilenetv2.pth")

# (learning rate, number of backbone blocks kept trainable) per phase.
# ``None`` means the whole backbone is trainable.
PHASES = {
    "head": (1e-3, 0),
    "partial": (1e-4, 4),
    "full": (1e-5, None),
}


def configure_trainable(net: nn.Module, trainable_blocks: int | None) -> None:
    """Freeze/unfreeze parts of the backbone.

    ``trainable_blocks=0``     -> classifier head only.
    ``trainable_blocks=N``     -> head + the last N inverted-residual blocks.
    ``trainable_blocks=None``  -> the entire network.
    """
    for param in net.features.parameters():
        param.requires_grad = trainable_blocks is None

    if trainable_blocks:  # unfreeze the trailing N blocks
        for block in net.features[-trainable_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    for param in net.classifier.parameters():
        param.requires_grad = True


def run_epoch(net, loader, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    net.train(train)
    total_loss, correct, seen = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.inference_mode()

    with context:
        for images, labels in tqdm(loader, leave=False, desc="train" if train else "eval"):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the malaria classifier.")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs per phase")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None, help="Override the phase LR")
    parser.add_argument(
        "--phase",
        choices=["head", "partial", "full", "progressive"],
        default="head",
        help="Which unfreezing schedule to run",
    )
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=2, help="PyTorch CPU threads")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    # --- data ---------------------------------------------------------------
    print("Loading dataset ...")
    train_ds, eval_ds = build_datasets(args.max_train_samples, args.max_eval_samples)
    print(f"  train: {len(train_ds)} images | eval: {len(eval_ds)} images")

    # num_workers=0 avoids process-spawn overhead on a 2-core machine.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- model --------------------------------------------------------------
    weights_path, source = resolve_weights()
    print(f"Starting from pretrained weights: {weights_path} ({source})")
    net = build_network()
    net.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    net.to(device)

    criterion = nn.CrossEntropyLoss()

    schedule = ["head", "partial", "full"] if args.phase == "progressive" else [args.phase]

    best_acc = 0.0
    for phase in schedule:
        phase_lr, trainable_blocks = PHASES[phase]
        lr = args.lr if args.lr is not None else phase_lr
        configure_trainable(net, trainable_blocks)

        n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"\n=== Phase '{phase}' | lr={lr} | trainable params={n_trainable:,} ===")

        optimizer = torch.optim.AdamW(
            (p for p in net.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4
        )

        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss, train_acc = run_epoch(net, train_loader, criterion, optimizer, device, True)
            eval_loss, eval_acc = run_epoch(net, eval_loader, criterion, optimizer, device, False)
            secs = time.perf_counter() - started
            print(
                f"[{phase} {epoch}/{args.epochs}] "
                f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"eval loss={eval_loss:.4f} acc={eval_acc:.4f} | {secs:.1f}s"
            )
            best_acc = max(best_acc, eval_acc)

    # --- save ---------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Save the raw state dict so it drops straight into app/model.py's loader
    # (which uses weights_only=True and expects a plain state_dict).
    torch.save(net.state_dict(), args.output)
    print(f"\nSaved fine-tuned weights to {args.output}")
    print(f"Best eval accuracy: {best_acc:.4f}")
    print("The API will use these weights automatically on next start.")


if __name__ == "__main__":
    main()
