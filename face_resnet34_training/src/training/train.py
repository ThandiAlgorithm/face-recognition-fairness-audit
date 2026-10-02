"""Step 10: training loop.

Refuses to start on a dirty git tree unless --allow-dirty is passed (see
environment.check_git_clean); the override is recorded in environment.json
either way. AMP (autocast/GradScaler) is a no-op on CPU and activates
automatically on CUDA -- same code path either machine.

--smoke-test-steps truncates train/val to a handful of batches per epoch.
It exists purely to verify the training loop is wired correctly (loss goes
somewhere sane, checkpoints save, history/environment get logged) without
running anything close to a real training run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from face_dataset import FaceDataset
from transforms import train_transform, eval_transform
from label_encoders import load_labels
from heads import build_model
from losses import build_loss
from seed import set_seed, worker_init_fn, make_generator, rng_state_dict
from environment import check_git_clean, capture_environment

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"


def accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1):
    mask = targets != ignore_index
    if mask.sum() == 0:
        return None
    preds = logits[mask].argmax(dim=1)
    return (preds == targets[mask]).float().mean().item()


def run_epoch(model, loader, loss_fn, optimizer, device, scaler, train: bool) -> dict:
    model.train(train)
    totals: dict = {}
    counts: dict = {}

    for imgs, targets, _meta in loader:
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(imgs)
                losses = loss_fn(outputs, targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(losses["total"]).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    optimizer.step()

        batch_size = imgs.size(0)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + value.item() * batch_size
            counts[key] = counts.get(key, 0) + batch_size

        for task, logits in outputs.items():
            if targets[task].dtype != torch.long:
                continue
            acc = accuracy(logits, targets[task])
            if acc is not None:
                acc_key = f"{task}_acc"
                totals[acc_key] = totals.get(acc_key, 0.0) + acc * batch_size
                counts[acc_key] = counts.get(acc_key, 0) + batch_size

    return {k: totals[k] / counts[k] for k in totals}


def _maybe_truncate(dataset, max_steps_per_epoch: int | None, batch_size: int):
    if max_steps_per_epoch is None:
        return dataset
    n = min(len(dataset), max_steps_per_epoch * batch_size)
    return Subset(dataset, range(n))


def train_dataset(
    dataset: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device_str: str,
    allow_dirty: bool,
    max_steps_per_epoch: int | None = None,
    out_dir: Path | None = None,
) -> dict:
    check_git_clean(allow_dirty)
    set_seed(seed)
    device = torch.device(device_str)

    labels = load_labels(dataset)
    model = build_model(dataset, labels).to(device)
    loss_fn = build_loss(dataset)
    # AdamW with a placeholder LR -- this baseline loop isn't tuned; Optuna
    # (src/tuning/) searches LR/weight-decay/etc. properly in the next step.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    train_ds = FaceDataset(dataset, "train", transform=train_transform())
    val_ds = FaceDataset(dataset, "val", transform=eval_transform())
    train_ds = _maybe_truncate(train_ds, max_steps_per_epoch, batch_size)
    val_ds = _maybe_truncate(val_ds, max_steps_per_epoch, batch_size)

    generator = make_generator(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    out_dir = out_dir or (EXPERIMENTS_DIR / dataset / "runs" / f"seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float("inf")
    for epoch in range(epochs):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, scaler, train=True)
        val_metrics = run_epoch(model, val_loader, loss_fn, optimizer, device, scaler, train=False)
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "elapsed_s": elapsed, "train": train_metrics, "val": val_metrics})
        print(
            f"[{dataset}] epoch {epoch}: train_loss={train_metrics['total']:.4f} "
            f"val_loss={val_metrics['total']:.4f} ({elapsed:.1f}s)"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rng_state": rng_state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                out_dir / "best_checkpoint.pt",
            )

    env = capture_environment(allow_dirty_used=allow_dirty)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    return {"history": history, "best_val_loss": best_val_loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface", "lfw"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--smoke-test-steps",
        type=int,
        default=None,
        help="Truncate train/val to this many batches per epoch -- wiring checks only, never a real run.",
    )
    args = parser.parse_args()

    train_dataset(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device_str=args.device,
        allow_dirty=args.allow_dirty,
        max_steps_per_epoch=args.smoke_test_steps,
    )


if __name__ == "__main__":
    main()
