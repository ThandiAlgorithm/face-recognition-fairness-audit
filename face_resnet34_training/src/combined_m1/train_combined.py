"""Combined-M1: one ResNet-34 baseline trained jointly on FairFace + UTKFace
+ LFW train splits (ds/model/losses in this same directory), no fairness
mitigation -- the shared-data counterpart to M1-M4, which each train one
model per dataset. Self-contained rather than reusing
src/training/train.py's train_dataset() wrapper (that function is built
around a single `dataset` string and FaceDataset/build_model/build_loss,
none of which fit a model spanning three datasets at once); it does reuse
that file's lower-level, dataset-agnostic pieces directly: run_epoch (the
train/eval step itself), the frozen SGD+cosine recipe, and the
environment/seed/checkpoint conventions, so Combined-M1's results stay
comparable to M1-M4's.

Same CPU-only constraint and AMP/grad-scaler setup as train.py -- see that
file's module docstring for why (the cluster's one GPU cannot run any
PyTorch CUDA kernel).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from dataset import CombinedDataset, build_age_harmonized_vocab, build_race_harmonized_vocab, lfw_attribute_vocab
from model import CombinedResNet34
from combined_loss import CombinedLoss
from transforms import train_transform, eval_transform
from seed import set_seed, worker_init_fn, make_generator, rng_state_dict
from environment import check_git_clean, capture_environment
from train import run_epoch, build_optimizer_and_scheduler, get_cpu_model

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
DEFAULT_WEIGHT_DECAY = 1e-4


def _build_model() -> CombinedResNet34:
    return CombinedResNet34(
        n_age=len(build_age_harmonized_vocab()),
        n_race=len(build_race_harmonized_vocab()),
        n_lfw_attributes=len(lfw_attribute_vocab()),
    )


def _build_loader(split: str, transform, batch_size: int, seed: int, num_workers: int, shuffle: bool) -> DataLoader:
    ds = CombinedDataset(split, transform=transform)
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, worker_init_fn=worker_init_fn)
    if shuffle:
        kwargs["shuffle"] = True
        kwargs["generator"] = make_generator(seed)
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def run_benchmark(
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device_str: str,
    num_threads: int | None,
    num_workers: int,
    benchmark_batches: int,
    warmup_batches: int,
) -> dict:
    import socket
    import statistics

    if num_threads is not None:
        torch.set_num_threads(num_threads)
    device = torch.device(device_str)
    if device.type == "cuda":
        raise RuntimeError("CPU-only -- see this file's module docstring.")

    set_seed(seed)
    model = _build_model().to(device)
    loss_fn = CombinedLoss()
    optimizer, _ = build_optimizer_and_scheduler(model, lr, weight_decay, max_epochs=1)
    train_loader = _build_loader("train", train_transform(), batch_size, seed, num_workers, shuffle=True)
    n_train_images = len(train_loader.dataset)
    batches_per_epoch = (n_train_images + batch_size - 1) // batch_size

    model.train()
    batch_times = []
    loader_iter = iter(train_loader)
    for i in range(warmup_batches + benchmark_batches):
        try:
            imgs, targets, _meta = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            imgs, targets, _meta = next(loader_iter)

        t0 = time.perf_counter()
        imgs = imgs.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        losses = loss_fn(outputs, targets)
        losses["total"].backward()
        optimizer.step()
        elapsed = time.perf_counter() - t0
        if i >= warmup_batches:
            batch_times.append(elapsed)

    mean_s = statistics.mean(batch_times)
    epoch_s = mean_s * batches_per_epoch
    result = {
        "hostname": socket.gethostname(),
        "cpu_model": get_cpu_model(),
        "torch_threads": torch.get_num_threads(),
        "requested_num_threads": num_threads,
        "dataloader_workers": num_workers,
        "n_train_images": n_train_images,
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "timed_batches": len(batch_times),
        "mean_s_per_batch": mean_s,
        "median_s_per_batch": statistics.median(batch_times),
        "images_per_sec": batch_size / mean_s,
        "estimated_epoch_s": epoch_s,
        "estimated_epoch_hm": f"{int(epoch_s // 3600)}h{int((epoch_s % 3600) // 60)}m",
        "estimated_30_epochs_s": epoch_s * 30,
    }
    print(json.dumps(result, indent=2))
    out_dir = EXPERIMENTS_DIR / "combined" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2))
    return result


def train_combined_m1(
    max_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device_str: str,
    allow_dirty: bool,
    num_threads: int | None,
    num_workers: int,
    early_stopping_patience: int | None,
) -> dict:
    check_git_clean(allow_dirty)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    set_seed(seed)
    device = torch.device(device_str)
    if device.type == "cuda":
        raise RuntimeError("CPU-only -- see this file's module docstring.")

    model = _build_model().to(device)
    loss_fn = CombinedLoss()
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr, weight_decay, max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    train_loader = _build_loader("train", train_transform(), batch_size, seed, num_workers, shuffle=True)
    val_loader = _build_loader("val", eval_transform(), batch_size, seed, num_workers, shuffle=False)

    out_dir = EXPERIMENTS_DIR / "combined" / "M1_baseline" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "method": "Combined-M1",
        "source_datasets": ["fairface", "utkface", "lfw"],
        "seed": seed,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping_patience,
        "heads": ["gender", "age_harmonized", "race_harmonized", "skin_tone", "lfw_attributes"],
        "race_harmonized_vocab": build_race_harmonized_vocab(),
        "age_harmonized_vocab": build_age_harmonized_vocab(),
        "n_lfw_attributes": len(lfw_attribute_vocab()),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, scaler, train=True)
        val_metrics = run_epoch(model, val_loader, loss_fn, optimizer, device, scaler, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "elapsed_s": elapsed, "train": train_metrics, "val": val_metrics})
        print(f"[combined_m1] epoch {epoch}: train_loss={train_metrics['total']:.4f} "
              f"val_loss={val_metrics['total']:.4f} ({elapsed:.1f}s)")

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            epochs_without_improvement = 0
            torch.save({
                "method": "Combined-M1",
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
                "rng_states": rng_state_dict(),
            }, out_dir / "best_checkpoint.pt")
        else:
            epochs_without_improvement += 1

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"[combined_m1] early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
            break

    env = capture_environment(allow_dirty_used=allow_dirty)
    env["method"] = "Combined-M1"
    env["seed"] = seed
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    return {"history": history, "best_val_loss": best_val_loss}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-batches", type=int, default=20)
    parser.add_argument("--warmup-batches", type=int, default=5)
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(
            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
            device_str=args.device, num_threads=args.num_threads, num_workers=args.num_workers,
            benchmark_batches=args.benchmark_batches, warmup_batches=args.warmup_batches,
        )
        return

    train_combined_m1(
        max_epochs=args.max_epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        seed=args.seed, device_str=args.device, allow_dirty=args.allow_dirty, num_threads=args.num_threads,
        num_workers=args.num_workers, early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    main()
