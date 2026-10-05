"""Step 10: training loop.

Refuses to start on a dirty git tree unless --allow-dirty is passed (see
environment.check_git_clean); the override is recorded in environment.json
either way. AMP (autocast/GradScaler) is a no-op on CPU and activates
automatically on CUDA -- same code path either machine.

Uses the older torch.cuda.amp.autocast/GradScaler API (not the unified
torch.amp one) deliberately: the only GPU this project has access to
(dica10's Tesla K40c) has a driver too old for any CUDA-12-era PyTorch
build, so the cluster runs an older PyTorch (~1.12, CUDA 11.3) than local
dev (2.3.1) does. torch.cuda.amp is the one AMP API both versions support
identically, so this file doesn't need two code paths.

IMPORTANT -- the K40c cannot actually be used at all, not just slowly:
torch.cuda.is_available() returns True (the driver initializes), but any
real GPU op fails with "CUDA error: no kernel image is available for
execution on the device" -- PyTorch's prebuilt binaries ship no compiled
kernels for compute capability 3.5 (Kepler), the K40c's architecture.
Confirmed empirically 2026-10-02/03; see K40_INCOMPATIBILITY.md (repo root).
Training here is CPU-only by necessity, not by choice, and no source build
of PyTorch targeting sm_35 is planned -- too large a time/risk cost for a
two-day training budget. --device therefore defaults to cpu, and passing
--device cuda will simply fail the moment a batch runs.

Frozen recipe: SGD(momentum=0.9) + CosineAnnealingLR + weight_decay=1e-4,
fixed across all three datasets. Only --lr is meant to be swept (a small
pilot, not a full Optuna search) -- see src/tuning/. Running the exact same
optimizer/scheduler/augmentation everywhere also makes the cross-dataset
comparison this project cares about cleaner: differences in results are
not confounded by per-dataset tuning choices.

--max-epochs is a ceiling, not a target: pass --early-stopping-patience to
stop once validation loss stops improving rather than running to the cap.

--benchmark runs the real train step (load -> forward -> loss -> backward
-> optimizer.step(), the same model/transforms/batch size/loss/optimizer/
DataLoader settings as an actual run) for a fixed number of batches and
reports measured images/sec and estimated epoch/multi-epoch time, so a
schedule is built from a measurement, not a guess. Always benchmark with
the same thread/worker allocation the real run will use -- these numbers
do not transfer across different core counts.

--smoke-test-steps truncates train/val to a handful of batches per epoch.
It exists purely to verify the training loop is wired correctly (loss goes
somewhere sane, checkpoints save, history/environment get logged) without
running anything close to a real training run.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
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

SGD_MOMENTUM = 0.9
DEFAULT_WEIGHT_DECAY = 1e-4


def accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1):
    mask = targets != ignore_index
    if mask.sum() == 0:
        return None
    preds = logits[mask].argmax(dim=1)
    return (preds == targets[mask]).float().mean().item()


def build_optimizer_and_scheduler(model, lr: float, weight_decay: float, max_epochs: int):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=SGD_MOMENTUM, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    return optimizer, scheduler


def get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run_epoch(model, loader, loss_fn, optimizer, device, scaler, train: bool) -> dict:
    model.train(train)
    if isinstance(loss_fn, torch.nn.Module):
        loss_fn.train(train)
    totals: dict = {}
    counts: dict = {}

    for imgs, targets, meta in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(imgs)
                # meta (ITA/race/gender/age metadata) is passed through so
                # M3's per-sample reweighted loss can look up each sample's
                # weight; M1/M2's plain loss classes just ignore the extra
                # argument (default meta=None on their forward()).
                losses = loss_fn(outputs, targets, meta)

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


def _build_train_loader(dataset: str, batch_size: int, seed: int, num_workers: int, max_steps_per_epoch=None):
    train_ds = FaceDataset(dataset, "train", transform=train_transform())
    train_ds = _maybe_truncate(train_ds, max_steps_per_epoch, batch_size)
    generator = make_generator(seed)
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(train_ds, **loader_kwargs), len(train_ds)


def run_benchmark(
    dataset: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device_str: str,
    num_threads: int | None,
    num_workers: int,
    benchmark_batches: int,
    warmup_batches: int,
    out_dir: Path | None = None,
) -> dict:
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    device = torch.device(device_str)
    if device.type == "cuda":
        raise RuntimeError(
            "The K40c cannot execute any PyTorch CUDA kernel on this build (compute "
            "capability 3.5 is unsupported) -- benchmark CPU only. See this file's module docstring."
        )

    labels = load_labels(dataset)
    model = build_model(dataset, labels).to(device)
    loss_fn = build_loss(dataset)
    optimizer, _ = build_optimizer_and_scheduler(model, lr, weight_decay, max_epochs=1)

    set_seed(seed)
    train_loader, n_train_images = _build_train_loader(dataset, batch_size, seed, num_workers)
    batches_per_epoch = (n_train_images + batch_size - 1) // batch_size

    model.train()
    batch_times = []
    n_timed = 0
    loader_iter = iter(train_loader)
    total_needed = warmup_batches + benchmark_batches

    for i in range(total_needed):
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
            n_timed += 1

    mean_s = statistics.mean(batch_times)
    median_s = statistics.median(batch_times)
    p90_s = sorted(batch_times)[int(len(batch_times) * 0.9) - 1]
    images_per_sec = batch_size / mean_s

    epoch_s = mean_s * batches_per_epoch
    result = {
        "hostname": socket.gethostname(),
        "cpu_model": get_cpu_model(),
        "torch_threads": torch.get_num_threads(),
        "requested_num_threads": num_threads,
        "dataloader_workers": num_workers,
        "dataset": dataset,
        "n_train_images": n_train_images,
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "warmup_batches": warmup_batches,
        "timed_batches": n_timed,
        "mean_s_per_batch": mean_s,
        "median_s_per_batch": median_s,
        "p90_s_per_batch": p90_s,
        "images_per_sec": images_per_sec,
        "estimated_epoch_s": epoch_s,
        "estimated_10_epochs_s": epoch_s * 10,
        "estimated_20_epochs_s": epoch_s * 20,
        "estimated_30_epochs_s": epoch_s * 30,
    }

    def fmt_hm(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h} hours {m} minutes"

    print("=" * 40)
    print("TRAINING BENCHMARK")
    print("=" * 40)
    print(f"Dataset: {dataset}")
    print(f"Host: {result['hostname']}")
    print(f"CPU: {result['cpu_model']}")
    print(f"Torch threads: {result['torch_threads']}")
    print()
    print(f"Batch size: {batch_size}")
    print(f"Workers: {num_workers}")
    print()
    print(f"Timed batches: {n_timed}")
    print()
    print(f"Mean batch:   {mean_s:.2f} sec")
    print(f"Median batch: {median_s:.2f} sec")
    print(f"P90 batch:    {p90_s:.2f} sec")
    print(f"Images/sec:   {images_per_sec:.2f}")
    print()
    print(f"Estimated epoch ({batches_per_epoch} batches):")
    print(f"  {fmt_hm(epoch_s)}")
    print()
    print("Estimated:")
    print(f"  10 epochs = {fmt_hm(epoch_s * 10)}")
    print(f"  20 epochs = {fmt_hm(epoch_s * 20)}")
    print(f"  30 epochs = {fmt_hm(epoch_s * 30)}")
    print("=" * 40)

    out_dir = out_dir or (EXPERIMENTS_DIR / dataset / "benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2))
    return result


def train_dataset(
    dataset: str,
    max_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device_str: str,
    allow_dirty: bool,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    num_threads: int | None = None,
    num_workers: int = 0,
    early_stopping_patience: int | None = None,
    max_steps_per_epoch: int | None = None,
    out_dir: Path | None = None,
    epoch_callback=None,
    train_loader_factory=None,
    method: str = "M1",
) -> dict:
    """epoch_callback(epoch, val_metrics), if given, runs after every epoch's
    validation pass -- src/tuning/ uses it to report intermediate values to
    Optuna and raise optuna.TrialPruned without train.py importing optuna.

    train_loader_factory(dataset, batch_size, seed, num_workers,
    max_steps_per_epoch) -> DataLoader, if given, replaces the default
    plain-shuffle loader -- this is the one hook M2 (resampling) uses, so
    everything else (model, loss, optimizer, scheduler, early stopping,
    checkpointing) stays byte-identical to M1 and every other mitigation
    variant. method is recorded in environment.json for traceability; it
    does not change any training behaviour by itself."""
    check_git_clean(allow_dirty)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    set_seed(seed)
    device = torch.device(device_str)
    if device.type == "cuda":
        raise RuntimeError(
            "The K40c cannot execute any PyTorch CUDA kernel on this build -- "
            "training is CPU-only. See this file's module docstring."
        )

    labels = load_labels(dataset)
    model = build_model(dataset, labels).to(device)
    loss_fn = build_loss(dataset)
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr, weight_decay, max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if train_loader_factory is not None:
        train_loader = train_loader_factory(dataset, batch_size, seed, num_workers, max_steps_per_epoch)
    else:
        train_loader, _ = _build_train_loader(dataset, batch_size, seed, num_workers, max_steps_per_epoch)
    val_ds = FaceDataset(dataset, "val", transform=eval_transform())
    val_ds = _maybe_truncate(val_ds, max_steps_per_epoch, batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    out_dir = out_dir or (EXPERIMENTS_DIR / dataset / "runs" / f"seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_hash_path = EXPERIMENTS_DIR / dataset / "split_manifest.sha256"
    split_manifest_sha256 = split_hash_path.read_text().strip() if split_hash_path.exists() else None
    config = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping_patience,
    }

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early_at = None

    for epoch in range(max_epochs):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, scaler, train=True)
        val_metrics = run_epoch(model, val_loader, loss_fn, optimizer, device, scaler, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        history.append({"epoch": epoch, "elapsed_s": elapsed, "train": train_metrics, "val": val_metrics})
        print(
            f"[{dataset}] epoch {epoch}: train_loss={train_metrics['total']:.4f} "
            f"val_loss={val_metrics['total']:.4f} ({elapsed:.1f}s)"
        )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "validation_score": best_val_loss,
                    "val_loss": best_val_loss,
                    "config": config,
                    "label_mapping": labels,
                    "split_manifest_sha256": split_manifest_sha256,
                    "rng_states": rng_state_dict(),
                },
                out_dir / "best_checkpoint.pt",
            )
        else:
            epochs_without_improvement += 1

        if epoch_callback is not None:
            epoch_callback(epoch, val_metrics)

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            stopped_early_at = epoch
            print(f"[{dataset}] early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
            break

    env = capture_environment(allow_dirty_used=allow_dirty)
    env["method"] = method
    env["dataset"] = dataset
    env["seed"] = seed
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    return {"history": history, "best_val_loss": best_val_loss, "stopped_early_at": stopped_early_at}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface", "lfw"])
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--smoke-test-steps",
        type=int,
        default=None,
        help="Truncate train/val to this many batches per epoch -- wiring checks only, never a real run.",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run a throughput benchmark instead of training.")
    parser.add_argument("--benchmark-batches", type=int, default=200)
    parser.add_argument("--warmup-batches", type=int, default=20)
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(
            dataset=args.dataset,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device_str=args.device,
            num_threads=args.num_threads,
            num_workers=args.num_workers,
            benchmark_batches=args.benchmark_batches,
            warmup_batches=args.warmup_batches,
        )
        return

    train_dataset(
        dataset=args.dataset,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_str=args.device,
        allow_dirty=args.allow_dirty,
        num_threads=args.num_threads,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        max_steps_per_epoch=args.smoke_test_steps,
    )


if __name__ == "__main__":
    main()
