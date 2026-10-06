"""M2 -- balanced/resampled training batches.

A pure data-level intervention: training images are drawn via a
WeightedRandomSampler instead of plain shuffling, so each epoch sees
roughly equal numbers of ITA<28 and ITA>=28 examples. Nothing else changes
-- same loss function, same images (no synthetic generation, no permanent
removal), same validation/test data, same model/optimizer/scheduler as M1.
That isolation is the entire point of M2: if fairness improves, we want to
be able to say resampling caused it, not some other simultaneous change.

Per-sample weight = 1 / (number of training images in that image's ITA
group). Rows with unknown/missing ITA (never inferred) form their own
third weight group, so they stay in the training pool at the same
proportion as any other group -- excluding them would silently shrink the
training set relative to M1, which is exactly the kind of uncontrolled
difference this experiment design is meant to avoid.

Logs, per spec:
  sampler_config.yaml   -- static: formula, group counts, seed, settings
  sampling_history.csv  -- per epoch: what was actually drawn
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from face_dataset import FaceDataset
from transforms import train_transform
from seed import make_generator, worker_init_fn

from train import train_dataset, EXPERIMENTS_DIR

ITA_GROUPS = ["<28", "≥28", "unknown"]


def _ita_bucket(value) -> str:
    if value == "≥28" or value == ">=28":
        return "≥28"
    if value == "<28":
        return "<28"
    return "unknown"


class CountingWeightedSampler(Sampler):
    """WeightedRandomSampler that also records what it actually drew each
    epoch, so sampling_history.csv reflects reality rather than the
    intended distribution."""

    def __init__(self, weights: np.ndarray, item_group: np.ndarray, num_samples: int, generator: torch.Generator):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.item_group = item_group
        self.num_samples = num_samples
        self.generator = generator
        self.last_group_counts: dict[str, int] = {}
        self.last_unique_images = 0
        self.last_duplicate_draws = 0

    def __iter__(self):
        idx = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=self.generator)
        idx_np = idx.numpy()
        drawn_groups = self.item_group[idx_np]
        unique_groups, counts = np.unique(drawn_groups, return_counts=True)
        self.last_group_counts = dict(zip(unique_groups.tolist(), counts.tolist()))
        unique_images = len(np.unique(idx_np))
        self.last_unique_images = unique_images
        self.last_duplicate_draws = self.num_samples - unique_images
        return iter(idx_np.tolist())

    def __len__(self) -> int:
        return self.num_samples


def build_m2_train_loader(dataset: str, batch_size: int, seed: int, num_workers: int, max_steps_per_epoch=None):
    train_ds = FaceDataset(dataset, "train", transform=train_transform())
    item_group = train_ds.df["ita_group"].map(_ita_bucket).to_numpy()

    group_counts = {g: int((item_group == g).sum()) for g in ITA_GROUPS}
    weights = np.array([1.0 / group_counts[g] if group_counts[g] > 0 else 0.0 for g in item_group])

    num_samples = max_steps_per_epoch * batch_size if max_steps_per_epoch else len(train_ds)
    generator = make_generator(seed)
    sampler = CountingWeightedSampler(weights, item_group, num_samples, generator)

    loader_kwargs = dict(
        batch_size=batch_size,
        sampler=sampler,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(train_ds, **loader_kwargs)
    loader._m2_sampler = sampler  # noqa: SLF001 -- read back by the epoch_callback below
    loader._m2_group_counts = group_counts
    return loader


def run_m2(
    dataset: str,
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
    max_steps_per_epoch: int | None = None,
) -> dict:
    out_dir = EXPERIMENTS_DIR / dataset / "M2_resampling" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sampling_log_path = out_dir / "sampling_history.csv"
    sampling_rows: list[dict] = []
    state: dict = {}

    def factory(dataset_, batch_size_, seed_, num_workers_, max_steps_per_epoch_):
        loader = build_m2_train_loader(dataset_, batch_size_, seed_, num_workers_, max_steps_per_epoch_)
        state["loader"] = loader
        return loader

    def epoch_callback(epoch: int, val_metrics: dict) -> None:
        sampler = state["loader"]._m2_sampler  # noqa: SLF001
        counts = {g: sampler.last_group_counts.get(g, 0) for g in ITA_GROUPS}
        total = sum(counts.values()) or 1
        sampling_rows.append(
            {
                "epoch": epoch,
                "number_sampled_ita_lt28": counts["<28"],
                "number_sampled_ita_ge28": counts["≥28"],
                "number_sampled_ita_unknown": counts["unknown"],
                "percentage_lt28": counts["<28"] / total * 100,
                "percentage_ge28": counts["≥28"] / total * 100,
                "unique_images_seen": sampler.last_unique_images,
                "duplicate_draws": sampler.last_duplicate_draws,
            }
        )
        with open(sampling_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(sampling_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sampling_rows)

    result = train_dataset(
        dataset=dataset,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=seed,
        device_str=device_str,
        allow_dirty=allow_dirty,
        num_threads=num_threads,
        num_workers=num_workers,
        early_stopping_patience=early_stopping_patience,
        max_steps_per_epoch=max_steps_per_epoch,
        out_dir=out_dir,
        epoch_callback=epoch_callback,
        train_loader_factory=factory,
        method="M2",
    )

    group_counts = state["loader"]._m2_group_counts  # noqa: SLF001
    sampler_config = {
        "balance_variable": "ita_group",
        "original_group_counts": group_counts,
        "sampling_formula": "weight(i) = 1 / count(train images in i's ITA group); unknown ITA forms its own group, never inferred",
        "replacement": True,
        "num_samples_per_epoch": len(sampling_rows) and (state["loader"]._m2_sampler.num_samples),  # noqa: SLF001
        "sampler_random_seed": seed,
    }
    (out_dir / "sampler_config.yaml").write_text(yaml.safe_dump(sampler_config, sort_keys=False))

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface", "lfw"])
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, required=True, help="Use M1's winning lr for a controlled comparison.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--smoke-test-steps", type=int, default=None)
    args = parser.parse_args()

    run_m2(
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
