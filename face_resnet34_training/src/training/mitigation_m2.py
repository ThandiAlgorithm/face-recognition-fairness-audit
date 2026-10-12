"""M2 -- balanced/resampled training batches.

A pure data-level intervention: training images are drawn via a
WeightedRandomSampler instead of plain shuffling, so each epoch sees
roughly equal numbers of examples from each group of --balance-variable.
Nothing else changes -- same loss function, same images (no synthetic
generation, no permanent removal), same validation/test data, same
model/optimizer/scheduler as M1. That isolation is the entire point of M2:
if fairness improves, we want to be able to say resampling caused it, not
some other simultaneous change.

--balance-variable selects which column to balance on: race_label,
gender_label, age_group, or ita_group (default, kept for backward
compatibility with earlier ita_group-only M2 runs -- their results and
output path are unchanged by this generalization). Per-sample weight = 1 /
(number of training images in that image's group for the chosen variable).
Rows with unknown/missing values (never inferred) form their own group, so
they stay in the training pool at the same proportion as any other group
-- excluding them would silently shrink the training set relative to M1.

Output path is experiments/<dataset>/M2_resampling/seed_<seed>/ for the
default ita_group (unchanged, so existing results keep their path) and
experiments/<dataset>/M2_resampling_by_<balance_variable>/seed_<seed>/
otherwise, so a race- or gender-targeted run never overwrites the original.

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
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from face_dataset import FaceDataset
from transforms import train_transform
from seed import make_generator, worker_init_fn

from train import train_dataset, EXPERIMENTS_DIR
from demographic_weights import compute_multi_demographic_weights

BALANCE_VARIABLES = ["ita_group", "race_label", "gender_label", "age_group", "multi"]
UNKNOWN_BUCKET = "unknown"


def _ita_bucket(value) -> str:
    if value == "≥28" or value == ">=28":
        return "≥28"
    if value == "<28":
        return "<28"
    return UNKNOWN_BUCKET


def _generic_bucket(value) -> str:
    # race_label/gender_label/age_group: real missing values are bare NaN
    # (not the string 'unknown') -- see FaceDataset/records.py -- so this
    # normalizes both to the same bucket name, same convention ita_group
    # already used for its own unknowns.
    if value is None:
        return UNKNOWN_BUCKET
    try:
        if value != value:  # NaN
            return UNKNOWN_BUCKET
    except TypeError:
        pass
    return str(value)


def _bucket_fn(balance_variable: str):
    return _ita_bucket if balance_variable == "ita_group" else _generic_bucket


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


def build_m2_train_loader(
    dataset: str, batch_size: int, seed: int, num_workers: int, max_steps_per_epoch=None,
    balance_variable: str = "ita_group",
):
    train_ds = FaceDataset(dataset, "train", transform=train_transform())

    if balance_variable == "multi":
        # Continuous per-image weight (race+gender+age+skin-tone rarity,
        # computed fresh from THIS dataset's own train split -- see
        # demographic_weights.py). CountingWeightedSampler's "group" concept
        # doesn't apply to a continuous weight, so samples are binned into
        # deciles purely for the sampling_history.csv report -- the bin
        # label plays no role in the actual sampling, which uses the real
        # continuous weight below.
        weights, diagnostics = compute_multi_demographic_weights(train_ds.df)
        (EXPERIMENTS_DIR / dataset / "M2_resampling_by_multi").mkdir(parents=True, exist_ok=True)
        (EXPERIMENTS_DIR / dataset / "M2_resampling_by_multi" / "weight_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2)
        )
        # labels=False avoids pd.qcut's "labels must be one fewer than bin
        # edges" error when many tied weight values force it to drop
        # duplicate edges (seen on LFW, whose weight distribution has more
        # ties than FairFace/UTKFace's) -- the resulting integer codes are
        # purely a reporting label for sampling_history.csv either way.
        decile_codes = pd.qcut(weights, 10, labels=False, duplicates="drop")
        item_group = np.asarray([f"decile_{c+1}" for c in decile_codes])
        groups = sorted(set(item_group.tolist()))
        group_counts = {g: int((item_group == g).sum()) for g in groups}
    else:
        bucket = _bucket_fn(balance_variable)
        item_group = train_ds.df[balance_variable].map(bucket).to_numpy()
        groups = sorted(set(item_group.tolist()))
        group_counts = {g: int((item_group == g).sum()) for g in groups}
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
    loader._m2_groups = groups
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
    balance_variable: str = "ita_group",
) -> dict:
    dir_name = "M2_resampling" if balance_variable == "ita_group" else f"M2_resampling_by_{balance_variable}"
    out_dir = EXPERIMENTS_DIR / dataset / dir_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sampling_log_path = out_dir / "sampling_history.csv"
    sampling_rows: list[dict] = []
    state: dict = {}

    def factory(dataset_, batch_size_, seed_, num_workers_, max_steps_per_epoch_):
        loader = build_m2_train_loader(
            dataset_, batch_size_, seed_, num_workers_, max_steps_per_epoch_, balance_variable=balance_variable,
        )
        state["loader"] = loader
        return loader

    def epoch_callback(epoch: int, val_metrics: dict) -> None:
        sampler = state["loader"]._m2_sampler  # noqa: SLF001
        groups = state["loader"]._m2_groups  # noqa: SLF001
        counts = {g: sampler.last_group_counts.get(g, 0) for g in groups}
        total = sum(counts.values()) or 1
        row = {"epoch": epoch}
        for g in groups:
            row[f"number_sampled_{g}"] = counts[g]
            row[f"percentage_{g}"] = counts[g] / total * 100
        row["unique_images_seen"] = sampler.last_unique_images
        row["duplicate_draws"] = sampler.last_duplicate_draws
        sampling_rows.append(row)
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
        method="M2" if balance_variable == "ita_group" else f"M2_{balance_variable}",
    )

    group_counts = state["loader"]._m2_group_counts  # noqa: SLF001
    sampler_config = {
        "balance_variable": balance_variable,
        "original_group_counts": group_counts,
        "sampling_formula": f"weight(i) = 1 / count(train images in i's {balance_variable} group); "
                             f"unknown/missing {balance_variable} forms its own group, never inferred",
        "replacement": True,
        "num_samples_per_epoch": len(sampling_rows) and (state["loader"]._m2_sampler.num_samples),  # noqa: SLF001
        "sampler_random_seed": seed,
    }
    (out_dir / "sampler_config.yaml").write_text(yaml.safe_dump(sampler_config, sort_keys=False))

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface", "lfw"])
    parser.add_argument("--balance-variable", choices=BALANCE_VARIABLES, default="ita_group",
                         help="Which column to balance the sampler on. Default (ita_group) matches the "
                              "original M2 run and its output path; any other value writes to "
                              "M2_resampling_by_<variable> instead, so it never overwrites the original.")
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
        balance_variable=args.balance_variable,
    )


if __name__ == "__main__":
    main()
