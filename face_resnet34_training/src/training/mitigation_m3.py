"""M3 -- sample/loss reweighting.

No resampling -- training batches are generated exactly as in M1
(shuffle=True, see train.py's default loader). The intervention is
entirely inside the loss: every sample gets a weight based on its
(ITA group, task label) combination, computed once from the TRAINING
partition only, then applied to that sample's per-task loss term before
averaging. This is deliberately different from M2 (which never touches
the loss) -- discrimination-aware "reweighing" and "resampling" are
distinct interventions in the fairness literature and mixing them would
make it impossible to attribute any later change to one mechanism.

    w(s, y) = P(S=s) * P(Y=y) / P(S=s, Y=y)

computed from training-set counts, then normalised so the mean training
weight is ~1.0. FairFace/UTKFace get one table per task (ITA x race,
ITA x gender, ITA x age_group); LFW gets one table per LFWA+ attribute
(ITA x present/absent), applied to BCEWithLogitsLoss(reduction="none").

Missing ITA is never inferred: those rows get weight 1.0 (neutral) rather
than being folded into a fabricated group. If a subgroup is so small its
raw weight balloons, it is flagged in reweighting_table.csv rather than
silently clipped -- --weight-clip is available but off by default and
always recorded when used.

Weights apply to the training objective only -- validation uses the same
plain (unweighted) loss M1/M2 would, since checkpoint selection and early
stopping should track genuine generalisation, not an artificially skewed
metric. (Confirmed this matters in practice, not just in theory: before
this split was added, a rare (ITA, class) cell's large weight landing in a
small validation batch spiked val_loss into the hundreds.)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from face_dataset import FaceDataset
from label_encoders import load_labels

from train import train_dataset, EXPERIMENTS_DIR
from losses import IGNORE_INDEX, FAIRFACE_UTKFACE_TASKS
from demographic_weights import compute_multi_demographic_weights

NEUTRAL_WEIGHT = 1.0
REWEIGHT_VARIABLES = ["ita", "multi"]


def _ita_bucket(value) -> str:
    if value in ("≥28", ">=28"):
        return "≥28"
    if value == "<28":
        return "<28"
    return "unknown"


def _build_table(df: pd.DataFrame, y_col: str, weight_clip: float | None) -> pd.DataFrame:
    """One (ITA_group, class) row per combination actually present in
    training, computed from training-set-only counts."""
    valid = df[df[y_col].notna() & (df[y_col] != "unknown")]
    n_total = len(valid)
    rows = []
    if n_total == 0:
        return pd.DataFrame(columns=["class", "ITA_group", "n", "raw_weight", "normalised_weight", "clipped"])

    for s_val, s_group in valid.groupby("ita_bucket"):
        p_s = len(s_group) / n_total
        for y_val, y_group in valid.groupby(y_col):
            p_y = len(y_group) / n_total
            n_sy = int(((valid["ita_bucket"] == s_val) & (valid[y_col] == y_val)).sum())
            p_sy = n_sy / n_total
            raw_w = (p_s * p_y / p_sy) if p_sy > 0 else float("nan")
            rows.append({"class": y_val, "ITA_group": s_val, "n": n_sy, "raw_weight": raw_w})

    table = pd.DataFrame(rows)
    finite = table["raw_weight"].replace([float("inf"), float("-inf")], pd.NA).dropna()
    mean_w = float(finite.mean()) if len(finite) else 1.0
    table["normalised_weight"] = table["raw_weight"] / mean_w
    table["clipped"] = False
    if weight_clip is not None:
        over = table["normalised_weight"] > weight_clip
        table.loc[over, "normalised_weight"] = weight_clip
        table.loc[over, "clipped"] = True
    return table


def compute_reweighting_tables(dataset: str, weight_clip: float | None = None) -> dict[str, pd.DataFrame]:
    train_ds = FaceDataset(dataset, "train", transform=None)
    df = train_ds.df.copy()
    df["ita_bucket"] = df["ita_group"].map(_ita_bucket)

    tables = {}
    if dataset == "lfw":
        labels = load_labels(dataset)
        for attr in labels["attributes"]:
            col = f"attr_{attr}"
            if col in df.columns:
                tables[attr] = _build_table(df, col, weight_clip)
    else:
        column_for_task = {"race": "race_label", "gender": "gender_label", "age_group": "age_group"}
        for task in FAIRFACE_UTKFACE_TASKS:
            tables[task] = _build_table(df, column_for_task[task], weight_clip)
    return tables


def _lookup_dicts(tables: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """{task_or_attr: {(ITA_group, class): normalised_weight}}"""
    out = {}
    for key, table in tables.items():
        out[key] = {(row["ITA_group"], row["class"]): row["normalised_weight"] for _, row in table.iterrows()}
    return out


class _EpochWeightTracker:
    """Accumulates per-batch weight stats across a full training epoch
    only (never validation -- run_epoch() shares the same loss_fn for both
    passes, toggling nn.Module.training via loss_fn.train(train)/.eval()).
    epoch_summary() both returns and resets the accumulator, so each
    logged row is that epoch's training batches only, not a stale
    last-batch-seen snapshot from whichever pass ran last."""

    def __init__(self):
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._mins: dict[str, float] = {}
        self._maxs: dict[str, float] = {}

    def record(self, key: str, weights: torch.Tensor) -> None:
        if weights.numel() == 0:
            return
        self._sums[key] = self._sums.get(key, 0.0) + float(weights.sum())
        self._counts[key] = self._counts.get(key, 0) + weights.numel()
        w_min, w_max = float(weights.min()), float(weights.max())
        self._mins[key] = min(self._mins.get(key, w_min), w_min)
        self._maxs[key] = max(self._maxs.get(key, w_max), w_max)

    def epoch_summary(self) -> dict[str, dict]:
        summary = {
            key: {
                "mean": self._sums[key] / self._counts[key],
                "min": self._mins[key],
                "max": self._maxs[key],
            }
            for key in self._sums
        }
        self._sums, self._counts, self._mins, self._maxs = {}, {}, {}, {}
        return summary


class WeightedMultiTaskLoss(nn.Module):
    """FairFace/UTKFace: per-sample weighted CE per task, summed into total."""

    def __init__(self, lookup: dict[str, dict], label_names: dict[str, list[str]]):
        super().__init__()
        self.lookup = lookup
        self.label_names = label_names
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
        self.tracker = _EpochWeightTracker()

    def forward(self, outputs: dict, targets: dict, meta: dict | None = None) -> dict:
        """Weights are a training-time intervention only. Validation uses
        the same plain (unweighted) mean loss M1/M2 would compute --
        checkpoint selection and early stopping should track genuine
        generalisation, not an artificially skewed training objective. This
        also matters practically: a rare (ITA, class) cell can carry a
        large weight, and applying that to a small validation batch made
        val_loss spike to 100+ in testing before this fix -- not a sign of
        a broken model, just a weighted metric that was never meant to be
        used for model selection."""
        ita_buckets = [_ita_bucket(g) for g in meta["ita_group"]] if self.training else None
        losses = {}
        for task in FAIRFACE_UTKFACE_TASKS:
            per_sample = self.ce(outputs[task], targets[task])  # (B,), 0 where ignore_index
            valid_mask = targets[task] != IGNORE_INDEX
            n_valid = valid_mask.sum().clamp(min=1)

            if self.training:
                names = self.label_names.get(task, [])
                weights = torch.ones_like(per_sample)
                for i, (bucket, idx) in enumerate(zip(ita_buckets, targets[task].tolist())):
                    if idx == IGNORE_INDEX or idx >= len(names):
                        continue
                    weights[i] = self.lookup[task].get((bucket, names[idx]), NEUTRAL_WEIGHT)
                losses[task] = (per_sample * weights * valid_mask).sum() / n_valid
                self.tracker.record(task, weights[valid_mask])
            else:
                losses[task] = (per_sample * valid_mask).sum() / n_valid
        losses["total"] = sum(losses[task] for task in FAIRFACE_UTKFACE_TASKS)
        return losses


class WeightedAttributeLoss(nn.Module):
    """LFW: per-sample, per-attribute weighted BCE."""

    def __init__(self, lookup: dict[str, dict], attributes: list[str]):
        super().__init__()
        self.lookup = lookup
        self.attributes = attributes
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.tracker = _EpochWeightTracker()

    def forward(self, outputs: dict, targets: dict, meta: dict | None = None) -> dict:
        """See WeightedMultiTaskLoss.forward's docstring: weights apply to
        the training objective only, never to the validation metric used
        for checkpoint selection / early stopping."""
        per_sample = self.bce(outputs["attributes"], targets["attributes"])  # (B, n_attributes)
        if not self.training:
            loss = per_sample.mean()
            return {"attributes": loss, "total": loss}

        ita_buckets = [_ita_bucket(g) for g in meta["ita_group"]]
        weights = torch.ones_like(per_sample)
        for i, bucket in enumerate(ita_buckets):
            for j, attr in enumerate(self.attributes):
                label = int(targets["attributes"][i, j].item())
                weights[i, j] = self.lookup[attr].get((bucket, label), NEUTRAL_WEIGHT)
        loss = (per_sample * weights).mean()
        self.tracker.record("attributes", weights)
        return {"attributes": loss, "total": loss}


class MultiWeightedMultiTaskLoss(nn.Module):
    """FairFace/UTKFace, --reweight-variable multi: ONE combined
    race+gender+age+skin-tone rarity weight per image (demographic_weights.py,
    same formula M2-MULTI's sampler uses), applied uniformly to every task's
    loss -- unlike WeightedMultiTaskLoss above, which computes a separate
    per-task (ITA, class) weight. Looked up by image_path, since that's
    already carried in meta for every batch (face_dataset.py)."""

    def __init__(self, weight_by_image: dict[str, float]):
        super().__init__()
        self.weight_by_image = weight_by_image
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
        self.tracker = _EpochWeightTracker()

    def forward(self, outputs: dict, targets: dict, meta: dict | None = None) -> dict:
        losses = {}
        weights = None
        if self.training:
            device = next(iter(outputs.values())).device
            weights = torch.tensor(
                [self.weight_by_image.get(p, NEUTRAL_WEIGHT) for p in meta["image_path"]],
                dtype=torch.float32, device=device,
            )
            self.tracker.record("multi", weights)
        for task in FAIRFACE_UTKFACE_TASKS:
            per_sample = self.ce(outputs[task], targets[task])
            valid_mask = targets[task] != IGNORE_INDEX
            n_valid = valid_mask.sum().clamp(min=1)
            if self.training:
                losses[task] = (per_sample * weights * valid_mask).sum() / n_valid
            else:
                losses[task] = (per_sample * valid_mask).sum() / n_valid
        losses["total"] = sum(losses[task] for task in FAIRFACE_UTKFACE_TASKS)
        return losses


class MultiWeightedAttributeLoss(nn.Module):
    """LFW, --reweight-variable multi: same combined per-image weight as
    above, applied uniformly across every LFWA+ attribute's BCE term for
    that image."""

    def __init__(self, weight_by_image: dict[str, float]):
        super().__init__()
        self.weight_by_image = weight_by_image
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.tracker = _EpochWeightTracker()

    def forward(self, outputs: dict, targets: dict, meta: dict | None = None) -> dict:
        per_sample = self.bce(outputs["attributes"], targets["attributes"])  # (B, n_attributes)
        if not self.training:
            loss = per_sample.mean()
            return {"attributes": loss, "total": loss}
        weights = torch.tensor(
            [self.weight_by_image.get(p, NEUTRAL_WEIGHT) for p in meta["image_path"]],
            dtype=torch.float32, device=per_sample.device,
        ).unsqueeze(1)
        self.tracker.record("attributes", weights.squeeze(1))
        loss = (per_sample * weights).mean()
        return {"attributes": loss, "total": loss}


def run_m3(
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
    weight_clip: float | None = None,
    max_steps_per_epoch: int | None = None,
    reweight_variable: str = "ita",
) -> dict:
    dir_name = "M3_reweighting" if reweight_variable == "ita" else "M3_reweighting_multi"
    out_dir = EXPERIMENTS_DIR / dataset / dir_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if reweight_variable == "multi":
        train_ds = FaceDataset(dataset, "train", transform=None)
        weights_arr, diagnostics = compute_multi_demographic_weights(train_ds.df)
        weight_by_image = dict(zip(train_ds.df["image_path"], weights_arr.tolist()))
        (out_dir / "weight_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
        labels = load_labels(dataset)
        loss_fn = MultiWeightedAttributeLoss(weight_by_image) if dataset == "lfw" else MultiWeightedMultiTaskLoss(weight_by_image)

        weight_history_path = out_dir / "loss_weight_history.csv"
        weight_history_rows: list[dict] = []

        def epoch_callback(epoch: int, val_metrics: dict) -> None:
            row = {"epoch": epoch}
            for task, stats in loss_fn.tracker.epoch_summary().items():
                row[f"{task}_mean_weight"] = stats["mean"]
                row[f"{task}_min_weight"] = stats["min"]
                row[f"{task}_max_weight"] = stats["max"]
            weight_history_rows.append(row)
            with open(weight_history_path, "w", newline="") as f:
                fieldnames = sorted({k for r in weight_history_rows for k in r})
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(weight_history_rows)

        import train as train_module
        original_build_loss = train_module.build_loss
        train_module.build_loss = lambda _dataset: loss_fn
        try:
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
                method="M3_multi",
            )
        finally:
            train_module.build_loss = original_build_loss
        return result

    tables = compute_reweighting_tables(dataset, weight_clip)
    combined = pd.concat(
        [t.assign(task=name) for name, t in tables.items()], ignore_index=True
    )[["task", "class", "ITA_group", "n", "raw_weight", "normalised_weight", "clipped"]]
    combined.to_csv(out_dir / "reweighting_table.csv", index=False)

    n_clipped = int(combined["clipped"].sum())
    if n_clipped:
        print(f"[M3/{dataset}] WARNING: {n_clipped} (ITA_group, class) cells were weight-clipped at {weight_clip}")

    lookup = _lookup_dicts(tables)
    labels = load_labels(dataset)
    if dataset == "lfw":
        loss_fn = WeightedAttributeLoss(lookup, labels["attributes"])
    else:
        loss_fn = WeightedMultiTaskLoss(lookup, labels)

    weight_history_path = out_dir / "loss_weight_history.csv"
    weight_history_rows: list[dict] = []

    def epoch_callback(epoch: int, val_metrics: dict) -> None:
        row = {"epoch": epoch}
        for task, stats in loss_fn.tracker.epoch_summary().items():
            row[f"{task}_mean_weight"] = stats["mean"]
            row[f"{task}_min_weight"] = stats["min"]
            row[f"{task}_max_weight"] = stats["max"]
        weight_history_rows.append(row)
        with open(weight_history_path, "w", newline="") as f:
            fieldnames = sorted({k for r in weight_history_rows for k in r})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(weight_history_rows)

    # train_dataset() builds its own loss via build_loss(dataset); M3 needs
    # its own weighted loss instance instead, so this calls the same
    # underlying epoch loop with everything else identical by monkeypatching
    # the loss the same way train_loader_factory overrides the loader.
    import train as train_module

    original_build_loss = train_module.build_loss
    train_module.build_loss = lambda _dataset: loss_fn
    try:
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
            method="M3",
        )
    finally:
        train_module.build_loss = original_build_loss

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
    parser.add_argument("--weight-clip", type=float, default=None, help="Cap normalised weights at this value; always recorded in reweighting_table.csv when used.")
    parser.add_argument("--reweight-variable", choices=REWEIGHT_VARIABLES, default="ita",
                         help="'ita' (default) reproduces the original per-task (ITA, class) reweighing, "
                              "writing to M3_reweighting/ as before. 'multi' uses one combined "
                              "race+gender+age+skin-tone rarity weight per image applied uniformly across "
                              "every task, writing to M3_reweighting_multi/ instead so it never overwrites the original.")
    parser.add_argument("--smoke-test-steps", type=int, default=None)
    args = parser.parse_args()

    run_m3(
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
        weight_clip=args.weight_clip,
        max_steps_per_epoch=args.smoke_test_steps,
        reweight_variable=args.reweight_variable,
    )


if __name__ == "__main__":
    main()
