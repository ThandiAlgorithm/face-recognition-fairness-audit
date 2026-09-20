from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_utkface_split(master: pd.DataFrame, preprocessing_cfg: dict) -> pd.DataFrame:
    cfg = preprocessing_cfg["splits"]["utkface"]
    seed = int(cfg["seed"])
    fractions = cfg.get("fractions", {"train": 0.70, "val": 0.15, "test": 0.15})
    total_fraction = sum(float(value) for value in fractions.values())
    if abs(total_fraction - 1.0) > 1e-9:
        raise RuntimeError("UTKFace split fractions must sum to 1.")

    subset = master[master["dataset"].eq("utkface")][["image_id"]].sort_values("image_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subset))
    labels = np.empty(len(subset), dtype=object)

    n_train = int(round(len(subset) * float(fractions["train"])))
    n_val = int(round(len(subset) * float(fractions["val"])))
    train_indices = order[:n_train]
    val_indices = order[n_train:n_train + n_val]
    test_indices = order[n_train + n_val:]
    labels[train_indices] = "train"
    labels[val_indices] = "val"
    labels[test_indices] = "test"

    output = subset.copy()
    output["split"] = labels
    return output


def lock_utkface_split(
    master: pd.DataFrame,
    preprocessing_cfg: dict,
    split_path: Path,
) -> pd.DataFrame:
    generated = build_utkface_split(master, preprocessing_cfg)
    if split_path.exists():
        existing = pd.read_csv(split_path)
        check = generated.merge(existing, on="image_id", suffixes=("_new", "_old"), how="outer", indicator=True)
        mismatch = check[
            check["_merge"].ne("both")
            | check["split_new"].ne(check["split_old"])
        ]
        if len(mismatch):
            raise RuntimeError(
                "Existing UTKFace split does not match the deterministic frozen split."
            )
        return existing[["image_id", "split"]].copy()

    generated.to_csv(split_path, index=False)
    return generated
