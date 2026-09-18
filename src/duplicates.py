
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src import paths


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def hash_images(rel_paths: list[str], progress_every: int = 20_000) -> pd.DataFrame:
    rows = []
    for i, rel in enumerate(rel_paths, start=1):
        abs_path = paths.to_abs(rel)
        rows.append({"path": rel, "sha256": sha256_file(abs_path)})
        if i % progress_every == 0:
            print(f"  hashed {i}/{len(rel_paths)}")
    return pd.DataFrame(rows)


def find_exact_duplicate_groups(hashes_df: pd.DataFrame) -> pd.DataFrame:
    df = hashes_df.copy()
    group_sizes = df.groupby("sha256")["sha256"].transform("size")
    dup_hashes = sorted(df.loc[group_sizes > 1, "sha256"].unique())
    hash_to_group = {h: f"DUPGRP-{i:05d}" for i, h in enumerate(dup_hashes, start=1)}
    df["exact_duplicate_group"] = df["sha256"].map(hash_to_group)
    return df


def summarise_within_dataset_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in df.groupby("dataset"):
        dup = group[group["exact_duplicate_group"].notna()]
        # Require two matches in the same dataset.
        within_dup_groups = dup.groupby("exact_duplicate_group")["dataset"].apply(
            lambda s: (s == dataset).sum() > 1
        )
        real_within_groups = within_dup_groups[within_dup_groups].index
        within_rows = dup[dup["exact_duplicate_group"].isin(real_within_groups)]
        rows.append(
            {
                "dataset": dataset,
                "total_images": len(group),
                "unique_hashes": group["sha256"].nunique(),
                "images_in_a_within_dataset_duplicate_group": len(within_rows),
                "within_dataset_duplicate_groups": len(real_within_groups),
            }
        )
    return pd.DataFrame(rows)


def summarise_cross_dataset_overlap(df: pd.DataFrame, dataset_a: str, dataset_b: str) -> dict:
    sub = df[df["dataset"].isin([dataset_a, dataset_b])]
    hash_datasets = sub.groupby("sha256")["dataset"].apply(set)
    overlap_hashes = hash_datasets[hash_datasets.apply(lambda s: {dataset_a, dataset_b}.issubset(s))]
    overlap_rows = sub[sub["sha256"].isin(overlap_hashes.index)].sort_values("sha256")
    return {
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "overlapping_hashes": len(overlap_hashes),
        "overlapping_images_total": len(overlap_rows),
        "rows": overlap_rows,
    }
