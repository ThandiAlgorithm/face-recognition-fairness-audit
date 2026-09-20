from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, master_table, paths, splits


def main() -> None:
    paths.ensure_dirs()
    preprocessing = config.load_preprocessing()
    exclusions = config.load_exclusions()

    manifest = pd.read_parquet(paths.METADATA_DIR / "file_manifest.parquet")
    integrity = pd.read_parquet(paths.METADATA_DIR / "image_integrity.parquet")
    labels = pd.read_parquet(paths.METADATA_DIR / "parsed_labels.parquet")
    duplicate_flags = pd.read_parquet(paths.METADATA_DIR / "duplicate_flags.parquet")

    master = master_table.build_master(
        manifest,
        integrity,
        labels,
        duplicate_flags,
        preprocessing,
    )

    split_path = paths.METADATA_DIR / "utkface_split.csv"
    utkface_split = splits.lock_utkface_split(master, preprocessing, split_path)
    master = master_table.apply_utkface_split(master, utkface_split)

    master = master_table.apply_eligibility(master, manifest, exclusions)
    master = master_table.finalise_master(master)

    output_path = paths.METADATA_DIR / "master_annotations.parquet"
    master.to_parquet(output_path, index=False)

    counts = master.groupby(["dataset", "dataset_version"]).size().rename("canonical_rows").reset_index()
    counts.to_csv(paths.DATA_PREP_OUTPUTS_DIR / "tables" / "master_counts.csv", index=False)

    print(counts.to_string(index=False))
    print("Total:", len(master))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
