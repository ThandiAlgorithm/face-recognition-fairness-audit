from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, duplicates, paths


def main() -> None:
    paths.ensure_dirs()
    preprocessing = config.load_preprocessing()
    manifest = pd.read_parquet(paths.METADATA_DIR / "file_manifest.parquet")
    integrity = pd.read_parquet(paths.METADATA_DIR / "image_integrity.parquet")

    hash_path = paths.CACHE_DIR / "perceptual_hashes.parquet"
    hashes = duplicates.compute_perceptual_hashes(
        manifest,
        integrity,
        preprocessing,
        hash_path,
    )

    review_path = paths.METADATA_DIR / "near_duplicate_review.csv"
    existing = pd.read_csv(review_path) if review_path.exists() else None
    review = duplicates.build_near_duplicate_review(hashes, preprocessing, existing)
    review.to_csv(review_path, index=False)

    flags = duplicates.duplicate_flags_from_review(manifest, review)
    flags_path = paths.METADATA_DIR / "duplicate_flags.parquet"
    flags.to_parquet(flags_path, index=False)

    exact_count = int(flags["exact_dup_group"].notna().sum())
    near_candidates = int(flags["near_dup_status"].isin(["candidate", "uncertain"]).sum())
    cross_candidates = int(flags["cross_dataset_dup_status"].isin(["candidate", "uncertain"]).sum())
    cross_confirmed = int(flags["cross_dataset_dup"].sum())

    print("Rows in exact duplicate groups:", exact_count)
    print("Rows with unresolved near duplicate review:", near_candidates)
    print("Rows with unresolved cross dataset perceptual review:", cross_candidates)
    print("Rows with confirmed cross dataset duplicates:", cross_confirmed)
    print("Review file:", review_path)
    print("Edit review_status to confirmed, false_alarm, or uncertain, then rerun this script.")


if __name__ == "__main__":
    main()
