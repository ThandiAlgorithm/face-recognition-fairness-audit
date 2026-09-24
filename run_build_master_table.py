from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, master_table, paths, splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the complete Phase 0 master table."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.METADATA_DIR / "master_annotations.parquet",
        help="Output path. Specify a separate file for testing.",
    )
    args = parser.parse_args()

    paths.ensure_dirs()

    preprocessing = config.load_preprocessing()
    categories_cfg = config.load_categories()
    exclusions = config.load_exclusions()

    metadata = paths.METADATA_DIR

    manifest = pd.read_parquet(
        metadata / "file_manifest.parquet"
    )
    integrity = pd.read_parquet(
        metadata / "image_integrity.parquet"
    )
    labels = pd.read_parquet(
        metadata / "parsed_labels.parquet"
    )
    duplicate_flags = pd.read_parquet(
        metadata / "duplicate_flags.parquet"
    )

    diagnostics_path = (
        metadata / "detection_alignment_diagnostics.parquet"
    )

    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            "Saved extraction diagnostics are required: "
            f"{diagnostics_path}"
        )

    diagnostics = pd.read_parquet(diagnostics_path)

    # 1. Reconstruct canonical inventory and annotations.
    master = master_table.build_master(
        manifest,
        integrity,
        labels,
        duplicate_flags,
        preprocessing,
    )

    # 2. Reproduce the locked, duplicate-aware UTKFace split.
    split_path = metadata / "utkface_split.csv"

    utkface_split = splits.lock_utkface_split(
        master,
        preprocessing,
        split_path,
    )

    master = master_table.apply_utkface_split(
        master,
        utkface_split,
    )

    # 3. Verify that saved extraction diagnostics cover
    # exactly the same canonical images.
    if (
        master["image_id"].isna().any()
        or master["image_id"].duplicated().any()
        or diagnostics["image_id"].isna().any()
        or diagnostics["image_id"].duplicated().any()
    ):
        raise RuntimeError(
            "Null or duplicate image IDs in rebuild inputs."
        )

    if (
        len(master) != 135039
        or len(diagnostics) != 135039
        or set(master["image_id"])
            != set(diagnostics["image_id"])
    ):
        raise RuntimeError(
            "Master and extraction diagnostics do not "
            "contain the same 135,039 image IDs."
        )

    # 4. Restore all detection and measurement columns
    # belonging to the canonical master schema.
    update_columns = [
        column
        for column in master.columns
        if column in diagnostics.columns
        and column != "image_id"
    ]

    required_updates = {
        "n_detections",
        "alignment_valid",
        "alignment_transform",
        "brightness",
        "contrast",
        "sharpness",
        "skin_roi_valid",
    }

    missing = required_updates - set(update_columns)

    if missing:
        raise RuntimeError(
            "Missing extraction fields: "
            f"{sorted(missing)}"
        )

    updates = diagnostics[
        ["image_id", *update_columns]
    ].copy()

    master = master_table.apply_detection_updates(
        master,
        updates,
    )

    # 5. Recreate derived categories and eligibility
    # in the same order as full feature extraction.
    master = master_table.apply_measurement_categories(
        master,
        categories_cfg,
    )

    master = master_table.apply_eligibility(
        master,
        manifest,
        exclusions,
    )

    master = master_table.finalise_master(master)

    if len(master) != 135039 or not master["image_id"].is_unique:
        raise RuntimeError(
            "Final master row-integrity check failed."
        )

    # 6. Write to the requested destination.
    output_path = args.output
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_parquet(
        output_path,
        index=False,
    )

    counts = (
        master.groupby(["dataset", "dataset_version"])
        .size()
        .rename("canonical_rows")
        .reset_index()
    )

    # Do not overwrite production summary outputs
    # when testing a separate rebuild.
    production_path = (
        metadata / "master_annotations.parquet"
    )

    if output_path.resolve() == production_path.resolve():
        counts.to_csv(
            paths.DATA_PREP_OUTPUTS_DIR
            / "tables"
            / "master_counts.csv",
            index=False,
        )

    print(counts.to_string(index=False))
    print("Total:", len(master))
    print("Valid alignments:",
          int(master["alignment_valid"].eq(True).sum()))
    print("No detections:",
          int(master["no_detection"].eq(True).sum()))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
