from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd


CLASS_ORDER = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino",
]

FOLDER_NAME = {
    "White": "White",
    "Black": "Black",
    "Indian": "Indian",
    "East Asian": "East_Asian",
    "Southeast Asian": "Southeast_Asian",
    "Middle Eastern": "Middle_Eastern",
    "Latino": "Latino",
}


def repo_root_from_script() -> Path:
    here = Path(__file__).resolve()
    # Expected location: <repo>/dataset_harmonization/materialize_harmonized_dataset.py
    candidate = here.parents[1]
    if (candidate / "results" / "harmonized_dataset").exists():
        return candidate
    # Fallback: walk upward for the expected results folder.
    for parent in here.parents:
        if (parent / "results" / "harmonized_dataset").exists():
            return parent
    raise RuntimeError(
        "Could not locate the repository root. Place this script inside "
        "<repo>/dataset_harmonization/ or pass --manifest and --output explicitly."
    )


def stable_record_id(source_dataset: str, image_id: str) -> str:
    token = f"{source_dataset}|{image_id}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()[:16]


def resolve_source_path(row: pd.Series, repo_root: Path) -> Path | None:
    candidates = []
    for col in ["image_path", "image_path_original"]:
        if col in row.index and pd.notna(row[col]):
            text = str(row[col]).strip()
            if text:
                p = Path(text)
                candidates.append(p if p.is_absolute() else repo_root / p)
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def human_size(num_bytes: int) -> str:
    n = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{num_bytes} B"


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            # Hard links can fail across drives/filesystems; fall back to copy.
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the frozen harmonized manifest into a conventional "
            "train/val/test image-folder dataset."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to training_manifest_frozen.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination folder for the standalone dataset",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "hardlink"],
        default="copy",
        help=(
            "copy = standalone files suitable for Drive upload; "
            "hardlink = saves local disk space when source and destination share a filesystem"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and estimate size without creating image files",
    )
    args = parser.parse_args()

    repo_root = repo_root_from_script()
    manifest = args.manifest or (
        repo_root
        / "results"
        / "harmonized_dataset"
        / "final_mvp_candidate"
        / "frozen_training"
        / "training_manifest_frozen.csv"
    )
    output = args.output or (
        repo_root
        / "results"
        / "harmonized_dataset"
        / "Harmonized_Face_Dataset_v1_MATERIALIZED"
    )

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    df = pd.read_csv(manifest, low_memory=False)

    required = {
        "source_dataset",
        "image_id",
        "candidate_demographic_label",
        "split",
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"Manifest is missing required columns: {sorted(missing_cols)}")

    unexpected = set(df["candidate_demographic_label"].dropna().unique()) - set(CLASS_ORDER)
    if unexpected:
        raise RuntimeError(f"Unexpected labels: {sorted(unexpected)}")

    resolved_paths: list[Path | None] = []
    total_bytes = 0
    missing_paths = []

    print("=" * 72)
    print("MATERIALIZE HARMONIZED FACE DATASET")
    print("=" * 72)
    print(f"\nManifest: {manifest}")
    print(f"Records:  {len(df):,}")
    print(f"Mode:     {args.mode}")
    print(f"Output:   {output}")

    print("\nChecking source image files...")
    for idx, row in df.iterrows():
        src = resolve_source_path(row, repo_root)
        resolved_paths.append(src)
        if src is None:
            missing_paths.append(idx)
        else:
            try:
                total_bytes += src.stat().st_size
            except OSError:
                pass
        if (idx + 1) % 10000 == 0:
            print(f"  checked {idx + 1:,}/{len(df):,}")

    if missing_paths:
        raise RuntimeError(
            f"{len(missing_paths)} source image files could not be resolved. "
            "The frozen manifest should normally have 0 missing files."
        )

    print(f"\nEstimated source-image volume: {human_size(total_bytes)}")

    # Build materialized metadata before copying.
    out_rows = []
    split_counts = {"train": 0, "val": 0, "test": 0}

    for i, row in df.iterrows():
        label = str(row["candidate_demographic_label"])
        split = str(row["split"]).lower()
        if split not in split_counts:
            raise RuntimeError(f"Unexpected split value: {split}")

        src = resolved_paths[i]
        assert src is not None
        ext = src.suffix.lower() if src.suffix else ".jpg"
        rec_id = stable_record_id(str(row["source_dataset"]), str(row["image_id"]))
        filename = f"{rec_id}{ext}"
        rel = Path("images") / split / FOLDER_NAME[label] / filename
        dst = output / rel

        out_row = row.to_dict()
        out_row["record_id"] = rec_id
        out_row["materialized_relative_path"] = rel.as_posix()
        out_row["class_index"] = CLASS_ORDER.index(label)
        out_rows.append(out_row)
        split_counts[split] += 1

    metadata_df = pd.DataFrame(out_rows)

    print("\nPlanned split counts:")
    for split in ["train", "val", "test"]:
        print(f"  {split:5s}: {split_counts[split]:,}")

    if args.dry_run:
        print("\nDRY RUN COMPLETE - no files were created.")
        return

    # Create folder structure and copy/link files.
    (output / "metadata").mkdir(parents=True, exist_ok=True)

    print("\nCreating image-folder dataset...")
    for i, row in metadata_df.iterrows():
        src = resolved_paths[i]
        assert src is not None
        dst = output / str(row["materialized_relative_path"])
        copy_or_link(src, dst, args.mode)
        if (i + 1) % 5000 == 0 or (i + 1) == len(metadata_df):
            print(f"  materialized {i + 1:,}/{len(metadata_df):,}")

    # Save metadata alongside the images.
    metadata_df.to_csv(output / "metadata" / "records.csv", index=False)
    shutil.copy2(manifest, output / "metadata" / "training_manifest_frozen.csv")

    class_map = {name: idx for idx, name in enumerate(CLASS_ORDER)}
    (output / "metadata" / "class_to_index.json").write_text(
        json.dumps(class_map, indent=2), encoding="utf-8"
    )

    summary = {
        "dataset_name": "Harmonized Diversity-Aware Face Dataset v1",
        "records": int(len(metadata_df)),
        "mode": args.mode,
        "estimated_image_bytes": int(total_bytes),
        "splits": split_counts,
        "classes": CLASS_ORDER,
        "folder_names": FOLDER_NAME,
        "source_manifest": str(manifest),
    }
    (output / "metadata" / "materialization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    readme = (
        "Harmonized Diversity-Aware Face Dataset v1\n\n"
        f"Records: {len(metadata_df):,}\n"
        "Folder structure: images/<train|val|test>/<class>/image\n"
        "Labels and provenance: metadata/records.csv\n"
        "Frozen source manifest: metadata/training_manifest_frozen.csv\n"
        "\nThis folder contains copied/hard-linked source images. "
        "Verify the source dataset terms before public redistribution.\n"
    )
    (output / "README.txt").write_text(readme, encoding="utf-8")

    print("\n" + "=" * 72)
    print("MATERIALIZATION COMPLETE")
    print("=" * 72)
    print(f"\nStandalone dataset: {output}")
    print(f"Metadata:           {output / 'metadata' / 'records.csv'}")
    print("\nFolder format is directly usable by common ImageFolder-style training loaders.")


if __name__ == "__main__":
    main()
