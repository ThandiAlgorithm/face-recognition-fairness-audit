"""Build metadata/detection_integrity_image.parquet and detection_integrity_by_dataset.csv
for the detection_audit/ scripts (DET-001).

Nothing in the committed pipeline produces these files. This script derives them from
metadata/detection_alignment_diagnostics.parquet and metadata/master_annotations.parquet.

The derivation was proposed and reviewed before running (see conversation record); it is
NOT part of the frozen Phase 0 governance config. Logic, by confidence:

- route: "project_retinaface" for lfw/utkface, "publisher_preprocessing" for fairface.
  High confidence: matches PR_DESCRIPTION_REVISED.md exactly, and matches the data itself
  (inference_success is null for every fairface row, non-null for every lfw/utkface row).
- corrupt: passthrough from master_annotations.parquet.
- detection_status: pre_inference_or_colour_failure > inference_failure > no_detection >
  face_detected, evaluated only for route == project_retinaface rows (fairface gets a
  distinct, honest label so it can never be silently counted as a project-detection
  outcome). Medium confidence: inferred precedence, but every row classifies cleanly with
  no leftover/ambiguous cases in the current data.
- primary_valid: landmarks.notna(). Medium confidence: in the current data this is 100%
  redundant with detection_status == face_detected (no observed case of a detected face
  with unresolved landmarks).
- project_alignment_valid: passthrough of alignment_valid. Medium-high confidence: every
  consumer script already gates on route == project_retinaface before reading this column,
  so FairFace's differently-sourced alignment_valid can never leak in regardless.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

META = Path("metadata")
DIAGNOSTICS = META / "detection_alignment_diagnostics.parquet"
MASTER = META / "master_annotations.parquet"
OUT_IMAGE = META / "detection_integrity_image.parquet"
OUT_DATASET = META / "detection_integrity_by_dataset.csv"

PROJECT_DETECTION_DATASETS = {"lfw", "utkface"}


def main() -> None:
    diag = pd.read_parquet(DIAGNOSTICS)
    master = pd.read_parquet(MASTER)

    if not diag["image_id"].is_unique or not master["image_id"].is_unique:
        raise RuntimeError("Duplicate image IDs in an input table.")

    df = diag.merge(master[["image_id", "dataset", "corrupt"]], on="image_id", how="left", validate="one_to_one", suffixes=("", "_master"))
    if "dataset" not in df.columns:
        raise RuntimeError("dataset column missing after merge.")

    df["route"] = np.where(df["dataset"].isin(PROJECT_DETECTION_DATASETS), "project_retinaface", "publisher_preprocessing")

    is_project = df["route"].eq("project_retinaface")
    colour_failed = df["colour_conversion_success"].isna()
    inference_failed = df["inference_success"].eq(False)
    no_detection = df["no_detection"].eq(True)

    status = pd.Series("not_applicable_publisher_route", index=df.index, dtype=object)
    status = status.where(~is_project, "face_detected")
    status = status.mask(is_project & no_detection, "no_detection")
    status = status.mask(is_project & inference_failed, "inference_failure")
    status = status.mask(is_project & colour_failed, "pre_inference_or_colour_failure")
    df["detection_status"] = status

    df["primary_valid"] = df["landmarks"].notna().astype("boolean")
    df["project_alignment_valid"] = df["alignment_valid"].astype("boolean")

    image = df[["image_id", "dataset", "route", "corrupt", "detection_status", "n_detections", "det_confidence", "primary_valid", "project_alignment_valid"]].copy()

    if len(image) != len(master):
        raise RuntimeError(f"Row count mismatch: {len(image)} vs master {len(master)}")
    if not image["image_id"].is_unique:
        raise RuntimeError("Duplicate image IDs in output.")

    # Reconciliation: every project-route row must land in exactly one known status.
    proj = image.loc[image["route"].eq("project_retinaface")]
    known = {"face_detected", "no_detection", "inference_failure", "pre_inference_or_colour_failure"}
    unexplained = proj.loc[~proj["detection_status"].isin(known)]
    if len(unexplained):
        raise RuntimeError(f"{len(unexplained)} project-route rows have an unexplained detection_status.")

    OUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    image.to_parquet(OUT_IMAGE, index=False)

    by_dataset = image.groupby(["dataset", "route", "detection_status"], dropna=False).size().reset_index(name="n")
    by_dataset.to_csv(OUT_DATASET, index=False)

    print("Rows:", len(image))
    print()
    print("Detection status by dataset:")
    print(by_dataset.to_string(index=False))
    print()
    print("Saved:", OUT_IMAGE)
    print("Saved:", OUT_DATASET)


if __name__ == "__main__":
    main()
