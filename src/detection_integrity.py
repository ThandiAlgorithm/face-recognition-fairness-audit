from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_MASTER_COLUMNS = {
    "image_id",
    "dataset",
    "original_path",
    "subject_id",
    "corrupt",
    "gender",
    "gender_source",
    "race",
    "race_source",
    "age_band",
    "det_confidence",
    "n_detections",
    "no_detection",
    "multi_detection",
    "bbox",
    "landmarks",
    "alignment_valid",
}

REQUIRED_MANIFEST_COLUMNS = {
    "file_role",
    "original_path",
    "source_preprocessed",
    "project_detection_required",
    "project_alignment_required",
}

REQUIRED_DIAGNOSTIC_COLUMNS = {
    "image_id",
    "inference_success",
    "inference_error",
}


@dataclass(frozen=True)
class DetectionAuditInputs:
    master: pd.DataFrame
    manifest: pd.DataFrame
    diagnostics: pd.DataFrame


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    name: str,
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(
            f"{name} is missing required columns: {missing}"
        )


def _is_valid_landmarks(value) -> bool:
    if value is None:
        return False

    try:
        points = list(value)
    except TypeError:
        return False

    if len(points) != 5:
        return False

    for point in points:
        try:
            coords = list(point)
        except TypeError:
            return False

        if len(coords) != 2:
            return False

        try:
            x = float(coords[0])
            y = float(coords[1])
        except (TypeError, ValueError):
            return False

        if not np.isfinite(x) or not np.isfinite(y):
            return False

    return True


def prepare_detection_frame(
    inputs: DetectionAuditInputs,
) -> pd.DataFrame:
    """
    Build the analysis frame from completed Phase 0 outputs.

    This function does not run RetinaFace and does not perform
    inferential statistics.
    """

    master = inputs.master.copy()
    manifest = inputs.manifest.copy()
    diagnostics = inputs.diagnostics.copy()

    _require_columns(
        master,
        REQUIRED_MASTER_COLUMNS,
        "master_annotations.parquet",
    )
    _require_columns(
        manifest,
        REQUIRED_MANIFEST_COLUMNS,
        "file_manifest.parquet",
    )
    _require_columns(
        diagnostics,
        REQUIRED_DIAGNOSTIC_COLUMNS,
        "detection_alignment_diagnostics.parquet",
    )

    if not master["image_id"].is_unique:
        raise RuntimeError("Master image IDs must be unique.")

    if not diagnostics["image_id"].is_unique:
        raise RuntimeError("Diagnostic image IDs must be unique.")

    if (
        master["image_id"].isna().any()
        or diagnostics["image_id"].isna().any()
    ):
        raise RuntimeError("Image IDs cannot be null.")

    if set(master["image_id"]) != set(diagnostics["image_id"]):
        raise RuntimeError(
            "Master and diagnostics must cover identical image IDs."
        )

    manifest_images = manifest.loc[
        manifest["file_role"].eq("image_candidate"),
        [
            "original_path",
            "source_preprocessed",
            "project_detection_required",
            "project_alignment_required",
        ],
    ].copy()

    if not manifest_images["original_path"].is_unique:
        raise RuntimeError(
            "Manifest image paths must be unique."
        )

    frame = master.merge(
        manifest_images,
        on="original_path",
        how="left",
        validate="one_to_one",
    )

    if frame["project_detection_required"].isna().any():
        raise RuntimeError(
            "Some master images have no manifest record."
        )

    diagnostic_columns = [
        column
        for column in [
            "image_id",
            "inference_success",
            "inference_error",
            "colour_conversion_success",
            "colour_conversion_error",
        ]
        if column in diagnostics.columns
    ]

    # The completed master already contains some diagnostic
    # columns. Verify matching values before merging.
    overlap = [
        column
        for column in diagnostic_columns
        if column != "image_id" and column in frame.columns
    ]

    if overlap:
        master_by_id = frame.set_index(
            "image_id"
        ).sort_index()

        diag_by_id = diagnostics.set_index(
            "image_id"
        ).sort_index()

        for column in overlap:
            master_values = (
                master_by_id[column]
                .astype("string")
                .fillna("<NULL>")
            )
            diag_values = (
                diag_by_id[column]
                .astype("string")
                .fillna("<NULL>")
            )

            if not master_values.equals(diag_values):
                raise RuntimeError(
                    "Production master disagrees with diagnostics: "
                    f"{column}"
                )

        frame = frame.drop(columns=overlap)

    frame = frame.merge(
        diagnostics[diagnostic_columns],
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    # Identify which images actually entered project RetinaFace.
    frame["project_detection_required"] = (
        frame["project_detection_required"]
        .fillna(False)
        .astype(bool)
    )

    frame["project_alignment_required"] = (
        frame["project_alignment_required"]
        .fillna(False)
        .astype(bool)
    )

    frame["source_preprocessed"] = (
        frame["source_preprocessed"]
        .fillna(False)
        .astype(bool)
    )

    frame["corrupt"] = (
        frame["corrupt"]
        .fillna(True)
        .astype(bool)
    )

    frame["entered_detector"] = (
        frame["project_detection_required"]
        & ~frame["corrupt"]
    )

    frame["detector_inference_error"] = (
        frame["entered_detector"]
        & ~frame["inference_success"].eq(True)
    )

    n_detections = pd.to_numeric(
        frame["n_detections"],
        errors="coerce",
    )

    frame["zero_detection"] = (
        frame["entered_detector"]
        & frame["inference_success"].eq(True)
        & n_detections.eq(0)
    )

    frame["detected"] = (
        frame["entered_detector"]
        & frame["inference_success"].eq(True)
        & n_detections.ge(1)
    )

    frame["multi_detection_flag"] = (
        frame["detected"]
        & n_detections.gt(1)
    )

    frame["primary_selected"] = (
        frame["detected"]
        & frame["bbox"].notna()
        & pd.to_numeric(
            frame["det_confidence"],
            errors="coerce",
        ).notna()
    )

    frame["primary_selection_failure"] = (
        frame["detected"]
        & ~frame["primary_selected"]
    )

    frame["landmark_valid"] = (
        frame["landmarks"].map(_is_valid_landmarks)
    )

    frame["landmark_failure"] = (
        frame["primary_selected"]
        & ~frame["landmark_valid"]
    )

    frame["alignment_attempted"] = (
        frame["project_alignment_required"]
        & frame["primary_selected"]
        & frame["landmark_valid"]
    )

    frame["alignment_failure"] = (
        frame["alignment_attempted"]
        & ~frame["alignment_valid"].eq(True)
    )

    frame["aligned_success"] = (
        frame["alignment_attempted"]
        & frame["alignment_valid"].eq(True)
    )

    frame["detected_no_project_alignment"] = (
        frame["detected"]
        & ~frame["project_alignment_required"]
    )

    frame["publisher_preprocessed_not_project_detected"] = (
        frame["source_preprocessed"]
        & ~frame["project_detection_required"]
    )

    # Every canonical image receives one terminal state.
    state = pd.Series(
        "unclassified",
        index=frame.index,
        dtype="object",
    )

    state.loc[
        frame["publisher_preprocessed_not_project_detected"]
        & ~frame["corrupt"]
    ] = "publisher_preprocessed_not_project_detected"

    state.loc[
        frame["corrupt"]
    ] = "corrupt_not_processed"

    state.loc[
        frame["detector_inference_error"]
    ] = "inference_error"

    state.loc[
        frame["zero_detection"]
    ] = "zero_detection"

    state.loc[
        frame["primary_selection_failure"]
    ] = "primary_selection_failure"

    state.loc[
        frame["landmark_failure"]
    ] = "landmark_failure"

    state.loc[
        frame["alignment_failure"]
    ] = "alignment_failure"

    state.loc[
        frame["aligned_success"]
    ] = "aligned_success"

    state.loc[
        frame["detected_no_project_alignment"]
    ] = "detected_no_project_alignment"

    frame["detector_terminal_state"] = state

    if frame["detector_terminal_state"].eq("unclassified").any():
        sample = frame.loc[
            frame["detector_terminal_state"].eq("unclassified"),
            ["image_id", "dataset", "original_path"],
        ].head(10)

        raise RuntimeError(
            "Unclassified detector outcomes:\n"
            f"{sample.to_string(index=False)}"
        )

    return frame


def build_detection_integrity_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Reconcile dataset-level detection outcomes."""

    rows = []

    for dataset, group in frame.groupby(
        "dataset",
        dropna=False,
        sort=True,
    ):
        entered = int(group["entered_detector"].sum())
        inference_error = int(
            group["detector_inference_error"].sum()
        )
        zero = int(group["zero_detection"].sum())
        detected = int(group["detected"].sum())

        if entered != inference_error + zero + detected:
            raise RuntimeError(
                f"Detection integrity reconciliation failed for {dataset}."
            )

        project_alignment_detected = group.loc[
            group["detected"]
            & group["project_alignment_required"]
        ]

        if len(project_alignment_detected):
            reconciled = (
                int(
                    project_alignment_detected[
                        "primary_selection_failure"
                    ].sum()
                )
                + int(
                    project_alignment_detected[
                        "landmark_failure"
                    ].sum()
                )
                + int(
                    project_alignment_detected[
                        "alignment_failure"
                    ].sum()
                )
                + int(
                    project_alignment_detected[
                        "aligned_success"
                    ].sum()
                )
            )

            if reconciled != len(project_alignment_detected):
                raise RuntimeError(
                    f"Alignment reconciliation failed for {dataset}."
                )

        rows.append(
            {
                "dataset": dataset,
                "canonical_images": int(len(group)),
                "source_preprocessed": int(
                    group["source_preprocessed"].sum()
                ),
                "project_detection_required": int(
                    group["project_detection_required"].sum()
                ),
                "corrupt_not_processed": int(
                    group["corrupt"].sum()
                ),
                "entered_detector": entered,
                "inference_success": int(
                    (
                        group["entered_detector"]
                        & group["inference_success"].eq(True)
                    ).sum()
                ),
                "inference_error": inference_error,
                "detected_ge1": detected,
                "zero_detection": zero,
                "multi_detection": int(
                    group["multi_detection_flag"].sum()
                ),
                "primary_selection_failure": int(
                    group["primary_selection_failure"].sum()
                ),
                "landmark_failure": int(
                    group["landmark_failure"].sum()
                ),
                "alignment_attempted": int(
                    group["alignment_attempted"].sum()
                ),
                "alignment_failure": int(
                    group["alignment_failure"].sum()
                ),
                "aligned_success": int(
                    group["aligned_success"].sum()
                ),
                "publisher_preprocessed_not_project_detected": int(
                    group[
                        "publisher_preprocessed_not_project_detected"
                    ].sum()
                ),
                "detector_reconciles": True,
            }
        )

    return pd.DataFrame(rows)


def build_terminal_state_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return (
        frame.groupby(
            ["dataset", "detector_terminal_state"],
            dropna=False,
        )
        .size()
        .rename("n")
        .reset_index()
        .sort_values(
            ["dataset", "detector_terminal_state"],
            ignore_index=True,
        )
    )


def _valid_subgroup_rows(
    frame: pd.DataFrame,
    axis: str,
) -> pd.DataFrame:
    if axis not in {"gender", "age_band", "race"}:
        raise KeyError(f"Unsupported subgroup axis: {axis}")

    values = frame[axis].astype("string")

    valid = (
        values.notna()
        & ~values.str.lower().isin(
            {"unknown", "none", "nan", ""}
        )
    )

    # Race labels are retained in their native dataset scheme.
    if axis == "race":
        valid &= (
            frame["race_source"]
            .astype("string")
            .str.lower()
            .eq("native")
        )

    return frame.loc[valid].copy()


def build_subgroup_failure_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Descriptive subgroup detection failures.

    Confidence intervals and inferential tests must be added
    using the approved statistical protocol.
    """

    source = frame.loc[
        frame["entered_detector"]
    ].copy()

    source["detection_failure"] = (
        source["detector_inference_error"]
        | source["zero_detection"]
    )

    rows = []

    for axis in ["gender", "age_band", "race"]:
        eligible = _valid_subgroup_rows(source, axis)

        for (dataset, subgroup), group in eligible.groupby(
            ["dataset", axis],
            dropna=False,
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "axis": axis,
                    "subgroup": subgroup,
                    "n": int(len(group)),
                    "detection_failures": int(
                        group["detection_failure"].sum()
                    ),
                    "inference_errors": int(
                        group["detector_inference_error"].sum()
                    ),
                    "zero_detections": int(
                        group["zero_detection"].sum()
                    ),
                    "detection_failure_rate": float(
                        group["detection_failure"].mean()
                    ),
                    "race_native_only": axis == "race",
                }
            )

    return pd.DataFrame(rows)


def build_confidence_rows(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Confidence summaries among detected faces only.

    Zero-detection cases have no primary-face confidence and
    must not be represented as zero-confidence detections.
    """

    source = frame.loc[
        frame["detected"]
        & frame["primary_selected"]
        & pd.to_numeric(
            frame["det_confidence"],
            errors="coerce",
        ).notna()
    ].copy()

    source["det_confidence"] = pd.to_numeric(
        source["det_confidence"],
        errors="coerce",
    )

    raw_columns = [
        "image_id",
        "dataset",
        "subject_id",
        "gender",
        "gender_source",
        "race",
        "race_source",
        "age_band",
        "det_confidence",
    ]

    raw = source[raw_columns].copy()

    rows = []

    for axis in ["gender", "age_band", "race"]:
        eligible = _valid_subgroup_rows(source, axis)

        for (dataset, subgroup), group in eligible.groupby(
            ["dataset", axis],
            dropna=False,
        ):
            values = (
                group["det_confidence"]
                .dropna()
                .astype(float)
            )

            rows.append(
                {
                    "dataset": dataset,
                    "axis": axis,
                    "subgroup": subgroup,
                    "n": int(len(values)),
                    "mean": float(values.mean())
                    if len(values) else np.nan,
                    "std": float(values.std(ddof=1))
                    if len(values) > 1 else np.nan,
                    "median": float(values.median())
                    if len(values) else np.nan,
                    "q1": float(values.quantile(0.25))
                    if len(values) else np.nan,
                    "q3": float(values.quantile(0.75))
                    if len(values) else np.nan,
                    "minimum": float(values.min())
                    if len(values) else np.nan,
                    "maximum": float(values.max())
                    if len(values) else np.nan,
                    "race_native_only": axis == "race",
                }
            )

    return pd.DataFrame(rows), raw


def build_alignment_failure_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distinguish landmark and project-alignment failures.

    FairFace's publisher alignment is not counted as a
    project alignment success or failure.
    """

    source = frame.loc[
        frame["project_alignment_required"]
        & frame["detected"]
        & ~frame["primary_selection_failure"]
    ].copy()

    source["landmark_or_alignment_failure"] = (
        source["landmark_failure"]
        | source["alignment_failure"]
    )

    rows = []

    for axis in ["gender", "age_band", "race"]:
        eligible = _valid_subgroup_rows(source, axis)

        for (dataset, subgroup), group in eligible.groupby(
            ["dataset", axis],
            dropna=False,
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "axis": axis,
                    "subgroup": subgroup,
                    "n_detected_primary": int(len(group)),
                    "landmark_failures": int(
                        group["landmark_failure"].sum()
                    ),
                    "alignment_failures": int(
                        group["alignment_failure"].sum()
                    ),
                    "landmark_or_alignment_failures": int(
                        group[
                            "landmark_or_alignment_failure"
                        ].sum()
                    ),
                    "landmark_failure_rate": float(
                        group["landmark_failure"].mean()
                    ),
                    "alignment_failure_rate": float(
                        group["alignment_failure"].mean()
                    ),
                    "combined_failure_rate": float(
                        group[
                            "landmark_or_alignment_failure"
                        ].mean()
                    ),
                    "race_native_only": axis == "race",
                }
            )

    return pd.DataFrame(rows)


def build_provenance_notes(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Record dataset provenance alongside the audit."""

    rows = []

    for dataset, group in frame.groupby(
        "dataset",
        sort=True,
    ):
        rows.append(
            {
                "dataset": dataset,
                "canonical_images": int(len(group)),
                "source_preprocessed_images": int(
                    group["source_preprocessed"].sum()
                ),
                "project_detection_images": int(
                    group["project_detection_required"].sum()
                ),
                "project_alignment_images": int(
                    group["project_alignment_required"].sum()
                ),
                "gender_sources": ";".join(
                    sorted(
                        group["gender_source"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                ),
                "race_sources": ";".join(
                    sorted(
                        group["race_source"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )
                ),
                "note": (
                    "Publisher-preprocessed images are not "
                    "treated as project RetinaFace detections."
                ),
            }
        )

    return pd.DataFrame(rows)
