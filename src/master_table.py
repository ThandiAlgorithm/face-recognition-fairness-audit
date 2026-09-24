from __future__ import annotations

import hashlib

import pandas as pd

from src import categories


MASTER_COLUMNS = [
    "image_id",
    "dataset",
    "dataset_version",
    "original_path",
    "split",
    "subject_id",
    "width_px",
    "height_px",
    "original_mode",
    "colour_profile",
    "colour_profile_assumed",
    "is_greyscale",
    "corrupt",
    "parse_failed",
    "gender",
    "gender_source",
    "gender_confidence",
    "race",
    "race_source",
    "race_confidence",
    "age",
    "age_source",
    "age_confidence",
    "age_band",
    "exact_dup_group",
    "near_dup_group",
    "near_dup_status",
    "cross_dataset_dup",
    "cross_dataset_dup_status",
    "bbox",
    "det_confidence",
    "n_detections",
    "no_detection",
    "multi_detection",
    "detections_all",
    "landmarks",
    "alignment_transform",
    "aligned_path",
    "alignment_valid",
    "brightness",
    "contrast",
    "sharpness",
    "face_px_area",
    "face_area_ratio",
    "yaw",
    "pitch",
    "roll",
    "L_star",
    "a_star",
    "b_star",
    "ITA",
    "hue_angle",
    "ita_group",
    "ita_ordinal",
    "fitz_classifier",
    "fitz_classifier_conf",
    "fitz_ordinal",
    "skin_group_ita",
    "skin_group_fitz",
    "skin_roi_valid",
    "ita_undefined",
    "low_confidence",
    "thin_cell",
    "exclusion_reason",
    "eligible_detection_audit",
    "eligible_visual_audit",
    "eligible_skin_audit",
    "eligible_demographic_audit",
    "eligible_cross_dataset_eval",
    "eligible_lfw_pair_eval",
]


def stable_image_id(dataset: str, dataset_version: str, original_path: str) -> str:
    payload = f"{dataset_version}|{original_path}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{dataset}-{digest}"


def build_master(
    manifest: pd.DataFrame,
    integrity: pd.DataFrame,
    labels: pd.DataFrame,
    duplicates: pd.DataFrame,
    preprocessing_cfg: dict,
) -> pd.DataFrame:
    images = manifest[manifest["file_role"].eq("image_candidate")].copy()
    base = images[
        [
            "dataset",
            "dataset_version",
            "variant",
            "original_path",
            "source_preprocessed",
            "project_detection_required",
            "project_alignment_required",
        ]
    ].copy()

    base["image_id"] = base.apply(
        lambda row: stable_image_id(
            row["dataset"],
            row["dataset_version"],
            row["original_path"],
        ),
        axis=1,
    )
    if not base["image_id"].is_unique:
        raise RuntimeError("Stable image IDs are not unique.")

    master = base.merge(
        integrity,
        on="original_path",
        how="left",
        validate="one_to_one",
    ).merge(
        labels,
        on="original_path",
        how="left",
        validate="one_to_one",
    ).merge(
        duplicates,
        on="original_path",
        how="left",
        validate="one_to_one",
    )

    master["corrupt"] = ~master["decode_success"].fillna(False)
    master["parse_failed"] = master["parse_failed"].fillna(True)
    master["split"] = master["split"].fillna("none")
    master["gender"] = master["gender"].fillna("unknown")
    master["gender_source"] = master["gender_source"].fillna("unavailable")
    master["race"] = master["race"].fillna("unknown")
    master["race_source"] = master["race_source"].fillna("unavailable")
    master["age_source"] = master["age_source"].fillna("unavailable")
    master["near_dup_status"] = master["near_dup_status"].fillna("none")
    master["cross_dataset_dup"] = master["cross_dataset_dup"].fillna(False)
    master["cross_dataset_dup_status"] = master["cross_dataset_dup_status"].fillna("none")

    nullable_columns = [
        "bbox",
        "det_confidence",
        "n_detections",
        "no_detection",
        "multi_detection",
        "detections_all",
        "landmarks",
        "alignment_transform",
        "aligned_path",
        "alignment_valid",
        "brightness",
        "contrast",
        "sharpness",
        "face_px_area",
        "face_area_ratio",
        "yaw",
        "pitch",
        "roll",
        "L_star",
        "a_star",
        "b_star",
        "ITA",
        "hue_angle",
        "ita_group",
        "ita_ordinal",
        "fitz_classifier",
        "fitz_classifier_conf",
        "fitz_ordinal",
        "skin_group_ita",
        "skin_group_fitz",
        "skin_roi_valid",
        "ita_undefined",
        "low_confidence",
        "thin_cell",
        "exclusion_reason",
        "eligible_detection_audit",
        "eligible_visual_audit",
        "eligible_skin_audit",
        "eligible_demographic_audit",
        "eligible_cross_dataset_eval",
        "eligible_lfw_pair_eval",
    ]
    for column in nullable_columns:
        if column not in master.columns:
            master[column] = None

    supplied = master["source_preprocessed"].fillna(False) & master["decode_success"].fillna(False)
    master.loc[supplied, "alignment_valid"] = True
    master.loc[supplied, "aligned_path"] = master.loc[supplied, "original_path"]

    return master


def apply_utkface_split(master: pd.DataFrame, split_table: pd.DataFrame) -> pd.DataFrame:
    output = master.copy()
    mapping = split_table.set_index("image_id")["split"]
    mask = output["dataset"].eq("utkface")
    output.loc[mask, "split"] = output.loc[mask, "image_id"].map(mapping)
    if output.loc[mask, "split"].isna().any():
        raise RuntimeError("UTKFace split assignment is incomplete.")
    return output


def apply_detection_updates(master: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    if updates.empty:
        return master.copy()
    if updates["image_id"].duplicated().any():
        raise RuntimeError("Detection updates contain duplicate image IDs.")
    output = master.set_index("image_id").copy()
    incoming = updates.set_index("image_id")
    common = output.index.intersection(incoming.index)
    for column in incoming.columns:
        if column not in output.columns:
            continue
        output.loc[common, column] = incoming.loc[common, column]
    return output.reset_index()


def apply_measurement_categories(master: pd.DataFrame, categories_cfg: dict) -> pd.DataFrame:
    output = master.copy()
    output["ita_group"] = output["ITA"].apply(lambda value: categories.assign_ita_group(value, categories_cfg))
    output["ita_ordinal"] = output["ITA"].apply(lambda value: categories.assign_ita_ordinal(value, categories_cfg))
    output["skin_group_ita"] = output["ITA"].apply(lambda value: categories.assign_ita_binary(value, categories_cfg))
    return output


def _primary_exclusion_reason(row: pd.Series, exclusions_cfg: dict) -> str | None:
    checks = {
        "corrupt_file": bool(row.get("corrupt", False)),
        "decode_failure": row.get("decode_success") is False,
        "label_parse_failure": bool(row.get("parse_failed", False)),
        "missing_required_label": row.get("gender", "unknown") == "unknown",
        "low_confidence_label": bool(row.get("low_confidence", False)),
        "zero_detection": row.get("no_detection") is True,
        "multi_detection_unresolved": False,
        "alignment_failure": row.get("alignment_valid") is False,
        "invalid_skin_roi": row.get("skin_roi_valid") is False and row.get("is_greyscale") is not True,
        "greyscale_no_chroma": row.get("is_greyscale") is True,
        "ita_undefined": row.get("ita_undefined") is True,
        "exact_duplicate": pd.notna(row.get("exact_dup_group")),
        "near_duplicate_confirmed": row.get("near_dup_status") == "confirmed",
        "cross_dataset_duplicate_confirmed": row.get("cross_dataset_dup") is True,
        "thin_cell": bool(row.get("thin_cell", False)),
    }
    for reason in exclusions_cfg["exclusion_precedence"]:
        if checks.get(reason, False):
            return reason
    return None


def apply_eligibility(
    master: pd.DataFrame,
    manifest: pd.DataFrame,
    exclusions_cfg: dict,
) -> pd.DataFrame:
    output = master.copy()
    source = manifest.loc[
        manifest["file_role"].eq("image_candidate"),
        ["original_path", "source_preprocessed", "project_detection_required"],
    ]
    needed = [
        column
        for column in ["source_preprocessed", "project_detection_required"]
        if column not in output.columns
    ]
    if needed:
        output = output.merge(
            source[["original_path", *needed]],
            on="original_path",
            how="left",
            validate="one_to_one",
        )

    threshold = float(exclusions_cfg["low_confidence_labels"]["threshold"])
    native = output["gender_source"].eq("native")
    confidence_values = pd.to_numeric(output["gender_confidence"], errors="coerce")
    confidence_ok = confidence_values.ge(threshold).fillna(False)
    output["low_confidence"] = (
        output["gender"].ne("unknown")
        & ~native
        & ~confidence_ok
    )

    output["eligible_detection_audit"] = ~output["corrupt"].eq(True)
    project_required = output["project_detection_required"].eq(True)
    n_detections = pd.to_numeric(output["n_detections"], errors="coerce")
    detector_ok = ~project_required | n_detections.ge(1).fillna(False)
    output["eligible_visual_audit"] = (
        output["eligible_detection_audit"]
        & detector_ok
        & output["alignment_valid"].eq(True)
    )
    output["eligible_skin_audit"] = (
        output["eligible_visual_audit"]
        & output["skin_roi_valid"].eq(True)
        & ~output["is_greyscale"].eq(True)
        & ~output["ita_undefined"].eq(True)
    )
    output["eligible_demographic_audit"] = (
        output["gender"].ne("unknown")
        & (native | confidence_ok)
    )
    output["eligible_cross_dataset_eval"] = (
        output["eligible_skin_audit"]
        & ~output["cross_dataset_dup"].eq(True)
    )
    output["eligible_lfw_pair_eval"] = False
    output["exclusion_reason"] = output.apply(
        lambda row: _primary_exclusion_reason(row, exclusions_cfg),
        axis=1,
    )

    output = output.drop(
        columns=["source_preprocessed", "project_detection_required"],
        errors="ignore",
    )
    return output


def finalise_master(master: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in MASTER_COLUMNS if column not in master.columns]
    if missing:
        raise RuntimeError(f"Master table is missing GOV-005 columns: {missing}")
    extra = [column for column in master.columns if column not in MASTER_COLUMNS]
    if extra:
        master = master.drop(columns=extra)
    return master[MASTER_COLUMNS].copy()
