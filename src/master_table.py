
from __future__ import annotations

import hashlib

import pandas as pd

from src import paths

# Dataset preprocessing provenance.
PROVENANCE = {
    "fairface": {
        "source_preprocessed": True,
        "source_preprocessing_method": "publisher_dlib_crop",
        "project_detection_applied": False,
        "project_alignment_applied": False,
    },
    "utkface": {
        "source_preprocessed": True,
        "source_preprocessing_method": "publisher_aligned_cropped",
        "project_detection_applied": False,
        "project_alignment_applied": False,
    },
    "lfw": {
        "source_preprocessed": False,
        "source_preprocessing_method": "none_original_unaligned",
        "project_detection_applied": True,
        "project_alignment_applied": True,
    },
}


def stable_image_id(dataset: str, relative_path: str) -> str:
    return hashlib.sha256(f"{dataset}::{relative_path}".encode("utf-8")).hexdigest()


def build_fairface_rows(fairface_categorised: pd.DataFrame) -> pd.DataFrame:
    df = fairface_categorised.copy()
    df["relative_path"] = df["file"].apply(lambda f: f"FairFace/{f}")
    df["original_filename"] = df["file"].apply(lambda f: f.split("/")[-1])
    df["subject_id"] = None
    return df


def build_utkface_rows(utkface_categorised: pd.DataFrame) -> pd.DataFrame:
    df = utkface_categorised.copy()
    df["relative_path"] = "UTKFace/UTKFace/" + df["relative_path"]
    df["subject_id"] = None
    return df


def build_lfw_rows(lfw_categorised: pd.DataFrame) -> pd.DataFrame:
    df = lfw_categorised.copy()
    df["relative_path"] = df["original_path"]
    df["original_filename"] = df["relative_path"].apply(lambda p: p.split("/")[-1])
    return df


def build_base_master_table(
    fairface_categorised: pd.DataFrame,
    utkface_categorised: pd.DataFrame,
    lfw_categorised: pd.DataFrame,
    image_integrity_df: pd.DataFrame,
    hashes_df: pd.DataFrame,
    cross_dataset_overlap_paths: set,
) -> pd.DataFrame:
    fairface_rows = build_fairface_rows(fairface_categorised)
    utkface_rows = build_utkface_rows(utkface_categorised)
    lfw_rows = build_lfw_rows(lfw_categorised)

    common_cols = [
        "dataset", "relative_path", "original_filename", "subject_id", "split",
        "age_native", "age_band_fine", "age_band_coarse", "age_source",
        "gender_native", "gender_code", "gender_source",
        "race_native", "race_source",
    ]

    frames = []
    for df in (fairface_rows, utkface_rows, lfw_rows):
        for col in common_cols:
            if col not in df.columns:
                df[col] = None
        frames.append(df[common_cols])

    master = pd.concat(frames, ignore_index=True)
    master["age_numeric"] = None
    master.loc[master["dataset"] == "utkface", "age_numeric"] = utkface_categorised["age_native"].values
    # Use one type for age values.
    master["age_native"] = master["age_native"].apply(lambda x: None if pd.isna(x) else str(x))
    master["gender_standard"] = master["gender_code"].map({1: "female", 0: "male"})
    master["age_source"] = master["dataset"].map(
        {"fairface": "native", "utkface": "native", "lfw": "unknown_no_metadata_supplied"}
    )

    master["image_id"] = master.apply(lambda r: stable_image_id(r["dataset"], r["relative_path"]), axis=1)

    integrity_small = image_integrity_df[
        ["path", "decode_success", "decoded_width", "decoded_height", "original_mode"]
    ].rename(columns={"path": "relative_path", "decoded_width": "width", "decoded_height": "height", "original_mode": "mode"})
    master = master.merge(integrity_small, on="relative_path", how="left")
    master["corrupt"] = ~master["decode_success"].fillna(False)
    master["format"] = master["relative_path"].apply(lambda p: p.rsplit(".", 1)[-1].lower() if "." in p else None)

    # Join duplicate flags.
    dup_small = hashes_df[["path", "exact_duplicate_group"]].rename(columns={"path": "relative_path"})
    master = master.merge(dup_small, on="relative_path", how="left")
    master["exact_duplicate"] = master["exact_duplicate_group"].notna()
    master["near_duplicate_candidate"] = None  # 9.2 cut, not run this pass
    master["cross_dataset_overlap"] = master["relative_path"].isin(cross_dataset_overlap_paths)

    # Add preprocessing provenance.
    for field in ["source_preprocessed", "source_preprocessing_method", "project_detection_applied", "project_alignment_applied"]:
        master[field] = master["dataset"].map(lambda d: PROVENANCE[d][field])

    # Reserve feature columns.
    for col in [
        "detection_success", "number_of_faces", "detection_confidence", "bbox",
        "alignment_success", "pilot_run",
        "brightness", "contrast", "sharpness", "face_bbox_area_px", "face_area_ratio",
        "yaw", "pitch", "roll",
        "skin_roi_valid", "skin_valid_fraction", "L_star", "a_star", "b_star",
        "ITA", "ita_undefined", "hue_angle",
    ]:
        master[col] = None
    master["pilot_run"] = False

    return master
