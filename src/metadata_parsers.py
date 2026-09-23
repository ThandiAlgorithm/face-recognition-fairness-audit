from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src import categories, paths

_UTKFACE_PATTERN = re.compile(
    r"^(?P<age>\d+)_(?P<gender>\d+)_(?P<race>\d+)_(?P<timestamp>\d+)(?P<suffix>.*)$",
    re.IGNORECASE,
)

_UTKFACE_RACE = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others",
}

_UTKFACE_GENDER = {0: "male", 1: "female"}


def parse_utkface_filename(filename: str) -> dict:
    match = _UTKFACE_PATTERN.match(filename)
    lowered = filename.lower()
    is_chip = ".chip." in lowered or "aligned" in lowered or "cropped" in lowered
    if match is None:
        return {
            "parse_failed": True,
            "parse_error": "filename does not contain the expected UTKFace fields",
            "is_disallowed_variant": is_chip,
            "age": None,
            "gender": "unknown",
            "race": "unknown",
        }

    age = int(match.group("age"))
    gender_code = int(match.group("gender"))
    race_code = int(match.group("race"))
    errors = []
    if not 0 <= age <= 116:
        errors.append("age_out_of_range")
    if gender_code not in _UTKFACE_GENDER:
        errors.append("unknown_gender_code")
    if race_code not in _UTKFACE_RACE:
        errors.append("unknown_race_code")
    if is_chip:
        errors.append("disallowed_aligned_or_cropped_variant")

    return {
        "parse_failed": bool(errors),
        "parse_error": ";".join(errors) if errors else None,
        "is_disallowed_variant": is_chip,
        "age": age if 0 <= age <= 116 else None,
        "gender": _UTKFACE_GENDER.get(gender_code, "unknown"),
        "race": _UTKFACE_RACE.get(race_code, "unknown"),
    }


def _fairface_labels(preprocessing_cfg: dict) -> pd.DataFrame:
    rows = []
    dataset_cfg = preprocessing_cfg["datasets"]["fairface"]
    for item in dataset_cfg.get("labels", []):
        csv_path = paths.DATA_DIR / item["path"]
        split = item["split"]
        if not csv_path.exists():
            raise FileNotFoundError(f"FairFace label file not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        frame["split"] = split
        rows.append(frame)
    if not rows:
        raise RuntimeError("No FairFace label files are configured.")
    return pd.concat(rows, ignore_index=True)


def build_label_table(
    manifest: pd.DataFrame,
    preprocessing_cfg: dict,
    categories_cfg: dict,
) -> pd.DataFrame:
    images = manifest[manifest["file_role"].eq("image_candidate")].copy()
    output_rows = []

    fairface_labels = _fairface_labels(preprocessing_cfg)
    fairface_lookup = {}
    for _, row in fairface_labels.iterrows():
        original_path = (Path("FairFace") / str(row["file"])).as_posix()
        fairface_lookup[original_path] = row

    utkface_age_bands = categories_cfg["age_bands"]

    for _, image in images.iterrows():
        dataset = image["dataset"]
        original_path = image["original_path"]
        row = {
            "original_path": original_path,
            "subject_id": None,
            "split": "none",
            "gender": "unknown",
            "gender_source": "unavailable",
            "gender_confidence": None,
            "race": "unknown",
            "race_source": "unavailable",
            "race_confidence": None,
            "age": None,
            "age_source": "unavailable",
            "age_confidence": None,
            "age_band": None,
            "parse_failed": False,
            "parse_error": None,
        }

        if dataset == "fairface":
            label = fairface_lookup.get(original_path)
            if label is None:
                row["parse_failed"] = True
                row["parse_error"] = "image has no FairFace label row"
            else:
                row["split"] = str(label["split"])
                row["gender"] = categories.normalize_gender(label.get("gender"))
                row["gender_source"] = "native"
                row["race"] = str(label.get("race", "unknown"))
                row["race_source"] = "native"
                row["age_band"] = categories.fairface_age_band(label.get("age"))
                row["age_source"] = "native"
                if row["gender"] == "unknown" or row["age_band"] is None:
                    row["parse_failed"] = True
                    row["parse_error"] = "unexpected FairFace demographic value"

        elif dataset == "utkface":
            parsed = parse_utkface_filename(Path(original_path).name)
            row["parse_failed"] = parsed["parse_failed"]
            row["parse_error"] = parsed["parse_error"]
            row["gender"] = parsed["gender"]
            row["gender_source"] = "native" if parsed["gender"] != "unknown" else "unavailable"
            row["race"] = parsed["race"]
            row["race_source"] = "native" if parsed["race"] != "unknown" else "unavailable"
            row["age"] = parsed["age"]
            row["age_source"] = "native" if parsed["age"] is not None else "unavailable"
            row["age_band"] = categories.bucket_numeric_age(parsed["age"], utkface_age_bands)
            if parsed["is_disallowed_variant"]:
                raise RuntimeError(
                    f"Aligned or cropped UTKFace file entered the canonical in the wild roots: {original_path}"
                )

        elif dataset == "lfw":
            rel = Path(original_path)
            if len(rel.parts) < 2:
                row["parse_failed"] = True
                row["parse_error"] = "LFW path does not contain an identity folder"
            else:
                row["subject_id"] = rel.parent.name
                match = re.match(r"^.+_(\d+)$", rel.stem)
                if match is None:
                    row["parse_failed"] = True
                    row["parse_error"] = "LFW filename does not contain an image number"
        else:
            raise RuntimeError(f"Unsupported dataset: {dataset}")

        output_rows.append(row)

    result = pd.DataFrame(output_rows)
    if result["original_path"].duplicated().any():
        raise RuntimeError("Label table contains duplicate original_path values.")
    return result


def lfw_image_number(original_path: str) -> int | None:
    match = re.match(r"^.+_(\d+)$", Path(original_path).stem)
    return int(match.group(1)) if match else None
