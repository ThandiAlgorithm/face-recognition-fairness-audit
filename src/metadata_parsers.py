
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src import paths
from src.data_inventory import list_image_files

# FairFace


def load_fairface_labels(csv_path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["split"] = split
    df["metadata_source"] = str(csv_path)
    return df


# UTKFace
# Filename fields: age, gender, race, timestamp.
# Gender: 0 male, 1 female.
# Race: 0 White, 1 Black, 2 Asian, 3 Indian, 4 Others.
# Accept the optional chip suffix.
_UTKFACE_PATTERN = re.compile(
    r"^(?P<age>\d+)_(?P<gender>\d+)_(?P<race>\d+)_(?P<timestamp>\d+)\.jpe?g(?:\.chip\.jpe?g)?$",
    re.IGNORECASE,
)

_UTKFACE_CHIP_PATTERN = re.compile(r"\.jpe?g\.chip\.jpe?g$", re.IGNORECASE)

_UTKFACE_RACE_LABELS = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}
_UTKFACE_GENDER_LABELS = {0: "male", 1: "female"}


def parse_utkface_filename(filename: str) -> dict:
    match = _UTKFACE_PATTERN.match(filename)
    is_chip = bool(_UTKFACE_CHIP_PATTERN.search(filename))
    if match is None:
        return {
            "original_filename": filename,
            "parsed_age": None,
            "parsed_gender_code": None,
            "parsed_gender": None,
            "parsed_race_code": None,
            "parsed_race": None,
            "is_chip_variant": is_chip,
            "parse_success": False,
            "parse_error": "filename does not match [age]_[gender]_[race]_[timestamp].jpg(.chip.jpg)?",
        }

    age = int(match.group("age"))
    gender_code = int(match.group("gender"))
    race_code = int(match.group("race"))

    errors = []
    if not (0 <= age <= 116):
        errors.append(f"age {age} outside plausible range 0-116")
    gender_label = _UTKFACE_GENDER_LABELS.get(gender_code)
    if gender_label is None:
        errors.append(f"unknown gender code {gender_code}")
    race_label = _UTKFACE_RACE_LABELS.get(race_code)
    if race_label is None:
        errors.append(f"unknown race code {race_code}")

    return {
        "original_filename": filename,
        "parsed_age": age,
        "parsed_gender_code": gender_code,
        "parsed_gender": gender_label,
        "parsed_race_code": race_code,
        "parsed_race": race_label,
        "is_chip_variant": is_chip,
        "parse_success": len(errors) == 0,
        "parse_error": "; ".join(errors) if errors else None,
    }


def load_utkface_directory(root: Path, location_label: str) -> pd.DataFrame:
    records = []
    for path in list_image_files(root):
        record = parse_utkface_filename(path.name)
        record["relative_path"] = str(path.relative_to(root))
        record["location"] = location_label
        records.append(record)

    df = pd.DataFrame(records)
    n_files = len(list_image_files(root))
    if len(df) != n_files:
        raise RuntimeError(
            f"{location_label}: {len(df)} records from {n_files} files. "
            "Every file must produce exactly one record."
        )
    return df


# LFW


def load_lfw_people(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["metadata_source"] = str(csv_path)
    return df


def load_lfw_pairs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["metadata_source"] = str(csv_path)
    return df


def build_lfw_identity_table(image_root: Path) -> pd.DataFrame:
    if not image_root.exists():
        return pd.DataFrame()

    records = []
    for identity_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        for image_path in sorted(list_image_files(identity_dir)):
            m = re.match(r"^.+_(?P<image_num>\d+)$", image_path.stem)
            records.append(
                {
                    "identity": identity_dir.name,
                    "image_num": int(m.group("image_num")) if m else None,
                    "original_path": paths.to_rel(image_path),  # relative to DATA_DIR
                    "subject_id": identity_dir.name,
                }
            )
    df = pd.DataFrame(records)
    if not df.empty:
        df["images_per_identity"] = df.groupby("identity")["identity"].transform("count")
    return df
