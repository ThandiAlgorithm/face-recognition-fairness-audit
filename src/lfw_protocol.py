from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from src import paths


def _pair_id(view: str, fold: int, identity_1: str, number_1: int, identity_2: str, number_2: int) -> str:
    payload = f"{view}|{fold}|{identity_1}|{number_1}|{identity_2}|{number_2}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"LFWPAIR-{digest}"


def _image_number(original_path: str) -> int | None:
    match = re.match(r"^.+_(\d+)$", Path(original_path).stem)
    return int(match.group(1)) if match else None


def normalize_view2_pairs(pairs_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(pairs_csv)
    columns = list(frame.columns)
    if len(columns) < 4:
        raise RuntimeError("pairs.csv must contain four columns after parsing.")

    rows = []
    for index, row in frame.iterrows():
        first_name = row[columns[0]]
        first_number = row[columns[1]]
        third = row[columns[2]]
        fourth = row[columns[3]]
        fold = int(index // 600) + 1
        if pd.isna(fourth):
            identity_1 = str(first_name)
            identity_2 = str(first_name)
            number_1 = int(first_number)
            number_2 = int(third)
            is_genuine = True
        else:
            identity_1 = str(first_name)
            identity_2 = str(third)
            number_1 = int(first_number)
            number_2 = int(fourth)
            is_genuine = False
        rows.append(
            {
                "fold": fold,
                "view": "view2",
                "is_genuine": is_genuine,
                "identity_1": identity_1,
                "identity_2": identity_2,
                "image_number_1": number_1,
                "image_number_2": number_2,
            }
        )
    return pd.DataFrame(rows)


def build_lfw_pairs(master: pd.DataFrame, pairs_csv: Path) -> pd.DataFrame:
    pairs = normalize_view2_pairs(pairs_csv)
    lfw = master[master["dataset"].eq("lfw")].copy()
    lfw["image_number"] = lfw["original_path"].apply(_image_number)
    lookup = lfw.set_index(["subject_id", "image_number"], drop=False)

    rows = []
    for item in pairs.itertuples(index=False):
        key_1 = (item.identity_1, item.image_number_1)
        key_2 = (item.identity_2, item.image_number_2)
        if key_1 not in lookup.index or key_2 not in lookup.index:
            raise RuntimeError(f"LFW pair does not resolve to master rows: {key_1}, {key_2}")
        left = lookup.loc[key_1]
        right = lookup.loc[key_2]
        if isinstance(left, pd.DataFrame) or isinstance(right, pd.DataFrame):
            raise RuntimeError("LFW lookup is not unique by identity and image number.")

        pair_demog_complete = all(
            [
                left["gender"] != "unknown",
                right["gender"] != "unknown",
                pd.notna(left["skin_group_ita"]),
                pd.notna(right["skin_group_ita"]),
                pd.notna(left["age_band"]),
                pd.notna(right["age_band"]),
            ]
        )
        same_gender = bool(pair_demog_complete and left["gender"] == right["gender"])
        same_skin = bool(pair_demog_complete and left["skin_group_ita"] == right["skin_group_ita"])
        same_age = bool(pair_demog_complete and left["age_band"] == right["age_band"])
        if not pair_demog_complete:
            yoked = "unknown"
        elif same_gender and same_skin and same_age:
            yoked = "fully_yoked"
        elif same_gender or same_skin or same_age:
            yoked = "partially_yoked"
        else:
            yoked = "cross_demographic"

        brightness_difference = None
        if pd.notna(left["brightness"]) and pd.notna(right["brightness"]):
            brightness_difference = abs(float(left["brightness"]) - float(right["brightness"]))

        rows.append(
            {
                "pair_id": _pair_id(
                    "view2",
                    item.fold,
                    item.identity_1,
                    item.image_number_1,
                    item.identity_2,
                    item.image_number_2,
                ),
                "fold": int(item.fold),
                "view": "view2",
                "is_genuine": bool(item.is_genuine),
                "identity_1": item.identity_1,
                "identity_2": item.identity_2,
                "image_id_1": left["image_id"],
                "image_id_2": right["image_id"],
                "gender_1": left["gender"],
                "gender_2": right["gender"],
                "same_gender": same_gender,
                "skin_group_1": left["skin_group_ita"] if pd.notna(left["skin_group_ita"]) else "unknown",
                "skin_group_2": right["skin_group_ita"] if pd.notna(right["skin_group_ita"]) else "unknown",
                "same_skin_group": same_skin,
                "age_band_1": left["age_band"] if pd.notna(left["age_band"]) else "unknown",
                "age_band_2": right["age_band"] if pd.notna(right["age_band"]) else "unknown",
                "same_age_band": same_age,
                "pair_demog_complete": bool(pair_demog_complete),
                "yoked": yoked,
                "brightness_difference": brightness_difference,
                "pose_difference": None,
                "model_name": None,
                "model_score": None,
                "threshold": None,
                "decision": None,
            }
        )

    output = pd.DataFrame(rows)
    if len(output) != 6000:
        raise RuntimeError(f"Expected 6000 View 2 pairs, found {len(output)}")
    fold_counts = output.groupby(["fold", "is_genuine"]).size().unstack(fill_value=0)
    if len(fold_counts) != 10:
        raise RuntimeError("Expected ten LFW View 2 folds.")
    return output


def validate_people_counts(people_csv: Path, image_root: Path) -> pd.DataFrame:
    people = pd.read_csv(people_csv)
    columns = list(people.columns)
    if len(columns) < 2:
        raise RuntimeError("people.csv must contain identity and image count columns.")
    rows = []
    for _, row in people.iterrows():
        identity = str(row[columns[0]])
        declared = int(row[columns[1]])
        folder = image_root / identity
        actual = len(list(folder.glob("*"))) if folder.exists() else 0
        rows.append(
            {
                "identity": identity,
                "declared_images": declared,
                "actual_files": actual,
                "folder_exists": folder.exists(),
                "count_matches": declared == actual,
            }
        )
    return pd.DataFrame(rows)
