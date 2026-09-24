from __future__ import annotations

from typing import Optional

import pandas as pd


FAIRFACE_AGE_MAP = {
    "0-2": "0-2",
    "3-9": "3-9",
    "10-19": "10-19",
    "20-29": "20-29",
    "30-39": "30-39",
    "40-49": "40-49",
    "50-59": "50-59",
    "60-69": "60-69",
    "more than 70": "70+",
    "70+": "70+",
}


def bucket_numeric_age(age: Optional[float], bands: list[dict]) -> Optional[str]:
    if age is None or pd.isna(age):
        return None
    value = float(age)
    for band in bands:
        if value >= band["min"] and (band["max"] is None or value <= band["max"]):
            return band["name"]
    return None


def normalize_gender(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    if text == "female":
        return "female"
    if text == "male":
        return "male"
    return "unknown"


def fairface_age_band(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return FAIRFACE_AGE_MAP.get(str(value).strip())


def assign_ita_group(ita: Optional[float], cfg: dict) -> Optional[str]:
    if ita is None or pd.isna(ita):
        return None
    for group in cfg["ita_groups"]:
        lower = group["min_exclusive"]
        upper = group["max_inclusive"]
        if (lower is None or ita > lower) and (upper is None or ita <= upper):
            return group["name"]
    return None


def assign_ita_ordinal(ita: Optional[float], cfg: dict) -> Optional[int]:
    group_name = assign_ita_group(ita, cfg)
    if group_name is None:
        return None
    for group in cfg["ita_groups"]:
        if group["name"] == group_name:
            return int(group["ordinal_rank"])
    return None


def assign_ita_binary(ita: Optional[float], cfg: dict) -> Optional[str]:
    if ita is None or pd.isna(ita):
        return None
    threshold = float(cfg["skin_binary"]["ita"]["threshold_degrees"])
    return "lighter" if float(ita) > threshold else "darker"
