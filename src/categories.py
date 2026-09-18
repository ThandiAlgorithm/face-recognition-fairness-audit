
from __future__ import annotations

from typing import Optional

import pandas as pd

FAIRFACE_AGE_TO_FINE_BAND = {
    "0-2": "0-2",
    "3-9": "3-9",
    "10-19": "10-19",
    "20-29": "20-29",
    "30-39": "30-39",
    "40-49": "40-49",
    "50-59": "50-59",
    "60-69": "60-69",
    "more than 70": "70+",
}

FAIRFACE_GENDER_TO_CODE = {"Female": 1, "Male": 0}


def verify_fine_in_coarse_nesting(age_bands_fine: list[dict], age_bands_coarse: list[dict]) -> pd.DataFrame:
    rows = []
    for fine in age_bands_fine:
        fine_min, fine_max = fine["min"], fine["max"]
        matches = []
        for coarse in age_bands_coarse:
            c_min, c_max = coarse["min"], coarse["max"]
            lower_ok = fine_min >= c_min
            if c_max is None:
                upper_ok = True
            else:
                upper_ok = fine_max is not None and fine_max <= c_max
            if lower_ok and upper_ok:
                matches.append(coarse["name"])
        rows.append(
            {
                "fine_band": fine["name"],
                "fine_min": fine_min,
                "fine_max": fine_max,
                "matching_coarse_bands": matches,
                "nests_cleanly": len(matches) == 1,
            }
        )
    return pd.DataFrame(rows)


def fine_to_coarse_map(nesting_df: pd.DataFrame) -> dict:
    if not nesting_df["nests_cleanly"].all():
        raise RuntimeError("Fine age bands do not nest cleanly in coarse age bands -- fix GOV-002 before proceeding.")
    return {row["fine_band"]: row["matching_coarse_bands"][0] for _, row in nesting_df.iterrows()}


def bucket_numeric_age(age: Optional[float], age_bands_fine: list[dict]) -> Optional[str]:
    if age is None or pd.isna(age):
        return None
    for band in age_bands_fine:
        lo, hi = band["min"], band["max"]
        if age >= lo and (hi is None or age <= hi):
            return band["name"]
    return None
