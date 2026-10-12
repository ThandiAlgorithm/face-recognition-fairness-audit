"""Cross-checks BAL-M1's equalized_odds_diff / demographic_parity_diff
against Fairlearn's own implementation, the same validation already run
for LFW/UTKFace/FairFace -- reuses crosscheck_one and
_has_degenerate_group directly from evaluation/fairlearn_crosscheck.py
rather than reimplementing them, since those are already parameterised
by task/sensitive_col and need no dataset-specific logic.

Run fairness_metrics.py first -- this reads its output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from fairlearn_crosscheck import crosscheck_one, TOLERANCE  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[2] / "experiments" / "harmoface" / "BAL-M1"
TASKS = ["race", "gender", "age", "skin_tone"]
SENSITIVE_COLS = ["candidate_demographic_label", "gender_clean", "age_group_harmonized", "fitzpatrick_approx"]


def main() -> None:
    df = pd.read_parquet(EXP_DIR / "test_predictions_joined.parquet")
    ours = pd.read_csv(EXP_DIR / "fairness_metrics.csv")
    ours["class"] = ours["class"].astype(str)

    rows = []
    for task in TASKS:
        for sens_col in SENSITIVE_COLS:
            rows.extend(crosscheck_one(df, task, sens_col))
    fl_df = pd.DataFrame(rows)
    fl_df["class"] = fl_df["class"].astype(str)

    merged = fl_df.merge(ours[["task", "class", "sensitive_attribute", "equalized_odds_diff", "demographic_parity_diff"]],
                          on=["task", "class", "sensitive_attribute"], how="left")
    merged["eo_diff_abs"] = (merged["fairlearn_equalized_odds_diff"] - merged["equalized_odds_diff"]).abs()
    merged["dp_diff_abs"] = (merged["fairlearn_demographic_parity_diff"] - merged["demographic_parity_diff"]).abs()
    merged["agrees"] = (merged["eo_diff_abs"] < TOLERANCE) & (merged["dp_diff_abs"] < TOLERANCE)
    ours_undefined = merged["has_degenerate_group"].fillna(False)
    merged["note"] = ""
    merged.loc[ours_undefined, "note"] = (
        "a sensitive-attribute group has zero ground-truth positives or negatives for this class -- "
        "ours drops that group's rate (NaN) from the gap, fairlearn/sklearn zero-fills it instead -- "
        "convention difference, not a bug"
    )

    out_path = EXP_DIR / "fairlearn_crosscheck.csv"
    merged.to_csv(out_path, index=False)
    n_total, n_agree = len(merged), int(merged["agrees"].sum())
    n_convention = int(ours_undefined.sum())
    n_real_mismatch = int((~merged["agrees"] & ~ours_undefined).sum())
    print(f"[bal_m1] {n_agree}/{n_total} rows agree with Fairlearn within {TOLERANCE}; "
          f"{n_convention} are the known NaN-vs-zero-fill convention difference; "
          f"{n_real_mismatch} are unexplained mismatches -> {out_path}")
    if n_real_mismatch > 0:
        print(merged[~merged["agrees"] & ~ours_undefined].to_string())


if __name__ == "__main__":
    main()
