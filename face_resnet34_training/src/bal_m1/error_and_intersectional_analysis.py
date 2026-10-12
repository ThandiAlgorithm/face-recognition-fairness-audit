"""Two analyses for BAL-M1, reusing the same machinery already built and
validated for LFW/UTKFace/FairFace, rather than inventing new methodology:

1. Error-condition linkage (error_condition_linkage.py's univariate_tests /
   demographic_control_test): is BAL-M1's remaining skin-tone/race gap a
   real model property, or partly explained by image-acquisition quality
   now that HarmoFace merges three source datasets with different average
   capture conditions?
2. Intersectional breakdown: per-task F1/accuracy for every pairwise
   combination of the four demographic attributes, flagging small cells
   rather than hiding them -- does the remaining gap concentrate in a
   specific intersection (e.g. dark skin + older age) or spread evenly?

Reads experiments/harmoface/BAL-M1/test_predictions_joined.parquet (the
evaluate.py predictions re-joined to the manifest's own demographic label
columns and image_id, built once locally -- see the join step in the
conversation, not re-derived here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from error_condition_linkage import load_quality_table, univariate_tests, demographic_control_test  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[2] / "experiments" / "harmoface" / "BAL-M1"
TASKS = ["race", "gender", "age", "skin_tone"]
DEMO_COLS = {
    "race": "candidate_demographic_label", "gender": "gender_clean",
    "age": "age_group_harmonized", "skin_tone": "fitzpatrick_approx",
}
IGNORE_INDEX = -1


def run_error_condition_linkage(df: pd.DataFrame) -> None:
    quality, _ = load_quality_table()
    # HarmoFace's own image_id is already VIS/MEA's image_id scheme
    # (e.g. "utkface-<hash>") -- verified by construction, so no path
    # normalization fallback is needed here, unlike FairFace's raw-path case.
    merged = df.merge(quality, on="image_id", how="left")
    n_missing = merged["brightness"].isna().sum() if "brightness" in merged.columns else len(merged)
    print(f"[bal_m1] {len(merged)} rows after quality join; missing quality data for {n_missing} rows")

    uni_rows, ctrl_rows = [], []
    for task in TASKS:
        true_col, pred_col = f"{task}_true", f"{task}_pred"
        valid = merged[merged[true_col] != IGNORE_INDEX]
        is_error = valid[pred_col] != valid[true_col]
        uni_rows.extend(univariate_tests(valid, is_error, task, "overall"))
        for other_task in TASKS:
            if other_task == task:
                continue
            demo_col = DEMO_COLS[other_task]
            if demo_col not in valid.columns:
                continue
            ctrl_rows.extend(demographic_control_test(valid, is_error, demo_col, task))

    uni_df, ctrl_df = pd.DataFrame(uni_rows), pd.DataFrame(ctrl_rows)
    uni_df.to_csv(EXP_DIR / "error_quality_univariate.csv", index=False)
    ctrl_df.to_csv(EXP_DIR / "error_quality_controlled.csv", index=False)
    print(f"wrote {len(uni_df)} univariate rows, {len(ctrl_df)} demographic-control rows -> {EXP_DIR}")
    if not ctrl_df.empty:
        survived = ctrl_df[ctrl_df["survives_control"] == True]
        explained = ctrl_df[ctrl_df["explained_by_quality"] == True]
        print(f"  {len(survived)} demographic associations SURVIVE quality control (real demographic effect)")
        print(f"  {len(explained)} demographic associations are EXPLAINED AWAY by quality covariates")
        if len(explained):
            print(explained[["task", "demographic", "dummy", "bare_p", "controlled_p"]].to_string())


def run_intersectional(df: pd.DataFrame) -> None:
    pairs = [("candidate_demographic_label", "age_group_harmonized"), ("candidate_demographic_label", "fitzpatrick_approx"),
             ("candidate_demographic_label", "gender_clean"), ("fitzpatrick_approx", "age_group_harmonized"),
             ("fitzpatrick_approx", "gender_clean"), ("gender_clean", "age_group_harmonized")]
    rows = []
    for attr_a, attr_b in pairs:
        for task in TASKS:
            true_col, pred_col = f"{task}_true", f"{task}_pred"
            valid = df[(df[true_col] != IGNORE_INDEX) & df[attr_a].notna() & df[attr_b].notna()
                       & (df[attr_a] != "unknown") & (df[attr_b] != "unknown")]
            for (a_val, b_val), group in valid.groupby([attr_a, attr_b]):
                n = len(group)
                f1 = f1_score(group[true_col], group[pred_col], average="macro") if n >= 5 else None
                acc = (group[pred_col] == group[true_col]).mean()
                rows.append({
                    "task": task, "attr_a": attr_a, "value_a": a_val, "attr_b": attr_b, "value_b": b_val,
                    "n": n, "macro_f1": f1, "accuracy": float(acc), "small_group": n < 20,
                })
    out = pd.DataFrame(rows)
    out.to_csv(EXP_DIR / "intersectional_breakdown.csv", index=False)
    print(f"wrote {len(out)} intersectional rows -> {EXP_DIR / 'intersectional_breakdown.csv'}")
    reliable = out[(~out["small_group"]) & out["macro_f1"].notna()]
    print("\nworst 10 reliable (n>=20) intersectional cells by macro_f1:")
    print(reliable.sort_values("macro_f1").head(10)[["task", "attr_a", "value_a", "attr_b", "value_b", "n", "macro_f1"]].to_string())


def main() -> None:
    df = pd.read_parquet(EXP_DIR / "test_predictions_joined.parquet")
    run_error_condition_linkage(df)
    print()
    run_intersectional(df)


if __name__ == "__main__":
    main()
