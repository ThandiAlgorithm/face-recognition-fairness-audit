"""Brings BAL-M1 up to the same audit rigor as LFW/UTKFace/FairFace's
M1/M2/M3: bootstrap 95% CIs on every fairness gap, not just point
estimates. Reuses the exact low-level machinery (_group_rates, _gap,
_bootstrap_fairness) already tested project-wide rather than
reimplementing the bootstrap loop -- same formulas, same conventions,
directly comparable numbers.

BAL-M1's own four demographic label columns (candidate_demographic_label,
gender_clean, age_group_harmonized, fitzpatrick_approx) stand in for the
other datasets' race_label/gender_label/age_group/fitzpatrick_approx --
the dataset-specific SENSITIVE_COLS set metrics.py.py hardcodes module-
wide, so that module's own fairness_gap_and_formal_metrics is reused by
copy with a different sensitive-column list, not monkeypatched, since
patching reusable module state is harder to reason about than a
parameterised copy of ~40 lines of orchestration code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from metrics import _bootstrap_fairness  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[2] / "experiments" / "harmoface" / "BAL-M1"
TASKS = ["race", "gender", "age", "skin_tone"]
SENSITIVE_COLS = ["candidate_demographic_label", "gender_clean", "age_group_harmonized", "fitzpatrick_approx"]
IGNORE_INDEX = -1


def fairness_gap_and_formal_metrics(df: pd.DataFrame, task: str) -> pd.DataFrame:
    valid = df[df[f"{task}_true"] != IGNORE_INDEX]
    rows = []
    for sens_col in SENSITIVE_COLS:
        if sens_col not in valid.columns or valid[sens_col].nunique() < 2:
            continue
        # Matches the established convention in metrics.py's own
        # fairness_gap_and_formal_metrics exactly: "unknown" is treated as
        # just another group, never filtered out -- for direct
        # comparability with LFW/UTKFace/FairFace's numbers, not a
        # BAL-M1-specific choice.
        group_series, group_names = pd.factorize(valid[sens_col].astype(str), sort=True)
        n_groups = len(group_names)
        if n_groups < 2:
            continue

        classes = sorted(valid[f"{task}_true"].unique().tolist())
        is_binary = len(classes) <= 2
        target_classes = [1] if is_binary else classes

        y_true_full = valid[f"{task}_true"].to_numpy()
        y_pred_full = valid[f"{task}_pred"].to_numpy()

        for cls in target_classes:
            y_true_bin = (y_true_full == cls).astype(int)
            y_pred_bin = (y_pred_full == cls).astype(int)
            result = _bootstrap_fairness(y_true_bin, y_pred_bin, group_series, n_groups, identity_codes=None)
            rows.append({
                "task": task, "class": cls if not is_binary else "positive", "sensitive_attribute": sens_col,
                "n_groups": n_groups,
                "tpr_gap": result["tpr_gap"][0], "tpr_gap_ci_lo": result["tpr_gap"][1], "tpr_gap_ci_hi": result["tpr_gap"][2],
                "fpr_gap": result["fpr_gap"][0], "fpr_gap_ci_lo": result["fpr_gap"][1], "fpr_gap_ci_hi": result["fpr_gap"][2],
                "precision_gap": result["precision_gap"][0], "precision_gap_ci_lo": result["precision_gap"][1], "precision_gap_ci_hi": result["precision_gap"][2],
                "equality_of_opportunity_diff": result["tpr_gap"][0],
                "equality_of_opportunity_ci_lo": result["tpr_gap"][1], "equality_of_opportunity_ci_hi": result["tpr_gap"][2],
                "equalized_odds_diff": result["equalized_odds_diff"][0], "equalized_odds_ci_lo": result["equalized_odds_diff"][1], "equalized_odds_ci_hi": result["equalized_odds_diff"][2],
                "demographic_parity_diff": result["demographic_parity_diff"][0], "demographic_parity_diff_ci_lo": result["demographic_parity_diff"][1], "demographic_parity_diff_ci_hi": result["demographic_parity_diff"][2],
                "demographic_parity_ratio": result["demographic_parity_ratio"][0], "demographic_parity_ratio_ci_lo": result["demographic_parity_ratio"][1], "demographic_parity_ratio_ci_hi": result["demographic_parity_ratio"][2],
                "predictive_parity_diff": result["precision_gap"][0], "predictive_parity_diff_ci_lo": result["precision_gap"][1], "predictive_parity_diff_ci_hi": result["precision_gap"][2],
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(EXP_DIR / "test_predictions_joined.parquet")
    frames = [fairness_gap_and_formal_metrics(df, task) for task in TASKS]
    out = pd.concat(frames, ignore_index=True)
    out_path = EXP_DIR / "fairness_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows -> {out_path}")

    gap_by_attr = out.groupby("sensitive_attribute")["equalized_odds_diff"].mean()
    print("\nmean equalized_odds_diff by sensitive attribute (point estimate, all tasks pooled):")
    print(gap_by_attr.to_string())


if __name__ == "__main__":
    main()
