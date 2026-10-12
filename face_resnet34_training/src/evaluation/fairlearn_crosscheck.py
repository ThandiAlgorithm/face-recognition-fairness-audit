"""Cross-checks this project's own equalized_odds_diff / demographic_parity_diff
(metrics.py's _bootstrap_fairness) against Fairlearn's implementation of the
same definitions, on the exact same one-vs-rest binary vectors and group
codes metrics.py builds internally (see fairness_gap_and_formal_metrics).

This does not re-run any model and does not replace our own
fairness_metrics.csv -- it only answers "does an independent, widely-used
library agree with our numbers," which is the point of step 6 in the
audit plan (reproduce selected metrics in Fairlearn, document agreement).

Point estimates only (no bootstrap CI here -- Fairlearn has no built-in
CI, and our own CI is already in fairness_metrics.csv); a mismatch beyond
floating-point tolerance means a real definitional bug somewhere and is
worth investigating immediately, not averaging away.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import SENSITIVE_COLS  # noqa: E402

TOLERANCE = 1e-9


def _has_degenerate_group(y_true_bin: np.ndarray, group_series: np.ndarray) -> bool:
    """True if any sensitive-attribute group has zero ground-truth
    positives or zero ground-truth negatives for this one-vs-rest class --
    the condition under which our own TPR/FPR convention (NaN, dropped
    from the gap) and Fairlearn's (sklearn's zero_division=0, kept in the
    gap) can legitimately diverge. Checked directly from group composition
    rather than from whether our own stored result happens to be NaN,
    since a degenerate group can get silently dropped from our internal
    max-min without the final value itself being NaN (e.g. when 3+ groups
    exist and only one is degenerate)."""
    for g in np.unique(group_series):
        vals = y_true_bin[group_series == g]
        if (vals == 1).sum() == 0 or (vals == 0).sum() == 0:
            return True
    return False


def crosscheck_one(df: pd.DataFrame, task: str, sens_col: str) -> list[dict]:
    valid = df[df[f"{task}_true"] != -1]
    if sens_col not in valid.columns or valid[sens_col].nunique() < 2:
        return []
    # Rows with a true-null sensitive value factorize to code -1 and are
    # correctly excluded by metrics.py's own _group_rates (it only loops
    # range(n_groups), 0..n_groups-1). Fairlearn has no such convention --
    # it would treat -1 as a real, extra group. Dropping null rows first
    # keeps both sides computing over the exact same support. Harmless
    # where sens_col uses a string "unknown" placeholder instead of a true
    # null (LFW/UTKFace/FairFace): .dropna() there is a no-op.
    valid = valid[valid[sens_col].notna()]
    if valid[sens_col].nunique() < 2:
        return []
    group_series, _ = pd.factorize(valid[sens_col].astype(str), sort=True)

    classes = sorted(valid[f"{task}_true"].unique().tolist())
    is_binary = len(classes) <= 2
    target_classes = [1] if is_binary else classes

    y_true_full = valid[f"{task}_true"].to_numpy()
    y_pred_full = valid[f"{task}_pred"].to_numpy()

    rows = []
    for cls in target_classes:
        y_true_bin = (y_true_full == cls).astype(int)
        y_pred_bin = (y_pred_full == cls).astype(int)
        if len(np.unique(group_series)) < 2:
            continue
        fl_eo = equalized_odds_difference(y_true_bin, y_pred_bin, sensitive_features=group_series)
        fl_dp = demographic_parity_difference(y_true_bin, y_pred_bin, sensitive_features=group_series)
        rows.append({
            "task": task, "class": cls if not is_binary else "positive", "sensitive_attribute": sens_col,
            "fairlearn_equalized_odds_diff": fl_eo, "fairlearn_demographic_parity_diff": fl_dp,
            "has_degenerate_group": _has_degenerate_group(y_true_bin, group_series),
        })
    return rows


def crosscheck_lfw_attribute(df: pd.DataFrame, attr: str, idx: int, sens_col: str) -> list[dict]:
    if sens_col not in df.columns or df[sens_col].nunique() < 2:
        return []
    group_series, _ = pd.factorize(df[sens_col].astype(str), sort=True)
    if len(np.unique(group_series)) < 2:
        return []
    y_true_bin = np.array(df["attributes_true"].tolist())[:, idx]
    y_pred_bin = np.array(df["attributes_pred"].tolist())[:, idx]
    fl_eo = equalized_odds_difference(y_true_bin, y_pred_bin, sensitive_features=group_series)
    fl_dp = demographic_parity_difference(y_true_bin, y_pred_bin, sensitive_features=group_series)
    return [{"attribute": attr, "sensitive_attribute": sens_col,
             "fairlearn_equalized_odds_diff": fl_eo, "fairlearn_demographic_parity_diff": fl_dp,
             "has_degenerate_group": _has_degenerate_group(y_true_bin, group_series)}]


def run(dataset: str, method: str) -> pd.DataFrame:
    eval_dir = EXPERIMENTS_DIR / dataset / "test_eval" if method == "M1_baseline" else None
    if eval_dir is None:
        candidates = list(EXPERIMENTS_DIR.glob(f"{dataset}/{method}/seed_*/test_eval"))
        if not candidates:
            raise FileNotFoundError(f"no test_eval dir found for {dataset}/{method}")
        eval_dir = candidates[0]

    pred_df = pd.read_parquet(eval_dir / "test_predictions.parquet")
    ours = pd.read_csv(eval_dir / "fairness_metrics.csv")

    if dataset == "lfw":
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "datasets"))
        from label_encoders import load_labels
        attrs = load_labels("lfw")["attributes"]
        rows = []
        for i, attr in enumerate(attrs):
            for sens_col in SENSITIVE_COLS:
                rows.extend(crosscheck_lfw_attribute(pred_df, attr, i, sens_col))
        fl_df = pd.DataFrame(rows)
        ours = ours.copy()
        merge_keys = ["attribute", "sensitive_attribute"]
        merged = fl_df.merge(ours[merge_keys + ["equalized_odds_diff", "demographic_parity_diff"]],
                              on=merge_keys, how="left")
        merged.insert(0, "task", "x")
        merged.insert(1, "class", "positive")
        return _finish(merged, eval_dir, dataset, method)

    tasks = [c[:-5] for c in pred_df.columns if c.endswith("_true") and c != "attributes_true"]
    rows = []
    for task in tasks:
        for sens_col in SENSITIVE_COLS:
            rows.extend(crosscheck_one(pred_df, task, sens_col))
    fl_df = pd.DataFrame(rows)
    fl_df["class"] = fl_df["class"].astype(str)
    ours = ours.copy()
    ours["class"] = ours["class"].astype(str)

    merged = fl_df.merge(ours[["task", "class", "sensitive_attribute", "equalized_odds_diff", "demographic_parity_diff"]],
                          on=["task", "class", "sensitive_attribute"], how="left")
    return _finish(merged, eval_dir, dataset, method)


def _finish(merged: pd.DataFrame, eval_dir: Path, dataset: str, method: str) -> pd.DataFrame:
    merged["eo_diff_abs"] = (merged["fairlearn_equalized_odds_diff"] - merged["equalized_odds_diff"]).abs()
    merged["dp_diff_abs"] = (merged["fairlearn_demographic_parity_diff"] - merged["demographic_parity_diff"]).abs()
    merged["agrees"] = (merged["eo_diff_abs"] < TOLERANCE) & (merged["dp_diff_abs"] < TOLERANCE)
    # Our TPR/FPR convention leaves a group's rate as NaN (and drops it from
    # the gap/max-min) when that group has zero ground-truth positives for
    # this one-vs-rest class -- common for self-referential combos like the
    # race task broken down by race_label itself. sklearn/Fairlearn instead
    # default to rate=0 for that group (zero_division=0), which Fairlearn's
    # max-min then treats as a real, meaningful gap. Neither is "wrong";
    # they're different conventions for an undefined rate, and ours is
    # the more conservative one (excludes rather than substitutes).
    ours_undefined = merged["has_degenerate_group"].fillna(False)
    merged["note"] = ""
    merged.loc[ours_undefined, "note"] = (
        "a sensitive-attribute group has zero ground-truth positives or negatives for this class -- "
        "ours drops that group's rate (NaN) from the gap, fairlearn/sklearn zero-fills it instead -- "
        "convention difference, not a bug"
    )

    out_path = eval_dir / "fairlearn_crosscheck.csv"
    merged.to_csv(out_path, index=False)
    n_total, n_agree = len(merged), int(merged["agrees"].sum())
    n_convention = int(ours_undefined.sum())
    n_real_mismatch = int((~merged["agrees"] & ~ours_undefined).sum())
    print(f"[{dataset}/{method}] {n_agree}/{n_total} rows agree with Fairlearn within {TOLERANCE}; "
          f"{n_convention} are the known NaN-vs-zero-fill convention difference; "
          f"{n_real_mismatch} are unexplained mismatches -> {out_path}")
    if n_real_mismatch > 0:
        print(merged[~merged["agrees"] & ~ours_undefined].to_string())
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--method", default="M1_baseline")
    args = parser.parse_args()
    run(args.dataset, args.method)


if __name__ == "__main__":
    main()
