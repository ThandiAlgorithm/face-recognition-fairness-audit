"""McNemar paired comparison: M1 baseline vs each mitigation variant, on
the SAME test examples, for classification outcomes -- the piece of the
reliability work metrics.py's bootstrap CIs don't cover. A gap metric
telling you fairness narrowed is a different claim from "the model's
per-example correctness actually changed between these two checkpoints";
McNemar's test is what turns the second claim into evidence rather than
eyeballing two bars.

Never re-runs a model -- reads test_predictions.parquet the same way
metrics.py does, and never touches test labels/thresholds (this is a
post-hoc comparison, not a tuning step).

Pairing is row-aligned on image_path within the frozen test split, since
M1/M2/M3/M4 for a given dataset all evaluate the identical split -- if the
two prediction files don't cover the exact same image set, that's a
real problem worth surfacing loudly, not silently inner-joining past it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from label_encoders import load_labels  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
CLASSIFICATION_TASKS = ["race", "gender", "age_group"]
# Exact (binomial) McNemar below this discordant-pair count, else the
# continuity-corrected chi-square -- the usual small-sample threshold
# (e.g. Fagerland et al. 2013) since the chi-square approximation is
# unreliable when there are few discordant pairs.
EXACT_THRESHOLD = 25


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    n10 = int(((correct_a) & (~correct_b)).sum())  # a right, b wrong
    n01 = int(((~correct_a) & (correct_b)).sum())  # a wrong, b right
    n_discordant = n10 + n01
    if n_discordant == 0:
        return {"n10": n10, "n01": n01, "n_discordant": 0, "test": "none", "statistic": 0.0, "p_value": 1.0}
    if n_discordant < EXACT_THRESHOLD:
        result = binomtest(min(n10, n01), n_discordant, 0.5, alternative="two-sided")
        return {"n10": n10, "n01": n01, "n_discordant": n_discordant, "test": "exact_binomial",
                "statistic": float(min(n10, n01)), "p_value": float(result.pvalue)}
    stat = (abs(n10 - n01) - 1) ** 2 / n_discordant
    p = float(chi2.sf(stat, df=1))
    return {"n10": n10, "n01": n01, "n_discordant": n_discordant, "test": "chi2_continuity_corrected",
            "statistic": float(stat), "p_value": p}


def _discover_variants(dataset: str) -> dict[str, Path]:
    """Baseline is experiments/<dataset>/test_eval; every other
    <dataset>/<method>/seed_*/test_eval with a predictions file found on
    disk is a comparison candidate, named after its method directory."""
    root = EXPERIMENTS_DIR / dataset
    variants = {}
    baseline = root / "test_eval" / "test_predictions.parquet"
    if baseline.exists():
        variants["M1_baseline"] = baseline
    for p in sorted(root.glob("*/seed_*/test_eval/test_predictions.parquet")):
        method_name = p.parents[2].name
        variants[method_name] = p
    return variants


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df["image_path"].duplicated().any():
        raise ValueError(f"{path}: duplicate image_path rows -- pairing would be ambiguous")
    return df.set_index("image_path")


def compare_pair(dataset: str, task: str, baseline_df: pd.DataFrame, compare_df: pd.DataFrame,
                  baseline_label: str, compare_label: str) -> dict | None:
    true_col, pred_col = f"{task}_true", f"{task}_pred"
    if true_col not in baseline_df.columns or true_col not in compare_df.columns:
        return None
    common = baseline_df.index.intersection(compare_df.index)
    dropped = len(baseline_df.index.union(compare_df.index)) - len(common)
    b = baseline_df.loc[common]
    c = compare_df.loc[common]
    valid = (b[true_col] != -1) & (c[true_col] != -1) & (b[true_col] == c[true_col])
    b, c = b[valid], c[valid]
    if b.empty:
        return None
    correct_a = (b[pred_col] == b[true_col]).to_numpy()
    correct_b = (c[pred_col] == c[true_col]).to_numpy()
    result = mcnemar_test(correct_a, correct_b)
    result.update({
        "dataset": dataset, "task": task,
        "baseline_method": baseline_label, "compare_method": compare_label,
        "n_paired": len(b), "n_dropped_unmatched": dropped,
        "baseline_accuracy": float(correct_a.mean()), "compare_accuracy": float(correct_b.mean()),
        "significant_0.05": result["p_value"] < 0.05,
    })
    return result


def compare_lfw_attribute(dataset: str, attr: str, idx: int, baseline_df: pd.DataFrame, compare_df: pd.DataFrame,
                           baseline_label: str, compare_label: str) -> dict | None:
    common = baseline_df.index.intersection(compare_df.index)
    dropped = len(baseline_df.index.union(compare_df.index)) - len(common)
    b, c = baseline_df.loc[common], compare_df.loc[common]
    b_true = np.array(b["attributes_true"].tolist())[:, idx]
    c_true = np.array(c["attributes_true"].tolist())[:, idx]
    b_pred = np.array(b["attributes_pred"].tolist())[:, idx]
    c_pred = np.array(c["attributes_pred"].tolist())[:, idx]
    valid = b_true == c_true
    if valid.sum() == 0:
        return None
    correct_a = (b_pred[valid] == b_true[valid])
    correct_b = (c_pred[valid] == c_true[valid])
    result = mcnemar_test(correct_a, correct_b)
    result.update({
        "dataset": dataset, "task": attr,
        "baseline_method": baseline_label, "compare_method": compare_label,
        "n_paired": int(valid.sum()), "n_dropped_unmatched": dropped,
        "baseline_accuracy": float(correct_a.mean()), "compare_accuracy": float(correct_b.mean()),
        "significant_0.05": result["p_value"] < 0.05,
    })
    return result


def run(dataset: str) -> pd.DataFrame:
    variants = _discover_variants(dataset)
    if "M1_baseline" not in variants:
        raise FileNotFoundError(f"No M1 baseline test_predictions.parquet found for {dataset}")
    if len(variants) < 2:
        print(f"[{dataset}] only the baseline is present -- nothing to compare yet")
        return pd.DataFrame()

    loaded = {label: _load(path) for label, path in variants.items()}
    baseline_label = "M1_baseline"
    baseline_df = loaded[baseline_label]
    rows = []

    if dataset == "lfw":
        labels = load_labels(dataset)
        for compare_label, compare_df in loaded.items():
            if compare_label == baseline_label:
                continue
            for i, attr in enumerate(labels["attributes"]):
                r = compare_lfw_attribute(dataset, attr, i, baseline_df, compare_df, baseline_label, compare_label)
                if r is not None:
                    rows.append(r)
    else:
        for compare_label, compare_df in loaded.items():
            if compare_label == baseline_label:
                continue
            for task in CLASSIFICATION_TASKS:
                r = compare_pair(dataset, task, baseline_df, compare_df, baseline_label, compare_label)
                if r is not None:
                    rows.append(r)

    result_df = pd.DataFrame(rows)
    out_path = EXPERIMENTS_DIR / dataset / "paired_comparison.csv"
    result_df.to_csv(out_path, index=False)
    print(f"[{dataset}] wrote {len(result_df)} paired comparisons -> {out_path}")
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["lfw", "utkface", "fairface"])
    args = parser.parse_args()
    run(args.dataset)


if __name__ == "__main__":
    main()
