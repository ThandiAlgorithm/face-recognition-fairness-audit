"""Post-hoc per-group threshold calibration for a binary classification
head, operating purely on an already-saved test_predictions.parquet -- no
retraining. Built around a specific, already-observed pattern: LFW M1's
"Male" head uses the default 0.5 cutoff, which gives male recall 0.983 but
female recall only 0.798 -- a 20-point gap that the global threshold itself
may be causing, not the model's underlying ranking ability (which still has
a healthy AUROC of 0.97 for Male). Also works for FairFace/UTKFace's gender
task, which is the same kind of binary head, saved in a different column
shape (gender_true/gender_probs per row, rather than LFW's
attributes_true/attributes_probs list indexed by attribute name).

Two things computed, from the saved probability scores only:
  1. A single GLOBAL balanced threshold: the cutoff where recall for the
     positive class (Male=1) and recall for the negative class (Male=0,
     i.e. Female) are as close to equal as possible -- the same "equal
     error rate" idea used for M0's face-verification threshold, applied
     here to a classification head instead of a verification score.
  2. The same balanced-threshold search repeated WITHIN each race group
     (Hardt et al. 2016's equalized-odds post-processing), to check
     whether one global threshold is enough or whether different race
     groups need their own calibration point.

Never touches training data or the model itself -- purely a decision-rule
change on top of already-computed probabilities.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from label_encoders import load_labels

THRESHOLDS = np.linspace(0.01, 0.99, 197)


def _recall_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, positive: int) -> float:
    mask = y_true == positive
    if mask.sum() == 0:
        return float("nan")
    pred = (y_prob >= threshold).astype(int)
    return float((pred[mask] == positive).mean())


def _sweep(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in THRESHOLDS:
        r_pos = _recall_at(y_true, y_prob, t, 1)
        r_neg = _recall_at(y_true, y_prob, t, 0)
        rows.append({"threshold": t, "recall_male": r_pos, "recall_female": r_neg, "gap": abs(r_pos - r_neg)})
    return pd.DataFrame(rows)


def _best_balanced(sweep: pd.DataFrame) -> dict:
    row = sweep.loc[sweep["gap"].idxmin()]
    return {"threshold": float(row["threshold"]), "recall_male": float(row["recall_male"]),
            "recall_female": float(row["recall_female"]), "gap": float(row["gap"])}


def _extract(df: pd.DataFrame, dataset: str, attribute: str, labels: dict):
    """Returns (y_true, y_prob_of_positive_class) as numpy arrays, for
    either LFW's attribute-list shape or FairFace/UTKFace's per-task
    softmax shape. positive class is 'male' for the gender task, or 1 for
    an LFW attribute (already binary 0/1)."""
    if dataset == "lfw":
        idx = labels["attributes"].index(attribute)
        y_true = df["attributes_true"].apply(lambda v: v[idx]).to_numpy()
        y_prob = df["attributes_probs"].apply(lambda v: v[idx]).to_numpy()
        return y_true, y_prob
    # fairface/utkface: attribute is a task name (only 'gender' is binary)
    task = attribute
    male_idx = labels[task].index("male")
    y_true = df[f"{task}_true"].to_numpy()
    y_prob = df[f"{task}_probs"].apply(lambda v: v[male_idx]).to_numpy()
    return y_true, y_prob


def calibrate(dataset: str, eval_dir: Path, attribute: str = "Male", min_group_n: int = 30) -> dict:
    df = pd.read_parquet(eval_dir / "test_predictions.parquet")
    labels = load_labels(dataset)
    y_true, y_prob = _extract(df, dataset, attribute, labels)

    default_sweep = _sweep(y_true, y_prob)
    default_at_05 = default_sweep.iloc[(default_sweep["threshold"] - 0.5).abs().idxmin()]
    global_best = _best_balanced(default_sweep)

    result = {
        "dataset": dataset,
        "attribute": attribute,
        "n_total": int(len(df)),
        "default_threshold_0.5": {
            "recall_male": float(default_at_05["recall_male"]),
            "recall_female": float(default_at_05["recall_female"]),
            "gap": float(default_at_05["gap"]),
        },
        "global_balanced_threshold": global_best,
        "per_race_balanced_threshold": {},
    }

    for race, g in df.groupby("race_label"):
        if race in ("unknown",) or len(g) < min_group_n:
            continue
        yt, yp = _extract(g, dataset, attribute, labels)
        if len(set(yt)) < 2:
            continue
        sweep = _sweep(yt, yp)
        best = _best_balanced(sweep)
        at_05_row = sweep.iloc[(sweep["threshold"] - 0.5).abs().idxmin()]
        result["per_race_balanced_threshold"][race] = {
            "n": int(len(g)),
            "at_0.5": {"recall_male": float(at_05_row["recall_male"]), "recall_female": float(at_05_row["recall_female"])},
            "balanced": best,
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["lfw", "fairface", "utkface"])
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--attribute", default=None,
                         help="LFW: an attribute name (default 'Male'). fairface/utkface: a task name "
                              "(default/only supported value is 'gender', the one binary task).")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    attribute = args.attribute or ("Male" if args.dataset == "lfw" else "gender")

    result = calibrate(args.dataset, args.eval_dir, attribute)
    out = args.out or (args.eval_dir / f"threshold_calibration_{attribute}.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
