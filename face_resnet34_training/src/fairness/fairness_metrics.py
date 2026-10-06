"""Step 13: subgroup/fairness metrics from locked test predictions.

Reads test_predictions.csv (written by evaluate.py) and computes
per-subgroup accuracy plus fairlearn fairness-gap metrics, broken down by
race, gender, and ITA group -- the three fairness variables this project
cares about. Never re-runs the model; this is pure analysis of
already-locked predictions, so it can be re-run freely as the fairness
breakdown is refined without touching the model or the test set again.

demographic_parity_difference / equalized_odds_difference are fairlearn
metrics defined for binary classification; they're computed directly for
gender (binary in this project's label encoding) and attempted for the
multi-class tasks (race, age_group) since fairlearn can still produce a
value for them in recent versions, but any failure is recorded as an error
string rather than crashing the whole report -- per-subgroup accuracy
(always valid for any number of classes) is the metric guaranteed to be
present for every task.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from label_encoders import load_labels

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
FAIRNESS_GROUPING_COLUMNS = ["race_label", "gender_label", "ita_group"]
CLASSIFICATION_TASKS = ["race", "gender", "age_group"]


def compute_task_fairness(predictions: pd.DataFrame, task: str) -> dict:
    valid = predictions[predictions[f"{task}_true"] != -1].copy()
    if valid.empty:
        return {"n_valid": 0}

    y_true = valid[f"{task}_true"]
    y_pred = valid[f"{task}_pred"]
    result = {"n_valid": len(valid), "overall_accuracy": float(accuracy_score(y_true, y_pred))}

    for group_col in FAIRNESS_GROUPING_COLUMNS:
        if group_col not in valid.columns or valid[group_col].nunique() < 2:
            continue
        groups = valid[group_col]

        frame = MetricFrame(metrics={"accuracy": accuracy_score}, y_true=y_true, y_pred=y_pred, sensitive_features=groups)
        result[f"accuracy_by_{group_col}"] = {str(k): float(v) for k, v in frame.by_group["accuracy"].to_dict().items()}
        result[f"accuracy_gap_{group_col}"] = float(frame.difference()["accuracy"])

        try:
            result[f"demographic_parity_diff_{group_col}"] = float(
                demographic_parity_difference(y_true, y_pred, sensitive_features=groups)
            )
        except Exception as exc:
            result[f"demographic_parity_diff_{group_col}_error"] = str(exc)
        try:
            result[f"equalized_odds_diff_{group_col}"] = float(
                equalized_odds_difference(y_true, y_pred, sensitive_features=groups)
            )
        except Exception as exc:
            result[f"equalized_odds_diff_{group_col}_error"] = str(exc)

    return result


def compute_lfw_attribute_fairness(predictions: pd.DataFrame, attributes: list[str]) -> dict:
    preds = np.array(predictions["attributes_pred"].tolist())
    trues = np.array(predictions["attributes_true"].tolist())

    result = {}
    for group_col in FAIRNESS_GROUPING_COLUMNS:
        if group_col not in predictions.columns or predictions[group_col].nunique() < 2:
            continue
        groups = predictions[group_col]
        per_attr_gap = {}
        for i, attr in enumerate(attributes):
            frame = MetricFrame(
                metrics={"accuracy": accuracy_score}, y_true=trues[:, i], y_pred=preds[:, i], sensitive_features=groups
            )
            per_attr_gap[attr] = float(frame.difference()["accuracy"])
        result[f"accuracy_gap_by_{group_col}_per_attribute"] = per_attr_gap
        result[f"mean_accuracy_gap_{group_col}"] = float(sum(per_attr_gap.values()) / len(per_attr_gap))
    return result


def run_fairness_analysis(dataset: str) -> dict:
    eval_dir = EXPERIMENTS_DIR / dataset / "test_eval"
    predictions = pd.read_csv(eval_dir / "test_predictions.csv")

    if dataset == "lfw":
        labels = load_labels(dataset)
        predictions["attributes_pred"] = predictions["attributes_pred"].apply(eval)
        predictions["attributes_true"] = predictions["attributes_true"].apply(eval)
        result = compute_lfw_attribute_fairness(predictions, labels["attributes"])
    else:
        result = {task: compute_task_fairness(predictions, task) for task in CLASSIFICATION_TASKS}

    (eval_dir / "fairness_report.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface", "lfw"])
    args = parser.parse_args()
    result = run_fairness_analysis(args.dataset)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
