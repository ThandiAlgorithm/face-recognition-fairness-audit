"""Post-hoc per-class calibration for a MULTI-class head (race, age_group),
generalising threshold_calibration.py's binary balanced-threshold idea to
K > 2 classes. A single 0..1 threshold doesn't apply to argmax-based
multi-class prediction, so instead this adds a learned per-class bias to
each class's logit before taking argmax ("logit adjustment", in the spirit
of Menon et al. 2021's long-tail correction), nudged iteratively until
every class's recall is as close as possible to every other class's.

CAVEAT, stated explicitly in every output: like threshold_calibration.py,
this is calibrated directly on the test set, for DIAGNOSTIC purposes --
to show whether a simple decision-rule change can close a gap, not to
produce a rule meant for deployment. A threshold/bias meant to actually be
used should be chosen on a held-out validation set instead (the rule
M0's own threshold.json already follows: "selected_on": "validation").
This script's bias vector is for illustration/diagnosis of the mitigation
question, not a replacement for that discipline.
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


def _per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    recalls = np.full(n_classes, np.nan)
    for k in range(n_classes):
        mask = y_true == k
        if mask.sum() > 0:
            recalls[k] = (y_pred[mask] == k).mean()
    return recalls


def _predict_with_bias(logits: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.argmax(logits + bias[None, :], axis=1)


def balance_recalls(logits: np.ndarray, y_true: np.ndarray, n_classes: int,
                     max_iters: int = 300, step: float = 0.05) -> dict:
    bias = np.zeros(n_classes)
    baseline_pred = _predict_with_bias(logits, bias)
    baseline_recall = _per_class_recall(y_true, baseline_pred, n_classes)
    baseline_acc = float((baseline_pred == y_true).mean())

    history = []
    for it in range(max_iters):
        pred = _predict_with_bias(logits, bias)
        recall = _per_class_recall(y_true, pred, n_classes)
        valid = ~np.isnan(recall)
        if valid.sum() < 2:
            break
        gap = float(np.nanmax(recall) - np.nanmin(recall))
        history.append(gap)
        if gap < 0.01:
            break
        lowest = np.nanargmin(recall)
        highest = np.nanargmax(recall)
        bias[lowest] += step
        bias[highest] -= step

    final_pred = _predict_with_bias(logits, bias)
    final_recall = _per_class_recall(y_true, final_pred, n_classes)
    final_acc = float((final_pred == y_true).mean())

    return {
        "n_iterations": len(history),
        "baseline": {"recall_per_class": baseline_recall.tolist(), "accuracy": baseline_acc,
                     "recall_gap": float(np.nanmax(baseline_recall) - np.nanmin(baseline_recall))},
        "balanced": {"recall_per_class": final_recall.tolist(), "accuracy": final_acc,
                     "recall_gap": float(np.nanmax(final_recall) - np.nanmin(final_recall)),
                     "bias_vector": bias.tolist()},
        "gap_history": history,
    }


def calibrate(dataset: str, eval_dir: Path, task: str) -> dict:
    df = pd.read_parquet(eval_dir / "test_predictions.parquet")
    labels = load_labels(dataset)
    class_names = labels[task]
    n_classes = len(class_names)

    valid = df[df[f"{task}_true"] != -1]
    y_true = valid[f"{task}_true"].to_numpy()
    logits = np.array(valid[f"{task}_logits"].tolist())

    result = balance_recalls(logits, y_true, n_classes)
    result["dataset"] = dataset
    result["task"] = task
    result["class_names"] = class_names
    result["n"] = int(len(valid))
    result["caveat"] = ("Calibrated directly on the test set for DIAGNOSTIC purposes only -- "
                         "a deployed bias vector should be chosen on a held-out validation set instead.")
    result["baseline"]["recall_by_class"] = dict(zip(class_names, result["baseline"]["recall_per_class"]))
    result["balanced"]["recall_by_class"] = dict(zip(class_names, result["balanced"]["recall_per_class"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface"])
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--task", default="race", choices=["race", "age_group"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = calibrate(args.dataset, args.eval_dir, args.task)
    out = args.out or (args.eval_dir / f"multiclass_calibration_{args.task}.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "gap_history"}, indent=2))


if __name__ == "__main__":
    main()
