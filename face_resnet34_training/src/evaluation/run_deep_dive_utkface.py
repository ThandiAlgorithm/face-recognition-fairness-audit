"""Generates all deep-dive figures for UTKFace: Baseline vs Resampling vs Reweighting.
Run from face_resnet34_training/ with the .venv-audit interpreter.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from deep_dive_plots import (binary_confusion_from_counts, confusion_from_predictions,
                              plot_confusion_matrix, grouped_bar_chart, n_vs_metric_scatter)
from label_encoders import load_labels

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "face_resnet34_training" / "experiments" / "utkface"
OUT = ROOT / "results" / "utkface_fairness_audit" / "deep_dive"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

METHOD_DIRS = {
    "Baseline": EXP / "test_eval",
    "Resampling": EXP / "M2_resampling_by_multi" / "seed_42" / "test_eval",
    "Reweighting": EXP / "M3_reweighting_multi" / "seed_42" / "test_eval",
}
LABELS = load_labels("utkface")
RACE_CLASSES = LABELS["race"]
AGE_CLASSES = LABELS["age_group"]
GENDER_CLASSES = LABELS["gender"]


def load_metrics(method: str) -> dict:
    return json.loads((METHOD_DIRS[method] / "metrics.json").read_text())


def load_subgroup(method: str) -> list[dict]:
    with open(METHOD_DIRS[method] / "subgroup_metrics.csv", newline="") as f:
        return list(csv.DictReader(f))


def load_predictions(method: str) -> pd.DataFrame:
    return pd.read_parquet(METHOD_DIRS[method] / "test_predictions.parquet")


def main() -> None:
    # 1. Confusion matrices
    for method in METHOD_DIRS:
        df = load_predictions(method)
        for task, classes in [("race", RACE_CLASSES), ("age_group", AGE_CLASSES), ("gender", GENDER_CLASSES)]:
            cm = confusion_from_predictions(df, f"{task}_true", f"{task}_pred", classes)
            plot_confusion_matrix(cm, classes, f"UTKFace {method} -- {task}",
                                   OUT / "figures" / f"confusion_{method}_{task}.png")
    print("confusion matrices done")

    # 2. Per-class recall bar charts, one per task, grouped by method
    for task, classes in [("race", RACE_CLASSES), ("age_group", AGE_CLASSES)]:
        method_vals = {}
        for method in METHOD_DIRS:
            m = load_metrics(method)
            method_vals[method] = [m[task]["per_class"][c]["recall_tpr"] * 100 for c in classes]
        grouped_bar_chart(classes, method_vals, f"UTKFace: {task} recall by class, Baseline vs Resampling vs Reweighting",
                           "Recall (%)", OUT / "figures" / f"{task}_recall_comparison.png", value_fmt="{:.1f}")
    print("per-class recall bar charts done")

    # 3. Gender F1 by race group, grouped by method
    race_gender_f1 = {}
    race_gender_n = {}
    for method in METHOD_DIRS:
        rows = load_subgroup(method)
        by_group = {r["group"]: r for r in rows
                    if r["task"] == "gender" and r["sensitive_attribute"] == "race_label"}
        races_present = [c for c in RACE_CLASSES if c in by_group]
        race_gender_f1[method] = [float(by_group[c]["f1"]) for c in races_present]
        race_gender_n[method] = {c: int(by_group[c]["n"]) for c in races_present}
    grouped_bar_chart(races_present, race_gender_f1,
                       "UTKFace: gender F1 by race, Baseline vs Resampling vs Reweighting",
                       "F1 (gender)", OUT / "figures" / "gender_f1_by_race_comparison.png", value_fmt="{:.2f}")
    print("gender-by-race bar chart done")

    # 4. Overall accuracy bar chart (race / gender / age)
    overall = {}
    for method in METHOD_DIRS:
        tm = json.loads((METHOD_DIRS[method] / "test_metrics.json").read_text())
        overall[method] = [tm["race_accuracy"] * 100, tm["gender_accuracy"] * 100, tm["age_group_accuracy"] * 100]
    grouped_bar_chart(["Race", "Gender", "Age"], overall,
                       "UTKFace: overall accuracy by task, Baseline vs Resampling vs Reweighting", "%",
                       OUT / "figures" / "overall_accuracy_comparison.png", value_fmt="{:.1f}")
    print("overall accuracy bar chart done")

    # 5. n vs F1 scatter for gender-by-race, all methods
    scatter_rows = []
    for method in METHOD_DIRS:
        for c in race_gender_n[method]:
            idx = races_present.index(c) if c in races_present else None
            if idx is not None:
                scatter_rows.append({"group": c, "n": race_gender_n[method][c],
                                      "value": race_gender_f1[method][idx], "method": method})
    n_vs_metric_scatter(scatter_rows, "UTKFace: race-group sample size vs gender F1 (all methods)",
                        OUT / "figures" / "n_vs_f1_scatter.png", metric_label="F1 (gender)")
    print("scatter done")


if __name__ == "__main__":
    main()
