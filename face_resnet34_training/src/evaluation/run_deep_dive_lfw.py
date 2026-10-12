"""Generates all deep-dive figures for LFW: Baseline vs Resampling vs Reweighting.
Run from face_resnet34_training/ with the .venv-audit interpreter.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deep_dive_plots import (binary_confusion_from_counts, plot_confusion_matrix,
                              grouped_bar_chart, n_vs_metric_scatter)

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "face_resnet34_training" / "experiments" / "lfw"
OUT = ROOT / "results" / "lfw_fairness_audit" / "deep_dive"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

METHOD_DIRS = {
    "Baseline": EXP / "test_eval",
    "Resampling": EXP / "M2_resampling_by_multi" / "seed_42" / "test_eval",
    "Reweighting": EXP / "M3_reweighting_multi" / "seed_42" / "test_eval",
}
ATTRS = ["Male", "White", "Black", "Asian", "Indian", "Pale_Skin"]
RACE_GENDER_GROUPS = ["White_female", "Black_female", "Asian_female", "Indian_female",
                       "White_male", "Black_male", "Asian_male", "Indian_male"]


def load_metrics(method: str) -> dict:
    return json.loads((METHOD_DIRS[method] / "metrics.json").read_text())


def load_subgroup(method: str) -> list[dict]:
    with open(METHOD_DIRS[method] / "subgroup_metrics.csv", newline="") as f:
        return list(csv.DictReader(f))


def load_test_metrics(method: str) -> dict:
    return json.loads((METHOD_DIRS[method] / "test_metrics.json").read_text())


def main() -> None:
    # 1. Confusion matrices: one per attribute per method
    for method in METHOD_DIRS:
        m = load_metrics(method)
        for attr in ATTRS:
            d = m[attr]
            cm = binary_confusion_from_counts(d["tp"], d["tn"], d["fp"], d["fn"])
            plot_confusion_matrix(
                cm, [f"not_{attr}", attr], f"LFW {method} -- {attr}",
                OUT / "figures" / f"confusion_{method}_{attr}.png",
            )
    print("confusion matrices done")

    # 2. Per-attribute F1 bar chart, grouped by method
    method_f1 = {}
    for method in METHOD_DIRS:
        m = load_metrics(method)
        method_f1[method] = [m[a]["f1"] for a in ATTRS]
    grouped_bar_chart(ATTRS, method_f1, "LFW: per-attribute F1, Baseline vs Resampling vs Reweighting",
                       "F1", OUT / "figures" / "attribute_f1_comparison.png", value_fmt="{:.2f}")
    print("attribute F1 bar chart done")

    # 3. Race x gender intersection F1 bar chart, grouped by method
    inter_f1 = {}
    inter_n = {}
    for method in METHOD_DIRS:
        rows = load_subgroup(method)
        by_group = {r["group"]: r for r in rows
                    if r["attribute"] == "Male" and r["sensitive_attribute"] == "race_gender_label"}
        inter_f1[method] = [float(by_group[g]["f1"]) if g in by_group else float("nan") for g in RACE_GENDER_GROUPS]
        inter_n[method] = {g: int(by_group[g]["n"]) for g in by_group}
    grouped_bar_chart(RACE_GENDER_GROUPS, inter_f1, "LFW: gender F1 by race x gender, Baseline vs Resampling vs Reweighting",
                       "F1 (gender/Male)", OUT / "figures" / "intersection_f1_comparison.png", value_fmt="{:.2f}")
    print("intersection F1 bar chart done")

    # 4. Overall mean-attribute-accuracy bar chart
    overall = {method: [load_test_metrics(method)["mean_attribute_accuracy"] * 100] for method in METHOD_DIRS}
    grouped_bar_chart(["Mean attribute accuracy (73 attrs)"], overall,
                       "LFW: overall accuracy, Baseline vs Resampling vs Reweighting", "%",
                       OUT / "figures" / "overall_accuracy_comparison.png", value_fmt="{:.1f}")
    print("overall accuracy bar chart done")

    # 5. n vs F1 scatter, race x gender intersections, all 3 methods
    scatter_rows = []
    for method in METHOD_DIRS:
        for g in RACE_GENDER_GROUPS:
            if g in inter_n[method]:
                val = inter_f1[method][RACE_GENDER_GROUPS.index(g)]
                scatter_rows.append({"group": g, "n": inter_n[method][g], "value": val, "method": method})
    n_vs_metric_scatter(scatter_rows, "LFW: subgroup sample size vs gender F1 (all methods)",
                        OUT / "figures" / "n_vs_f1_scatter.png", metric_label="F1 (gender/Male)")
    print("scatter done")


if __name__ == "__main__":
    main()
