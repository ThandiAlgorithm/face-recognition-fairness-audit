"""Extra deep-dive figures: weight-cap histograms (why resampling/
reweighting aren't near-perfect), UTKFace intersectional analysis
(race x gender, race x age, gender x age), and ROC curve overlays for
LFW/UTKFace's gender task. Run from face_resnet34_training/ with the
.venv-audit interpreter (needs numpy/pandas/matplotlib only).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from deep_dive_plots import intersection_heatmap, grouped_bar_chart, roc_curve_overlay, weight_cap_histogram
from label_encoders import load_labels

ROOT = Path(__file__).resolve().parents[3]
FACE_TRAINING = ROOT / "face_resnet34_training"
SCRATCH = Path(r"C:\Users\User1\AppData\Local\Temp\claude\c--Users-User1-projects-Algorithmic-Fairness-in-Computer-Vision\41c6a40f-7cfa-4bdf-bc13-3cf3c0ac94be\scratchpad")

LFW_OUT = ROOT / "results" / "lfw_fairness_audit" / "deep_dive" / "figures"
UTK_OUT = ROOT / "results" / "utkface_fairness_audit" / "deep_dive" / "figures"


def roc_from_scores(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    tpr = tp / n_pos if n_pos else tp * 0
    fpr = fp / n_neg if n_neg else fp * 0
    tpr = np.concatenate([[0], tpr, [1]])
    fpr = np.concatenate([[0], fpr, [1]])
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))  # manual trapezoidal rule
    return fpr, tpr, auc


def weight_cap_section() -> None:
    lfw_w = np.load(SCRATCH / "lfw_precap_weights.npy")
    utk_w = np.load(SCRATCH / "utk_precap_weights.npy")
    weight_cap_histogram(lfw_w, 4.0, "LFW: pre-cap rarity weight distribution vs the 4x cap applied",
                          LFW_OUT / "weight_cap_histogram.png")
    weight_cap_histogram(utk_w, 4.0, "UTKFace: pre-cap rarity weight distribution vs the 4x cap applied",
                          UTK_OUT / "weight_cap_histogram.png")
    print(f"weight cap histograms done (LFW max={lfw_w.max():.1f}x, UTKFace max={utk_w.max():.1f}x, both capped at 4.0x)")


def utkface_intersections() -> None:
    EXP = FACE_TRAINING / "experiments" / "utkface"
    METHOD_DIRS = {
        "Baseline": EXP / "test_eval",
        "Resampling": EXP / "M2_resampling_by_multi" / "seed_42" / "test_eval",
        "Reweighting": EXP / "M3_reweighting_multi" / "seed_42" / "test_eval",
    }
    LABELS = load_labels("utkface")
    RACES = LABELS["race"]
    AGES = LABELS["age_group"]

    def load_sub(method):
        with open(METHOD_DIRS[method] / "subgroup_metrics.csv", newline="") as f:
            return list(csv.DictReader(f))

    # 1. Race x gender F1 (gender task), full 10-cell grouped bar, all 3 methods
    groups10 = [f"{r}_{g}" for r in RACES for g in ("female", "male")]
    f1_by_method = {}
    for method in METHOD_DIRS:
        rows = load_sub(method)
        by_group = {r["group"]: r for r in rows if r["task"] == "gender" and r["sensitive_attribute"] == "race_gender_label"}
        f1_by_method[method] = [float(by_group[g]["f1"]) if g in by_group else float("nan") for g in groups10]
    grouped_bar_chart(groups10, f1_by_method, "UTKFace: gender F1 by race x gender, Baseline vs Resampling vs Reweighting",
                       "F1 (gender)", UTK_OUT / "intersection_race_gender_f1.png", value_fmt="{:.2f}")
    print("race x gender intersection bar chart done")

    # 2. Race x age F1 (gender task) heatmap, one per method
    for method in METHOD_DIRS:
        rows = load_sub(method)
        by_group = {r["group"]: r for r in rows if r["task"] == "gender" and r["sensitive_attribute"] == "race_age_label"}
        mat = np.full((len(RACES), len(AGES)), np.nan)
        nmat = np.full((len(RACES), len(AGES)), np.nan)
        for i, r in enumerate(RACES):
            for j, a in enumerate(AGES):
                key = f"{r}_{a}"
                if key in by_group:
                    mat[i, j] = float(by_group[key]["f1"])
                    nmat[i, j] = int(by_group[key]["n"])
        intersection_heatmap(mat, RACES, AGES, nmat,
                              f"UTKFace {method}: gender F1 by race x age", UTK_OUT / f"intersection_race_age_{method}.png",
                              value_label="F1 (gender)")
    print("race x age intersection heatmaps done")

    # 3. Gender x age accuracy (race task), grouped bar, all 3 methods
    groups8 = [f"{g}_{a}" for g in ("female", "male") for a in AGES]
    acc_by_method = {}
    for method in METHOD_DIRS:
        rows = load_sub(method)
        by_group = {r["group"]: r for r in rows if r["task"] == "race" and r["sensitive_attribute"] == "gender_age_label"}
        acc_by_method[method] = [float(by_group[g]["accuracy"]) * 100 if g in by_group else float("nan") for g in groups8]
    grouped_bar_chart(groups8, acc_by_method, "UTKFace: race accuracy by gender x age, Baseline vs Resampling vs Reweighting",
                       "Accuracy (%)", UTK_OUT / "intersection_gender_age_race_acc.png", value_fmt="{:.0f}")
    print("gender x age intersection bar chart done")


def roc_overlays() -> None:
    # LFW: Male attribute
    labels = load_labels("lfw")
    a_idx = labels["attributes"].index("Male")
    EXP = FACE_TRAINING / "experiments" / "lfw"
    dirs = {"Baseline": EXP / "test_eval",
            "Resampling": EXP / "M2_resampling_by_multi" / "seed_42" / "test_eval",
            "Reweighting": EXP / "M3_reweighting_multi" / "seed_42" / "test_eval"}
    curves = []
    for method, d in dirs.items():
        df = pd.read_parquet(d / "test_predictions.parquet")
        y_true = df["attributes_true"].apply(lambda v: v[a_idx]).to_numpy().astype(int)
        y_prob = df["attributes_probs"].apply(lambda v: v[a_idx]).to_numpy()
        fpr, tpr, auc = roc_from_scores(y_true, y_prob)
        curves.append({"method": method, "fpr": fpr, "tpr": tpr, "auc": auc})
    roc_curve_overlay(curves, "LFW: Male attribute ROC, Baseline vs Resampling vs Reweighting",
                       LFW_OUT / "roc_overlay_Male.png")
    print("LFW ROC overlay done")

    # UTKFace: gender task
    EXP2 = FACE_TRAINING / "experiments" / "utkface"
    utk_labels = load_labels("utkface")
    male_idx = utk_labels["gender"].index("male")
    dirs2 = {"Baseline": EXP2 / "test_eval",
             "Resampling": EXP2 / "M2_resampling_by_multi" / "seed_42" / "test_eval",
             "Reweighting": EXP2 / "M3_reweighting_multi" / "seed_42" / "test_eval"}
    curves2 = []
    for method, d in dirs2.items():
        df = pd.read_parquet(d / "test_predictions.parquet")
        valid = df[df["gender_true"] != -1]
        y_true = valid["gender_true"].to_numpy().astype(int)
        y_prob = valid["gender_probs"].apply(lambda v: v[male_idx]).to_numpy()
        fpr, tpr, auc = roc_from_scores(y_true, y_prob)
        curves2.append({"method": method, "fpr": fpr, "tpr": tpr, "auc": auc})
    roc_curve_overlay(curves2, "UTKFace: gender task ROC, Baseline vs Resampling vs Reweighting",
                       UTK_OUT / "roc_overlay_gender.png")
    print("UTKFace ROC overlay done")


if __name__ == "__main__":
    weight_cap_section()
    utkface_intersections()
    roc_overlays()
