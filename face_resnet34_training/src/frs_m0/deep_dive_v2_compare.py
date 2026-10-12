"""M0 v1 (softmax) vs v2 (ArcFace) comparison figures: score-distribution
overlay, confusion matrices side by side, and a subgroup TAR bar chart.
Run from face_resnet34_training/ with the .venv-audit interpreter.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
V1_DIR = ROOT / "face_resnet34_training" / "outputs" / "M0_LFW"
V2_DIR = ROOT / "face_resnet34_training" / "outputs" / "M0_LFW_arcface"
OUT = ROOT / "results" / "m0_identity_audit"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def main() -> None:
    v1 = pd.read_csv(V1_DIR / "test_scores.csv")
    v2 = pd.read_csv(V2_DIR / "test_scores.csv")

    # 1. Score distribution: v1 vs v2, genuine vs impostor, side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, df, label, thr in [
        (axes[0], v1, "v1 (plain softmax)", df_threshold(v1)),
        (axes[1], v2, "v2 (ArcFace)", df_threshold(v2)),
    ]:
        genuine = df[df["same_identity"] == 1]
        impostor = df[df["same_identity"] == 0]
        ax.hist(impostor["similarity_score"], bins=80, alpha=0.6, label="Impostor", color="#C00000", density=True)
        ax.hist(genuine["similarity_score"], bins=80, alpha=0.6, label="Genuine", color="#4472C4", density=True)
        ax.axvline(thr, color="black", linestyle="--", label=f"Threshold ({thr:.3f})")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Cosine similarity score")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Density")
    fig.suptitle("M0: genuine vs. impostor score separation, v1 vs v2", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "score_distribution_v1_vs_v2.png")
    plt.close(fig)
    print("score distribution comparison done")

    # 2. Overall metrics bar chart
    m1 = json.loads((V1_DIR / "overall_metrics.json").read_text())
    m2 = json.loads((V2_DIR / "overall_metrics.json").read_text())
    metrics = ["roc_auc", "accuracy_at_global_threshold", "tar_at_global_threshold"]
    labels = ["AUROC", "Accuracy", "TAR"]
    v1_vals = [m1[m] * 100 if m != "roc_auc" else m1[m] * 100 for m in metrics]
    v2_vals = [m2[m] * 100 if m != "roc_auc" else m2[m] * 100 for m in metrics]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - width/2, v1_vals, width, label="v1 (softmax)", color="#ED7D31")
    b2 = ax.bar(x + width/2, v2_vals, width, label="v2 (ArcFace)", color="#4472C4")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{b.get_height():.1f}",
                     ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_title("M0: overall quality, v1 vs v2")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "overall_metrics_v1_vs_v2.png")
    plt.close(fig)
    print("overall metrics bar chart done")

    # 3. EER bar chart (lower is better, flip framing)
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(["v1 (softmax)", "v2 (ArcFace)"], [m1["eer"]*100, m2["eer"]*100], color=["#ED7D31", "#4472C4"])
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{b.get_height():.1f}%", ha="center")
    ax.set_ylabel("EER (%) -- lower is better")
    ax.set_title("M0: Equal Error Rate, v1 vs v2")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "eer_v1_vs_v2.png")
    plt.close(fig)
    print("EER bar chart done")

    # 4. Race-group TAR bar chart, v1 vs v2
    f1 = json.loads((V1_DIR / "fairness_metrics.json").read_text())
    f2 = json.loads((V2_DIR / "fairness_metrics.json").read_text())
    sub1 = pd.read_csv(V1_DIR / "subgroup_metrics.csv")
    sub2 = pd.read_csv(V2_DIR / "subgroup_metrics.csv")
    races = ["White", "Black", "Asian", "Indian"]
    v1_tar = [float(sub1[(sub1["axis"]=="race") & (sub1["group"]==r)]["tar"].iloc[0]) * 100 if len(sub1[(sub1["axis"]=="race") & (sub1["group"]==r)]) else float("nan") for r in races]
    v2_tar = [float(sub2[(sub2["axis"]=="race") & (sub2["group"]==r)]["tar"].iloc[0]) * 100 if len(sub2[(sub2["axis"]=="race") & (sub2["group"]==r)]) else float("nan") for r in races]
    x = np.arange(len(races))
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width/2, v1_tar, width, label="v1 (softmax)", color="#ED7D31")
    b2 = ax.bar(x + width/2, v2_tar, width, label="v2 (ArcFace)", color="#4472C4")
    for bars in (b1, b2):
        for b, v in zip(bars, bars.datavalues):
            if v == v:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(races)
    ax.set_ylabel("TAR (%)")
    ax.set_title("M0: true accept rate by race, v1 vs v2")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "tar_by_race_v1_vs_v2.png")
    plt.close(fig)
    print("race TAR comparison done")


def df_threshold(df: pd.DataFrame) -> float:
    return float(df["global_threshold"].iloc[0])


if __name__ == "__main__":
    main()
