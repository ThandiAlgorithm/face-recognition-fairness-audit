"""Figures to go alongside metrics.py's numeric output: confusion
matrices, ROC curves, PR curves, and calibration (reliability) diagrams.
Reads the same test_predictions.parquet, never re-runs the model.

Written to the folder layout the mitigation-experiment spec asks for:
  confusion_matrices/<task>.png
  roc_curves/<task>.png
  pr_curves/<task>.png
  calibration/<task>.png

For FairFace/UTKFace tasks (race/gender/age_group): one multi-class
confusion matrix per task, one ROC/PR plot per task with one curve per
class (one-vs-rest), one calibration plot per task averaging over classes.

For LFW: one confusion matrix + ROC + PR + calibration plot per attribute
(40 of each) -- matplotlib's Agg backend (non-interactive) is used
explicitly since this runs headless on the cluster.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from label_encoders import load_labels

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
CLASSIFICATION_TASKS = ["race", "gender", "age_group"]


def plot_confusion_matrix(y_true, y_pred, class_names: list[str], title: str, out_path: Path) -> None:
    """Row-normalized (% of each true class) -- raw counts alone don't say
    much when classes are imbalanced; this answers "of all true X, what %
    got predicted as each class", the standard convention for a %-based
    confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_pct = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100)
    fig, ax = plt.subplots(figsize=(max(4, len(class_names) * 0.8), max(4, len(class_names) * 0.8)))
    disp = ConfusionMatrixDisplay(cm_pct, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format=".1f")
    for text in disp.text_.ravel():
        text.set_text(text.get_text() + "%")
    ax.set_title(f"{title} (% of true class)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc_curves(y_true, prob_matrix, class_names: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    y_true = np.asarray(y_true)
    for i, cname in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        if len(set(y_true_bin)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin, prob_matrix[:, i])
        ax.plot(fpr, tpr, label=str(cname))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pr_curves(y_true, prob_matrix, class_names: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    y_true = np.asarray(y_true)
    for i, cname in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        if len(set(y_true_bin)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin, prob_matrix[:, i])
        ax.plot(recall, precision, label=str(cname))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_calibration(y_true, prob_matrix, class_names: list[str], title: str, out_path: Path, n_bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    y_true = np.asarray(y_true)
    for i, cname in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        if len(set(y_true_bin)) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(y_true_bin, prob_matrix[:, i], n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", markersize=3, label=str(cname))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_binary_task(y_true, y_pred, y_prob, task_label: str, out_dirs: dict) -> None:
    class_names = ["negative", "positive"]
    prob_matrix = np.column_stack([1 - np.asarray(y_prob), np.asarray(y_prob)])
    plot_confusion_matrix(y_true, y_pred, class_names, task_label, out_dirs["confusion_matrices"] / f"{task_label}.png")
    plot_roc_curves(y_true, prob_matrix, class_names, task_label, out_dirs["roc_curves"] / f"{task_label}.png")
    plot_pr_curves(y_true, prob_matrix, class_names, task_label, out_dirs["pr_curves"] / f"{task_label}.png")
    plot_calibration(y_true, prob_matrix, class_names, task_label, out_dirs["calibration"] / f"{task_label}.png")


def generate_all_plots(dataset: str, eval_dir: Path | None = None) -> None:
    eval_dir = eval_dir or (EXPERIMENTS_DIR / dataset / "test_eval")
    df = pd.read_parquet(eval_dir / "test_predictions.parquet")
    labels = load_labels(dataset)

    out_dirs = {name: eval_dir / name for name in ("confusion_matrices", "roc_curves", "pr_curves", "calibration")}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    if dataset == "lfw":
        for i, attr in enumerate(labels["attributes"]):
            y_true = df["attributes_true"].apply(lambda v, i=i: v[i])
            y_pred = df["attributes_pred"].apply(lambda v, i=i: v[i])
            y_prob = df["attributes_probs"].apply(lambda v, i=i: v[i])
            if y_true.nunique() < 2:
                continue
            _plot_binary_task(y_true, y_pred, y_prob, attr, out_dirs)
        return

    for task in CLASSIFICATION_TASKS:
        valid = df[df[f"{task}_true"] != -1]
        if valid.empty or f"{task}_probs" not in valid.columns:
            continue
        class_names = labels.get(task, [])
        prob_matrix = np.array(valid[f"{task}_probs"].tolist())
        plot_confusion_matrix(valid[f"{task}_true"], valid[f"{task}_pred"], class_names, task, out_dirs["confusion_matrices"] / f"{task}.png")
        plot_roc_curves(valid[f"{task}_true"], prob_matrix, class_names, task, out_dirs["roc_curves"] / f"{task}.png")
        plot_pr_curves(valid[f"{task}_true"], prob_matrix, class_names, task, out_dirs["pr_curves"] / f"{task}.png")
        plot_calibration(valid[f"{task}_true"], prob_matrix, class_names, task, out_dirs["calibration"] / f"{task}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface", "lfw"])
    parser.add_argument("--eval-dir", type=Path, default=None, help="Matches evaluate.py's --out-dir / metrics.py's --eval-dir.")
    args = parser.parse_args()
    generate_all_plots(args.dataset, eval_dir=args.eval_dir)
    print(f"Plots written under {args.eval_dir or f'experiments/{args.dataset}/test_eval'}/{{confusion_matrices,roc_curves,pr_curves,calibration}}/")


if __name__ == "__main__":
    main()
