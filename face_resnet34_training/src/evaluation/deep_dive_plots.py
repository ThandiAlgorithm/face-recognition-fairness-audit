"""Deep-dive visualizations for M1 vs M2-MULTI vs M3-MULTI: confusion
matrices (row-normalized to %, i.e. per-true-class recall), grouped bar
charts of per-group metrics across methods, and sample-size-vs-metric
scatter plots (to make the "small group = unreliable" pattern visible,
not just stated in prose).

Reads only already-computed outputs (metrics.json, subgroup_metrics.csv,
test_predictions.parquet from evaluate.py/metrics.py) -- never re-runs the
model. Headless (Agg backend), no display needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

METHOD_COLORS = {
    "Baseline (M1)": "#0057FF", "Resampling (M2)": "#FF6D00", "Reweighting (M3)": "#00C853",
    "Adversarial (M4)": "#FF1744", "Baseline": "#0057FF", "Resampling": "#FF6D00",
    "Reweighting": "#00C853", "Adversarial": "#FF1744",
}


def binary_confusion_from_counts(tp: int, tn: int, fp: int, fn: int) -> np.ndarray:
    """[[TN, FP], [FN, TP]] -- rows = true (neg, pos), cols = pred (neg, pos)."""
    return np.array([[tn, fp], [fn, tp]], dtype=float)


def plot_confusion_matrix(counts: np.ndarray, class_names: list[str], title: str, out_path: Path) -> None:
    """counts: raw confusion matrix, rows=true, cols=pred. Annotates both the
    row-normalized % (what's actually comparable across classes of different
    size) and the raw n underneath."""
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    pct = counts / row_sums * 100

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(1.3 * n + 2, 1.1 * n + 2))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    for i in range(n):
        for j in range(n):
            color = "white" if pct[i, j] > 55 else "black"
            ax.text(j, i, f"{pct[i, j]:.1f}%\n(n={int(counts[i, j])})",
                     ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="% of true class (row-normalized)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def confusion_from_predictions(df: pd.DataFrame, true_col: str, pred_col: str, class_names: list[str],
                                ignore_value=-1) -> np.ndarray:
    """true_col/pred_col hold integer class indices (0..len(class_names)-1,
    matching load_labels()'s frozen category order), not string names."""
    valid = df[df[true_col] != ignore_value]
    n = len(class_names)
    mat = np.zeros((n, n))
    for t, p in zip(valid[true_col], valid[pred_col]):
        if 0 <= t < n and 0 <= p < n:
            mat[int(t), int(p)] += 1
    return mat


def grouped_bar_chart(groups: list[str], method_values: dict[str, list[float]], title: str, ylabel: str,
                       out_path: Path, value_fmt: str = "{:.1f}") -> None:
    methods = list(method_values.keys())
    n_groups = len(groups)
    n_methods = len(methods)
    x = np.arange(n_groups)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * n_groups), 5))
    for i, method in enumerate(methods):
        vals = method_values[method]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=method,
                        color=METHOD_COLORS.get(method, None))
        for b, v in zip(bars, vals):
            if v == v:  # not NaN
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                         value_fmt.format(v), ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def intersection_heatmap(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
                          n_matrix: np.ndarray, title: str, out_path: Path,
                          value_label: str = "Value") -> None:
    """Grid too big for a readable grouped bar chart (e.g. 5 races x 4 age
    bands) -- a heatmap instead. NaN cells (no data for that combination)
    render blank with 'n/a'."""
    nrows, ncols = matrix.shape
    fig, ax = plt.subplots(figsize=(1.3 * ncols + 2, 1.0 * nrows + 2))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#EEEEEE")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=np.nanmax(matrix) if np.isfinite(np.nanmax(matrix)) else 1)
    ax.set_xticks(range(ncols)); ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(nrows)); ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=11)
    for i in range(nrows):
        for j in range(ncols):
            v = matrix[i, j]
            n = n_matrix[i, j] if n_matrix is not None else None
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="gray")
            else:
                txt = f"{v:.2f}" + (f"\n(n={int(n)})" if n is not None else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label=value_label)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def roc_curve_overlay(curves: list[dict], title: str, out_path: Path) -> None:
    """curves: [{'method': str, 'fpr': array, 'tpr': array, 'auc': float}, ...]"""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for c in curves:
        ax.plot(c["fpr"], c["tpr"], label=f"{c['method']} (AUC={c['auc']:.3f})",
                 color=METHOD_COLORS.get(c["method"], None), linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def weight_cap_histogram(pre_cap: np.ndarray, cap: float, title: str, out_path: Path) -> None:
    """Visualizes how much correction the MAX_WEIGHT cap actually throws
    away -- the uncapped weight distribution vs. the hard ceiling every
    rare image's real weight gets squashed down to."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.minimum(pre_cap, cap * 5), bins=100, color="#4472C4", alpha=0.8)
    ax.axvline(cap, color="#C00000", linestyle="--", linewidth=2, label=f"Applied cap ({cap:.1f}x)")
    ax.set_xlabel("Pre-cap combined rarity weight (x average), clipped at 5x cap for display")
    ax.set_ylabel("Number of training images")
    ax.set_title(title, fontsize=11)
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def n_vs_metric_scatter(rows: list[dict], title: str, out_path: Path, metric_label: str = "F1") -> None:
    """rows: [{'group': str, 'n': int, 'value': float, 'method': str}, ...]"""
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, color in METHOD_COLORS.items():
        pts = [r for r in rows if r["method"] == method]
        if not pts:
            continue
        ax.scatter([r["n"] for r in pts], [r["value"] for r in pts],
                    label=method, color=color, s=50, alpha=0.75, edgecolors="black", linewidths=0.5)
        for r in pts:
            ax.annotate(r["group"], (r["n"], r["value"]), fontsize=6, alpha=0.8,
                         xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Group sample size, n (log scale)")
    ax.set_ylabel(metric_label)
    ax.set_title(title, fontsize=11)
    ax.axhspan(0, 0.3, color="red", alpha=0.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
