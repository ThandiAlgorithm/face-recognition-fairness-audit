"""M0 deep dive: score-distribution histogram, genuine/impostor confusion
matrix (with %), subgroup TAR bar charts, deeper error analysis (what
characterizes false matches/false non-matches beyond the single-axis
subgroup table already in fairness_metrics.json), and a genuine attempt at
per-race-group EER threshold calibration -- checked against overall
accuracy/TAR, not just the FMR/FNMR gap (the LFW classification deep dive
found that gap-only checks can hide a precision collapse; this verifies
that doesn't happen here too).

Run from face_resnet34_training/ with the .venv-audit interpreter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
M0_DIR = ROOT / "face_resnet34_training" / "outputs" / "M0_LFW"
OUT = ROOT / "results" / "m0_identity_audit"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
(OUT / "data").mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def eer_threshold(scores_genuine: np.ndarray, scores_impostor: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds, return (threshold, eer) at the point FMR == FNMR."""
    all_scores = np.concatenate([scores_genuine, scores_impostor])
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0, 1, 400)))
    best_thr, best_gap, best_err = None, float("inf"), None
    for t in thresholds:
        fnmr = (scores_genuine < t).mean() if len(scores_genuine) else 0.0
        fmr = (scores_impostor >= t).mean() if len(scores_impostor) else 0.0
        gap = abs(fnmr - fmr)
        if gap < best_gap:
            best_gap, best_thr, best_err = gap, t, (fnmr + fmr) / 2
    return float(best_thr), float(best_err)


def main() -> None:
    df = pd.read_csv(M0_DIR / "test_scores.csv")
    genuine = df[df["same_identity"] == 1]
    impostor = df[df["same_identity"] == 0]
    global_thr = df["global_threshold"].iloc[0]

    # 1. Score distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(impostor["similarity_score"], bins=80, alpha=0.6, label=f"Impostor (n={len(impostor)})",
            color="#C00000", density=True)
    ax.hist(genuine["similarity_score"], bins=80, alpha=0.6, label=f"Genuine (n={len(genuine)})",
            color="#4472C4", density=True)
    ax.axvline(global_thr, color="black", linestyle="--", label=f"Global threshold ({global_thr:.3f})")
    ax.set_xlabel("Cosine similarity score")
    ax.set_ylabel("Density")
    ax.set_title("M0: genuine vs. impostor similarity score distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "score_distribution.png")
    plt.close(fig)
    print("score distribution histogram done")

    # 2. Confusion matrix: genuine/impostor x match/no-match, with %
    cm = np.array([
        [((impostor["predicted_match"] == 0)).sum(), ((impostor["predicted_match"] == 1)).sum()],
        [((genuine["predicted_match"] == 0)).sum(), ((genuine["predicted_match"] == 1)).sum()],
    ], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = cm / row_sums * 100
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    labels = ["No-match", "Match"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Impostor pair\n(true: different people)", "Genuine pair\n(true: same person)"])
    ax.set_xlabel("Predicted")
    ax.set_title("M0: verification confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            color = "white" if pct[i, j] > 55 else "black"
            ax.text(j, i, f"{pct[i, j]:.1f}%\n(n={int(cm[i, j])})", ha="center", va="center", color=color)
    fig.colorbar(im, ax=ax, label="% of row")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "confusion_matrix.png")
    plt.close(fig)
    print("confusion matrix done")

    # 3. Subgroup TAR bar chart (from fairness_metrics via direct recompute per race)
    race_groups = sorted(set(genuine["race_label_a"].dropna().unique()) - {"unknown"})
    tar_by_race = []
    n_by_race = []
    for g in race_groups:
        gen_g = genuine[genuine["race_label_a"] == g]
        tar_by_race.append((gen_g["predicted_match"] == 1).mean() * 100)
        n_by_race.append(len(gen_g))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(race_groups, tar_by_race, color="#4472C4")
    for b, n in zip(bars, n_by_race):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"n={n}", ha="center", fontsize=8)
    ax.set_ylabel("TAR at global threshold (%)")
    ax.set_title("M0: true accept rate by race")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "tar_by_race.png")
    plt.close(fig)
    print("TAR by race bar chart done")

    # 4. Deeper error analysis: what characterizes false matches / false non-matches
    fm = impostor[impostor["predicted_match"] == 1]
    fnm = genuine[genuine["predicted_match"] == 0]
    error_analysis = {
        "n_genuine": int(len(genuine)), "n_impostor": int(len(impostor)),
        "fnmr_overall": float((genuine["predicted_match"] == 0).mean()),
        "fmr_overall": float((impostor["predicted_match"] == 1).mean()),
        "false_match_rate_by_same_race": impostor.groupby("same_race_label")["predicted_match"].mean().to_dict(),
        "false_match_rate_by_same_gender": impostor.groupby("same_gender_label")["predicted_match"].mean().to_dict(),
        "note_age_group": "age_group_a/b are 100% null in test_scores.csv -- M0's pair table never "
                           "populated age for LFW pairs, so no age-based false-match/false-non-match "
                           "breakdown is possible from this data. This is a real gap in the M0 pipeline, "
                           "not a finding about the model.",
        "mean_similarity_false_match": float(fm["similarity_score"].mean()) if len(fm) else None,
        "mean_similarity_true_reject": float(impostor[impostor["predicted_match"] == 0]["similarity_score"].mean()),
        "mean_similarity_false_nonmatch": float(fnm["similarity_score"].mean()) if len(fnm) else None,
        "mean_similarity_true_accept": float(genuine[genuine["predicted_match"] == 1]["similarity_score"].mean()),
    }
    (OUT / "data" / "error_analysis.json").write_text(json.dumps(error_analysis, indent=2, default=str))
    print(json.dumps(error_analysis, indent=2, default=str))

    # 5. Per-race-group EER threshold: compute, then check net effect on
    # accuracy/TAR (not just the gap) -- same discipline as the LFW
    # classification deep dive.
    per_group_results = {}
    for g in race_groups:
        gen_scores = genuine[genuine["race_label_a"] == g]["similarity_score"].to_numpy()
        imp_scores = impostor[(impostor["race_label_a"] == g) & (impostor["same_race_label"] == 1)]["similarity_score"].to_numpy()
        if len(gen_scores) < 10 or len(imp_scores) < 10:
            per_group_results[g] = {"skipped": "too few pairs (<10) for a stable per-group threshold"}
            continue
        thr, eer = eer_threshold(gen_scores, imp_scores)
        tar_default = float((gen_scores >= global_thr).mean())
        frr_default = 1 - tar_default
        fmr_default = float((imp_scores >= global_thr).mean())
        tar_new = float((gen_scores >= thr).mean())
        fmr_new = float((imp_scores >= thr).mean())
        per_group_results[g] = {
            "n_genuine": int(len(gen_scores)), "n_impostor": int(len(imp_scores)),
            "default_threshold": float(global_thr), "default_tar": tar_default, "default_fmr": fmr_default,
            "per_group_threshold": thr, "per_group_tar": tar_new, "per_group_fmr": fmr_new,
            "per_group_eer": eer,
        }
    (OUT / "data" / "per_group_threshold_calibration.json").write_text(json.dumps(per_group_results, indent=2))
    print(json.dumps(per_group_results, indent=2))

    # Overall effect of switching to per-group thresholds, applied to the
    # WHOLE test set at once (not just within-race pairs, to see the real
    # net effect on the actual evaluation protocol)
    def predict_with_group_threshold(row):
        g = row["race_label_a"]
        thr = per_group_results.get(g, {}).get("per_group_threshold", global_thr)
        return 1 if row["similarity_score"] >= thr else 0

    df["predicted_match_per_group"] = df.apply(predict_with_group_threshold, axis=1)
    acc_default = float((df["predicted_match"] == (df["same_identity"])).mean())
    acc_per_group = float((df["predicted_match_per_group"] == (df["same_identity"])).mean())
    overall_comparison = {
        "accuracy_global_threshold": acc_default,
        "accuracy_per_group_threshold": acc_per_group,
        "tar_global_threshold": float((genuine["predicted_match"] == 1).mean()),
        "tar_per_group_threshold": float((df[df["same_identity"] == 1]["predicted_match_per_group"] == 1).mean()),
        "fmr_global_threshold": float((impostor["predicted_match"] == 1).mean()),
        "fmr_per_group_threshold": float((df[df["same_identity"] == 0]["predicted_match_per_group"] == 1).mean()),
    }
    (OUT / "data" / "per_group_threshold_overall_effect.json").write_text(json.dumps(overall_comparison, indent=2))
    print(json.dumps(overall_comparison, indent=2))


if __name__ == "__main__":
    main()
