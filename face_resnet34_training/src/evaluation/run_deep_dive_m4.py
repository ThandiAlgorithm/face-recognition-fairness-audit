"""Generates M4 (adversarial debiasing) figures: training trajectories
across all three datasets, and FairFace/UTKFace/LFW comparisons against
Baseline/Resampling/Reweighting now that M4 has evaluated results for
all three datasets. FairFace M4 training was stopped manually at epoch
11 (not early-stopped -- loss was still flat/stable); the evaluated
checkpoint is from epoch 9 (checkpoint_epoch in test_metrics.json).

Run from face_resnet34_training/ with the face-demo-app backend venv.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from deep_dive_plots import confusion_from_predictions, plot_confusion_matrix, grouped_bar_chart
from label_encoders import load_labels

ROOT = Path(__file__).resolve().parents[3]
FRT = ROOT / "face_resnet34_training"
OUT = ROOT / "report_writing_supporting" / "cross_dataset_findings"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 11})


def read_history(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


# ---------- 1. Adversarial training trajectory, all three datasets ----------
histories = {
    "FairFace": (FRT / "experiments/fairface/M4_adversarial/seed_42/adversarial_history.csv", 9, "stopped manually epoch 11 (not early-stopped); evaluated checkpoint is epoch 9"),
    "UTKFace": (FRT / "experiments/utkface/M4_adversarial/seed_42/adversarial_history.csv", 5, "early-stopped epoch 15; checkpoint saved at epoch 5"),
    "LFW": (FRT / "experiments/lfw/M4_adversarial/seed_42/adversarial_history.csv", 0, "early-stopped epoch 10; checkpoint saved at epoch 0"),
}
colors = {"FairFace": "#0057FF", "UTKFace": "#FFD600", "LFW": "#FF1744"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for name, (path, ckpt_epoch, note) in histories.items():
    if not path.exists():
        print(f"WARNING: missing history for {name}: {path}")
        continue
    rows = read_history(path)
    epochs = [r["epoch"] for r in rows]
    loss = [r["mean_loss"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    ax1.plot(epochs, loss, marker="o", label=name, color=colors[name])
    if ckpt_epoch is not None and int(ckpt_epoch) < len(loss):
        ax1.scatter([ckpt_epoch], [loss[int(ckpt_epoch)]],
                    marker="*", s=220, color=colors[name], edgecolor="black", zorder=5)
    ax2.plot(epochs, acc, marker="o", label=name, color=colors[name])

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Adversary mean loss")
ax1.set_title("Adversarial loss by epoch")
ax1.legend()
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Adversary accuracy, guessing ITA group (%)")
ax2.set_title("Adversary accuracy by epoch")
ax2.legend()
plt.suptitle("M4 adversarial debiasing: training trajectory by dataset", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "m4_training_trajectories.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved m4_training_trajectories.png")


# ---------- 2. UTKFace: M1/M2/M3/M4 comparison ----------
UTK_EXP = FRT / "experiments" / "utkface"
METHOD_DIRS = {
    "Baseline (M1)": UTK_EXP / "test_eval",
    "Resampling (M2)": UTK_EXP / "M2_resampling" / "seed_42" / "test_eval",
    "Reweighting (M3)": UTK_EXP / "M3_reweighting" / "seed_42" / "test_eval",
    "Adversarial (M4)": UTK_EXP / "M4_adversarial" / "seed_42" / "test_eval",
}
LABELS = load_labels("utkface")
RACE_CLASSES, AGE_CLASSES, GENDER_CLASSES = LABELS["race"], LABELS["age_group"], LABELS["gender"]

# confusion matrices for M4 specifically (the other three already exist from an earlier pass)
df_m4 = pd.read_parquet(METHOD_DIRS["Adversarial (M4)"] / "test_predictions.parquet")
for task, classes in [("race", RACE_CLASSES), ("age_group", AGE_CLASSES), ("gender", GENDER_CLASSES)]:
    cm = confusion_from_predictions(df_m4, f"{task}_true", f"{task}_pred", classes)
    plot_confusion_matrix(cm, classes, f"UTKFace Adversarial (M4) -- {task}",
                           OUT / f"m4_utkface_confusion_{task}.png")
print("saved UTKFace M4 confusion matrices")

# overall accuracy, all four methods
overall = {}
for method, d in METHOD_DIRS.items():
    tm = json.loads((d / "test_metrics.json").read_text())
    overall[method] = [tm["race_accuracy"] * 100, tm["gender_accuracy"] * 100, tm["age_group_accuracy"] * 100]
grouped_bar_chart(["Race", "Gender", "Age"], overall,
                   "UTKFace: overall accuracy by task, all four methods", "%",
                   OUT / "m4_utkface_overall_accuracy_comparison.png", value_fmt="{:.1f}")
print("saved UTKFace overall accuracy comparison (M1-M4)")

# per-class race recall, all four methods -- M4 has no metrics.json (only the
# simpler evaluate.py was run for it), so derive its recall directly from the
# predictions parquet instead; M1/M2/M3 keep using their existing metrics.json
race_vals = {}
for method, d in METHOD_DIRS.items():
    metrics_path = d / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        race_vals[method] = [m["race"]["per_class"][c]["recall_tpr"] * 100 for c in RACE_CLASSES]
    else:
        df = pd.read_parquet(d / "test_predictions.parquet")
        recalls = []
        for i, c in enumerate(RACE_CLASSES):
            mask = df["race_true"] == i
            recalls.append(float((df.loc[mask, "race_pred"] == i).mean()) * 100 if mask.sum() else 0.0)
        race_vals[method] = recalls
grouped_bar_chart(RACE_CLASSES, race_vals, "UTKFace: race recall by class, all four methods",
                   "Recall (%)", OUT / "m4_utkface_race_recall_comparison.png", value_fmt="{:.1f}")
print("saved UTKFace race recall comparison (M1-M4)")


# ---------- 2b. FairFace: M1/M2/M3/M4 comparison ----------
FF_EXP = FRT / "experiments" / "fairface"
FF_METHOD_DIRS = {
    "Baseline (M1)": FF_EXP / "test_eval",
    "Resampling (M2)": FF_EXP / "M2_resampling" / "seed_42" / "test_eval",
    "Reweighting (M3)": FF_EXP / "M3_reweighting" / "seed_42" / "test_eval",
    "Adversarial (M4)": FF_EXP / "M4_adversarial" / "seed_42" / "test_eval",
}
FF_LABELS = load_labels("fairface")
FF_RACE_CLASSES, FF_AGE_CLASSES, FF_GENDER_CLASSES = FF_LABELS["race"], FF_LABELS["age_group"], FF_LABELS["gender"]

df_ff_m4 = pd.read_parquet(FF_METHOD_DIRS["Adversarial (M4)"] / "test_predictions.parquet")
for task, classes in [("race", FF_RACE_CLASSES), ("age_group", FF_AGE_CLASSES), ("gender", FF_GENDER_CLASSES)]:
    cm = confusion_from_predictions(df_ff_m4, f"{task}_true", f"{task}_pred", classes)
    plot_confusion_matrix(cm, classes, f"FairFace Adversarial (M4) -- {task}",
                           OUT / f"m4_fairface_confusion_{task}.png")
print("saved FairFace M4 confusion matrices")

ff_overall = {}
for method, d in FF_METHOD_DIRS.items():
    tm = json.loads((d / "test_metrics.json").read_text())
    ff_overall[method] = [tm["race_accuracy"] * 100, tm["gender_accuracy"] * 100, tm["age_group_accuracy"] * 100]
grouped_bar_chart(["Race", "Gender", "Age"], ff_overall,
                   "FairFace: overall accuracy by task, all four methods", "%",
                   OUT / "m4_fairface_overall_accuracy_comparison.png", value_fmt="{:.1f}")
print("saved FairFace overall accuracy comparison (M1-M4)")

# per-class race recall, all four methods -- M4 (and M2/M3 here) have no
# metrics.json (only the simpler evaluate.py was run), so derive recall
# directly from the predictions parquet for any method missing one.
ff_race_vals = {}
for method, d in FF_METHOD_DIRS.items():
    metrics_path = d / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        ff_race_vals[method] = [m["race"]["per_class"][c]["recall_tpr"] * 100 for c in FF_RACE_CLASSES]
    else:
        df = pd.read_parquet(d / "test_predictions.parquet")
        recalls = []
        for i, c in enumerate(FF_RACE_CLASSES):
            mask = df["race_true"] == i
            recalls.append(float((df.loc[mask, "race_pred"] == i).mean()) * 100 if mask.sum() else 0.0)
        ff_race_vals[method] = recalls
grouped_bar_chart(FF_RACE_CLASSES, ff_race_vals, "FairFace: race recall by class, all four methods",
                   "Recall (%)", OUT / "m4_fairface_race_recall_comparison.png", value_fmt="{:.1f}")
print("saved FairFace race recall comparison (M1-M4)")


# ---------- 3. LFW: M1/M2/M3/M4 gender comparison ----------
LFW_EXP = FRT / "experiments" / "lfw"
LFW_DIRS = {
    "Baseline (M1)": LFW_EXP / "test_eval",
    "Resampling (M2)": LFW_EXP / "M2_resampling" / "seed_42" / "test_eval",
    "Reweighting (M3)": LFW_EXP / "M3_reweighting" / "seed_42" / "test_eval",
    "Adversarial (M4)": LFW_EXP / "M4_adversarial" / "seed_42" / "test_eval",
}
male_acc = {}
for method, d in LFW_DIRS.items():
    tm_path = d / "test_metrics.json"
    if not tm_path.exists():
        continue
    tm = json.loads(tm_path.read_text())
    # LFW's test_metrics stores per-attribute accuracy; Male is the gender proxy
    if "per_attribute_accuracy" in tm:
        male_acc[method] = tm["per_attribute_accuracy"].get("Male")
    elif "male_accuracy" in tm:
        male_acc[method] = tm["male_accuracy"]
print("LFW Male attribute accuracy by method:", male_acc)

fig, ax = plt.subplots(figsize=(7, 5))
methods = list(male_acc.keys())
vals = [male_acc[m] * 100 for m in methods]
bar_colors = ["#0057FF", "#FF6D00", "#00C853", "#FF1744"][: len(methods)]
ax.bar(methods, vals, color=bar_colors)
for i, v in enumerate(vals):
    ax.text(i, v + 0.3, f"{v:.1f}", ha="center")
ax.set_ylabel("Male attribute accuracy (%)")
ax.set_title("LFW: gender (Male attribute) accuracy, all four methods")
ax.set_ylim(0, 100)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUT / "m4_lfw_gender_accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("saved LFW gender accuracy comparison (M1-M4)")

print("\nALL M4 FIGURES SAVED")
