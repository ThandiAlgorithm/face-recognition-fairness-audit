"""Evaluates BAL-M1 on its frozen HarmoFace-95K test split: per-task macro-F1
and balanced accuracy, plus equalized-odds gaps for each task broken down by
every OTHER demographic attribute (does the race head's accuracy vary by
gender, does the gender head's accuracy vary by race, and so on) -- the
piece needed for the poster's hero comparison table, which did not exist
before this script (BAL-M1 had never been evaluated past raw training
mean_macro_f1).

Point estimates only, no bootstrap CI -- matches how master_comparison.py
already reports M1/M2/M3 for UTKFace/FairFace without CIs, so this is
directly comparable to those numbers, not a different standard of rigor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from dataset import BalM1Dataset, RACE_VOCAB, GENDER_VOCAB, AGE_VOCAB, IGNORE_INDEX  # noqa: E402
from model import build_bal_m1_model  # noqa: E402
from transforms import eval_transform  # noqa: E402
from metrics import _group_rates, _gap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "harmoface" / "BAL-M1"
MANIFEST_PATH = EXP_DIR / "harmoface_manifest_enriched.csv"
TASKS = ["race", "gender", "age", "skin_tone"]


def load_model(device: str) -> tuple[torch.nn.Module, dict]:
    label_mappings = json.loads((EXP_DIR / "label_mappings.json").read_text())
    n_classes = {t: len(label_mappings[t]) for t in TASKS}
    model = build_bal_m1_model(n_classes["race"], n_classes["gender"], n_classes["age"], n_classes["skin_tone"])
    try:
        ckpt = torch.load(EXP_DIR / "best.pt", map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(EXP_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"loaded checkpoint from epoch {ckpt.get('epoch')}, mean_macro_f1={ckpt.get('mean_macro_f1')}")
    return model, label_mappings


def run_inference(model: torch.nn.Module, device: str) -> pd.DataFrame:
    skin_vocab = json.loads((EXP_DIR / "label_mappings.json").read_text())["skin_tone"]
    ds = BalM1Dataset(MANIFEST_PATH, "test", eval_transform(), skin_tone_vocab=skin_vocab)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    print(f"test set n={len(ds)}")

    rows = []
    with torch.no_grad():
        seen = 0
        for imgs, targets, row_batch in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = {t: outputs[t].argmax(dim=1).cpu().numpy() for t in TASKS}
            bs = imgs.shape[0]
            for i in range(bs):
                r = {
                    "race_true": int(targets["race"][i]), "race_pred": int(preds["race"][i]),
                    "gender_true": int(targets["gender"][i]), "gender_pred": int(preds["gender"][i]),
                    "age_true": int(targets["age"][i]), "age_pred": int(preds["age"][i]),
                    "skin_tone_true": int(targets["skin_tone"][i]), "skin_tone_pred": int(preds["skin_tone"][i]),
                }
                rows.append(r)
            seen += bs
            if seen % 1280 == 0:
                print(f"  {seen}/{len(ds)}")
    df = pd.DataFrame(rows)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(EXP_DIR / "test_predictions.parquet", index=False)
    print(f"wrote {len(df)} rows -> {EXP_DIR / 'test_predictions.parquet'}")
    return df


def task_metrics(df: pd.DataFrame, task: str) -> dict:
    valid = df[df[f"{task}_true"] != IGNORE_INDEX]
    y_true, y_pred = valid[f"{task}_true"], valid[f"{task}_pred"]
    return {
        "n": len(valid),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def equalized_odds_gap(df: pd.DataFrame, task: str, group_task: str) -> float | None:
    """Gap in task's one-vs-rest TPR/FPR across groups defined by
    group_task's TRUE label -- same point-estimate formula metrics.py
    uses project-wide (max-min across groups, NaN-dropped)."""
    valid = df[(df[f"{task}_true"] != IGNORE_INDEX) & (df[f"{group_task}_true"] != IGNORE_INDEX)]
    if valid.empty:
        return None
    y_true_full = valid[f"{task}_true"].to_numpy()
    y_pred_full = valid[f"{task}_pred"].to_numpy()
    group_codes = valid[f"{group_task}_true"].to_numpy()
    n_groups = int(group_codes.max()) + 1
    classes = sorted(valid[f"{task}_true"].unique().tolist())
    eo_vals = []
    for cls in classes:
        y_true_bin = (y_true_full == cls).astype(int)
        y_pred_bin = (y_pred_full == cls).astype(int)
        tpr, fpr, _, _ = _group_rates(y_true_bin, y_pred_bin, group_codes, n_groups)
        tpr_gap, fpr_gap = _gap(tpr), _gap(fpr)
        if not (np.isnan(tpr_gap) or np.isnan(fpr_gap)):
            eo_vals.append(max(tpr_gap, fpr_gap))
    return float(np.mean(eo_vals)) if eo_vals else None


def run() -> None:
    device = "cpu"  # gpu-node's K40c (compute capability 3.5) has no usable CUDA kernels for this torch build
    model, label_mappings = load_model(device)
    df = run_inference(model, device)

    metrics = {t: task_metrics(df, t) for t in TASKS}
    mean_macro_f1 = float(np.mean([metrics[t]["macro_f1"] for t in TASKS]))
    mean_balanced_acc = float(np.mean([metrics[t]["balanced_accuracy"] for t in TASKS]))

    gaps = {}
    for task in TASKS:
        for group_task in TASKS:
            if group_task == task:
                continue
            gaps[f"{task}_by_{group_task}"] = equalized_odds_gap(df, task, group_task)

    gap_by_attr = {}
    for attr in TASKS:
        vals = [v for k, v in gaps.items() if k.endswith(f"_by_{attr}") and v is not None]
        gap_by_attr[attr] = float(np.mean(vals)) if vals else None

    result = {
        "n_test": len(df),
        "per_task": metrics,
        "mean_macro_f1": mean_macro_f1,
        "mean_balanced_accuracy": mean_balanced_acc,
        "equalized_odds_gap_detail": gaps,
        "equalized_odds_gap_by_attribute": gap_by_attr,
    }
    (EXP_DIR / "test_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nwrote -> {EXP_DIR / 'test_metrics.json'}")


if __name__ == "__main__":
    run()
