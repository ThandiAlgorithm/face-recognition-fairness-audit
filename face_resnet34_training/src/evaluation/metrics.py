"""Comprehensive metrics computed purely from saved test predictions
(evaluate.py's test_predictions.parquet) -- never re-runs the model.

Three outputs per dataset/method/seed:
  - metrics.json            overall + per-class classification metrics
  - subgroup_metrics.csv    per-group performance (race/gender/ita_group)
  - fairness_metrics.csv    gap metrics + formal fairness criteria, with
                             bootstrap 95% confidence intervals

Bootstrap resampling (1000 resamples) resamples whole identities for LFW
(images from the same person aren't independent) and individual rows for
FairFace/UTKFace (no reliable identity IDs there).

FairFace/UTKFace tasks (race, gender, age_group) are treated as multiclass
(one-vs-rest per class + macro/micro/weighted averages); LFW's 40 attributes
are each treated as an independent binary task.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from label_encoders import load_labels

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
# race_gender_label is derived (see _add_intersectional_columns), not a raw
# saved column -- a true joint subgroup ("Black_Female" vs "White_Male"),
# not just race-alone or gender-alone conditioning. Intersectional gaps can
# be large even when both marginal (single-attribute) views look fine --
# the whole point of auditing this separately (Buolamwini & Gebru 2018).
SENSITIVE_COLS = [
    "race_label", "gender_label", "ita_group", "age_group",
    "race_gender_label", "race_ita_label", "gender_ita_label", "fitzpatrick_approx",
]
N_BOOTSTRAP = 1000
CLASSIFICATION_TASKS = ["race", "gender", "age_group"]


# Approximate ITA(deg) -> Fitzpatrick-type band, from the commonly cited
# dermatology correspondence (e.g. Chardon et al. 1991; used in several
# skin-tone CV fairness papers as the standard ITA<->Fitzpatrick bridge).
# This is NOT a clinical Fitzpatrick assignment -- no annotator or
# self-report exists for these datasets -- it is an image-derived
# approximation, documented as such everywhere it's used (see
# ITA_VS_FITZPATRICK.md). Breakpoints are upper-bound-exclusive except the
# first.
FITZPATRICK_ITA_BREAKPOINTS = [
    (55.0, "I-II (very light)"),
    (41.0, "II-III (light)"),
    (28.0, "III-IV (intermediate)"),
    (10.0, "IV-V (tan)"),
    (-30.0, "V (brown)"),
    (float("-inf"), "VI (dark)"),
]


def _approximate_fitzpatrick(ita_value) -> str:
    if ita_value is None or pd.isna(ita_value):
        return "unknown"
    for lower, label in FITZPATRICK_ITA_BREAKPOINTS:
        if ita_value > lower:
            return label
    return "unknown"


def _add_intersectional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived grouping columns not present in the raw saved
    predictions:
      race_gender_label       = '<race>_<gender>' (e.g. 'Black_Female')
      fitzpatrick_approx       = ITA-derived approximate Fitzpatrick band
      race_ita_label           = '<race>_<ita_group>' (race x skin-tone)
      gender_ita_label         = '<gender>_<ita_group>' (gender x skin-tone)
    Either side missing/unknown -> 'unknown' for the joint column too,
    rather than a half-formed label that would silently pollute a real
    subgroup's statistics."""
    df = df.copy()
    race = df.get("race_label")
    gender = df.get("gender_label")
    ita_group = df.get("ita_group")
    ita_value = df.get("ita_value")

    def _join(a, b):
        if pd.isna(a) or pd.isna(b) or a == "unknown" or b == "unknown":
            return "unknown"
        return f"{a}_{b}"

    df["race_gender_label"] = [_join(r, g) for r, g in zip(race, gender)] if race is not None and gender is not None else "unknown"
    df["fitzpatrick_approx"] = [_approximate_fitzpatrick(v) for v in ita_value] if ita_value is not None else "unknown"
    df["race_ita_label"] = [_join(r, s) for r, s in zip(race, ita_group)] if race is not None and ita_group is not None else "unknown"
    df["gender_ita_label"] = [_join(g, s) for g, s in zip(gender, ita_group)] if gender is not None and ita_group is not None else "unknown"
    return df


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        return {"error": str(exc)}


def expected_calibration_error(y_true_bin: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(y_prob)
    if n == 0:
        return float("nan")
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true_bin[mask].mean() - y_prob[mask].mean())
    return float(ece)


def binary_metrics(y_true, y_pred, y_prob=None) -> dict:
    """y_true/y_pred in {0,1}; y_prob = P(class=1), optional."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"n": 0}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    m = {
        "n": int(len(y_true)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_ppv": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "recall_tpr": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity_tnr": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "fpr": float(fp / (tn + fp)) if (tn + fp) else float("nan"),
        "fnr": float(fn / (tp + fn)) if (tp + fn) else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan"),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        y_prob = np.asarray(y_prob)
        m["auroc"] = float(roc_auc_score(y_true, y_prob))
        m["auprc"] = float(average_precision_score(y_true, y_prob))
        try:
            m["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
        except ValueError:
            m["log_loss"] = float("nan")
        m["brier"] = float(brier_score_loss(y_true, y_prob))
        m["ece"] = expected_calibration_error(y_true, y_prob)
    return m


def multiclass_metrics(y_true, y_pred, y_prob_matrix=None, class_names: list[str] | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"n": 0}
    class_names = class_names or sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    result = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan"),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob_matrix is not None:
        y_prob_matrix = np.asarray(y_prob_matrix)
        labels_present = sorted(set(y_true.tolist()))
        if len(labels_present) > 1:
            auroc = _safe(
                roc_auc_score,
                y_true,
                y_prob_matrix,
                multi_class="ovr",
                average="macro",
                labels=list(range(len(class_names))),
            )
            result["macro_auroc_ovr"] = auroc if isinstance(auroc, dict) else float(auroc)

    per_class = {}
    for i, cname in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        y_pred_bin = (y_pred == i).astype(int)
        y_prob_bin = y_prob_matrix[:, i] if y_prob_matrix is not None else None
        per_class[str(cname)] = binary_metrics(y_true_bin, y_pred_bin, y_prob_bin)
    result["per_class"] = per_class
    return result


def subgroup_table(df: pd.DataFrame, task: str, class_names: list[str] | None) -> pd.DataFrame:
    rows = []
    valid = df[df[f"{task}_true"] != -1] if class_names else df
    for sens_col in SENSITIVE_COLS:
        if sens_col not in valid.columns:
            continue
        for group_val, g in valid.groupby(sens_col):
            if len(g) == 0:
                continue
            if class_names and len(class_names) > 2:
                m = multiclass_metrics(g[f"{task}_true"], g[f"{task}_pred"], class_names=class_names)
                flat = {k: v for k, v in m.items() if not isinstance(v, dict)}
            else:
                flat = binary_metrics(g[f"{task}_true"], g[f"{task}_pred"])
            rows.append({"task": task, "sensitive_attribute": sens_col, "group": str(group_val), "n": len(g), **flat})
    return pd.DataFrame(rows)


def _group_rates(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, group_codes: np.ndarray, n_groups: int):
    """Per-group TPR/FPR/precision/selection-rate via direct boolean sums --
    no sklearn/pandas calls, since this runs inside a 1000-iteration
    bootstrap loop and sklearn's per-call overhead dominated runtime when
    this was first written using binary_metrics() here (confirmed via a
    smoke test that took minutes on just 48 rows)."""
    tpr = np.full(n_groups, np.nan)
    fpr = np.full(n_groups, np.nan)
    prec = np.full(n_groups, np.nan)
    sel = np.full(n_groups, np.nan)
    for g in range(n_groups):
        mask = group_codes == g
        if not mask.any():
            continue
        yt, yp = y_true_bin[mask], y_pred_bin[mask]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tpr[g] = tp / (tp + fn) if (tp + fn) else np.nan
        fpr[g] = fp / (fp + tn) if (fp + tn) else np.nan
        prec[g] = tp / (tp + fp) if (tp + fp) else np.nan
        sel[g] = float(yp.mean())
    return tpr, fpr, prec, sel


def _gap(arr: np.ndarray) -> float:
    valid = arr[~np.isnan(arr)]
    return float(valid.max() - valid.min()) if len(valid) >= 2 else float("nan")


def _ratio(arr: np.ndarray) -> float:
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2 or valid.max() == 0:
        return float("nan")
    return float(valid.min() / valid.max())


def _bootstrap_fairness(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    group_codes: np.ndarray,
    n_groups: int,
    identity_codes: np.ndarray | None,
    n: int = N_BOOTSTRAP,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    """One bootstrap loop computing all six gap/ratio metrics together per
    resample (rather than one loop per metric) -- a 6x reduction in
    iterations on top of the per-iteration numpy speedup above."""

    def point_values(yt, yp, gc):
        tpr, fpr, prec, sel = _group_rates(yt, yp, gc, n_groups)
        tpr_gap, fpr_gap, prec_gap = _gap(tpr), _gap(fpr), _gap(prec)
        dp_diff, dp_ratio = _gap(sel), _ratio(sel)
        eo = max(tpr_gap, fpr_gap) if not (np.isnan(tpr_gap) or np.isnan(fpr_gap)) else float("nan")
        return tpr_gap, fpr_gap, prec_gap, dp_diff, dp_ratio, eo

    keys = ["tpr_gap", "fpr_gap", "precision_gap", "demographic_parity_diff", "demographic_parity_ratio", "equalized_odds_diff"]
    point = dict(zip(keys, point_values(y_true_bin, y_pred_bin, group_codes)))

    rng = np.random.default_rng(seed)
    samples = {k: [] for k in keys}
    n_rows = len(y_true_bin)

    if identity_codes is not None:
        unique_ids = np.unique(identity_codes)
        id_to_rows: dict = {}
        for idx, iid in enumerate(identity_codes):
            id_to_rows.setdefault(iid, []).append(idx)
        for _ in range(n):
            sampled_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)
            rows = np.concatenate([id_to_rows[i] for i in sampled_ids])
            for k, v in zip(keys, point_values(y_true_bin[rows], y_pred_bin[rows], group_codes[rows])):
                samples[k].append(v)
    else:
        for _ in range(n):
            rows = rng.integers(0, n_rows, size=n_rows)
            for k, v in zip(keys, point_values(y_true_bin[rows], y_pred_bin[rows], group_codes[rows])):
                samples[k].append(v)

    out = {}
    for k in keys:
        arr = np.array([v for v in samples[k] if not np.isnan(v)])
        lo, hi = (np.percentile(arr, [2.5, 97.5]) if len(arr) else (float("nan"), float("nan")))
        out[k] = (point[k], float(lo), float(hi))
    return out


def fairness_gap_and_formal_metrics(df: pd.DataFrame, task: str, identity_col: str | None) -> pd.DataFrame:
    """Gap metrics (best-worst, TPR/FPR/precision gaps) plus formal
    fairness criteria (equalized odds / equality of opportunity /
    demographic parity diff+ratio / predictive parity), each with a
    bootstrap 95% CI, for every sensitive attribute and -- for multiclass
    tasks -- every one-vs-rest class."""
    valid = df[df[f"{task}_true"] != -1]
    rows = []
    identity_codes_full = (
        valid[identity_col].to_numpy() if identity_col and identity_col in valid.columns else None
    )
    for sens_col in SENSITIVE_COLS:
        if sens_col not in valid.columns or valid[sens_col].nunique() < 2:
            continue
        group_series, group_names = pd.factorize(valid[sens_col].astype(str), sort=True)
        n_groups = len(group_names)
        if n_groups < 2:
            continue

        classes = sorted(valid[f"{task}_true"].unique().tolist())
        is_binary = len(classes) <= 2
        target_classes = [1] if is_binary else classes

        y_true_full = valid[f"{task}_true"].to_numpy()
        y_pred_full = valid[f"{task}_pred"].to_numpy()

        for cls in target_classes:
            y_true_bin = (y_true_full == cls).astype(int)
            y_pred_bin = (y_pred_full == cls).astype(int)

            result = _bootstrap_fairness(y_true_bin, y_pred_bin, group_series, n_groups, identity_codes_full)

            rows.append(
                {
                    "task": task,
                    "class": cls if not is_binary else "positive",
                    "sensitive_attribute": sens_col,
                    "n_groups": n_groups,
                    "tpr_gap": result["tpr_gap"][0], "tpr_gap_ci_lo": result["tpr_gap"][1], "tpr_gap_ci_hi": result["tpr_gap"][2],
                    "fpr_gap": result["fpr_gap"][0], "fpr_gap_ci_lo": result["fpr_gap"][1], "fpr_gap_ci_hi": result["fpr_gap"][2],
                    "precision_gap": result["precision_gap"][0], "precision_gap_ci_lo": result["precision_gap"][1], "precision_gap_ci_hi": result["precision_gap"][2],
                    "equality_of_opportunity_diff": result["tpr_gap"][0],
                    "equality_of_opportunity_ci_lo": result["tpr_gap"][1], "equality_of_opportunity_ci_hi": result["tpr_gap"][2],
                    "equalized_odds_diff": result["equalized_odds_diff"][0], "equalized_odds_ci_lo": result["equalized_odds_diff"][1], "equalized_odds_ci_hi": result["equalized_odds_diff"][2],
                    "demographic_parity_diff": result["demographic_parity_diff"][0], "demographic_parity_diff_ci_lo": result["demographic_parity_diff"][1], "demographic_parity_diff_ci_hi": result["demographic_parity_diff"][2],
                    "demographic_parity_ratio": result["demographic_parity_ratio"][0], "demographic_parity_ratio_ci_lo": result["demographic_parity_ratio"][1], "demographic_parity_ratio_ci_hi": result["demographic_parity_ratio"][2],
                    "predictive_parity_diff": result["precision_gap"][0], "predictive_parity_diff_ci_lo": result["precision_gap"][1], "predictive_parity_diff_ci_hi": result["precision_gap"][2],
                }
            )
    return pd.DataFrame(rows)


def compute_classification_section(df: pd.DataFrame, dataset: str, labels: dict) -> dict:
    if dataset == "lfw":
        result = {}
        for i, attr in enumerate(labels["attributes"]):
            y_true = df["attributes_true"].apply(lambda v: v[i])
            y_pred = df["attributes_pred"].apply(lambda v: v[i])
            y_prob = df["attributes_probs"].apply(lambda v: v[i])
            result[attr] = binary_metrics(y_true, y_pred, y_prob)
        return result

    result = {}
    for task in CLASSIFICATION_TASKS:
        valid = df[df[f"{task}_true"] != -1]
        if valid.empty:
            continue
        class_names = labels.get(task, [])
        prob_matrix = np.array(valid[f"{task}_probs"].tolist()) if f"{task}_probs" in valid.columns else None
        result[task] = multiclass_metrics(valid[f"{task}_true"], valid[f"{task}_pred"], prob_matrix, class_names)
    return result


def compute_all_metrics(dataset: str, eval_dir: Path | None = None) -> dict:
    eval_dir = eval_dir or (EXPERIMENTS_DIR / dataset / "test_eval")
    df = pd.read_parquet(eval_dir / "test_predictions.parquet")
    df = _add_intersectional_columns(df)
    labels = load_labels(dataset)
    identity_col = "identity_id" if dataset == "lfw" else None

    classification = compute_classification_section(df, dataset, labels)
    (eval_dir / "metrics.json").write_text(json.dumps(classification, indent=2, default=str))

    if dataset == "lfw":
        subgroup_frames = []
        fairness_frames = []
        for i, attr in enumerate(labels["attributes"]):
            tmp = df.copy()
            tmp[f"{attr}_true"] = tmp["attributes_true"].apply(lambda v, i=i: v[i])
            tmp[f"{attr}_pred"] = tmp["attributes_pred"].apply(lambda v, i=i: v[i])
            sub = subgroup_table(tmp.rename(columns={f"{attr}_true": "x_true", f"{attr}_pred": "x_pred"}), "x", None)
            sub.insert(0, "attribute", attr)
            subgroup_frames.append(sub)
            fair = fairness_gap_and_formal_metrics(
                tmp.rename(columns={f"{attr}_true": "x_true", f"{attr}_pred": "x_pred"}), "x", identity_col
            )
            fair.insert(0, "attribute", attr)
            fairness_frames.append(fair)
        subgroup_df = pd.concat(subgroup_frames, ignore_index=True) if subgroup_frames else pd.DataFrame()
        fairness_df = pd.concat(fairness_frames, ignore_index=True) if fairness_frames else pd.DataFrame()
    else:
        subgroup_frames, fairness_frames = [], []
        for task in CLASSIFICATION_TASKS:
            if f"{task}_true" not in df.columns:
                continue
            class_names = labels.get(task, [])
            subgroup_frames.append(subgroup_table(df, task, class_names))
            fairness_frames.append(fairness_gap_and_formal_metrics(df, task, identity_col))
        subgroup_df = pd.concat(subgroup_frames, ignore_index=True) if subgroup_frames else pd.DataFrame()
        fairness_df = pd.concat(fairness_frames, ignore_index=True) if fairness_frames else pd.DataFrame()

    subgroup_df.to_csv(eval_dir / "subgroup_metrics.csv", index=False)
    fairness_df.to_csv(eval_dir / "fairness_metrics.csv", index=False)
    generate_highlights(dataset, classification, subgroup_df, fairness_df, eval_dir)

    return {
        "classification": classification,
        "n_subgroup_rows": len(subgroup_df),
        "n_fairness_rows": len(fairness_df),
    }


def generate_highlights(
    dataset: str, classification: dict, subgroup_df: pd.DataFrame, fairness_df: pd.DataFrame, eval_dir: Path
) -> dict:
    """Pulls gender (and, for FairFace/UTKFace, race) results out to their
    own file. LFW's model predicts 40 LFWA+ attributes, of which gender
    ("Male") is just one, easy to lose among 39 unrelated look attributes
    like Big_Nose or Smiling -- this file exists so the results that
    actually matter for this project's fairness question don't require
    digging through all 40."""

    def _rows(df: pd.DataFrame, task_col: str, task_val: str, sensitive: str) -> list[dict]:
        if df.empty or task_col not in df.columns:
            return []
        mask = (df[task_col] == task_val) & (df["sensitive_attribute"] == sensitive)
        # for LFW, "task" is always the meaningless placeholder "x" left
        # over from reusing the generic binary-task helpers per attribute;
        # drop it there, keep it for FairFace/UTKFace where it's real.
        drop_cols = [task_col] + (["task"] if dataset == "lfw" and "task" in df.columns else [])
        return df[mask].drop(columns=drop_cols).to_dict(orient="records")

    highlights: dict = {"dataset": dataset}

    if dataset == "lfw":
        task_col = "attribute"
        gender_key = "Male"
        highlights["gender_task"] = "LFWA+ attribute: Male"
        highlights["gender_overall"] = classification.get(gender_key, {})
    else:
        task_col = "task"
        gender_key = "gender"
        highlights["gender_task"] = "gender (native classification head)"
        highlights["gender_overall"] = classification.get(gender_key, {})
        if "race" in classification:
            highlights["race_task"] = "race (native classification head)"
            highlights["race_overall"] = {k: v for k, v in classification["race"].items() if k != "per_class"}
            highlights["race_by_gender"] = _rows(subgroup_df, task_col, "race", "gender_label")
            highlights["race_fairness_by_gender"] = _rows(fairness_df, task_col, "race", "gender_label")

    highlights["gender_by_race"] = _rows(subgroup_df, task_col, gender_key, "race_label")
    highlights["gender_fairness_by_race"] = _rows(fairness_df, task_col, gender_key, "race_label")
    highlights["gender_by_ita"] = _rows(subgroup_df, task_col, gender_key, "ita_group")
    highlights["gender_fairness_by_ita"] = _rows(fairness_df, task_col, gender_key, "ita_group")
    highlights["gender_by_age"] = _rows(subgroup_df, task_col, gender_key, "age_group")
    highlights["gender_fairness_by_age"] = _rows(fairness_df, task_col, gender_key, "age_group")
    highlights["gender_by_fitzpatrick_approx"] = _rows(subgroup_df, task_col, gender_key, "fitzpatrick_approx")
    highlights["gender_fairness_by_fitzpatrick_approx"] = _rows(fairness_df, task_col, gender_key, "fitzpatrick_approx")
    highlights["gender_by_race_ita"] = _rows(subgroup_df, task_col, gender_key, "race_ita_label")
    highlights["gender_fairness_by_race_ita"] = _rows(fairness_df, task_col, gender_key, "race_ita_label")
    if dataset != "lfw" and "race" in classification:
        highlights["race_by_age"] = _rows(subgroup_df, task_col, "race", "age_group")
        highlights["race_fairness_by_age"] = _rows(fairness_df, task_col, "race", "age_group")
        highlights["race_by_fitzpatrick_approx"] = _rows(subgroup_df, task_col, "race", "fitzpatrick_approx")
        highlights["race_fairness_by_fitzpatrick_approx"] = _rows(fairness_df, task_col, "race", "fitzpatrick_approx")

    # True joint race x gender subgroups (e.g. "Black_Female" vs
    # "White_Male"), not just race-alone or gender-alone conditioning --
    # intersectional gaps can be large even when both marginal views look
    # fine (Buolamwini & Gebru 2018). Covers every available classification
    # task, not just gender, so a gap specific to e.g. the age_group head
    # for a particular intersection isn't missed either.
    intersection_tasks = ([gender_key, "race"] if dataset != "lfw" and "race" in classification else [gender_key])
    highlights["intersectional_tasks"] = intersection_tasks
    for task_val in intersection_tasks:
        key_prefix = "gender" if task_val == gender_key else task_val
        rows = _rows(subgroup_df, task_col, task_val, "race_gender_label")
        fair_rows = _rows(fairness_df, task_col, task_val, "race_gender_label")
        highlights[f"{key_prefix}_by_race_gender_intersection"] = rows
        highlights[f"{key_prefix}_fairness_by_race_gender_intersection"] = fair_rows

        # Worst-performing intersectional cell by accuracy, among cells with
        # at least 10 examples (smaller than that, "worst" is mostly noise,
        # not a real gap worth acting on).
        reliable = [r for r in rows if r.get("group") != "unknown" and r.get("n", 0) >= 10 and "accuracy" in r]
        if reliable:
            worst = min(reliable, key=lambda r: r["accuracy"])
            best = max(reliable, key=lambda r: r["accuracy"])
            highlights[f"{key_prefix}_worst_intersection"] = {
                "group": worst["group"], "n": worst["n"], "accuracy": worst["accuracy"],
            }
            highlights[f"{key_prefix}_best_intersection"] = {
                "group": best["group"], "n": best["n"], "accuracy": best["accuracy"],
            }
            highlights[f"{key_prefix}_intersection_accuracy_gap"] = best["accuracy"] - worst["accuracy"]

    (eval_dir / "race_gender_highlights.json").write_text(json.dumps(highlights, indent=2, default=str))
    return highlights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface", "lfw"])
    parser.add_argument(
        "--eval-dir", type=Path, default=None,
        help="Directory holding test_predictions.parquet (from evaluate.py's matching --out-dir). "
             "Defaults to experiments/<dataset>/test_eval/ (M1's location).",
    )
    args = parser.parse_args()
    summary = compute_all_metrics(args.dataset, eval_dir=args.eval_dir)
    print(json.dumps({"n_subgroup_rows": summary["n_subgroup_rows"], "n_fairness_rows": summary["n_fairness_rows"]}, indent=2))


if __name__ == "__main__":
    main()
