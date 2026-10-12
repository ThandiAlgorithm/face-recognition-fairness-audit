"""Links classifier errors to image-acquisition conditions (brightness,
contrast, sharpness, pose, face size, ITA/L*) rather than only to
demographic group -- the piece the fairness-gap tables can't answer on
their own. A race gap could mean "the model is biased by race" or it
could mean "this dataset's darker-skinned subjects happen to have lower
average sharpness/face-area," which is a measurement/acquisition problem,
not (only) a modelling one. Buolamwini & Gebru's original Gender Shades
critique is exactly this kind of confound, which is why the project's own
VIS/MEA audit exists -- this script is what actually joins that audit to
the classifiers' errors instead of leaving them as two separate stories.

Never re-runs a model and never re-derives VIS/MEA's own measurements --
reads test_predictions.parquet (evaluate.py) and the already-computed
VIS_image_measurements.parquet / MEA_image_measurements.csv, joined on
image_id (the content hash in image_path's filename stem, e.g.
"data/derived/aligned/lfw/lfw-<hash>.png" -> "lfw-<hash>", which matches
VIS/MEA's own image_id column exactly -- verified by a full-overlap check,
not assumed).

Two analyses per task, both standard and appropriate for a binary
(error/correct) outcome against continuous covariates:
  1. Univariate: Mann-Whitney U, error rows vs correct rows, per covariate
     -- nonparametric, no normality assumption, robust to the long tails
     these quality measures actually have.
  2. Multivariate: logistic regression of is_error on demographic group
     dummies alone, then again with the image-quality covariates added.
     If a demographic coefficient's p-value survives (stays significant)
     once quality covariates are controlled for, that's evidence of a
     real demographic association beyond acquisition conditions; if it
     drops out, the apparent gap was better explained by image quality.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
VIS_PATH = ROOT / "outputs/data_prep/visual_audit/tables/VIS_image_measurements.parquet"
MEA_PATH = ROOT / "outputs/phase1/measurement_audit/tables/MEA_image_measurements.csv"

COVARIATES = [
    "brightness", "contrast", "sharpness", "laplacian_variance", "tenengrad_mean",
    "face_area_ratio", "yaw", "pitch", "roll", "noise_residual_std",
    "skin_ITA_deg", "skin_L_star",
]
DATASET_TASKS = {
    "lfw": [],  # handled separately: 40 binary attributes via attributes_true/pred
    "utkface": ["race", "gender", "age_group"],
    "fairface": ["race", "gender", "age_group"],
}


def _image_id_from_path(image_path: str) -> str:
    return Path(image_path).stem


def load_quality_table() -> tuple[pd.DataFrame, pd.Series]:
    """Returns (quality_df, path_to_id) -- path_to_id maps VIS/MEA's own
    original_path (e.g. "FairFace/train/10.jpg") to their image_id, for
    datasets whose test_predictions.parquet stores the pre-alignment
    original path rather than the aligned-hash path. LFW/UTKFace's
    image_path IS the aligned-hash path (its stem already equals
    image_id directly, verified by a full-overlap check); FairFace's
    image_path is the raw original path instead, so it needs this
    second lookup -- both conventions live in the same project, not a
    bug in either, just two different `evaluate.py` eras."""
    vis = pd.read_parquet(VIS_PATH)
    mea = pd.read_csv(MEA_PATH)
    vis_cols = ["image_id"] + [c for c in COVARIATES if c in vis.columns]
    mea_cols = ["image_id"] + [c for c in COVARIATES if c in mea.columns and c not in vis.columns]
    merged = vis[vis_cols].merge(mea[mea_cols], on="image_id", how="left")
    path_to_id = vis.set_index("original_path")["image_id"]
    return merged, path_to_id


def resolve_image_ids(preds: pd.DataFrame, quality_ids: set[str], path_to_id: pd.Series) -> pd.Series:
    stem_ids = preds["image_path"].apply(_image_id_from_path)
    resolved = stem_ids.where(stem_ids.isin(quality_ids))
    unresolved = resolved.isna()
    if unresolved.any():
        normalized = preds.loc[unresolved, "image_path"].str.replace(r"^data/", "", regex=True)
        resolved.loc[unresolved] = normalized.map(path_to_id)
    return resolved


def univariate_tests(df: pd.DataFrame, is_error: pd.Series, task: str, group_label: str) -> list[dict]:
    rows = []
    for cov in COVARIATES:
        if cov not in df.columns:
            continue
        vals = pd.to_numeric(df[cov], errors="coerce")
        valid = vals.notna()
        err_vals = vals[valid & is_error]
        ok_vals = vals[valid & ~is_error]
        if len(err_vals) < 5 or len(ok_vals) < 5:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p = mannwhitneyu(err_vals, ok_vals, alternative="two-sided")
        rows.append({
            "task": task, "group": group_label, "covariate": cov,
            "n_error": len(err_vals), "n_correct": len(ok_vals),
            "median_error": float(err_vals.median()), "median_correct": float(ok_vals.median()),
            "u_statistic": float(stat), "p_value": float(p), "significant_0.05": p < 0.05,
        })
    return rows


def demographic_control_test(df: pd.DataFrame, is_error: pd.Series, demographic_col: str, task: str) -> list[dict]:
    """Logistic regression: does a demographic group's association with
    error survive once image-quality covariates are added?"""
    # Require >=50% non-null coverage -- yaw/pitch/roll are entirely null
    # for some datasets (pose wasn't estimated there), and a single
    # all-null covariate would otherwise wipe out every row via dropna.
    available_covs = [c for c in COVARIATES if c in df.columns and df[c].notna().mean() >= 0.5]
    if not available_covs:
        return [{"task": task, "demographic": demographic_col, "dummy": None,
                 "note": "no image-quality covariates available after the join -- "
                         "'survives_control' would be meaningless here, skipped"}]
    sub = df[[demographic_col] + available_covs].copy()
    sub["is_error"] = is_error.astype(int).values
    sub = sub.dropna(subset=available_covs)
    sub = sub[sub[demographic_col].notna() & (sub[demographic_col] != "unknown")]
    groups = sub[demographic_col].value_counts()
    groups = groups[groups >= 20].index.tolist()
    sub = sub[sub[demographic_col].isin(groups)]
    if sub[demographic_col].nunique() < 2 or len(sub) < 50:
        return []

    dummies = pd.get_dummies(sub[demographic_col], prefix=demographic_col, drop_first=True)
    rows = []

    X_bare = sm.add_constant(dummies.astype(float))
    try:
        model_bare = sm.Logit(sub["is_error"], X_bare).fit(disp=0)
    except Exception as e:
        return [{"task": task, "demographic": demographic_col, "model": "error", "note": str(e)}]

    X_full = sm.add_constant(pd.concat([dummies.astype(float), sub[available_covs].astype(float)], axis=1))
    try:
        model_full = sm.Logit(sub["is_error"], X_full).fit(disp=0)
    except Exception as e:
        model_full = None

    for col in dummies.columns:
        bare_p = float(model_bare.pvalues.get(col, np.nan))
        full_p = float(model_full.pvalues.get(col, np.nan)) if model_full is not None else np.nan
        rows.append({
            "task": task, "demographic": demographic_col, "dummy": col,
            "n": len(sub),
            "bare_coef": float(model_bare.params.get(col, np.nan)), "bare_p": bare_p,
            "controlled_coef": float(model_full.params.get(col, np.nan)) if model_full is not None else np.nan,
            "controlled_p": full_p,
            "survives_control": bool(bare_p < 0.05 and full_p < 0.05) if not np.isnan(full_p) else None,
            "explained_by_quality": bool(bare_p < 0.05 and full_p >= 0.05) if not np.isnan(full_p) else None,
        })
    return rows


def run(dataset: str, method: str = "M1_baseline") -> None:
    if method == "M1_baseline":
        eval_dir = EXPERIMENTS_DIR / dataset / "test_eval"
    else:
        candidates = list(EXPERIMENTS_DIR.glob(f"{dataset}/{method}/seed_*/test_eval"))
        if not candidates:
            raise FileNotFoundError(f"no test_eval dir for {dataset}/{method}")
        eval_dir = candidates[0]

    preds = pd.read_parquet(eval_dir / "test_predictions.parquet")
    quality, path_to_id = load_quality_table()
    preds["image_id"] = resolve_image_ids(preds, set(quality["image_id"]), path_to_id)
    n_unresolved = int(preds["image_id"].isna().sum())
    df = preds.merge(quality, on="image_id", how="left")
    print(f"[{dataset}/{method}] {len(df)} rows after join; {n_unresolved} rows had no resolvable "
          f"image_id at all; missing VIS/MEA quality data for "
          f"{int(df['brightness'].isna().sum()) if 'brightness' in df.columns else 'N/A'} rows")

    uni_rows, ctrl_rows = [], []
    tasks = DATASET_TASKS.get(dataset, [])

    if dataset == "lfw":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
        from label_encoders import load_labels
        attrs = load_labels("lfw")["attributes"]
        y_true_all = np.array(df["attributes_true"].tolist())
        y_pred_all = np.array(df["attributes_pred"].tolist())
        for i, attr in enumerate(attrs):
            is_error = pd.Series(y_pred_all[:, i] != y_true_all[:, i], index=df.index)
            uni_rows.extend(univariate_tests(df, is_error, attr, "overall"))
            for demo_col in ["race_label", "gender_label", "age_group"]:
                if demo_col not in df.columns:
                    continue
                ctrl_rows.extend(demographic_control_test(df, is_error, demo_col, attr))
    else:
        for task in tasks:
            true_col, pred_col = f"{task}_true", f"{task}_pred"
            if true_col not in df.columns:
                continue
            valid = df[df[true_col] != -1]
            is_error = valid[pred_col] != valid[true_col]
            uni_rows.extend(univariate_tests(valid, is_error, task, "overall"))
            for demo_col in ["race_label", "gender_label", "age_group"]:
                if demo_col == task or demo_col not in valid.columns:
                    continue
                ctrl_rows.extend(demographic_control_test(valid, is_error, demo_col, task))

    uni_df = pd.DataFrame(uni_rows)
    ctrl_df = pd.DataFrame(ctrl_rows)
    uni_path = eval_dir / "error_quality_univariate.csv"
    ctrl_path = eval_dir / "error_quality_controlled.csv"
    uni_df.to_csv(uni_path, index=False)
    ctrl_df.to_csv(ctrl_path, index=False)
    print(f"[{dataset}/{method}] wrote {len(uni_df)} univariate rows -> {uni_path}")
    print(f"[{dataset}/{method}] wrote {len(ctrl_df)} demographic-control rows -> {ctrl_path}")
    if not ctrl_df.empty:
        survived = ctrl_df[ctrl_df["survives_control"] == True]
        explained = ctrl_df[ctrl_df["explained_by_quality"] == True]
        print(f"  {len(survived)} demographic associations SURVIVE quality control (real demographic effect)")
        print(f"  {len(explained)} demographic associations are EXPLAINED AWAY by quality covariates")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["lfw", "utkface", "fairface"])
    parser.add_argument("--method", default="M1_baseline")
    args = parser.parse_args()
    run(args.dataset, args.method)


if __name__ == "__main__":
    main()
