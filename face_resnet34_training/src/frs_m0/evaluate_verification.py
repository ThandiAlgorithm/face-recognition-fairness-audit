"""M0 spec sections 23-34: locked test-set verification evaluation.
Threshold comes from threshold.json (selected on validation ONLY, see
select_threshold.py) -- never re-chosen here. This script runs once
against the frozen test pairs and saves everything downstream fairness
analysis needs, so it never has to re-run the model either.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import cosine_similarity_matrix_pairs, compute_roc, eer_from_roc, tar_at_fmr, bootstrap_ci

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "M0_LFW"
N_BOOTSTRAP = 1000
MIN_CELL_N = 10  # below this, flag as low statistical support rather than hide


def _group_metrics(pairs: pd.DataFrame, group_genuine_mask, group_impostor_mask, label: str) -> dict:
    genuine = pairs[group_genuine_mask]
    impostor = pairs[group_impostor_mask]
    n_gen, n_imp = len(genuine), len(impostor)
    row = {"group": label, "n_genuine": n_gen, "n_impostor": n_imp, "low_support": (n_gen < MIN_CELL_N or n_imp < MIN_CELL_N)}
    if n_gen == 0 or n_imp == 0:
        row.update({"fmr": None, "fnmr": None, "tar": None, "roc_auc": None})
        return row

    sub = pd.concat([genuine, impostor])
    scores = sub["similarity_score"].to_numpy()
    labels = sub["same_identity"].to_numpy()
    threshold = pairs.attrs["threshold"]

    fmr = float((impostor["similarity_score"] >= threshold).mean())
    fnmr = float((genuine["similarity_score"] < threshold).mean())
    tar = 1.0 - fnmr
    try:
        auc = float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else None
    except ValueError:
        auc = None
    row.update({"fmr": fmr, "fnmr": fnmr, "tar": tar, "roc_auc": auc})

    # Bootstrap CIs resample IDENTITIES, not raw pairs (pairs from the same
    # identity aren't independent -- spec section 31).
    gen_ids = genuine["identity_a"].to_numpy()
    imp_ids = pd.concat([impostor["identity_a"], impostor["identity_b"]]).unique()

    def fmr_fn(idx):
        chosen = set(np.asarray(imp_ids)[idx % len(imp_ids)])
        mask = impostor["identity_a"].isin(chosen) | impostor["identity_b"].isin(chosen)
        sel = impostor[mask]
        return float((sel["similarity_score"] >= threshold).mean()) if len(sel) else np.nan

    def fnmr_fn(idx):
        chosen = set(np.asarray(gen_ids)[idx % len(gen_ids)]) if len(gen_ids) else set()
        sel = genuine[genuine["identity_a"].isin(chosen)]
        return float((sel["similarity_score"] < threshold).mean()) if len(sel) else np.nan

    if len(gen_ids) >= 2:
        _, fnmr_lo, fnmr_hi = bootstrap_ci(fnmr_fn, len(gen_ids), n_boot=N_BOOTSTRAP)
        row["fnmr_ci_lo"], row["fnmr_ci_hi"] = fnmr_lo, fnmr_hi
    if len(imp_ids) >= 2:
        _, fmr_lo, fmr_hi = bootstrap_ci(fmr_fn, len(imp_ids), n_boot=N_BOOTSTRAP)
        row["fmr_ci_lo"], row["fmr_ci_hi"] = fmr_lo, fmr_hi
    return row


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir

    threshold_info = json.loads((out_dir / "threshold.json").read_text())
    threshold = threshold_info["threshold"]

    embs = np.load(out_dir / "embeddings" / "test_embeddings.npy")
    meta = pd.read_csv(out_dir / "embeddings" / "test_metadata.csv")
    emb_by_id = {iid: e for iid, e in zip(meta["image_id"], embs)}

    pairs = pd.read_csv(out_dir / "pairs" / "test_pairs.csv")
    a = np.stack([emb_by_id[i] for i in pairs["image_id_a"]])
    b = np.stack([emb_by_id[i] for i in pairs["image_id_b"]])
    pairs["similarity_score"] = cosine_similarity_matrix_pairs(a, b)
    pairs["global_threshold"] = threshold
    pairs["predicted_match"] = (pairs["similarity_score"] >= threshold).astype(int)
    pairs["correct"] = (pairs["predicted_match"] == pairs["same_identity"]).astype(int)
    pairs.attrs["threshold"] = threshold
    pairs.to_csv(out_dir / "test_scores.csv", index=False)

    labels = pairs["same_identity"].to_numpy()
    scores = pairs["similarity_score"].to_numpy()
    thresholds_sweep, fmr_sweep, fnmr_sweep = compute_roc(scores, labels)
    eer, eer_threshold = eer_from_roc(thresholds_sweep, fmr_sweep, fnmr_sweep)
    auc = float(roc_auc_score(labels, scores))

    genuine_all = pairs[pairs["same_identity"] == 1]
    impostor_all = pairs[pairs["same_identity"] == 0]
    fmr_at_thresh = float((impostor_all["similarity_score"] >= threshold).mean())
    fnmr_at_thresh = float((genuine_all["similarity_score"] < threshold).mean())

    overall = {
        "n_pairs": len(pairs), "n_genuine": len(genuine_all), "n_impostor": len(impostor_all),
        "roc_auc": auc, "eer": eer, "eer_threshold": eer_threshold,
        "global_threshold_used": threshold, "threshold_selected_on": threshold_info["selected_on"],
        "fmr_at_global_threshold": fmr_at_thresh, "fnmr_at_global_threshold": fnmr_at_thresh,
        "tar_at_global_threshold": 1.0 - fnmr_at_thresh,
        "accuracy_at_global_threshold": float((pairs["correct"]).mean()),
        "tar_at_fmr_0.01": tar_at_fmr(thresholds_sweep, fmr_sweep, fnmr_sweep, 0.01),
        "tar_at_fmr_0.001": tar_at_fmr(thresholds_sweep, fmr_sweep, fnmr_sweep, 0.001),
    }
    (out_dir / "overall_metrics.json").write_text(json.dumps(overall, indent=2))
    print(json.dumps(overall, indent=2))

    # --- Subgroup analysis: race, gender, age, skin-tone. Primary FMR uses
    # SAME-group impostor pairs (spec section 25); genuine pairs are always
    # within one identity so they're trivially "same group" already. ---
    subgroup_rows = []
    for axis, col_a, col_b, same_col in [
        ("race", "race_label_a", "race_label_b", "same_race_label"),
        ("gender", "gender_label_a", "gender_label_b", "same_gender_label"),
        ("age_group", "age_group_a", "age_group_b", "same_age_group"),
        ("fitzpatrick_approx", "fitzpatrick_approx_a", "fitzpatrick_approx_b", "same_fitzpatrick_approx"),
        ("ita_group", "ita_group_a", "ita_group_b", "same_ita_group"),
    ]:
        for group_val in sorted(set(pairs[col_a].dropna().unique()) | set(pairs[col_b].dropna().unique())):
            if group_val in ("unknown",):
                continue
            gen_mask = (pairs["same_identity"] == 1) & (pairs[col_a] == group_val)
            imp_mask = (pairs["same_identity"] == 0) & (pairs[same_col] == 1) & (pairs[col_a] == group_val)
            row = _group_metrics(pairs, gen_mask, imp_mask, group_val)
            row["axis"] = axis
            subgroup_rows.append(row)
    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(out_dir / "subgroup_metrics.csv", index=False)

    # --- Intersectional: race x gender (the representative intersection --
    # LFW's test set is too small to support finer 3-way/4-way cells with
    # any statistical support; those cells would all be low_support anyway). ---
    pairs["race_gender_a"] = pairs["race_label_a"].astype(str) + "_" + pairs["gender_label_a"].astype(str)
    inter_rows = []
    for group_val in sorted(pairs["race_gender_a"].unique()):
        if "unknown" in group_val or "nan" in group_val:
            continue
        gen_mask = (pairs["same_identity"] == 1) & (pairs["race_gender_a"] == group_val)
        imp_mask = (pairs["same_identity"] == 0) & (pairs["same_race_label"] == 1) & (pairs["same_gender_label"] == 1) & (pairs["race_gender_a"] == group_val)
        row = _group_metrics(pairs, gen_mask, imp_mask, group_val)
        row["axis"] = "race_x_gender"
        inter_rows.append(row)
    inter_df = pd.DataFrame(inter_rows)
    inter_df.to_csv(out_dir / "intersection_metrics.csv", index=False)

    # --- Fairness gap summary (spec section 32) ---
    reliable = subgroup_df[(~subgroup_df["low_support"]) & subgroup_df["fmr"].notna()]
    fairness = {}
    for axis in reliable["axis"].unique():
        a = reliable[reliable["axis"] == axis]
        if len(a) < 2:
            continue
        fairness[axis] = {
            "fnmr_gap": float(a["fnmr"].max() - a["fnmr"].min()),
            "fmr_gap": float(a["fmr"].max() - a["fmr"].min()),
            "worst_group_tar": float(a["tar"].min()),
            "worst_group": a.loc[a["tar"].idxmin(), "group"],
            "best_group_tar": float(a["tar"].max()),
            "best_group": a.loc[a["tar"].idxmax(), "group"],
            "equality_of_opportunity_gap": float(a["tar"].max() - a["tar"].min()),
        }
    (out_dir / "fairness_metrics.json").write_text(json.dumps(fairness, indent=2, default=str))
    print(json.dumps(fairness, indent=2, default=str))


if __name__ == "__main__":
    main()
