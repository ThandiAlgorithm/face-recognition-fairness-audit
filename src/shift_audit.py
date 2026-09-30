from __future__ import annotations

from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import stats as S
from src.phase1_common import (
    load_audit_config, load_master, module_table, output_dirs, register_results,
    save_run_metadata, write_json, write_table,
)

MODULE = "shift_audit"


def _load_joined(cfg: dict) -> pd.DataFrame:
    mea_p = module_table(cfg, "measurement_audit", "MEA_image_measurements.csv")
    vis_p = module_table(cfg, "visual_audit", "VIS_image_measurements.csv")
    if not mea_p.exists() or not vis_p.exists():
        raise FileNotFoundError("Run MEA and VIS before SHIFT.")
    mea = pd.read_csv(mea_p, low_memory=False)
    vis = pd.read_csv(vis_p, low_memory=False)
    wanted = list(dict.fromkeys(cfg["shift"]["continuous_axes"] + cfg["shift"]["effective_balance_dimensions"]))
    keep_vis = ["image_id", "status", "in_analysis", *[c for c in wanted if c in vis.columns and c not in mea.columns]]
    vis = vis[keep_vis].rename(columns={"status": "vis_status", "in_analysis": "vis_in_analysis"})
    df = mea.merge(vis, on="image_id", how="left", validate="one_to_one").rename(columns={"in_analysis": "mea_in_analysis"})
    cross = df["cross_dataset_dup"].fillna(False).astype(str).str.lower().isin(["true", "1"]) if "cross_dataset_dup" in df.columns else False
    df["shift_in_analysis"] = df["mea_in_analysis"].fillna(False).astype(bool) & ~cross
    df["shift_exclusion"] = np.where(df["shift_in_analysis"], None,
                                     np.where(df["mea_in_analysis"].fillna(False).astype(bool), "cross_dataset_duplicate", "not_in_mea_analysis"))
    return df


def freeze_axes(cfg: dict, od: dict) -> dict:
    sc = cfg["shift"]
    banned = set(sc.get("within_dataset_only", []))
    clash = banned & set(sc["categorical_axes"] + sc["continuous_axes"])
    if clash:
        raise ValueError(f"Within-dataset-only variables listed as cross-dataset axes: {sorted(clash)}")
    frozen = {
        "categorical_axes": sc["categorical_axes"],
        "continuous_axes": sc["continuous_axes"],
        "within_dataset_only": sorted(banned),
        "gender_rule": sc.get("gender_rule"),
        "age_rule": sc.get("age_rule"),
        "skin_rule": "ITA, L*, hue and skin groups use one pixel pipeline (MEA) for every dataset",
        "visual_rule": "visual measures use one pipeline (VIS) for every dataset",
    }
    write_json(frozen, od["metadata"] / "SHIFT_frozen_axes.json")
    return frozen


def categorical_shift(df: pd.DataFrame, axes: list[str], min_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    props, tests = [], []
    for axis in axes:
        if axis not in df.columns:
            continue
        props.append(S.proportion_table(df, "dataset", axis, min_n))
        lab = S.clean_labels(df[axis])
        known = df[lab != "unknown"]
        tab = pd.crosstab(known["dataset"], S.clean_labels(known[axis]))
        if tab.shape[0] >= 2:
            tests.append({"variable": axis, "comparison": "all datasets", **S.chi_square(tab)})
        for a, b in combinations(tab.index, 2):
            tests.append({"variable": axis, "comparison": f"{a} vs {b}", **S.chi_square(tab.loc[[a, b]])})
        missing = df.groupby("dataset")[axis].apply(lambda s: float((S.clean_labels(s) == "unknown").mean() * 100))
        for ds, pct in missing.items():
            props.append(pd.DataFrame([{"dataset": ds, "variable": axis, "category": "__pct_unknown__", "pct": pct}]))
    return (pd.concat(props, ignore_index=True) if props else pd.DataFrame(), pd.DataFrame(tests))


def continuous_shift(df: pd.DataFrame, axes: list[str], min_n: int, boot: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    summ, tests = [], []
    for m in axes:
        if m not in df.columns:
            continue
        arrays = {}
        for ds, d in df.groupby("dataset"):
            x = pd.to_numeric(d[m], errors="coerce").to_numpy()
            summ.append({"dataset": ds, "metric": m, **S.summarise(x)})
            x = x[np.isfinite(x)]
            if len(x) >= min_n:
                arrays[ds] = x
        for a, b in combinations(sorted(arrays), 2):
            tests.append({"metric": m, "dataset_a": a, "dataset_b": b, "n_a": len(arrays[a]), "n_b": len(arrays[b]),
                          **S.compare_two(arrays[a], arrays[b], boot), **S.distribution_distance(arrays[a], arrays[b])})
    return pd.DataFrame(summ), pd.DataFrame(tests)


def nominal_balance(df: pd.DataFrame, cfg: dict, min_n: int) -> pd.DataFrame:
    rows = []
    for ds, d in df.groupby("dataset"):
        for axis in cfg["shift"]["nominal_balance_axes"]:
            if axis not in d.columns:
                continue
            lab = S.clean_labels(d[axis])
            counts = lab[lab != "unknown"].value_counts()
            if counts.empty:
                rows.append({"dataset": ds, "axis": axis, "n_known": 0, "note": "no labels in this dataset"})
                continue
            rows.append({"dataset": ds, "axis": axis, "n_known": int(counts.sum()), "n_categories": int(len(counts)),
                         "largest_category": counts.index[0], "largest_pct": 100 * counts.iloc[0] / counts.sum(),
                         "smallest_category": counts.index[-1], "smallest_pct": 100 * counts.iloc[-1] / counts.sum(),
                         "max_min_ratio": float(counts.iloc[0] / counts.iloc[-1]),
                         "normalized_entropy": S.normalized_entropy(counts.to_numpy()),
                         "n_thin_cells": int((counts < min_n).sum())})
        for a, b in cfg["shift"].get("intersections", []):
            if a in d.columns and b in d.columns:
                la, lb = S.clean_labels(d[a]), S.clean_labels(d[b])
                ok = (la != "unknown") & (lb != "unknown")
                counts = (la[ok] + " x " + lb[ok]).value_counts()
                if len(counts):
                    rows.append({"dataset": ds, "axis": f"{a} x {b}", "n_known": int(counts.sum()), "n_categories": int(len(counts)),
                                 "largest_category": counts.index[0], "largest_pct": 100 * counts.iloc[0] / counts.sum(),
                                 "smallest_category": counts.index[-1], "smallest_pct": 100 * counts.iloc[-1] / counts.sum(),
                                 "max_min_ratio": float(counts.iloc[0] / counts.iloc[-1]),
                                 "normalized_entropy": S.normalized_entropy(counts.to_numpy()),
                                 "n_thin_cells": int((counts < min_n).sum())})
    return pd.DataFrame(rows)


def effective_balance(df: pd.DataFrame, cfg: dict, min_n: int, boot: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sc = cfg["shift"]
    group_axes = [a for a in sc["effective_balance_groups"] if a in df.columns]
    summaries, tests = [], []
    for m in [x for x in sc["effective_balance_dimensions"] if x in df.columns]:
        for axis in group_axes:
            s, o, p = S.subgroup_analysis(df, m, axis, "dataset", min_n, boot)
            summaries.append(s)
            if not o.empty:
                tests.append(o.assign(level="omnibus"))
            if not p.empty:
                tests.append(p.assign(level="pairwise"))
    return (pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(),
            pd.concat(tests, ignore_index=True) if tests else pd.DataFrame())


def identity_depth(df: pd.DataFrame, cfg: dict, min_n_identity: int) -> pd.DataFrame:
    rows = []
    for ds in cfg["shift"].get("identity_datasets", ["lfw"]):
        d = df[df["dataset"].eq(ds) & df["subject_id"].notna()] if "subject_id" in df.columns else pd.DataFrame()
        if d.empty:
            continue
        ident = d.groupby("subject_id").agg(n_images=("image_id", "size"), median_ita=("skin_ITA_deg", "median"))
        thr = float(cfg["shift"]["identity_skin_threshold_deg"])
        ident["identity_skin_group"] = np.where(ident["median_ita"] > thr, "lighter", "darker")
        for g, z in ident.groupby("identity_skin_group"):
            rows.append({"dataset": ds, "identity_skin_group": g, "n_identities": int(len(z)), "n_images": int(z["n_images"].sum()),
                         "median_images_per_identity": float(z["n_images"].median()),
                         "pct_identities_with_2plus_images": float(100 * (z["n_images"] >= 2).mean()),
                         "max_images_one_identity": int(z["n_images"].max()),
                         "thin_cell": len(z) < min_n_identity})
    return pd.DataFrame(rows)


def run(config_path: str = "configs/phase1_audit.json") -> pd.DataFrame:
    cfg, cfg_path = load_audit_config(config_path)
    _, master_path = load_master(cfg)
    od = output_dirs(cfg, MODULE)
    scfg = cfg["stats"]
    boot = scfg["bootstrap"]
    min_n = int(scfg["min_n"]["image"])
    frozen = freeze_axes(cfg, od)

    df = _load_joined(cfg)
    written = [write_table(df.groupby(["dataset", "shift_exclusion"], dropna=False).size().reset_index(name="n"),
                           od["tables"] / "SHIFT_analysis_set_counts.csv")]
    ana = df[df["shift_in_analysis"]].copy()

    props, cat_tests = categorical_shift(ana, frozen["categorical_axes"], min_n)
    written.append(write_table(props, od["tables"] / "SHIFT_categorical_distributions.csv"))
    written.append(write_table(S.add_bh(cat_tests, ["variable"], alpha=scfg["bh_alpha"]), od["tables"] / "SHIFT_categorical_tests.csv"))

    cont_summ, cont_tests = continuous_shift(ana, frozen["continuous_axes"], min_n, boot)
    written.append(write_table(cont_summ, od["tables"] / "SHIFT_continuous_summary.csv"))
    written.append(write_table(S.add_bh(cont_tests, ["metric"], alpha=scfg["bh_alpha"]), od["tables"] / "SHIFT_continuous_tests.csv"))

    written.append(write_table(nominal_balance(ana, cfg, min_n), od["tables"] / "SHIFT_nominal_balance.csv"))
    eff_s, eff_t = effective_balance(ana, cfg, min_n, boot)
    written.append(write_table(eff_s, od["tables"] / "SHIFT_effective_balance_summary.csv"))
    written.append(write_table(S.add_bh(eff_t, ["dataset", "group_variable", "level"], alpha=scfg["bh_alpha"]), od["tables"] / "SHIFT_effective_balance_tests.csv"))

    within = []
    if "race" in ana.columns:
        s, o, p = S.subgroup_analysis(ana, "skin_ITA_deg", "race", "dataset", min_n, boot)
        within = [s.assign(note="native race labels, within-dataset only")]
    written.append(write_table(pd.concat(within, ignore_index=True) if within else pd.DataFrame(), od["tables"] / "SHIFT_ita_within_native_race.csv"))
    written.append(write_table(identity_depth(ana, cfg, int(scfg["min_n"]["identity"])), od["tables"] / "SHIFT_identity_depth.csv"))

    if "skin_group_ita" in ana.columns:
        tab = pd.crosstab(ana["dataset"], S.clean_labels(ana["skin_group_ita"]), normalize="index") * 100
        ax = tab.plot(kind="bar", stacked=True, figsize=(6, 4))
        ax.set_ylabel("% of analysed images")
        ax.figure.tight_layout()
        ax.figure.savefig(od["figures"] / "SHIFT_skin_group_by_dataset.png", dpi=160)
        plt.close(ax.figure)

    n_by = ana.groupby("dataset").size().to_dict()
    lines = [
        "# Dataset-shift interpretation limits (auto-generated draft, edit wording before use)", "",
        f"Images in the shift analysis set: {n_by}.", "",
        f"1. Only these axes are compared across datasets: {frozen['categorical_axes'] + frozen['continuous_axes']}. "
        f"Native race labels ({frozen['within_dataset_only']}) use different schemes in each dataset and are only analysed within a dataset.",
        f"2. Gender: {frozen['gender_rule']}",
        f"3. Age: {frozen['age_rule']}",
        "4. Skin tone and visual measures come from one pixel pipeline, but the images differ in capture, cropping and compression. "
        "FairFace crops are publisher-made with padding, LFW and UTKFace crops are project-aligned. A difference in a measure can come "
        "from the image source, not only from the people shown.",
        "5. Nominal balance (label counts) and effective balance (image conditions per group) are reported separately. No composite "
        "fairness score is created.",
        "6. With large n, small differences reach significance. Read effect sizes (Cliff's delta, Cramér's V, Wasserstein distance) "
        "with their intervals, not p-values alone.",
    ]
    lim = od["root"] / "SHIFT_interpretation_limits_draft.md"
    lim.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(lim)

    save_run_metadata(MODULE, cfg, cfg_path, master_path, od, {"n_analysis_by_dataset": {str(k): int(v) for k, v in n_by.items()}})
    register_results(cfg, cfg_path, MODULE, written, "src/shift_audit.py")
    return ana
