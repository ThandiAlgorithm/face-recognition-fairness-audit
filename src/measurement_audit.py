from __future__ import annotations

import copy
import math
from itertools import combinations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import rgb2lab

from src import phase1_categories as categories, phase1_config as config, phase1_measurements as measurements
from src import stats as S
from src.phase1_image_integrity import load_rgb_for_analysis
from src.phase1_common import (
    analysis_flags, as_bool, contact_sheet, load_audit_config, load_master, output_dirs,
    process_records, register_results, resolve_file, save_run_metadata, valid_pixel_mask,
    write_json, write_table,
)

MODULE = "measurement_audit"
MEA_METRICS = ["skin_L_star", "skin_a_star", "skin_b_star", "skin_ITA_deg", "skin_hue_deg", "skin_chroma"]
RECORD_COLS = ["image_id", "dataset", "aligned_path", "alignment_transform", "width_px", "height_px", "is_greyscale"]


def _lab(pixels_rgb: np.ndarray) -> np.ndarray:
    return rgb2lab((pixels_rgb.astype(np.float64) / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)


def _single_patch_cfg(pre: dict, name: str) -> dict:
    sub = copy.deepcopy(pre)
    sub["skin_measurement"]["roi_patches"] = {name: pre["skin_measurement"]["roi_patches"][name]}
    return sub


def measure_skin(record: dict, ctx: dict) -> dict:
    pre, cat = ctx["pre"], ctx["cat"]
    sm = pre["skin_measurement"]
    out = {"image_id": record["image_id"], "status": "ok", "error": None}
    if as_bool(record.get("is_greyscale")):
        out["status"] = "greyscale_no_chroma"
        return out
    path = resolve_file(record.get("aligned_path"))
    if path is None:
        out["status"] = "image_missing"
        return out
    rgb, _ = load_rgb_for_analysis(path)
    valid, mask_method = valid_pixel_mask(record, rgb.shape[:2])
    out["valid_mask_method"] = mask_method
    out["aligned_width_px"], out["aligned_height_px"] = int(rgb.shape[1]), int(rgb.shape[0])

    roi = measurements.extract_skin_roi(rgb, pre, valid)
    out["skin_roi_total_px"] = roi["total_roi_pixels"]
    out["skin_roi_valid_px"] = roi["valid_pixel_count"]
    out["skin_roi_valid_fraction"] = roi["valid_fraction"]
    min_px = int(sm["roi_min_valid_pixels"])
    if roi["valid_pixel_count"] < min_px:
        out["status"] = "invalid_skin_roi"
        return out

    lab = _lab(roi["pixels"])
    L, a, b = [float(v) for v in np.median(lab, axis=0)]
    guard = float(sm["ita_guard"]["b_star_min_abs"])
    ita, undefined = measurements.compute_ita(L, b, guard)
    out.update(skin_L_star=L, skin_a_star=a, skin_b_star=b, skin_ITA_deg=ita, ita_undefined=bool(undefined),
               skin_hue_deg=measurements.compute_hue(a, b), skin_chroma=float(math.hypot(a, b)))
    if undefined:
        out["status"] = "ita_undefined"
    out["ita_group"] = categories.assign_ita_group(ita, cat)
    out["ita_ordinal"] = categories.assign_ita_ordinal(ita, cat)
    out["skin_group_ita"] = categories.assign_ita_binary(ita, cat)

    Lm, _, bm = [float(v) for v in lab.mean(axis=0)]
    out["rel_ITA_mean_based"] = measurements.compute_ita(Lm, bm, guard)[0]

    patches = sm["roi_patches"]
    per_patch_min = max(1, math.ceil(min_px / max(1, len(patches))))
    for name in patches:
        r = measurements.extract_skin_roi(rgb, _single_patch_cfg(pre, name), valid)
        out[f"rel_px_{name}"] = r["valid_pixel_count"]
        if r["valid_pixel_count"] >= per_patch_min:
            pl = _lab(r["pixels"])
            out[f"rel_ITA_{name}"] = measurements.compute_ita(float(np.median(pl[:, 0])), float(np.median(pl[:, 2])), guard)[0]
        else:
            out[f"rel_ITA_{name}"] = None
    return out


def _hist(df: pd.DataFrame, metric: str, path, vline=None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    drawn = False
    for ds, g in df.groupby("dataset"):
        x = pd.to_numeric(g[metric], errors="coerce").dropna()
        if len(x):
            ax.hist(x, bins=60, alpha=0.45, density=True, label=f"{ds} (n={len(x):,})")
            drawn = True
    if not drawn:
        plt.close(fig)
        return
    if vline is not None:
        ax.axvline(vline, color="k", ls="--", lw=1, label=f"threshold {vline:g}")
    ax.set_xlabel(metric)
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _bland_altman(x, y, title, path) -> None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return
    x, y = x[ok], y[ok]
    m, d = (x + y) / 2, x - y
    bias, sd = d.mean(), d.std(ddof=1)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(m, d, s=3, alpha=0.2)
    for v in [bias, bias - 1.96 * sd, bias + 1.96 * sd]:
        ax.axhline(v, color="k", ls="--", lw=1)
    ax.set_xlabel("mean of the two ITA values (deg)")
    ax.set_ylabel("difference (deg)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _roi_overlay(pre: dict):
    patches = pre["skin_measurement"]["roi_patches"]

    def draw(img):
        h, w = img.shape[:2]
        for p in patches.values():
            x1, x2 = sorted(p["x"])
            y1, y2 = sorted(p["y"])
            cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 1)
        return img
    return draw


def run(config_path: str = "configs/phase1_audit.json", workers: int | None = None) -> pd.DataFrame:
    cfg, cfg_path = load_audit_config(config_path)
    master, master_path = load_master(cfg)
    dep_cfg = cfg.get("dependency_configs", {})
    pre = config.load_preprocessing(dep_cfg.get("preprocessing", "configs/phase1_preprocessing.yaml"))
    cat = config.load_categories(dep_cfg.get("categories", "configs/phase1_categories.yaml"))
    od = output_dirs(cfg, MODULE)
    scfg = cfg["stats"]
    boot = scfg["bootstrap"]
    min_n = int(scfg["min_n"]["image"])
    fam = scfg.get("bh_family", ["dataset", "group_variable"])
    workers = workers if workers is not None else int(cfg.get("workers", 1))

    # MEA has its own eligibility. If the project has not created that flag yet,
    # process every row that has an aligned crop path rather than inheriting VIS eligibility.
    elig_col = "eligible_measurement_audit"
    if elig_col in master.columns:
        eligible = master[elig_col].fillna(False).astype(bool)
    elif "aligned_path" in master.columns:
        eligible = master["aligned_path"].notna() & master["aligned_path"].astype(str).str.strip().ne("")
    else:
        eligible = pd.Series(False, index=master.index)
    cols = [c for c in RECORD_COLS if c in master.columns]
    records = master.loc[eligible, cols].to_dict("records")
    res = process_records(measure_skin, records, {"pre": pre, "cat": cat}, od["tables"] / "MEA_checkpoint.csv", workers)

    skipped = master.loc[~eligible, ["image_id"]].assign(status="not_eligible_measurement_audit")
    mea = pd.concat([res, skipped], ignore_index=True)
    carry = [c for c in ["dataset", "subject_id", "original_path", "aligned_path", "gender", "race", "age_band",
                         "exact_dup_group", "near_dup_group", "cross_dataset_dup", "L_star", "ITA"] if c in master.columns]
    mea = mea.merge(master[["image_id", *carry]], on="image_id", how="left", validate="one_to_one")
    mea = mea.rename(columns={"L_star": "phase0_L_star", "ITA": "phase0_ITA"})
    mea = pd.concat([mea, analysis_flags(mea, mea["status"].eq("ok"), cfg.get("dedup", {}))], axis=1)
    written = [write_table(mea, od["tables"] / "MEA_image_measurements.csv", parquet=True)]

    status = mea.groupby(["dataset", "status"], dropna=False).size().reset_index(name="n")
    written.append(write_table(status, od["tables"] / "MEA_status_counts.csv"))

    measured = mea[~mea["status"].eq("not_eligible_measurement_audit")].copy()
    measured["roi_failed"] = ~measured["status"].eq("ok")
    fail_rows, fail_tests = [], []
    for axis in [a for a in cfg["axes"]["within_dataset"] if a in measured.columns]:
        for ds, d in measured.groupby("dataset"):
            tab = pd.crosstab(S.clean_labels(d[axis]), d["roi_failed"])
            for g, row in tab.iterrows():
                n, k = int(row.sum()), int(row.get(True, 0))
                lo, hi = S.wilson_ci(k, n)
                fail_rows.append({"dataset": ds, "group_variable": axis, "group": g, "n": n, "n_failed": k,
                                  "failed_pct": 100 * k / n if n else np.nan, "ci_low": 100 * lo, "ci_high": 100 * hi,
                                  "thin_cell": n < min_n})
            known = tab.drop(index="unknown", errors="ignore")
            known = known[known.sum(axis=1) >= min_n]
            if len(known) >= 2:
                fail_tests.append({"dataset": ds, "group_variable": axis, **S.chi_square(known)})
    written.append(write_table(pd.DataFrame(fail_rows), od["tables"] / "MEA_roi_failure_by_group.csv"))
    written.append(write_table(S.add_bh(pd.DataFrame(fail_tests), ["group_variable"], alpha=scfg["bh_alpha"]),
                               od["tables"] / "MEA_roi_failure_tests.csv"))

    ana = mea[mea["in_analysis"]].copy()
    summ = []
    for ds, d in ana.groupby("dataset"):
        for m in MEA_METRICS:
            summ.append({"dataset": ds, "metric": m, **S.summarise(pd.to_numeric(d[m], errors="coerce").to_numpy())})
    written.append(write_table(pd.DataFrame(summ), od["tables"] / "MEA_summary_by_dataset.csv"))

    dist = pd.concat([S.proportion_table(ana, "dataset", c, min_n) for c in ["skin_group_ita", "ita_group"]], ignore_index=True)
    written.append(write_table(dist, od["tables"] / "MEA_skin_group_distribution.csv"))

    sums, omni, pair = [], [], []
    for m in ["skin_ITA_deg", "skin_L_star", "skin_hue_deg"]:
        for axis in [a for a in cfg["axes"]["within_dataset"] if a in ana.columns]:
            s, o, p = S.subgroup_analysis(ana, m, axis, "dataset", min_n, boot)
            sums.append(s), omni.append(o), pair.append(p)
    written.append(write_table(pd.concat(sums, ignore_index=True), od["tables"] / "MEA_subgroup_summary.csv"))
    written.append(write_table(S.add_bh(pd.concat(omni, ignore_index=True), fam, alpha=scfg["bh_alpha"]), od["tables"] / "MEA_subgroup_omnibus.csv"))
    written.append(write_table(S.add_bh(pd.concat(pair, ignore_index=True), fam, alpha=scfg["bh_alpha"]), od["tables"] / "MEA_subgroup_pairwise.csv"))

    patch_names = list(pre["skin_measurement"]["roi_patches"])
    threshold = float(cat["skin_binary"]["ita"]["threshold_degrees"])
    order = [g["name"] for g in sorted(cat["ita_groups"], key=lambda g: g["ordinal_rank"])]
    comparisons = [(f"rel_ITA_{a}", f"rel_ITA_{b}", f"patch {a} vs patch {b}") for a, b in combinations(patch_names, 2)]
    comparisons.append(("skin_ITA_deg", "rel_ITA_mean_based", "median-based vs mean-based ITA"))
    if "phase0_ITA" in ana.columns and pd.to_numeric(ana["phase0_ITA"], errors="coerce").notna().any():
        comparisons.append(("skin_ITA_deg", "phase0_ITA", "MEA ITA vs Phase 0 master ITA"))
    rel_rows = []
    for ds, d in ana.groupby("dataset"):
        for x, y, label in comparisons:
            if x not in d.columns or y not in d.columns:
                continue
            xv = pd.to_numeric(d[x], errors="coerce").to_numpy()
            yv = pd.to_numeric(d[y], errors="coerce").to_numpy()
            ok = np.isfinite(xv) & np.isfinite(yv)
            cont = S.agreement_continuous(xv, yv, boot)
            kap = S.cohens_kappa(np.where(xv[ok] > threshold, "lighter", "darker"),
                                 np.where(yv[ok] > threshold, "lighter", "darker"), ["darker", "lighter"])
            kap6 = S.cohens_kappa([categories.assign_ita_group(v, cat) for v in xv[ok]],
                                  [categories.assign_ita_group(v, cat) for v in yv[ok]], order, weights="quadratic")
            rel_rows.append({"dataset": ds, "comparison": label, "x": x, "y": y, **cont,
                             "binary_kappa": kap["kappa"], "binary_percent_agreement": kap["percent_agreement"],
                             "ita_group_weighted_kappa": kap6["kappa"], "ita_group_percent_agreement": kap6["percent_agreement"]})
            _bland_altman(xv, yv, f"{ds}: {label}", od["figures"] / f"MEA_BA_{ds}_{x}_vs_{y}.png")
    written.append(write_table(pd.DataFrame(rel_rows), od["tables"] / "MEA_reliability_agreement.csv"))

    margin = float(cfg["mea"].get("boundary_margin_deg", 5.0))
    patch_cols = [f"rel_ITA_{p}" for p in patch_names]
    bnd = []
    for ds, d in ana.groupby("dataset"):
        ita = pd.to_numeric(d["skin_ITA_deg"], errors="coerce")
        near = (ita - threshold).abs() <= margin
        pv = d[patch_cols].apply(pd.to_numeric, errors="coerce")
        all_ok = pv.notna().all(axis=1)
        flips = (pv[all_ok] > threshold).nunique(axis=1).gt(1)
        lo, hi = S.wilson_ci(int(near.sum()), int(ita.notna().sum()))
        bnd.append({"dataset": ds, "threshold_deg": threshold, "margin_deg": margin, "n": int(ita.notna().sum()),
                    "n_within_margin": int(near.sum()), "pct_within_margin": 100 * near.mean(),
                    "ci_low": 100 * lo, "ci_high": 100 * hi, "n_all_patches_valid": int(all_ok.sum()),
                    "pct_binary_group_changes_across_patches": 100 * flips.mean() if len(flips) else np.nan})
    written.append(write_table(pd.DataFrame(bnd), od["tables"] / "MEA_boundary_sensitivity.csv"))

    _hist(ana, "skin_ITA_deg", od["figures"] / "MEA_hist_ITA.png", vline=threshold)
    for m in ["skin_L_star", "skin_hue_deg", "skin_chroma"]:
        _hist(ana, m, od["figures"] / f"MEA_hist_{m}.png")

    rng = np.random.default_rng(int(boot.get("seed", 26041)))
    qa_items = []
    for (ds, grp), d in ana.groupby(["dataset", "ita_group"]):
        for _, r in d.iloc[rng.permutation(len(d))[:3]].iterrows():
            p = resolve_file(r["aligned_path"])
            if p is not None:
                qa_items.append((p, f"{ds} {grp} {r['skin_ITA_deg']:.0f}", _roi_overlay(pre)))
    contact_sheet(qa_items, od["qa"] / "MEA_roi_overlay_sample.png")

    counts = {str(k): int(v) for k, v in mea["status"].value_counts().items()}
    lines = [
        "# Measurement validity limitations (auto-generated draft, edit wording before use)", "",
        f"Skin tone was measured once per aligned, non-augmented crop with the frozen ROI in preprocessing.yaml. Status counts: {counts}.",
        "",
        "1. There is no ground-truth skin-tone oracle in LFW, FairFace or UTKFace. The reliability tables report agreement "
        "between measurement variants (patches, median vs mean, Phase 0 vs MEA). Agreement shows the measurement is stable. "
        "It does not show the measurement is accurate.",
        "2. ITA depends on scene lighting, white balance and camera processing, not only on skin. Check the VIS brightness "
        "results before reading group differences as skin differences.",
        "3. ROI failure is not random if MEA_roi_failure_tests.csv shows differences by group. Invalid ROI, greyscale and "
        "undefined ITA images are flagged, not deleted.",
        f"4. Within {margin:g} degrees of the {threshold:g} degree threshold, group membership is unstable. See MEA_boundary_sensitivity.csv.",
        "5. FairFace images are publisher crops with padding. The same fixed ROI covers a different face area than on "
        "project-aligned LFW and UTKFace crops.",
        "6. Albinism and other visible phenotypes are not inferred from real photographs. Scars, acne, facial difference and "
        "disability are not labelled. No manual review of real images was performed.",
    ]
    lim = od["root"] / "MEA_limitations_draft.md"
    lim.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(lim)

    method = {
        "manual_review": False,
        "roi": "frozen preprocessing.yaml skin_measurement.roi_patches on the aligned crop, same for all datasets",
        "pixel_filter": "preprocessing.yaml roi_skin_filter YCrCb Cb/Cr ranges, restricted to the warped source-extent mask",
        "image_loading": "src.image_integrity.load_rgb_for_analysis (EXIF applied, ICC converted to sRGB)",
        "colour_space": "scikit-image rgb2lab, sRGB D65; per-channel median of valid ROI pixels, computed once",
        "ITA": "degrees(atan((L* - 50) / b*)), range -90 to 90; null and flagged when |b*| < ita_guard.b_star_min_abs",
        "hue": "degrees(atan2(b*, a*)) mod 360, range 0 to 360; not independent of ITA because both use b*",
        "groups": "categories.yaml ita_groups and skin_binary threshold, applied after the continuous value",
        "greyscale": "not measured, status greyscale_no_chroma",
        "reliability": "per-patch ITA, mean-based ITA and Phase 0 ITA agreement; Lin CCC, Bland-Altman, kappa",
        "fitzpatrick_classifier": "not used",
    }
    write_json(method, od["metadata"] / "MEA_method.json")
    write_json(S.EFFECT_SIZE_MAPPING, od["metadata"] / "effect_size_mapping.json")
    save_run_metadata(MODULE, cfg, cfg_path, master_path, od, {"status_counts": counts, "workers": workers})
    register_results(cfg, cfg_path, MODULE, written, "src/measurement_audit.py")
    return mea
