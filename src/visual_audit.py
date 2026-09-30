from __future__ import annotations

import math

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import rgb2lab

from src import phase1_measurements as measurements
from src import stats as S
from src.phase1_image_integrity import load_rgb_for_analysis
from src.phase1_common import (
    analysis_flags, as_array, contact_sheet, erode, is_missing, load_audit_config, load_master,
    module_table, output_dirs, process_records, read_table, register_results, resolve_file,
    save_run_metadata, valid_pixel_mask, write_json, write_table,
)

MODULE = "visual_audit"
RECORD_COLS = ["image_id", "dataset", "aligned_path", "alignment_transform", "width_px", "height_px", "bbox", "landmarks"]
PIXEL_METRICS = [
    "brightness_L_mean", "brightness_L_median", "lstar_p5_p95_range", "contrast_L_sd", "illumination_asymmetry",
    "shadow_clip_fraction", "highlight_clip_fraction", "sharpness_lapvar", "sharpness_tenengrad",
    "noise_residual_sd", "eye_band_L_mean", "eye_band_lapvar",
]
GEOMETRY_METRICS = ["face_px_area", "face_short_side_px", "face_area_ratio", "interocular_px"]
POSE_METRICS = ["yaw_deg", "pitch_deg", "roll_deg", "yaw_proxy", "pitch_proxy", "roll_eye_line_deg"]
ALL_METRICS = PIXEL_METRICS + GEOMETRY_METRICS + POSE_METRICS

HEAD_MODEL = np.array([
    [-225.0, -170.0, 135.0],
    [225.0, -170.0, 135.0],
    [0.0, 0.0, 0.0],
    [-150.0, 150.0, 125.0],
    [150.0, 150.0, 125.0],
])


def ellipse_mask(h: int, w: int, wf: float, hf: float, cy: float = 0.5) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, int(round(h * cy))), (max(1, int(w * wf / 2)), max(1, int(h * hf / 2))), 0, 0, 360, 1, -1)
    return m.astype(bool)


def rotation_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    return yaw, pitch, roll


def estimate_pose(landmarks: np.ndarray, width: int, height: int) -> tuple[float, float, float, str]:
    f = float(max(width, height))
    cam = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float64)
    pts = landmarks.astype(np.float64).reshape(5, 2)
    ok, rvec, tvec = cv2.solvePnP(HEAD_MODEL, pts, cam, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return np.nan, np.nan, np.nan, "pnp_failed"
    ok, rvec, tvec = cv2.solvePnP(HEAD_MODEL, pts, cam, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return np.nan, np.nan, np.nan, "pnp_failed"
    if float(tvec[2, 0]) <= 0:
        return np.nan, np.nan, np.nan, "pnp_behind_camera"
    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = rotation_to_euler(R)
    return yaw, pitch, roll, "ok"


def landmark_geometry(lm: np.ndarray) -> dict:
    le, re, nose, ml, mr = lm
    eye_mid = (le + re) / 2
    mouth_mid = (ml + mr) / 2
    io = float(np.linalg.norm(re - le))
    vert = float(mouth_mid[1] - eye_mid[1])
    return {
        "interocular_px": io,
        "roll_eye_line_deg": math.degrees(math.atan2(re[1] - le[1], re[0] - le[0])),
        "yaw_proxy": float((nose[0] - eye_mid[0]) / io) if io > 0 else np.nan,
        "pitch_proxy": float((nose[1] - eye_mid[1]) / vert) if vert > 0 else np.nan,
    }


def measure_visual(record: dict, ctx: dict) -> dict:
    v = ctx["vis"]
    out = {"image_id": record["image_id"], "status": "ok", "error": None}
    path = resolve_file(record.get("aligned_path"))
    if path is None:
        out["status"] = "image_missing"
        return out
    rgb, _ = load_rgb_for_analysis(path)
    h, w = rgb.shape[:2]
    valid, mask_method = valid_pixel_mask(record, (h, w))
    ell = ellipse_mask(h, w, v["ellipse_width_fraction"], v["ellipse_height_fraction"], v.get("ellipse_centre_y", 0.5))
    roi = ell & erode(valid, int(v.get("mask_erode_px", 3)))
    out.update(valid_mask_method=mask_method, aligned_width_px=w, aligned_height_px=h,
               roi_px=int(roi.sum()), roi_valid_fraction=float(roi.sum() / max(1, ell.sum())))

    if roi.sum() >= int(v.get("min_roi_px", 500)):
        L = rgb2lab(rgb.astype(np.float64) / 255.0)[..., 0]
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        vals = L[roi]
        gv = grey[roi]
        left = roi.copy()
        left[:, w // 2:] = False
        right = roi.copy()
        right[:, : w // 2] = False
        lap = cv2.Laplacian(grey, cv2.CV_64F)
        gx = cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3)
        blur = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), 1.0)
        out.update(
            brightness_L_mean=float(vals.mean()),
            brightness_L_median=float(np.median(vals)),
            lstar_p5_p95_range=float(np.percentile(vals, 95) - np.percentile(vals, 5)),
            contrast_L_sd=float(vals.std()),
            illumination_asymmetry=float(abs(L[left].mean() - L[right].mean())) if left.any() and right.any() else np.nan,
            shadow_clip_fraction=float(np.mean(gv <= int(v.get("shadow_clip_level", 10)))),
            highlight_clip_fraction=float(np.mean(gv >= int(v.get("highlight_clip_level", 245)))),
            sharpness_lapvar=float(lap[roi].var()),
            sharpness_tenengrad=float((gx[roi] ** 2 + gy[roi] ** 2).mean()),
            noise_residual_sd=float((grey.astype(np.float32) - blur)[roi].std()),
        )
        y1, y2 = [int(round(h * f)) for f in v.get("eye_band_y", [0.30, 0.50])]
        band = np.zeros_like(roi)
        band[y1:y2] = True
        band &= roi
        if band.sum() >= 50:
            out["eye_band_L_mean"] = float(L[band].mean())
            out["eye_band_lapvar"] = float(lap[band].var())
    else:
        out["status"] = "roi_too_small"

    W, H = record.get("width_px"), record.get("height_px")
    bbox = as_array(record.get("bbox"), (4,))
    if bbox is not None and not is_missing(W) and not is_missing(H):
        fs = measurements.face_size(bbox.tolist(), int(W), int(H))
        bw = max(0.0, min(bbox[2], W) - max(bbox[0], 0))
        bh = max(0.0, min(bbox[3], H) - max(bbox[1], 0))
        out.update(face_px_area=fs["face_px_area"], face_area_ratio=fs["face_area_ratio"],
                   face_short_side_px=float(min(bw, bh)), geometry_status="ok" if bw > 1 and bh > 1 else "invalid_bbox")
    else:
        out["geometry_status"] = "no_project_detection"

    lm = as_array(record.get("landmarks"), (5, 2))
    if lm is not None and not is_missing(W) and not is_missing(H):
        out.update(landmark_geometry(lm))
        out["yaw_deg"], out["pitch_deg"], out["roll_deg"], out["pose_status"] = estimate_pose(lm, int(W), int(H))
    else:
        out["pose_status"] = "no_landmarks"
    return out


def robust_outliers(df: pd.DataFrame, metrics: list[str], z: float) -> pd.DataFrame:
    flags = pd.DataFrame(index=df.index)
    for m in metrics:
        x = pd.to_numeric(df[m], errors="coerce")
        med = x.groupby(df["dataset"]).transform("median")
        mad = (x - med).abs().groupby(df["dataset"]).transform("median") * 1.4826
        flags[f"outlier_{m}"] = ((x - med).abs() / mad.replace(0, np.nan)) > z
    return flags


def _box(df, metric, group, path, min_n):
    t = df[["dataset", group, metric]].copy()
    t[group] = S.clean_labels(t[group])
    t[metric] = pd.to_numeric(t[metric], errors="coerce")
    arrays, labels = [], []
    for (ds, g), z in t.dropna(subset=[metric]).groupby(["dataset", group]):
        if g != "unknown" and len(z) >= min_n:
            arrays.append(z[metric].to_numpy())
            labels.append(f"{ds}\n{g}\nn={len(z)}")
    if not arrays:
        return
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(arrays)), 4.8))
    ax.boxplot(arrays, showfliers=False)
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(config_path: str = "configs/phase1_audit.json", workers: int | None = None) -> pd.DataFrame:
    cfg, cfg_path = load_audit_config(config_path)
    master, master_path = load_master(cfg)
    od = output_dirs(cfg, MODULE)
    scfg = cfg["stats"]
    boot = scfg["bootstrap"]
    min_n = int(scfg["min_n"]["image"])
    fam = scfg.get("bh_family", ["dataset", "group_variable"])
    workers = workers if workers is not None else int(cfg.get("workers", 1))

    eligible = master["eligible_visual_audit"].fillna(False).astype(bool) if "eligible_visual_audit" in master.columns else pd.Series(True, index=master.index)
    cols = [c for c in RECORD_COLS if c in master.columns]
    res = process_records(measure_visual, master.loc[eligible, cols].to_dict("records"), {"vis": cfg["vis"]},
                          od["tables"] / "VIS_checkpoint.csv", workers)
    vis = pd.concat([res, master.loc[~eligible, ["image_id"]].assign(status="not_eligible_visual_audit")], ignore_index=True)
    for m in ALL_METRICS:
        if m not in vis.columns:
            vis[m] = np.nan

    carry = [c for c in ["dataset", "subject_id", "aligned_path", "gender", "race", "age_band",
                         "exact_dup_group", "near_dup_group", "cross_dataset_dup"] if c in master.columns]
    vis = vis.merge(master[["image_id", *carry]], on="image_id", how="left", validate="one_to_one")
    mea_path = module_table(cfg, "measurement_audit", "MEA_image_measurements.csv")
    skin_axes = cfg["axes"]["skin"]
    if mea_path.exists():
        mea = pd.read_csv(mea_path, usecols=lambda c: c in {"image_id", "skin_L_star", "skin_ITA_deg", *skin_axes}, low_memory=False)
        vis = vis.merge(mea, on="image_id", how="left", validate="one_to_one")
    else:
        for a in skin_axes:
            vis[a] = None

    vis = pd.concat([vis, analysis_flags(vis, vis["status"].eq("ok"), cfg.get("dedup", {}))], axis=1)
    qc = robust_outliers(vis, PIXEL_METRICS + GEOMETRY_METRICS, float(cfg["vis"].get("outlier_robust_z", 5.0)))
    qc["flag_pose_implausible"] = vis[["yaw_deg", "pitch_deg", "roll_deg"]].abs().max(axis=1) > float(cfg["vis"].get("pose_max_abs_deg", 90))
    qc["flag_pose_failed"] = vis["pose_status"].isin(["pnp_failed", "pnp_behind_camera"])
    qc["flag_face_ratio_invalid"] = ~pd.to_numeric(vis["face_area_ratio"], errors="coerce").between(0, 1, inclusive="right") & vis["face_area_ratio"].notna()
    qc["n_flags"] = qc.sum(axis=1)
    vis = pd.concat([vis, qc], axis=1)
    written = [write_table(vis, od["tables"] / "VIS_image_measurements.csv", parquet=True)]

    written.append(write_table(vis.groupby(["dataset", "status"], dropna=False).size().reset_index(name="n"), od["tables"] / "VIS_status_counts.csv"))
    written.append(write_table(vis.groupby("dataset")[[c for c in qc.columns if c != "n_flags"]].sum().reset_index(), od["tables"] / "VIS_qc_flag_counts.csv"))

    ana = vis[vis["in_analysis"]].copy()
    summ = [{"dataset": ds, "metric": m, **S.summarise(pd.to_numeric(d[m], errors="coerce").to_numpy())}
            for ds, d in ana.groupby("dataset") for m in ALL_METRICS]
    written.append(write_table(pd.DataFrame(summ), od["tables"] / "VIS_summary_by_dataset.csv"))

    axes = [a for a in cfg["axes"]["within_dataset"] + skin_axes if a in ana.columns]
    sums, omni, pair = [], [], []
    for m in cfg["vis"]["compare_metrics"]:
        for axis in axes:
            s, o, p = S.subgroup_analysis(ana, m, axis, "dataset", min_n, boot)
            sums.append(s), omni.append(o), pair.append(p)
    written.append(write_table(pd.concat(sums, ignore_index=True), od["tables"] / "VIS_subgroup_summary.csv"))
    written.append(write_table(S.add_bh(pd.concat(omni, ignore_index=True), fam, alpha=scfg["bh_alpha"]), od["tables"] / "VIS_subgroup_omnibus.csv"))
    written.append(write_table(S.add_bh(pd.concat(pair, ignore_index=True), fam, alpha=scfg["bh_alpha"]), od["tables"] / "VIS_subgroup_pairwise.csv"))

    conf = []
    if "skin_L_star" in ana.columns:
        for ds, d in ana.groupby("dataset"):
            x = pd.to_numeric(d["brightness_L_mean"], errors="coerce")
            y = pd.to_numeric(d["skin_L_star"], errors="coerce")
            ok = x.notna() & y.notna()
            if ok.sum() >= 3:
                from scipy.stats import spearmanr
                rho, p = spearmanr(x[ok], y[ok])
                conf.append({"dataset": ds, "n": int(ok.sum()), "spearman_rho_brightness_vs_skin_L": float(rho), "p_value": float(p)})
    written.append(write_table(pd.DataFrame(conf), od["tables"] / "VIS_brightness_vs_skin_L.csv"))

    for m in cfg["vis"]["compare_metrics"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for ds, d in ana.groupby("dataset"):
            x = pd.to_numeric(d[m], errors="coerce").dropna()
            if len(x):
                ax.hist(x.clip(*np.nanpercentile(x, [0.5, 99.5])), bins=60, alpha=0.45, density=True, label=f"{ds} (n={len(x):,})")
        ax.set_xlabel(m)
        ax.legend()
        fig.tight_layout()
        fig.savefig(od["figures"] / f"VIS_hist_{m}.png", dpi=160)
        plt.close(fig)
        if "skin_group_ita" in ana.columns:
            _box(ana, m, "skin_group_ita", od["figures"] / f"VIS_box_{m}_by_skin_group_ita.png", min_n)

    for m in ["sharpness_lapvar", "brightness_L_mean", "yaw_deg"]:
        d = ana.dropna(subset=[m]).sort_values(m)
        pick = pd.concat([d.head(12), d.tail(12)])
        items = [(p, f"{r['dataset']} {r[m]:.1f}", None) for _, r in pick.iterrows() if (p := resolve_file(r["aligned_path"])) is not None]
        contact_sheet(items, od["qa"] / f"VIS_extremes_{m}.png")

    method = {
        "manual_review": False,
        "face_region": (f"central ellipse on the aligned crop (width {cfg['vis']['ellipse_width_fraction']}, height "
                        f"{cfg['vis']['ellipse_height_fraction']} of crop), intersected with the warped source-extent mask "
                        f"eroded by {cfg['vis'].get('mask_erode_px', 3)} px so alignment padding is never measured"),
        "image_loading": "src.image_integrity.load_rgb_for_analysis (EXIF applied, ICC converted to sRGB)",
        "brightness": "mean CIELAB L* (scikit-image rgb2lab, D65) in the face region; 0 black to 100 white; higher is brighter",
        "contrast": "SD of L* in the face region; higher is more contrast",
        "sharpness_primary": "variance of the 3x3 Laplacian of 8-bit grey inside the face region; higher is sharper",
        "sharpness_secondary": "mean Sobel gradient energy (Tenengrad); higher is sharper; exploratory",
        "face_size": "RetinaFace bbox from the original image, clipped to image bounds: area px^2, shorter side px, area/image area",
        "pose": ("solvePnP (EPnP then iterative) with a generic 5-point head model in OpenCV camera axes (x right, y down, "
                 "z forward), focal = max(W, H), principal point = image centre, on original-image landmarks. "
                 "Euler angles in degrees: yaw about y, pitch about x, roll about z. Approximate."),
        "pose_proxies": "yaw_proxy = (nose x - eye midpoint x) / interocular; pitch_proxy = (nose y - eye mid y) / (mouth mid y - eye mid y)",
        "fairface": "publisher crops; no project detection, so geometry and pose are missing by design",
        "qc": "robust z > outlier_robust_z within dataset, pose beyond pose_max_abs_deg, PnP failures; flagged, not removed",
    }
    write_json(method, od["metadata"] / "VIS_method.json")
    save_run_metadata(MODULE, cfg, cfg_path, master_path, od, {"status_counts": {str(k): int(v) for k, v in vis["status"].value_counts().items()}, "workers": workers})
    register_results(cfg, cfg_path, MODULE, written, "src/visual_audit.py")
    return vis
