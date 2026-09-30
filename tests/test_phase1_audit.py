from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from src import phase1_measurements as measurements, phase1_paths as paths
from src import stats as S
from src.lfw_audit import build_pair_table, parse_pair_file
from src.phase1_common import analysis_flags, valid_pixel_mask
from src.visual_audit import HEAD_MODEL, estimate_pose, landmark_geometry


def test_bh_matches_known_values():
    got = S.bh_adjust([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(got, [0.04, 0.0533333, 0.0533333, 0.2], atol=1e-6)


def test_cliffs_delta_direction_and_bounds():
    assert S.cliffs_delta([5, 6, 7], [1, 2, 3]) == 1.0
    assert S.cliffs_delta([1, 2, 3], [5, 6, 7]) == -1.0
    assert abs(S.cliffs_delta([1, 2, 3], [1, 2, 3])) < 1e-12


def test_kruskal_effect_sizes():
    rng = np.random.default_rng(0)
    arrays = [rng.normal(i, 1, 50) for i in range(3)]
    r = S.compare_many(arrays)
    n, k = 150, 3
    assert math.isclose(r["effect_size"], r["statistic"] / (n - 1))
    assert math.isclose(r["eta_squared_h"], (r["statistic"] - k + 1) / (n - k))


def test_wilson_and_median_ci():
    lo, hi = S.wilson_ci(50, 100)
    assert lo < 0.5 < hi
    x = np.arange(101, dtype=float)
    lo, hi = S.median_ci(x)
    assert lo <= 50 <= hi


def test_cluster_bootstrap_runs():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 60), rng.normal(1, 1, 60)
    ca, cb = np.repeat(np.arange(20), 3), np.repeat(np.arange(20), 3)
    r = S.compare_two(a, b, {"reps": 200, "seed": 1}, ca, cb)
    assert r["effect_ci_method"] == "bootstrap_cluster"
    assert r["effect_ci_low"] <= r["effect_size"] <= r["effect_ci_high"]


def test_agreement_statistics():
    x = np.linspace(-20, 60, 100)
    assert math.isclose(S.lins_ccc(x, x), 1.0)
    assert S.lins_ccc(x, x + 10) < 1.0
    k = S.cohens_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"])
    assert k["kappa"] == 1.0


def test_ita_guard_and_hue_quadrants():
    assert measurements.compute_ita(60, 0.2, 1.0) == (None, True)
    assert math.isclose(measurements.compute_ita(50, 10, 1.0)[0], 0.0)
    assert math.isclose(measurements.compute_hue(1, 1), 45.0)
    assert math.isclose(measurements.compute_hue(-1, 1), 135.0)
    assert math.isclose(measurements.compute_hue(-1, -1), 225.0)
    assert math.isclose(measurements.compute_hue(1, -1), 315.0)


def _project(yaw, pitch, roll, w=500, h=500):
    Rx = cv2.Rodrigues(np.array([math.radians(pitch), 0, 0.0]))[0]
    Ry = cv2.Rodrigues(np.array([0, math.radians(yaw), 0.0]))[0]
    Rz = cv2.Rodrigues(np.array([0, 0, math.radians(roll)]))[0]
    cam = np.array([[max(w, h), 0, w / 2], [0, max(w, h), h / 2], [0, 0, 1.0]])
    pts, _ = cv2.projectPoints(HEAD_MODEL, cv2.Rodrigues(Rz @ Ry @ Rx)[0], np.array([0, 0, 2000.0]), cam, None)
    return pts.reshape(5, 2)


@pytest.mark.parametrize("angles", [(0, 0, 0), (30, 0, 0), (0, -20, 0), (0, 0, 15), (20, 10, -10)])
def test_pose_recovers_known_angles(angles):
    yaw, pitch, roll, status = estimate_pose(_project(*angles), 500, 500)
    assert status == "ok"
    assert np.allclose([yaw, pitch, roll], angles, atol=0.5)


def test_positive_yaw_moves_nose_to_image_left():
    assert landmark_geometry(_project(30, 0, 0))["yaw_proxy"] < 0


def test_valid_mask_excludes_padding():
    rec = {"alignment_transform": [[1, 0, 20], [0, 1, 0]], "width_px": 100, "height_px": 100}
    m, method = valid_pixel_mask(rec, (100, 100))
    assert method == "warped_source_extent"
    assert not m[:, :20].any() and m[:, 20:].all()


def test_dedup_keeps_one_copy():
    df = pd.DataFrame({"image_id": ["a", "b", "c"], "exact_dup_group": ["G", "G", None], "near_dup_group": [None, None, None]})
    f = analysis_flags(df, pd.Series([True, True, True]), {})
    assert f["in_analysis"].tolist() == [True, False, True]


def test_pair_parsing_and_yoking(tmp_path):
    p = tmp_path / "pairs.txt"
    p.write_text("2\nA 1 2\nB 1 C 1\n", encoding="utf-8")
    pairs = parse_pair_file(p, "view1", "train")
    assert pairs["is_genuine"].tolist() == [True, False]
    images = pd.DataFrame({
        "image_id": ["a1", "a2", "b1", "c1"], "subject_id": ["A", "A", "B", "C"], "image_number": [1, 2, 1, 1],
        "mea_ok": [True] * 4, "skin_group_ita": ["lighter", "lighter", "darker", "darker"],
        "gender": ["unknown"] * 4, "skin_ITA_deg": [40, 41, 10, 12],
    })
    cfg = {"lfw": {"yoking_axes": ["skin_group_ita", "gender"], "visual_difference_metrics": []}}
    t = build_pair_table(pairs, images, cfg)
    assert t["resolved"].all()
    assert t.loc[1, "yoking"] == "yoked_on_known_axes"


SKIN_RGB = [(255, 224, 196), (234, 192, 160), (214, 160, 120), (190, 130, 90), (150, 95, 60), (100, 60, 40)]


def _face(rng, colour, size=224):
    img = np.full((size, size, 3), 40, np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (80, 100), 0, 0, 360, colour, -1)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


@pytest.fixture()
def project(tmp_path, monkeypatch):
    for name, value in {"PROJECT_ROOT": tmp_path, "DATA_DIR": tmp_path / "data", "CONFIGS_DIR": tmp_path / "configs",
                        "ALIGNED_DIR": tmp_path / "data/derived/aligned"}.items():
        monkeypatch.setattr(paths, name, value)
    (tmp_path / "configs").mkdir()
    (tmp_path / "metadata").mkdir()
    pre = {"skin_measurement": {
        "roi_patches": {"forehead": {"x": [0.38, 0.62], "y": [0.20, 0.32]},
                        "left_cheek": {"x": [0.25, 0.40], "y": [0.50, 0.66]},
                        "right_cheek": {"x": [0.60, 0.75], "y": [0.50, 0.66]}},
        "roi_skin_filter": {"cb_range": [77, 127], "cr_range": [133, 173]},
        "roi_min_valid_pixels": 150, "ita_guard": {"b_star_min_abs": 1.0}}}
    cat = {"ita_groups": [
        {"name": "very_light", "min_exclusive": 55, "max_inclusive": None, "ordinal_rank": 1},
        {"name": "light", "min_exclusive": 41, "max_inclusive": 55, "ordinal_rank": 2},
        {"name": "intermediate", "min_exclusive": 28, "max_inclusive": 41, "ordinal_rank": 3},
        {"name": "tan", "min_exclusive": 10, "max_inclusive": 28, "ordinal_rank": 4},
        {"name": "brown", "min_exclusive": -30, "max_inclusive": 10, "ordinal_rank": 5},
        {"name": "dark", "min_exclusive": None, "max_inclusive": -30, "ordinal_rank": 6}],
        "skin_binary": {"ita": {"threshold_degrees": 28}}}
    (tmp_path / "configs/phase1_preprocessing.yaml").write_text(yaml.safe_dump(pre))
    (tmp_path / "configs/phase1_categories.yaml").write_text(yaml.safe_dump(cat))

    rng = np.random.default_rng(26041)
    rows = []
    aligned = tmp_path / "data/derived/aligned"
    aligned.mkdir(parents=True)
    for ds, n in [("lfw", 180), ("fairface", 160), ("utkface", 160)]:
        for i in range(n):
            colour = SKIN_RGB[i % len(SKIN_RGB)]
            path = aligned / f"{ds}_{i:04d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(_face(rng, colour), cv2.COLOR_RGB2BGR))
            ident = f"Person_{i // 3:03d}"
            dx = float(rng.uniform(-8, 8))
            lm = [[90 + dx, 110], [160 + dx, 110], [125 + 1.5 * dx, 145], [98, 180], [152, 180]]
            rows.append({
                "image_id": f"{ds}-{i:05d}", "dataset": ds,
                "original_path": f"lfw/{ident}/{ident}_{i % 3 + 1:04d}.jpg" if ds == "lfw" else f"{ds}/{i}.jpg",
                "subject_id": ident if ds == "lfw" else None,
                "aligned_path": str(path.relative_to(tmp_path)),
                "alignment_transform": None if ds == "fairface" else [[1.0, 0.0, -10.0], [0.0, 1.0, -5.0]],
                "width_px": 250, "height_px": 250,
                "bbox": None if ds == "fairface" else [70.0, 60.0, 180.0, 200.0],
                "landmarks": None if ds == "fairface" else lm,
                "is_greyscale": i == 5,
                "gender": "unknown" if ds == "lfw" else ("female" if i % 2 else "male"),
                "race": "unknown" if ds == "lfw" else ["White", "Black", "Asian", "Indian"][i % 4],
                "age_band": None if ds == "lfw" else ["20-29", "30-39"][i % 2],
                "exact_dup_group": "EXACT-1" if (ds == "utkface" and i in (1, 2)) else None,
                "near_dup_group": None, "cross_dataset_dup": False,
                "eligible_visual_audit": i != 7, "eligible_measurement_audit": i != 9, "L_star": None, "ITA": None,
            })
    master = pd.DataFrame(rows)
    master["alignment_transform"] = master["alignment_transform"].map(lambda v: json.dumps(v) if v is not None else None)
    master["bbox"] = master["bbox"].map(lambda v: json.dumps(v) if v is not None else None)
    master["landmarks"] = master["landmarks"].map(lambda v: json.dumps(v) if v is not None else None)
    master.to_csv(tmp_path / "metadata/master_annotations.csv", index=False)

    lfw_ids = sorted({r["subject_id"] for r in rows if r["dataset"] == "lfw"})
    lines = ["30"]
    for ident in lfw_ids[:30]:
        lines.append(f"{ident} 1 2")
    for a, b in zip(lfw_ids[:30], lfw_ids[30:60]):
        lines.append(f"{a} 1 {b} 3")
    (tmp_path / "data/lfw").mkdir(parents=True)
    (tmp_path / "data/lfw/pairsDevTrain.txt").write_text("\n".join(lines) + "\n")

    from pathlib import Path
    real = json.loads((Path(__file__).resolve().parents[1] / "configs/phase1_audit.json").read_text())
    real["workers"] = 1
    real["master_table"] = "metadata/master_annotations.csv"
    real["stats"]["bootstrap"]["reps"] = 100
    real["lfw"]["pair_files"] = [{"view": "view1", "split": "train", "path": "data/lfw/pairsDevTrain.txt"}]
    (tmp_path / "configs/phase1_audit.json").write_text(json.dumps(real))
    return tmp_path


def test_end_to_end(project):
    from src import lfw_audit, measurement_audit, shift_audit, visual_audit
    mea = measurement_audit.run()
    assert (mea["status"] == "greyscale_no_chroma").sum() == 3
    assert (mea["status"] == "not_eligible_measurement_audit").sum() == 3
    assert not mea.loc[mea["image_id"].str.endswith("00007"), "status"].eq("not_eligible_measurement_audit").any()
    ok = mea[mea["status"].eq("ok")]
    assert len(ok) > 400
    assert ok.groupby("dataset")["skin_ITA_deg"].apply(lambda s: s.max() - s.min()).min() > 30
    assert mea.loc[mea["image_id"].eq("utkface-00002"), "analysis_exclusion"].iloc[0] == "duplicate_copy:exact_dup_group"

    vis = visual_audit.run()
    lfw_rows = vis[vis["dataset"].eq("lfw") & vis["status"].eq("ok")]
    assert lfw_rows["valid_mask_method"].eq("warped_source_extent").all()
    assert lfw_rows["shadow_clip_fraction"].max() < 0.05
    assert lfw_rows["pose_status"].eq("ok").all()
    assert vis.loc[vis["dataset"].eq("fairface"), "yaw_deg"].isna().all()

    shift_audit.run()
    table = lfw_audit.run()
    assert len(table) == 60 and table["resolved"].all()
    out = project / "outputs/phase1"
    for f in ["measurement_audit/tables/MEA_reliability_agreement.csv", "visual_audit/tables/VIS_subgroup_pairwise.csv",
              "shift_audit/tables/SHIFT_nominal_balance.csv", "lfw_audit/tables/LFW_impostor_same_group_vs_random.csv",
              "result_registry.csv"]:
        assert (out / f).exists(), f
    reg = pd.read_csv(out / "result_registry.csv")
    assert set(reg["module"]) == {"measurement_audit", "visual_audit", "shift_audit", "lfw_audit"}
    assert reg["file"].str.endswith(".png").any(), "PNG figures/QA images must be registered"
    assert reg["file"].str.endswith(".json").any(), "JSON metadata must be registered"
    assert reg["file"].str.contains("/qa/").any(), "QA images must be registered"


def test_lfw_is_optional_when_pair_files_missing(project):
    cfg_path = project / "configs/phase1_audit.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["lfw"]["pair_files"] = [{"view": "view1", "split": "train", "path": "data/lfw/DOES_NOT_EXIST.txt"}]
    cfg_path.write_text(json.dumps(cfg))
    from src import lfw_audit
    out = lfw_audit.run()
    assert out.empty
    status = json.loads((project / "outputs/phase1/lfw_audit/metadata/LFW_status.json").read_text())
    assert status["status"] == "skipped"
    reg = pd.read_csv(project / "outputs/phase1/result_registry.csv")
    lfw_files = reg.loc[reg["module"].eq("lfw_audit"), "file"].tolist()
    assert any(x.endswith("LFW_status.json") for x in lfw_files)
