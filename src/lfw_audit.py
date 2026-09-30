from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src import stats as S
from src.phase1_common import (
    image_number, is_missing, load_audit_config, load_master, module_table, output_dirs,
    project_path, register_results, save_run_metadata, write_json, write_table,
)

MODULE = "lfw_audit"
EXPECTED = {
    ("view1", "train"): (1100, 1100),
    ("view1", "test"): (500, 500),
    ("view2", "10fold"): (3000, 3000),
}


def pair_id(view: str, split: str, fold, id1: str, n1: int, id2: str, n2: int) -> str:
    payload = f"{view}|{split}|{fold}|{id1}|{n1}|{id2}|{n2}"
    return "LFWPAIR-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _rows_from_file(path: Path) -> list[list[str]]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, header=0, dtype=str)
        return [[str(v).strip() for v in row if not is_missing(v)] for row in frame.itertuples(index=False)]
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            rows.append(parts)
    if rows and all(p.isdigit() for p in rows[0]):
        rows = rows[1:]
    return rows


def parse_pair_file(path: Path, view: str, split: str) -> pd.DataFrame:
    out = []
    for i, parts in enumerate(_rows_from_file(path)):
        if len(parts) == 3:
            out.append({"row": i, "is_genuine": True, "identity_1": parts[0], "image_number_1": int(parts[1]),
                        "identity_2": parts[0], "image_number_2": int(parts[2])})
        elif len(parts) == 4:
            out.append({"row": i, "is_genuine": False, "identity_1": parts[0], "image_number_1": int(parts[1]),
                        "identity_2": parts[2], "image_number_2": int(parts[3])})
        else:
            out.append({"row": i, "is_genuine": None, "parse_error": f"unexpected field count {len(parts)}"})
    df = pd.DataFrame(out)
    df["view"], df["split"] = view, split
    df["fold"] = (df["row"] // 600 + 1) if view == "view2" else 1
    return df


def parse_dev_pair_files(match_path: Path, mismatch_path: Path, view: str, split: str) -> pd.DataFrame:
    """Parse the LFW development protocol when genuine/impostor pairs are supplied as two
    separate files (matchpairsDevTrain.csv / mismatchpairsDevTrain.csv style) instead of one
    combined file. Columns are read by position, not by name, since pandas renames the
    mismatch file's duplicate "name" header to "name.1"."""
    match = pd.read_csv(match_path)
    mismatch = pd.read_csv(mismatch_path)
    genuine = pd.DataFrame({
        "is_genuine": True,
        "identity_1": match.iloc[:, 0].astype(str),
        "image_number_1": match.iloc[:, 1].astype(int),
        "identity_2": match.iloc[:, 0].astype(str),
        "image_number_2": match.iloc[:, 2].astype(int),
    })
    impostor = pd.DataFrame({
        "is_genuine": False,
        "identity_1": mismatch.iloc[:, 0].astype(str),
        "image_number_1": mismatch.iloc[:, 1].astype(int),
        "identity_2": mismatch.iloc[:, 2].astype(str),
        "image_number_2": mismatch.iloc[:, 3].astype(int),
    })
    df = pd.concat([genuine, impostor], ignore_index=True)
    df["row"] = df.index
    df["view"], df["split"], df["fold"] = view, split, 1
    return df


def validate_pairs(df: pd.DataFrame, view: str, split: str) -> dict:
    g, i = int((df["is_genuine"] == True).sum()), int((df["is_genuine"] == False).sum())
    exp = EXPECTED.get((view, split))
    res = {"view": view, "split": split, "n_genuine": g, "n_impostor": i, "expected": exp,
           "counts_match_protocol": exp is None or (g, i) == exp, "parse_errors": int(df.get("parse_error", pd.Series(dtype=object)).notna().sum())}
    if view == "view2":
        fc = df.groupby(["fold", "is_genuine"]).size().unstack(fill_value=0)
        res["folds_ok"] = bool(len(fc) == 10 and (fc == 300).all().all())
        if not res["folds_ok"]:
            df["fold"] = np.nan
    return res


def load_external_gender(cfg: dict) -> pd.Series | None:
    spec = cfg["lfw"].get("external_gender_labels")
    if not spec:
        return None
    p = project_path(spec["path"])
    if not p.exists():
        raise FileNotFoundError(f"External LFW gender label file not found: {p}")
    t = pd.read_csv(p, dtype=str)
    g = t[spec["gender_column"]].str.strip().str.lower().map({"m": "male", "male": "male", "f": "female", "female": "female", "1": "male", "0": "female"})
    return pd.Series(g.to_numpy(), index=t[spec["identity_column"]].str.strip()).groupby(level=0).first()


def _yoke(row: pd.Series, axes: list[str]) -> str:
    if row["is_genuine"]:
        return "genuine"
    known = [a for a in axes if row[f"{a}_known"]]
    if not known:
        return "unknown"
    same = [row[f"same_{a}"] for a in known]
    if all(same):
        return "fully_yoked" if len(known) == len(axes) else "yoked_on_known_axes"
    if any(same):
        return "partially_yoked"
    return "cross_demographic"


def build_pair_table(pairs: pd.DataFrame, images: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    axes = [a for a in cfg["lfw"]["yoking_axes"] if a in images.columns]
    metrics = [m for m in cfg["lfw"]["visual_difference_metrics"] if m in images.columns]
    key = images.dropna(subset=["subject_id", "image_number"]).assign(image_number=lambda d: d["image_number"].astype(int)).set_index(["subject_id", "image_number"])
    if key.index.duplicated().any():
        raise RuntimeError("LFW images are not unique by identity and image number.")
    cols = ["image_id", "mea_ok", *axes, *metrics, "skin_ITA_deg"]
    cols = list(dict.fromkeys(c for c in cols if c in key.columns))
    out = pairs.copy()
    for s in ["1", "2"]:
        idx = pd.MultiIndex.from_arrays([out[f"identity_{s}"], out[f"image_number_{s}"]])
        found = key.reindex(idx)[cols].reset_index(drop=True)
        found.columns = [f"{c}_{s}" for c in cols]
        out = pd.concat([out.reset_index(drop=True), found], axis=1)
    out["resolved"] = out["image_id_1"].notna() & out["image_id_2"].notna()
    out["both_measured"] = out["mea_ok_1"].fillna(False).astype(bool) & out["mea_ok_2"].fillna(False).astype(bool)
    for a in axes:
        k1 = S.clean_labels(out[f"{a}_1"]) != "unknown"
        k2 = S.clean_labels(out[f"{a}_2"]) != "unknown"
        out[f"{a}_known"] = k1 & k2
        out[f"same_{a}"] = out[f"{a}_known"] & (out[f"{a}_1"].astype(str) == out[f"{a}_2"].astype(str))
    for m in metrics:
        out[f"{m}_abs_diff"] = (pd.to_numeric(out[f"{m}_1"], errors="coerce") - pd.to_numeric(out[f"{m}_2"], errors="coerce")).abs()
    out["yoking"] = out.apply(lambda r: _yoke(r, axes), axis=1)
    out["yoking_axes"] = ",".join(axes)
    out["pair_id"] = [pair_id(r.view, r.split, r.fold, r.identity_1, r.image_number_1, r.identity_2, r.image_number_2)
                      for r in out.itertuples()]
    return out


def composition(images: pd.DataFrame, pairs: pd.DataFrame, axes: list[str], min_n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, tests = [], []
    measured = images[images["mea_ok"]]
    for (view, split), p in pairs[pairs["resolved"]].groupby(["view", "split"]):
        in_pairs = set(p["image_id_1"]) | set(p["image_id_2"])
        strata = {
            "all_lfw_images": measured,
            "images_in_pair_list": measured[measured["image_id"].isin(in_pairs)],
            "images_not_in_pair_list": measured[~measured["image_id"].isin(in_pairs)],
        }
        for kind, flag in [("genuine_pair_slots", True), ("impostor_pair_slots", False)]:
            q = p[p["is_genuine"] == flag]
            ids = pd.concat([q["image_id_1"], q["image_id_2"]])
            strata[kind] = measured.set_index("image_id").reindex(ids).dropna(subset=["dataset"]).reset_index()
        for axis in axes:
            for name, d in strata.items():
                t = S.proportion_table(d.assign(stratum=name), "stratum", axis, min_n)
                rows.append(t.assign(view=view, split=split))
            a, b = strata["images_in_pair_list"], strata["images_not_in_pair_list"]
            la, lb = S.clean_labels(a[axis]), S.clean_labels(b[axis])
            tab = pd.DataFrame({"in_pairs": la[la != "unknown"].value_counts(), "not_in_pairs": lb[lb != "unknown"].value_counts()}).fillna(0).T
            tests.append({"view": view, "split": split, "variable": axis, "comparison": "images in pair list vs not", **S.chi_square(tab)})
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.DataFrame(tests))


def yoking_tables(images: pd.DataFrame, pairs: pd.DataFrame, axes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    imp = pairs[(pairs["is_genuine"] == False) & pairs["resolved"]]
    counts = imp.groupby(["view", "split", "yoking"]).size().reset_index(name="n")
    counts["pct"] = 100 * counts["n"] / counts.groupby(["view", "split"])["n"].transform("sum")
    measured = images[images["mea_ok"]]
    exp_rows = []
    for axis in axes:
        lab = S.clean_labels(measured[axis])
        p = lab[lab != "unknown"].value_counts(normalize=True)
        expected = float((p ** 2).sum()) if len(p) else np.nan
        for (view, split), d in imp.groupby(["view", "split"]):
            known = d[d[f"{axis}_known"]]
            n, k = len(known), int(known[f"same_{axis}"].sum())
            test = binomtest(k, n, expected) if n and np.isfinite(expected) else None
            lo, hi = S.wilson_ci(k, n)
            exp_rows.append({"view": view, "split": split, "axis": axis, "n_impostor_known": n, "n_missing": int(len(d) - n),
                             "n_same": k, "observed_same_pct": 100 * k / n if n else np.nan, "ci_low": 100 * lo, "ci_high": 100 * hi,
                             "expected_same_pct_random_pairing": 100 * expected, "p_value": test.pvalue if test else np.nan,
                             "risk_difference_pp": (100 * k / n - 100 * expected) if n else np.nan})
    return counts, pd.DataFrame(exp_rows)


def run(config_path: str = "configs/phase1_audit.json") -> pd.DataFrame:
    cfg, cfg_path = load_audit_config(config_path)
    master, master_path = load_master(cfg)
    od = output_dirs(cfg, MODULE)
    lc = cfg["lfw"]
    scfg = cfg["stats"]
    boot = scfg["bootstrap"]
    min_pair = int(scfg["min_n"]["pair"])
    min_img = int(scfg["min_n"]["image"])

    protocol = {
        "source": "Huang, Ramesh, Berg and Learned-Miller (2007), UMass Amherst Technical Report 07-49; "
                  "Huang and Learned-Miller (2014), UMass Amherst Technical Report UM-CS-2014-003",
        "view1": "development only: pairsDevTrain (1100 genuine + 1100 impostor) and pairsDevTest (500 + 500)",
        "view2": "reporting only: 10 folds, each 300 genuine + 300 impostor; results reported as mean over folds",
        "view2_metadata_opened_in_phase1": bool(lc.get("allow_view2_metadata", False)),
        "declared_training_paradigm": lc["declared_training_paradigm"],
        "project_rules": [
            "No model is tuned, selected or thresholded on View 2.",
            "Yoking categories are frozen here, before any matcher score exists.",
            "Pairs that do not resolve to master rows are kept and flagged, not dropped.",
        ],
        "optional_stage": True,
    }
    write_json(protocol, od["metadata"] / "LFW_protocol_rules.json")

    # Check pair-file availability before requiring any LFW-specific upstream output.
    # This makes the normal Phase 1 workflow usable while LFW access/pair files are pending.
    files, checks = [], []
    for spec in lc.get("pair_files", []):
        if spec["view"] == "view2" and not lc.get("allow_view2_metadata", False):
            checks.append({"view": "view2", "split": spec["split"], "status": "sealed, not opened (allow_view2_metadata=false)"})
            continue
        if "match_path" in spec:
            mp, xp = project_path(spec["match_path"]), project_path(spec["mismatch_path"])
            missing = [str(p) for p in (mp, xp) if not p.exists()]
            if missing:
                checks.append({"view": spec["view"], "split": spec["split"], "status": f"file not found: {'; '.join(missing)}"})
                continue
            df = parse_dev_pair_files(mp, xp, spec["view"], spec["split"])
        else:
            pp = project_path(spec["path"])
            if not pp.exists():
                checks.append({"view": spec["view"], "split": spec["split"], "status": f"file not found: {pp}"})
                continue
            df = parse_pair_file(pp, spec["view"], spec["split"])
        checks.append({**validate_pairs(df, spec["view"], spec["split"]), "status": "parsed"})
        files.append(df)

    written = [write_table(pd.DataFrame(checks), od["tables"] / "LFW_pair_file_checks.csv")]
    if not files:
        status_path = write_json({
            "status": "skipped",
            "reason": "No eligible LFW pair files are currently available. This is an optional Phase 1 stage.",
            "checks": checks,
            "how_to_run_later": "python run_phase1_audit.py --stages lfw",
        }, od["metadata"] / "LFW_status.json")
        written.append(status_path)
        save_run_metadata(MODULE, cfg, cfg_path, master_path, od, {"status": "skipped_no_pair_files", "pair_file_checks": checks})
        register_results(cfg, cfg_path, MODULE, written, "src/lfw_audit.py")
        print("LFW stage skipped: no available pair files.", flush=True)
        return pd.DataFrame()

    lfw = master[master["dataset"].astype(str).str.lower().eq("lfw")].copy()
    if lfw.empty:
        status_path = write_json({
            "status": "skipped",
            "reason": "Pair files exist, but the master table currently contains no LFW rows.",
        }, od["metadata"] / "LFW_status.json")
        written.append(status_path)
        save_run_metadata(MODULE, cfg, cfg_path, master_path, od, {"status": "skipped_no_lfw_rows", "pair_file_checks": checks})
        register_results(cfg, cfg_path, MODULE, written, "src/lfw_audit.py")
        print("LFW stage skipped: no LFW rows in master table.", flush=True)
        return pd.DataFrame()

    if "original_path" not in lfw.columns:
        raise ValueError("LFW rows require original_path so image numbers can be resolved against official pair lists.")
    lfw["image_number"] = lfw["original_path"].map(image_number)

    mea_p = module_table(cfg, "measurement_audit", "MEA_image_measurements.csv")
    vis_p = module_table(cfg, "visual_audit", "VIS_image_measurements.csv")
    if not mea_p.exists():
        status_path = write_json({
            "status": "blocked",
            "reason": "LFW pair files are available, but MEA_image_measurements.csv is missing. Run MEA first.",
        }, od["metadata"] / "LFW_status.json")
        written.append(status_path)
        save_run_metadata(MODULE, cfg, cfg_path, master_path, od, {"status": "blocked_missing_mea", "pair_file_checks": checks})
        register_results(cfg, cfg_path, MODULE, written, "src/lfw_audit.py")
        print("LFW stage blocked: run MEA first.", flush=True)
        return pd.DataFrame()

    mea = pd.read_csv(mea_p, usecols=lambda c: c in {"image_id", "status", "skin_ITA_deg", "skin_group_ita", "ita_group"}, low_memory=False)
    lfw = lfw.drop(columns=[c for c in ["skin_group_ita", "ita_group"] if c in lfw.columns]).merge(
        mea.rename(columns={"status": "mea_status"}), on="image_id", how="left"
    )
    lfw["mea_ok"] = lfw["mea_status"].eq("ok")

    if vis_p.exists():
        vcols = set(lc["visual_difference_metrics"]) | {"image_id"}
        lfw = lfw.merge(pd.read_csv(vis_p, usecols=lambda c: c in vcols, low_memory=False), on="image_id", how="left")

    ext = load_external_gender(cfg)
    if ext is not None:
        lfw["gender"] = lfw["subject_id"].map(ext).fillna("unknown")
        lfw["gender_source"] = f"external:{lc['external_gender_labels'].get('source', 'unnamed')}"
    elif "gender" not in lfw.columns:
        lfw["gender"] = "unknown"

    pairs = pd.concat(files, ignore_index=True)
    pairs = pairs[pairs["is_genuine"].notna()]
    pairs["is_genuine"] = pairs["is_genuine"].astype(bool)

    axes = [a for a in lc["yoking_axes"] if a in lfw.columns]
    table = build_pair_table(pairs, lfw, cfg)
    written.append(write_table(table, od["tables"] / "LFW_pair_table.csv", parquet=True))
    written.append(write_table(table.groupby(["view", "split", "is_genuine", "resolved", "both_measured"]).size().reset_index(name="n"),
                               od["tables"] / "LFW_pair_resolution_counts.csv"))

    comp, comp_tests = composition(lfw, table, axes, min_img)
    written.append(write_table(comp, od["tables"] / "LFW_composition_images_vs_pairs.csv"))
    written.append(write_table(S.add_bh(comp_tests, ["view", "split"], alpha=scfg["bh_alpha"]), od["tables"] / "LFW_composition_tests.csv"))

    yk, yk_exp = yoking_tables(lfw, table, axes)
    written.append(write_table(yk, od["tables"] / "LFW_impostor_yoking_counts.csv"))
    written.append(write_table(S.add_bh(yk_exp, ["view", "split"], alpha=scfg["bh_alpha"]), od["tables"] / "LFW_impostor_same_group_vs_random.csv"))

    gen = table[table["is_genuine"] & table["both_measured"]].copy()
    if "skin_group_ita" in axes and not gen.empty:
        gen["identity_skin_group"] = np.where(gen["same_skin_group_ita"], gen["skin_group_ita_1"], "mixed_within_identity")
        sums, pw = [], []
        for m in [c for c in gen.columns if c.endswith("_abs_diff")]:
            s, _, pp = S.subgroup_analysis(gen, m, "identity_skin_group", "view", min_pair, boot, cluster_col="identity_1")
            sums.append(s), pw.append(pp)
        if sums:
            written.append(write_table(pd.concat(sums, ignore_index=True), od["tables"] / "LFW_genuine_visual_difference_by_skin_group.csv"))
        if pw:
            written.append(write_table(S.add_bh(pd.concat(pw, ignore_index=True), ["view"], alpha=scfg["bh_alpha"]),
                                       od["tables"] / "LFW_genuine_visual_difference_tests.csv"))

    diff_cols = [c for c in table.columns if c.endswith("_abs_diff")]
    by_type = table[table["both_measured"]].groupby(["view", "split", "yoking"])[diff_cols].median().reset_index() if diff_cols else pd.DataFrame()
    written.append(write_table(by_type, od["tables"] / "LFW_visual_difference_median_by_yoking.csv"))

    gender_known = float(100 * (S.clean_labels(lfw["gender"]) != "unknown").mean())
    status_path = write_json({"status": "complete", "pair_files_parsed": len(files)}, od["metadata"] / "LFW_status.json")
    written.append(status_path)
    save_run_metadata(MODULE, cfg, cfg_path, master_path, od, {"pair_file_checks": checks, "yoking_axes": axes,
                                                               "lfw_gender_known_pct": gender_known})
    register_results(cfg, cfg_path, MODULE, written, "src/lfw_audit.py")
    return table
