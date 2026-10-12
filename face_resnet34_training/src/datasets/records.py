"""Step 2/3: resolve real image paths and build one record per image.

Reads directly from the project's own master table rather than the PDF
spec's assumed `data/<dataset>/aligned_cropped/` layout -- that layout was
never actually produced. What exists on disk is:

  - LFW / UTKFace: Phase 0 wrote aligned crops to
    data/derived/aligned/{lfw,utkface}/<dataset>-<hash>.png, and
    master_annotations.parquet's aligned_path column already points at them,
    relative to the repo root.
  - FairFace: Phase 0 never re-wrote aligned crops; aligned_path equals
    original_path (FairFace's own publisher layout), and that path is
    relative to data/, not the repo root.

resolve_image_path() tries both roots so one function covers both
conventions without hardcoding per-dataset branches.

LFW has no native race/gender (master_annotations.parquet reports
'unknown' for every row); this reads them from
metadata/master_annotations_lfw_demographics_derived.parquet instead, which
carries the *_source columns needed to distinguish derived from native
labels everywhere downstream.

ita_value/ita_group are read straight from the master table (ITA,
skin_group_ita) and never recomputed: skin_group_ita already is the frozen
GOV-002 binary split (ITA > 28 -> lighter, <= 28 -> darker;
configs/categories.yaml, skin_binary.ita), verified to match that rule on
100% of non-null rows.

LFW's own training target (a multi-label attribute head) comes from
lfw_att_73.mat -- the fuller LFWA+ release, which includes native
Male/Asian/White/Black/Indian flags alongside the usual appearance
attributes. race_label/gender_label (below) stay sourced from
master_annotations_lfw_demographics_derived.parquet regardless -- they're
audit-only grouping columns, independent of whatever the attribute head is
trained on.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

try:
    import imagehash
except ImportError:
    imagehash = None

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data"
MASTER_PATH = REPO_ROOT / "metadata" / "master_annotations.parquet"
LFW_DEMOGRAPHICS_PATH = REPO_ROOT / "metadata" / "master_annotations_lfw_demographics_derived.parquet"
LFW_ATTR_PATH = REPO_ROOT / "lfw_att_73.mat"
RECORDS_DIR = Path(__file__).resolve().parents[2] / "data" / "records"

RECORD_COLUMNS = [
    "image_id",
    "image_path",
    "dataset",
    "identity_id",
    "race_label",
    "race_source",
    "gender_label",
    "gender_source",
    "age_value",
    "age_source",
    "age_group",
    "ita_value",
    "ita_group",
    "alignment_valid",
    "exclusion_reason",
    "eligible_demographic_audit",
    "eligible_demographic_audit_overridden",
    "exact_dup_group",
    "near_dup_group",
    "cross_dataset_dup",
    "image_sha256",
    "phash",
]


def resolve_image_path(aligned_path: str | float | None) -> Path | None:
    if not isinstance(aligned_path, str):
        return None
    repo_root_candidate = REPO_ROOT / aligned_path
    if repo_root_candidate.exists():
        return repo_root_candidate
    data_root_candidate = DATA_ROOT / aligned_path
    if data_root_candidate.exists():
        return data_root_candidate
    return None


def compute_image_hashes(path: Path) -> tuple[str | None, str | None]:
    try:
        raw = path.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        phash = None
        if imagehash is not None:
            with Image.open(path) as img:
                phash = str(imagehash.phash(img))
        return sha256, phash
    except Exception:
        return None, None


def _deref_mat_string(f: h5py.File, ref) -> str:
    arr = f[ref][()]
    return "".join(chr(c) for c in arr.flatten())


def load_lfw_attributes() -> pd.DataFrame:
    """Returns one row per LFW image: path_key (identity/filename.jpg, forward
    slashes) plus one 0/1 column per LFWA+ attribute. Attribute names are
    sanitised (spaces -> underscores) so they match the existing
    underscore_case convention used everywhere downstream (plot filenames,
    per_attribute_accuracy keys, etc.)."""
    with h5py.File(LFW_ATTR_PATH, "r") as f:
        attr_names = [
            _deref_mat_string(f, r[0]).replace(" ", "_") for r in f["AttrName"][()]
        ]
        path_keys = [
            _deref_mat_string(f, r[0]).replace("\\", "/") for r in f["name"][()]
        ]
        labels = f["label"][()].T  # (n_images, n_attributes)

    df = pd.DataFrame(labels, columns=[f"attr_{name}" for name in attr_names])
    df.insert(0, "path_key", path_keys)
    return df


LFW_ATTR_SCORES_PATH = REPO_ROOT / "data" / "LFW" / "lfw_attributes.txt"


def load_lfw_attribute_scores() -> pd.DataFrame:
    """Returns one row per LFW image: path_key (identity/filename.jpg) plus
    one continuous-score column per attribute (attr_<Name>_score), parsed
    from data/LFW/lfw_attributes.txt -- the original Kumar et al. 2009
    "Describable Visual Attributes" classifier output, and the actual
    source load_lfw_attributes()'s binary attr_ columns are thresholded
    from (verified: attr_Male == 1 iff this file's Male score > 0, spot
    checked). Never replaces the binary columns -- training already
    depends on those -- this is purely an additive, optional signal (e.g.
    attribute confidence/margin rather than just presence/absence) for
    future finer-grained analysis."""
    with open(LFW_ATTR_SCORES_PATH, encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    header = lines[1].lstrip("#").strip().split("\t")
    attr_names = [h.strip().replace(" ", "_") for h in header[2:]]  # skip 'person', 'imagenum'

    rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        person, imagenum = parts[0], parts[1]
        scores = [float(x) for x in parts[2:]]
        person_key = person.replace(" ", "_")
        path_key = f"{person_key}/{person_key}_{int(imagenum):04d}.jpg"
        rows.append([path_key, *scores])

    return pd.DataFrame(rows, columns=["path_key", *[f"attr_{n}_score" for n in attr_names]])


def _lfw_path_key(original_path: str) -> str:
    # original_path looks like "LFW/lfw/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg";
    # lfw_att_73.mat keys images as "Aaron_Eckhart/Aaron_Eckhart_0001.jpg".
    parts = original_path.split("/")
    return "/".join(parts[-2:])


def _load_master() -> pd.DataFrame:
    master = pd.read_parquet(MASTER_PATH)
    lfw_demographics = pd.read_parquet(LFW_DEMOGRAPHICS_PATH)
    lfw_cols = [
        "image_id",
        "race",
        "race_source",
        "gender",
        "gender_source",
        "eligible_demographic_audit",
        "eligible_demographic_audit_overridden",
    ]
    is_lfw = master["dataset"] == "lfw"
    master["eligible_demographic_audit_overridden"] = False
    lfw_lookup = lfw_demographics.set_index("image_id")[lfw_cols[1:]].to_dict("index")
    for col in lfw_cols[1:]:
        master.loc[is_lfw, col] = master.loc[is_lfw, "image_id"].map(
            lambda iid: lfw_lookup.get(iid, {}).get(col)
        )
    return master


def build_records(dataset: str, master: pd.DataFrame) -> pd.DataFrame:
    sub = master[master["dataset"] == dataset].copy()

    lfw_attrs = None
    lfw_attr_scores = None
    if dataset == "lfw":
        lfw_attrs = load_lfw_attributes()
        lfw_attr_scores = load_lfw_attribute_scores()
        sub["path_key"] = sub["original_path"].map(_lfw_path_key)

    rows = []
    missing_files = 0
    for _, row in sub.iterrows():
        resolved = resolve_image_path(row["aligned_path"])
        if resolved is None:
            missing_files += 1
            sha256, phash = None, None
            image_path = None
        else:
            sha256, phash = compute_image_hashes(resolved)
            image_path = resolved.relative_to(REPO_ROOT).as_posix()

        record = {
            "image_id": row["image_id"],
            "image_path": image_path,
            "dataset": dataset,
            "identity_id": row["subject_id"] if pd.notna(row.get("subject_id")) else None,
            "race_label": row["race"],
            "race_source": row["race_source"],
            "gender_label": row["gender"],
            "gender_source": row["gender_source"],
            "age_value": row["age"],
            "age_source": row["age_source"],
            "age_group": row["age_band"],
            "ita_value": row["ITA"],
            "ita_group": row["skin_group_ita"],
            "alignment_valid": row["alignment_valid"],
            "exclusion_reason": row["exclusion_reason"],
            "eligible_demographic_audit": row["eligible_demographic_audit"],
            "eligible_demographic_audit_overridden": row["eligible_demographic_audit_overridden"],
            "exact_dup_group": row["exact_dup_group"],
            "near_dup_group": row["near_dup_group"],
            "cross_dataset_dup": row["cross_dataset_dup"],
            "image_sha256": sha256,
            "phash": phash,
        }
        rows.append(record)

    records = pd.DataFrame(rows, columns=RECORD_COLUMNS)

    if lfw_attrs is not None:
        attr_lookup = lfw_attrs.set_index("path_key")
        attr_cols = [c for c in lfw_attrs.columns if c != "path_key"]
        path_keys = sub["path_key"].to_numpy()
        attr_values = attr_lookup.reindex(path_keys)[attr_cols].reset_index(drop=True)
        records = pd.concat([records.reset_index(drop=True), attr_values], axis=1)

    if lfw_attr_scores is not None:
        score_lookup = lfw_attr_scores.set_index("path_key")
        score_cols = [c for c in lfw_attr_scores.columns if c != "path_key"]
        path_keys = sub["path_key"].to_numpy()
        score_values = score_lookup.reindex(path_keys)[score_cols].reset_index(drop=True)
        records = pd.concat([records.reset_index(drop=True), score_values], axis=1)

    if missing_files:
        print(f"[{dataset}] WARNING: {missing_files} rows had no resolvable image file on disk")

    return records


def main() -> None:
    master = _load_master()
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in ["lfw", "fairface", "utkface"]:
        print(f"Building records for {dataset}...")
        records = build_records(dataset, master)
        out_path = RECORDS_DIR / f"{dataset}_records.csv"
        records.to_csv(out_path, index=False)
        n_missing_image = records["image_path"].isna().sum()
        n_missing_hash = records["image_sha256"].isna().sum()
        print(
            f"  {len(records)} rows written to {out_path.relative_to(REPO_ROOT)} "
            f"({n_missing_image} missing image file, {n_missing_hash} missing hash)"
        )


if __name__ == "__main__":
    main()
