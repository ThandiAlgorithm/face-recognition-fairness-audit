"""Derive LFW race and gender fields, kept clearly separate from the native
labels used for FairFace/UTKFace.

These are NOT native/human-annotated labels in the same sense as FairFace's
or UTKFace's. They are derived from external sources, kept under their own
provenance columns so they are never silently confused with the project's
native-vocabulary fields.

- Race: lfw_att_73.mat (Kumar et al. 2009 attribute classifier), four
  non-exclusive binary flags (Asian/White/Black/Indian). A single positive
  flag -> that category. Zero or multiple positive flags -> "unknown"
  rather than guessed (~16% of LFW).
- Gender: data/LFW/{female,male}_names.txt, per-image name lists. Verified
  against the attribute classifier's Male flag -- the name lists are more
  reliable (e.g. correctly all-female for every Condoleezza_Rice image,
  where the attribute classifier mislabeled one as male). The two list
  files have 5 known-conflicting entries (appear in both); those 5 are
  marked "unknown" rather than guessed, as is any image in neither list.

Writes an augmented copy of the master table -- the canonical frozen
metadata/master_annotations.parquet is never modified.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

MASTER_PATH = Path("metadata/master_annotations.parquet")
ATTR_MAT_PATH = Path("lfw_att_73.mat")
FEMALE_NAMES_PATH = Path("data/LFW/female_names.txt")
MALE_NAMES_PATH = Path("data/LFW/male_names.txt")
OUTPUT_PATH = Path("metadata/master_annotations_lfw_demographics_derived.parquet")

RACE_ATTRS = ["Asian", "White", "Black", "Indian"]


def parse_name_list(path: Path) -> set[str]:
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.endswith(".jpg"):
            line = line[:-4]
        out.add(line)
    return out


def load_gender_from_name_lists() -> dict[tuple[str, int], str]:
    female = parse_name_list(FEMALE_NAMES_PATH)
    male = parse_name_list(MALE_NAMES_PATH)
    conflicting = female & male

    lookup: dict[tuple[str, int], str] = {}
    for stem in female - conflicting:
        parts = stem.rsplit("_", 1)
        lookup[(parts[0], int(parts[1]))] = "female"
    for stem in male - conflicting:
        parts = stem.rsplit("_", 1)
        lookup[(parts[0], int(parts[1]))] = "male"
    for stem in conflicting:
        parts = stem.rsplit("_", 1)
        lookup[(parts[0], int(parts[1]))] = "unknown"
    return lookup


def deref_strings(f: h5py.File, ds: h5py.Dataset) -> list[str]:
    return ["".join(chr(c) for c in f[r][()].flatten()) for r in ds[()].flatten()]


def load_race_attributes() -> pd.DataFrame:
    with h5py.File(ATTR_MAT_PATH, "r") as f:
        attrnames = deref_strings(f, f["AttrName"])
        names = deref_strings(f, f["name"])
        label = f["label"][()]

    idx = {a: attrnames.index(a) for a in RACE_ATTRS}
    race_mat = np.stack([label[idx[a]] for a in RACE_ATTRS], axis=1)
    n_positive = race_mat.sum(axis=1)
    argmax = np.array(RACE_ATTRS)[race_mat.argmax(axis=1)]
    race_derived = np.where(n_positive == 1, argmax, "Unknown")

    # name format: "Aaron_Eckhart\Aaron_Eckhart_0001.jpg" -> subject_id, image_number
    subject_ids, image_numbers = [], []
    for n in names:
        stem = n.split("\\")[-1].rsplit(".", 1)[0]
        parts = stem.rsplit("_", 1)
        subject_ids.append(parts[0])
        image_numbers.append(int(parts[1]))

    return pd.DataFrame({
        "subject_id": subject_ids,
        "image_number": image_numbers,
        "race_lfw_derived_kumar2009": race_derived,
        "race_lfw_derived_n_positive_flags": n_positive.astype(int),
    })


def main() -> None:
    master = pd.read_parquet(MASTER_PATH)
    is_lfw = master["dataset"].str.lower().eq("lfw")

    from src.phase1_common import image_number as extract_image_number
    lfw = master[is_lfw].copy()
    lfw["image_number"] = lfw["original_path"].map(extract_image_number).astype("int64")

    attr = load_race_attributes()
    race_lookup = dict(zip(zip(attr["subject_id"], attr["image_number"]), attr["race_lfw_derived_kumar2009"]))
    flag_lookup = dict(zip(zip(attr["subject_id"], attr["image_number"]), attr["race_lfw_derived_n_positive_flags"]))
    gender_lookup = load_gender_from_name_lists()

    keys = list(zip(lfw["subject_id"].astype(str), lfw["image_number"]))
    lfw["race_lfw_derived_kumar2009"] = [race_lookup.get(k) for k in keys]
    lfw["race_lfw_derived_n_positive_flags"] = [flag_lookup.get(k) for k in keys]
    lfw["gender_lfw_derived_name_list"] = [gender_lookup.get(k) for k in keys]
    lfw = lfw.drop(columns=["image_number"])

    # Populate the actual `race`/`gender` columns REP-00X read, so LFW is
    # included in composition/intersection/base-rate tables. Rows the
    # source doesn't cover, or couldn't be cleanly assigned, stay "unknown"
    # -- never guessed. The provenance columns are kept alongside.
    lfw["race"] = lfw["race_lfw_derived_kumar2009"].fillna("unknown").replace("Unknown", "unknown")
    race_assigned = lfw["race"].ne("unknown")
    lfw["race_source"] = lfw["race_source"].where(~race_assigned, "derived_kumar2009_attribute_classifier")

    lfw["gender"] = lfw["gender_lfw_derived_name_list"].fillna("unknown")
    gender_assigned = lfw["gender"].ne("unknown")
    lfw["gender_source"] = lfw["gender_source"].where(~gender_assigned, "derived_lfw_name_lists")

    # Phase 0 marked every LFW row ineligible for demographic analysis
    # (no native labels existed at freeze time). Now that gender is
    # populated from a verified external source, override eligibility for
    # exactly those rows -- in this derived copy only, never touching the
    # canonical frozen master table or its eligibility determination for
    # any other row.
    eligibility_overridden = gender_assigned & lfw["eligible_demographic_audit"].eq(False)
    lfw["eligible_demographic_audit"] = lfw["eligible_demographic_audit"].where(~gender_assigned, True)
    lfw["eligible_demographic_audit_overridden"] = eligibility_overridden

    non_lfw = master[~is_lfw].copy()
    non_lfw["eligible_demographic_audit_overridden"] = False
    non_lfw["race_lfw_derived_kumar2009"] = pd.NA
    non_lfw["race_lfw_derived_n_positive_flags"] = pd.NA
    non_lfw["gender_lfw_derived_name_list"] = pd.NA

    merged = pd.concat([non_lfw, lfw], ignore_index=True).sort_values("image_id").reset_index(drop=True)
    merged.to_parquet(OUTPUT_PATH, index=False)

    lfw = merged[merged["dataset"].str.lower().eq("lfw")]
    print("LFW rows:", len(lfw))
    print("\nrace (now populated for REP-00X):")
    print(lfw["race"].value_counts(dropna=False))
    print("\ngender (now populated for REP-00X):")
    print(lfw["gender"].value_counts(dropna=False))
    print("\neligible_demographic_audit (after override):")
    print(lfw["eligible_demographic_audit"].value_counts(dropna=False))
    print("rows where eligibility was overridden:", int(lfw["eligible_demographic_audit_overridden"].sum()))
    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
