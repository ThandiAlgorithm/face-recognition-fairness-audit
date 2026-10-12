"""BAL-M1 manifest enrichment: reads the dataset creator's own frozen
training_manifest_frozen.csv (never modified -- it's their source of
truth) and adds the two things it's missing, writing a new file alongside
it rather than overwriting anything:

- age_group_harmonized: training_manifest_frozen.csv's own age_group
  column mixes THREE incompatible binning schemes as-is -- FairFace's
  9-band native scheme, UTKFace's 4-band native scheme, and LFW's empty
  string (no native age at all). Harmonized here using the exact same
  mapping already proven for Combined-M1 (configs/age_mapping.yaml) --
  FairFace's 9 bands collapse cleanly into UTKFace's 4-band scheme
  (same underlying continuous age, just rebinned), LFW stays null (no
  numeric age exists to harmonize, never fabricated).
- fitzpatrick_approx: not present in the source manifest at all (it has
  ita_value/ita_group only). Computed from ita_value via the same
  approximate_fitzpatrick() breakpoints used everywhere else in this
  project (frs_m0/utils.py) -- explicitly named "_approx", per this
  project's standing rule that an ITA-derived estimate is never called
  "fitzpatrick" outright (no real Fitzpatrick annotation exists for any
  row here, so there is no bare `fitzpatrick` column -- only this
  clearly-labeled approximation).

Also verifies, for every row, whether the image file is actually present
on disk yet (the dataset was still transferring when this was built) --
written as a `file_present` column so training/benchmarking can honestly
report how much of the frozen manifest it could actually see, not just
assume completeness.

Everything else (race_label vs candidate_demographic_label, gender_label,
identity_id, split) is used exactly as the source manifest provides it --
already correctly harmonized by whoever built this dataset, not
re-derived here.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frs_m0"))
from utils import approximate_fitzpatrick  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
HARMOFACE_ROOT = REPO_ROOT / "data" / "New Dataset" / "HarmoFace-95K_v1"
# records_labeled.csv, not training_manifest_frozen.csv: verified identical
# 95,258 rows and identical split assignments (frozen rows are a strict
# subset of labeled's columns), but only records_labeled.csv carries
# materialized_relative_path -- the actual on-disk location under this
# project's images/ tree. training_manifest_frozen.csv's own image_path/
# image_path_original columns are both paths on the dataset-builder's own
# machine (nicol's), not usable here.
SOURCE_MANIFEST = HARMOFACE_ROOT / "metadata" / "records_labeled.csv"
AGE_MAPPING_PATH = Path(__file__).resolve().parents[1].parent / "configs" / "age_mapping.yaml"
OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "harmoface" / "BAL-M1"


def _load_age_mapping() -> dict:
    return yaml.safe_load(AGE_MAPPING_PATH.read_text())


def harmonize_age(source_dataset: str, age_group: str, mapping: dict) -> str | None:
    if source_dataset == "lfw":
        return None
    table = mapping.get(source_dataset, {})
    return table.get(age_group)  # None if genuinely unmapped -- never guessed


def build() -> None:
    age_mapping = _load_age_mapping()
    with open(SOURCE_MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_present = 0
    for row in rows:
        ita_raw = row.get("ita_value", "")
        ita_value = float(ita_raw) if ita_raw not in (None, "") else None
        row["fitzpatrick_approx"] = approximate_fitzpatrick(ita_value)
        row["age_group_original"] = row["age_group"]
        harmonized = harmonize_age(row["source_dataset"], row["age_group"], age_mapping)
        row["age_group_harmonized"] = harmonized if harmonized is not None else ""
        img_path = HARMOFACE_ROOT / row["materialized_relative_path"]
        present = img_path.is_file()
        row["file_present"] = "1" if present else "0"
        row["harmoface_abs_path"] = str(img_path)
        if present:
            n_present += 1

    out_path = OUT_DIR / "harmoface_manifest_enriched.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"n_rows={len(rows)} n_file_present={n_present} ({n_present/len(rows)*100:.1f}%)")
    print(f"wrote {out_path}")

    # Quick per-split, per-source completeness breakdown -- the thing that
    # actually determines whether training can start yet.
    from collections import Counter
    by_split_source = Counter((r["split"], r["source_dataset"]) for r in rows if r["file_present"] == "1")
    total_by_split_source = Counter((r["split"], r["source_dataset"]) for r in rows)
    print("\nFile availability by split x source:")
    for key in sorted(total_by_split_source):
        have = by_split_source.get(key, 0)
        total = total_by_split_source[key]
        print(f"  {key[0]:5s} / {key[1]:10s}: {have:6d} / {total:6d} ({have/total*100:5.1f}%)")


if __name__ == "__main__":
    build()
