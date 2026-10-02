"""Step 4: integrity checks on records.csv, before any split gets frozen.

Read-only -- reports problems, changes nothing. Same philosophy the
project's own Phase 0 audits use throughout: flag and report, don't
silently fix or drop rows. Downstream split/training code decides what to
do with what's reported here.

Duplicate-group and cross_dataset_dup fields are Phase 0's own
already-computed results (exact_dup_group/near_dup_group/cross_dataset_dup
in master_annotations.parquet, carried through by records.py) -- this
script reports on them, it does not recompute duplicate detection itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
RECORDS_DIR = Path(__file__).resolve().parents[2] / "data" / "records"
REPORT_PATH = RECORDS_DIR / "integrity_report.json"

DECODE_SAMPLE_SIZE = 200
DECODE_SAMPLE_SEED = 42


def check_dataset(name: str) -> dict:
    df = pd.read_csv(RECORDS_DIR / f"{name}_records.csv")
    usable = df[df["image_path"].notna() & df["image_sha256"].notna()]

    report: dict = {
        "dataset": name,
        "n_rows": len(df),
        "n_usable": len(usable),
        "n_missing_file": int(df["image_path"].isna().sum()),
        "n_null_image_id": int(df["image_id"].isna().sum()),
    }

    sha_counts = usable["image_sha256"].value_counts()
    dup_hashes = sha_counts[sha_counts > 1]
    report["n_duplicate_hash_groups"] = int(len(dup_hashes))
    report["n_rows_in_duplicate_hash_groups"] = int(dup_hashes.sum())

    report["n_in_exact_dup_group"] = int(df["exact_dup_group"].notna().sum())
    report["n_in_near_dup_group"] = int(df["near_dup_group"].notna().sum())
    report["n_cross_dataset_dup"] = int(df["cross_dataset_dup"].astype(bool).sum())

    report["race_label_unknown_rate"] = float(df["race_label"].eq("unknown").mean())
    report["gender_label_unknown_rate"] = float(df["gender_label"].eq("unknown").mean())
    report["eligible_demographic_audit_rate"] = float(df["eligible_demographic_audit"].mean())

    n_identities = df["identity_id"].notna().sum()
    if n_identities:
        report["n_identities"] = int(df["identity_id"].nunique())
        report["images_per_identity_median"] = float(df.groupby("identity_id").size().median())
        report["images_per_identity_max"] = int(df.groupby("identity_id").size().max())

    sample = usable.sample(min(DECODE_SAMPLE_SIZE, len(usable)), random_state=DECODE_SAMPLE_SEED)
    decode_failures = []
    for _, row in sample.iterrows():
        try:
            with Image.open(REPO_ROOT / row["image_path"]) as img:
                img.verify()
        except Exception as exc:
            decode_failures.append({"image_id": row["image_id"], "error": str(exc)})
    report["decode_sample_size"] = len(sample)
    report["decode_failures"] = decode_failures

    return report


def main() -> None:
    reports = [check_dataset(name) for name in ["lfw", "fairface", "utkface"]]
    REPORT_PATH.write_text(json.dumps(reports, indent=2, default=str))

    for r in reports:
        print(f"--- {r['dataset']} ---")
        for key, value in r.items():
            if key == "decode_failures":
                continue
            print(f"  {key}: {value}")
        if r["decode_failures"]:
            print(f"  DECODE FAILURES ({len(r['decode_failures'])}): {r['decode_failures']}")
        print()

    print(f"Full report written to {REPORT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
