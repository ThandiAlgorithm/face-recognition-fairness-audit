"""M0 spec section 4: audit LFW identity frequency before training.

Reuses the existing frozen, identity-disjoint LFW split (experiments/lfw/
split_manifest.csv, built by src/datasets/splits.py and already used by
M1) rather than generating a new one -- it already satisfies every
requirement spec section 5 asks for (verified empirically: 0 identities
span more than one split, ~80/10/10 split by identity). Rebuilding it
would risk a DIFFERENT identity partition from M1's, which would make the
"M0 is the missing recognition component of the same investigation"
framing harder to defend, not easier.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_m0_table

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "M0_LFW"


def _bucket(n: int) -> str:
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 4:
        return "3-4"
    if n <= 9:
        return "5-9"
    return "10+"


def audit_split(df, split: str) -> dict:
    sub = df[df["split"] == split]
    counts = Counter(sub["identity_id"])
    values = list(counts.values())
    buckets = Counter(_bucket(v) for v in values)
    return {
        "split": split,
        "n_images": int(len(sub)),
        "n_identities": int(len(counts)),
        "median_images_per_identity": statistics.median(values),
        "mean_images_per_identity": round(statistics.mean(values), 3),
        "max_images_per_identity": max(values),
        "identities_with_1_image": buckets.get("1", 0),
        "identities_with_2_images": buckets.get("2", 0),
        "identities_with_3_4_images": buckets.get("3-4", 0),
        "identities_with_5_9_images": buckets.get("5-9", 0),
        "identities_with_10plus_images": buckets.get("10+", 0),
        "identities_genuine_pair_eligible": sum(1 for v in values if v >= 2),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_m0_table()

    rows = [audit_split(df, s) for s in ["train", "val", "test"]]

    import csv
    with open(OUT_DIR / "data_audit.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_images": int(len(df)),
        "total_identities": int(df["identity_id"].nunique()),
        "per_split": rows,
        "reused_existing_split": True,
        "split_source": "experiments/lfw/split_manifest.csv",
    }
    (OUT_DIR / "data_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
