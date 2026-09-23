from pathlib import Path

import pandas as pd


MASTER_PATH = Path("metadata/master_annotations.parquet")
SPLIT_PATH = Path("metadata/utkface_split.csv")

EXPECTED_ROWS = 24108

EXPECTED_COUNTS = {
    "train": 16876,
    "val": 3616,
    "test": 3616,
}


def report_group_leakage(df, group_col, label):
    grouped = (
        df.dropna(subset=[group_col])
        .groupby(group_col)["split"]
        .agg(lambda x: sorted(set(x)))
    )

    leaking = grouped[
        grouped.map(len) > 1
    ]

    print()
    print(label)
    print("-" * 60)
    print(
        "Groups represented:",
        len(grouped),
    )
    print(
        "Groups crossing splits:",
        len(leaking),
    )

    if len(leaking):
        print("\nFirst leaking groups:")
        print(
            leaking.head(20)
            .to_string()
        )

    return leaking


def main():

    master = pd.read_parquet(MASTER_PATH)
    split = pd.read_csv(SPLIT_PATH)

    print("=" * 70)
    print("UTKFACE SPLIT VALIDATION")
    print("=" * 70)

    # Canonical UTKFace population

    utk = master[
        master["dataset"]
        .astype(str)
        .str.lower()
        .eq("utkface")
    ].copy()

    print("Master UTKFace rows:", len(utk))
    print("Split-file rows:", len(split))

    if len(utk) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} UTKFace master rows, "
            f"found {len(utk)}"
        )

    if len(split) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} split rows, "
            f"found {len(split)}"
        )

    # Required columns

    required = {
        "image_id",
        "split",
    }

    missing = required - set(split.columns)

    if missing:
        raise RuntimeError(
            f"Split file missing columns: {sorted(missing)}"
        )

    # image_id integrity

    print(
        "Null split image_id:",
        split["image_id"].isna().sum(),
    )

    print(
        "Duplicate split image_id:",
        split["image_id"].duplicated().sum(),
    )

    if split["image_id"].isna().any():
        raise RuntimeError(
            "Split file contains null image_id"
        )

    if split["image_id"].duplicated().any():
        raise RuntimeError(
            "Split file contains duplicate image_id"
        )

    master_ids = set(utk["image_id"])
    split_ids = set(split["image_id"])

    missing_from_split = (
        master_ids - split_ids
    )

    extra_in_split = (
        split_ids - master_ids
    )

    print(
        "Master IDs missing from split:",
        len(missing_from_split),
    )

    print(
        "Unknown IDs in split:",
        len(extra_in_split),
    )

    if missing_from_split:
        raise RuntimeError(
            "Some canonical UTKFace images are missing "
            "from the split file"
        )

    if extra_in_split:
        raise RuntimeError(
            "Split file contains non-canonical/unknown IDs"
        )

    # Allowed split labels and counts

    split["split"] = (
        split["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    counts = (
        split["split"]
        .value_counts()
        .to_dict()
    )

    print("\nSplit counts:")
    print(
        split["split"]
        .value_counts()
        .to_string()
    )

    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            "Unexpected UTKFace split counts.\n"
            f"Expected: {EXPECTED_COUNTS}\n"
            f"Actual:   {counts}"
        )

    # Verify master table has exactly the same assignment

    current = utk[
        [
            "image_id",
            "split",
            "exact_dup_group",
            "near_dup_group",
            "near_dup_status",
        ]
    ].copy()

    current = current.rename(
        columns={
            "split": "master_split"
        }
    )

    merged = current.merge(
        split,
        on="image_id",
        how="outer",
        validate="one_to_one",
    )

    mismatch = (
        merged["master_split"]
        .astype(str)
        .str.lower()
        != merged["split"]
    )

    print(
        "\nMaster/split assignment mismatches:",
        int(mismatch.sum()),
    )

    if mismatch.any():
        print(
            merged.loc[
                mismatch,
                [
                    "image_id",
                    "master_split",
                    "split",
                ],
            ]
            .head(20)
            .to_string(index=False)
        )

        raise RuntimeError(
            "Master UTKFace split assignments do not "
            "match metadata/utkface_split.csv"
        )

    # Duplicate leakage audit

    audit = merged.copy()

    exact_leaks = report_group_leakage(
        audit,
        "exact_dup_group",
        "EXACT DUPLICATE SPLIT AUDIT",
    )

    near_leaks = report_group_leakage(
        audit,
        "near_dup_group",
        "CONFIRMED NEAR-DUPLICATE SPLIT AUDIT",
    )

    # Final result

    print()
    print("=" * 70)

    if len(exact_leaks) == 0 and len(near_leaks) == 0:
        print("UTKFACE SPLIT VALIDATION PASSED")
    else:
        print("UTKFACE SPLIT STRUCTURE VALID, BUT LEAKAGE FOUND")

    print("=" * 70)

    print("Rows:", len(split))
    print("Train:", counts["train"])
    print("Val:", counts["val"])
    print("Test:", counts["test"])
    print(
        "Exact duplicate groups crossing splits:",
        len(exact_leaks),
    )
    print(
        "Confirmed near-duplicate groups crossing splits:",
        len(near_leaks),
    )


if __name__ == "__main__":
    main()
