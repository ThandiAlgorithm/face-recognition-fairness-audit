from pathlib import Path

import pandas as pd


META = Path("metadata")
AUDIT = META / "detection_integrity_image.parquet"
MASTER = META / "master_annotations.parquet"

OUTPUT = META / "alignment_failures_by_subgroup.csv"

ATTRIBUTES = [
    "gender",
    "race",
    "age_band",
]


def summarize(group, dataset, attribute, native_group):
    detected = group[
        group["detection_status"].eq("face_detected")
    ].copy()

    n = len(detected)

    missing_primary = ~detected["primary_valid"].fillna(False)

    primary_ok = detected["primary_valid"].fillna(False)

    alignment_ok = (
        detected["project_alignment_valid"].fillna(False)
    )

    # This is a downstream alignment failure only when
    # a valid primary face is already available.
    alignment_failed = primary_ok & ~alignment_ok

    if (alignment_ok & ~primary_ok).any():
        raise RuntimeError(
            "Valid alignment without valid primary landmarks."
        )

    return {
        "dataset": dataset,
        "attribute": attribute,
        "native_group": str(native_group),

        "images_in_group": len(group),
        "faces_detected": n,

        "missing_or_invalid_primary": int(
            missing_primary.sum()
        ),

        "primary_valid": int(
            primary_ok.sum()
        ),

        "alignment_failed_after_valid_primary": int(
            alignment_failed.sum()
        ),

        "valid_alignment": int(
            alignment_ok.sum()
        ),

        "primary_failure_rate": (
            float(missing_primary.mean()) if n else None
        ),

        "alignment_failure_rate_given_valid_primary": (
            float(
                alignment_failed.sum() / primary_ok.sum()
            )
            if primary_ok.sum()
            else None
        ),

        "end_to_end_failure_rate_given_detection": (
            float((~alignment_ok).mean()) if n else None
        ),
    }


def main():
    audit = pd.read_parquet(AUDIT)
    master = pd.read_parquet(MASTER)

    if len(audit) != 135039:
        raise RuntimeError("Expected full DET-001 image table.")

    if not audit["image_id"].is_unique:
        raise RuntimeError("Duplicate audit image IDs.")

    if not master["image_id"].is_unique:
        raise RuntimeError("Duplicate master image IDs.")

    labels = [
        attribute
        for attribute in ATTRIBUTES
        if attribute in master.columns
    ]

    merged = audit.merge(
        master[["image_id", *labels]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    raw = merged[
        merged["route"].eq("project_retinaface")
        & ~merged["corrupt"].fillna(False)
    ].copy()

    rows = []

    for dataset, part in raw.groupby("dataset"):
        rows.append(
            summarize(part, dataset, "ALL", "ALL")
        )

        for attribute in labels:
            subgroup = part.copy()

            subgroup["native_group"] = (
                subgroup[attribute]
                .astype("string")
                .fillna("unknown")
                .replace("", "unknown")
            )

            for name, group in subgroup.groupby(
                "native_group",
                dropna=False,
            ):
                rows.append(
                    summarize(
                        group,
                        dataset,
                        attribute,
                        name,
                    )
                )

    result = pd.DataFrame(rows)

    # Reconciliation checks.
    assert (
        result["missing_or_invalid_primary"]
        + result["primary_valid"]
        == result["faces_detected"]
    ).all()

    assert (
        result["alignment_failed_after_valid_primary"]
        + result["valid_alignment"]
        == result["primary_valid"]
    ).all()

    # The original no-detection count remains in DET-001,
    # not in the landmark-failure denominator.
    result["thin_cell"] = result["faces_detected"] < 10

    result.to_csv(OUTPUT, index=False)

    print("\nDATASET-WIDE LANDMARK/ALIGNMENT RESULTS")
    print(
        result[result["attribute"].eq("ALL")]
        .to_string(index=False)
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
