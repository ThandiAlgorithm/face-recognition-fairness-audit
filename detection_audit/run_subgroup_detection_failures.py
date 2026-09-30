from pathlib import Path

import pandas as pd


META = Path("metadata")
IMAGE_AUDIT = META / "detection_integrity_image.parquet"
MASTER = META / "master_annotations.parquet"
OUTPUT = META / "subgroup_detection_failures.csv"

EXPECTED_TOTAL = 135039

FAILURES = {
    "no_detection",
    "inference_failure",
    "pre_inference_or_colour_failure",
}

SUCCESS = "face_detected"

ATTRIBUTES = [
    "gender",
    "race",
    "age_band",
]


def main():
    audit = pd.read_parquet(IMAGE_AUDIT)
    master = pd.read_parquet(MASTER)

    if len(audit) != EXPECTED_TOTAL:
        raise RuntimeError("DET-001 audit has an unexpected row count.")

    if not audit["image_id"].is_unique:
        raise RuntimeError("Duplicate image IDs in DET-001.")

    if not master["image_id"].is_unique:
        raise RuntimeError("Duplicate image IDs in the master table.")

    available = [
        attribute
        for attribute in ATTRIBUTES
        if attribute in master.columns
    ]

    native_labels = master[
        ["image_id", *available]
    ].copy()

    df = audit.merge(
        native_labels,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    # FairFace is excluded because RetinaFace was not run
    # on the publisher-preprocessed images.
    eligible = df[
        df["route"].eq("project_retinaface")
        & ~df["corrupt"].fillna(False)
    ].copy()

    valid_outcomes = FAILURES | {SUCCESS}

    invalid = ~eligible["detection_status"].isin(valid_outcomes)

    if invalid.any():
        raise RuntimeError(
            "Some eligible images have unexplained outcomes: "
            + str(
                eligible.loc[
                    invalid,
                    "detection_status",
                ].value_counts().to_dict()
            )
        )

    eligible["failed"] = (
        eligible["detection_status"].isin(FAILURES)
    )

    eligible["detected"] = (
        eligible["detection_status"].eq(SUCCESS)
    )

    print("Eligible project-detection images:", len(eligible))

    rows = []

    for dataset, dataset_rows in eligible.groupby("dataset"):

        # Dataset-wide result, for comparison with DET-001.
        n = len(dataset_rows)

        rows.append({
            "dataset": dataset,
            "attribute": "ALL",
            "native_group": "ALL",
            "eligible_images": n,
            "detected": int(dataset_rows["detected"].sum()),
            "failures": int(dataset_rows["failed"].sum()),
            "failure_rate": float(dataset_rows["failed"].mean()),
            "no_detection": int(
                dataset_rows["detection_status"].eq(
                    "no_detection"
                ).sum()
            ),
            "inference_failure": int(
                dataset_rows["detection_status"].eq(
                    "inference_failure"
                ).sum()
            ),
            "pre_inference_or_colour_failure": int(
                dataset_rows["detection_status"].eq(
                    "pre_inference_or_colour_failure"
                ).sum()
            ),
        })

        for attribute in available:

            part = dataset_rows.copy()

            # Do not invent missing demographic labels.
            part["native_group"] = (
                part[attribute]
                .astype("string")
                .fillna("unknown")
            )

            part["native_group"] = (
                part["native_group"]
                .replace("", "unknown")
            )

            for group, group_rows in part.groupby(
                "native_group",
                dropna=False,
            ):
                n = len(group_rows)
                failures = int(group_rows["failed"].sum())

                rows.append({
                    "dataset": dataset,
                    "attribute": attribute,
                    "native_group": str(group),
                    "eligible_images": n,
                    "detected": int(group_rows["detected"].sum()),
                    "failures": failures,
                    "failure_rate": failures / n,
                    "no_detection": int(
                        group_rows["detection_status"].eq(
                            "no_detection"
                        ).sum()
                    ),
                    "inference_failure": int(
                        group_rows["detection_status"].eq(
                            "inference_failure"
                        ).sum()
                    ),
                    "pre_inference_or_colour_failure": int(
                        group_rows["detection_status"].eq(
                            "pre_inference_or_colour_failure"
                        ).sum()
                    ),
                })

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError("No subgroup results produced.")

    # Every row must reconcile.
    assert (
        result["detected"] + result["failures"]
        == result["eligible_images"]
    ).all()

    # Avoid presenting tiny cells as stable estimates.
    result["thin_cell"] = result["eligible_images"] < 10

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("\nDataset-wide detection outcomes:")
    print(
        result[result["attribute"].eq("ALL")]
        .to_string(index=False)
    )

    print("\nUnknown-label groups are retained.")
    print("Rows in subgroup table:", len(result))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
