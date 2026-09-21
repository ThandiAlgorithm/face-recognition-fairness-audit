from __future__ import annotations

import argparse
from pathlib import Path

from src.representation import (
    REP001_FIELDS,
    audit_flag_summary,
    composition_table,
    dataset_summary,
    demographic_availability_table,
    harmonise_age_band,
    load_master_table,
    provenance_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "REP-001: image-level representation "
            "composition audit."
        )
    )

    parser.add_argument(
        "--master",
        required=True,
        help=(
            "Path to the Phase 0 "
            "master_annotations parquet/csv."
        ),
    )

    parser.add_argument(
        "--config",
        default="configs/categories.yaml",
        help=(
            "Path to the frozen governance "
            "category configuration."
        ),
    )

    parser.add_argument(
        "--output",
        default="results/representation/REP-001",
        help=(
            "Directory for REP-001 outputs."
        ),
    )

    args = parser.parse_args()

    master_path = Path(
        args.master
    )

    config_path = Path(
        args.config
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Remove only previous REP-001 generated CSV outputs.
    # This prevents stale files from earlier runs being mistaken
    # for current results.
    # --------------------------------------------------------------

    for old_file in output_dir.glob(
        "rep001_*.csv"
    ):
        old_file.unlink()

    print("=" * 70)
    print(
        "REP-001 IMAGE-LEVEL "
        "REPRESENTATION AUDIT"
    )
    print("=" * 70)

    print(
        f"\nInput master table:\n"
        f"{master_path.resolve()}"
    )

    print(
        f"\nGovernance config:\n"
        f"{config_path.resolve()}"
    )

    # --------------------------------------------------------------
    # Load Phase 0 hand-off
    # --------------------------------------------------------------

    df = load_master_table(
        master_path
    )

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Unique image IDs: "
        f"{df['image_id'].nunique():,}"
    )

    # --------------------------------------------------------------
    # Reporting-only copy
    # --------------------------------------------------------------

    df_report = df.copy()

    df_report[
        "age_band_harmonised"
    ] = harmonise_age_band(
        df_report,
        config_path=config_path,
    )

    # --------------------------------------------------------------
    # 1. Dataset-level composition
    # --------------------------------------------------------------

    summary = dataset_summary(
        df
    )

    summary.to_csv(
        output_dir
        / "rep001_dataset_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # 2. Native/source demographic composition
    # --------------------------------------------------------------

    available_fields = [
        field
        for field in REP001_FIELDS
        if field in df.columns
    ]

    for field in available_fields:

        table = composition_table(
            df,
            field,
        )

        table.to_csv(
            output_dir
            / f"rep001_{field}_composition.csv",
            index=False,
        )

    # --------------------------------------------------------------
    # 3. Harmonised age composition
    # --------------------------------------------------------------

    harmonised_age = composition_table(
        df_report,
        "age_band_harmonised",
    )

    harmonised_age.to_csv(
        output_dir
        / (
            "rep001_age_band_"
            "harmonised_composition.csv"
        ),
        index=False,
    )

    # --------------------------------------------------------------
    # 4. Demographic availability
    # --------------------------------------------------------------

    availability = (
        demographic_availability_table(
            df
        )
    )

    availability.to_csv(
        output_dir
        / "rep001_demographic_availability.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # 5. Demographic provenance
    # --------------------------------------------------------------

    provenance = provenance_summary(
        df
    )

    provenance.to_csv(
        output_dir
        / "rep001_demographic_provenance.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # 6. Phase 0 audit flags
    # --------------------------------------------------------------

    flags = audit_flag_summary(
        df
    )

    flags.to_csv(
        output_dir
        / "rep001_phase0_flags.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # 7. Input schema snapshot
    # --------------------------------------------------------------

    schema = (
        df.dtypes
        .astype(str)
        .rename("dtype")
        .reset_index()
        .rename(
            columns={
                "index": "column"
            }
        )
    )

    schema.to_csv(
        output_dir
        / "rep001_input_schema.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------------

    if summary["n_images"].sum() != len(df):
        raise RuntimeError(
            "Dataset-summary counts do not "
            "sum to master-table row count."
        )

    expected_datasets = {
        "fairface",
        "lfw",
        "utkface",
    }

    actual_datasets = set(
        df["dataset"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    unexpected = (
        actual_datasets
        - expected_datasets
    )

    if unexpected:
        print(
            "\nWARNING: Unexpected dataset labels found:"
        )

        for dataset in sorted(
            unexpected
        ):
            print(
                f"  - {dataset}"
            )

    # --------------------------------------------------------------
    # Print generated outputs
    # --------------------------------------------------------------

    print("\nCreated:")

    for path in sorted(
        output_dir.glob("*.csv")
    ):
        print(
            f"  {path}"
        )

    print(
        "\nREP-001 completed successfully."
    )


if __name__ == "__main__":
    main()