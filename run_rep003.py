from __future__ import annotations

import argparse
from pathlib import Path

from src.representation import (
    REP003_SUBGROUP_FIELDS,
    build_identity_attribute_table,
    harmonise_age_band,
    identity_demographic_availability,
    identity_depth_distribution,
    identity_depth_subgroup_distribution,
    identity_depth_subgroup_summary,
    identity_depth_summary,
    identity_image_counts,
    identity_support_summary,
    load_master_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "REP-003: images-per-identity "
            "representation audit."
        )
    )

    parser.add_argument(
        "--master",
        required=True,
        help=(
            "Path to Phase 0 "
            "master_annotations parquet/csv."
        ),
    )

    parser.add_argument(
        "--config",
        default="configs/categories.yaml",
        help=(
            "Path to frozen governance "
            "category configuration."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/representation/REP-003"
        ),
        help=(
            "Directory for REP-003 outputs."
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

    # Remove stale REP-003 CSV files.
    for old_file in output_dir.glob(
        "rep003_*.csv"
    ):
        old_file.unlink()

    print("=" * 70)
    print(
        "REP-003 IMAGES-PER-IDENTITY "
        "DISTRIBUTION AUDIT"
    )
    print("=" * 70)

    print(
        f"\nInput master table:\n"
        f"{master_path.resolve()}"
    )

    # ----------------------------------------------------------
    # Load hand-off
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Reporting-only copy with governance-compliant age bands
    # ----------------------------------------------------------

    df_report = df.copy()

    df_report[
        "age_band_harmonised"
    ] = harmonise_age_band(
        df_report,
        config_path=config_path,
    )

    # ----------------------------------------------------------
    # 1. Identity support
    # ----------------------------------------------------------

    support = identity_support_summary(
        df_report
    )

    # ----------------------------------------------------------
    # 2. Images per identity
    # ----------------------------------------------------------

    counts = identity_image_counts(
        df_report
    )

    counts.to_csv(
        output_dir
        / "rep003_identity_image_counts.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 3. Overall identity-depth summary
    # ----------------------------------------------------------

    summary = identity_depth_summary(
        df_report,
        counts,
    )

    summary.to_csv(
        output_dir
        / "rep003_identity_depth_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 4. Exact identity-depth distribution
    # ----------------------------------------------------------

    distribution = (
        identity_depth_distribution(
            counts
        )
    )

    distribution.to_csv(
        output_dir
        / "rep003_identity_depth_distribution.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 5. Build identity-level demographic status table
    # ----------------------------------------------------------

    identity_table = (
        build_identity_attribute_table(
            df_report,
            fields=REP003_SUBGROUP_FIELDS,
        )
    )

    # ----------------------------------------------------------
    # 6. Record subgroup-analysis availability
    # ----------------------------------------------------------

    subgroup_status = (
        identity_demographic_availability(
            df_report,
            identity_table,
            fields=REP003_SUBGROUP_FIELDS,
        )
    )

    subgroup_status.to_csv(
        output_dir
        / "rep003_subgroup_analysis_status.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 7. Images-per-identity summaries by subgroup
    # ----------------------------------------------------------

    subgroup_summary = (
        identity_depth_subgroup_summary(
            counts,
            identity_table,
        )
    )

    subgroup_summary.to_csv(
        output_dir
        / "rep003_subgroup_depth_summary.csv",
        index=False,
    )

    subgroup_distribution = (
        identity_depth_subgroup_distribution(
            counts,
            identity_table,
        )
    )

    subgroup_distribution.to_csv(
        output_dir
        / "rep003_subgroup_depth_distribution.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    for _, row in support.iterrows():

        dataset = row["dataset"]

        expected_identities = int(
            row["n_unique_identities"]
        )

        expected_images = int(
            row["n_images_with_subject_id"]
        )

        subset = counts[
            counts["dataset"]
            == dataset
        ]

        observed_identities = len(
            subset
        )

        observed_images = int(
            subset["n_images"].sum()
        ) if not subset.empty else 0

        if (
            observed_identities
            != expected_identities
        ):
            raise RuntimeError(
                f"REP-003 identity-count mismatch "
                f"for {dataset}: "
                f"{observed_identities} != "
                f"{expected_identities}"
            )

        if (
            observed_images
            != expected_images
        ):
            raise RuntimeError(
                f"REP-003 image-count mismatch "
                f"for {dataset}: "
                f"{observed_images} != "
                f"{expected_images}"
            )

    # Distribution accounting
    for dataset, group in distribution.groupby(
        "dataset"
    ):

        n_identities = int(
            group["n_identities"].sum()
        )

        n_images = int(
            group[
                "images_contributed"
            ].sum()
        )

        expected = counts[
            counts["dataset"]
            == dataset
        ]

        if n_identities != len(expected):
            raise RuntimeError(
                f"REP-003 distribution identity "
                f"accounting failed for {dataset}."
            )

        if (
            n_images
            != int(
                expected["n_images"].sum()
            )
        ):
            raise RuntimeError(
                f"REP-003 distribution image "
                f"accounting failed for {dataset}."
            )

    # ----------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------

    print(
        "\nIDENTITY-DEPTH SUMMARY"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSUBGROUP ANALYSIS STATUS"
    )

    print(
        subgroup_status.to_string(
            index=False
        )
    )

    if subgroup_summary.empty:

        print(
            "\nNo demographic subgroup "
            "images-per-identity distributions "
            "are available in the current "
            "Phase 0 hand-off."
        )

    else:

        print(
            "\nSUBGROUP IDENTITY-DEPTH SUMMARY"
        )

        print(
            subgroup_summary.to_string(
                index=False
            )
        )

    print(
        "\nCreated:"
    )

    for path in sorted(
        output_dir.glob("*.csv")
    ):
        print(
            f"  {path}"
        )

    print(
        "\nREP-003 completed successfully."
    )


if __name__ == "__main__":
    main()


    