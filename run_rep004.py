from __future__ import annotations

import argparse
from pathlib import Path

from src.representation import (
    REP004_INTERSECTIONS,
    all_intersectional_composition,
    demographic_eligibility_summary,
    harmonise_age_band,
    intersection_reporting_summary,
    load_image_cell_thresholds,
    load_master_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "REP-004: intersectional "
            "representation composition audit."
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
        "--category-config",
        default="configs/categories.yaml",
        help=(
            "Frozen GOV-002 category configuration."
        ),
    )

    parser.add_argument(
        "--reporting-config",
        default="configs/reporting.yaml",
        help=(
            "Frozen GOV-003 reporting configuration."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "results/representation/REP-004"
        ),
        help=(
            "Directory for REP-004 outputs."
        ),
    )

    args = parser.parse_args()

    master_path = Path(
        args.master
    )

    category_config = Path(
        args.category_config
    )

    reporting_config = Path(
        args.reporting_config
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove stale REP-004 CSV files.
    for old_file in output_dir.glob(
        "rep004_*.csv"
    ):
        old_file.unlink()

    print("=" * 70)
    print(
        "REP-004 INTERSECTIONAL "
        "REPRESENTATION AUDIT"
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
    # Reporting-only harmonised age
    # ----------------------------------------------------------

    df_report = df.copy()

    df_report[
        "age_band_harmonised"
    ] = harmonise_age_band(
        df_report,
        config_path=category_config,
    )

    # ----------------------------------------------------------
    # Frozen thresholds
    # ----------------------------------------------------------

    thresholds = (
        load_image_cell_thresholds(
            reporting_config
        )
    )

    print(
        "\nFrozen image-cell thresholds:"
    )

    print(
        "  Suppress below:",
        thresholds[
            "suppress_below"
        ],
    )

    print(
        "  Low confidence below:",
        thresholds[
            "low_confidence_below"
        ],
    )

    # ----------------------------------------------------------
    # 1. Demographic analysis eligibility
    # ----------------------------------------------------------

    eligibility = (
        demographic_eligibility_summary(
            df_report
        )
    )

    eligibility.to_csv(
        output_dir
        / "rep004_demographic_eligibility.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 2. All pre-specified intersections
    # ----------------------------------------------------------

    composition = (
        all_intersectional_composition(
            df_report,
            reporting_config_path=
                reporting_config,
        )
    )

    composition.to_csv(
        output_dir
        / "rep004_intersectional_composition.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 3. Separate tables for readability
    # ----------------------------------------------------------

    for intersection_name in (
        REP004_INTERSECTIONS
    ):
        subset = composition[
            composition[
                "intersection"
            ]
            == intersection_name
        ]

        subset.to_csv(
            output_dir
            / (
                f"rep004_{intersection_name}.csv"
            ),
            index=False,
        )

    # ----------------------------------------------------------
    # 4. Reporting-status summary
    # ----------------------------------------------------------

    status_summary = (
        intersection_reporting_summary(
            composition
        )
    )

    status_summary.to_csv(
        output_dir
        / "rep004_reporting_status_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    eligibility_lookup = (
        eligibility.set_index(
            "dataset"
        )[
            "n_eligible_images"
        ]
        .to_dict()
    )

    for (
        dataset,
        intersection,
    ), group in composition.groupby(
        [
            "dataset",
            "intersection",
        ]
    ):
        observed = int(
            group[
                "n_images"
            ].sum()
        )

        expected = int(
            eligibility_lookup[
                dataset
            ]
        )

        if observed != expected:
            raise RuntimeError(
                f"REP-004 accounting mismatch for "
                f"{dataset}/{intersection}: "
                f"{observed} != {expected}"
            )

        pct_total = float(
            group[
                "percentage"
            ].sum()
        )

        if expected > 0 and not (
            99.999
            <= pct_total
            <= 100.001
        ):
            raise RuntimeError(
                f"REP-004 percentages do not sum "
                f"to 100 for "
                f"{dataset}/{intersection}: "
                f"{pct_total}"
            )

    # ----------------------------------------------------------
    # Console summary
    # ----------------------------------------------------------

    print(
        "\nDEMOGRAPHIC ELIGIBILITY"
    )

    print(
        eligibility.to_string(
            index=False
        )
    )

    print(
        "\nINTERSECTION REPORTING STATUS"
    )

    print(
        status_summary.to_string(
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
        "\nREP-004 completed successfully."
    )


if __name__ == "__main__":
    main()
    