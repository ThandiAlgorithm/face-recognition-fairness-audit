from __future__ import annotations

import argparse
from pathlib import Path

from src.representation import (
    load_master_table,
)

from src.representation_missingness import (
    affected_analysis_summary,
    demographic_missingness_patterns,
    label_missingness_summary,
    missingness_parse_failure_summary,
    prepare_rep006_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "REP-006: missing and unknown "
            "demographic-label audit."
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
        "--output",
        default=(
            "results/representation/REP-006"
        ),
        help=(
            "Directory for REP-006 outputs."
        ),
    )

    args = parser.parse_args()

    master_path = Path(
        args.master
    )

    category_config = Path(
        args.category_config
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_file in output_dir.glob(
        "rep006_*.csv"
    ):
        old_file.unlink()

    print("=" * 70)
    print(
        "REP-006 MISSING / UNKNOWN LABEL AUDIT"
    )
    print("=" * 70)

    print(
        f"\nInput master table:\n"
        f"{master_path.resolve()}"
    )

    # ----------------------------------------------------------
    # Load Phase 0 hand-off
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
    # Reporting-only preparation
    # ----------------------------------------------------------

    work = prepare_rep006_data(
        df,
        category_config_path=
            category_config,
    )

    # ----------------------------------------------------------
    # 1. Label missingness
    # ----------------------------------------------------------

    missingness = (
        label_missingness_summary(
            work
        )
    )

    missingness.to_csv(
        output_dir
        / "rep006_label_missingness.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 2. Joint missingness patterns
    # ----------------------------------------------------------

    patterns = (
        demographic_missingness_patterns(
            work
        )
    )

    patterns.to_csv(
        output_dir
        / "rep006_missingness_patterns.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 3. Affected analyses
    # ----------------------------------------------------------

    affected = (
        affected_analysis_summary(
            work
        )
    )

    affected.to_csv(
        output_dir
        / "rep006_affected_analyses.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 4. Relationship with parse failures
    # ----------------------------------------------------------

    parse_relation = (
        missingness_parse_failure_summary(
            work
        )
    )

    parse_relation.to_csv(
        output_dir
        / "rep006_missingness_vs_parse_failure.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Validation: label counts
    # ----------------------------------------------------------

    for _, row in missingness.iterrows():
        total = int(
            row["n_total"]
        )

        accounted = (
            int(row["n_available"])
            + int(
                row[
                    "n_missing_or_unknown"
                ]
            )
        )

        if accounted != total:
            raise RuntimeError(
                "REP-006 label accounting failed "
                f"for {row['dataset']} / "
                f"{row['label']}."
            )

    # ----------------------------------------------------------
    # Validation: missingness pattern totals
    # ----------------------------------------------------------

    for dataset, group in patterns.groupby(
        "dataset"
    ):
        observed = int(
            group[
                "n_images"
            ].sum()
        )

        expected = int(
            (
                work[
                    "dataset"
                ]
                == dataset
            ).sum()
        )

        if observed != expected:
            raise RuntimeError(
                "REP-006 missingness-pattern "
                f"accounting failed for {dataset}: "
                f"{observed} != {expected}"
            )

    # ----------------------------------------------------------
    # Validation: affected-analysis denominators
    # ----------------------------------------------------------

    for _, row in affected.iterrows():
        total = int(
            row[
                "n_total_images"
            ]
        )

        accounted = (
            int(
                row[
                    "n_complete_label_rows"
                ]
            )
            + int(
                row[
                    "n_incomplete_label_rows"
                ]
            )
        )

        if accounted != total:
            raise RuntimeError(
                "REP-006 affected-analysis "
                f"accounting failed for "
                f"{row['dataset']} / "
                f"{row['analysis']}."
            )

    # ----------------------------------------------------------
    # Console output
    # ----------------------------------------------------------

    print(
        "\nLABEL MISSINGNESS"
    )

    print(
        missingness.to_string(
            index=False
        )
    )

    print(
        "\nMISSINGNESS VS PARSE FAILURE"
    )

    print(
        parse_relation.to_string(
            index=False
        )
    )

    print(
        "\nAFFECTED ANALYSES"
    )

    print(
        affected.to_string(
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
        "\nREP-006 completed successfully."
    )


if __name__ == "__main__":
    main()
    