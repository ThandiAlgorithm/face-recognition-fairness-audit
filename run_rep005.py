from __future__ import annotations

import argparse
from pathlib import Path

from src.representation import (
    load_master_table,
)

from src.representation_base_rates import (
    load_target_coding,
    overall_base_rate,
    prepare_rep005_data,
    rep005_analysis_coverage,
    subgroup_base_rate,
    target_coding_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "REP-005: base-rate tables "
            "for the frozen binary classification target."
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
            "results/representation/REP-005"
        ),
        help=(
            "Directory for REP-005 outputs."
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

    for old_file in output_dir.glob(
        "rep005_*.csv"
    ):
        old_file.unlink()

    print("=" * 70)
    print(
        "REP-005 BASE-RATE AUDIT"
    )
    print("=" * 70)

    print(
        f"\nInput master table:\n"
        f"{master_path.resolve()}"
    )

    # ----------------------------------------------------------
    # Load inputs
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
    # Frozen target coding
    # ----------------------------------------------------------

    coding = load_target_coding(
        category_config
    )

    print(
        "\nFROZEN TARGET CODING"
    )

    print(
        f"  Target: "
        f"{coding['target_field']}"
    )

    print(
        f"  Y=1: "
        f"{coding['positive_class']}"
    )

    print(
        "  Y=0: "
        + ", ".join(
            coding[
                "negative_classes"
            ]
        )
    )

    coding_table = (
        target_coding_table(
            coding
        )
    )

    coding_table.to_csv(
        output_dir
        / "rep005_target_coding.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Prepare reporting-only dataframe
    # ----------------------------------------------------------

    work = prepare_rep005_data(
        df,
        category_config_path=
            category_config,
    )

    # ----------------------------------------------------------
    # Coverage / denominator accounting
    # ----------------------------------------------------------

    coverage = (
        rep005_analysis_coverage(
            work,
            coding,
        )
    )

    coverage.to_csv(
        output_dir
        / "rep005_analysis_coverage.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Overall base rate
    # ----------------------------------------------------------

    overall = overall_base_rate(
        work,
        coding,
        reporting_config_path=
            reporting_config,
    )

    overall.to_csv(
        output_dir
        / "rep005_overall_base_rate.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # P(Y=1 | race=g)
    # ----------------------------------------------------------

    race = subgroup_base_rate(
        work,
        coding,
        subgroup_axis="race",
        reporting_config_path=
            reporting_config,
    )

    race.to_csv(
        output_dir
        / "rep005_base_rate_by_race.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # P(Y=1 | age=g)
    # ----------------------------------------------------------

    age = subgroup_base_rate(
        work,
        coding,
        subgroup_axis="age",
        reporting_config_path=
            reporting_config,
    )

    age.to_csv(
        output_dir
        / "rep005_base_rate_by_age.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Validation — binary accounting
    # ----------------------------------------------------------

    for table_name, table in [
        (
            "overall",
            overall,
        ),
        (
            "race",
            race,
        ),
        (
            "age",
            age,
        ),
    ]:
        if table.empty:
            continue

        available = table[
            table["n"].astype(int)
            > 0
        ]

        for _, row in available.iterrows():
            n = int(
                row["n"]
            )

            accounted = (
                int(row["n_y1"])
                + int(row["n_y0"])
            )

            if accounted != n:
                raise RuntimeError(
                    f"REP-005 binary accounting "
                    f"failed in {table_name}: "
                    f"{accounted} != {n}"
                )

    # ----------------------------------------------------------
    # Validation — subgroup denominators
    # ----------------------------------------------------------

    table_lookup = {
        "race":
            race,
        "age":
            age,
    }

    for _, row in coverage.iterrows():
        if (
            row["analysis_status"]
            != "available"
        ):
            continue

        dataset = row[
            "dataset"
        ]

        subgroup_axis = row[
            "subgroup_axis"
        ]

        expected = int(
            row[
                "n_complete_cases"
            ]
        )

        table = table_lookup[
            subgroup_axis
        ]

        observed = int(
            table.loc[
                table["dataset"]
                == dataset,
                "n",
            ]
            .astype(int)
            .sum()
        )

        if observed != expected:
            raise RuntimeError(
                f"REP-005 denominator mismatch for "
                f"{dataset}/{subgroup_axis}: "
                f"{observed} != {expected}"
            )

    # ----------------------------------------------------------
    # Console output
    # ----------------------------------------------------------

    print(
        "\nANALYSIS COVERAGE"
    )

    print(
        coverage.to_string(
            index=False
        )
    )

    print(
        "\nOVERALL BASE RATE"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print(
        "\nBASE RATE BY RACE"
    )

    if race.empty:
        print(
            "No valid race-conditioned "
            "base rates available."
        )
    else:
        print(
            race.to_string(
                index=False
            )
        )

    print(
        "\nBASE RATE BY AGE"
    )

    if age.empty:
        print(
            "No valid age-conditioned "
            "base rates available."
        )
    else:
        print(
            age.to_string(
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
        "\nREP-005 completed successfully."
    )


if __name__ == "__main__":
    main()

    