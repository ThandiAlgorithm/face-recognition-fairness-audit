from __future__ import annotations

import argparse
from pathlib import Path

from src.representation import (
    REP002_DEMOGRAPHIC_FIELDS,
    build_identity_attribute_table,
    identity_demographic_availability,
    identity_demographic_composition,
    identity_support_summary,
    load_master_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REP-002: identity-level representation audit."
    )

    parser.add_argument(
        "--master",
        required=True,
        help="Path to Phase 0 master_annotations parquet/csv.",
    )

    parser.add_argument(
        "--output",
        default="results/representation/REP-002",
        help="Directory for REP-002 outputs.",
    )

    args = parser.parse_args()

    master_path = Path(args.master)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove stale REP-002 CSV outputs from previous runs.
    for old_file in output_dir.glob("rep002_*.csv"):
        old_file.unlink()

    print("=" * 70)
    print("REP-002 IDENTITY-LEVEL REPRESENTATION AUDIT")
    print("=" * 70)

    print(f"\nInput master table:\n{master_path.resolve()}")

    # ----------------------------------------------------------
    # Load Phase 0 hand-off
    # ----------------------------------------------------------

    df = load_master_table(master_path)

    print(f"\nRows: {len(df):,}")
    print(f"Unique image IDs: {df['image_id'].nunique():,}")

    # ----------------------------------------------------------
    # 1. Identity support
    # ----------------------------------------------------------

    support = identity_support_summary(df)

    support.to_csv(
        output_dir / "rep002_identity_support.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 2. Build identity-level demographic status table
    # ----------------------------------------------------------

    identity_table = build_identity_attribute_table(
        df,
        fields=REP002_DEMOGRAPHIC_FIELDS,
    )

    # ----------------------------------------------------------
    # 3. Demographic availability at identity level
    # ----------------------------------------------------------

    availability = identity_demographic_availability(
        df,
        identity_table,
        fields=REP002_DEMOGRAPHIC_FIELDS,
    )

    availability.to_csv(
        output_dir
        / "rep002_identity_demographic_availability.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 4. Identity-level demographic composition
    # ----------------------------------------------------------

    composition = identity_demographic_composition(
        identity_table
    )

    composition.to_csv(
        output_dir
        / "rep002_identity_demographic_composition.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    for _, row in availability.iterrows():
        total = int(row["n_total_identities"])

        if total == 0:
            continue

        accounted = (
            int(row["n_consistent"])
            + int(row["n_unavailable"])
            + int(row["n_conflicting"])
        )

        if accounted != total:
            raise RuntimeError(
                f"Identity accounting failed for "
                f"{row['dataset']} / {row['field']}"
            )

    # ----------------------------------------------------------
    # Console output
    # ----------------------------------------------------------

    print("\nIDENTITY SUPPORT")
    print(support.to_string(index=False))

    print("\nIDENTITY DEMOGRAPHIC AVAILABILITY")
    print(availability.to_string(index=False))

    if composition.empty:
        print(
            "\nNo consistently labelled identity-level "
            "demographic composition is available "
            "in the current hand-off."
        )
    else:
        print("\nIDENTITY DEMOGRAPHIC COMPOSITION")
        print(composition.to_string(index=False))

    print("\nCreated:")

    for path in sorted(output_dir.glob("*.csv")):
        print(f"  {path}")

    print("\nREP-002 completed successfully.")


if __name__ == "__main__":
    main()