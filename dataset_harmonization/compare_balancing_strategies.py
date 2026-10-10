from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "candidate_taxonomy_v2"
    / "candidate_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "balancing_scenarios"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TAXONOMY GROUPS
# ============================================================

ALL_TRAINABLE = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino",
    "Asian",
]

EXPLICIT_FINE_GROUPS = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino",
]

NON_TARGET_GROUPS = [
    "Other",
    "Unknown",
]


def load_data() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    return pd.read_csv(
        INPUT,
        low_memory=False,
    )


def get_counts(
    df: pd.DataFrame,
) -> pd.Series:

    return (
        df["candidate_demographic_label"]
        .value_counts()
    )


def scenario_strict_all(
    counts: pd.Series,
) -> pd.Series:
    """
    Equal count across all usable demographic groups.

    No oversampling.
    Therefore the smallest category determines the size.
    """

    minimum = min(
        counts.get(group, 0)
        for group in ALL_TRAINABLE
    )

    return pd.Series(
        {
            group: minimum
            for group in ALL_TRAINABLE
        },
        name="strict_parity_all",
    )


def scenario_strict_explicit(
    counts: pd.Series,
) -> pd.Series:
    """
    Equal count using only categories with explicit fine-grained
    source labels.

    Broad Asian, Other and Unknown are not destroyed; they are
    retained in the full manifest but excluded from this candidate
    training subset.
    """

    minimum = min(
        counts.get(group, 0)
        for group in EXPLICIT_FINE_GROUPS
    )

    return pd.Series(
        {
            group: minimum
            for group in EXPLICIT_FINE_GROUPS
        },
        name="strict_parity_explicit",
    )


def scenario_volume_aware(
    counts: pd.Series,
) -> pd.Series:
    """
    Reduce dominance without forcing full parity.

    Cap very large explicit groups at the median group size.
    Smaller groups keep all available unique examples.

    No duplication / oversampling.
    """

    available = pd.Series(
        {
            group: counts.get(group, 0)
            for group in EXPLICIT_FINE_GROUPS
        }
    )

    cap = int(
        available.median()
    )

    target = available.clip(
        upper=cap
    )

    target.name = "volume_aware_cap"

    return target


def build_comparison(
    counts: pd.Series,
) -> pd.DataFrame:

    current = pd.Series(
        {
            group: counts.get(group, 0)
            for group in ALL_TRAINABLE
        },
        name="current_pool",
    )

    strict_all = scenario_strict_all(
        counts
    )

    strict_explicit = scenario_strict_explicit(
        counts
    )

    volume = scenario_volume_aware(
        counts
    )

    table = pd.concat(
        [
            current,
            strict_all,
            strict_explicit,
            volume,
        ],
        axis=1,
    ).fillna(0)

    table = table.astype(int)

    table.index.name = (
        "candidate_demographic_label"
    )

    return table


def build_summary(
    comparison: pd.DataFrame,
    full_n: int,
) -> pd.DataFrame:

    rows = []

    for column in comparison.columns:

        total = int(
            comparison[column].sum()
        )

        nonzero = comparison[
            column
        ][comparison[column] > 0]

        minimum = (
            int(nonzero.min())
            if len(nonzero)
            else 0
        )

        maximum = (
            int(nonzero.max())
            if len(nonzero)
            else 0
        )

        ratio = (
            maximum / minimum
            if minimum
            else float("nan")
        )

        rows.append(
            {
                "scenario": column,
                "n_images": total,
                "percent_of_full_pool": (
                    total
                    / full_n
                    * 100
                ),
                "smallest_group": minimum,
                "largest_group": maximum,
                "max_to_min_ratio": ratio,
            }
        )

    return pd.DataFrame(rows)


def plot_comparison(
    comparison: pd.DataFrame,
) -> None:

    ax = comparison.plot(
        kind="bar",
        figsize=(13, 7),
    )

    ax.set_title(
        "Candidate Dataset Balancing Strategies"
    )

    ax.set_xlabel(
        "Demographic category"
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.tick_params(
        axis="x",
        rotation=40,
    )

    ax.legend(
        title="Scenario"
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "01_balancing_strategy_comparison.png",
        dpi=220,
    )

    plt.close(fig)


def plot_total_volume(
    summary: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        summary["scenario"],
        summary["n_images"],
    )

    ax.set_title(
        "Training Volume Retained by Balancing Strategy"
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    for i, row in summary.iterrows():

        ax.text(
            i,
            row["n_images"],
            (
                f"{int(row['n_images']):,}\n"
                f"{row['percent_of_full_pool']:.1f}%"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "02_training_volume_comparison.png",
        dpi=220,
    )

    plt.close(fig)


def plot_imbalance_ratio(
    summary: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        summary["scenario"],
        summary["max_to_min_ratio"],
    )

    ax.axhline(
        1.0,
        linestyle="--",
    )

    ax.set_title(
        "Largest-to-Smallest Group Ratio"
    )

    ax.set_ylabel(
        "Ratio (1.0 = exact parity)"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "03_group_imbalance_ratio.png",
        dpi=220,
    )

    plt.close(fig)


def write_notes(
    summary: pd.DataFrame,
) -> None:

    text = """# Candidate Balancing Strategies

These scenarios are exploratory only.

No image has been deleted, duplicated, copied or relabelled.

## Scenario 1 — Current pool

Uses all currently mapped demographic records.

Advantage:
- maximum volume.

Disadvantage:
- White representation remains dominant.

## Scenario 2 — Strict parity across all trainable groups

Every demographic category is reduced to the size of the
smallest category.

Advantage:
- exact numerical parity.

Disadvantage:
- potentially severe loss of training volume.
- does not guarantee diversity or better model performance.

## Scenario 3 — Strict parity across explicit fine-grained groups

Uses:
- White
- Black
- Indian
- East Asian
- Southeast Asian
- Middle Eastern
- Latino

Broad Asian, Other and Unknown records remain preserved in
the full dataset but are not used in this candidate subset.

Advantage:
- clean source-supported taxonomy.
- no fabricated demographic refinement.

Disadvantage:
- excludes useful images from the candidate training subset.

## Scenario 4 — Volume-aware capped dataset

Large groups are capped at the median explicit-group size.
Smaller groups retain every available unique image.

Advantage:
- reduces majority dominance.
- preserves substantially more data than strict parity.
- does not duplicate minority examples.

Disadvantage:
- demographic counts are not exactly equal.

## Next stage

Whichever scenario is selected must then be constrained by:

- gender
- ITA / measured skin tone
- age
- source dataset
- brightness
- contrast
- sharpness
- face size
- image quality
- duplicate status
- identity concentration where identity is known

Numerical balance alone is not treated as sufficient evidence
of dataset fairness.
"""

    (
        OUTPUT_DIR
        / "balancing_strategy_notes.md"
    ).write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:

    df = load_data()

    counts = get_counts(
        df
    )

    comparison = build_comparison(
        counts
    )

    summary = build_summary(
        comparison,
        len(df),
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "balancing_targets.csv"
    )

    summary.to_csv(
        OUTPUT_DIR
        / "balancing_scenario_summary.csv",
        index=False,
    )

    plot_comparison(
        comparison
    )

    plot_total_volume(
        summary
    )

    plot_imbalance_ratio(
        summary
    )

    write_notes(
        summary
    )

    print("=" * 72)
    print("BALANCING STRATEGY COMPARISON")
    print("=" * 72)

    print("\nTarget counts:")
    print(
        comparison.to_string()
    )

    print("\n")
    print("=" * 72)
    print("SCENARIO SUMMARY")
    print("=" * 72)

    print(
        summary.to_string(
            index=False,
            formatters={
                "percent_of_full_pool": (
                    lambda x: f"{x:.2f}%"
                ),
                "max_to_min_ratio": (
                    lambda x: f"{x:.2f}"
                ),
            },
        )
    )

    print("\nOutputs:")
    print(OUTPUT_DIR)

    print(
        "\nNo images were deleted, copied, "
        "duplicated or modified."
    )


if __name__ == "__main__":
    main()
    