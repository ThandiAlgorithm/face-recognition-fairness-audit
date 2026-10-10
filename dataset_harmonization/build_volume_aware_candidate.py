from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    / "mvp_volume_aware"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 2026


# ============================================================
# PRIMARY FINE-GRAINED TRAINING TAXONOMY
# ============================================================

TARGET_COUNTS = {
    "White": 14990,
    "Black": 14990,
    "Indian": 14990,
    "East Asian": 13837,
    "Southeast Asian": 12210,
    "Middle Eastern": 10425,
    "Latino": 14990,
}


# Preserved in the combined master dataset but not used
# in this first fine-grained MVP training candidate.
HELD_ASIDE_LABELS = [
    "Asian",
    "Other",
    "Unknown",
]


def clean_strata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in [
        "source_dataset",
        "gender_label",
        "ita_group",
    ]:
        if column not in df.columns:
            df[column] = "missing"

        df[column] = (
            df[column]
            .fillna("missing")
            .astype(str)
            .str.strip()
            .str.lower()
        )

    return df


def proportional_stratified_sample(
    group_df: pd.DataFrame,
    target_n: int,
    seed: int,
) -> pd.DataFrame:
    """
    Downsample while approximately preserving the original
    source × gender × ITA composition.

    No oversampling.
    No image duplication.
    """

    if len(group_df) <= target_n:
        return group_df.copy()

    group_df = clean_strata(
        group_df
    )

    strata_cols = [
        "source_dataset",
        "gender_label",
        "ita_group",
    ]

    grouped = list(
        group_df.groupby(
            strata_cols,
            dropna=False,
        )
    )

    total = len(group_df)

    allocations = []

    for stratum_key, stratum_df in grouped:
        ideal = (
            len(stratum_df)
            / total
            * target_n
        )

        base = int(
            np.floor(ideal)
        )

        allocations.append(
            {
                "key": stratum_key,
                "df": stratum_df,
                "ideal": ideal,
                "allocated": min(
                    base,
                    len(stratum_df),
                ),
                "remainder": ideal - base,
            }
        )

    allocated_total = sum(
        item["allocated"]
        for item in allocations
    )

    remaining = (
        target_n
        - allocated_total
    )

    allocations.sort(
        key=lambda x: x["remainder"],
        reverse=True,
    )

    while remaining > 0:
        changed = False

        for item in allocations:
            if remaining <= 0:
                break

            if (
                item["allocated"]
                < len(item["df"])
            ):
                item["allocated"] += 1
                remaining -= 1
                changed = True

        if not changed:
            break

    sampled_frames = []

    for i, item in enumerate(
        allocations
    ):
        n = item["allocated"]

        if n <= 0:
            continue

        sample = item["df"].sample(
            n=n,
            replace=False,
            random_state=seed + i,
        )

        sampled_frames.append(
            sample
        )

    result = pd.concat(
        sampled_frames,
        ignore_index=False,
    )

    if len(result) != target_n:
        raise RuntimeError(
            f"Expected {target_n} rows, "
            f"selected {len(result)}."
        )

    return result


def build_candidate(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    selected = []

    for i, (
        label,
        target_n,
    ) in enumerate(
        TARGET_COUNTS.items()
    ):
        group = df[
            df[
                "candidate_demographic_label"
            ]
            == label
        ].copy()

        print(
            f"{label:18s} "
            f"available={len(group):6,d} "
            f"target={target_n:6,d}"
        )

        sampled = (
            proportional_stratified_sample(
                group,
                target_n,
                SEED + i * 100,
            )
        )

        selected.append(
            sampled
        )

    candidate = pd.concat(
        selected,
        ignore_index=True,
    )

    held_aside = df[
        df[
            "candidate_demographic_label"
        ].isin(
            HELD_ASIDE_LABELS
        )
    ].copy()

    return (
        candidate,
        held_aside,
    )


def distribution(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:

    if column not in df.columns:
        return pd.DataFrame()

    table = (
        df.groupby(
            [
                "candidate_demographic_label",
                column,
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    totals = (
        table.groupby(
            "candidate_demographic_label"
        )["count"]
        .transform("sum")
    )

    table["percentage"] = (
        table["count"]
        / totals
        * 100
    )

    return table


def plot_demographic_counts(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> None:

    before_counts = (
        before[
            before[
                "candidate_demographic_label"
            ].isin(
                TARGET_COUNTS
            )
        ][
            "candidate_demographic_label"
        ]
        .value_counts()
        .reindex(
            TARGET_COUNTS.keys()
        )
    )

    after_counts = (
        after[
            "candidate_demographic_label"
        ]
        .value_counts()
        .reindex(
            TARGET_COUNTS.keys()
        )
    )

    plot_df = pd.DataFrame(
        {
            "Before": before_counts,
            "MVP candidate": after_counts,
        }
    )

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title(
        "Demographic Distribution Before and After Volume-Aware Sampling"
    )

    ax.set_xlabel(
        "Demographic label"
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "01_before_after_demographic_counts.png",
        dpi=220,
    )

    plt.close(fig)


def plot_composition(
    table: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
) -> None:

    if table.empty:
        return

    pivot = table.pivot(
        index="candidate_demographic_label",
        columns=column,
        values="percentage",
    ).fillna(0)

    pivot = pivot.reindex(
        TARGET_COUNTS.keys()
    )

    ax = pivot.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Demographic label"
    )

    ax.set_ylabel(
        "Percentage within group"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    ax.legend(
        title=column,
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / filename,
        dpi=220,
    )

    plt.close(fig)


def main() -> None:

    print("=" * 72)
    print("BUILDING VOLUME-AWARE MVP DATASET")
    print("=" * 72)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    candidate, held_aside = (
        build_candidate(
            df
        )
    )

    # --------------------------------------------------------
    # Save manifests
    # --------------------------------------------------------

    candidate.to_csv(
        OUTPUT_DIR
        / "mvp_training_manifest.csv",
        index=False,
    )

    held_aside.to_csv(
        OUTPUT_DIR
        / "held_aside_records.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary counts
    # --------------------------------------------------------

    counts = (
        candidate[
            "candidate_demographic_label"
        ]
        .value_counts()
        .reindex(
            TARGET_COUNTS.keys()
        )
        .rename_axis(
            "candidate_demographic_label"
        )
        .reset_index(
            name="count"
        )
    )

    counts["percentage"] = (
        counts["count"]
        / len(candidate)
        * 100
    )

    counts.to_csv(
        OUTPUT_DIR
        / "mvp_demographic_counts.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Diversity diagnostics
    # --------------------------------------------------------

    gender = distribution(
        candidate,
        "gender_label",
    )

    ita = distribution(
        candidate,
        "ita_group",
    )

    source = distribution(
        candidate,
        "source_dataset",
    )

    gender.to_csv(
        OUTPUT_DIR
        / "mvp_gender_distribution.csv",
        index=False,
    )

    ita.to_csv(
        OUTPUT_DIR
        / "mvp_ita_distribution.csv",
        index=False,
    )

    source.to_csv(
        OUTPUT_DIR
        / "mvp_source_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_demographic_counts(
        df,
        candidate,
    )

    plot_composition(
        gender,
        "gender_label",
        "Gender Composition After Volume-Aware Sampling",
        "02_gender_composition.png",
    )

    plot_composition(
        ita,
        "ita_group",
        "ITA Composition After Volume-Aware Sampling",
        "03_ita_composition.png",
    )

    plot_composition(
        source,
        "source_dataset",
        "Source Dataset Contribution After Volume-Aware Sampling",
        "04_source_composition.png",
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("MVP DATASET COUNTS")
    print("=" * 72)

    print(
        counts.to_string(
            index=False,
            formatters={
                "percentage": (
                    lambda x: f"{x:.2f}%"
                )
            },
        )
    )

    print("\nTotal selected:")
    print(
        f"{len(candidate):,}"
    )

    print("\nHeld aside:")
    print(
        f"{len(held_aside):,}"
    )

    print("\nHeld-aside breakdown:")

    print(
        held_aside[
            "candidate_demographic_label"
        ]
        .value_counts()
        .to_string()
    )

    print("\nOutputs:")
    print(OUTPUT_DIR)

    print(
        "\nNo images were duplicated, "
        "deleted or modified."
    )


if __name__ == "__main__":
    main()
    