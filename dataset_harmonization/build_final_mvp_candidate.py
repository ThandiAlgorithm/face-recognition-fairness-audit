from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "mvp_volume_aware"
    / "mvp_training_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "final_mvp_candidate"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 2026

TARGET_GROUP = "Middle Eastern"
MAX_MALE_N = 6022


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["gender_clean"] = (
        df["gender_label"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["ita_clean"] = (
        df["ita_group"]
        .fillna("missing")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df


def sample_middle_eastern_males(
    males: pd.DataFrame,
) -> pd.DataFrame:
    """
    Downsample Middle Eastern male-labelled records while
    approximately preserving their ITA composition.
    """

    if len(males) <= MAX_MALE_N:
        return males.copy()

    pieces = []

    counts = (
        males["ita_clean"]
        .value_counts()
    )

    total = len(males)

    allocations = {}

    for ita_group, n in counts.items():
        target = round(
            n / total
            * MAX_MALE_N
        )

        allocations[ita_group] = min(
            target,
            n,
        )

    difference = (
        MAX_MALE_N
        - sum(allocations.values())
    )

    labels = list(
        counts.index
    )

    i = 0

    while difference != 0:
        label = labels[
            i % len(labels)
        ]

        if difference > 0:
            if (
                allocations[label]
                < counts[label]
            ):
                allocations[label] += 1
                difference -= 1

        else:
            if allocations[label] > 0:
                allocations[label] -= 1
                difference += 1

        i += 1

    for i, (
        ita_group,
        n_target,
    ) in enumerate(
        allocations.items()
    ):
        subset = males[
            males["ita_clean"]
            == ita_group
        ]

        selected = subset.sample(
            n=n_target,
            replace=False,
            random_state=SEED + i,
        )

        pieces.append(
            selected
        )

    return pd.concat(
        pieces,
        ignore_index=False,
    )


def build_final_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    target_group = df[
        df[
            "candidate_demographic_label"
        ]
        == TARGET_GROUP
    ].copy()

    other_groups = df[
        df[
            "candidate_demographic_label"
        ]
        != TARGET_GROUP
    ].copy()

    females = target_group[
        target_group["gender_clean"]
        == "female"
    ]

    males = target_group[
        target_group["gender_clean"]
        == "male"
    ]

    unknown = target_group[
        ~target_group[
            "gender_clean"
        ].isin(
            [
                "female",
                "male",
            ]
        )
    ]

    selected_males = (
        sample_middle_eastern_males(
            males
        )
    )

    final_target_group = pd.concat(
        [
            females,
            selected_males,
            unknown,
        ],
        ignore_index=False,
    )

    final = pd.concat(
        [
            other_groups,
            final_target_group,
        ],
        ignore_index=True,
    )

    return final


def build_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for group, g in df.groupby(
        "candidate_demographic_label"
    ):

        female = int(
            (
                g["gender_clean"]
                == "female"
            ).sum()
        )

        male = int(
            (
                g["gender_clean"]
                == "male"
            ).sum()
        )

        unknown = (
            len(g)
            - female
            - male
        )

        known = female + male

        rows.append(
            {
                "demographic_group": group,
                "n_total": len(g),
                "female_n": female,
                "male_n": male,
                "unknown_gender_n": unknown,
                "female_pct_known": (
                    female / known * 100
                    if known
                    else 0
                ),
                "male_pct_known": (
                    male / known * 100
                    if known
                    else 0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_ita_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    table = pd.crosstab(
        df[
            "candidate_demographic_label"
        ],
        df[
            "ita_clean"
        ],
        normalize="index",
    ) * 100

    return table.reset_index()


def plot_final_counts(
    df: pd.DataFrame,
) -> None:

    counts = (
        df[
            "candidate_demographic_label"
        ]
        .value_counts()
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.bar(
        counts.index,
        counts.values,
    )

    ax.set_title(
        "Final MVP Candidate — Demographic Composition"
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    for i, value in enumerate(
        counts.values
    ):
        ax.text(
            i,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "01_final_demographic_counts.png",
        dpi=220,
    )

    plt.close(fig)


def plot_gender(
    summary: pd.DataFrame,
) -> None:

    plot_df = summary.set_index(
        "demographic_group"
    )[
        [
            "female_pct_known",
            "male_pct_known",
        ]
    ]

    plot_df.columns = [
        "Female",
        "Male",
    ]

    ax = plot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
    )

    ax.axhline(
        35,
        linestyle="--",
    )

    ax.axhline(
        65,
        linestyle="--",
    )

    ax.set_title(
        "Final MVP Candidate — Gender Composition"
    )

    ax.set_ylabel(
        "Percentage of known-gender records"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "02_final_gender_composition.png",
        dpi=220,
    )

    plt.close(fig)


def main() -> None:

    print("=" * 72)
    print("BUILD FINAL MVP CANDIDATE")
    print("=" * 72)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    df = clean(
        df
    )

    original_n = len(df)

    final = build_final_dataset(
        df
    )

    final_n = len(final)

    removed_n = (
        original_n
        - final_n
    )

    summary = build_summary(
        final
    )

    ita_summary = build_ita_summary(
        final
    )

    final.to_csv(
        OUTPUT_DIR
        / "final_mvp_training_manifest.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "final_demographic_gender_summary.csv",
        index=False,
    )

    ita_summary.to_csv(
        OUTPUT_DIR
        / "final_ita_summary_percent.csv",
        index=False,
    )

    plot_final_counts(
        final
    )

    plot_gender(
        summary
    )

    print("\nOriginal MVP:")
    print(
        f"{original_n:,}"
    )

    print("\nFinal MVP:")
    print(
        f"{final_n:,}"
    )

    print("\nRemoved:")
    print(
        f"{removed_n:,}"
    )

    print("\n")
    print("=" * 72)
    print("FINAL DEMOGRAPHIC + GENDER SUMMARY")
    print("=" * 72)

    print(
        summary.to_string(
            index=False,
            formatters={
                "female_pct_known":
                    lambda x: f"{x:.1f}%",

                "male_pct_known":
                    lambda x: f"{x:.1f}%",
            },
        )
    )

    print("\n")
    print("=" * 72)
    print("FINAL ITA COMPOSITION (%)")
    print("=" * 72)

    print(
        ita_summary.round(
            2
        ).to_string(
            index=False
        )
    )

    print("\nOutputs:")
    print(OUTPUT_DIR)

    print(
        "\nNo source image was modified or duplicated."
    )


if __name__ == "__main__":
    main()
    