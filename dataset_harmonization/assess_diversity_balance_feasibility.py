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
    / "mvp_volume_aware"
    / "diversity_audit"
    / "mvp_training_manifest_enriched.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "mvp_volume_aware"
    / "balance_feasibility"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


GROUPS = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino",
]


def clean_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["gender_clean"] = (
        df["gender_label"]
        .fillna("unknown")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["ita_clean"] = (
        df["ita_group"]
        .fillna("missing")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if "age_group" in df.columns:
        df["age_group_clean"] = (
            df["age_group"]
            .fillna("missing")
            .astype(str)
            .str.strip()
        )

    return df


def build_gender_ita_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:

    usable = df[
        df["gender_clean"].isin(
            ["female", "male"]
        )
        & df["ita_clean"].isin(
            ["darker", "lighter"]
        )
    ].copy()

    counts = (
        usable.groupby(
            [
                "candidate_demographic_label",
                "gender_clean",
                "ita_clean",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    return counts


def build_feasibility_summary(
    df: pd.DataFrame,
    cell_counts: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for group in GROUPS:

        g = df[
            df[
                "candidate_demographic_label"
            ]
            == group
        ]

        total = len(g)

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

        darker = int(
            (
                g["ita_clean"]
                == "darker"
            ).sum()
        )

        lighter = int(
            (
                g["ita_clean"]
                == "lighter"
            ).sum()
        )

        # Maximum possible size under exact
        # female/male parity.
        gender_balanced_n = (
            2 * min(
                female,
                male,
            )
        )

        # Maximum possible size under exact
        # darker/lighter parity.
        ita_balanced_n = (
            2 * min(
                darker,
                lighter,
            )
        )

        group_cells = (
            cell_counts[
                cell_counts[
                    "candidate_demographic_label"
                ]
                == group
            ]
        )

        lookup = {
            (
                row["gender_clean"],
                row["ita_clean"],
            ): int(row["count"])
            for _, row
            in group_cells.iterrows()
        }

        four_cells = [
            lookup.get(
                ("female", "darker"),
                0,
            ),
            lookup.get(
                ("female", "lighter"),
                0,
            ),
            lookup.get(
                ("male", "darker"),
                0,
            ),
            lookup.get(
                ("male", "lighter"),
                0,
            ),
        ]

        joint_min = min(
            four_cells
        )

        joint_balanced_n = (
            4 * joint_min
        )

        rows.append(
            {
                "demographic_group": group,
                "current_n": total,

                "female_n": female,
                "male_n": male,

                "darker_n": darker,
                "lighter_n": lighter,

                "female_darker_n":
                    four_cells[0],

                "female_lighter_n":
                    four_cells[1],

                "male_darker_n":
                    four_cells[2],

                "male_lighter_n":
                    four_cells[3],

                "max_gender_parity_n":
                    gender_balanced_n,

                "gender_parity_retention_pct":
                    (
                        gender_balanced_n
                        / total
                        * 100
                        if total
                        else np.nan
                    ),

                "max_ita_parity_n":
                    ita_balanced_n,

                "ita_parity_retention_pct":
                    (
                        ita_balanced_n
                        / total
                        * 100
                        if total
                        else np.nan
                    ),

                "max_joint_gender_ita_parity_n":
                    joint_balanced_n,

                "joint_parity_retention_pct":
                    (
                        joint_balanced_n
                        / total
                        * 100
                        if total
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_age_coverage(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if "age_group_clean" not in df.columns:
        return pd.DataFrame()

    rows = []

    for group in GROUPS:

        g = df[
            df[
                "candidate_demographic_label"
            ]
            == group
        ]

        known = g[
            ~g[
                "age_group_clean"
            ].str.lower().isin(
                [
                    "missing",
                    "unknown",
                    "nan",
                    "none",
                ]
            )
        ]

        rows.append(
            {
                "demographic_group": group,
                "n_total": len(g),
                "n_age_group_known": len(
                    known
                ),
                "age_group_coverage_pct":
                    (
                        len(known)
                        / len(g)
                        * 100
                        if len(g)
                        else np.nan
                    ),
                "n_unique_age_groups":
                    known[
                        "age_group_clean"
                    ].nunique(),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_age_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if "age_group_clean" not in df.columns:
        return pd.DataFrame()

    return (
        df.groupby(
            [
                "candidate_demographic_label",
                "age_group_clean",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
    )


def build_brightness_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    # Global quartile cut-points give every group
    # the same definition of low / medium / high
    # brightness.
    values = pd.to_numeric(
        df["brightness"],
        errors="coerce",
    ).dropna()

    q25 = values.quantile(
        0.25
    )

    q75 = values.quantile(
        0.75
    )

    temp = df.copy()

    brightness = pd.to_numeric(
        temp["brightness"],
        errors="coerce",
    )

    temp["brightness_band"] = (
        "missing"
    )

    temp.loc[
        brightness < q25,
        "brightness_band",
    ] = "low"

    temp.loc[
        (
            brightness >= q25
        )
        & (
            brightness <= q75
        ),
        "brightness_band",
    ] = "medium"

    temp.loc[
        brightness > q75,
        "brightness_band",
    ] = "high"

    for group in GROUPS:

        g = temp[
            temp[
                "candidate_demographic_label"
            ]
            == group
        ]

        counts = (
            g["brightness_band"]
            .value_counts()
        )

        rows.append(
            {
                "demographic_group": group,
                "low_pct":
                    counts.get(
                        "low",
                        0,
                    )
                    / len(g)
                    * 100,

                "medium_pct":
                    counts.get(
                        "medium",
                        0,
                    )
                    / len(g)
                    * 100,

                "high_pct":
                    counts.get(
                        "high",
                        0,
                    )
                    / len(g)
                    * 100,

                "missing_pct":
                    counts.get(
                        "missing",
                        0,
                    )
                    / len(g)
                    * 100,
            }
        )

    summary = pd.DataFrame(
        rows
    )

    summary.attrs[
        "brightness_q25"
    ] = q25

    summary.attrs[
        "brightness_q75"
    ] = q75

    return summary


def plot_retention(
    summary: pd.DataFrame,
) -> None:

    plot_df = (
        summary[
            [
                "demographic_group",
                "gender_parity_retention_pct",
                "ita_parity_retention_pct",
                "joint_parity_retention_pct",
            ]
        ]
        .set_index(
            "demographic_group"
        )
    )

    plot_df.columns = [
        "Gender parity",
        "ITA parity",
        "Gender + ITA parity",
    ]

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title(
        "Data Retained Under Candidate Diversity Constraints"
    )

    ax.set_ylabel(
        "Percentage of current MVP retained"
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_xlabel(
        "Demographic group"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "01_balance_retention.png",
        dpi=220,
    )

    plt.close(fig)


def plot_brightness(
    summary: pd.DataFrame,
) -> None:

    plot_df = (
        summary.set_index(
            "demographic_group"
        )[
            [
                "low_pct",
                "medium_pct",
                "high_pct",
            ]
        ]
    )

    ax = plot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
    )

    ax.set_title(
        "Brightness Distribution Within MVP Groups"
    )

    ax.set_ylabel(
        "Percentage"
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_xlabel(
        "Demographic group"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "02_brightness_composition.png",
        dpi=220,
    )

    plt.close(fig)


def main() -> None:

    print("=" * 72)
    print("DIVERSITY-BALANCE FEASIBILITY")
    print("=" * 72)

    df = pd.read_parquet(
        INPUT
    )

    df = clean_data(
        df
    )

    cell_counts = (
        build_gender_ita_counts(
            df
        )
    )

    summary = (
        build_feasibility_summary(
            df,
            cell_counts,
        )
    )

    age_coverage = (
        build_age_coverage(
            df
        )
    )

    age_counts = (
        build_age_counts(
            df
        )
    )

    brightness = (
        build_brightness_summary(
            df
        )
    )

    cell_counts.to_csv(
        OUTPUT_DIR
        / "gender_ita_cell_counts.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "balance_feasibility_summary.csv",
        index=False,
    )

    age_coverage.to_csv(
        OUTPUT_DIR
        / "age_group_coverage.csv",
        index=False,
    )

    age_counts.to_csv(
        OUTPUT_DIR
        / "age_group_counts.csv",
        index=False,
    )

    brightness.to_csv(
        OUTPUT_DIR
        / "brightness_composition.csv",
        index=False,
    )

    plot_retention(
        summary
    )

    plot_brightness(
        brightness
    )

    print("\n")
    print("=" * 72)
    print("GENDER × ITA CELL COUNTS")
    print("=" * 72)

    print(
        cell_counts.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 72)
    print("BALANCE FEASIBILITY")
    print("=" * 72)

    display_cols = [
        "demographic_group",
        "current_n",
        "max_gender_parity_n",
        "gender_parity_retention_pct",
        "max_ita_parity_n",
        "ita_parity_retention_pct",
        "max_joint_gender_ita_parity_n",
        "joint_parity_retention_pct",
    ]

    print(
        summary[
            display_cols
        ].to_string(
            index=False,
            formatters={
                "gender_parity_retention_pct":
                    lambda x: f"{x:.1f}%",

                "ita_parity_retention_pct":
                    lambda x: f"{x:.1f}%",

                "joint_parity_retention_pct":
                    lambda x: f"{x:.1f}%",
            },
        )
    )

    print("\n")
    print("=" * 72)
    print("AGE-GROUP COVERAGE")
    print("=" * 72)

    print(
        age_coverage.to_string(
            index=False,
            formatters={
                "age_group_coverage_pct":
                    lambda x: f"{x:.1f}%"
            },
        )
    )

    print("\n")
    print("=" * 72)
    print("BRIGHTNESS COMPOSITION")
    print("=" * 72)

    print(
        brightness.to_string(
            index=False,
            formatters={
                "low_pct":
                    lambda x: f"{x:.1f}%",

                "medium_pct":
                    lambda x: f"{x:.1f}%",

                "high_pct":
                    lambda x: f"{x:.1f}%",

                "missing_pct":
                    lambda x: f"{x:.1f}%",
            },
        )
    )

    print("\nOutputs:")
    print(OUTPUT_DIR)

    print(
        "\nNo dataset membership was changed."
    )


if __name__ == "__main__":
    main()
    