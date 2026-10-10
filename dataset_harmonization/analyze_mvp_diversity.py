from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MVP_PATH = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "mvp_volume_aware"
    / "mvp_training_manifest.csv"
)

QUALITY_PATH = (
    ROOT
    / "data"
    / "derived"
    / "cache"
    / "detection_alignment_measurement.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "mvp_volume_aware"
    / "diversity_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


GROUP_ORDER = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino",
]


QUALITY_COLUMNS = [
    "brightness",
    "contrast",
    "sharpness",
    "face_area_ratio",
]

NUMERIC_COLUMNS = [
    "age_value",
    "ita_value",
    "brightness",
    "contrast",
    "sharpness",
    "face_area_ratio",
]


def load_data() -> pd.DataFrame:
    if not MVP_PATH.exists():
        raise FileNotFoundError(
            f"Missing MVP manifest:\n{MVP_PATH}"
        )

    if not QUALITY_PATH.exists():
        raise FileNotFoundError(
            f"Missing Phase 1 quality cache:\n{QUALITY_PATH}"
        )

    mvp = pd.read_csv(
        MVP_PATH,
        low_memory=False,
    )

    quality = pd.read_parquet(
        QUALITY_PATH
    )

    available_quality = [
        "image_id",
        *[
            col
            for col in QUALITY_COLUMNS
            if col in quality.columns
        ],
    ]

    quality = quality[
        available_quality
    ].drop_duplicates(
        subset=["image_id"]
    )

    df = mvp.merge(
        quality,
        on="image_id",
        how="left",
        validate="many_to_one",
    )

    return df


def build_coverage(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        rows.append(
            {
                "variable": column,
                "n_total": len(df),
                "n_available": int(
                    values.notna().sum()
                ),
                "n_missing": int(
                    values.isna().sum()
                ),
                "coverage_percent": float(
                    values.notna().mean()
                    * 100
                ),
            }
        )

    return pd.DataFrame(rows)


def build_group_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for group in GROUP_ORDER:
        g = df[
            df[
                "candidate_demographic_label"
            ]
            == group
        ]

        for column in NUMERIC_COLUMNS:
            if column not in g.columns:
                continue

            values = pd.to_numeric(
                g[column],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            rows.append(
                {
                    "demographic_group": group,
                    "variable": column,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "p10": values.quantile(0.10),
                    "p25": values.quantile(0.25),
                    "median": values.median(),
                    "p75": values.quantile(0.75),
                    "p90": values.quantile(0.90),
                    "max": values.max(),
                }
            )

    return pd.DataFrame(rows)


def build_source_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:

    table = pd.crosstab(
        df[
            "candidate_demographic_label"
        ],
        df[
            "source_dataset"
        ],
    )

    table = table.reindex(
        GROUP_ORDER,
        fill_value=0,
    )

    percentages = (
        table
        .div(
            table.sum(axis=1),
            axis=0,
        )
        * 100
    )

    percentages.index.name = (
        "candidate_demographic_label"
    )

    return percentages.reset_index()


def plot_box(
    df: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
) -> None:

    if column not in df.columns:
        return

    data = []
    labels = []

    for group in GROUP_ORDER:
        values = pd.to_numeric(
            df.loc[
                df[
                    "candidate_demographic_label"
                ]
                == group,
                column,
            ],
            errors="coerce",
        ).dropna()

        if len(values) > 0:
            data.append(values.values)
            labels.append(group)

    if not data:
        return

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
    )

    ax.set_title(title)

    ax.set_ylabel(column)

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=220,
    )

    plt.close(fig)


def plot_source_mix(
    source_table: pd.DataFrame,
) -> None:

    temp = source_table.set_index(
        "candidate_demographic_label"
    )

    ax = temp.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
    )

    ax.set_title(
        "Source Dataset Composition Within MVP Groups"
    )

    ax.set_xlabel(
        "Demographic group"
    )

    ax.set_ylabel(
        "Percentage"
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
        title="Source dataset",
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
        / "01_source_mix.png",
        dpi=220,
    )

    plt.close(fig)


def print_medians(
    summary: pd.DataFrame,
    variable: str,
) -> None:

    temp = summary[
        summary["variable"]
        == variable
    ][
        [
            "demographic_group",
            "n",
            "median",
            "p10",
            "p90",
        ]
    ]

    if temp.empty:
        return

    print()
    print(variable.upper())
    print(
        temp.to_string(
            index=False,
            formatters={
                "median": lambda x: f"{x:.3f}",
                "p10": lambda x: f"{x:.3f}",
                "p90": lambda x: f"{x:.3f}",
            },
        )
    )


def main() -> None:

    print("=" * 72)
    print("MVP DIVERSITY AUDIT")
    print("=" * 72)

    df = load_data()

    print(
        f"\nMVP images: {len(df):,}"
    )

    coverage = build_coverage(
        df
    )

    summary = build_group_summary(
        df
    )

    source_table = build_source_counts(
        df
    )

    coverage.to_csv(
        OUTPUT_DIR
        / "variable_coverage.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "demographic_diversity_summary.csv",
        index=False,
    )

    source_table.to_csv(
        OUTPUT_DIR
        / "source_mix_percent.csv",
        index=False,
    )

    df.to_parquet(
        OUTPUT_DIR
        / "mvp_training_manifest_enriched.parquet",
        index=False,
    )

    plot_source_mix(
        source_table
    )

    plot_box(
        df,
        "age_value",
        "Age Distribution by Demographic Group",
        "02_age_distribution.png",
    )

    plot_box(
        df,
        "ita_value",
        "ITA Distribution by Demographic Group",
        "03_ita_distribution.png",
    )

    plot_box(
        df,
        "brightness",
        "Brightness Distribution by Demographic Group",
        "04_brightness_distribution.png",
    )

    plot_box(
        df,
        "contrast",
        "Contrast Distribution by Demographic Group",
        "05_contrast_distribution.png",
    )

    plot_box(
        df,
        "sharpness",
        "Sharpness Distribution by Demographic Group",
        "06_sharpness_distribution.png",
    )

    plot_box(
        df,
        "face_area_ratio",
        "Face Area Ratio by Demographic Group",
        "07_face_area_distribution.png",
    )

    print("\n")
    print("=" * 72)
    print("VARIABLE COVERAGE")
    print("=" * 72)

    print(
        coverage.to_string(
            index=False,
            formatters={
                "coverage_percent":
                    lambda x: f"{x:.2f}%"
            },
        )
    )

    print("\n")
    print("=" * 72)
    print("GROUP MEDIANS + 10–90% RANGE")
    print("=" * 72)

    for variable in NUMERIC_COLUMNS:
        print_medians(
            summary,
            variable,
        )

    print("\n")
    print("=" * 72)
    print("SOURCE COMPOSITION (%)")
    print("=" * 72)

    print(
        source_table.round(
            2
        ).to_string(
            index=False
        )
    )

    print("\nOutputs:")
    print(OUTPUT_DIR)

    print(
        "\nNo dataset membership was changed."
    )


if __name__ == "__main__":
    main()
    