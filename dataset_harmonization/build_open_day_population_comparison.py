from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE_MANIFEST = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "candidate_taxonomy_v2"
    / "candidate_manifest.csv"
)

FINAL_MANIFEST = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "final_mvp_candidate"
    / "final_mvp_training_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "open_day_pack"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# UN WORLD POPULATION PROSPECTS 2024
# 2025 medium-variant projections, millions
# ============================================================

WORLD_REGIONS = {
    "Asia": 4835,
    "Africa": 1550,
    "Americas": 1055,
    "Europe": 744,
    "Oceania": 47,
}

WORLD_TOTAL = 8232


ASIA_SUBREGIONS = {
    "Southern Asia": 2085,
    "Eastern Asia": 1652,
    "South-Eastern Asia": 700,
    "Western Asia": 314,
    "Central Asia": 84,
}


TAXONOMY_ORDER = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino",
    "Asian",
    "Other",
    "Unknown",
]


def load_data():
    source = pd.read_csv(
        SOURCE_MANIFEST,
        low_memory=False,
    )

    final = pd.read_csv(
        FINAL_MANIFEST,
        low_memory=False,
    )

    return source, final


def build_world_region_table():
    rows = []

    for region, population_m in WORLD_REGIONS.items():
        rows.append(
            {
                "region": region,
                "population_millions": population_m,
                "world_share_percent":
                    population_m
                    / WORLD_TOTAL
                    * 100,
            }
        )

    return pd.DataFrame(rows)


def build_asia_table():
    rows = []

    for region, population_m in ASIA_SUBREGIONS.items():
        rows.append(
            {
                "region": region,
                "population_millions": population_m,
                "world_share_percent":
                    population_m
                    / WORLD_TOTAL
                    * 100,
                "asia_share_percent":
                    population_m
                    / WORLD_REGIONS["Asia"]
                    * 100,
            }
        )

    return pd.DataFrame(rows)


def build_dataset_comparison(
    source: pd.DataFrame,
    final: pd.DataFrame,
):
    rows = []

    for dataset in [
        "fairface",
        "utkface",
        "lfw",
    ]:
        df = source[
            source["source_dataset"]
            == dataset
        ]

        total = len(df)

        counts = (
            df[
                "candidate_demographic_label"
            ]
            .value_counts()
        )

        for label in TAXONOMY_ORDER:
            count = int(
                counts.get(
                    label,
                    0,
                )
            )

            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "count": count,
                    "percentage":
                        count / total * 100,
                }
            )

    total = len(final)

    counts = (
        final[
            "candidate_demographic_label"
        ]
        .value_counts()
    )

    for label in TAXONOMY_ORDER:
        count = int(
            counts.get(
                label,
                0,
            )
        )

        rows.append(
            {
                "dataset": "Our MVP",
                "label": label,
                "count": count,
                "percentage":
                    count / total * 100,
            }
        )

    return pd.DataFrame(rows)


def build_gender_comparison(
    source: pd.DataFrame,
    final: pd.DataFrame,
):
    rows = []

    frames = {
        "FairFace": source[
            source["source_dataset"]
            == "fairface"
        ],
        "UTKFace": source[
            source["source_dataset"]
            == "utkface"
        ],
        "LFW": source[
            source["source_dataset"]
            == "lfw"
        ],
        "Our MVP": final,
    }

    for name, df in frames.items():
        gender = (
            df["gender_label"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
        )

        known = gender.isin(
            ["female", "male"]
        )

        female = int(
            (gender == "female").sum()
        )

        male = int(
            (gender == "male").sum()
        )

        known_n = int(
            known.sum()
        )

        rows.append(
            {
                "dataset": name,
                "female_n": female,
                "male_n": male,
                "female_percent":
                    female / known_n * 100,
                "male_percent":
                    male / known_n * 100,
            }
        )

    return pd.DataFrame(rows)


def plot_world_regions(df):
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        df["region"],
        df["world_share_percent"],
    )

    ax.set_title(
        "World Population by Region — 2025"
    )

    ax.set_ylabel(
        "Share of world population (%)"
    )

    for i, row in df.iterrows():
        ax.text(
            i,
            row["world_share_percent"],
            (
                f"{row['world_share_percent']:.1f}%\n"
                f"{row['population_millions']:,}M"
            ),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "01_world_population_2025.png",
        dpi=220,
    )

    plt.close(fig)


def plot_asia_subregions(df):
    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        df["region"],
        df["population_millions"],
    )

    ax.set_title(
        "Population of Asian Subregions — 2025"
    )

    ax.set_ylabel(
        "Population (millions)"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    for i, row in df.iterrows():
        ax.text(
            i,
            row["population_millions"],
            (
                f"{row['population_millions']:,}M\n"
                f"{row['world_share_percent']:.1f}% world"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "02_asia_population_2025.png",
        dpi=220,
    )

    plt.close(fig)


def plot_dataset_comparison(df):
    pivot = df.pivot(
        index="label",
        columns="dataset",
        values="percentage",
    ).fillna(0)

    pivot = pivot.reindex(
        TAXONOMY_ORDER
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(14, 7),
    )

    ax.set_title(
        "Demographic Composition: Source Datasets vs Our MVP"
    )

    ax.set_xlabel(
        "Harmonised demographic label"
    )

    ax.set_ylabel(
        "Percentage of dataset"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "03_source_vs_mvp_composition.png",
        dpi=220,
    )

    plt.close(fig)


def plot_gender_comparison(df):
    plot_df = (
        df.set_index(
            "dataset"
        )[
            [
                "female_percent",
                "male_percent",
            ]
        ]
    )

    plot_df.columns = [
        "Female",
        "Male",
    ]

    ax = plot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6),
    )

    ax.set_title(
        "Gender Composition: Source Datasets vs Our MVP"
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
        rotation=0,
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "04_gender_source_vs_mvp.png",
        dpi=220,
    )

    plt.close(fig)


def write_facts(
    world,
    asia,
    source,
    final,
):
    final_counts = (
        final[
            "candidate_demographic_label"
        ]
        .value_counts()
    )

    largest = int(
        final_counts.max()
    )

    smallest = int(
        final_counts.min()
    )

    ratio = (
        largest / smallest
    )

    lfw = source[
        source["source_dataset"]
        == "lfw"
    ]

    lfw_white = (
        (
            lfw[
                "candidate_demographic_label"
            ]
            == "White"
        ).mean()
        * 100
    )

    lfw_gender = (
        lfw["gender_label"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    lfw_known_gender = (
        lfw_gender.isin(
            ["male", "female"]
        )
    )

    lfw_male = (
        (
            lfw_gender[
                lfw_known_gender
            ]
            == "male"
        ).mean()
        * 100
    )

    asia_share = float(
        world.loc[
            world["region"]
            == "Asia",
            "world_share_percent",
        ].iloc[0]
    )

    southern_asia = float(
        asia.loc[
            asia["region"]
            == "Southern Asia",
            "world_share_percent",
        ].iloc[0]
    )

    eastern_asia = float(
        asia.loc[
            asia["region"]
            == "Eastern Asia",
            "world_share_percent",
        ].iloc[0]
    )

    southeast_asia = float(
        asia.loc[
            asia["region"]
            == "South-Eastern Asia",
            "world_share_percent",
        ].iloc[0]
    )

    text = f"""
OPEN DAY — DID YOU KNOW?

WORLD POPULATION
----------------
• The 2025 world population is approximately 8.232 billion.
• Asia contains approximately {asia_share:.1f}% of the world's population.
• Southern Asia alone represents approximately {southern_asia:.1f}% of humanity.
• Eastern Asia represents approximately {eastern_asia:.1f}%.
• South-Eastern Asia represents approximately {southeast_asia:.1f}%.

SOURCE DATASET FINDING
----------------------
• In our LFW metadata, approximately {lfw_white:.1f}% of images are labelled White.
• Approximately {lfw_male:.1f}% of known-gender LFW images are labelled male.

OUR MVP
-------
• Final candidate: {len(final):,} image records.
• Seven explicit demographic categories are retained.
• Largest group: {largest:,} records.
• Smallest group: {smallest:,} records.
• Largest-to-smallest ratio: {ratio:.2f}:1.
• No minority category was duplicated to manufacture numerical balance.
• A 35/65 gender floor was used rather than forcing exact 50/50 parity.

IMPORTANT INTERPRETATION
------------------------
World geography is NOT treated as race or ethnicity.

Population statistics are used as contextual evidence about global
representation. Demographic labels retain their original source provenance.
Skin tone (ITA) is analysed separately from demographic category.
"""

    (
        OUTPUT_DIR
        / "open_day_did_you_know.txt"
    ).write_text(
        text.strip(),
        encoding="utf-8",
    )

    print(text)


def main():
    print("=" * 72)
    print("OPEN DAY POPULATION + DATASET COMPARISON")
    print("=" * 72)

    source, final = load_data()

    world = build_world_region_table()
    asia = build_asia_table()

    comparison = build_dataset_comparison(
        source,
        final,
    )

    gender = build_gender_comparison(
        source,
        final,
    )

    world.to_csv(
        OUTPUT_DIR
        / "world_population_2025.csv",
        index=False,
    )

    asia.to_csv(
        OUTPUT_DIR
        / "asia_subregions_2025.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "source_vs_mvp_demographics.csv",
        index=False,
    )

    gender.to_csv(
        OUTPUT_DIR
        / "source_vs_mvp_gender.csv",
        index=False,
    )

    plot_world_regions(
        world
    )

    plot_asia_subregions(
        asia
    )

    plot_dataset_comparison(
        comparison
    )

    plot_gender_comparison(
        gender
    )

    write_facts(
        world,
        asia,
        source,
        final,
    )

    print("\nOutputs:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
    