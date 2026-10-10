from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RECORDS_DIR = (
    ROOT
    / "face_resnet34_training"
    / "data"
    / "records"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "candidate_taxonomy_v2"
)

DATASETS = {
    "fairface": RECORDS_DIR / "fairface_records.csv",
    "utkface": RECORDS_DIR / "utkface_records.csv",
    "lfw": RECORDS_DIR / "lfw_records.csv",
}


# ============================================================
# CANDIDATE TAXONOMY V2
# ============================================================
#
# White
# Black
# Indian
# East Asian
# Southeast Asian
# Middle Eastern
# Latino
# Asian
# Other
# Unknown
#
# Principles:
# - preserve original source labels
# - no visual relabelling
# - no inference from facial appearance
# - only map when source metadata supports it
# ============================================================


TARGET_ORDER = [
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


MAPPINGS = {
    "fairface": {
        "White": ("White", "direct"),
        "Black": ("Black", "direct"),
        "Indian": ("Indian", "direct"),
        "East Asian": ("East Asian", "direct"),
        "Southeast Asian": (
            "Southeast Asian",
            "direct",
        ),
        "Middle Eastern": (
            "Middle Eastern",
            "direct",
        ),
        "Latino_Hispanic": (
            "Latino",
            "harmonised from FairFace Latino_Hispanic",
        ),
    },

    "utkface": {
        "White": ("White", "direct"),
        "Black": ("Black", "direct"),
        "Indian": ("Indian", "direct"),
        "Asian": ("Asian", "direct broad source label"),
        "Others": (
            "Other",
            "source residual category",
        ),
        "unknown": (
            "Unknown",
            "source unknown",
        ),
    },

    "lfw": {
        "White": (
            "White",
            "direct derived label",
        ),
        "Black": (
            "Black",
            "direct derived label",
        ),
        "Indian": (
            "Indian",
            "direct derived label",
        ),
        "Asian": (
            "Asian",
            "direct broad derived label",
        ),
        "unknown": (
            "Unknown",
            "derived label unavailable",
        ),
    },
}


def load_records() -> pd.DataFrame:
    frames = []

    for dataset, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing records file: {path}"
            )

        df = pd.read_csv(
            path,
            low_memory=False,
        )

        df["source_dataset"] = dataset

        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def map_label(
    dataset: str,
    source_label,
) -> tuple[str, str]:

    if pd.isna(source_label):
        return "Unknown", "missing source label"

    label = str(source_label).strip()

    mapping = MAPPINGS.get(
        dataset,
        {},
    )

    if label in mapping:
        return mapping[label]

    return (
        "Unknown",
        f"no mapping rule for source label: {label}",
    )


def add_candidate_taxonomy(
    df: pd.DataFrame,
) -> pd.DataFrame:

    mapped = df.apply(
        lambda row: map_label(
            row["source_dataset"],
            row.get("race_label"),
        ),
        axis=1,
    )

    df = df.copy()

    df[
        [
            "candidate_demographic_label",
            "mapping_reason",
        ]
    ] = pd.DataFrame(
        mapped.tolist(),
        index=df.index,
    )

    return df


def build_taxonomy_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:

    counts = (
        df["candidate_demographic_label"]
        .value_counts()
        .reindex(
            TARGET_ORDER,
            fill_value=0,
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
        / len(df)
        * 100
    )

    return counts


def build_source_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    table = pd.crosstab(
        df["candidate_demographic_label"],
        df["source_dataset"],
    )

    table = table.reindex(
        TARGET_ORDER,
        fill_value=0,
    )

    return table.reset_index()


def build_gender_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if "gender_label" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()

    temp["gender_label"] = (
        temp["gender_label"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    table = pd.crosstab(
        temp["candidate_demographic_label"],
        temp["gender_label"],
        normalize="index",
    ) * 100

    table = table.reindex(
        TARGET_ORDER
    )

    return table.reset_index()


def build_ita_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if "ita_group" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()

    temp["ita_group"] = (
        temp["ita_group"]
        .fillna("missing")
        .astype(str)
        .str.lower()
    )

    table = pd.crosstab(
        temp["candidate_demographic_label"],
        temp["ita_group"],
        normalize="index",
    ) * 100

    table = table.reindex(
        TARGET_ORDER
    )

    return table.reset_index()


def build_mapping_audit(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        df.groupby(
            [
                "source_dataset",
                "race_label",
                "candidate_demographic_label",
                "mapping_reason",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            [
                "candidate_demographic_label",
                "source_dataset",
                "count",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
    )


def plot_taxonomy_counts(
    counts: pd.DataFrame,
) -> None:

    plot_df = counts[
        counts["count"] > 0
    ].copy()

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.bar(
        plot_df["candidate_demographic_label"],
        plot_df["count"],
    )

    ax.set_title(
        "Candidate Harmonised Dataset Composition — Taxonomy v2"
    )

    ax.set_xlabel(
        "Candidate demographic label"
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    for i, row in plot_df.reset_index(
        drop=True
    ).iterrows():

        ax.text(
            i,
            row["count"],
            f"{int(row['count']):,}\n"
            f"{row['percentage']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "01_candidate_taxonomy_counts.png",
        dpi=220,
    )

    plt.close(fig)


def plot_source_contribution(
    source_table: pd.DataFrame,
) -> None:

    temp = source_table.set_index(
        "candidate_demographic_label"
    )

    temp = temp.loc[
        temp.sum(axis=1) > 0
    ]

    ax = temp.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
    )

    ax.set_title(
        "Source Dataset Contribution by Candidate Label"
    )

    ax.set_xlabel(
        "Candidate demographic label"
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    ax.legend(
        title="Source dataset"
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "02_source_contribution.png",
        dpi=220,
    )

    plt.close(fig)


def plot_percentage_table(
    table: pd.DataFrame,
    title: str,
    output_name: str,
) -> None:

    if table.empty:
        return

    temp = table.set_index(
        "candidate_demographic_label"
    )

    temp = temp.dropna(
        how="all"
    )

    if temp.empty:
        return

    ax = temp.plot(
        kind="bar",
        stacked=True,
        figsize=(12, 6),
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Candidate demographic label"
    )

    ax.set_ylabel(
        "Percentage within category"
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
        title="Group",
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
        / output_name,
        dpi=220,
    )

    plt.close(fig)


def write_justification(
    counts: pd.DataFrame,
) -> None:

    total = int(
        counts["count"].sum()
    )

    text = f"""# Candidate Harmonised Taxonomy v2

## Candidate labels

- White
- Black
- Indian
- East Asian
- Southeast Asian
- Middle Eastern
- Latino
- Asian
- Other
- Unknown

## Dataset size

Total pooled source images: {total:,}

## Mapping principles

1. Original source labels are preserved.
2. No demographic label is assigned from facial appearance.
3. FairFace East Asian remains East Asian.
4. FairFace Southeast Asian remains Southeast Asian.
5. UTKFace Asian remains Asian because its source metadata is broader.
6. LFW derived Asian remains Asian because the derived metadata is broader.
7. FairFace Indian remains Indian.
8. FairFace Middle Eastern remains Middle Eastern.
9. FairFace Latino_Hispanic is harmonised to Latino.
10. UTKFace Others remains Other.
11. Missing or unavailable labels remain Unknown.

## Design rationale

The taxonomy prioritises traceability over forced completeness.

A more specific demographic category is only used when supported by the
source annotation. Broad labels are not visually subdivided.

The final dataset will later be evaluated not only for demographic counts,
but also for:

- gender composition,
- ITA / skin-tone composition,
- age coverage,
- source diversity,
- image quality,
- brightness,
- contrast,
- sharpness,
- face size,
- identity concentration where identity labels exist,
- and downstream model performance.

This version is still a candidate taxonomy and does not modify any source
image or source annotation.
"""

    (
        OUTPUT_DIR
        / "taxonomy_justification.md"
    ).write_text(
        text,
        encoding="utf-8",
    )


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("CANDIDATE HARMONISED TAXONOMY V2")
    print("=" * 72)

    df = load_records()

    print(
        f"\nCombined source rows: {len(df):,}"
    )

    df = add_candidate_taxonomy(
        df
    )

    counts = build_taxonomy_counts(
        df
    )

    source_table = build_source_table(
        df
    )

    gender_table = build_gender_table(
        df
    )

    ita_table = build_ita_table(
        df
    )

    mapping_audit = build_mapping_audit(
        df
    )

    manifest_columns = [
        column
        for column in [
            "image_id",
            "image_path",
            "source_dataset",
            "race_label",
            "race_source",
            "gender_label",
            "gender_source",
            "age_value",
            "age_group",
            "ita_value",
            "ita_group",
            "identity_id",
            "candidate_demographic_label",
            "mapping_reason",
        ]
        if column in df.columns
    ]

    df[
        manifest_columns
    ].to_csv(
        OUTPUT_DIR
        / "candidate_manifest.csv",
        index=False,
    )

    counts.to_csv(
        OUTPUT_DIR
        / "taxonomy_counts.csv",
        index=False,
    )

    source_table.to_csv(
        OUTPUT_DIR
        / "source_by_taxonomy.csv",
        index=False,
    )

    gender_table.to_csv(
        OUTPUT_DIR
        / "gender_by_taxonomy_percent.csv",
        index=False,
    )

    ita_table.to_csv(
        OUTPUT_DIR
        / "ita_by_taxonomy_percent.csv",
        index=False,
    )

    mapping_audit.to_csv(
        OUTPUT_DIR
        / "mapping_audit.csv",
        index=False,
    )

    plot_taxonomy_counts(
        counts
    )

    plot_source_contribution(
        source_table
    )

    plot_percentage_table(
        gender_table,
        "Gender Composition Within Candidate Demographic Labels",
        "03_gender_composition.png",
    )

    plot_percentage_table(
        ita_table,
        "ITA Composition Within Candidate Demographic Labels",
        "04_ita_composition.png",
    )

    write_justification(
        counts
    )

    print("\n")
    print("=" * 72)
    print("CURRENT CANDIDATE COUNTS")
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

    print("\n")
    print("=" * 72)
    print("SOURCE MAPPING AUDIT")
    print("=" * 72)

    print(
        mapping_audit.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 72)
    print("OUTPUTS")
    print("=" * 72)

    print(OUTPUT_DIR)

    print(
        "\nNo source images or source labels were modified."
    )


if __name__ == "__main__":
    main()
    