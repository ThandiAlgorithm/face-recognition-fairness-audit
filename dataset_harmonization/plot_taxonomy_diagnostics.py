from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "candidate_taxonomy_v1"
)

OUTPUT_DIR = INPUT_DIR / "diagnostics"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


manifest = pd.read_csv(
    INPUT_DIR / "candidate_manifest.csv",
    low_memory=False,
)


# ============================================================
# 1. SOURCE LABEL -> CANDIDATE LABEL FLOW TABLE
# ============================================================

flow = (
    manifest.groupby(
        [
            "source_dataset",
            "race_label",
            "candidate_demographic_label",
        ],
        dropna=False,
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        "count",
        ascending=False,
    )
)

flow.to_csv(
    OUTPUT_DIR / "source_label_mapping_counts.csv",
    index=False,
)


# ============================================================
# 2. DEMOGRAPHIC DISTRIBUTION BY SOURCE DATASET
# ============================================================

source_counts = pd.crosstab(
    manifest[
        "candidate_demographic_label"
    ],
    manifest[
        "source_dataset"
    ],
)

source_percent = (
    source_counts
    .div(
        source_counts.sum(axis=0),
        axis=1,
    )
    * 100
)

source_percent.to_csv(
    OUTPUT_DIR
    / "candidate_distribution_by_source_percent.csv"
)

ax = source_percent.plot(
    kind="bar",
    figsize=(12, 6),
)

ax.set_title(
    "Candidate Demographic Composition by Source Dataset"
)

ax.set_xlabel(
    "Candidate demographic label"
)

ax.set_ylabel(
    "Percentage of source dataset"
)

ax.tick_params(
    axis="x",
    rotation=40,
)

ax.legend(
    title="Dataset"
)

fig = ax.get_figure()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "05_candidate_distribution_by_source.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# 3. GENDER BALANCE INSIDE EACH CANDIDATE GROUP
# ============================================================

gender = manifest.copy()

gender["gender_label"] = (
    gender["gender_label"]
    .fillna("unknown")
    .astype(str)
    .str.lower()
)

gender_table = pd.crosstab(
    gender[
        "candidate_demographic_label"
    ],
    gender[
        "gender_label"
    ],
)

gender_percent = (
    gender_table
    .div(
        gender_table.sum(axis=1),
        axis=0,
    )
    * 100
)

gender_percent.to_csv(
    OUTPUT_DIR
    / "gender_within_candidate_group_percent.csv"
)

ax = gender_percent.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6),
)

ax.set_title(
    "Gender Composition Within Candidate Demographic Groups"
)

ax.set_xlabel(
    "Candidate demographic label"
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
    rotation=40,
)

ax.legend(
    title="Gender"
)

fig = ax.get_figure()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "06_gender_within_candidate_groups.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# 4. ITA BALANCE INSIDE EACH CANDIDATE GROUP
# ============================================================

ita = manifest.copy()

ita["ita_group"] = (
    ita["ita_group"]
    .fillna("missing")
    .astype(str)
    .str.lower()
)

ita_table = pd.crosstab(
    ita[
        "candidate_demographic_label"
    ],
    ita[
        "ita_group"
    ],
)

ita_percent = (
    ita_table
    .div(
        ita_table.sum(axis=1),
        axis=0,
    )
    * 100
)

ita_percent.to_csv(
    OUTPUT_DIR
    / "ita_within_candidate_group_percent.csv"
)

ax = ita_percent.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6),
)

ax.set_title(
    "ITA Composition Within Candidate Demographic Groups"
)

ax.set_xlabel(
    "Candidate demographic label"
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
    rotation=40,
)

ax.legend(
    title="ITA group"
)

fig = ax.get_figure()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR
    / "07_ita_within_candidate_groups.png",
    dpi=220,
)

plt.close(fig)


# ============================================================
# 5. UNRESOLVED BREAKDOWN
# ============================================================

unresolved = manifest[
    manifest[
        "candidate_demographic_label"
    ]
    == "Unresolved"
].copy()

unresolved_counts = (
    unresolved.groupby(
        [
            "source_dataset",
            "race_label",
        ],
        dropna=False,
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        "count",
        ascending=False,
    )
)

unresolved_counts.to_csv(
    OUTPUT_DIR
    / "unresolved_breakdown.csv",
    index=False,
)

if not unresolved_counts.empty:

    labels = (
        unresolved_counts[
            "source_dataset"
        ].astype(str)
        + " · "
        + unresolved_counts[
            "race_label"
        ].astype(str)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        labels,
        unresolved_counts[
            "count"
        ],
    )

    ax.set_title(
        "Why Images Remain Unresolved"
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    for i, value in enumerate(
        unresolved_counts[
            "count"
        ]
    ):
        ax.text(
            i,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "08_unresolved_breakdown.png",
        dpi=220,
    )

    plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("TAXONOMY DIAGNOSTICS COMPLETE")
print("=" * 70)

print("\nUnresolved labels:")
print(
    unresolved_counts.to_string(
        index=False
    )
)

print("\nGender composition (%):")
print(
    gender_percent.round(2).to_string()
)

print("\nITA composition (%):")
print(
    ita_percent.round(2).to_string()
)

print("\nOutputs:")
print(OUTPUT_DIR)
