from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
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
    / "final_mvp_candidate"
    / "splits"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 2026

TRAIN_FRACTION = 0.80
VAL_FRACTION = 0.10
TEST_FRACTION = 0.10


EXPECTED_LABELS = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino",
]


def deterministic_value(
    key: str,
) -> float:
    """
    Convert a stable key into a deterministic number in [0, 1).

    This makes the split fully reproducible.
    """

    token = (
        f"{SEED}|{key}"
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        token
    ).digest()

    number = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return (
        number
        / float(2**64)
    )


def make_split_key(
    row: pd.Series,
) -> str:
    """
    LFW:
        split by identity so the same known person cannot
        appear in train and test.

    FairFace / UTKFace:
        split by image record because identity IDs are not
        available in the current metadata.
    """

    source = str(
        row.get(
            "source_dataset",
            "unknown",
        )
    ).lower()

    identity = row.get(
        "identity_id"
    )

    image_id = row.get(
        "image_id"
    )

    if (
        source == "lfw"
        and pd.notna(identity)
        and str(identity).strip()
        not in [
            "",
            "nan",
            "none",
        ]
    ):
        return (
            f"lfw_identity::{identity}"
        )

    return (
        f"{source}::image::{image_id}"
    )


def assign_split(
    value: float,
) -> str:

    if value < TRAIN_FRACTION:
        return "train"

    if value < (
        TRAIN_FRACTION
        + VAL_FRACTION
    ):
        return "val"

    return "test"


def validate_labels(
    df: pd.DataFrame,
) -> None:

    actual = set(
        df[
            "candidate_demographic_label"
        ]
        .dropna()
        .unique()
    )

    expected = set(
        EXPECTED_LABELS
    )

    unexpected = (
        actual - expected
    )

    missing = (
        expected - actual
    )

    if unexpected:
        raise RuntimeError(
            "Unexpected demographic labels: "
            + str(
                sorted(unexpected)
            )
        )

    if missing:
        raise RuntimeError(
            "Expected demographic labels missing: "
            + str(
                sorted(missing)
            )
        )


def structural_checks(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    duplicate_source_image = (
        df.duplicated(
            subset=[
                "source_dataset",
                "image_id",
            ],
            keep=False,
        ).sum()
        if (
            "source_dataset"
            in df.columns
            and "image_id"
            in df.columns
        )
        else 0
    )

    duplicate_paths = (
        df["image_path"]
        .dropna()
        .duplicated(
            keep=False
        )
        .sum()
        if "image_path"
        in df.columns
        else 0
    )

    missing_paths = (
        df["image_path"]
        .isna()
        .sum()
        if "image_path"
        in df.columns
        else len(df)
    )

    rows.append(
        {
            "check":
                "duplicate source_dataset + image_id rows",
            "value":
                int(
                    duplicate_source_image
                ),
        }
    )

    rows.append(
        {
            "check":
                "duplicate image_path rows",
            "value":
                int(
                    duplicate_paths
                ),
        }
    )

    rows.append(
        {
            "check":
                "missing image_path",
            "value":
                int(
                    missing_paths
                ),
        }
    )

    return pd.DataFrame(
        rows
    )


def validate_lfw_identity_leakage(
    df: pd.DataFrame,
) -> int:

    if "identity_id" not in df.columns:
        return 0

    lfw = df[
        (
            df["source_dataset"]
            .astype(str)
            .str.lower()
            == "lfw"
        )
        & df[
            "identity_id"
        ].notna()
    ].copy()

    if lfw.empty:
        return 0

    identity_split_counts = (
        lfw.groupby(
            "identity_id"
        )["split"]
        .nunique()
    )

    leakage = int(
        (
            identity_split_counts
            > 1
        ).sum()
    )

    return leakage


def build_split_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    total = len(df)

    for split in [
        "train",
        "val",
        "test",
    ]:

        subset = df[
            df["split"]
            == split
        ]

        rows.append(
            {
                "split": split,
                "n_records":
                    len(subset),
                "percentage":
                    len(subset)
                    / total
                    * 100,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_label_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    table = pd.crosstab(
        df[
            "candidate_demographic_label"
        ],
        df[
            "split"
        ],
    )

    table = table.reindex(
        EXPECTED_LABELS,
        fill_value=0,
    )

    for split in [
        "train",
        "val",
        "test",
    ]:
        if split not in table.columns:
            table[split] = 0

    return table[
        [
            "train",
            "val",
            "test",
        ]
    ].reset_index()


def build_source_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        pd.crosstab(
            df[
                "source_dataset"
            ],
            df[
                "split"
            ],
        )
        .reset_index()
    )


def build_gender_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    temp = df.copy()

    temp["gender_clean"] = (
        temp["gender_label"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return (
        temp.groupby(
            [
                "split",
                "candidate_demographic_label",
                "gender_clean",
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )


def plot_split_sizes(
    summary: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        summary["split"],
        summary["n_records"],
    )

    ax.set_title(
        "Final MVP Train / Validation / Test Split"
    )

    ax.set_ylabel(
        "Number of records"
    )

    for i, row in summary.iterrows():

        ax.text(
            i,
            row["n_records"],
            (
                f"{int(row['n_records']):,}\n"
                f"{row['percentage']:.1f}%"
            ),
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "01_split_sizes.png",
        dpi=220,
    )

    plt.close(fig)


def plot_demographic_split(
    table: pd.DataFrame,
) -> None:

    plot_df = table.set_index(
        "candidate_demographic_label"
    )

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title(
        "Demographic Composition Across Data Splits"
    )

    ax.set_xlabel(
        "Demographic label"
    )

    ax.set_ylabel(
        "Number of records"
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "02_demographic_split_counts.png",
        dpi=220,
    )

    plt.close(fig)


def main() -> None:

    print("=" * 72)
    print("FINALISE MVP TRAIN / VAL / TEST SPLITS")
    print("=" * 72)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    validate_labels(
        df
    )

    print(
        f"\nInput records: {len(df):,}"
    )

    checks = structural_checks(
        df
    )

    df["split_key"] = df.apply(
        make_split_key,
        axis=1,
    )

    df["split_value"] = (
        df["split_key"]
        .map(
            deterministic_value
        )
    )

    df["split"] = (
        df["split_value"]
        .map(
            assign_split
        )
    )

    leakage = (
        validate_lfw_identity_leakage(
            df
        )
    )

    summary = build_split_summary(
        df
    )

    label_table = build_label_table(
        df
    )

    source_table = build_source_table(
        df
    )

    gender_table = build_gender_table(
        df
    )

    # Save final training manifest.
    df.to_csv(
        OUTPUT_DIR
        / "final_mvp_with_splits.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "split_summary.csv",
        index=False,
    )

    label_table.to_csv(
        OUTPUT_DIR
        / "demographic_split_counts.csv",
        index=False,
    )

    source_table.to_csv(
        OUTPUT_DIR
        / "source_split_counts.csv",
        index=False,
    )

    gender_table.to_csv(
        OUTPUT_DIR
        / "gender_split_counts.csv",
        index=False,
    )

    checks.to_csv(
        OUTPUT_DIR
        / "structural_checks.csv",
        index=False,
    )

    plot_split_sizes(
        summary
    )

    plot_demographic_split(
        label_table
    )

    print("\n")
    print("=" * 72)
    print("STRUCTURAL CHECKS")
    print("=" * 72)

    print(
        checks.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 72)
    print("SPLIT SUMMARY")
    print("=" * 72)

    print(
        summary.to_string(
            index=False,
            formatters={
                "percentage":
                    lambda x:
                    f"{x:.2f}%"
            },
        )
    )

    print("\n")
    print("=" * 72)
    print("DEMOGRAPHIC COUNTS BY SPLIT")
    print("=" * 72)

    print(
        label_table.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 72)
    print("SOURCE COUNTS BY SPLIT")
    print("=" * 72)

    print(
        source_table.to_string(
            index=False
        )
    )

    print(
        "\nLFW identities appearing "
        "in more than one split:"
    )

    print(
        leakage
    )

    if leakage != 0:
        raise RuntimeError(
            "Identity leakage detected."
        )

    print("\nOutputs:")
    print(OUTPUT_DIR)

    print(
        "\nSplit is deterministic and reproducible."
    )

    print(
        "No images were copied, modified or duplicated."
    )


if __name__ == "__main__":
    main()
    