from __future__ import annotations

import json
from pathlib import Path

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
    / "label_inventory"
)

DATASETS = {
    "fairface": RECORDS_DIR / "fairface_records.csv",
    "utkface": RECORDS_DIR / "utkface_records.csv",
    "lfw": RECORDS_DIR / "lfw_records.csv",
}

LFW_LABEL_CONFIG = (
    ROOT
    / "face_resnet34_training"
    / "configs"
    / "lfw_labels.json"
)

CORE_CATEGORICAL_COLUMNS = [
    "race_label",
    "race_source",
    "gender_label",
    "gender_source",
    "age_group",
    "ita_group",
    "alignment_valid",
    "eligible_demographic_audit",
    "eligible_demographic_audit_overridden",
    "cross_dataset_dup",
]

CORE_NUMERIC_COLUMNS = [
    "age_value",
    "ita_value",
]


def normalise_missing(series: pd.Series) -> pd.Series:
    """Convert common empty / null-like strings to actual missing values."""
    if series.dtype != "object":
        return series

    cleaned = series.copy()

    missing_tokens = {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "<na>",
    }

    cleaned = cleaned.apply(
        lambda value: (
            np.nan
            if isinstance(value, str)
            and value.strip().lower() in missing_tokens
            else value
        )
    )

    return cleaned


def load_dataset(
    dataset: str,
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {dataset} records file:\n{path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    for column in df.columns:
        df[column] = normalise_missing(df[column])

    return df


def build_dataset_summary(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for dataset, df in frames.items():
        row = {
            "dataset": dataset,
            "n_rows": len(df),
            "n_columns": len(df.columns),
        }

        if "image_id" in df.columns:
            row["unique_image_ids"] = (
                df["image_id"].nunique(
                    dropna=True
                )
            )
        else:
            row["unique_image_ids"] = np.nan

        if "identity_id" in df.columns:
            identity = df["identity_id"].dropna()

            row["n_known_identities"] = (
                identity.nunique()
            )

            row["n_rows_with_identity"] = (
                identity.shape[0]
            )
        else:
            row["n_known_identities"] = np.nan
            row["n_rows_with_identity"] = np.nan

        if "image_path" in df.columns:
            row["n_rows_with_image_path"] = (
                df["image_path"].notna().sum()
            )
        else:
            row["n_rows_with_image_path"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def build_missingness(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    important_columns = [
        "image_id",
        "image_path",
        "identity_id",
        "race_label",
        "race_source",
        "gender_label",
        "gender_source",
        "age_value",
        "age_group",
        "ita_value",
        "ita_group",
        "alignment_valid",
        "eligible_demographic_audit",
    ]

    for dataset, df in frames.items():
        for column in important_columns:
            if column not in df.columns:
                rows.append(
                    {
                        "dataset": dataset,
                        "column": column,
                        "column_present": False,
                        "n_total": len(df),
                        "n_known": 0,
                        "n_missing": len(df),
                        "coverage_fraction": 0.0,
                    }
                )
                continue

            known = df[column].notna()

            if df[column].dtype == "object":
                text_values = (
                    df[column]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                known = (
                    known
                    & ~text_values.isin(
                        [
                            "",
                            "unknown",
                            "nan",
                            "none",
                            "null",
                        ]
                    )
                )

            n_known = int(known.sum())
            n_total = len(df)

            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "column_present": True,
                    "n_total": n_total,
                    "n_known": n_known,
                    "n_missing": n_total - n_known,
                    "coverage_fraction": (
                        n_known / n_total
                        if n_total
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_categorical_counts(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for dataset, df in frames.items():
        for column in CORE_CATEGORICAL_COLUMNS:
            if column not in df.columns:
                continue

            series = df[column].copy()

            series = series.where(
                series.notna(),
                "MISSING",
            )

            counts = (
                series
                .astype(str)
                .value_counts(
                    dropna=False
                )
            )

            total = len(df)

            for value, count in counts.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "column": column,
                        "value": value,
                        "count": int(count),
                        "fraction": (
                            count / total
                            if total
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_numeric_summary(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for dataset, df in frames.items():
        for column in CORE_NUMERIC_COLUMNS:
            if column not in df.columns:
                continue

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            ).dropna()

            if values.empty:
                rows.append(
                    {
                        "dataset": dataset,
                        "column": column,
                        "n": 0,
                        "min": np.nan,
                        "p05": np.nan,
                        "p25": np.nan,
                        "median": np.nan,
                        "mean": np.nan,
                        "p75": np.nan,
                        "p95": np.nan,
                        "max": np.nan,
                        "std": np.nan,
                    }
                )
                continue

            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "n": len(values),
                    "min": values.min(),
                    "p05": values.quantile(0.05),
                    "p25": values.quantile(0.25),
                    "median": values.median(),
                    "mean": values.mean(),
                    "p75": values.quantile(0.75),
                    "p95": values.quantile(0.95),
                    "max": values.max(),
                    "std": values.std(),
                }
            )

    return pd.DataFrame(rows)


def build_identity_summary(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for dataset, df in frames.items():
        if "identity_id" not in df.columns:
            continue

        known = df.dropna(
            subset=["identity_id"]
        ).copy()

        if known.empty:
            rows.append(
                {
                    "dataset": dataset,
                    "n_images_with_identity": 0,
                    "n_identities": 0,
                    "singleton_identities": 0,
                    "repeated_identities": 0,
                    "mean_images_per_identity": np.nan,
                    "median_images_per_identity": np.nan,
                    "max_images_per_identity": np.nan,
                }
            )
            continue

        counts = (
            known["identity_id"]
            .astype(str)
            .value_counts()
        )

        rows.append(
            {
                "dataset": dataset,
                "n_images_with_identity": len(known),
                "n_identities": len(counts),
                "singleton_identities": int(
                    (counts == 1).sum()
                ),
                "repeated_identities": int(
                    (counts > 1).sum()
                ),
                "mean_images_per_identity": counts.mean(),
                "median_images_per_identity": counts.median(),
                "max_images_per_identity": counts.max(),
            }
        )

    return pd.DataFrame(rows)


def build_intersection_counts(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    for dataset, df in frames.items():
        required = {
            "race_label",
            "gender_label",
        }

        if not required.issubset(df.columns):
            continue

        temp = df[
            [
                "race_label",
                "gender_label",
            ]
        ].copy()

        temp["race_label"] = (
            temp["race_label"]
            .fillna("MISSING")
            .astype(str)
        )

        temp["gender_label"] = (
            temp["gender_label"]
            .fillna("MISSING")
            .astype(str)
        )

        counts = (
            temp.groupby(
                [
                    "race_label",
                    "gender_label",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="count"
            )
        )

        counts["dataset"] = dataset
        counts["fraction"] = (
            counts["count"] / len(df)
        )

        rows.extend(
            counts[
                [
                    "dataset",
                    "race_label",
                    "gender_label",
                    "count",
                    "fraction",
                ]
            ].to_dict(
                orient="records"
            )
        )

    return pd.DataFrame(rows)


def build_lfw_attribute_inventory(
    lfw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    attr_columns = sorted(
        column
        for column in lfw.columns
        if column.startswith("attr_")
    )

    rows = []

    for column in attr_columns:
        values = pd.to_numeric(
            lfw[column],
            errors="coerce",
        )

        known = values.notna()

        rows.append(
            {
                "column": column,
                "attribute": column.removeprefix(
                    "attr_"
                ),
                "n_total": len(lfw),
                "n_known": int(known.sum()),
                "n_missing": int(
                    (~known).sum()
                ),
                "coverage_fraction": float(
                    known.mean()
                ),
                "n_positive": int(
                    (values == 1).sum()
                ),
                "n_negative": int(
                    (values == 0).sum()
                ),
            }
        )

    config_attributes = []

    if LFW_LABEL_CONFIG.exists():
        config = json.loads(
            LFW_LABEL_CONFIG.read_text(
                encoding="utf-8"
            )
        )

        config_attributes = config.get(
            "attributes",
            [],
        )

    diagnostic = {
        "n_attr_columns_in_lfw_records": (
            len(attr_columns)
        ),
        "n_attributes_in_lfw_labels_config": (
            len(config_attributes)
        ),
        "records_attribute_names": [
            c.removeprefix("attr_")
            for c in attr_columns
        ],
        "config_attribute_names": (
            config_attributes
        ),
        "count_match": (
            len(attr_columns)
            == len(config_attributes)
        ),
    }

    return (
        pd.DataFrame(rows),
        diagnostic,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("HARMONISED DATASET — LABEL INVENTORY")
    print("=" * 72)

    frames = {}

    for dataset, path in DATASETS.items():
        print(
            f"\nLoading {dataset}:"
        )
        print(path)

        df = load_dataset(
            dataset,
            path,
        )

        frames[dataset] = df

        print(
            f"  rows:    {len(df):,}"
        )
        print(
            f"  columns: {len(df.columns):,}"
        )

    dataset_summary = (
        build_dataset_summary(
            frames
        )
    )

    missingness = (
        build_missingness(
            frames
        )
    )

    categorical_counts = (
        build_categorical_counts(
            frames
        )
    )

    numeric_summary = (
        build_numeric_summary(
            frames
        )
    )

    identity_summary = (
        build_identity_summary(
            frames
        )
    )

    intersection_counts = (
        build_intersection_counts(
            frames
        )
    )

    (
        lfw_attributes,
        lfw_diagnostic,
    ) = build_lfw_attribute_inventory(
        frames["lfw"]
    )

    dataset_summary.to_csv(
        OUTPUT_DIR
        / "dataset_summary.csv",
        index=False,
    )

    missingness.to_csv(
        OUTPUT_DIR
        / "label_missingness.csv",
        index=False,
    )

    categorical_counts.to_csv(
        OUTPUT_DIR
        / "categorical_label_counts.csv",
        index=False,
    )

    numeric_summary.to_csv(
        OUTPUT_DIR
        / "numeric_label_summary.csv",
        index=False,
    )

    identity_summary.to_csv(
        OUTPUT_DIR
        / "identity_summary.csv",
        index=False,
    )

    intersection_counts.to_csv(
        OUTPUT_DIR
        / "race_gender_intersections.csv",
        index=False,
    )

    lfw_attributes.to_csv(
        OUTPUT_DIR
        / "lfw_attribute_inventory.csv",
        index=False,
    )

    (
        OUTPUT_DIR
        / "lfw_attribute_diagnostic.json"
    ).write_text(
        json.dumps(
            lfw_diagnostic,
            indent=2,
        ),
        encoding="utf-8",
    )

    combined_metadata = {
        "datasets": list(
            frames.keys()
        ),
        "record_files": {
            dataset: str(path)
            for dataset, path
            in DATASETS.items()
        },
        "lfw_attribute_diagnostic": (
            lfw_diagnostic
        ),
        "note": (
            "This inventory preserves source labels "
            "and does not harmonise or relabel any "
            "image."
        ),
    }

    (
        OUTPUT_DIR
        / "inventory_metadata.json"
    ).write_text(
        json.dumps(
            combined_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n")
    print("=" * 72)
    print("DATASET SUMMARY")
    print("=" * 72)
    print(
        dataset_summary.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 72)
    print("RACE LABELS")
    print("=" * 72)

    race_rows = categorical_counts[
        categorical_counts["column"]
        == "race_label"
    ]

    if race_rows.empty:
        print(
            "No race_label columns found."
        )
    else:
        print(
            race_rows[
                [
                    "dataset",
                    "value",
                    "count",
                    "fraction",
                ]
            ].to_string(
                index=False
            )
        )

    print("\n")
    print("=" * 72)
    print("GENDER LABELS")
    print("=" * 72)

    gender_rows = categorical_counts[
        categorical_counts["column"]
        == "gender_label"
    ]

    if gender_rows.empty:
        print(
            "No gender_label columns found."
        )
    else:
        print(
            gender_rows[
                [
                    "dataset",
                    "value",
                    "count",
                    "fraction",
                ]
            ].to_string(
                index=False
            )
        )

    print("\n")
    print("=" * 72)
    print("ITA GROUPS")
    print("=" * 72)

    ita_rows = categorical_counts[
        categorical_counts["column"]
        == "ita_group"
    ]

    if ita_rows.empty:
        print(
            "No ita_group columns found."
        )
    else:
        print(
            ita_rows[
                [
                    "dataset",
                    "value",
                    "count",
                    "fraction",
                ]
            ].to_string(
                index=False
            )
        )

    print("\n")
    print("=" * 72)
    print("LFW ATTRIBUTE CHECK")
    print("=" * 72)

    print(
        "attr_* columns in records:",
        lfw_diagnostic[
            "n_attr_columns_in_lfw_records"
        ],
    )

    print(
        "attributes in config:",
        lfw_diagnostic[
            "n_attributes_in_lfw_labels_config"
        ],
    )

    print(
        "counts match:",
        lfw_diagnostic[
            "count_match"
        ],
    )

    print("\nOutputs written to:")
    print(OUTPUT_DIR)

    print("\nNo images or source labels were modified.")


if __name__ == "__main__":
    main()
    