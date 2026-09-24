from __future__ import annotations

import re
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


# =====================================================================
# SHARED CONFIGURATION
# =====================================================================

MISSING_TOKENS = {
    "",
    "unknown",
    "missing",
    "none",
    "nan",
    "na",
    "n/a",
    "null",
}

REP001_FIELDS = [
    "gender",
    "race",
    "age_band",
]

REP002_DEMOGRAPHIC_FIELDS = [
    "gender",
    "race",
    "age_band",
]

REP003_SUBGROUP_FIELDS = [
    "gender",
    "race",
    "age_band_harmonised",
]

REP004_INTERSECTIONS = {
    "gender_x_race": [
        "gender",
        "race",
    ],
    "gender_x_age": [
        "gender",
        "age_band_harmonised",
    ],
    "race_x_age": [
        "race",
        "age_band_harmonised",
    ],
    "gender_x_race_x_age": [
        "gender",
        "race",
        "age_band_harmonised",
    ],
}


# =====================================================================
# SHARED — MASTER TABLE LOADING
# =====================================================================

def load_master_table(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load the Phase 0 master annotation table.

    No rows are filtered or removed.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Master table not found: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)

    elif suffix == ".csv":
        df = pd.read_csv(path)

    else:
        raise ValueError(
            f"Unsupported master-table format: {suffix}. "
            "Expected .parquet or .csv."
        )

    required = {
        "image_id",
        "dataset",
        "dataset_version",
        "original_path",
    }

    missing_required = sorted(
        required - set(df.columns)
    )

    if missing_required:
        raise ValueError(
            "Master table is missing required columns: "
            + ", ".join(missing_required)
        )

    if df["image_id"].isna().any():
        raise ValueError(
            "Master table contains missing image_id values."
        )

    duplicated_ids = int(
        df["image_id"].duplicated().sum()
    )

    if duplicated_ids:
        raise ValueError(
            f"Master table contains "
            f"{duplicated_ids} duplicate image_id values."
        )

    return df


# =====================================================================
# SHARED — MISSING / UNKNOWN HANDLING
# =====================================================================

def normalise_reporting_category(
    series: pd.Series,
) -> pd.Series:
    """
    Convert missing/unknown-like values to the reporting label
    'unknown'.

    The original master-table column is never modified.
    """
    out = (
        series
        .astype("string")
        .str.strip()
    )

    lower = out.str.lower()

    missing = (
        out.isna()
        | lower.isin(MISSING_TOKENS)
    )

    return out.mask(
        missing,
        "unknown",
    )


def missing_mask(
    series: pd.Series,
) -> pd.Series:
    """
    True where a value is considered missing/unknown.
    """
    reported = normalise_reporting_category(
        series
    )

    return (
        reported
        .str.lower()
        .eq("unknown")
    )


# =====================================================================
# SHARED — FROZEN AGE GOVERNANCE
# =====================================================================

def load_frozen_age_bands(
    config_path: str | Path = "configs/categories.yaml",
) -> list[dict]:
    """
    Load frozen GOV-002 age-band definitions.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Category config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            "Category configuration is not a valid mapping."
        )

    meta = config.get(
        "meta",
        {},
    )

    if not meta.get(
        "frozen",
        False,
    ):
        raise ValueError(
            "Category configuration is not marked as frozen."
        )

    bands = config.get(
        "age_bands"
    )

    if not bands:
        raise ValueError(
            f"No age_bands found in {config_path}"
        )

    return bands


def _numeric_age_to_band(
    age,
    bands: list[dict],
) -> str:
    """
    Map an exact numeric age to a frozen GOV-002 age band.
    """
    if pd.isna(age):
        return "unknown"

    try:
        age_value = float(age)

    except (TypeError, ValueError):
        return "unknown"

    for band in bands:
        low = float(
            band["min"]
        )

        high = (
            None
            if band["max"] is None
            else float(band["max"])
        )

        if age_value < low:
            continue

        if (
            high is None
            or age_value <= high
        ):
            return str(
                band["name"]
            )

    return "unknown"


def _parse_source_age_band(
    value,
) -> tuple[float, float] | None:
    """
    Parse source/native age bands such as:
        0-2
        10-19
        20-39
        60+
        70+
    """
    if pd.isna(value):
        return None

    text = (
        str(value)
        .strip()
        .lower()
    )

    if text in MISSING_TOKENS:
        return None

    range_match = re.fullmatch(
        r"(\d+)\s*-\s*(\d+)",
        text,
    )

    if range_match:
        return (
            float(range_match.group(1)),
            float(range_match.group(2)),
        )

    plus_match = re.fullmatch(
        r"(\d+)\s*\+",
        text,
    )

    if plus_match:
        return (
            float(plus_match.group(1)),
            float("inf"),
        )

    return None


def _source_band_to_frozen_band(
    source_band,
    bands: list[dict],
) -> str:
    """
    Map a source age interval into a frozen GOV-002 band
    only when it lies completely inside one target band.
    """
    interval = _parse_source_age_band(
        source_band
    )

    if interval is None:
        return "unknown"

    source_low, source_high = interval

    for band in bands:
        target_low = float(
            band["min"]
        )

        target_high = (
            float("inf")
            if band["max"] is None
            else float(band["max"])
        )

        if (
            source_low >= target_low
            and source_high <= target_high
        ):
            return str(
                band["name"]
            )

    return "unknown"


def harmonise_age_band(
    df: pd.DataFrame,
    config_path: str | Path = "configs/categories.yaml",
) -> pd.Series:
    """
    Create a reporting-only harmonised age-band field.

    Priority:
      1. exact numeric age where available;
      2. source/native age band;
      3. unknown.
    """
    required = {
        "age",
        "age_band",
    }

    missing_required = sorted(
        required - set(df.columns)
    )

    if missing_required:
        raise KeyError(
            "Cannot harmonise age. Missing columns: "
            + ", ".join(missing_required)
        )

    bands = load_frozen_age_bands(
        config_path
    )

    values: list[str] = []

    for age, source_band in zip(
        df["age"],
        df["age_band"],
    ):
        if pd.notna(age):
            value = _numeric_age_to_band(
                age,
                bands,
            )
        else:
            value = _source_band_to_frozen_band(
                source_band,
                bands,
            )

        values.append(value)

    return pd.Series(
        values,
        index=df.index,
        dtype="string",
        name="age_band_harmonised",
    )


# =====================================================================
# REP-001 — IMAGE-LEVEL COMPOSITION
# =====================================================================

def dataset_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Image counts and percentages by dataset.
    """
    counts = (
        df.groupby(
            "dataset",
            dropna=False,
        )
        .size()
        .rename("n_images")
        .reset_index()
    )

    total = len(df)

    counts["pct_all_images"] = (
        counts["n_images"]
        / total
        * 100.0
    )

    counts[
        "total_images_all_datasets"
    ] = total

    return (
        counts
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def composition_table(
    df: pd.DataFrame,
    field: str,
) -> pd.DataFrame:
    """
    Image-level categorical composition within each dataset.
    """
    if field not in df.columns:
        raise KeyError(
            f"Required field is absent: {field}"
        )

    work = df[
        [
            "dataset",
            field,
        ]
    ].copy()

    work["category"] = (
        normalise_reporting_category(
            work[field]
        )
    )

    counts = (
        work.groupby(
            [
                "dataset",
                "category",
            ],
            dropna=False,
        )
        .size()
        .rename("n_images")
        .reset_index()
    )

    totals = (
        work.groupby(
            "dataset",
            dropna=False,
        )
        .size()
        .rename("dataset_total")
        .reset_index()
    )

    counts = counts.merge(
        totals,
        on="dataset",
        how="left",
        validate="many_to_one",
    )

    counts["percentage"] = (
        counts["n_images"]
        / counts["dataset_total"]
        * 100.0
    )

    counts.insert(
        1,
        "field",
        field,
    )

    return (
        counts[
            [
                "dataset",
                "field",
                "category",
                "n_images",
                "dataset_total",
                "percentage",
            ]
        ]
        .sort_values(
            [
                "dataset",
                "category",
            ]
        )
        .reset_index(drop=True)
    )


def demographic_availability_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distinguish demographic-label availability from exact
    numeric-age availability.
    """
    specifications = [
        (
            "gender_label",
            "gender",
        ),
        (
            "race_label",
            "race",
        ),
        (
            "age_label",
            "age_band",
        ),
        (
            "exact_numeric_age",
            "age",
        ),
    ]

    rows: list[dict] = []

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        for measure, field in specifications:
            if field not in group.columns:
                continue

            missing = missing_mask(
                group[field]
            )

            n_total = len(group)

            n_missing = int(
                missing.sum()
            )

            n_available = (
                n_total
                - n_missing
            )

            rows.append(
                {
                    "dataset": dataset,
                    "measure": measure,
                    "source_column": field,
                    "n_total": n_total,
                    "n_available": n_available,
                    "n_missing_or_unknown": n_missing,
                    "pct_available": (
                        n_available
                        / n_total
                        * 100.0
                        if n_total
                        else 0.0
                    ),
                    "pct_missing_or_unknown": (
                        n_missing
                        / n_total
                        * 100.0
                        if n_total
                        else 0.0
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "dataset",
                "measure",
            ]
        )
        .reset_index(drop=True)
    )


def missingness_table(
    df: pd.DataFrame,
    fields: Iterable[str],
) -> pd.DataFrame:
    """
    Generic descriptive missingness helper.
    """
    rows: list[dict] = []

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        for field in fields:
            if field not in group.columns:
                continue

            missing = missing_mask(
                group[field]
            )

            n_total = len(group)

            n_missing = int(
                missing.sum()
            )

            n_observed = (
                n_total
                - n_missing
            )

            rows.append(
                {
                    "dataset": dataset,
                    "field": field,
                    "n_total": n_total,
                    "n_observed": n_observed,
                    "n_missing_or_unknown": n_missing,
                    "pct_observed": (
                        n_observed
                        / n_total
                        * 100.0
                        if n_total
                        else 0.0
                    ),
                    "pct_missing_or_unknown": (
                        n_missing
                        / n_total
                        * 100.0
                        if n_total
                        else 0.0
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "dataset",
                "field",
            ]
        )
        .reset_index(drop=True)
    )


def provenance_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise demographic label provenance.
    """
    source_fields = [
        "gender_source",
        "race_source",
        "age_source",
    ]

    frames: list[pd.DataFrame] = []

    for field in source_fields:
        if field not in df.columns:
            continue

        frames.append(
            composition_table(
                df,
                field,
            )
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def audit_flag_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise Phase 0 flags relevant to representation.

    Null flag values remain distinct from False.
    """
    flag_fields = [
        "corrupt",
        "parse_failed",
        "cross_dataset_dup",
        "low_confidence",
        "thin_cell",
        "eligible_demographic_audit",
    ]

    rows: list[dict] = []

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        for field in flag_fields:
            if field not in group.columns:
                continue

            values = group[field]

            n_total = len(group)

            n_null = int(
                values.isna().sum()
            )

            n_known = (
                n_total
                - n_null
            )

            n_true = int(
                values.eq(True)
                .fillna(False)
                .sum()
            )

            n_false = int(
                values.eq(False)
                .fillna(False)
                .sum()
            )

            if (
                n_true
                + n_false
                + n_null
                != n_total
            ):
                raise ValueError(
                    f"Unexpected flag encoding in "
                    f"{dataset}/{field}."
                )

            rows.append(
                {
                    "dataset": dataset,
                    "flag": field,
                    "n_total": n_total,
                    "n_known": n_known,
                    "n_true": n_true,
                    "n_false": n_false,
                    "n_null": n_null,
                    "pct_known": (
                        n_known
                        / n_total
                        * 100.0
                        if n_total
                        else 0.0
                    ),
                    "pct_true_among_known": (
                        n_true
                        / n_known
                        * 100.0
                        if n_known
                        else None
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "dataset",
                "flag",
            ]
        )
        .reset_index(drop=True)
    )


# =====================================================================
# REP-002 — IDENTITY-LEVEL REPRESENTATION
# =====================================================================

def subject_id_present_mask(
    series: pd.Series,
) -> pd.Series:
    """
    True where a usable identity identifier exists.
    """
    reported = normalise_reporting_category(
        series
    )

    return (
        reported
        .str.lower()
        .ne("unknown")
    )


def identity_support_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise subject-ID availability by dataset.
    """
    if "subject_id" not in df.columns:
        raise KeyError(
            "Master table does not contain subject_id."
        )

    rows: list[dict] = []

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        subject = normalise_reporting_category(
            group["subject_id"]
        )

        present = (
            subject
            .str.lower()
            .ne("unknown")
        )

        valid_subjects = subject[
            present
        ]

        counts = (
            valid_subjects
            .value_counts()
        )

        n_rows = len(group)

        n_present = int(
            present.sum()
        )

        n_missing = (
            n_rows
            - n_present
        )

        n_identities = int(
            valid_subjects.nunique()
        )

        n_singletons = (
            int(
                (counts == 1).sum()
            )
            if not counts.empty
            else 0
        )

        n_repeated = (
            int(
                (counts > 1).sum()
            )
            if not counts.empty
            else 0
        )

        max_images = (
            int(counts.max())
            if not counts.empty
            else 0
        )

        rows.append(
            {
                "dataset": dataset,
                "n_images": n_rows,
                "n_images_with_subject_id":
                    n_present,
                "n_images_without_subject_id":
                    n_missing,
                "pct_images_with_subject_id": (
                    n_present
                    / n_rows
                    * 100.0
                    if n_rows
                    else 0.0
                ),
                "n_unique_identities":
                    n_identities,
                "n_singleton_identities":
                    n_singletons,
                "n_repeated_identities":
                    n_repeated,
                "max_images_per_identity":
                    max_images,
                "subject_id_available":
                    n_present > 0,
                "repeated_identity_structure_present":
                    n_repeated > 0,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def build_identity_attribute_table(
    df: pd.DataFrame,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build one demographic-status row per identity and field.

    status:
        unavailable
        consistent
        conflicting
    """
    if fields is None:
        fields = REP002_DEMOGRAPHIC_FIELDS

    if "subject_id" not in df.columns:
        raise KeyError(
            "Master table does not contain subject_id."
        )

    rows: list[dict] = []

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        work = group.copy()

        work["_subject_rep"] = (
            normalise_reporting_category(
                work["subject_id"]
            )
        )

        work = work[
            work["_subject_rep"]
            .str.lower()
            .ne("unknown")
        ]

        if work.empty:
            continue

        for (
            subject_id,
            subject_group,
        ) in work.groupby(
            "_subject_rep",
            dropna=False,
        ):
            n_images = len(
                subject_group
            )

            for field in fields:
                if field not in subject_group.columns:
                    continue

                values = (
                    normalise_reporting_category(
                        subject_group[field]
                    )
                )

                observed = (
                    values[
                        values
                        .str.lower()
                        .ne("unknown")
                    ]
                    .drop_duplicates()
                    .tolist()
                )

                if len(observed) == 0:
                    status = "unavailable"
                    category = "unknown"

                elif len(observed) == 1:
                    status = "consistent"
                    category = observed[0]

                else:
                    status = "conflicting"
                    category = "conflicting"

                rows.append(
                    {
                        "dataset": dataset,
                        "subject_id":
                            subject_id,
                        "field": field,
                        "n_images_for_identity":
                            n_images,
                        "status": status,
                        "category":
                            category,
                        "n_unique_observed_values":
                            len(observed),
                    }
                )

    columns = [
        "dataset",
        "subject_id",
        "field",
        "n_images_for_identity",
        "status",
        "category",
        "n_unique_observed_values",
    ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def identity_demographic_availability(
    df: pd.DataFrame,
    identity_table: pd.DataFrame,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """
    Determine whether identity-level demographic analysis
    is supported for each dataset and field.
    """
    if fields is None:
        fields = REP002_DEMOGRAPHIC_FIELDS

    support = identity_support_summary(
        df
    )

    rows: list[dict] = []

    for _, support_row in support.iterrows():
        dataset = support_row[
            "dataset"
        ]

        n_identities = int(
            support_row[
                "n_unique_identities"
            ]
        )

        for field in fields:
            if n_identities == 0:
                rows.append(
                    {
                        "dataset": dataset,
                        "field": field,
                        "n_total_identities": 0,
                        "n_consistent": 0,
                        "n_unavailable": 0,
                        "n_conflicting": 0,
                        "pct_consistent": None,
                        "pct_unavailable": None,
                        "pct_conflicting": None,
                        "analysis_status":
                            "subject_id_unavailable",
                    }
                )

                continue

            subset = identity_table[
                (
                    identity_table[
                        "dataset"
                    ]
                    == dataset
                )
                &
                (
                    identity_table[
                        "field"
                    ]
                    == field
                )
            ]

            n_consistent = int(
                (
                    subset["status"]
                    == "consistent"
                ).sum()
            )

            n_unavailable = int(
                (
                    subset["status"]
                    == "unavailable"
                ).sum()
            )

            n_conflicting = int(
                (
                    subset["status"]
                    == "conflicting"
                ).sum()
            )

            total_checked = (
                n_consistent
                + n_unavailable
                + n_conflicting
            )

            if total_checked != n_identities:
                raise ValueError(
                    f"Identity accounting mismatch for "
                    f"{dataset}/{field}: "
                    f"{total_checked} != {n_identities}"
                )

            if n_consistent == n_identities:
                analysis_status = (
                    "fully_available"
                )

            elif n_consistent > 0:
                analysis_status = (
                    "partially_available"
                )

            else:
                analysis_status = (
                    "demographic_label_unavailable"
                )

            rows.append(
                {
                    "dataset": dataset,
                    "field": field,
                    "n_total_identities":
                        n_identities,
                    "n_consistent":
                        n_consistent,
                    "n_unavailable":
                        n_unavailable,
                    "n_conflicting":
                        n_conflicting,
                    "pct_consistent": (
                        n_consistent
                        / n_identities
                        * 100.0
                    ),
                    "pct_unavailable": (
                        n_unavailable
                        / n_identities
                        * 100.0
                    ),
                    "pct_conflicting": (
                        n_conflicting
                        / n_identities
                        * 100.0
                    ),
                    "analysis_status":
                        analysis_status,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "dataset",
                "field",
            ]
        )
        .reset_index(drop=True)
    )


def identity_demographic_composition(
    identity_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identity-level demographic composition using only
    consistently labelled identities.
    """
    output_columns = [
        "dataset",
        "field",
        "category",
        "n_identities",
        "n_consistently_labelled_identities",
        "n_total_identities",
        "pct_of_labelled_identities",
        "pct_of_all_identities",
    ]

    if identity_table.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    totals = (
        identity_table.groupby(
            [
                "dataset",
                "field",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "n_total_identities"
        )
        .reset_index()
    )

    labelled = identity_table[
        identity_table[
            "status"
        ]
        == "consistent"
    ].copy()

    if labelled.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    labelled_totals = (
        labelled.groupby(
            [
                "dataset",
                "field",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "n_consistently_labelled_identities"
        )
        .reset_index()
    )

    counts = (
        labelled.groupby(
            [
                "dataset",
                "field",
                "category",
            ],
            dropna=False,
        )
        .size()
        .rename("n_identities")
        .reset_index()
    )

    counts = counts.merge(
        labelled_totals,
        on=[
            "dataset",
            "field",
        ],
        how="left",
        validate="many_to_one",
    )

    counts = counts.merge(
        totals,
        on=[
            "dataset",
            "field",
        ],
        how="left",
        validate="many_to_one",
    )

    counts[
        "pct_of_labelled_identities"
    ] = (
        counts["n_identities"]
        / counts[
            "n_consistently_labelled_identities"
        ]
        * 100.0
    )

    counts[
        "pct_of_all_identities"
    ] = (
        counts["n_identities"]
        / counts["n_total_identities"]
        * 100.0
    )

    return (
        counts[
            output_columns
        ]
        .sort_values(
            [
                "dataset",
                "field",
                "category",
            ]
        )
        .reset_index(drop=True)
    )


# =====================================================================
# REP-003 — IMAGES-PER-IDENTITY DISTRIBUTION
# =====================================================================

def identity_image_counts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per usable identity with the number of images.
    """
    if "subject_id" not in df.columns:
        raise KeyError(
            "Master table does not contain subject_id."
        )

    frames: list[pd.DataFrame] = []

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        subject = normalise_reporting_category(
            group["subject_id"]
        )

        present = (
            subject
            .str.lower()
            .ne("unknown")
        )

        if not present.any():
            continue

        work = pd.DataFrame(
            {
                "dataset": dataset,
                "subject_id":
                    subject[present],
            }
        )

        counts = (
            work.groupby(
                [
                    "dataset",
                    "subject_id",
                ],
                dropna=False,
            )
            .size()
            .rename("n_images")
            .reset_index()
        )

        frames.append(counts)

    columns = [
        "dataset",
        "subject_id",
        "n_images",
    ]

    if not frames:
        return pd.DataFrame(
            columns=columns
        )

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "dataset",
                "n_images",
                "subject_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def identity_depth_summary(
    df: pd.DataFrame,
    identity_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summary statistics for images per identity.
    """
    support = identity_support_summary(
        df
    )

    rows: list[dict] = []

    for _, support_row in support.iterrows():
        dataset = support_row[
            "dataset"
        ]

        subset = identity_counts[
            identity_counts[
                "dataset"
            ]
            == dataset
        ]

        if subset.empty:
            rows.append(
                {
                    "dataset":
                        dataset,
                    "analysis_status":
                        "subject_id_unavailable",
                    "n_identities":
                        0,
                    "n_images_with_identity":
                        0,
                    "min_images_per_identity":
                        None,
                    "q25_images_per_identity":
                        None,
                    "median_images_per_identity":
                        None,
                    "mean_images_per_identity":
                        None,
                    "q75_images_per_identity":
                        None,
                    "p90_images_per_identity":
                        None,
                    "p95_images_per_identity":
                        None,
                    "p99_images_per_identity":
                        None,
                    "max_images_per_identity":
                        None,
                    "n_singleton_identities":
                        0,
                    "pct_singleton_identities":
                        None,
                    "n_repeated_identities":
                        0,
                    "pct_repeated_identities":
                        None,
                }
            )

            continue

        values = (
            subset["n_images"]
            .astype(float)
        )

        n_identities = len(
            subset
        )

        n_singletons = int(
            (values == 1).sum()
        )

        n_repeated = int(
            (values > 1).sum()
        )

        rows.append(
            {
                "dataset":
                    dataset,
                "analysis_status":
                    "identity_distribution_available",
                "n_identities":
                    n_identities,
                "n_images_with_identity":
                    int(values.sum()),
                "min_images_per_identity":
                    int(values.min()),
                "q25_images_per_identity":
                    float(values.quantile(0.25)),
                "median_images_per_identity":
                    float(values.median()),
                "mean_images_per_identity":
                    float(values.mean()),
                "q75_images_per_identity":
                    float(values.quantile(0.75)),
                "p90_images_per_identity":
                    float(values.quantile(0.90)),
                "p95_images_per_identity":
                    float(values.quantile(0.95)),
                "p99_images_per_identity":
                    float(values.quantile(0.99)),
                "max_images_per_identity":
                    int(values.max()),
                "n_singleton_identities":
                    n_singletons,
                "pct_singleton_identities": (
                    n_singletons
                    / n_identities
                    * 100.0
                ),
                "n_repeated_identities":
                    n_repeated,
                "pct_repeated_identities": (
                    n_repeated
                    / n_identities
                    * 100.0
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def identity_depth_distribution(
    identity_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Exact empirical images-per-identity distribution.
    """
    columns = [
        "dataset",
        "n_images_per_identity",
        "n_identities",
        "total_identities",
        "pct_identities",
        "images_contributed",
        "total_images_with_identity",
        "pct_images_contributed",
    ]

    if identity_counts.empty:
        return pd.DataFrame(
            columns=columns
        )

    distribution = (
        identity_counts.groupby(
            [
                "dataset",
                "n_images",
            ],
            dropna=False,
        )
        .size()
        .rename("n_identities")
        .reset_index()
        .rename(
            columns={
                "n_images":
                    "n_images_per_identity"
            }
        )
    )

    identity_totals = (
        identity_counts.groupby(
            "dataset"
        )
        .size()
        .rename(
            "total_identities"
        )
        .reset_index()
    )

    image_totals = (
        identity_counts.groupby(
            "dataset"
        )["n_images"]
        .sum()
        .rename(
            "total_images_with_identity"
        )
        .reset_index()
    )

    distribution = distribution.merge(
        identity_totals,
        on="dataset",
        how="left",
        validate="many_to_one",
    )

    distribution = distribution.merge(
        image_totals,
        on="dataset",
        how="left",
        validate="many_to_one",
    )

    distribution[
        "pct_identities"
    ] = (
        distribution["n_identities"]
        / distribution["total_identities"]
        * 100.0
    )

    distribution[
        "images_contributed"
    ] = (
        distribution[
            "n_images_per_identity"
        ]
        * distribution[
            "n_identities"
        ]
    )

    distribution[
        "pct_images_contributed"
    ] = (
        distribution[
            "images_contributed"
        ]
        / distribution[
            "total_images_with_identity"
        ]
        * 100.0
    )

    return (
        distribution[
            columns
        ]
        .sort_values(
            [
                "dataset",
                "n_images_per_identity",
            ]
        )
        .reset_index(drop=True)
    )


def identity_depth_subgroup_summary(
    identity_counts: pd.DataFrame,
    identity_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Images-per-identity summary by demographic subgroup
    where consistent identity-level labels exist.
    """
    columns = [
        "dataset",
        "field",
        "category",
        "n_identities",
        "n_images",
        "min_images_per_identity",
        "q25_images_per_identity",
        "median_images_per_identity",
        "mean_images_per_identity",
        "q75_images_per_identity",
        "p90_images_per_identity",
        "max_images_per_identity",
        "n_singleton_identities",
        "pct_singleton_identities",
    ]

    if (
        identity_counts.empty
        or identity_table.empty
    ):
        return pd.DataFrame(
            columns=columns
        )

    labels = identity_table[
        identity_table[
            "status"
        ]
        == "consistent"
    ][
        [
            "dataset",
            "subject_id",
            "field",
            "category",
        ]
    ].copy()

    if labels.empty:
        return pd.DataFrame(
            columns=columns
        )

    merged = labels.merge(
        identity_counts,
        on=[
            "dataset",
            "subject_id",
        ],
        how="inner",
        validate="many_to_one",
    )

    rows: list[dict] = []

    for (
        dataset,
        field,
        category,
    ), group in merged.groupby(
        [
            "dataset",
            "field",
            "category",
        ],
        dropna=False,
    ):
        values = (
            group["n_images"]
            .astype(float)
        )

        n_identities = len(
            group
        )

        n_singletons = int(
            (values == 1).sum()
        )

        rows.append(
            {
                "dataset":
                    dataset,
                "field":
                    field,
                "category":
                    category,
                "n_identities":
                    n_identities,
                "n_images":
                    int(values.sum()),
                "min_images_per_identity":
                    int(values.min()),
                "q25_images_per_identity":
                    float(values.quantile(0.25)),
                "median_images_per_identity":
                    float(values.median()),
                "mean_images_per_identity":
                    float(values.mean()),
                "q75_images_per_identity":
                    float(values.quantile(0.75)),
                "p90_images_per_identity":
                    float(values.quantile(0.90)),
                "max_images_per_identity":
                    int(values.max()),
                "n_singleton_identities":
                    n_singletons,
                "pct_singleton_identities": (
                    n_singletons
                    / n_identities
                    * 100.0
                ),
            }
        )

    return (
        pd.DataFrame(
            rows,
            columns=columns,
        )
        .sort_values(
            [
                "dataset",
                "field",
                "category",
            ]
        )
        .reset_index(drop=True)
    )


def identity_depth_subgroup_distribution(
    identity_counts: pd.DataFrame,
    identity_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Exact images-per-identity distribution by subgroup.
    """
    columns = [
        "dataset",
        "field",
        "category",
        "n_images_per_identity",
        "n_identities",
        "total_identities_in_subgroup",
        "pct_identities_in_subgroup",
    ]

    if (
        identity_counts.empty
        or identity_table.empty
    ):
        return pd.DataFrame(
            columns=columns
        )

    labels = identity_table[
        identity_table[
            "status"
        ]
        == "consistent"
    ][
        [
            "dataset",
            "subject_id",
            "field",
            "category",
        ]
    ].copy()

    if labels.empty:
        return pd.DataFrame(
            columns=columns
        )

    merged = labels.merge(
        identity_counts,
        on=[
            "dataset",
            "subject_id",
        ],
        how="inner",
        validate="many_to_one",
    )

    counts = (
        merged.groupby(
            [
                "dataset",
                "field",
                "category",
                "n_images",
            ],
            dropna=False,
        )
        .size()
        .rename("n_identities")
        .reset_index()
        .rename(
            columns={
                "n_images":
                    "n_images_per_identity"
            }
        )
    )

    totals = (
        merged.groupby(
            [
                "dataset",
                "field",
                "category",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "total_identities_in_subgroup"
        )
        .reset_index()
    )

    counts = counts.merge(
        totals,
        on=[
            "dataset",
            "field",
            "category",
        ],
        how="left",
        validate="many_to_one",
    )

    counts[
        "pct_identities_in_subgroup"
    ] = (
        counts["n_identities"]
        / counts[
            "total_identities_in_subgroup"
        ]
        * 100.0
    )

    return (
        counts[
            columns
        ]
        .sort_values(
            [
                "dataset",
                "field",
                "category",
                "n_images_per_identity",
            ]
        )
        .reset_index(drop=True)
    )


# =====================================================================
# REP-004 — INTERSECTIONAL COMPOSITION
# =====================================================================

def load_image_cell_thresholds(
    config_path: str | Path = "configs/reporting.yaml",
) -> dict:
    """
    Load frozen GOV-003 image-level cell-size rules.
    """
    config_path = Path(
        config_path
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Reporting config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if not isinstance(
        config,
        dict,
    ):
        raise ValueError(
            "Reporting configuration is not a valid mapping."
        )

    meta = config.get(
        "meta",
        {},
    )

    if not meta.get(
        "frozen",
        False,
    ):
        raise ValueError(
            "Reporting configuration is not marked as frozen."
        )

    minimum = (
        config
        .get(
            "minimum_cell_size",
            {},
        )
        .get(
            "image",
            {},
        )
    )

    suppress_below = minimum.get(
        "suppress_below"
    )

    low_confidence_below = minimum.get(
        "low_confidence_below"
    )

    if (
        suppress_below is None
        or low_confidence_below is None
    ):
        raise ValueError(
            "Image minimum-cell-size thresholds "
            "are missing."
        )

    suppress_below = int(
        suppress_below
    )

    low_confidence_below = int(
        low_confidence_below
    )

    if suppress_below < 0:
        raise ValueError(
            "suppress_below must be non-negative."
        )

    if (
        low_confidence_below
        < suppress_below
    ):
        raise ValueError(
            "low_confidence_below cannot be "
            "smaller than suppress_below."
        )

    return {
        "suppress_below":
            suppress_below,
        "low_confidence_below":
            low_confidence_below,
    }


def _analysis_boolean_mask(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a boolean-like eligibility column into
    a strict analysis mask.

    Null values are treated as False.
    """
    if pd.api.types.is_bool_dtype(
        series.dtype
    ):
        return (
            series
            .fillna(False)
            .astype(bool)
        )

    text = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    true_values = {
        "true",
        "1",
        "yes",
    }

    false_values = {
        "false",
        "0",
        "no",
        "",
        "unknown",
        "nan",
        "none",
        "null",
    }

    known = (
        text.isna()
        | text.isin(
            true_values
        )
        | text.isin(
            false_values
        )
    )

    if not known.all():
        bad = sorted(
            text[
                ~known
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unexpected boolean encoding: "
            + ", ".join(
                map(str, bad)
            )
        )

    return text.isin(
        true_values
    )


def demographic_eligibility_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise Phase 0 demographic-analysis eligibility.
    """
    field = (
        "eligible_demographic_audit"
    )

    if field not in df.columns:
        raise KeyError(
            f"Master table does not contain {field}."
        )

    rows: list[dict] = []

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        eligible = (
            _analysis_boolean_mask(
                group[field]
            )
        )

        n_total = len(group)

        n_eligible = int(
            eligible.sum()
        )

        n_ineligible = (
            n_total
            - n_eligible
        )

        rows.append(
            {
                "dataset":
                    dataset,
                "n_total_images":
                    n_total,
                "n_eligible_images":
                    n_eligible,
                "n_ineligible_images":
                    n_ineligible,
                "pct_eligible_images": (
                    n_eligible
                    / n_total
                    * 100.0
                    if n_total
                    else 0.0
                ),
                "analysis_status": (
                    "available"
                    if n_eligible > 0
                    else
                    "no_eligible_demographic_rows"
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def _reporting_status(
    n: int,
    suppress_below: int,
    low_confidence_below: int,
) -> tuple[str, bool, bool]:
    """
    Apply frozen GOV-003 image-cell reporting rules.

    n < suppress_below
        -> suppressed

    suppress_below <= n < low_confidence_below
        -> low confidence

    n >= low_confidence_below
        -> reportable
    """
    if n < suppress_below:
        return (
            "suppressed",
            True,
            False,
        )

    if n < low_confidence_below:
        return (
            "low_confidence",
            False,
            True,
        )

    return (
        "reportable",
        False,
        False,
    )


def intersectional_composition(
    df: pd.DataFrame,
    intersection_name: str,
    fields: list[str],
    reporting_config_path: str | Path = "configs/reporting.yaml",
) -> pd.DataFrame:
    """
    Calculate one pre-specified intersectional composition table.

    Only rows eligible for demographic analysis contribute
    to the denominator.

    Zero-count cells remain represented.

    Raw percentages are retained for reproducibility.
    percentage_reported is blank for suppressed cells.
    """
    required = {
        "dataset",
        "eligible_demographic_audit",
        *fields,
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"Cannot calculate {intersection_name}. "
            "Missing fields: "
            + ", ".join(missing)
        )

    thresholds = (
        load_image_cell_thresholds(
            reporting_config_path
        )
    )

    suppress_below = thresholds[
        "suppress_below"
    ]

    low_confidence_below = thresholds[
        "low_confidence_below"
    ]

    rows: list[dict] = []

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        eligible_mask = (
            _analysis_boolean_mask(
                group[
                    "eligible_demographic_audit"
                ]
            )
        )

        eligible = group[
            eligible_mask
        ].copy()

        if eligible.empty:
            continue

        for field in fields:
            eligible[field] = (
                normalise_reporting_category(
                    eligible[field]
                )
            )

        category_levels: list[
            list[str]
        ] = []

        for field in fields:
            values = sorted(
                eligible[field]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if not values:
                values = [
                    "unknown"
                ]

            category_levels.append(
                values
            )

        counts = (
            eligible.groupby(
                fields,
                dropna=False,
            )
            .size()
            .rename("n_images")
        )

        dataset_eligible_n = len(
            eligible
        )

        for combination in product(
            *category_levels
        ):
            if len(fields) == 1:
                lookup_key = (
                    combination[0]
                )
            else:
                lookup_key = tuple(
                    combination
                )

            n_images = int(
                counts.get(
                    lookup_key,
                    0,
                )
            )

            (
                status,
                thin_cell,
                low_confidence,
            ) = _reporting_status(
                n_images,
                suppress_below,
                low_confidence_below,
            )

            percentage_raw = (
                n_images
                / dataset_eligible_n
                * 100.0
                if dataset_eligible_n
                else 0.0
            )

            percentage_reported = (
                None
                if status == "suppressed"
                else percentage_raw
            )

            row = {
                "dataset":
                    dataset,
                "intersection":
                    intersection_name,
                "n_images":
                    n_images,
                "dataset_eligible_n":
                    dataset_eligible_n,
                "percentage":
                    percentage_raw,
                "percentage_reported":
                    percentage_reported,
                "suppress_below":
                    suppress_below,
                "low_confidence_below":
                    low_confidence_below,
                "thin_cell":
                    thin_cell,
                "low_confidence":
                    low_confidence,
                "reporting_status":
                    status,
            }

            labels: list[str] = []

            for field, value in zip(
                fields,
                combination,
            ):
                row[field] = value

                labels.append(
                    f"{field}={value}"
                )

            row["cell_label"] = (
                " | ".join(
                    labels
                )
            )

            rows.append(
                row
            )

    base_columns = [
        "dataset",
        "intersection",
        "cell_label",
    ]

    category_columns = [
        "gender",
        "race",
        "age_band_harmonised",
    ]

    result_columns = [
        "n_images",
        "dataset_eligible_n",
        "percentage",
        "percentage_reported",
        "suppress_below",
        "low_confidence_below",
        "thin_cell",
        "low_confidence",
        "reporting_status",
    ]

    columns = (
        base_columns
        + category_columns
        + result_columns
    )

    if not rows:
        return pd.DataFrame(
            columns=columns
        )

    result = pd.DataFrame(
        rows
    )

    for column in category_columns:
        if column not in result.columns:
            result[column] = pd.NA

    return (
        result[
            columns
        ]
        .sort_values(
            [
                "dataset",
                "intersection",
                "cell_label",
            ]
        )
        .reset_index(drop=True)
    )


def all_intersectional_composition(
    df: pd.DataFrame,
    reporting_config_path: str | Path = "configs/reporting.yaml",
) -> pd.DataFrame:
    """
    Run all pre-specified REP-004 intersections.
    """
    frames: list[pd.DataFrame] = []

    for (
        intersection_name,
        fields,
    ) in REP004_INTERSECTIONS.items():

        frames.append(
            intersectional_composition(
                df,
                intersection_name,
                fields,
                reporting_config_path,
            )
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


def intersection_reporting_summary(
    composition: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count reportable, low-confidence, suppressed and zero
    cells for each dataset/intersection.
    """
    columns = [
        "dataset",
        "intersection",
        "n_cells",
        "n_reportable_cells",
        "n_low_confidence_cells",
        "n_suppressed_cells",
        "n_zero_cells",
    ]

    if composition.empty:
        return pd.DataFrame(
            columns=columns
        )

    rows: list[dict] = []

    for (
        dataset,
        intersection,
    ), group in composition.groupby(
        [
            "dataset",
            "intersection",
        ],
        dropna=False,
    ):
        rows.append(
            {
                "dataset":
                    dataset,
                "intersection":
                    intersection,
                "n_cells":
                    len(group),
                "n_reportable_cells":
                    int(
                        (
                            group[
                                "reporting_status"
                            ]
                            == "reportable"
                        ).sum()
                    ),
                "n_low_confidence_cells":
                    int(
                        (
                            group[
                                "reporting_status"
                            ]
                            == "low_confidence"
                        ).sum()
                    ),
                "n_suppressed_cells":
                    int(
                        (
                            group[
                                "reporting_status"
                            ]
                            == "suppressed"
                        ).sum()
                    ),
                "n_zero_cells":
                    int(
                        (
                            group[
                                "n_images"
                            ]
                            == 0
                        ).sum()
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows,
            columns=columns,
        )
        .sort_values(
            [
                "dataset",
                "intersection",
            ]
        )
        .reset_index(drop=True)
    )