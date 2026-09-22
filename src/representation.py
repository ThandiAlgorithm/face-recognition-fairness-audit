from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


# =====================================================================
# REP-001 configuration
# =====================================================================

# Native/source demographic fields used for descriptive composition.
REP001_FIELDS = [
    "gender",
    "race",
    "age_band",
]

# Values that should be represented as "unknown" for reporting.
# The source master table itself is never modified.
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


# =====================================================================
# Master-table loading and structural validation
# =====================================================================

def load_master_table(path: str | Path) -> pd.DataFrame:
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
# Missing / unknown handling
# =====================================================================

def normalise_reporting_category(
    series: pd.Series,
) -> pd.Series:
    """
    Create a reporting-only version of a categorical column.

    Missing/null/unknown-like values are represented as "unknown".

    The original Series and master table are not modified.
    """
    out = series.astype("string").str.strip()

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
    Return True for values treated as missing/unknown
    for reporting purposes.
    """
    reported = normalise_reporting_category(
        series
    )

    return reported.str.lower().eq(
        "unknown"
    )


# =====================================================================
# Frozen age-governance handling
# =====================================================================

def load_frozen_age_bands(
    config_path: str | Path = "configs/categories.yaml",
) -> list[dict]:
    """
    Load the frozen harmonised age-band definitions
    from the governance configuration.
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

    meta = config.get("meta", {})

    if not meta.get("frozen", False):
        raise ValueError(
            "The category configuration is not marked as frozen."
        )

    bands = config.get("age_bands")

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
    Map an exact numeric age to one of the frozen GOV-002
    harmonised age bands.
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

        if high is None or age_value <= high:
            return str(
                band["name"]
            )

    return "unknown"


def _parse_source_age_band(
    value,
) -> tuple[float, float] | None:
    """
    Parse native/source age-band strings.

    Supported examples:
        0-2
        3-9
        10-19
        20-29
        0-19
        60+
        70+

    Returns:
        (lower_bound, upper_bound)

    Open-ended upper bounds use positive infinity.
    """
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in MISSING_TOKENS:
        return None

    range_match = re.fullmatch(
        r"(\d+)\s*-\s*(\d+)",
        text,
    )

    if range_match:
        lower = float(
            range_match.group(1)
        )

        upper = float(
            range_match.group(2)
        )

        return lower, upper

    plus_match = re.fullmatch(
        r"(\d+)\s*\+",
        text,
    )

    if plus_match:
        lower = float(
            plus_match.group(1)
        )

        return lower, float("inf")

    return None


def _source_band_to_frozen_band(
    source_band,
    bands: list[dict],
) -> str:
    """
    Map a native/source categorical age interval into one
    frozen GOV-002 age band.

    A source interval is mapped only if it lies completely
    inside a single frozen target interval.
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
        1. Exact numeric age where available.
        2. Native/source age_band where the interval nests
           completely inside a frozen GOV-002 age band.
        3. "unknown".

    This function does not modify the master table.
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

        # Prefer exact age where available.
        if pd.notna(age):

            harmonised = _numeric_age_to_band(
                age,
                bands,
            )

        else:

            harmonised = _source_band_to_frozen_band(
                source_band,
                bands,
            )

        values.append(
            harmonised
        )

    return pd.Series(
        values,
        index=df.index,
        dtype="string",
        name="age_band_harmonised",
    )


# =====================================================================
# REP-001 composition tables
# =====================================================================

def dataset_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Overall image counts and percentages by dataset.
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
    Calculate image-level counts and percentages for
    one categorical field within each dataset.

    All rows remain in the denominator.
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


# =====================================================================
# Demographic availability
# =====================================================================

def demographic_availability_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distinguish demographic-label availability from
    exact numeric-age availability.

    This prevents a dataset such as FairFace from being
    incorrectly described as having no age information
    simply because exact numeric ages are unavailable.
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


# =====================================================================
# Optional generic missingness helper
# =====================================================================

def missingness_table(
    df: pd.DataFrame,
    fields: Iterable[str],
) -> pd.DataFrame:
    """
    Generic descriptive missing/unknown summary.

    Retained as a reusable helper for later REP analyses.
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


# =====================================================================
# Label provenance
# =====================================================================

def provenance_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise demographic provenance.

    This distinguishes native, derived and unavailable
    demographic information.
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

        table = composition_table(
            df,
            field,
        )

        frames.append(
            table
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# =====================================================================
# Phase 0 flag summary
# =====================================================================

def audit_flag_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise Phase 0 flags relevant to representation.

    Missing/null flag values remain distinct from False.

    REP-001 does not use these flags to delete rows.
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
# REP-002 — Identity-level representation
# =====================================================================

REP002_DEMOGRAPHIC_FIELDS = [
    "gender",
    "race",
    "age_band",
]


def subject_id_present_mask(
    series: pd.Series,
) -> pd.Series:
    """
    Return True where a usable subject/identity identifier exists.

    Missing and unknown-like values are treated as unavailable.
    """
    reported = normalise_reporting_category(
        series
    )

    return ~reported.str.lower().eq(
        "unknown"
    )


def identity_support_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise identity-ID availability by dataset.

    This function reports what the Phase 0 hand-off contains.
    It does not infer or manufacture identities.
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

        present = ~subject.str.lower().eq(
            "unknown"
        )

        valid_subjects = subject[present]

        counts = valid_subjects.value_counts()

        n_rows = len(group)
        n_present = int(present.sum())
        n_missing = n_rows - n_present
        n_identities = int(
            valid_subjects.nunique()
        )

        n_singletons = int(
            (counts == 1).sum()
        ) if not counts.empty else 0

        n_repeated = int(
            (counts > 1).sum()
        ) if not counts.empty else 0

        max_images = (
            int(counts.max())
            if not counts.empty
            else 0
        )

        rows.append(
            {
                "dataset": dataset,
                "n_images": n_rows,
                "n_images_with_subject_id": n_present,
                "n_images_without_subject_id": n_missing,
                "pct_images_with_subject_id": (
                    n_present
                    / n_rows
                    * 100.0
                    if n_rows
                    else 0.0
                ),
                "n_unique_identities": n_identities,
                "n_singleton_identities": n_singletons,
                "n_repeated_identities": n_repeated,
                "max_images_per_identity": max_images,
                "subject_id_available": (
                    n_present > 0
                ),
                "repeated_identity_structure_present": (
                    n_repeated > 0
                ),
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
    Construct an in-memory identity-level demographic assessment.

    For each identity and demographic field:

      unavailable:
          no usable demographic value occurs on any image

      consistent:
          exactly one non-missing demographic value occurs

      conflicting:
          more than one non-missing demographic value occurs

    This table is intended as an intermediate analysis object.
    It does not modify the image-level master table.
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

        work["_subject_rep002"] = (
            normalise_reporting_category(
                work["subject_id"]
            )
        )

        work = work[
            work["_subject_rep002"]
            .str.lower()
            .ne("unknown")
        ]

        if work.empty:
            continue

        for subject_id, subject_group in work.groupby(
            "_subject_rep002",
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
                        "subject_id": subject_id,
                        "field": field,
                        "n_images_for_identity": n_images,
                        "status": status,
                        "category": category,
                        "n_unique_observed_values": len(
                            observed
                        ),
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

    if not rows:
        return pd.DataFrame(
            columns=columns
        )

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
    Summarise whether demographic identity-level composition
    can actually be estimated.

    Datasets without subject IDs remain explicitly represented.
    """
    if fields is None:
        fields = REP002_DEMOGRAPHIC_FIELDS

    support = identity_support_summary(
        df
    )

    rows: list[dict] = []

    for _, support_row in support.iterrows():

        dataset = support_row["dataset"]

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
                (identity_table["dataset"] == dataset)
                & (identity_table["field"] == field)
            ]

            n_consistent = int(
                (subset["status"] == "consistent")
                .sum()
            )

            n_unavailable = int(
                (subset["status"] == "unavailable")
                .sum()
            )

            n_conflicting = int(
                (subset["status"] == "conflicting")
                .sum()
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
                    "n_total_identities": n_identities,
                    "n_consistent": n_consistent,
                    "n_unavailable": n_unavailable,
                    "n_conflicting": n_conflicting,
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

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "dataset",
            "field",
        ]
    ).reset_index(
        drop=True
    )


def identity_demographic_composition(
    identity_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate identity-level demographic composition only
    from identities with one consistent observed label.

    Both labelled-identity and all-identity denominators
    are retained.

    Current LFW results may legitimately be empty if the
    Phase 0 hand-off contains no demographic annotations.
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
        identity_table["status"]
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
        .rename(
            "n_identities"
        )
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