from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.representation import (
    harmonise_age_band,
    normalise_reporting_category,
)


# =====================================================================
# REP-006 — MISSING / UNKNOWN LABEL AUDIT
# =====================================================================

REP006_CORE_LABELS = {
    "gender": "gender",
    "race": "race",
    "age_label": "age_band_harmonised",
    "exact_numeric_age": "age",
}


REP006_ANALYSIS_REQUIREMENTS = {
    "REP-001_gender_composition": [
        "gender",
    ],
    "REP-001_race_composition": [
        "race",
    ],
    "REP-001_age_composition": [
        "age_band_harmonised",
    ],
    "REP-004_gender_x_race": [
        "gender",
        "race",
    ],
    "REP-004_gender_x_age": [
        "gender",
        "age_band_harmonised",
    ],
    "REP-004_race_x_age": [
        "race",
        "age_band_harmonised",
    ],
    "REP-004_gender_x_race_x_age": [
        "gender",
        "race",
        "age_band_harmonised",
    ],
    "REP-005_overall_base_rate": [
        "gender",
    ],
    "REP-005_base_rate_by_race": [
        "gender",
        "race",
    ],
    "REP-005_base_rate_by_age": [
        "gender",
        "age_band_harmonised",
    ],
}


def _missing_label_mask(
    series: pd.Series,
) -> pd.Series:
    """
    Return True where a demographic field is missing
    or represented by an unknown-like value.
    """
    normalised = normalise_reporting_category(
        series
    )

    return (
        normalised
        .str.lower()
        .eq("unknown")
    )


def _strict_boolean_mask(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a boolean-like flag into a strict boolean mask.

    Null values are treated as False.
    Unexpected encodings raise an error.
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
        | text.isin(true_values)
        | text.isin(false_values)
    )

    if not known.all():
        unexpected = sorted(
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
                map(str, unexpected)
            )
        )

    return text.isin(
        true_values
    )


def prepare_rep006_data(
    df: pd.DataFrame,
    category_config_path: str | Path = "configs/categories.yaml",
) -> pd.DataFrame:
    """
    Prepare a reporting-only dataframe.

    No rows or source fields in the Phase 0 master table
    are modified.
    """
    required = {
        "dataset",
        "gender",
        "race",
        "age",
        "age_band",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise KeyError(
            "REP-006 master table is missing: "
            + ", ".join(missing)
        )

    out = df.copy()

    out[
        "age_band_harmonised"
    ] = harmonise_age_band(
        out,
        config_path=category_config_path,
    )

    return out


def label_missingness_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Audit demographic-label availability over every master-table row.

    exact_numeric_age is reported separately because an image can
    have a valid categorical age label without an exact numeric age.
    """
    rows: list[dict] = []

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        for label_name, column in (
            REP006_CORE_LABELS.items()
        ):
            if column not in group.columns:
                continue

            missing = _missing_label_mask(
                group[column]
            )

            n_total = len(
                group
            )

            n_missing = int(
                missing.sum()
            )

            n_available = (
                n_total
                - n_missing
            )

            if n_missing == 0:
                status = "complete"

            elif n_available == 0:
                status = "unavailable"

            else:
                status = "partial_missingness"

            rows.append(
                {
                    "dataset":
                        dataset,
                    "label":
                        label_name,
                    "source_column":
                        column,
                    "n_total":
                        n_total,
                    "n_available":
                        n_available,
                    "n_missing_or_unknown":
                        n_missing,
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
                    "missingness_status":
                        status,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "dataset",
                "label",
            ]
        )
        .reset_index(drop=True)
    )


def demographic_missingness_patterns(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count joint missingness patterns for the three core
    demographic labels:

        gender
        race
        harmonised age

    This reveals whether missing fields occur on the same rows.
    """
    required = {
        "dataset",
        "gender",
        "race",
        "age_band_harmonised",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise KeyError(
            "Cannot build REP-006 missingness patterns. "
            "Missing columns: "
            + ", ".join(missing)
        )

    work = pd.DataFrame(
        {
            "dataset":
                df["dataset"],
            "gender_missing":
                _missing_label_mask(
                    df["gender"]
                ),
            "race_missing":
                _missing_label_mask(
                    df["race"]
                ),
            "age_label_missing":
                _missing_label_mask(
                    df[
                        "age_band_harmonised"
                    ]
                ),
        }
    )

    work[
        "any_core_label_missing"
    ] = (
        work[
            [
                "gender_missing",
                "race_missing",
                "age_label_missing",
            ]
        ]
        .any(
            axis=1
        )
    )

    work[
        "all_core_labels_missing"
    ] = (
        work[
            [
                "gender_missing",
                "race_missing",
                "age_label_missing",
            ]
        ]
        .all(
            axis=1
        )
    )

    counts = (
        work.groupby(
            [
                "dataset",
                "gender_missing",
                "race_missing",
                "age_label_missing",
                "any_core_label_missing",
                "all_core_labels_missing",
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

    counts[
        "percentage"
    ] = (
        counts["n_images"]
        / counts["dataset_total"]
        * 100.0
    )

    return (
        counts
        .sort_values(
            [
                "dataset",
                "any_core_label_missing",
                "gender_missing",
                "race_missing",
                "age_label_missing",
            ]
        )
        .reset_index(drop=True)
    )


def affected_analysis_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quantify how demographic-label missingness affects each
    representation analysis.

    n_complete_label_rows considers label availability only.

    n_eligible_complete_rows additionally applies the existing
    Phase 0 eligible_demographic_audit mask when that field exists.
    """
    rows: list[dict] = []

    has_eligibility = (
        "eligible_demographic_audit"
        in df.columns
    )

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        if has_eligibility:
            eligible = _strict_boolean_mask(
                group[
                    "eligible_demographic_audit"
                ]
            )
        else:
            eligible = pd.Series(
                True,
                index=group.index,
                dtype=bool,
            )

        for (
            analysis,
            required_fields,
        ) in (
            REP006_ANALYSIS_REQUIREMENTS.items()
        ):
            missing_fields = [
                field
                for field in required_fields
                if field not in group.columns
            ]

            if missing_fields:
                raise KeyError(
                    f"{analysis} requires missing columns: "
                    + ", ".join(missing_fields)
                )

            complete = pd.Series(
                True,
                index=group.index,
                dtype=bool,
            )

            for field in required_fields:
                complete &= (
                    ~_missing_label_mask(
                        group[field]
                    )
                )

            n_total = len(
                group
            )

            n_complete = int(
                complete.sum()
            )

            n_incomplete = (
                n_total
                - n_complete
            )

            n_eligible_complete = int(
                (
                    complete
                    & eligible
                ).sum()
            )

            if n_complete == 0:
                label_status = (
                    "blocked_by_missing_labels"
                )

            elif n_incomplete == 0:
                label_status = (
                    "label_complete"
                )

            else:
                label_status = (
                    "partially_affected"
                )

            if n_eligible_complete == 0:
                operational_status = (
                    "no_eligible_complete_rows"
                )

            elif n_eligible_complete < n_total:
                operational_status = (
                    "available_with_reduced_denominator"
                )

            else:
                operational_status = (
                    "fully_available"
                )

            rows.append(
                {
                    "dataset":
                        dataset,
                    "analysis":
                        analysis,
                    "required_fields":
                        ";".join(
                            required_fields
                        ),
                    "n_total_images":
                        n_total,
                    "n_complete_label_rows":
                        n_complete,
                    "n_incomplete_label_rows":
                        n_incomplete,
                    "pct_complete_label_rows": (
                        n_complete
                        / n_total
                        * 100.0
                        if n_total
                        else 0.0
                    ),
                    "n_eligible_complete_rows":
                        n_eligible_complete,
                    "label_missingness_status":
                        label_status,
                    "operational_status":
                        operational_status,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "dataset",
                "analysis",
            ]
        )
        .reset_index(drop=True)
    )


def missingness_parse_failure_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare core demographic missingness with Phase 0 parse failures.

    This determines whether missing labels coincide with rows already
    flagged by metadata parsing, without assuming that they do.
    """
    required = {
        "dataset",
        "gender",
        "race",
        "age_band_harmonised",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise KeyError(
            "Cannot compare missingness with parse failure. "
            "Missing columns: "
            + ", ".join(missing)
        )

    rows: list[dict] = []

    has_parse_failed = (
        "parse_failed"
        in df.columns
    )

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        missing_any = (
            _missing_label_mask(
                group["gender"]
            )
            |
            _missing_label_mask(
                group["race"]
            )
            |
            _missing_label_mask(
                group[
                    "age_band_harmonised"
                ]
            )
        )

        if has_parse_failed:
            parse_failed = (
                _strict_boolean_mask(
                    group[
                        "parse_failed"
                    ]
                )
            )
        else:
            parse_failed = pd.Series(
                False,
                index=group.index,
                dtype=bool,
            )

        n_total = len(
            group
        )

        n_missing_any = int(
            missing_any.sum()
        )

        n_parse_failed = int(
            parse_failed.sum()
        )

        n_both = int(
            (
                missing_any
                & parse_failed
            ).sum()
        )

        n_missing_without_parse_failure = int(
            (
                missing_any
                & ~parse_failed
            ).sum()
        )

        n_parse_failure_without_missing = int(
            (
                parse_failed
                & ~missing_any
            ).sum()
        )

        rows.append(
            {
                "dataset":
                    dataset,
                "n_total_images":
                    n_total,
                "n_any_core_label_missing":
                    n_missing_any,
                "n_parse_failed":
                    n_parse_failed,
                "n_missing_and_parse_failed":
                    n_both,
                "n_missing_without_parse_failure":
                    n_missing_without_parse_failure,
                "n_parse_failure_without_missing":
                    n_parse_failure_without_missing,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("dataset")
        .reset_index(drop=True)
    )
