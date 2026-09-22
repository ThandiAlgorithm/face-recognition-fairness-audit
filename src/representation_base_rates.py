from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.representation import (
    harmonise_age_band,
    load_image_cell_thresholds,
    normalise_reporting_category,
)


# =====================================================================
# REP-005 — BASE-RATE TABLES
# =====================================================================

REP005_TARGET_FIELD = "gender"

REP005_SUBGROUPS = {
    "race": "race",
    "age": "age_band_harmonised",
}


def load_target_coding(
    config_path: str | Path = "configs/categories.yaml",
    target_field: str = REP005_TARGET_FIELD,
) -> dict:
    """
    Load and validate the frozen binary target coding.

    REP-005 defines the base rate as:

        P(Y = 1 | A = g)

    The positive class is read from frozen governance rather
    than hard-coded into the analysis.
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

    target_config = config.get(
        target_field
    )

    if not isinstance(
        target_config,
        dict,
    ):
        raise ValueError(
            f"No configuration found for target "
            f"field '{target_field}'."
        )

    positive_class = target_config.get(
        "positive_class"
    )

    codes = target_config.get(
        "codes"
    )

    if positive_class is None:
        raise ValueError(
            f"No positive_class defined for "
            f"'{target_field}'."
        )

    if not isinstance(
        codes,
        dict,
    ):
        raise ValueError(
            f"No codes mapping defined for "
            f"'{target_field}'."
        )

    canonical_codes = {
        str(label).strip().lower(): code
        for label, code in codes.items()
    }

    positive_label = (
        str(positive_class)
        .strip()
        .lower()
    )

    if positive_label not in canonical_codes:
        raise ValueError(
            "Configured positive class is not "
            "present in the target codes mapping."
        )

    positive_code = canonical_codes[
        positive_label
    ]

    if positive_code != 1:
        raise ValueError(
            f"Positive class '{positive_label}' "
            f"must be coded as 1 for REP-005; "
            f"found {positive_code!r}."
        )

    binary_classes = [
        label
        for label, code in canonical_codes.items()
        if code in {
            0,
            1,
        }
    ]

    negative_classes = [
        label
        for label, code in canonical_codes.items()
        if code == 0
    ]

    if len(binary_classes) < 2:
        raise ValueError(
            "REP-005 requires a binary target "
            "with both 0 and 1 classes."
        )

    if not negative_classes:
        raise ValueError(
            "No class coded as 0 was found."
        )

    return {
        "target_field":
            target_field,
        "positive_class":
            positive_label,
        "positive_code":
            1,
        "negative_classes":
            negative_classes,
        "binary_classes":
            binary_classes,
        "config_version":
            meta.get("version"),
        "frozen_commit":
            meta.get("frozen_commit"),
    }


def _strict_boolean_mask(
    series: pd.Series,
) -> pd.Series:
    """
    Safely convert a boolean-like eligibility column.

    Null is treated as False. Unexpected encodings raise
    an error instead of being silently interpreted.
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
        bad = sorted(
            text[
                ~known
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unexpected eligibility encoding: "
            + ", ".join(
                map(str, bad)
            )
        )

    return text.isin(
        true_values
    )


def prepare_rep005_data(
    df: pd.DataFrame,
    category_config_path: str | Path = "configs/categories.yaml",
) -> pd.DataFrame:
    """
    Prepare a reporting-only dataframe for REP-005.

    The source master table is not modified.
    """
    required = {
        "dataset",
        "gender",
        "race",
        "age",
        "age_band",
        "eligible_demographic_audit",
    }

    missing = sorted(
        required - set(df.columns)
    )

    if missing:
        raise KeyError(
            "REP-005 master table is missing: "
            + ", ".join(missing)
        )

    out = df.copy()

    out[
        "age_band_harmonised"
    ] = harmonise_age_band(
        out,
        config_path=category_config_path,
    )

    out[
        "_rep005_eligible"
    ] = _strict_boolean_mask(
        out[
            "eligible_demographic_audit"
        ]
    )

    out[
        "_rep005_target"
    ] = (
        normalise_reporting_category(
            out["gender"]
        )
        .str.lower()
    )

    out[
        "_rep005_race"
    ] = normalise_reporting_category(
        out["race"]
    )

    out[
        "_rep005_age"
    ] = normalise_reporting_category(
        out[
            "age_band_harmonised"
        ]
    )

    return out


def _reporting_status(
    n: int,
    suppress_below: int,
    low_confidence_below: int,
) -> tuple[str, bool, bool]:
    """
    Apply frozen image-level minimum-n reporting rules.
    """
    if n <= 0:
        return (
            "not_available",
            False,
            False,
        )

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


def target_coding_table(
    coding: dict,
) -> pd.DataFrame:
    """
    Machine-readable record of REP-005 target coding.
    """
    return pd.DataFrame(
        [
            {
                "target_field":
                    coding[
                        "target_field"
                    ],
                "positive_class":
                    coding[
                        "positive_class"
                    ],
                "positive_code":
                    coding[
                        "positive_code"
                    ],
                "negative_classes":
                    ";".join(
                        coding[
                            "negative_classes"
                        ]
                    ),
                "binary_classes":
                    ";".join(
                        coding[
                            "binary_classes"
                        ]
                    ),
                "config_version":
                    coding[
                        "config_version"
                    ],
                "frozen_commit":
                    coding[
                        "frozen_commit"
                    ],
            }
        ]
    )


def rep005_analysis_coverage(
    df: pd.DataFrame,
    coding: dict,
) -> pd.DataFrame:
    """
    Reconcile the denominator available for each subgroup
    base-rate analysis.

    Unknown subgroup labels are not used as protected-group
    base-rate cells, but their counts remain visible here.
    """
    binary_classes = set(
        coding[
            "binary_classes"
        ]
    )

    rows: list[dict] = []

    subgroup_specs = [
        (
            "race",
            "_rep005_race",
        ),
        (
            "age",
            "_rep005_age",
        ),
    ]

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        eligible = group[
            "_rep005_eligible"
        ]

        target_known = (
            group[
                "_rep005_target"
            ]
            .isin(
                binary_classes
            )
        )

        for (
            subgroup_axis,
            subgroup_column,
        ) in subgroup_specs:

            subgroup_known = (
                group[
                    subgroup_column
                ]
                .str.lower()
                .ne("unknown")
            )

            complete = (
                eligible
                & target_known
                & subgroup_known
            )

            n_total = len(
                group
            )

            n_eligible = int(
                eligible.sum()
            )

            n_target_known = int(
                (
                    eligible
                    & target_known
                ).sum()
            )

            n_subgroup_known = int(
                (
                    eligible
                    & subgroup_known
                ).sum()
            )

            n_complete = int(
                complete.sum()
            )

            if n_eligible == 0:
                status = (
                    "no_eligible_demographic_rows"
                )

            elif n_target_known == 0:
                status = (
                    "target_unavailable"
                )

            elif n_subgroup_known == 0:
                status = (
                    "subgroup_unavailable"
                )

            elif n_complete == 0:
                status = (
                    "no_complete_cases"
                )

            else:
                status = (
                    "available"
                )

            rows.append(
                {
                    "dataset":
                        dataset,
                    "subgroup_axis":
                        subgroup_axis,
                    "n_total_images":
                        n_total,
                    "n_eligible_demographic":
                        n_eligible,
                    "n_binary_target_known":
                        n_target_known,
                    "n_subgroup_known":
                        n_subgroup_known,
                    "n_complete_cases":
                        n_complete,
                    "n_target_unknown_among_eligible":
                        (
                            n_eligible
                            - n_target_known
                        ),
                    "n_subgroup_unknown_among_eligible":
                        (
                            n_eligible
                            - n_subgroup_known
                        ),
                    "analysis_status":
                        status,
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "dataset",
                "subgroup_axis",
            ]
        )
        .reset_index(drop=True)
    )


def overall_base_rate(
    df: pd.DataFrame,
    coding: dict,
    reporting_config_path: str | Path = "configs/reporting.yaml",
) -> pd.DataFrame:
    """
    Calculate the overall dataset base rate P(Y=1).

    This is useful context for subgroup-conditioned rates.
    """
    thresholds = (
        load_image_cell_thresholds(
            reporting_config_path
        )
    )

    binary_classes = set(
        coding[
            "binary_classes"
        ]
    )

    positive_class = coding[
        "positive_class"
    ]

    rows: list[dict] = []

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        eligible = group[
            "_rep005_eligible"
        ]

        target_known = (
            group[
                "_rep005_target"
            ]
            .isin(
                binary_classes
            )
        )

        analysis_mask = (
            eligible
            & target_known
        )

        values = group.loc[
            analysis_mask,
            "_rep005_target",
        ]

        n = len(
            values
        )

        n_positive = int(
            (
                values
                == positive_class
            ).sum()
        )

        n_negative = (
            n
            - n_positive
        )

        base_rate_raw = (
            n_positive
            / n
            if n
            else None
        )

        (
            reporting_status,
            thin_cell,
            low_confidence,
        ) = _reporting_status(
            n,
            thresholds[
                "suppress_below"
            ],
            thresholds[
                "low_confidence_below"
            ],
        )

        base_rate_reported = (
            None
            if reporting_status
            in {
                "suppressed",
                "not_available",
            }
            else base_rate_raw
        )

        if int(eligible.sum()) == 0:
            analysis_status = (
                "no_eligible_demographic_rows"
            )

        elif n == 0:
            analysis_status = (
                "target_unavailable"
            )

        else:
            analysis_status = (
                "available"
            )

        rows.append(
            {
                "dataset":
                    dataset,
                "target_field":
                    coding[
                        "target_field"
                    ],
                "positive_class":
                    positive_class,
                "n":
                    n,
                "n_y1":
                    n_positive,
                "n_y0":
                    n_negative,
                "base_rate_raw":
                    base_rate_raw,
                "base_rate_pct_raw": (
                    base_rate_raw
                    * 100.0
                    if base_rate_raw
                    is not None
                    else None
                ),
                "base_rate_reported":
                    base_rate_reported,
                "base_rate_pct_reported": (
                    base_rate_reported
                    * 100.0
                    if base_rate_reported
                    is not None
                    else None
                ),
                "thin_cell":
                    thin_cell,
                "low_confidence":
                    low_confidence,
                "reporting_status":
                    reporting_status,
                "analysis_status":
                    analysis_status,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("dataset")
        .reset_index(drop=True)
    )


def subgroup_base_rate(
    df: pd.DataFrame,
    coding: dict,
    subgroup_axis: str,
    reporting_config_path: str | Path = "configs/reporting.yaml",
) -> pd.DataFrame:
    """
    Calculate:

        P(Y = 1 | A = g)

    for one subgroup axis.

    Current supported axes:
        race
        age

    Race remains dataset-specific. No cross-dataset race
    categories are merged.
    """
    subgroup_columns = {
        "race":
            "_rep005_race",
        "age":
            "_rep005_age",
    }

    if subgroup_axis not in subgroup_columns:
        raise ValueError(
            f"Unsupported subgroup axis: "
            f"{subgroup_axis}"
        )

    subgroup_column = subgroup_columns[
        subgroup_axis
    ]

    thresholds = (
        load_image_cell_thresholds(
            reporting_config_path
        )
    )

    binary_classes = set(
        coding[
            "binary_classes"
        ]
    )

    positive_class = coding[
        "positive_class"
    ]

    rows: list[dict] = []

    for dataset, group in df.groupby(
        "dataset",
        dropna=False,
    ):
        eligible = group[
            "_rep005_eligible"
        ]

        target_known = (
            group[
                "_rep005_target"
            ]
            .isin(
                binary_classes
            )
        )

        subgroup_known = (
            group[
                subgroup_column
            ]
            .str.lower()
            .ne("unknown")
        )

        work = group[
            eligible
            & target_known
            & subgroup_known
        ].copy()

        if work.empty:
            continue

        for subgroup_value, cell in work.groupby(
            subgroup_column,
            dropna=False,
        ):
            n = len(
                cell
            )

            n_positive = int(
                (
                    cell[
                        "_rep005_target"
                    ]
                    == positive_class
                ).sum()
            )

            n_negative = (
                n
                - n_positive
            )

            base_rate_raw = (
                n_positive
                / n
            )

            (
                reporting_status,
                thin_cell,
                low_confidence,
            ) = _reporting_status(
                n,
                thresholds[
                    "suppress_below"
                ],
                thresholds[
                    "low_confidence_below"
                ],
            )

            base_rate_reported = (
                None
                if reporting_status
                == "suppressed"
                else base_rate_raw
            )

            rows.append(
                {
                    "dataset":
                        dataset,
                    "subgroup_axis":
                        subgroup_axis,
                    "subgroup":
                        subgroup_value,
                    "target_field":
                        coding[
                            "target_field"
                        ],
                    "positive_class":
                        positive_class,
                    "n":
                        n,
                    "n_y1":
                        n_positive,
                    "n_y0":
                        n_negative,
                    "base_rate_raw":
                        base_rate_raw,
                    "base_rate_pct_raw":
                        (
                            base_rate_raw
                            * 100.0
                        ),
                    "base_rate_reported":
                        base_rate_reported,
                    "base_rate_pct_reported": (
                        base_rate_reported
                        * 100.0
                        if base_rate_reported
                        is not None
                        else None
                    ),
                    "thin_cell":
                        thin_cell,
                    "low_confidence":
                        low_confidence,
                    "reporting_status":
                        reporting_status,
                }
            )

    columns = [
        "dataset",
        "subgroup_axis",
        "subgroup",
        "target_field",
        "positive_class",
        "n",
        "n_y1",
        "n_y0",
        "base_rate_raw",
        "base_rate_pct_raw",
        "base_rate_reported",
        "base_rate_pct_reported",
        "thin_cell",
        "low_confidence",
        "reporting_status",
    ]

    if not rows:
        return pd.DataFrame(
            columns=columns
        )

    return (
        pd.DataFrame(
            rows,
            columns=columns,
        )
        .sort_values(
            [
                "dataset",
                "subgroup",
            ]
        )
        .reset_index(drop=True)
    )
