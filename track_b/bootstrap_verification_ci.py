from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "arcface_reference"
    / "verification"
    / "fairness"
    / "view1_test"
    / "pair_scores_with_demographics.csv"
)

DEFAULT_ITERATIONS = 2000
DEFAULT_SEED = 2026
CONFIDENCE_LEVEL = 0.95


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap confidence intervals for overall "
            "and demographic subgroup verification metrics."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Pair-level verification file with demographics."
        ),
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="R0_ArcFace",
        help="Model identifier.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of bootstrap iterations.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Bootstrap random seed.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom output directory.",
    )

    return parser.parse_args()


# ============================================================
# BOOLEAN HANDLING
# ============================================================

def to_bool_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype(bool)

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    cleaned = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if cleaned.isna().any():
        bad = (
            series[
                cleaned.isna()
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Could not parse boolean values: "
            + ", ".join(
                bad[:10]
            )
        )

    return cleaned.astype(bool)


# ============================================================
# METRICS
# ============================================================

def safe_divide(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return np.nan

    return float(
        numerator / denominator
    )


def compute_metrics(
    frame: pd.DataFrame,
) -> dict:
    actual = frame[
        "is_genuine"
    ].to_numpy(
        dtype=bool
    )

    predicted = frame[
        "predicted_genuine"
    ].to_numpy(
        dtype=bool
    )

    tp = int(
        np.sum(
            actual
            & predicted
        )
    )

    fn = int(
        np.sum(
            actual
            & ~predicted
        )
    )

    fp = int(
        np.sum(
            ~actual
            & predicted
        )
    )

    tn = int(
        np.sum(
            ~actual
            & ~predicted
        )
    )

    n_genuine = (
        tp + fn
    )

    n_impostor = (
        tn + fp
    )

    total = (
        n_genuine
        + n_impostor
    )

    tpr = safe_divide(
        tp,
        n_genuine,
    )

    fnmr = safe_divide(
        fn,
        n_genuine,
    )

    fmr = safe_divide(
        fp,
        n_impostor,
    )

    specificity = safe_divide(
        tn,
        n_impostor,
    )

    accuracy = safe_divide(
        tp + tn,
        total,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    if (
        not np.isnan(
            precision
        )
        and not np.isnan(
            tpr
        )
        and (
            precision + tpr
        ) > 0
    ):
        f1 = (
            2.0
            * precision
            * tpr
            / (
                precision
                + tpr
            )
        )
    else:
        f1 = np.nan

    if (
        not np.isnan(
            tpr
        )
        and not np.isnan(
            specificity
        )
    ):
        balanced_accuracy = (
            tpr
            + specificity
        ) / 2.0
    else:
        balanced_accuracy = np.nan

    return {
        "n_pairs": int(
            total
        ),
        "n_genuine": int(
            n_genuine
        ),
        "n_impostor": int(
            n_impostor
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "precision": precision,
        "F1": f1,
        "TPR_TAR": tpr,
        "FMR_FPR": fmr,
        "FNMR_FNR": fnmr,
        "specificity_TNR": (
            specificity
        ),
    }


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "F1",
    "TPR_TAR",
    "FMR_FPR",
    "FNMR_FNR",
    "specificity_TNR",
]


# ============================================================
# STRATIFIED BOOTSTRAP
# ============================================================

def stratified_bootstrap_sample(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    genuine = frame[
        frame[
            "is_genuine"
        ]
    ]

    impostor = frame[
        ~frame[
            "is_genuine"
        ]
    ]

    samples = []

    if not genuine.empty:
        indices = rng.integers(
            0,
            len(genuine),
            size=len(genuine),
        )

        samples.append(
            genuine.iloc[
                indices
            ]
        )

    if not impostor.empty:
        indices = rng.integers(
            0,
            len(impostor),
            size=len(impostor),
        )

        samples.append(
            impostor.iloc[
                indices
            ]
        )

    if not samples:
        return frame.iloc[
            0:0
        ].copy()

    return pd.concat(
        samples,
        ignore_index=True,
    )


def percentile_interval(
    values: np.ndarray,
    confidence: float,
) -> tuple[
    float,
    float,
    int,
]:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan,
            0,
        )

    alpha = (
        1.0 - confidence
    )

    lower = float(
        np.quantile(
            values,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            values,
            1.0 - alpha / 2.0,
        )
    )

    return (
        lower,
        upper,
        len(values),
    )


def bootstrap_frame(
    frame: pd.DataFrame,
    iterations: int,
    seed: int,
    confidence: float,
) -> pd.DataFrame:
    point = compute_metrics(
        frame
    )

    rng = np.random.default_rng(
        seed
    )

    samples = {
        metric: []
        for metric in METRIC_COLUMNS
    }

    for _ in range(
        iterations
    ):
        boot = (
            stratified_bootstrap_sample(
                frame,
                rng,
            )
        )

        result = compute_metrics(
            boot
        )

        for metric in METRIC_COLUMNS:
            samples[
                metric
            ].append(
                result[
                    metric
                ]
            )

    rows = []

    for metric in METRIC_COLUMNS:
        lower, upper, n_valid = (
            percentile_interval(
                np.asarray(
                    samples[
                        metric
                    ],
                    dtype=float,
                ),
                confidence,
            )
        )

        rows.append(
            {
                "metric": metric,
                "estimate": point[
                    metric
                ],
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": (
                    confidence
                ),
                "bootstrap_iterations": (
                    iterations
                ),
                "valid_bootstrap_iterations": (
                    n_valid
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# OVERALL ANALYSIS
# ============================================================

def overall_bootstrap(
    pairs: pd.DataFrame,
    model_id: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    result = bootstrap_frame(
        frame=pairs,
        iterations=iterations,
        seed=seed,
        confidence=CONFIDENCE_LEVEL,
    )

    counts = compute_metrics(
        pairs
    )

    result.insert(
        0,
        "model_id",
        model_id,
    )

    result.insert(
        1,
        "analysis",
        "overall",
    )

    result.insert(
        2,
        "attribute",
        "overall",
    )

    result.insert(
        3,
        "group",
        "all_test_pairs",
    )

    result[
        "n_pairs"
    ] = counts[
        "n_pairs"
    ]

    result[
        "n_genuine"
    ] = counts[
        "n_genuine"
    ]

    result[
        "n_impostor"
    ] = counts[
        "n_impostor"
    ]

    return result


# ============================================================
# SUBGROUP ANALYSIS
# ============================================================

def subgroup_bootstrap(
    pairs: pd.DataFrame,
    model_id: str,
    attribute: str,
    iterations: int,
    base_seed: int,
) -> pd.DataFrame:
    col1 = (
        f"{attribute}1"
    )

    col2 = (
        f"{attribute}2"
    )

    eligible = pairs[
        pairs[col1].notna()
        & pairs[col2].notna()
        & pairs[col1].eq(
            pairs[col2]
        )
    ].copy()

    groups = sorted(
        eligible[
            col1
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    frames = []

    for index, group_name in enumerate(
        groups
    ):
        group = eligible[
            eligible[
                col1
            ]
            .astype(str)
            .eq(
                group_name
            )
        ].copy()

        if group.empty:
            continue

        result = bootstrap_frame(
            frame=group,
            iterations=iterations,
            seed=(
                base_seed
                + 1000
                + index
            ),
            confidence=CONFIDENCE_LEVEL,
        )

        counts = compute_metrics(
            group
        )

        result.insert(
            0,
            "model_id",
            model_id,
        )

        result.insert(
            1,
            "analysis",
            "subgroup",
        )

        result.insert(
            2,
            "attribute",
            attribute,
        )

        result.insert(
            3,
            "group",
            group_name,
        )

        result[
            "n_pairs"
        ] = counts[
            "n_pairs"
        ]

        result[
            "n_genuine"
        ] = counts[
            "n_genuine"
        ]

        result[
            "n_impostor"
        ] = counts[
            "n_impostor"
        ]

        frames.append(
            result
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# DATA QUALITY FLAGS
# ============================================================

def add_precision_flags(
    results: pd.DataFrame,
) -> pd.DataFrame:
    out = results.copy()

    out[
        "small_genuine_sample"
    ] = (
        out[
            "n_genuine"
        ]
        < 20
    )

    out[
        "small_impostor_sample"
    ] = (
        out[
            "n_impostor"
        ]
        < 20
    )

    out[
        "interpret_with_caution"
    ] = (
        out[
            "small_genuine_sample"
        ]
        | out[
            "small_impostor_sample"
        ]
    )

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    if args.iterations < 100:
        raise ValueError(
            "--iterations should be at least 100."
        )

    input_path = (
        args.input
        if args.input.is_absolute()
        else PROJECT_ROOT
        / args.input
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input:\n{input_path}"
        )

    if args.output_dir is None:
        output_dir = (
            input_path.parent
            / "bootstrap"
        )
    else:
        output_dir = (
            args.output_dir
            if args.output_dir.is_absolute()
            else PROJECT_ROOT
            / args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = pd.read_csv(
        input_path,
        low_memory=False,
    )

    required = {
        "is_genuine",
        "predicted_genuine",
        "gender1",
        "gender2",
        "race1",
        "race2",
        "intersection1",
        "intersection2",
    }

    missing = required - set(
        pairs.columns
    )

    if missing:
        raise KeyError(
            "Input is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    pairs[
        "is_genuine"
    ] = to_bool_series(
        pairs[
            "is_genuine"
        ]
    )

    pairs[
        "predicted_genuine"
    ] = to_bool_series(
        pairs[
            "predicted_genuine"
        ]
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TRACK B BOOTSTRAP CONFIDENCE INTERVALS"
    )

    print(
        "=" * 72
    )

    print(
        f"Model: {args.model_id}"
    )

    print(
        f"Pairs: {len(pairs):,}"
    )

    print(
        f"Iterations: {args.iterations:,}"
    )

    print(
        f"Confidence level: "
        f"{CONFIDENCE_LEVEL:.0%}"
    )

    print(
        f"Seed: {args.seed}"
    )

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    frames = [
        overall_bootstrap(
            pairs=pairs,
            model_id=args.model_id,
            iterations=args.iterations,
            seed=args.seed,
        )
    ]

    # --------------------------------------------------------
    # DEMOGRAPHIC SUBGROUPS
    # --------------------------------------------------------

    for offset, attribute in enumerate(
        [
            "gender",
            "race",
            "intersection",
        ]
    ):
        result = subgroup_bootstrap(
            pairs=pairs,
            model_id=args.model_id,
            attribute=attribute,
            iterations=args.iterations,
            base_seed=(
                args.seed
                + (
                    offset
                    * 10000
                )
            ),
        )

        if not result.empty:
            frames.append(
                result
            )

    results = pd.concat(
        frames,
        ignore_index=True,
    )

    results = add_precision_flags(
        results
    )

    # --------------------------------------------------------
    # SAVE FULL TABLE
    # --------------------------------------------------------

    output_path = (
        output_dir
        / "bootstrap_confidence_intervals.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # REPORT-FRIENDLY TABLE
    # --------------------------------------------------------

    report_metrics = {
        "accuracy",
        "balanced_accuracy",
        "TPR_TAR",
        "FMR_FPR",
        "FNMR_FNR",
    }

    report = results[
        results[
            "metric"
        ].isin(
            report_metrics
        )
    ].copy()

    report[
        "estimate_ci"
    ] = report.apply(
        lambda row: (
            f"{row['estimate']:.4f} "
            f"[{row['ci_lower']:.4f}, "
            f"{row['ci_upper']:.4f}]"
            if (
                pd.notna(
                    row[
                        "estimate"
                    ]
                )
                and pd.notna(
                    row[
                        "ci_lower"
                    ]
                )
                and pd.notna(
                    row[
                        "ci_upper"
                    ]
                )
            )
            else "NA"
        ),
        axis=1,
    )

    report_path = (
        output_dir
        / "bootstrap_report_table.csv"
    )

    report.to_csv(
        report_path,
        index=False,
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {
        "model_id": args.model_id,
        "input": str(
            input_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "evaluation_split": (
            "LFW View 1 test"
        ),
        "confidence_level": (
            CONFIDENCE_LEVEL
        ),
        "bootstrap_iterations": (
            args.iterations
        ),
        "random_seed": (
            args.seed
        ),
        "bootstrap_method": (
            "non-parametric stratified pair bootstrap"
        ),
        "stratification": (
            "genuine and impostor pairs resampled "
            "separately within each analysis group"
        ),
        "subgroup_rule": (
            "both images must share the same known "
            "demographic group label"
        ),
        "small_sample_flag_threshold": {
            "genuine_pairs": 20,
            "impostor_pairs": 20
        },
        "view2_used": False,
        "note": (
            "Current R0 ArcFace raw-input run is "
            "diagnostic only until aligned inputs "
            "replace it."
        ),
    }

    metadata_path = (
        output_dir
        / "bootstrap_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    overall = results[
        (
            results[
                "analysis"
            ]
            == "overall"
        )
        & results[
            "metric"
        ].isin(
            [
                "accuracy",
                "balanced_accuracy",
                "TPR_TAR",
                "FMR_FPR",
                "FNMR_FNR",
            ]
        )
    ]

    print(
        "\nOverall View 1 test estimates:"
    )

    for row in overall.itertuples():
        print(
            f"  {row.metric:<20} "
            f"{row.estimate:.4f} "
            f"[{row.ci_lower:.4f}, "
            f"{row.ci_upper:.4f}]"
        )

    caution_groups = (
        results[
            (
                results[
                    "analysis"
                ]
                == "subgroup"
            )
            & results[
                "interpret_with_caution"
            ]
        ][
            [
                "attribute",
                "group",
                "n_genuine",
                "n_impostor",
            ]
        ]
        .drop_duplicates()
    )

    print(
        "\nSubgroups evaluated:",
        results[
            results[
                "analysis"
            ]
            == "subgroup"
        ][
            [
                "attribute",
                "group",
            ]
        ]
        .drop_duplicates()
        .shape[0],
    )

    print(
        "Small-sample subgroup warnings:",
        len(
            caution_groups
        ),
    )

    if not caution_groups.empty:
        print(
            "\nGroups requiring caution:"
        )

        print(
            caution_groups.to_string(
                index=False
            )
        )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )

    print(
        report_path
    )

    print(
        metadata_path
    )

    print(
        "\nNOTE:"
    )

    print(
        "These confidence intervals currently test "
        "the statistical pipeline only. The present "
        "raw-input ArcFace scores are not final R0 results."
    )


if __name__ == "__main__":
    main()
    