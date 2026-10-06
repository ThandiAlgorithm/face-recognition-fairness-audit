from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired McNemar comparison of two facial verification "
            "models on identical LFW View 1 test pairs."
        )
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help=(
            "Baseline pair-score CSV, e.g. M0 verification output."
        ),
    )

    parser.add_argument(
        "--comparison",
        type=Path,
        required=True,
        help=(
            "Comparison pair-score CSV, e.g. M1/M2/M3 output."
        ),
    )

    parser.add_argument(
        "--baseline-id",
        type=str,
        required=True,
        help="Baseline model label, e.g. M0.",
    )

    parser.add_argument(
        "--comparison-id",
        type=str,
        required=True,
        help="Comparison model label, e.g. M1.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom output directory.",
    )

    return parser.parse_args()


# ============================================================
# PATH
# ============================================================

def resolve_path(
    path: Path,
) -> Path:
    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    )


# ============================================================
# BOOLEAN PARSING
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
        invalid = (
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
                invalid[:10]
            )
        )

    return cleaned.astype(bool)


# ============================================================
# LOAD ONE MODEL
# ============================================================

def load_model_scores(
    path: Path,
    model_id: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing score file:\n{path}"
        )

    frame = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "pair_id",
        "split",
        "is_genuine",
        "predicted_genuine",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:
        raise KeyError(
            f"{model_id} score file is missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    frame = frame[
        frame["split"]
        .astype(str)
        .str.lower()
        .eq("test")
    ].copy()

    if len(frame) != 1000:
        raise ValueError(
            f"{model_id}: expected 1,000 View 1 "
            f"test pairs, found {len(frame)}."
        )

    frame[
        "is_genuine"
    ] = to_bool_series(
        frame[
            "is_genuine"
        ]
    )

    frame[
        "predicted_genuine"
    ] = to_bool_series(
        frame[
            "predicted_genuine"
        ]
    )

    if frame[
        "pair_id"
    ].duplicated().any():
        raise RuntimeError(
            f"{model_id}: duplicate pair_id values."
        )

    frame[
        "correct"
    ] = (
        frame[
            "is_genuine"
        ]
        == frame[
            "predicted_genuine"
        ]
    )

    return frame[
        [
            "pair_id",
            "is_genuine",
            "predicted_genuine",
            "correct",
        ]
    ].copy()


# ============================================================
# ALIGN TWO MODELS
# ============================================================

def align_models(
    baseline: pd.DataFrame,
    comparison: pd.DataFrame,
    baseline_id: str,
    comparison_id: str,
) -> pd.DataFrame:
    merged = baseline.merge(
        comparison,
        on="pair_id",
        how="inner",
        suffixes=(
            "_baseline",
            "_comparison",
        ),
        validate="one_to_one",
    )

    if len(merged) != 1000:
        raise RuntimeError(
            "The two models do not contain the same "
            "1,000 View 1 test pairs."
        )

    if not (
        merged[
            "is_genuine_baseline"
        ]
        == merged[
            "is_genuine_comparison"
        ]
    ).all():
        raise RuntimeError(
            "Ground-truth pair labels differ between "
            f"{baseline_id} and {comparison_id}."
        )

    merged[
        "is_genuine"
    ] = merged[
        "is_genuine_baseline"
    ]

    return merged


# ============================================================
# MCNEMAR
# ============================================================

def mcnemar_exact(
    frame: pd.DataFrame,
) -> dict:
    baseline_correct = frame[
        "correct_baseline"
    ].astype(bool)

    comparison_correct = frame[
        "correct_comparison"
    ].astype(bool)

    both_correct = int(
        (
            baseline_correct
            & comparison_correct
        ).sum()
    )

    baseline_only = int(
        (
            baseline_correct
            & ~comparison_correct
        ).sum()
    )

    comparison_only = int(
        (
            ~baseline_correct
            & comparison_correct
        ).sum()
    )

    both_wrong = int(
        (
            ~baseline_correct
            & ~comparison_correct
        ).sum()
    )

    discordant = (
        baseline_only
        + comparison_only
    )

    if discordant == 0:
        p_value = 1.0

    else:
        p_value = float(
            binomtest(
                k=comparison_only,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    baseline_accuracy = float(
        baseline_correct.mean()
    )

    comparison_accuracy = float(
        comparison_correct.mean()
    )

    return {
        "n_pairs": int(
            len(frame)
        ),
        "both_correct": (
            both_correct
        ),
        "baseline_correct_comparison_wrong": (
            baseline_only
        ),
        "baseline_wrong_comparison_correct": (
            comparison_only
        ),
        "both_wrong": (
            both_wrong
        ),
        "discordant_pairs": int(
            discordant
        ),
        "baseline_accuracy": (
            baseline_accuracy
        ),
        "comparison_accuracy": (
            comparison_accuracy
        ),
        "accuracy_difference": float(
            comparison_accuracy
            - baseline_accuracy
        ),
        "mcnemar_exact_p": (
            p_value
        ),
    }


# ============================================================
# ANALYSE SUBSET
# ============================================================

def analyse_subset(
    frame: pd.DataFrame,
    subset_name: str,
    baseline_id: str,
    comparison_id: str,
) -> dict:
    result = mcnemar_exact(
        frame
    )

    result.update(
        {
            "baseline_id": baseline_id,
            "comparison_id": (
                comparison_id
            ),
            "subset": subset_name,
        }
    )

    return result


# ============================================================
# SPLIT BY PAIR TYPE
# ============================================================

def analyse_pair_types(
    frame: pd.DataFrame,
    baseline_id: str,
    comparison_id: str,
) -> pd.DataFrame:
    rows = []

    rows.append(
        analyse_subset(
            frame,
            "all_test_pairs",
            baseline_id,
            comparison_id,
        )
    )

    genuine = frame[
        frame[
            "is_genuine"
        ]
    ].copy()

    impostor = frame[
        ~frame[
            "is_genuine"
        ]
    ].copy()

    rows.append(
        analyse_subset(
            genuine,
            "genuine_pairs",
            baseline_id,
            comparison_id,
        )
    )

    rows.append(
        analyse_subset(
            impostor,
            "impostor_pairs",
            baseline_id,
            comparison_id,
        )
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SIGNIFICANCE LABEL
# ============================================================

def add_interpretation(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    out = frame.copy()

    out[
        "significant_at_0_05"
    ] = (
        out[
            "mcnemar_exact_p"
        ]
        < 0.05
    )

    def direction(
        row,
    ) -> str:
        difference = row[
            "accuracy_difference"
        ]

        significant = row[
            "significant_at_0_05"
        ]

        if not significant:
            return (
                "no_statistically_significant_difference"
            )

        if difference > 0:
            return (
                "comparison_model_better"
            )

        if difference < 0:
            return (
                "baseline_model_better"
            )

        return (
            "no_accuracy_difference"
        )

    out[
        "interpretation"
    ] = out.apply(
        direction,
        axis=1,
    )

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    baseline_path = resolve_path(
        args.baseline
    )

    comparison_path = resolve_path(
        args.comparison
    )

    if args.output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "results"
            / "track_b"
            / "model_comparisons"
            / (
                f"{args.baseline_id}_vs_"
                f"{args.comparison_id}"
            )
        )

    else:
        output_dir = resolve_path(
            args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline = load_model_scores(
        baseline_path,
        args.baseline_id,
    )

    comparison = load_model_scores(
        comparison_path,
        args.comparison_id,
    )

    paired = align_models(
        baseline,
        comparison,
        args.baseline_id,
        args.comparison_id,
    )

    results = analyse_pair_types(
        paired,
        args.baseline_id,
        args.comparison_id,
    )

    results = add_interpretation(
        results
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results_path = (
        output_dir
        / "mcnemar_results.csv"
    )

    paired_path = (
        output_dir
        / "paired_predictions.csv"
    )

    metadata_path = (
        output_dir
        / "mcnemar_metadata.json"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    paired.to_csv(
        paired_path,
        index=False,
    )

    metadata = {
        "baseline_id": (
            args.baseline_id
        ),
        "comparison_id": (
            args.comparison_id
        ),
        "baseline_file": str(
            baseline_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "comparison_file": str(
            comparison_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "evaluation_split": (
            "LFW View 1 test"
        ),
        "n_test_pairs": 1000,
        "statistical_test": (
            "exact two-sided McNemar test"
        ),
        "significance_level": 0.05,
        "paired_unit": (
            "same LFW verification pair"
        ),
        "primary_comparison": (
            "correct versus incorrect verification decision"
        ),
        "view2_used": False,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TRACK B PAIRED MODEL COMPARISON"
    )

    print(
        "=" * 72
    )

    print(
        f"Baseline:   {args.baseline_id}"
    )

    print(
        f"Comparison: {args.comparison_id}"
    )

    print(
        "\nLFW View 1 test pairs aligned:",
        len(paired),
    )

    print(
        "\nMcNemar results:"
    )

    for row in results.itertuples():
        print(
            f"\n  {row.subset}"
        )

        print(
            f"    Baseline accuracy:   "
            f"{row.baseline_accuracy:.4f}"
        )

        print(
            f"    Comparison accuracy: "
            f"{row.comparison_accuracy:.4f}"
        )

        print(
            f"    Difference:          "
            f"{row.accuracy_difference:+.4f}"
        )

        print(
            f"    Baseline-only wins:  "
            f"{row.baseline_correct_comparison_wrong}"
        )

        print(
            f"    Comparison-only wins:"
            f"  {row.baseline_wrong_comparison_correct}"
        )

        print(
            f"    Discordant pairs:    "
            f"{row.discordant_pairs}"
        )

        print(
            f"    Exact p-value:       "
            f"{row.mcnemar_exact_p:.6f}"
        )

        print(
            f"    Interpretation:      "
            f"{row.interpretation}"
        )

    print(
        "\nSaved:"
    )

    print(
        results_path
    )

    print(
        paired_path
    )

    print(
        metadata_path
    )


if __name__ == "__main__":
    main()
    