from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VERIFICATION_ROOT = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "arcface_reference_aligned"
    / "verification"
)

FAIRNESS_ROOT = (
    VERIFICATION_ROOT
    / "fairness"
    / "view1_test"
)

BOOTSTRAP_ROOT = (
    FAIRNESS_ROOT
    / "bootstrap"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "reference_a"
    / "report_pack"
)

PAIR_SCORES = (
    VERIFICATION_ROOT
    / "pair_scores.csv"
)

METRICS = (
    VERIFICATION_ROOT
    / "verification_metrics.csv"
)

DISTRIBUTIONS = (
    VERIFICATION_ROOT
    / "score_distributions.csv"
)

SUBGROUP = (
    FAIRNESS_ROOT
    / "subgroup_fairness.csv"
)

PAIR_COVERAGE = (
    FAIRNESS_ROOT
    / "pair_demographic_coverage.csv"
)

IMAGE_COVERAGE = (
    FAIRNESS_ROOT
    / "image_demographic_coverage.csv"
)

BOOTSTRAP = (
    BOOTSTRAP_ROOT
    / "bootstrap_confidence_intervals.csv"
)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Reference A result is missing:\n{path}"
        )


def safe_divide(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return np.nan

    return float(
        numerator / denominator
    )


def roc_points(
    labels: np.ndarray,
    scores: np.ndarray,
) -> pd.DataFrame:
    labels = np.asarray(
        labels,
        dtype=bool,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    labels = labels[order]
    scores = scores[order]

    positives = int(
        labels.sum()
    )

    negatives = int(
        (~labels).sum()
    )

    tp = np.cumsum(
        labels
    )

    fp = np.cumsum(
        ~labels
    )

    distinct_end = np.r_[
        scores[:-1] != scores[1:],
        True,
    ]

    idx = np.where(
        distinct_end
    )[0]

    frame = pd.DataFrame(
        {
            "threshold": scores[idx],
            "TPR": (
                tp[idx]
                / positives
            ),
            "FPR": (
                fp[idx]
                / negatives
            ),
        }
    )

    origin = pd.DataFrame(
        {
            "threshold": [np.inf],
            "TPR": [0.0],
            "FPR": [0.0],
        }
    )

    return pd.concat(
        [
            origin,
            frame,
        ],
        ignore_index=True,
    )


def corrected_disparities(
    subgroup: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    definitions = {
        "TPR_TAR": "n_genuine",
        "FNMR_FNR": "n_genuine",
        "FMR_FPR": "n_impostor",
    }

    for attribute, group in subgroup.groupby(
        "attribute"
    ):
        row = {
            "attribute": attribute,
            "n_groups": int(
                len(group)
            ),
        }

        for metric, denominator in (
            definitions.items()
        ):
            eligible = group[
                group[denominator] > 0
            ][
                [
                    "group",
                    metric,
                ]
            ].dropna()

            n_eligible = int(
                len(eligible)
            )

            row[
                f"{metric}_estimable_groups"
            ] = n_eligible

            if n_eligible < 2:
                row[
                    f"{metric}_min"
                ] = np.nan

                row[
                    f"{metric}_max"
                ] = np.nan

                row[
                    f"{metric}_gap"
                ] = np.nan

            else:
                values = eligible[
                    metric
                ].astype(float)

                row[
                    f"{metric}_min"
                ] = float(
                    values.min()
                )

                row[
                    f"{metric}_max"
                ] = float(
                    values.max()
                )

                row[
                    f"{metric}_gap"
                ] = float(
                    values.max()
                    - values.min()
                )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


def build_confusion_matrix(
    test: pd.DataFrame,
) -> pd.DataFrame:
    actual = (
        test[
            "is_genuine"
        ].astype(bool)
    )

    predicted = (
        test[
            "predicted_genuine"
        ].astype(bool)
    )

    tp = int(
        (
            actual
            & predicted
        ).sum()
    )

    fn = int(
        (
            actual
            & ~predicted
        ).sum()
    )

    fp = int(
        (
            ~actual
            & predicted
        ).sum()
    )

    tn = int(
        (
            ~actual
            & ~predicted
        ).sum()
    )

    genuine_n = (
        tp + fn
    )

    impostor_n = (
        tn + fp
    )

    return pd.DataFrame(
        [
            {
                "actual_class": "Genuine",
                "predicted_genuine_pct": (
                    safe_divide(
                        tp,
                        genuine_n,
                    )
                ),
                "predicted_impostor_pct": (
                    safe_divide(
                        fn,
                        genuine_n,
                    )
                ),
                "n": genuine_n,
            },
            {
                "actual_class": "Impostor",
                "predicted_genuine_pct": (
                    safe_divide(
                        fp,
                        impostor_n,
                    )
                ),
                "predicted_impostor_pct": (
                    safe_divide(
                        tn,
                        impostor_n,
                    )
                ),
                "n": impostor_n,
            },
        ]
    )


def select_examples(
    test: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    frame = test.copy()

    actual = (
        frame[
            "is_genuine"
        ].astype(bool)
    )

    predicted = (
        frame[
            "predicted_genuine"
        ].astype(bool)
    )

    frame[
        "outcome"
    ] = np.select(
        [
            actual & predicted,
            ~actual & ~predicted,
            ~actual & predicted,
            actual & ~predicted,
        ],
        [
            "TP",
            "TN",
            "FP",
            "FN",
        ],
        default="Unknown",
    )

    selected = []

    specifications = [
        (
            "TP",
            False,
        ),
        (
            "TN",
            True,
        ),
        (
            "FP",
            False,
        ),
        (
            "FN",
            True,
        ),
    ]

    for outcome, ascending in specifications:
        candidates = frame[
            frame[
                "outcome"
            ].eq(
                outcome
            )
        ].copy()

        if candidates.empty:
            continue

        candidates = candidates.sort_values(
            "cosine_similarity",
            ascending=ascending,
        )

        chosen = candidates.head(
            2
        ).copy()

        selected.append(
            chosen
        )

    if not selected:
        return pd.DataFrame()

    examples = pd.concat(
        selected,
        ignore_index=False,
    )

    examples = examples.reset_index(
        names="pair_index"
    )

    examples[
        "threshold"
    ] = threshold

    examples[
        "actual_pair_type"
    ] = np.where(
        examples[
            "is_genuine"
        ].astype(bool),
        "Genuine",
        "Impostor",
    )

    examples[
        "model_decision"
    ] = np.where(
        examples[
            "predicted_genuine"
        ].astype(bool),
        "Genuine",
        "Impostor",
    )

    return examples[
        [
            "pair_index",
            "outcome",
            "actual_pair_type",
            "model_decision",
            "cosine_similarity",
            "threshold",
        ]
    ]


def main() -> None:
    required = [
        PAIR_SCORES,
        METRICS,
        DISTRIBUTIONS,
        SUBGROUP,
        PAIR_COVERAGE,
        IMAGE_COVERAGE,
        BOOTSTRAP,
    ]

    for path in required:
        require(
            path
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores = pd.read_csv(
        PAIR_SCORES
    )

    metrics = pd.read_csv(
        METRICS
    )

    distributions = pd.read_csv(
        DISTRIBUTIONS
    )

    subgroup = pd.read_csv(
        SUBGROUP
    )

    pair_coverage = pd.read_csv(
        PAIR_COVERAGE
    )

    image_coverage = pd.read_csv(
        IMAGE_COVERAGE
    )

    bootstrap = pd.read_csv(
        BOOTSTRAP
    )

    test = scores[
        scores[
            "split"
        ]
        .astype(str)
        .str.lower()
        .eq("test")
    ].copy()

    train = scores[
        scores[
            "split"
        ]
        .astype(str)
        .str.lower()
        .eq("train")
    ].copy()

    test_metrics = metrics[
        metrics[
            "split"
        ]
        .astype(str)
        .str.lower()
        .eq("test")
    ].iloc[0]

    train_metrics = metrics[
        metrics[
            "split"
        ]
        .astype(str)
        .str.lower()
        .eq("train")
    ].iloc[0]

    threshold = float(
        test_metrics[
            "threshold"
        ]
    )

    # --------------------------------------------------------
    # Sanitised pair scores
    # --------------------------------------------------------

    safe_scores = scores[
        [
            "split",
            "is_genuine",
            "cosine_similarity",
            "predicted_genuine",
        ]
    ].copy()

    safe_scores.insert(
        0,
        "pair_index",
        np.arange(
            len(safe_scores),
            dtype=int,
        ),
    )

    safe_scores.to_csv(
        OUTPUT_ROOT
        / "pair_scores_sanitized.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Verification metrics
    # --------------------------------------------------------

    metrics.to_csv(
        OUTPUT_ROOT
        / "verification_metrics.csv",
        index=False,
    )

    distributions.to_csv(
        OUTPUT_ROOT
        / "score_distributions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    roc = roc_points(
        test[
            "is_genuine"
        ].to_numpy(
            dtype=bool
        ),
        test[
            "cosine_similarity"
        ].to_numpy(
            dtype=float
        ),
    )

    roc.to_csv(
        OUTPUT_ROOT
        / "roc_test.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Confusion matrix percentages
    # --------------------------------------------------------

    confusion = (
        build_confusion_matrix(
            test
        )
    )

    confusion.to_csv(
        OUTPUT_ROOT
        / "confusion_matrix_percent.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Fairness
    # --------------------------------------------------------

    subgroup.to_csv(
        OUTPUT_ROOT
        / "subgroup_fairness.csv",
        index=False,
    )

    pair_coverage.to_csv(
        OUTPUT_ROOT
        / "pair_demographic_coverage.csv",
        index=False,
    )

    image_coverage.to_csv(
        OUTPUT_ROOT
        / "image_demographic_coverage.csv",
        index=False,
    )

    disparities = (
        corrected_disparities(
            subgroup
        )
    )

    disparities.to_csv(
        OUTPUT_ROOT
        / "fairness_disparity_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    bootstrap_overall = bootstrap[
        bootstrap[
            "analysis"
        ].eq(
            "overall"
        )
    ].copy()

    bootstrap_overall.to_csv(
        OUTPUT_ROOT
        / "bootstrap_overall.csv",
        index=False,
    )

    subgroup_bootstrap = bootstrap[
        bootstrap[
            "analysis"
        ].eq(
            "subgroup"
        )
    ].copy()

    subgroup_bootstrap.to_csv(
        OUTPUT_ROOT
        / "bootstrap_subgroups.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Raw diagnostic versus aligned result
    # --------------------------------------------------------

    test_distribution = distributions[
        distributions[
            "split"
        ]
        .astype(str)
        .str.lower()
        .eq("test")
    ]

    genuine_mean = float(
        test_distribution[
            test_distribution[
                "pair_type"
            ].eq(
                "genuine"
            )
        ][
            "mean"
        ].iloc[0]
    )

    impostor_mean = float(
        test_distribution[
            test_distribution[
                "pair_type"
            ].eq(
                "impostor"
            )
        ][
            "mean"
        ].iloc[0]
    )

    aligned_separation = (
        genuine_mean
        - impostor_mean
    )

    comparison = pd.DataFrame(
        [
            {
                "pipeline": (
                    "Raw-image diagnostic"
                ),
                "official_result": False,
                "accuracy": 0.5810,
                "TPR_TAR": 0.4400,
                "FMR_FPR": 0.2780,
                "FNMR_FNR": 0.5600,
                "ROC_AUC": 0.5921,
                "EER": 0.4330,
                "mean_score_separation": (
                    0.013381
                ),
            },
            {
                "pipeline": (
                    "Project-aligned Reference A"
                ),
                "official_result": True,
                "accuracy": float(
                    test_metrics[
                        "accuracy"
                    ]
                ),
                "TPR_TAR": float(
                    test_metrics[
                        "recall_TPR_TAR"
                    ]
                ),
                "FMR_FPR": float(
                    test_metrics[
                        "FMR_FPR"
                    ]
                ),
                "FNMR_FNR": float(
                    test_metrics[
                        "FNMR_FNR"
                    ]
                ),
                "ROC_AUC": float(
                    test_metrics[
                        "ROC_AUC"
                    ]
                ),
                "EER": float(
                    test_metrics[
                        "EER"
                    ]
                ),
                "mean_score_separation": (
                    aligned_separation
                ),
            },
        ]
    )

    comparison.to_csv(
        OUTPUT_ROOT
        / "preprocessing_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Qualitative example selection
    # --------------------------------------------------------

    examples = select_examples(
        test,
        threshold,
    )

    examples.to_csv(
        OUTPUT_ROOT
        / "qualitative_examples.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary metadata
    # --------------------------------------------------------

    summary = {
        "model_id": "ReferenceA_ArcFace",
        "role": (
            "pretrained recognition reference"
        ),
        "mitigation_baseline": False,
        "dataset": "LFW",
        "protocol": "View 1 development",
        "unique_images": 4992,
        "train_pairs": int(
            len(train)
        ),
        "test_pairs": int(
            len(test)
        ),
        "embedding_dimension": 512,
        "similarity": (
            "cosine similarity"
        ),
        "threshold": threshold,
        "threshold_source": (
            "View 1 train only"
        ),
        "accuracy": float(
            test_metrics[
                "accuracy"
            ]
        ),
        "TPR_TAR": float(
            test_metrics[
                "recall_TPR_TAR"
            ]
        ),
        "FMR_FPR": float(
            test_metrics[
                "FMR_FPR"
            ]
        ),
        "FNMR_FNR": float(
            test_metrics[
                "FNMR_FNR"
            ]
        ),
        "ROC_AUC": float(
            test_metrics[
                "ROC_AUC"
            ]
        ),
        "EER": float(
            test_metrics[
                "EER"
            ]
        ),
        "view2_used": False,
        "demographics": (
            "externally derived LFW demographic annotations; "
            "not demographic ground truth"
        ),
        "bootstrap_iterations": 2000,
        "bootstrap_confidence": 0.95,
    }

    (
        OUTPUT_ROOT
        / "reference_a_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "REFERENCE A REPORT PACK COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Output: {OUTPUT_ROOT}"
    )

    print(
        "\nOfficial test result:"
    )

    print(
        f"  Accuracy: "
        f"{summary['accuracy']:.2%}"
    )

    print(
        f"  TAR:      "
        f"{summary['TPR_TAR']:.2%}"
    )

    print(
        f"  FMR:      "
        f"{summary['FMR_FPR']:.2%}"
    )

    print(
        f"  FNMR:     "
        f"{summary['FNMR_FNR']:.2%}"
    )

    print(
        f"  AUC:      "
        f"{summary['ROC_AUC']:.4f}"
    )

    print(
        f"  EER:      "
        f"{summary['EER']:.2%}"
    )

    print(
        "\nNo source image paths or embeddings "
        "are written to the report pack."
    )


if __name__ == "__main__":
    main()
    