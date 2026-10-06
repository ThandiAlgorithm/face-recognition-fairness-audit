from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PAIRS = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "arcface_reference"
    / "lfw_view1_pairs_resolved.csv"
)


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic LFW View 1 verification evaluator for "
            "ArcFace, M0, M1, M2, M3 or any embedding model."
        )
    )

    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="NPZ containing embeddings and raw_image_paths.",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model identifier.",
    )

    parser.add_argument(
        "--pairs",
        type=Path,
        default=DEFAULT_PAIRS,
        help="Resolved LFW View 1 pair CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for verification outputs.",
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def canonical_lfw_path(value) -> str:
    path = (
        str(value)
        .strip()
        .replace("\\", "/")
    )

    lower = path.lower()

    marker = "/data/lfw/"

    if marker in lower:
        index = lower.index(marker)
        path = path[
            index + len("/data/"):
        ]

    else:
        parts = path.split("/")

        for i, part in enumerate(parts):
            if part.lower() == "lfw":
                path = "/".join(parts[i:])
                break

    return path.lower()


def to_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    result = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if result.isna().any():
        bad = (
            series[result.isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Could not parse boolean labels: "
            + ", ".join(bad[:10])
        )

    return result.astype(bool)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= 0):
        raise ValueError(
            "At least one embedding has zero norm."
        )

    return matrix / norms


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing embedding file:\n{path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        if "embeddings" not in data:
            raise KeyError(
                "NPZ does not contain 'embeddings'."
            )

        if "raw_image_paths" not in data:
            raise KeyError(
                "NPZ does not contain 'raw_image_paths'."
            )

        embeddings = (
            data["embeddings"]
            .astype(np.float32)
        )

        raw_paths = (
            data["raw_image_paths"]
            .astype(str)
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2-D matrix."
        )

    if len(raw_paths) != len(embeddings):
        raise ValueError(
            "raw_image_paths length does not match "
            "embedding matrix."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embedding matrix contains NaN or infinity."
        )

    embeddings = l2_normalize(
        embeddings
    )

    keys = [
        canonical_lfw_path(path)
        for path in raw_paths
    ]

    if len(set(keys)) != len(keys):
        raise RuntimeError(
            "Duplicate image paths detected in "
            "embedding file."
        )

    lookup = {
        key: index
        for index, key in enumerate(keys)
    }

    return (
        embeddings,
        raw_paths,
        lookup,
    )


# ============================================================
# LOAD PAIRS
# ============================================================

def load_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pair file:\n{path}"
        )

    pairs = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "image1_path",
        "image2_path",
        "is_genuine",
        "split",
    }

    missing = (
        required
        - set(pairs.columns)
    )

    if missing:
        raise KeyError(
            "Pair file missing columns: "
            + ", ".join(sorted(missing))
        )

    pairs["is_genuine"] = (
        to_bool_series(
            pairs["is_genuine"]
        )
    )

    pairs["split"] = (
        pairs["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    pairs["image1_key"] = (
        pairs["image1_path"]
        .map(canonical_lfw_path)
    )

    pairs["image2_key"] = (
        pairs["image2_path"]
        .map(canonical_lfw_path)
    )

    return pairs


# ============================================================
# SCORE PAIRS
# ============================================================

def score_pairs(
    pairs: pd.DataFrame,
    embeddings: np.ndarray,
    lookup: dict,
) -> pd.DataFrame:

    missing1 = (
        ~pairs["image1_key"]
        .isin(lookup)
    )

    missing2 = (
        ~pairs["image2_key"]
        .isin(lookup)
    )

    if missing1.any() or missing2.any():
        raise RuntimeError(
            "Some pair images do not have embeddings. "
            f"image1 missing={missing1.sum()}, "
            f"image2 missing={missing2.sum()}"
        )

    index1 = (
        pairs["image1_key"]
        .map(lookup)
        .to_numpy(dtype=int)
    )

    index2 = (
        pairs["image2_key"]
        .map(lookup)
        .to_numpy(dtype=int)
    )

    # Embeddings are L2-normalized,
    # therefore dot product = cosine similarity.
    similarity = np.sum(
        embeddings[index1]
        * embeddings[index2],
        axis=1,
    )

    result = pairs.copy()

    result[
        "cosine_similarity"
    ] = similarity.astype(float)

    return result


# ============================================================
# ROC / THRESHOLD
# ============================================================

def roc_points(
    labels: np.ndarray,
    scores: np.ndarray,
):
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

    scores_sorted = scores[
        order
    ]

    labels_sorted = labels[
        order
    ]

    positives = int(
        labels.sum()
    )

    negatives = int(
        (~labels).sum()
    )

    if positives == 0 or negatives == 0:
        raise ValueError(
            "ROC requires both genuine and impostor pairs."
        )

    tp_cumulative = np.cumsum(
        labels_sorted
    )

    fp_cumulative = np.cumsum(
        ~labels_sorted
    )

    # Evaluate only at the final occurrence
    # of each tied score.
    distinct_end = np.r_[
        scores_sorted[:-1]
        != scores_sorted[1:],
        True,
    ]

    indices = np.where(
        distinct_end
    )[0]

    thresholds = scores_sorted[
        indices
    ]

    tp = tp_cumulative[
        indices
    ]

    fp = fp_cumulative[
        indices
    ]

    tpr = tp / positives
    fpr = fp / negatives

    # Add operating point above maximum score.
    thresholds = np.r_[
        np.inf,
        thresholds,
    ]

    tpr = np.r_[
        0.0,
        tpr,
    ]

    fpr = np.r_[
        0.0,
        fpr,
    ]

    return (
        thresholds,
        tpr,
        fpr,
    )


def choose_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
):
    thresholds, tpr, fpr = (
        roc_points(
            labels,
            scores,
        )
    )

    youden = (
        tpr - fpr
    )

    best_value = np.max(
        youden
    )

    best_indices = np.where(
        np.isclose(
            youden,
            best_value,
            rtol=0.0,
            atol=1e-12,
        )
    )[0]

    # Deterministic tie rule:
    # ROC points are ordered from highest threshold
    # to lowest threshold, so choose the first maximum.
    best_index = int(
        best_indices[0]
    )

    threshold = float(
        thresholds[
            best_index
        ]
    )

    if not np.isfinite(
        threshold
    ):
        raise RuntimeError(
            "Threshold selection produced infinity."
        )

    return {
        "threshold": threshold,
        "youden_j": float(
            youden[
                best_index
            ]
        ),
        "train_tpr": float(
            tpr[
                best_index
            ]
        ),
        "train_fpr": float(
            fpr[
                best_index
            ]
        ),
        "tie_rule": (
            "highest threshold among equal "
            "maximum-Youden-J solutions"
        ),
    }


# ============================================================
# METRICS
# ============================================================

def safe_divide(
    numerator,
    denominator,
):
    if denominator == 0:
        return np.nan

    return float(
        numerator
        / denominator
    )


def auc_rank(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    labels = np.asarray(
        labels,
        dtype=bool,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    n_pos = int(
        labels.sum()
    )

    n_neg = int(
        (~labels).sum()
    )

    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = (
        pd.Series(scores)
        .rank(
            method="average"
        )
        .to_numpy()
    )

    rank_sum_pos = float(
        ranks[
            labels
        ].sum()
    )

    auc = (
        rank_sum_pos
        - n_pos
        * (
            n_pos + 1
        )
        / 2.0
    ) / (
        n_pos
        * n_neg
    )

    return float(
        auc
    )


def eer_estimate(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    _, tpr, fpr = (
        roc_points(
            labels,
            scores,
        )
    )

    fnr = (
        1.0 - tpr
    )

    index = int(
        np.argmin(
            np.abs(
                fpr - fnr
            )
        )
    )

    return float(
        (
            fpr[index]
            + fnr[index]
        )
        / 2.0
    )


def compute_metrics(
    frame: pd.DataFrame,
    threshold: float,
) -> dict:
    actual = (
        frame[
            "is_genuine"
        ]
        .to_numpy(
            dtype=bool
        )
    )

    scores = (
        frame[
            "cosine_similarity"
        ]
        .to_numpy(
            dtype=float
        )
    )

    predicted = (
        scores
        >= threshold
    )

    tp = int(
        np.sum(
            actual
            & predicted
        )
    )

    tn = int(
        np.sum(
            ~actual
            & ~predicted
        )
    )

    fp = int(
        np.sum(
            ~actual
            & predicted
        )
    )

    fn = int(
        np.sum(
            actual
            & ~predicted
        )
    )

    positives = (
        tp + fn
    )

    negatives = (
        tn + fp
    )

    total = (
        positives
        + negatives
    )

    tpr = safe_divide(
        tp,
        positives,
    )

    fmr = safe_divide(
        fp,
        negatives,
    )

    fnmr = safe_divide(
        fn,
        positives,
    )

    specificity = safe_divide(
        tn,
        negatives,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    accuracy = safe_divide(
        tp + tn,
        total,
    )

    if (
        not np.isnan(tpr)
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

    return {
        "n_pairs": total,
        "n_genuine": positives,
        "n_impostor": negatives,
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "precision": precision,
        "recall_TPR_TAR": tpr,
        "F1": f1,
        "FMR_FPR": fmr,
        "FNMR_FNR": fnmr,
        "specificity_TNR": (
            specificity
        ),
        "ROC_AUC": auc_rank(
            actual,
            scores,
        ),
        "EER": eer_estimate(
            actual,
            scores,
        ),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


# ============================================================
# DISTRIBUTIONS
# ============================================================

def distribution_summary(
    frame: pd.DataFrame,
    split_name: str,
) -> list[dict]:
    rows = []

    for genuine_value, label in [
        (True, "genuine"),
        (False, "impostor"),
    ]:
        values = (
            frame[
                frame[
                    "is_genuine"
                ]
                == genuine_value
            ][
                "cosine_similarity"
            ]
            .to_numpy(
                dtype=float
            )
        )

        rows.append(
            {
                "split": split_name,
                "pair_type": label,
                "n": len(values),
                "mean": float(
                    np.mean(values)
                ),
                "std": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                ),
                "median": float(
                    np.median(
                        values
                    )
                ),
                "min": float(
                    np.min(values)
                ),
                "max": float(
                    np.max(values)
                ),
            }
        )

    return rows


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    embedding_path = resolve(
        args.embeddings
    )

    pair_path = resolve(
        args.pairs
    )

    output_dir = resolve(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings, raw_paths, lookup = (
        load_embeddings(
            embedding_path
        )
    )

    pairs = load_pairs(
        pair_path
    )

    scored = score_pairs(
        pairs,
        embeddings,
        lookup,
    )

    train = scored[
        scored[
            "split"
        ]
        == "train"
    ].copy()

    test = scored[
        scored[
            "split"
        ]
        == "test"
    ].copy()

    if len(train) != 2200:
        raise RuntimeError(
            "Expected 2,200 View 1 train pairs, "
            f"found {len(train)}."
        )

    if len(test) != 1000:
        raise RuntimeError(
            "Expected 1,000 View 1 test pairs, "
            f"found {len(test)}."
        )

    threshold_info = (
        choose_threshold(
            train[
                "is_genuine"
            ].to_numpy(
                dtype=bool
            ),
            train[
                "cosine_similarity"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    threshold = (
        threshold_info[
            "threshold"
        ]
    )

    scored[
        "predicted_genuine"
    ] = (
        scored[
            "cosine_similarity"
        ]
        >= threshold
    )

    train = scored[
        scored[
            "split"
        ]
        == "train"
    ].copy()

    test = scored[
        scored[
            "split"
        ]
        == "test"
    ].copy()

    train_metrics = (
        compute_metrics(
            train,
            threshold,
        )
    )

    test_metrics = (
        compute_metrics(
            test,
            threshold,
        )
    )

    metrics = pd.DataFrame(
        [
            {
                "model_id": (
                    args.model_id
                ),
                "split": "train",
                **train_metrics,
            },
            {
                "model_id": (
                    args.model_id
                ),
                "split": "test",
                **test_metrics,
            },
        ]
    )

    distributions = pd.DataFrame(
        distribution_summary(
            train,
            "train",
        )
        + distribution_summary(
            test,
            "test",
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    score_path = (
        output_dir
        / "pair_scores.csv"
    )

    metrics_path = (
        output_dir
        / "verification_metrics.csv"
    )

    distribution_path = (
        output_dir
        / "score_distributions.csv"
    )

    metadata_path = (
        output_dir
        / "verification_metadata.json"
    )

    scored.to_csv(
        score_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    distributions.to_csv(
        distribution_path,
        index=False,
    )

    metadata = {
        "model_id": (
            args.model_id
        ),
        "embedding_file": str(
            embedding_path
        ),
        "pair_file": str(
            pair_path
        ),
        "n_embeddings": int(
            len(embeddings)
        ),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "embedding_l2_normalized": True,
        "similarity": (
            "cosine similarity"
        ),
        "higher_score_more_similar": True,
        "threshold_source": (
            "LFW View 1 train only"
        ),
        "threshold_criterion": (
            "maximum Youden J"
        ),
        "threshold_tie_rule": (
            threshold_info[
                "tie_rule"
            ]
        ),
        "frozen_threshold": (
            threshold
        ),
        "view1_train_pairs": (
            len(train)
        ),
        "view1_test_pairs": (
            len(test)
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
        "GENERIC LFW VIEW 1 VERIFICATION EVALUATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Model: {args.model_id}"
    )

    print(
        f"Embeddings: "
        f"{embeddings.shape}"
    )

    print(
        f"Train pairs: {len(train):,}"
    )

    print(
        f"Test pairs:  {len(test):,}"
    )

    print(
        "\nFrozen threshold "
        "(selected from train only):"
    )

    print(
        f"  {threshold:.6f}"
    )

    print(
        f"  Youden J: "
        f"{threshold_info['youden_j']:.4f}"
    )

    print(
        "\nTRAIN:"
    )

    for key in [
        "accuracy",
        "balanced_accuracy",
        "recall_TPR_TAR",
        "FMR_FPR",
        "FNMR_FNR",
        "ROC_AUC",
        "EER",
    ]:
        print(
            f"  {key:<22}"
            f"{train_metrics[key]:.4f}"
        )

    print(
        "\nTEST:"
    )

    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_TPR_TAR",
        "F1",
        "FMR_FPR",
        "FNMR_FNR",
        "ROC_AUC",
        "EER",
    ]:
        print(
            f"  {key:<22}"
            f"{test_metrics[key]:.4f}"
        )

    print(
        "\nTEST confusion counts:"
    )

    print(
        f"  TP={test_metrics['TP']}  "
        f"TN={test_metrics['TN']}  "
        f"FP={test_metrics['FP']}  "
        f"FN={test_metrics['FN']}"
    )

    print(
        "\nTEST score distributions:"
    )

    test_distribution = (
        distributions[
            distributions[
                "split"
            ]
            == "test"
        ]
    )

    for row in (
        test_distribution
        .itertuples()
    ):
        print(
            f"  {row.pair_type:<9} "
            f"mean={row.mean:.6f}  "
            f"std={row.std:.6f}  "
            f"median={row.median:.6f}"
        )

    genuine_mean = float(
        test_distribution[
            test_distribution[
                "pair_type"
            ]
            == "genuine"
        ][
            "mean"
        ].iloc[0]
    )

    impostor_mean = float(
        test_distribution[
            test_distribution[
                "pair_type"
            ]
            == "impostor"
        ][
            "mean"
        ].iloc[0]
    )

    print(
        "\nMean score separation:"
    )

    print(
        f"  {genuine_mean - impostor_mean:.6f}"
    )

    print(
        "\nSaved:"
    )

    print(score_path)
    print(metrics_path)
    print(distribution_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
    