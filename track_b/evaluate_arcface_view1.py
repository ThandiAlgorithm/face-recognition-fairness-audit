from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "arcface_reference"
)

PAIR_PATH = (
    RESULT_ROOT
    / "lfw_view1_pairs_resolved.csv"
)

EMBEDDING_PATH = (
    RESULT_ROOT
    / "embeddings"
    / "arcface_embeddings.npz"
)

OUTPUT_ROOT = (
    RESULT_ROOT
    / "verification"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

PAIR_SCORE_PATH = (
    OUTPUT_ROOT
    / "arcface_view1_pair_scores.csv"
)

METRICS_PATH = (
    OUTPUT_ROOT
    / "arcface_view1_metrics.csv"
)

THRESHOLD_PATH = (
    OUTPUT_ROOT
    / "arcface_view1_threshold.json"
)

ROC_POINTS_PATH = (
    OUTPUT_ROOT
    / "arcface_view1_roc_points.csv"
)

ROC_FIGURE_PATH = (
    OUTPUT_ROOT
    / "01_arcface_view1_roc.png"
)

SCORE_FIGURE_PATH = (
    OUTPUT_ROOT
    / "02_arcface_view1_score_distributions.png"
)

CONFUSION_FIGURE_PATH = (
    OUTPUT_ROOT
    / "03_arcface_view1_test_confusion_matrix.png"
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embedding_cache():
    if not EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"Missing embedding cache:\n{EMBEDDING_PATH}"
        )

    with np.load(
        EMBEDDING_PATH,
        allow_pickle=False,
    ) as data:

        embeddings = (
            data["embeddings"]
            .astype(np.float32)
        )

        image_paths = (
            data["image_paths"]
            .astype(str)
        )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
    ):
        raise ValueError(
            "Embeddings are not L2-normalized."
        )

    path_to_index = {
        path: index
        for index, path
        in enumerate(image_paths)
    }

    return (
        embeddings,
        path_to_index,
    )


# ============================================================
# PAIR SCORING
# ============================================================

def score_pairs(
    pairs: pd.DataFrame,
    embeddings: np.ndarray,
    path_to_index: dict[str, int],
) -> pd.DataFrame:

    out = pairs.copy()

    scores = []

    missing_paths = []

    for row in out.itertuples():

        path1 = str(
            row.image1_path
        )

        path2 = str(
            row.image2_path
        )

        if path1 not in path_to_index:
            missing_paths.append(path1)
            continue

        if path2 not in path_to_index:
            missing_paths.append(path2)
            continue

        emb1 = embeddings[
            path_to_index[path1]
        ]

        emb2 = embeddings[
            path_to_index[path2]
        ]

        # Embeddings are already L2-normalized,
        # therefore dot product = cosine similarity.
        similarity = float(
            np.dot(
                emb1,
                emb2,
            )
        )

        scores.append(
            similarity
        )

    if missing_paths:
        unique_missing = sorted(
            set(missing_paths)
        )

        raise KeyError(
            "Some pair images do not have cached embeddings:\n"
            + "\n".join(
                unique_missing[:20]
            )
        )

    if len(scores) != len(out):
        raise RuntimeError(
            "Pair score count does not match pair table."
        )

    out["cosine_similarity"] = scores

    return out


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
):
    """
    Select one global threshold using View 1 TRAIN ONLY.

    Criterion:
        maximise Youden's J = TPR - FPR

    This is equivalent to maximising balanced accuracy.

    Higher cosine similarity means "same identity".
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
    )

    youden_j = (
        tpr - fpr
    )

    best_index = int(
        np.argmax(
            youden_j
        )
    )

    threshold = float(
        thresholds[
            best_index
        ]
    )

    return {
        "threshold": threshold,
        "criterion": (
            "Maximum Youden J / balanced accuracy"
        ),
        "tpr_at_selection": float(
            tpr[
                best_index
            ]
        ),
        "fpr_at_selection": float(
            fpr[
                best_index
            ]
        ),
        "youden_j": float(
            youden_j[
                best_index
            ]
        ),
    }


# ============================================================
# EER
# ============================================================

def calculate_eer(
    y_true: np.ndarray,
    scores: np.ndarray,
):
    """
    Approximate EER using the ROC operating point
    where FPR and FNR are closest.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
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

    eer = float(
        (
            fpr[index]
            + fnr[index]
        )
        / 2.0
    )

    return {
        "eer": eer,
        "eer_threshold": float(
            thresholds[index]
        ),
        "fpr_at_eer": float(
            fpr[index]
        ),
        "fnr_at_eer": float(
            fnr[index]
        ),
    }


# ============================================================
# METRICS
# ============================================================

def evaluate_split(
    split_name: str,
    frame: pd.DataFrame,
    threshold: float,
) -> dict:

    y_true = (
        frame["is_genuine"]
        .astype(bool)
        .to_numpy()
    )

    scores = (
        frame["cosine_similarity"]
        .astype(float)
        .to_numpy()
    )

    y_pred = (
        scores >= threshold
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[
                False,
                True,
            ],
        )
        .ravel()
    )

    tpr = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    fnmr = (
        fn / (tp + fn)
        if (tp + fn) > 0
        else np.nan
    )

    fmr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else np.nan
    )

    tnr = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    auc = roc_auc_score(
        y_true,
        scores,
    )

    eer_data = calculate_eer(
        y_true,
        scores,
    )

    return {
        "split": split_name,
        "n_pairs": int(
            len(frame)
        ),
        "n_genuine": int(
            y_true.sum()
        ),
        "n_impostor": int(
            (~y_true).sum()
        ),
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall_tpr_tar": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity_tnr": float(
            tnr
        ),
        "fmr_fpr": float(
            fmr
        ),
        "fnmr_fnr": float(
            fnmr
        ),
        "roc_auc": float(
            auc
        ),
        "eer": float(
            eer_data[
                "eer"
            ]
        ),
        "eer_threshold_descriptive": float(
            eer_data[
                "eer_threshold"
            ]
        ),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# ============================================================
# ROC OUTPUT
# ============================================================

def build_roc_table(
    frame: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:

    y_true = (
        frame["is_genuine"]
        .astype(bool)
        .to_numpy()
    )

    scores = (
        frame["cosine_similarity"]
        .astype(float)
        .to_numpy()
    )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
    )

    return pd.DataFrame(
        {
            "split": split_name,
            "fpr": fpr,
            "tpr": tpr,
            "fnr": 1.0 - tpr,
            "threshold": thresholds,
        }
    )


# ============================================================
# FIGURES
# ============================================================

def make_roc_figure(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for name, frame in [
        ("View 1 train", train),
        ("View 1 test", test),
    ]:

        y_true = (
            frame["is_genuine"]
            .astype(bool)
            .to_numpy()
        )

        scores = (
            frame["cosine_similarity"]
            .astype(float)
            .to_numpy()
        )

        fpr, tpr, _ = roc_curve(
            y_true,
            scores,
        )

        auc = roc_auc_score(
            y_true,
            scores,
        )

        ax.plot(
            fpr,
            tpr,
            label=(
                f"{name} "
                f"(AUC={auc:.4f})"
            ),
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Chance",
    )

    ax.set_xlabel(
        "False Match Rate / FPR"
    )

    ax.set_ylabel(
        "True Match Rate / TPR"
    )

    ax.set_title(
        "ArcFace Verification ROC — LFW View 1"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        ROC_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def make_score_distribution_figure(
    pairs: pd.DataFrame,
    threshold: float,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    genuine = pairs[
        pairs["is_genuine"]
    ][
        "cosine_similarity"
    ]

    impostor = pairs[
        ~pairs["is_genuine"]
    ][
        "cosine_similarity"
    ]

    ax.hist(
        impostor,
        bins=50,
        alpha=0.6,
        label="Impostor pairs",
    )

    ax.hist(
        genuine,
        bins=50,
        alpha=0.6,
        label="Genuine pairs",
    )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=(
            f"Frozen threshold = "
            f"{threshold:.4f}"
        ),
    )

    ax.set_xlabel(
        "Cosine similarity"
    )

    ax.set_ylabel(
        "Number of pairs"
    )

    ax.set_title(
        "ArcFace Genuine vs Impostor Similarity"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SCORE_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def make_confusion_figure(
    test: pd.DataFrame,
    threshold: float,
) -> None:

    y_true = (
        test["is_genuine"]
        .astype(bool)
        .to_numpy()
    )

    scores = (
        test["cosine_similarity"]
        .astype(float)
        .to_numpy()
    )

    y_pred = (
        scores >= threshold
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            False,
            True,
        ],
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        matrix
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    labels = [
        "Impostor",
        "Genuine",
    ]

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        "ArcFace LFW View 1 Test Confusion Matrix"
    )

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(
                    matrix[
                        row,
                        column,
                    ]
                ),
                ha="center",
                va="center",
            )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_FIGURE_PATH,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not PAIR_PATH.exists():
        raise FileNotFoundError(
            f"Missing pair file:\n{PAIR_PATH}"
        )

    pairs = pd.read_csv(
        PAIR_PATH,
        low_memory=False,
    )

    embeddings, path_to_index = (
        load_embedding_cache()
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ARCFACE LFW VIEW 1 VERIFICATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Pairs: {len(pairs):,}"
    )

    print(
        f"Cached embeddings: "
        f"{len(path_to_index):,}"
    )

    pairs = score_pairs(
        pairs,
        embeddings,
        path_to_index,
    )

    train = (
        pairs[
            pairs["split"]
            .eq("train")
        ]
        .copy()
    )

    test = (
        pairs[
            pairs["split"]
            .eq("test")
        ]
        .copy()
    )

    if len(train) != 2200:
        raise ValueError(
            f"Expected 2,200 train pairs, "
            f"found {len(train)}"
        )

    if len(test) != 1000:
        raise ValueError(
            f"Expected 1,000 test pairs, "
            f"found {len(test)}"
        )

    threshold_data = choose_threshold(
        train[
            "is_genuine"
        ]
        .astype(bool)
        .to_numpy(),
        train[
            "cosine_similarity"
        ]
        .astype(float)
        .to_numpy(),
    )

    frozen_threshold = (
        threshold_data[
            "threshold"
        ]
    )

    print(
        "\nThreshold selected on "
        "View 1 TRAIN only:"
    )

    print(
        f"{frozen_threshold:.6f}"
    )

    print(
        "Selection criterion:"
    )

    print(
        threshold_data[
            "criterion"
        ]
    )

    pairs[
        "predicted_genuine"
    ] = (
        pairs[
            "cosine_similarity"
        ]
        >= frozen_threshold
    )

    pairs.to_csv(
        PAIR_SCORE_PATH,
        index=False,
    )

    train_metrics = (
        evaluate_split(
            "view1_train",
            train,
            frozen_threshold,
        )
    )

    test_metrics = (
        evaluate_split(
            "view1_test",
            test,
            frozen_threshold,
        )
    )

    metrics = pd.DataFrame(
        [
            train_metrics,
            test_metrics,
        ]
    )

    metrics.to_csv(
        METRICS_PATH,
        index=False,
    )

    roc_table = pd.concat(
        [
            build_roc_table(
                train,
                "view1_train",
            ),
            build_roc_table(
                test,
                "view1_test",
            ),
        ],
        ignore_index=True,
    )

    roc_table.to_csv(
        ROC_POINTS_PATH,
        index=False,
    )

    threshold_record = {
        **threshold_data,
        "selected_on": (
            "LFW View 1 train"
        ),
        "used_for": (
            "LFW View 1 test"
        ),
        "score": (
            "cosine similarity"
        ),
        "decision_rule": (
            "predict genuine when "
            "cosine_similarity >= threshold"
        ),
        "view2_used": False,
    }

    THRESHOLD_PATH.write_text(
        json.dumps(
            threshold_record,
            indent=2,
        ),
        encoding="utf-8",
    )

    make_roc_figure(
        train,
        test,
    )

    make_score_distribution_figure(
        pairs,
        frozen_threshold,
    )

    make_confusion_figure(
        test,
        frozen_threshold,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 72
    )

    for row in metrics.itertuples():

        print(
            f"\n{row.split}"
        )

        print(
            f"  Accuracy:        "
            f"{row.accuracy:.4f}"
        )

        print(
            f"  Balanced acc.:   "
            f"{row.balanced_accuracy:.4f}"
        )

        print(
            f"  Precision:       "
            f"{row.precision:.4f}"
        )

        print(
            f"  Recall / TPR:    "
            f"{row.recall_tpr_tar:.4f}"
        )

        print(
            f"  F1:              "
            f"{row.f1:.4f}"
        )

        print(
            f"  FMR / FPR:       "
            f"{row.fmr_fpr:.4f}"
        )

        print(
            f"  FNMR / FNR:      "
            f"{row.fnmr_fnr:.4f}"
        )

        print(
            f"  ROC AUC:         "
            f"{row.roc_auc:.4f}"
        )

        print(
            f"  EER:             "
            f"{row.eer:.4f}"
        )

        print(
            f"  TP/TN/FP/FN:     "
            f"{row.tp}/"
            f"{row.tn}/"
            f"{row.fp}/"
            f"{row.fn}"
        )

    print(
        "\nFrozen threshold:"
    )

    print(
        f"{frozen_threshold:.6f}"
    )

    print(
        "\nView 2 used: NO"
    )

    print(
        "\nSaved:"
    )

    for path in [
        PAIR_SCORE_PATH,
        METRICS_PATH,
        THRESHOLD_PATH,
        ROC_POINTS_PATH,
        ROC_FIGURE_PATH,
        SCORE_FIGURE_PATH,
        CONFUSION_FIGURE_PATH,
    ]:
        print(
            path
        )


if __name__ == "__main__":
    main()
    