from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from deepface import DeepFace
from deepface.modules import preprocessing

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 26041

DEFAULT_MASTER = Path(
    "metadata/master_annotations.parquet"
)

DEFAULT_OUTPUT = Path(
    "results/model_audit/deepface/fairface_validation"
)

DEEPFACE_RACE_LABELS_RAW = [
    "asian",
    "indian",
    "black",
    "white",
    "middle eastern",
    "latino hispanic",
]

COMMON_RACE_LABELS = [
    "White",
    "Black",
    "Asian",
    "Indian",
    "Latino_Hispanic",
    "Middle Eastern",
]

GENDER_LABELS = [
    "Male",
    "Female",
]


# ============================================================
# LABEL NORMALISATION
# ============================================================

def normalise_gender_truth(
    value,
) -> str | None:

    if pd.isna(value):
        return None

    text = str(
        value
    ).strip().lower()

    mapping = {
        "male": "Male",
        "m": "Male",
        "0": "Male",

        "female": "Female",
        "f": "Female",
        "1": "Female",
    }

    return mapping.get(
        text
    )


def normalise_fairface_race_truth(
    value,
) -> str | None:

    if pd.isna(value):
        return None

    text = (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    text = " ".join(
        text.split()
    )

    mapping = {
        "white": "White",
        "black": "Black",

        "east asian": "Asian",
        "southeast asian": "Asian",
        "asian": "Asian",

        "indian": "Indian",

        "latino hispanic": "Latino_Hispanic",
        "latino": "Latino_Hispanic",
        "hispanic": "Latino_Hispanic",

        "middle eastern": "Middle Eastern",
    }

    return mapping.get(
        text
    )


def normalise_deepface_race_prediction(
    value: str,
) -> str:

    mapping = {
        "white": "White",
        "black": "Black",
        "asian": "Asian",
        "indian": "Indian",
        "latino hispanic": "Latino_Hispanic",
        "middle eastern": "Middle Eastern",
    }

    return mapping[
        value
    ]


def normalise_deepface_gender_prediction(
    value: str,
) -> str:

    mapping = {
        "Man": "Male",
        "Woman": "Female",
    }

    return mapping[
        value
    ]


# ============================================================
# INPUT PATHS
# ============================================================

def resolve_image_path(
    value,
) -> Path:

    raw = Path(
        str(
            value
        )
    )

    candidates = [
        raw,
        Path.cwd() / raw,
        Path.cwd() / "data" / raw,
    ]

    for candidate in candidates:

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve image path: {value}"
    )


# ============================================================
# LOAD FAIRFACE VALIDATION
# ============================================================

def load_fairface_validation(
    master_path: Path,
    limit: int | None,
) -> pd.DataFrame:

    master = pd.read_parquet(
        master_path
    )

    dataset_text = (
        master[
            "dataset"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    split_text = (
        master[
            "split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    fairface = master[
        dataset_text.eq(
            "fairface"
        )
    ].copy()

    validation_mask = (
        split_text.loc[
            fairface.index
        ]
        .str.contains(
            "val",
            na=False,
        )
    )

    frame = fairface[
        validation_mask
    ].copy()

    if frame.empty:

        print(
            "FairFace split values found:"
        )

        print(
            fairface[
                "split"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

        raise RuntimeError(
            "Could not identify the FairFace "
            "validation split."
        )

    frame[
        "race_true"
    ] = frame[
        "race"
    ].map(
        normalise_fairface_race_truth
    )

    frame[
        "gender_true"
    ] = frame[
        "gender"
    ].map(
        normalise_gender_truth
    )

    valid_labels = (
        frame[
            "race_true"
        ].notna()
        &
        frame[
            "gender_true"
        ].notna()
    )

    frame = frame[
        valid_labels
    ].copy()

    frame = (
        frame
        .sort_values(
            "image_id"
        )
        .reset_index(
            drop=True
        )
    )

    if limit is not None:

        if limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        if len(frame) > limit:

            frame = (
                frame.sample(
                    n=limit,
                    random_state=SEED,
                )
                .sort_values(
                    "image_id"
                )
                .reset_index(
                    drop=True
                )
            )

    return frame


# ============================================================
# DEEPFACE PREPROCESSING
# ============================================================

def load_deepface_image(
    path: Path,
) -> np.ndarray:
    """
    Reproduce DeepFace demographic preprocessing with the
    detector skipped.

    cv2.imread() gives BGR pixels. DeepFace demographic
    analysis ultimately sends a BGR 224x224 image in [0,1]
    to its Gender and Race models.
    """

    image = cv2.imread(
        str(
            path
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    prepared = preprocessing.resize_image(
        img=image,
        target_size=(
            224,
            224,
        ),
    )

    # resize_image returns (1, 224, 224, 3)
    return prepared[
        0
    ].astype(
        np.float32
    )


# ============================================================
# DEEPFACE MODELS
# ============================================================

def load_models():

    print(
        "Loading DeepFace Gender model..."
    )

    gender_model = DeepFace.build_model(
        "Gender",
        task="facial_attribute",
    )

    print(
        "Gender model loaded."
    )

    print(
        "Loading DeepFace Race model..."
    )

    race_model = DeepFace.build_model(
        "Race",
        task="facial_attribute",
    )

    print(
        "Race model loaded."
    )

    return (
        gender_model,
        race_model,
    )


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    frame: pd.DataFrame,
    gender_model,
    race_model,
    batch_size: int,
) -> tuple[pd.DataFrame, float]:

    records = []

    total = len(
        frame
    )

    processed = 0

    start = time.perf_counter()

    for batch_start in range(
        0,
        total,
        batch_size,
    ):

        batch_end = min(
            batch_start
            + batch_size,
            total,
        )

        batch_frame = frame.iloc[
            batch_start:batch_end
        ]

        images = []

        for _, row in batch_frame.iterrows():

            image_path = resolve_image_path(
                row[
                    "original_path"
                ]
            )

            images.append(
                load_deepface_image(
                    image_path
                )
            )

        batch = np.stack(
            images,
            axis=0,
        ).astype(
            np.float32
        )

        gender_probs = np.asarray(
            gender_model.predict(
                batch
            )
        )

        race_probs = np.asarray(
            race_model.predict(
                batch
            )
        )

        # Ensure two-dimensional output even for a 1-image batch.
        if gender_probs.ndim == 1:
            gender_probs = np.expand_dims(
                gender_probs,
                axis=0,
            )

        if race_probs.ndim == 1:
            race_probs = np.expand_dims(
                race_probs,
                axis=0,
            )

        for local_index, (
            _,
            row,
        ) in enumerate(
            batch_frame.iterrows()
        ):

            gender_vector = gender_probs[
                local_index
            ]

            race_vector = race_probs[
                local_index
            ]

            gender_index = int(
                np.argmax(
                    gender_vector
                )
            )

            race_index = int(
                np.argmax(
                    race_vector
                )
            )

            deepface_gender_raw = [
                "Woman",
                "Man",
            ][
                gender_index
            ]

            deepface_race_raw = (
                DEEPFACE_RACE_LABELS_RAW[
                    race_index
                ]
            )

            record = {
                "image_id": (
                    row[
                        "image_id"
                    ]
                ),

                "input_source": (
                    "fairface_prepared_image_detector_skipped"
                ),

                "gender_pred_raw": (
                    deepface_gender_raw
                ),

                "gender_pred": (
                    normalise_deepface_gender_prediction(
                        deepface_gender_raw
                    )
                ),

                "model_gender_confidence": float(
                    gender_vector[
                        gender_index
                    ]
                ),

                "race_pred_raw": (
                    deepface_race_raw
                ),

                "race_pred": (
                    normalise_deepface_race_prediction(
                        deepface_race_raw
                    )
                ),

                "model_race_confidence": float(
                    race_vector[
                        race_index
                    ]
                    /
                    max(
                        float(
                            race_vector.sum()
                        ),
                        1e-12,
                    )
                ),
            }

            # Gender probabilities
            record[
                "gender_prob_female"
            ] = float(
                gender_vector[
                    0
                ]
            )

            record[
                "gender_prob_male"
            ] = float(
                gender_vector[
                    1
                ]
            )

            # Race probabilities
            race_sum = max(
                float(
                    race_vector.sum()
                ),
                1e-12,
            )

            for race_position, race_name in enumerate(
                DEEPFACE_RACE_LABELS_RAW
            ):

                column_name = (
                    "race_prob_"
                    + race_name
                    .replace(
                        " ",
                        "_",
                    )
                )

                record[
                    column_name
                ] = float(
                    race_vector[
                        race_position
                    ]
                    /
                    race_sum
                )

            records.append(
                record
            )

        processed += len(
            batch_frame
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        rate = (
            processed
            / elapsed
            if elapsed > 0
            else 0.0
        )

        remaining = (
            total
            - processed
        )

        eta_seconds = (
            remaining
            / rate
            if rate > 0
            else 0.0
        )

        if (
            processed == total
            or processed % 500
            < len(
                batch_frame
            )
        ):

            print(
                f"Processed "
                f"{processed:,}/"
                f"{total:,} | "
                f"{rate:.2f} images/s | "
                f"ETA "
                f"{eta_seconds / 60:.1f} min"
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    return (
        pd.DataFrame(
            records
        ),
        elapsed,
    )


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    task: str,
    y_true,
    y_pred,
    labels,
) -> dict:

    return {
        "task": task,

        "n": int(
            len(
                y_true
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),

        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),

        "precision_weighted": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),

        "recall_weighted": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),

        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def build_overall_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            classification_metrics(
                "race_harmonised_6class",
                predictions[
                    "race_true"
                ],
                predictions[
                    "race_pred"
                ],
                COMMON_RACE_LABELS,
            ),

            classification_metrics(
                "gender",
                predictions[
                    "gender_true"
                ],
                predictions[
                    "gender_pred"
                ],
                GENDER_LABELS,
            ),
        ]
    )


# ============================================================
# INTERSECTIONAL PERFORMANCE
# ============================================================

def build_intersectional_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    frame = predictions.copy()

    frame[
        "race_correct"
    ] = (
        frame[
            "race_true"
        ]
        ==
        frame[
            "race_pred"
        ]
    )

    frame[
        "gender_correct"
    ] = (
        frame[
            "gender_true"
        ]
        ==
        frame[
            "gender_pred"
        ]
    )

    frame[
        "both_correct"
    ] = (
        frame[
            "race_correct"
        ]
        &
        frame[
            "gender_correct"
        ]
    )

    grouped = (
        frame.groupby(
            [
                "race_true",
                "gender_true",
            ],
            observed=True,
        )
        .agg(
            n=(
                "image_id",
                "count",
            ),

            race_accuracy=(
                "race_correct",
                "mean",
            ),

            gender_accuracy=(
                "gender_correct",
                "mean",
            ),

            both_correct_rate=(
                "both_correct",
                "mean",
            ),

            mean_race_confidence=(
                "model_race_confidence",
                "mean",
            ),

            mean_gender_confidence=(
                "model_gender_confidence",
                "mean",
            ),
        )
        .reset_index()
    )

    grouped[
        "reporting_status"
    ] = np.select(
        [
            grouped[
                "n"
            ] < 30,

            grouped[
                "n"
            ] < 100,
        ],
        [
            "suppressed",
            "low_confidence",
        ],
        default="reportable",
    )

    return grouped


# ============================================================
# GENDER FAIRNESS ACROSS RACE
# ============================================================

def build_gender_fairness(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    rows = []

    for race in COMMON_RACE_LABELS:

        frame = predictions[
            predictions[
                "race_true"
            ].eq(
                race
            )
        ].copy()

        actual_female = (
            frame[
                "gender_true"
            ].eq(
                "Female"
            )
        )

        actual_male = (
            frame[
                "gender_true"
            ].eq(
                "Male"
            )
        )

        predicted_female = (
            frame[
                "gender_pred"
            ].eq(
                "Female"
            )
        )

        tp = int(
            (
                actual_female
                &
                predicted_female
            ).sum()
        )

        fn = int(
            (
                actual_female
                &
                ~predicted_female
            ).sum()
        )

        fp = int(
            (
                actual_male
                &
                predicted_female
            ).sum()
        )

        tn = int(
            (
                actual_male
                &
                ~predicted_female
            ).sum()
        )

        female_n = (
            tp + fn
        )

        male_n = (
            fp + tn
        )

        tpr = (
            tp / female_n
            if female_n > 0
            else np.nan
        )

        fpr = (
            fp / male_n
            if male_n > 0
            else np.nan
        )

        rows.append(
            {
                "race": race,

                "n": int(
                    len(
                        frame
                    )
                ),

                "n_female": (
                    female_n
                ),

                "n_male": (
                    male_n
                ),

                "true_positive": (
                    tp
                ),

                "false_negative": (
                    fn
                ),

                "false_positive": (
                    fp
                ),

                "true_negative": (
                    tn
                ),

                "female_tpr": (
                    tpr
                ),

                "female_fpr": (
                    fpr
                ),
            }
        )

    rates = pd.DataFrame(
        rows
    )

    valid_tpr = (
        rates[
            "female_tpr"
        ]
        .dropna()
    )

    valid_fpr = (
        rates[
            "female_fpr"
        ]
        .dropna()
    )

    tpr_difference = float(
        valid_tpr.max()
        -
        valid_tpr.min()
    )

    fpr_difference = float(
        valid_fpr.max()
        -
        valid_fpr.min()
    )

    summary = {
        "task": (
            "gender classification"
        ),

        "positive_class": (
            "Female"
        ),

        "sensitive_feature": (
            "FairFace harmonised race"
        ),

        "tpr_difference": (
            tpr_difference
        ),

        "fpr_difference": (
            fpr_difference
        ),

        "equalized_odds_difference": float(
            max(
                tpr_difference,
                fpr_difference,
            )
        ),
    }

    return (
        rates,
        summary,
    )


# ============================================================
# RACE FAIRNESS ACROSS GENDER
# ============================================================

def build_race_fairness_by_gender(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for target_race in COMMON_RACE_LABELS:

        group_rates = {}

        for gender in GENDER_LABELS:

            frame = predictions[
                predictions[
                    "gender_true"
                ].eq(
                    gender
                )
            ]

            actual_positive = (
                frame[
                    "race_true"
                ].eq(
                    target_race
                )
            )

            predicted_positive = (
                frame[
                    "race_pred"
                ].eq(
                    target_race
                )
            )

            tp = int(
                (
                    actual_positive
                    &
                    predicted_positive
                ).sum()
            )

            fn = int(
                (
                    actual_positive
                    &
                    ~predicted_positive
                ).sum()
            )

            fp = int(
                (
                    ~actual_positive
                    &
                    predicted_positive
                ).sum()
            )

            tn = int(
                (
                    ~actual_positive
                    &
                    ~predicted_positive
                ).sum()
            )

            positive_n = (
                tp + fn
            )

            negative_n = (
                fp + tn
            )

            tpr = (
                tp / positive_n
                if positive_n > 0
                else np.nan
            )

            fpr = (
                fp / negative_n
                if negative_n > 0
                else np.nan
            )

            group_rates[
                gender
            ] = {
                "positive_n": (
                    positive_n
                ),

                "negative_n": (
                    negative_n
                ),

                "tpr": (
                    tpr
                ),

                "fpr": (
                    fpr
                ),
            }

        male = group_rates[
            "Male"
        ]

        female = group_rates[
            "Female"
        ]

        tpr_difference = abs(
            male[
                "tpr"
            ]
            -
            female[
                "tpr"
            ]
        )

        fpr_difference = abs(
            male[
                "fpr"
            ]
            -
            female[
                "fpr"
            ]
        )

        rows.append(
            {
                "target_race": (
                    target_race
                ),

                "male_positive_n": (
                    male[
                        "positive_n"
                    ]
                ),

                "female_positive_n": (
                    female[
                        "positive_n"
                    ]
                ),

                "male_tpr": (
                    male[
                        "tpr"
                    ]
                ),

                "female_tpr": (
                    female[
                        "tpr"
                    ]
                ),

                "tpr_difference": (
                    tpr_difference
                ),

                "male_fpr": (
                    male[
                        "fpr"
                    ]
                ),

                "female_fpr": (
                    female[
                        "fpr"
                    ]
                ),

                "fpr_difference": (
                    fpr_difference
                ),

                "equalized_odds_difference": float(
                    max(
                        tpr_difference,
                        fpr_difference,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# KEY INTERSECTIONAL COMPARISONS
# ============================================================

def build_key_comparisons(
    intersections: pd.DataFrame,
) -> pd.DataFrame:

    lookup = {}

    for _, row in intersections.iterrows():

        lookup[
            (
                row[
                    "race_true"
                ],
                row[
                    "gender_true"
                ],
            )
        ] = row

    pairs = [
        (
            "White Male",
            (
                "White",
                "Male",
            ),
            "Black Female",
            (
                "Black",
                "Female",
            ),
        ),

        (
            "Black Male",
            (
                "Black",
                "Male",
            ),
            "White Male",
            (
                "White",
                "Male",
            ),
        ),

        (
            "White Female",
            (
                "White",
                "Female",
            ),
            "Black Female",
            (
                "Black",
                "Female",
            ),
        ),
    ]

    rows = []

    for (
        label_a,
        key_a,
        label_b,
        key_b,
    ) in pairs:

        if (
            key_a not in lookup
            or key_b not in lookup
        ):
            continue

        a = lookup[
            key_a
        ]

        b = lookup[
            key_b
        ]

        rows.append(
            {
                "group_a": (
                    label_a
                ),

                "group_b": (
                    label_b
                ),

                "n_a": int(
                    a[
                        "n"
                    ]
                ),

                "n_b": int(
                    b[
                        "n"
                    ]
                ),

                "race_accuracy_a": float(
                    a[
                        "race_accuracy"
                    ]
                ),

                "race_accuracy_b": float(
                    b[
                        "race_accuracy"
                    ]
                ),

                "race_accuracy_gap_a_minus_b": float(
                    a[
                        "race_accuracy"
                    ]
                    -
                    b[
                        "race_accuracy"
                    ]
                ),

                "gender_accuracy_a": float(
                    a[
                        "gender_accuracy"
                    ]
                ),

                "gender_accuracy_b": float(
                    b[
                        "gender_accuracy"
                    ]
                ),

                "gender_accuracy_gap_a_minus_b": float(
                    a[
                        "gender_accuracy"
                    ]
                    -
                    b[
                        "gender_accuracy"
                    ]
                ),

                "both_correct_a": float(
                    a[
                        "both_correct_rate"
                    ]
                ),

                "both_correct_b": float(
                    b[
                        "both_correct_rate"
                    ]
                ),

                "both_correct_gap_a_minus_b": float(
                    a[
                        "both_correct_rate"
                    ]
                    -
                    b[
                        "both_correct_rate"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

def save_confusion_matrix(
    predictions: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    labels: list[str],
    title: str,
    csv_path: Path,
    png_path: Path,
) -> None:

    matrix = confusion_matrix(
        predictions[
            truth_column
        ],
        predictions[
            prediction_column
        ],
        labels=labels,
    )

    table = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    table.index.name = (
        "actual"
    )

    table.columns.name = (
        "predicted"
    )

    table.to_csv(
        csv_path
    )

    row_totals = (
        matrix.sum(
            axis=1,
            keepdims=True,
        )
    )

    normalised = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=row_totals != 0,
    )

    normalised_table = pd.DataFrame(
        normalised,
        index=labels,
        columns=labels,
    )

    normalised_table.index.name = (
        "actual"
    )

    normalised_table.columns.name = (
        "predicted"
    )

    normalised_table.to_csv(
        csv_path.with_name(
            csv_path.stem
            + "_normalised.csv"
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            max(
                6,
                len(
                    labels
                )
                * 1.25,
            ),
            max(
                5,
                len(
                    labels
                )
                * 1.1,
            ),
        )
    )

    display = ax.imshow(
        matrix
    )

    ax.set_xticks(
        range(
            len(
                labels
            )
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(
        range(
            len(
                labels
            )
        )
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "Actual label"
    )

    ax.set_title(
        title
    )

    maximum = (
        matrix.max()
        if matrix.size
        else 0
    )

    midpoint = (
        maximum / 2
        if maximum
        else 0
    )

    for row in range(
        matrix.shape[
            0
        ]
    ):

        for column in range(
            matrix.shape[
                1
            ]
        ):

            value = int(
                matrix[
                    row,
                    column,
                ]
            )

            ax.text(
                column,
                row,
                str(
                    value
                ),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > midpoint
                    else "black"
                ),
                fontsize=8,
            )

    fig.colorbar(
        display,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        png_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# CONFIDENCE SUMMARY
# ============================================================

def build_confidence_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for task, column in [
        (
            "race",
            "model_race_confidence",
        ),
        (
            "gender",
            "model_gender_confidence",
        ),
    ]:

        values = predictions[
            column
        ]

        rows.append(
            {
                "task": task,

                "n": int(
                    len(
                        values
                    )
                ),

                "mean_confidence": float(
                    values.mean()
                ),

                "median_confidence": float(
                    values.median()
                ),

                "p05": float(
                    values.quantile(
                        0.05
                    )
                ),

                "p25": float(
                    values.quantile(
                        0.25
                    )
                ),

                "p75": float(
                    values.quantile(
                        0.75
                    )
                ),

                "p95": float(
                    values.quantile(
                        0.95
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "DeepFace demographic classification audit "
            "on the FairFace validation split."
        )
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if not args.master.exists():

        raise FileNotFoundError(
            f"Master table not found: "
            f"{args.master}"
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be positive."
        )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.random.seed(
        SEED
    )

    tf.random.set_seed(
        SEED
    )

    print()
    print(
        "DeepFace → FairFace Validation Audit"
    )

    print(
        "===================================="
    )

    print(
        f"TensorFlow: "
        f"{tf.__version__}"
    )

    print(
        f"Batch size: "
        f"{args.batch_size}"
    )

    devices = (
        tf.config.list_physical_devices()
    )

    print(
        "TensorFlow devices:"
    )

    for device in devices:

        print(
            f"  {device}"
        )

    frame = load_fairface_validation(
        args.master,
        args.limit,
    )

    print()
    print(
        f"FairFace validation images selected: "
        f"{len(frame):,}"
    )

    print()
    print(
        "Harmonised ground-truth race labels:"
    )

    print(
        frame[
            "race_true"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Ground-truth gender labels:"
    )

    print(
        frame[
            "gender_true"
        ]
        .value_counts()
        .to_string()
    )

    print()

    (
        gender_model,
        race_model,
    ) = load_models()

    print()
    print(
        "Running batched inference..."
    )

    (
        inference,
        elapsed,
    ) = run_inference(
        frame,
        gender_model,
        race_model,
        args.batch_size,
    )

    predictions = frame.merge(
        inference,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if predictions[
        "race_pred"
    ].isna().any():

        missing = int(
            predictions[
                "race_pred"
            ]
            .isna()
            .sum()
        )

        raise RuntimeError(
            f"{missing} images did not "
            "receive predictions."
        )

    predictions.to_csv(
        args.output
        / "predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    overall = build_overall_metrics(
        predictions
    )

    overall.to_csv(
        args.output
        / "overall_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORTS
    # --------------------------------------------------------

    race_report = classification_report(
        predictions[
            "race_true"
        ],
        predictions[
            "race_pred"
        ],
        labels=COMMON_RACE_LABELS,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        race_report
    ).T.to_csv(
        args.output
        / "race_classification_report.csv"
    )

    gender_report = classification_report(
        predictions[
            "gender_true"
        ],
        predictions[
            "gender_pred"
        ],
        labels=GENDER_LABELS,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        gender_report
    ).T.to_csv(
        args.output
        / "gender_classification_report.csv"
    )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    save_confusion_matrix(
        predictions,
        "race_true",
        "race_pred",
        COMMON_RACE_LABELS,
        (
            "DeepFace on FairFace Validation: "
            "Race Confusion Matrix"
        ),
        args.output
        / "race_confusion_matrix.csv",
        args.output
        / "race_confusion_matrix.png",
    )

    save_confusion_matrix(
        predictions,
        "gender_true",
        "gender_pred",
        GENDER_LABELS,
        (
            "DeepFace on FairFace Validation: "
            "Gender Confusion Matrix"
        ),
        args.output
        / "gender_confusion_matrix.csv",
        args.output
        / "gender_confusion_matrix.png",
    )

    # --------------------------------------------------------
    # INTERSECTIONAL
    # --------------------------------------------------------

    intersectional = (
        build_intersectional_metrics(
            predictions
        )
    )

    intersectional.to_csv(
        args.output
        / "intersectional_metrics.csv",
        index=False,
    )

    key_comparisons = (
        build_key_comparisons(
            intersectional
        )
    )

    key_comparisons.to_csv(
        args.output
        / "key_intersectional_comparisons.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FAIRNESS
    # --------------------------------------------------------

    (
        gender_fairness,
        gender_eo,
    ) = build_gender_fairness(
        predictions
    )

    gender_fairness.to_csv(
        args.output
        / "gender_fairness_by_race.csv",
        index=False,
    )

    with open(
        args.output
        / "gender_equalized_odds_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            gender_eo,
            handle,
            indent=2,
        )

    race_fairness = (
        build_race_fairness_by_gender(
            predictions
        )
    )

    race_fairness.to_csv(
        args.output
        / "race_fairness_by_gender.csv",
        index=False,
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = (
        build_confidence_summary(
            predictions
        )
    )

    confidence.to_csv(
        args.output
        / "confidence_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # RUNTIME / PROVENANCE
    # --------------------------------------------------------

    throughput = (
        len(
            predictions
        )
        /
        elapsed
        if elapsed > 0
        else np.nan
    )

    runtime = {
        "library": (
            "DeepFace"
        ),

        "dataset": (
            "FairFace validation"
        ),

        "tasks": [
            "race",
            "gender",
        ],

        "race_taxonomy": (
            "6-class common taxonomy; "
            "FairFace East Asian and Southeast Asian "
            "merged to Asian"
        ),

        "input_preprocessing": (
            "FairFace prepared images; "
            "DeepFace detector skipped; "
            "BGR resize/pad to 224x224; "
            "pixel scale [0,1]"
        ),

        "n_images": int(
            len(
                predictions
            )
        ),

        "elapsed_seconds": float(
            elapsed
        ),

        "images_per_second": float(
            throughput
        ),

        "batch_size": int(
            args.batch_size
        ),

        "tensorflow_version": (
            tf.__version__
        ),

        "seed": (
            SEED
        ),
    }

    with open(
        args.output
        / "runtime.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            runtime,
            handle,
            indent=2,
        )

    # --------------------------------------------------------
    # CONSOLE OUTPUT
    # --------------------------------------------------------

    print()
    print(
        "Experiment complete"
    )

    print(
        "===================================="
    )

    print(
        f"Images:      "
        f"{len(predictions):,}"
    )

    print(
        f"Elapsed:     "
        f"{elapsed:.1f} s"
    )

    print(
        f"Throughput:  "
        f"{throughput:.2f} images/s"
    )

    print()
    print(
        "OVERALL PERFORMANCE"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print()
    print(
        "INTERSECTIONAL PERFORMANCE"
    )

    print(
        intersectional.to_string(
            index=False
        )
    )

    print()
    print(
        "KEY INTERSECTIONAL COMPARISONS"
    )

    print(
        key_comparisons.to_string(
            index=False
        )
    )

    print()
    print(
        "GENDER FAIRNESS BY RACE"
    )

    print(
        gender_fairness.to_string(
            index=False
        )
    )

    print()
    print(
        "GENDER EQUALIZED ODDS"
    )

    print(
        json.dumps(
            gender_eo,
            indent=2,
        )
    )

    print()
    print(
        "RACE FAIRNESS BY GENDER"
    )

    print(
        race_fairness.to_string(
            index=False
        )
    )

    print()
    print(
        "CONFIDENCE SUMMARY"
    )

    print(
        confidence.to_string(
            index=False
        )
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        args.output.resolve()
    )


if __name__ == "__main__":
    main()
    