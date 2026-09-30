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
import yaml

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

from src import detection
from src import image_integrity
from src import paths


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 26041

DEFAULT_MASTER = Path(
    "metadata/master_annotations.parquet"
)

DEFAULT_PREPROCESSING = Path(
    "configs/preprocessing.yaml"
)

DEFAULT_OUTPUT = Path(
    "results/model_audit/deepface/utkface_aligned"
)


DEEPFACE_RACE_LABELS_RAW = [
    "asian",
    "indian",
    "black",
    "white",
    "middle eastern",
    "latino hispanic",
]

UTK_NATIVE_RACES = [
    "White",
    "Black",
    "Asian",
    "Indian",
    "Others",
]

HARMONISED_RACES = [
    "White",
    "Black",
    "Asian",
    "Indian",
]

PREDICTED_HARMONISED_CLASSES = [
    "White",
    "Black",
    "Asian",
    "Indian",
    "Out_of_scope",
]

GENDER_LABELS = [
    "Male",
    "Female",
]


# ============================================================
# LABEL NORMALISATION
# ============================================================

def normalise_gender(
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


def normalise_utk_race(
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
        "0": "White",

        "black": "Black",
        "1": "Black",

        "asian": "Asian",
        "2": "Asian",

        "indian": "Indian",
        "3": "Indian",

        "others": "Others",
        "other": "Others",
        "4": "Others",
    }

    return mapping.get(
        text
    )


def normalise_deepface_gender(
    value: str,
) -> str:

    mapping = {
        "Man": "Male",
        "Woman": "Female",
    }

    return mapping[
        value
    ]


def normalise_deepface_race(
    value: str,
) -> str:

    mapping = {
        "white": "White",
        "black": "Black",
        "asian": "Asian",
        "indian": "Indian",

        "middle eastern": "Out_of_scope",
        "latino hispanic": "Out_of_scope",
    }

    return mapping[
        value
    ]


# ============================================================
# PROJECT ALIGNMENT CONFIGURATION
# ============================================================

def load_preprocessing_config(
    path: Path,
) -> dict:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:

        config = yaml.safe_load(
            handle
        )

    if not isinstance(
        config,
        dict,
    ):
        raise RuntimeError(
            "Could not load preprocessing configuration."
        )

    return config


def detection_compatible_config(
    preprocessing_cfg: dict,
) -> dict:

    config = dict(
        preprocessing_cfg
    )

    alignment = dict(
        preprocessing_cfg.get(
            "alignment",
            {},
        )
    )

    if "output_size" not in alignment:

        if "output_size_px" not in alignment:
            raise KeyError(
                "Alignment configuration contains "
                "no output size."
            )

        alignment[
            "output_size"
        ] = alignment[
            "output_size_px"
        ]

    if (
        "canonical_landmark_fractions"
        not in alignment
    ):

        if (
            "canonical_landmark_positions"
            not in alignment
        ):
            raise KeyError(
                "Alignment configuration contains "
                "no canonical landmarks."
            )

        alignment[
            "canonical_landmark_fractions"
        ] = alignment[
            "canonical_landmark_positions"
        ]

    config[
        "alignment"
    ] = alignment

    return config


# ============================================================
# LANDMARK NORMALISATION
# ============================================================

def normalise_five_point_landmarks(
    value,
) -> list[list[float]]:

    if value is None:
        raise RuntimeError(
            "Landmarks unavailable."
        )

    points = np.asarray(
        value,
        dtype=object,
    ).reshape(-1)

    if len(points) != 5:
        raise RuntimeError(
            f"Expected 5 landmarks; found {len(points)}."
        )

    normalised = []

    for index, point in enumerate(
        points
    ):

        numeric = np.asarray(
            point,
            dtype=np.float32,
        ).reshape(-1)

        if numeric.size != 2:
            raise RuntimeError(
                f"Landmark {index} does not "
                "contain two coordinates."
            )

        x = float(
            numeric[0]
        )

        y = float(
            numeric[1]
        )

        if (
            not np.isfinite(x)
            or not np.isfinite(y)
        ):
            raise RuntimeError(
                f"Landmark {index} contains "
                "non-finite coordinates."
            )

        normalised.append(
            [
                x,
                y,
            ]
        )

    return normalised


# ============================================================
# LOAD UTKFACE
# ============================================================

def load_utkface(
    master_path: Path,
    limit: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

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

    frame = master[
        dataset_text.eq(
            "utkface"
        )
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No UTKFace rows found."
        )

    frame[
        "race_true_native"
    ] = frame[
        "race"
    ].map(
        normalise_utk_race
    )

    frame[
        "gender_true"
    ] = frame[
        "gender"
    ].map(
        normalise_gender
    )

    frame[
        "race_comparable"
    ] = frame[
        "race_true_native"
    ].isin(
        HARMONISED_RACES
    )

    eligible = (
        frame[
            "alignment_valid"
        ].eq(
            True
        )
        &
        frame[
            "landmarks"
        ].notna()
    )

    excluded = frame[
        ~eligible
    ].copy()

    excluded[
        "model_audit_exclusion_reason"
    ] = np.where(
        excluded[
            "landmarks"
        ].isna(),
        "landmarks_unavailable",
        "alignment_not_valid",
    )

    frame = frame[
        eligible
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

    return (
        frame,
        excluded,
    )


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prepare_image(
    row: pd.Series,
    preprocessing_cfg: dict,
) -> np.ndarray:

    rgb, _ = (
        image_integrity.load_rgb_for_analysis(
            paths.data_abs(
                str(
                    row[
                        "original_path"
                    ]
                )
            )
        )
    )

    landmarks = (
        normalise_five_point_landmarks(
            row[
                "landmarks"
            ]
        )
    )

    (
        aligned_rgb,
        transform_matrix,
        valid_mask,
    ) = detection.align_face(
        rgb,
        landmarks,
        preprocessing_cfg,
    )

    if aligned_rgb is None:
        raise RuntimeError(
            f"Alignment failed for "
            f"{row['image_id']}."
        )

    if transform_matrix is None:
        raise RuntimeError(
            f"Alignment transform missing for "
            f"{row['image_id']}."
        )

    if valid_mask is None:
        raise RuntimeError(
            f"Alignment mask missing for "
            f"{row['image_id']}."
        )

    # DeepFace demographic preprocessing operates
    # on BGR images.
    aligned_bgr = cv2.cvtColor(
        aligned_rgb,
        cv2.COLOR_RGB2BGR,
    )

    prepared = preprocessing.resize_image(
        img=aligned_bgr,
        target_size=(
            224,
            224,
        ),
    )

    return prepared[
        0
    ].astype(
        np.float32
    )


# ============================================================
# LOAD DEEPFACE MODELS
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
    preprocessing_cfg: dict,
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

            images.append(
                prepare_image(
                    row,
                    preprocessing_cfg,
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

            gender_raw = [
                "Woman",
                "Man",
            ][
                gender_index
            ]

            race_raw = (
                DEEPFACE_RACE_LABELS_RAW[
                    race_index
                ]
            )

            race_sum = max(
                float(
                    race_vector.sum()
                ),
                1e-12,
            )

            record = {
                "image_id": (
                    row[
                        "image_id"
                    ]
                ),

                "input_source": (
                    "reconstructed_project_alignment_"
                    "deepface_detector_skipped"
                ),

                "gender_pred_raw": (
                    gender_raw
                ),

                "gender_pred": (
                    normalise_deepface_gender(
                        gender_raw
                    )
                ),

                "model_gender_confidence": float(
                    gender_vector[
                        gender_index
                    ]
                ),

                "race_pred_raw": (
                    race_raw
                ),

                "race_pred_harmonised": (
                    normalise_deepface_race(
                        race_raw
                    )
                ),

                "model_race_confidence": float(
                    race_vector[
                        race_index
                    ]
                    /
                    race_sum
                ),

                "gender_prob_female": float(
                    gender_vector[
                        0
                    ]
                ),

                "gender_prob_male": float(
                    gender_vector[
                        1
                    ]
                ),
            }

            for race_position, race_name in enumerate(
                DEEPFACE_RACE_LABELS_RAW
            ):

                column_name = (
                    "race_prob_"
                    + race_name.replace(
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
# METRICS
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

    rows = []

    gender = predictions[
        predictions[
            "gender_true"
        ].notna()
    ].copy()

    rows.append(
        classification_metrics(
            "gender_all_utkface",
            gender[
                "gender_true"
            ],
            gender[
                "gender_pred"
            ],
            GENDER_LABELS,
        )
    )

    race = predictions[
        predictions[
            "race_comparable"
        ].eq(
            True
        )
        &
        predictions[
            "race_true_native"
        ].notna()
    ].copy()

    rows.append(
        classification_metrics(
            "race_harmonised_4class",
            race[
                "race_true_native"
            ],
            race[
                "race_pred_harmonised"
            ],
            HARMONISED_RACES,
        )
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# INTERSECTIONAL PERFORMANCE
# ============================================================

def build_intersectional_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    frame = predictions[
        predictions[
            "race_true_native"
        ].notna()
        &
        predictions[
            "gender_true"
        ].notna()
    ].copy()

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
        "race_correct"
    ] = np.where(
        frame[
            "race_comparable"
        ].eq(
            True
        ),
        (
            frame[
                "race_true_native"
            ]
            ==
            frame[
                "race_pred_harmonised"
            ]
        ).astype(
            float
        ),
        np.nan,
    )

    grouped = (
        frame.groupby(
            [
                "race_true_native",
                "gender_true",
            ],
            observed=True,
        )
        .agg(
            n=(
                "image_id",
                "count",
            ),

            gender_accuracy=(
                "gender_correct",
                "mean",
            ),

            race_accuracy=(
                "race_correct",
                "mean",
            ),

            mean_model_race_confidence=(
                "model_race_confidence",
                "mean",
            ),

            mean_model_gender_confidence=(
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
# GENDER FAIRNESS ACROSS UTK RACE
# ============================================================

def build_gender_fairness(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    rows = []

    for race in UTK_NATIVE_RACES:

        frame = predictions[
            predictions[
                "race_true_native"
            ].eq(
                race
            )
            &
            predictions[
                "gender_true"
            ].notna()
        ].copy()

        if frame.empty:
            continue

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

                "n_female": int(
                    female_n
                ),

                "n_male": int(
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
        ].dropna()
    )

    valid_fpr = (
        rates[
            "female_fpr"
        ].dropna()
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
            "UTKFace native race"
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

    comparable = predictions[
        predictions[
            "race_comparable"
        ].eq(
            True
        )
        &
        predictions[
            "gender_true"
        ].notna()
    ].copy()

    rows = []

    for target_race in HARMONISED_RACES:

        group_rates = {}

        for gender in GENDER_LABELS:

            frame = comparable[
                comparable[
                    "gender_true"
                ].eq(
                    gender
                )
            ]

            actual_positive = (
                frame[
                    "race_true_native"
                ].eq(
                    target_race
                )
            )

            predicted_positive = (
                frame[
                    "race_pred_harmonised"
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
# TAXONOMY SUMMARY
# ============================================================

def build_taxonomy_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    comparable = predictions[
        predictions[
            "race_comparable"
        ].eq(
            True
        )
    ].copy()

    out_of_scope = comparable[
        "race_pred_harmonised"
    ].eq(
        "Out_of_scope"
    )

    others = predictions[
        predictions[
            "race_true_native"
        ].eq(
            "Others"
        )
    ]

    return pd.DataFrame(
        [
            {
                "measure": (
                    "All aligned UTKFace images"
                ),

                "n": int(
                    len(
                        predictions
                    )
                ),

                "proportion": (
                    1.0
                ),
            },

            {
                "measure": (
                    "Race-comparable UTKFace images"
                ),

                "n": int(
                    len(
                        comparable
                    )
                ),

                "proportion": float(
                    len(
                        comparable
                    )
                    /
                    len(
                        predictions
                    )
                ),
            },

            {
                "measure": (
                    "Comparable images predicted outside "
                    "the harmonised taxonomy"
                ),

                "n": int(
                    out_of_scope.sum()
                ),

                "proportion": float(
                    out_of_scope.mean()
                ),
            },

            {
                "measure": (
                    "UTKFace native Others images"
                ),

                "n": int(
                    len(
                        others
                    )
                ),

                "proportion": float(
                    len(
                        others
                    )
                    /
                    len(
                        predictions
                    )
                ),
            },
        ]
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

def save_gender_confusion(
    predictions: pd.DataFrame,
    output: Path,
) -> None:

    frame = predictions[
        predictions[
            "gender_true"
        ].notna()
    ]

    matrix = confusion_matrix(
        frame[
            "gender_true"
        ],
        frame[
            "gender_pred"
        ],
        labels=GENDER_LABELS,
    )

    table = pd.DataFrame(
        matrix,
        index=GENDER_LABELS,
        columns=GENDER_LABELS,
    )

    table.index.name = (
        "actual"
    )

    table.columns.name = (
        "predicted"
    )

    table.to_csv(
        output
        / "gender_confusion_matrix.csv"
    )

    fig, ax = plt.subplots(
        figsize=(
            6,
            5,
        )
    )

    display = ax.imshow(
        matrix
    )

    ax.set_xticks(
        range(
            len(
                GENDER_LABELS
            )
        )
    )

    ax.set_xticklabels(
        GENDER_LABELS
    )

    ax.set_yticks(
        range(
            len(
                GENDER_LABELS
            )
        )
    )

    ax.set_yticklabels(
        GENDER_LABELS
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "Actual label"
    )

    ax.set_title(
        "DeepFace on UTKFace: Gender Confusion Matrix"
    )

    maximum = matrix.max()

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
            )

    fig.colorbar(
        display,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output
        / "gender_confusion_matrix.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_race_confusion(
    predictions: pd.DataFrame,
    output: Path,
) -> None:

    frame = predictions[
        predictions[
            "race_comparable"
        ].eq(
            True
        )
    ]

    table = pd.crosstab(
        frame[
            "race_true_native"
        ],
        frame[
            "race_pred_harmonised"
        ],
    )

    table = table.reindex(
        index=HARMONISED_RACES,
        columns=PREDICTED_HARMONISED_CLASSES,
        fill_value=0,
    )

    table.index.name = (
        "actual"
    )

    table.columns.name = (
        "predicted"
    )

    table.to_csv(
        output
        / "race_confusion_matrix_harmonised.csv"
    )

    matrix = table.values

    fig, ax = plt.subplots(
        figsize=(
            9,
            7,
        )
    )

    display = ax.imshow(
        matrix
    )

    ax.set_xticks(
        range(
            len(
                table.columns
            )
        )
    )

    ax.set_xticklabels(
        table.columns,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(
        range(
            len(
                table.index
            )
        )
    )

    ax.set_yticklabels(
        table.index
    )

    ax.set_xlabel(
        "Predicted category"
    )

    ax.set_ylabel(
        "Actual UTKFace category"
    )

    ax.set_title(
        "DeepFace on UTKFace: "
        "Race Confusion Matrix"
    )

    maximum = matrix.max()

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
            )

    fig.colorbar(
        display,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output
        / "race_confusion_matrix_harmonised.png",
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

    parser = argparse.ArgumentParser(
        description=(
            "DeepFace demographic audit on "
            "deterministically aligned UTKFace images."
        )
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
    )

    parser.add_argument(
        "--preprocessing",
        type=Path,
        default=DEFAULT_PREPROCESSING,
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

    preprocessing_cfg = (
        detection_compatible_config(
            load_preprocessing_config(
                args.preprocessing
            )
        )
    )

    (
        frame,
        excluded,
    ) = load_utkface(
        args.master,
        args.limit,
    )

    excluded.to_csv(
        args.output
        / "excluded_from_aligned_model_audit.csv",
        index=False,
    )

    print()
    print(
        "DeepFace → UTKFace Aligned Audit"
    )

    print(
        "================================"
    )

    print(
        f"TensorFlow: {tf.__version__}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print()
    print(
        f"Aligned/landmark-eligible images: "
        f"{len(frame):,}"
    )

    print(
        f"Excluded before inference:        "
        f"{len(excluded):,}"
    )

    print()
    print(
        "Native UTKFace race labels:"
    )

    print(
        frame[
            "race_true_native"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        "UTKFace gender labels:"
    )

    print(
        frame[
            "gender_true"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()

    (
        gender_model,
        race_model,
    ) = load_models()

    print()
    print(
        "Reconstructing alignment and "
        "running DeepFace inference..."
    )

    (
        inference,
        elapsed,
    ) = run_inference(
        frame,
        preprocessing_cfg,
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

    predictions.to_csv(
        args.output
        / "predictions.csv",
        index=False,
    )

    overall = build_overall_metrics(
        predictions
    )

    overall.to_csv(
        args.output
        / "overall_metrics.csv",
        index=False,
    )

    comparable = predictions[
        predictions[
            "race_comparable"
        ].eq(
            True
        )
    ]

    race_report = classification_report(
        comparable[
            "race_true_native"
        ],
        comparable[
            "race_pred_harmonised"
        ],
        labels=HARMONISED_RACES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        race_report
    ).T.to_csv(
        args.output
        / "race_classification_report.csv"
    )

    gender_valid = predictions[
        predictions[
            "gender_true"
        ].notna()
    ]

    gender_report = classification_report(
        gender_valid[
            "gender_true"
        ],
        gender_valid[
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

    taxonomy = build_taxonomy_summary(
        predictions
    )

    taxonomy.to_csv(
        args.output
        / "taxonomy_summary.csv",
        index=False,
    )

    save_gender_confusion(
        predictions,
        args.output,
    )

    save_race_confusion(
        predictions,
        args.output,
    )

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
            "UTKFace"
        ),

        "input_preprocessing": (
            "project five-point alignment reconstructed "
            "from stored landmarks; DeepFace detector skipped"
        ),

        "n_images": int(
            len(
                predictions
            )
        ),

        "excluded_before_inference": int(
            len(
                excluded
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

    print()
    print(
        "Experiment complete"
    )

    print(
        "================================"
    )

    print(
        f"Images processed: "
        f"{len(predictions):,}"
    )

    print(
        f"Excluded:         "
        f"{len(excluded):,}"
    )

    print(
        f"Elapsed:          "
        f"{elapsed:.1f} seconds"
    )

    print(
        f"Throughput:       "
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
        "TAXONOMY SUMMARY"
    )

    print(
        taxonomy.to_string(
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
    