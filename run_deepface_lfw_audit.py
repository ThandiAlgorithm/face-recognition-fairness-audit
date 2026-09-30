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
    "results/model_audit/deepface/lfw_aligned"
)


DEEPFACE_RACE_LABELS_RAW = [
    "asian",
    "indian",
    "black",
    "white",
    "middle eastern",
    "latino hispanic",
]

RACE_LABELS = [
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
# NORMALISATION
# ============================================================

def normalise_deepface_gender(
    value: str,
) -> str:

    mapping = {
        "Man": "Male",
        "Woman": "Female",
    }

    return mapping[value]


def normalise_deepface_race(
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

    return mapping[value]


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

        config = yaml.safe_load(handle)

    if not isinstance(config, dict):

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
                "Alignment configuration contains no output size."
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
# LANDMARKS
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
                f"Landmark {index} does not contain "
                "two coordinates."
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
# LOAD LFW
# ============================================================

def load_lfw(
    master_path: Path,
    limit: int | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

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

    lfw = master[
        dataset_text.eq(
            "lfw"
        )
    ].copy()

    if lfw.empty:

        raise RuntimeError(
            "No LFW rows found."
        )

    eligible = (
        lfw[
            "alignment_valid"
        ].eq(
            True
        )
        &
        lfw[
            "landmarks"
        ].notna()
    )

    excluded = lfw[
        ~eligible
    ].copy()

    if not excluded.empty:

        excluded[
            "model_audit_exclusion_reason"
        ] = np.where(
            excluded[
                "landmarks"
            ].isna(),
            "landmarks_unavailable",
            "alignment_not_valid",
        )

    frame = lfw[
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
# MODELS
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
) -> tuple[
    pd.DataFrame,
    float,
]:

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

            gender_vector = (
                gender_probs[
                    local_index
                ]
                .astype(
                    float
                )
            )

            race_vector = (
                race_probs[
                    local_index
                ]
                .astype(
                    float
                )
            )

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

            gender_sum = max(
                float(
                    gender_vector.sum()
                ),
                1e-12,
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

                "subject_id": (
                    row[
                        "subject_id"
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
                    /
                    gender_sum
                ),

                "race_pred_raw": (
                    race_raw
                ),

                "race_pred": (
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
                    /
                    gender_sum
                ),

                "gender_prob_male": float(
                    gender_vector[
                        1
                    ]
                    /
                    gender_sum
                ),
            }

            for (
                race_position,
                race_name,
            ) in enumerate(
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
            /
            elapsed
            if elapsed > 0
            else 0.0
        )

        remaining = (
            total
            -
            processed
        )

        eta_seconds = (
            remaining
            /
            rate
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
# PREDICTED COMPOSITION
# ============================================================

def build_distribution(
    predictions: pd.DataFrame,
    column: str,
    confidence_column: str,
    category_name: str,
) -> pd.DataFrame:

    rows = []

    for category, group in predictions.groupby(
        column,
        observed=True,
    ):

        rows.append(
            {
                category_name: (
                    category
                ),

                "n": int(
                    len(
                        group
                    )
                ),

                "proportion": float(
                    len(
                        group
                    )
                    /
                    len(
                        predictions
                    )
                ),

                "mean_confidence": float(
                    group[
                        confidence_column
                    ].mean()
                ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "n",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# INTERSECTIONAL PREDICTION DISTRIBUTION
# ============================================================

def build_intersections(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    grouped = (
        predictions.groupby(
            [
                "race_pred",
                "gender_pred",
            ],
            observed=True,
        )
        .agg(
            n=(
                "image_id",
                "count",
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
        "proportion"
    ] = (
        grouped[
            "n"
        ]
        /
        len(
            predictions
        )
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

    return (
        grouped
        .sort_values(
            [
                "race_pred",
                "gender_pred",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# CONFIDENCE
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
                "task": (
                    task
                ),

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

                "below_0_5": int(
                    (
                        values
                        < 0.5
                    ).sum()
                ),

                "below_0_7": int(
                    (
                        values
                        < 0.7
                    ).sum()
                ),

                "below_0_8": int(
                    (
                        values
                        < 0.8
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# REPEATED-IDENTITY CONSISTENCY
# ============================================================

def modal_value_and_count(
    series: pd.Series,
) -> tuple[
    str,
    int,
]:

    counts = series.value_counts(
        dropna=False
    )

    maximum = int(
        counts.max()
    )

    candidates = sorted(
        [
            str(
                value
            )
            for value, count
            in counts.items()
            if count == maximum
        ]
    )

    return (
        candidates[
            0
        ],
        maximum,
    )


def build_identity_consistency(
    predictions: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    subject_counts = (
        predictions[
            "subject_id"
        ]
        .value_counts(
            dropna=True
        )
    )

    repeated_subjects = set(
        subject_counts[
            subject_counts > 1
        ].index
    )

    repeated = predictions[
        predictions[
            "subject_id"
        ].isin(
            repeated_subjects
        )
    ].copy()

    rows = []

    for subject_id, group in repeated.groupby(
        "subject_id",
        observed=True,
    ):

        n_images = int(
            len(
                group
            )
        )

        (
            modal_race,
            race_mode_count,
        ) = modal_value_and_count(
            group[
                "race_pred"
            ]
        )

        (
            modal_gender,
            gender_mode_count,
        ) = modal_value_and_count(
            group[
                "gender_pred"
            ]
        )

        race_consistency = (
            race_mode_count
            /
            n_images
        )

        gender_consistency = (
            gender_mode_count
            /
            n_images
        )

        rows.append(
            {
                "subject_id": (
                    subject_id
                ),

                "n_images": (
                    n_images
                ),

                "modal_predicted_race": (
                    modal_race
                ),

                "race_mode_count": (
                    race_mode_count
                ),

                "race_consistency": float(
                    race_consistency
                ),

                "race_fully_consistent": bool(
                    race_mode_count
                    ==
                    n_images
                ),

                "modal_predicted_gender": (
                    modal_gender
                ),

                "gender_mode_count": (
                    gender_mode_count
                ),

                "gender_consistency": float(
                    gender_consistency
                ),

                "gender_fully_consistent": bool(
                    gender_mode_count
                    ==
                    n_images
                ),

                "mean_race_confidence": float(
                    group[
                        "model_race_confidence"
                    ].mean()
                ),

                "mean_gender_confidence": float(
                    group[
                        "model_gender_confidence"
                    ].mean()
                ),
            }
        )

    identity = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "n_images",
                "subject_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    if identity.empty:

        summary = pd.DataFrame(
            [
                {
                    "measure": (
                        "Repeated identities"
                    ),
                    "value": 0,
                },
                {
                    "measure": (
                        "Images belonging to repeated identities"
                    ),
                    "value": 0,
                },
            ]
        )

        by_depth = pd.DataFrame()

        return (
            identity,
            summary,
            by_depth,
        )

    summary = pd.DataFrame(
        [
            {
                "measure": (
                    "Repeated identities"
                ),

                "value": int(
                    len(
                        identity
                    )
                ),
            },

            {
                "measure": (
                    "Images belonging to repeated identities"
                ),

                "value": int(
                    len(
                        repeated
                    )
                ),
            },

            {
                "measure": (
                    "Mean race consistency"
                ),

                "value": float(
                    identity[
                        "race_consistency"
                    ].mean()
                ),
            },

            {
                "measure": (
                    "Median race consistency"
                ),

                "value": float(
                    identity[
                        "race_consistency"
                    ].median()
                ),
            },

            {
                "measure": (
                    "Fully race-consistent identities"
                ),

                "value": int(
                    identity[
                        "race_fully_consistent"
                    ].sum()
                ),
            },

            {
                "measure": (
                    "Mean gender consistency"
                ),

                "value": float(
                    identity[
                        "gender_consistency"
                    ].mean()
                ),
            },

            {
                "measure": (
                    "Median gender consistency"
                ),

                "value": float(
                    identity[
                        "gender_consistency"
                    ].median()
                ),
            },

            {
                "measure": (
                    "Fully gender-consistent identities"
                ),

                "value": int(
                    identity[
                        "gender_fully_consistent"
                    ].sum()
                ),
            },
        ]
    )

    identity[
        "identity_depth_band"
    ] = pd.cut(
        identity[
            "n_images"
        ],
        bins=[
            1,
            2,
            4,
            9,
            19,
            np.inf,
        ],
        labels=[
            "2",
            "3-4",
            "5-9",
            "10-19",
            "20+",
        ],
        right=True,
    )

    by_depth = (
        identity.groupby(
            "identity_depth_band",
            observed=True,
        )
        .agg(
            n_identities=(
                "subject_id",
                "count",
            ),

            mean_race_consistency=(
                "race_consistency",
                "mean",
            ),

            median_race_consistency=(
                "race_consistency",
                "median",
            ),

            mean_gender_consistency=(
                "gender_consistency",
                "mean",
            ),

            median_gender_consistency=(
                "gender_consistency",
                "median",
            ),
        )
        .reset_index()
    )

    return (
        identity,
        summary,
        by_depth,
    )


# ============================================================
# FIGURES
# ============================================================

def save_distribution_figure(
    table: pd.DataFrame,
    category_column: str,
    title: str,
    path: Path,
) -> None:

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    bars = ax.bar(
        table[
            category_column
        ],
        table[
            "proportion"
        ],
    )

    ax.set_ylabel(
        "Proportion of LFW images"
    )

    ax.set_title(
        title
    )

    ax.set_ylim(
        0,
        max(
            table[
                "proportion"
            ].max()
            * 1.18,
            0.05,
        ),
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    for bar, value in zip(
        bars,
        table[
            "proportion"
        ],
    ):

        ax.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            bar.get_height(),
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_intersection_heatmap(
    predictions: pd.DataFrame,
    path: Path,
) -> None:

    table = pd.crosstab(
        predictions[
            "race_pred"
        ],
        predictions[
            "gender_pred"
        ],
    )

    table = table.reindex(
        index=RACE_LABELS,
        columns=GENDER_LABELS,
        fill_value=0,
    )

    proportions = (
        table
        /
        len(
            predictions
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            8,
            7,
        )
    )

    display = ax.imshow(
        proportions.values
    )

    ax.set_xticks(
        range(
            len(
                proportions.columns
            )
        )
    )

    ax.set_xticklabels(
        proportions.columns
    )

    ax.set_yticks(
        range(
            len(
                proportions.index
            )
        )
    )

    ax.set_yticklabels(
        proportions.index
    )

    ax.set_xlabel(
        "Predicted gender"
    )

    ax.set_ylabel(
        "Predicted race"
    )

    ax.set_title(
        "DeepFace Predicted Race × Gender on LFW"
    )

    maximum = float(
        proportions.values.max()
    )

    midpoint = (
        maximum / 2
        if maximum > 0
        else 0
    )

    for row in range(
        proportions.shape[
            0
        ]
    ):

        for column in range(
            proportions.shape[
                1
            ]
        ):

            value = float(
                proportions.iloc[
                    row,
                    column,
                ]
            )

            ax.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > midpoint
                    else "black"
                ),
                fontsize=9,
            )

    fig.colorbar(
        display,
        ax=ax,
        label="Proportion",
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=240,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_consistency_figure(
    identity: pd.DataFrame,
    path: Path,
) -> None:

    if identity.empty:

        return

    values = [
        float(
            identity[
                "race_consistency"
            ].mean()
        ),

        float(
            identity[
                "gender_consistency"
            ].mean()
        ),
    ]

    labels = [
        "Race",
        "Gender",
    ]

    fig, ax = plt.subplots(
        figsize=(
            7,
            5,
        )
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Mean modal-label consistency"
    )

    ax.set_title(
        "DeepFace Repeated-Identity Consistency on LFW"
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            value + 0.015,
            f"{value:.1%}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=240,
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
            "Prediction-only DeepFace demographic audit "
            "on deterministically aligned LFW images."
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
    ) = load_lfw(
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
        "DeepFace → LFW Prediction-Only Audit"
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

    print()

    print(
        f"Alignment-eligible images: "
        f"{len(frame):,}"
    )

    print(
        f"Excluded before inference: "
        f"{len(excluded):,}"
    )

    repeated_identities = int(
        (
            frame[
                "subject_id"
            ]
            .value_counts()
            > 1
        ).sum()
    )

    print(
        f"Repeated identities represented: "
        f"{repeated_identities:,}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "LFW has no independent demographic "
        "ground truth in the canonical master table."
    )

    print(
        "Outputs below describe model-assigned "
        "demographic predictions, not verified demographics."
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

    if len(
        inference
    ) != len(
        frame
    ):

        raise RuntimeError(
            "Inference row count does not match "
            "input row count."
        )

    inference_for_merge = inference.drop(
        columns=[
            "subject_id",
        ]
    )

    predictions = frame.merge(
        inference_for_merge,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    required_columns = [
        "race_pred",
        "gender_pred",
        "model_race_confidence",
        "model_gender_confidence",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Expected DeepFace output columns are "
            f"missing after merge: {missing_columns}"
        )

    if predictions[
        "race_pred"
    ].isna().any():

        raise RuntimeError(
            "One or more LFW images did not "
            "receive a race prediction."
        )

    if predictions[
        "gender_pred"
    ].isna().any():

        raise RuntimeError(
            "One or more LFW images did not "
            "receive a gender prediction."
        )

    predictions.to_csv(
        args.output
        / "predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # PREDICTED DISTRIBUTIONS
    # --------------------------------------------------------

    race_distribution = (
        build_distribution(
            predictions,
            "race_pred",
            "model_race_confidence",
            "predicted_race",
        )
    )

    race_distribution.to_csv(
        args.output
        / "predicted_race_distribution.csv",
        index=False,
    )

    gender_distribution = (
        build_distribution(
            predictions,
            "gender_pred",
            "model_gender_confidence",
            "predicted_gender",
        )
    )

    gender_distribution.to_csv(
        args.output
        / "predicted_gender_distribution.csv",
        index=False,
    )

    intersections = (
        build_intersections(
            predictions
        )
    )

    intersections.to_csv(
        args.output
        / "predicted_race_gender_intersections.csv",
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
    # REPEATED IDENTITIES
    # --------------------------------------------------------

    (
        identity_consistency,
        identity_summary,
        consistency_by_depth,
    ) = build_identity_consistency(
        predictions
    )

    identity_consistency.to_csv(
        args.output
        / "repeated_identity_consistency.csv",
        index=False,
    )

    identity_summary.to_csv(
        args.output
        / "repeated_identity_consistency_summary.csv",
        index=False,
    )

    consistency_by_depth.to_csv(
        args.output
        / "identity_consistency_by_depth.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    save_distribution_figure(
        race_distribution,
        "predicted_race",
        "DeepFace Predicted Race Distribution on LFW",
        args.output
        / "01_predicted_race_distribution.png",
    )

    save_distribution_figure(
        gender_distribution,
        "predicted_gender",
        "DeepFace Predicted Gender Distribution on LFW",
        args.output
        / "02_predicted_gender_distribution.png",
    )

    save_intersection_heatmap(
        predictions,
        args.output
        / "03_predicted_race_gender_heatmap.png",
    )

    save_consistency_figure(
        identity_consistency,
        args.output
        / "04_repeated_identity_consistency.png",
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
            "LFW"
        ),

        "audit_type": (
            "prediction_only"
        ),

        "ground_truth_demographics_available": (
            False
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

        "repeated_identities": int(
            len(
                identity_consistency
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
        "PREDICTED RACE DISTRIBUTION"
    )

    print(
        race_distribution.to_string(
            index=False
        )
    )

    print()

    print(
        "PREDICTED GENDER DISTRIBUTION"
    )

    print(
        gender_distribution.to_string(
            index=False
        )
    )

    print()

    print(
        "PREDICTED RACE × GENDER"
    )

    print(
        intersections.to_string(
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
        "REPEATED-IDENTITY CONSISTENCY"
    )

    print(
        identity_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "CONSISTENCY BY IDENTITY DEPTH"
    )

    if consistency_by_depth.empty:

        print(
            "No repeated identities available."
        )

    else:

        print(
            consistency_by_depth.to_string(
                index=False
            )
        )

    print()

    print(
        "IMPORTANT INTERPRETATION"
    )

    print(
        "These are DeepFace model-assigned demographic "
        "labels. They are not LFW demographic ground truth."
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
    