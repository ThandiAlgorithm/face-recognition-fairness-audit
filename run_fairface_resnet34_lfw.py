from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

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

DEFAULT_CHECKPOINT = Path(
    "models/fairface/res34_fair_align_multi_7_20190809.pt"
)

DEFAULT_OUTPUT = Path(
    "results/model_audit/fairface_resnet34/lfw_predictions"
)


RACE_LABELS = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
]

GENDER_LABELS = [
    "Male",
    "Female",
]

AGE_LABELS = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+",
]


# ============================================================
# CONFIG LOADING
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
    """
    Adapt the frozen preprocessing configuration in memory
    to the key names expected by src.detection.align_face().

    The frozen YAML itself is not modified.
    """

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
                "No alignment output size found."
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
                "No canonical landmark positions found."
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
            f"Expected 5 landmarks; "
            f"found {len(points)}."
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
# LOAD LFW
# ============================================================

def load_lfw(
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

    frame = master[
        dataset_text.eq(
            "lfw"
        )
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No LFW rows found in the master table."
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

    excluded = int(
        (~eligible).sum()
    )

    if excluded:
        print(
            f"WARNING: {excluded} LFW rows "
            "are not alignment eligible."
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

    return frame


# ============================================================
# PYTORCH DATASET
# ============================================================

class LFWAlignedDataset(
    Dataset
):

    def __init__(
        self,
        frame: pd.DataFrame,
        preprocessing_cfg: dict,
    ) -> None:

        self.frame = (
            frame.reset_index(
                drop=True
            )
        )

        self.preprocessing_cfg = (
            detection_compatible_config(
                preprocessing_cfg
            )
        )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),

                transforms.Resize(
                    (
                        224,
                        224,
                    )
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.frame
        )

    def __getitem__(
        self,
        index: int,
    ):

        row = self.frame.iloc[
            index
        ]

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
            self.preprocessing_cfg,
        )

        if aligned_rgb is None:
            raise RuntimeError(
                f"Alignment failed for "
                f"{row['image_id']}."
            )

        if transform_matrix is None:
            raise RuntimeError(
                f"No alignment transform for "
                f"{row['image_id']}."
            )

        if valid_mask is None:
            raise RuntimeError(
                f"No alignment mask for "
                f"{row['image_id']}."
            )

        tensor = self.transform(
            aligned_rgb
        )

        return (
            tensor,
            str(
                row[
                    "image_id"
                ]
            ),
        )


# ============================================================
# FAIRFACE MODEL
# ============================================================

def load_checkpoint(
    checkpoint_path: Path,
) -> dict:

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "state_dict" in checkpoint
    ):

        checkpoint = checkpoint[
            "state_dict"
        ]

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Unexpected checkpoint structure."
        )

    cleaned = {}

    for key, value in checkpoint.items():

        clean_key = str(
            key
        )

        if clean_key.startswith(
            "module."
        ):
            clean_key = clean_key[
                len("module.") :
            ]

        cleaned[
            clean_key
        ] = value

    return cleaned


def build_model(
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:

    model = models.resnet34(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        18,
    )

    model.load_state_dict(
        load_checkpoint(
            checkpoint_path
        ),
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[pd.DataFrame, float]:

    records = []

    total = len(
        loader.dataset
    )

    processed = 0

    start = time.perf_counter()

    with torch.inference_mode():

        for (
            images,
            image_ids,
        ) in loader:

            images = images.to(
                device
            )

            outputs = model(
                images
            )

            race_logits = outputs[
                :,
                0:7,
            ]

            gender_logits = outputs[
                :,
                7:9,
            ]

            age_logits = outputs[
                :,
                9:18,
            ]

            race_probs = (
                torch.softmax(
                    race_logits,
                    dim=1,
                )
                .cpu()
                .numpy()
            )

            gender_probs = (
                torch.softmax(
                    gender_logits,
                    dim=1,
                )
                .cpu()
                .numpy()
            )

            age_probs = (
                torch.softmax(
                    age_logits,
                    dim=1,
                )
                .cpu()
                .numpy()
            )

            race_indices = np.argmax(
                race_probs,
                axis=1,
            )

            gender_indices = np.argmax(
                gender_probs,
                axis=1,
            )

            age_indices = np.argmax(
                age_probs,
                axis=1,
            )

            for index, image_id in enumerate(
                image_ids
            ):

                race_index = int(
                    race_indices[
                        index
                    ]
                )

                gender_index = int(
                    gender_indices[
                        index
                    ]
                )

                age_index = int(
                    age_indices[
                        index
                    ]
                )

                record = {
                    "image_id": (
                        image_id
                    ),

                    "input_source": (
                        "reconstructed_project_alignment"
                    ),

                    "predicted_race": (
                        RACE_LABELS[
                            race_index
                        ]
                    ),

                    "predicted_race_confidence": float(
                        race_probs[
                            index,
                            race_index,
                        ]
                    ),

                    "predicted_gender": (
                        GENDER_LABELS[
                            gender_index
                        ]
                    ),

                    "predicted_gender_confidence": float(
                        gender_probs[
                            index,
                            gender_index,
                        ]
                    ),

                    "predicted_age_band": (
                        AGE_LABELS[
                            age_index
                        ]
                    ),

                    "predicted_age_confidence": float(
                        age_probs[
                            index,
                            age_index,
                        ]
                    ),

                    "gender_prob_male": float(
                        gender_probs[
                            index,
                            0,
                        ]
                    ),

                    "gender_prob_female": float(
                        gender_probs[
                            index,
                            1,
                        ]
                    ),
                }

                for race_number, race_name in enumerate(
                    RACE_LABELS
                ):

                    column_name = (
                        "race_prob_"
                        + race_name
                        .lower()
                        .replace(
                            " ",
                            "_",
                        )
                        .replace(
                            "/",
                            "_",
                        )
                    )

                    record[
                        column_name
                    ] = float(
                        race_probs[
                            index,
                            race_number,
                        ]
                    )

                records.append(
                    record
                )

            processed += len(
                image_ids
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
                    image_ids
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
# DISTRIBUTIONS
# ============================================================

def categorical_distribution(
    frame: pd.DataFrame,
    column: str,
    category_name: str,
) -> pd.DataFrame:

    table = (
        frame[
            column
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            category_name
        )
        .reset_index(
            name="n"
        )
    )

    table[
        "proportion"
    ] = (
        table[
            "n"
        ]
        /
        len(
            frame
        )
    )

    return table


# ============================================================
# INTERSECTIONAL PREDICTION DISTRIBUTION
# ============================================================

def build_predicted_intersections(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    table = (
        predictions.groupby(
            [
                "predicted_race",
                "predicted_gender",
            ],
            observed=True,
        )
        .agg(
            n=(
                "image_id",
                "count",
            ),

            mean_race_confidence=(
                "predicted_race_confidence",
                "mean",
            ),

            mean_gender_confidence=(
                "predicted_gender_confidence",
                "mean",
            ),
        )
        .reset_index()
    )

    table[
        "proportion"
    ] = (
        table[
            "n"
        ]
        /
        len(
            predictions
        )
    )

    table[
        "reporting_status"
    ] = np.select(
        [
            table[
                "n"
            ] < 30,

            table[
                "n"
            ] < 100,
        ],
        [
            "suppressed",
            "low_confidence",
        ],
        default="reportable",
    )

    return table


# ============================================================
# CONFIDENCE SUMMARY
# ============================================================

def build_confidence_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    metrics = [
        (
            "race",
            "predicted_race_confidence",
        ),
        (
            "gender",
            "predicted_gender_confidence",
        ),
        (
            "age",
            "predicted_age_confidence",
        ),
    ]

    for task, column in metrics:

        values = predictions[
            column
        ].dropna()

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

                "below_0_50": int(
                    (
                        values < 0.50
                    ).sum()
                ),

                "below_0_70": int(
                    (
                        values < 0.70
                    ).sum()
                ),

                "below_0_80": int(
                    (
                        values < 0.80
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# IDENTITY CONSISTENCY
# ============================================================

def modal_value_and_consistency(
    series: pd.Series,
) -> tuple[str, float]:

    counts = series.value_counts(
        dropna=True
    )

    if counts.empty:
        return (
            "",
            np.nan,
        )

    mode = str(
        counts.index[
            0
        ]
    )

    consistency = float(
        counts.iloc[
            0
        ]
        /
        counts.sum()
    )

    return (
        mode,
        consistency,
    )


def build_identity_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    if "subject_id" not in predictions.columns:
        return pd.DataFrame()

    frame = predictions[
        predictions[
            "subject_id"
        ].notna()
    ].copy()

    rows = []

    for subject_id, group in frame.groupby(
        "subject_id",
        observed=True,
    ):

        (
            modal_race,
            race_consistency,
        ) = modal_value_and_consistency(
            group[
                "predicted_race"
            ]
        )

        (
            modal_gender,
            gender_consistency,
        ) = modal_value_and_consistency(
            group[
                "predicted_gender"
            ]
        )

        (
            modal_age,
            age_consistency,
        ) = modal_value_and_consistency(
            group[
                "predicted_age_band"
            ]
        )

        rows.append(
            {
                "subject_id": (
                    subject_id
                ),

                "n_images": int(
                    len(
                        group
                    )
                ),

                "modal_predicted_race": (
                    modal_race
                ),

                "race_prediction_consistency": (
                    race_consistency
                ),

                "modal_predicted_gender": (
                    modal_gender
                ),

                "gender_prediction_consistency": (
                    gender_consistency
                ),

                "modal_predicted_age_band": (
                    modal_age
                ),

                "age_prediction_consistency": (
                    age_consistency
                ),

                "mean_race_confidence": float(
                    group[
                        "predicted_race_confidence"
                    ].mean()
                ),

                "mean_gender_confidence": float(
                    group[
                        "predicted_gender_confidence"
                    ].mean()
                ),

                "mean_age_confidence": float(
                    group[
                        "predicted_age_confidence"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_repeated_identity_summary(
    identity_summary: pd.DataFrame,
) -> pd.DataFrame:

    if identity_summary.empty:
        return pd.DataFrame()

    repeated = identity_summary[
        identity_summary[
            "n_images"
        ] >= 2
    ].copy()

    if repeated.empty:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "n_repeated_identities": int(
                    len(
                        repeated
                    )
                ),

                "n_images_repeated_identities": int(
                    repeated[
                        "n_images"
                    ].sum()
                ),

                "mean_race_prediction_consistency": float(
                    repeated[
                        "race_prediction_consistency"
                    ].mean()
                ),

                "median_race_prediction_consistency": float(
                    repeated[
                        "race_prediction_consistency"
                    ].median()
                ),

                "fully_race_consistent_identities": int(
                    repeated[
                        "race_prediction_consistency"
                    ]
                    .eq(
                        1.0
                    )
                    .sum()
                ),

                "mean_gender_prediction_consistency": float(
                    repeated[
                        "gender_prediction_consistency"
                    ].mean()
                ),

                "median_gender_prediction_consistency": float(
                    repeated[
                        "gender_prediction_consistency"
                    ].median()
                ),

                "fully_gender_consistent_identities": int(
                    repeated[
                        "gender_prediction_consistency"
                    ]
                    .eq(
                        1.0
                    )
                    .sum()
                ),

                "mean_age_prediction_consistency": float(
                    repeated[
                        "age_prediction_consistency"
                    ].mean()
                ),
            }
        ]
    )


# ============================================================
# IDENTITY DEPTH BY PREDICTED GROUP
# ============================================================

def build_identity_depth_by_predicted_race(
    identity_summary: pd.DataFrame,
) -> pd.DataFrame:

    if identity_summary.empty:
        return pd.DataFrame()

    grouped = (
        identity_summary.groupby(
            "modal_predicted_race",
            observed=True,
        )[
            "n_images"
        ]
        .agg(
            [
                "count",
                "min",
                "median",
                "mean",
                "max",
            ]
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "count": (
                "n_identities"
            ),
            "min": (
                "min_images_per_identity"
            ),
            "median": (
                "median_images_per_identity"
            ),
            "mean": (
                "mean_images_per_identity"
            ),
            "max": (
                "max_images_per_identity"
            ),
        }
    )

    return grouped


# ============================================================
# LOW-CONFIDENCE CASES
# ============================================================

def save_low_confidence_cases(
    predictions: pd.DataFrame,
    output: Path,
) -> None:

    race_low = (
        predictions
        .sort_values(
            "predicted_race_confidence"
        )
        .head(
            250
        )
    )

    gender_low = (
        predictions
        .sort_values(
            "predicted_gender_confidence"
        )
        .head(
            250
        )
    )

    age_low = (
        predictions
        .sort_values(
            "predicted_age_confidence"
        )
        .head(
            250
        )
    )

    race_low.to_csv(
        output
        / "lowest_race_confidence_250.csv",
        index=False,
    )

    gender_low.to_csv(
        output
        / "lowest_gender_confidence_250.csv",
        index=False,
    )

    age_low.to_csv(
        output
        / "lowest_age_confidence_250.csv",
        index=False,
    )


# ============================================================
# BASIC FIGURES
# ============================================================

def save_distribution_plot(
    table: pd.DataFrame,
    category_column: str,
    title: str,
    filename: Path,
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
        ].astype(str),
        table[
            "n"
        ],
    )

    ax.set_title(
        title
    )

    ax.set_ylabel(
        "Number of LFW images"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
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
        filename,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_intersection_plot(
    intersections: pd.DataFrame,
    output: Path,
) -> None:

    pivot = intersections.pivot(
        index="predicted_race",
        columns="predicted_gender",
        values="n",
    )

    pivot = pivot.reindex(
        RACE_LABELS
    )

    pivot = pivot.reindex(
        columns=GENDER_LABELS
    )

    pivot = pivot.fillna(
        0
    )

    matrix = pivot.values

    fig, ax = plt.subplots(
        figsize=(
            8,
            8,
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
                RACE_LABELS
            )
        )
    )

    ax.set_yticklabels(
        RACE_LABELS
    )

    ax.set_xlabel(
        "Predicted gender"
    )

    ax.set_ylabel(
        "Predicted race"
    )

    ax.set_title(
        "LFW: FairFace-Predicted Race × Gender Distribution"
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
            )

    fig.colorbar(
        display,
        ax=ax,
        label="Image count",
    )

    fig.tight_layout()

    fig.savefig(
        output
        / "predicted_race_gender_intersection.png",
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
            "Generate FairFace ResNet-34 demographic "
            "predictions for deterministically aligned LFW images."
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
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    for required in [
        args.master,
        args.preprocessing,
        args.checkpoint,
    ]:

        if not required.exists():

            raise FileNotFoundError(
                f"Required file not found: "
                f"{required}"
            )

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        hardware = (
            torch.cuda.get_device_name(
                0
            )
        )

        default_batch_size = 64

    else:

        device = torch.device(
            "cpu"
        )

        hardware = "CPU"

        default_batch_size = 16

    batch_size = (
        args.batch_size
        if args.batch_size
        is not None
        else default_batch_size
    )

    print()
    print(
        "FairFace ResNet-34 → LFW Prediction Audit"
    )

    print(
        "========================================="
    )

    print(
        f"Device:     {device}"
    )

    print(
        f"Hardware:   {hardware}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    preprocessing_cfg = (
        load_preprocessing_config(
            args.preprocessing
        )
    )

    frame = load_lfw(
        args.master,
        args.limit,
    )

    print()
    print(
        f"LFW images selected: "
        f"{len(frame):,}"
    )

    print(
        "Ground-truth demographic labels: "
        "not available in canonical LFW table"
    )

    print()
    print(
        "Loading FairFace ResNet-34..."
    )

    model = build_model(
        args.checkpoint,
        device,
    )

    print(
        "Model loaded successfully."
    )

    dataset = LFWAlignedDataset(
        frame,
        preprocessing_cfg,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        "Reconstructing alignment and "
        "running demographic predictions..."
    )

    inference, elapsed = (
        run_inference(
            model,
            loader,
            device,
        )
    )

    predictions = frame.merge(
        inference,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if predictions[
        "predicted_race"
    ].isna().any():

        missing = int(
            predictions[
                "predicted_race"
            ]
            .isna()
            .sum()
        )

        raise RuntimeError(
            f"{missing} LFW images did not "
            "receive predictions."
        )

    # --------------------------------------------------------
    # RAW PREDICTIONS
    # --------------------------------------------------------

    predictions.to_csv(
        args.output
        / "predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # PREDICTED DISTRIBUTIONS
    # --------------------------------------------------------

    race_distribution = (
        categorical_distribution(
            predictions,
            "predicted_race",
            "predicted_race",
        )
    )

    race_distribution.to_csv(
        args.output
        / "predicted_race_distribution.csv",
        index=False,
    )

    gender_distribution = (
        categorical_distribution(
            predictions,
            "predicted_gender",
            "predicted_gender",
        )
    )

    gender_distribution.to_csv(
        args.output
        / "predicted_gender_distribution.csv",
        index=False,
    )

    age_distribution = (
        categorical_distribution(
            predictions,
            "predicted_age_band",
            "predicted_age_band",
        )
    )

    age_distribution.to_csv(
        args.output
        / "predicted_age_distribution.csv",
        index=False,
    )

    intersections = (
        build_predicted_intersections(
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

    confidence_summary = (
        build_confidence_summary(
            predictions
        )
    )

    confidence_summary.to_csv(
        args.output
        / "confidence_summary.csv",
        index=False,
    )

    save_low_confidence_cases(
        predictions,
        args.output,
    )

    # --------------------------------------------------------
    # IDENTITY CONSISTENCY
    # --------------------------------------------------------

    identity_summary = (
        build_identity_summary(
            predictions
        )
    )

    if not identity_summary.empty:

        identity_summary.to_csv(
            args.output
            / "identity_prediction_summary.csv",
            index=False,
        )

        repeated_identity_summary = (
            build_repeated_identity_summary(
                identity_summary
            )
        )

        repeated_identity_summary.to_csv(
            args.output
            / "repeated_identity_consistency_summary.csv",
            index=False,
        )

        identity_depth = (
            build_identity_depth_by_predicted_race(
                identity_summary
            )
        )

        identity_depth.to_csv(
            args.output
            / "identity_depth_by_modal_predicted_race.csv",
            index=False,
        )

    else:

        repeated_identity_summary = (
            pd.DataFrame()
        )

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    save_distribution_plot(
        race_distribution,
        "predicted_race",
        (
            "LFW: FairFace-Predicted "
            "Race Distribution"
        ),
        args.output
        / "predicted_race_distribution.png",
    )

    save_distribution_plot(
        gender_distribution,
        "predicted_gender",
        (
            "LFW: FairFace-Predicted "
            "Gender Distribution"
        ),
        args.output
        / "predicted_gender_distribution.png",
    )

    save_distribution_plot(
        age_distribution,
        "predicted_age_band",
        (
            "LFW: FairFace-Predicted "
            "Age-Band Distribution"
        ),
        args.output
        / "predicted_age_distribution.png",
    )

    save_intersection_plot(
        intersections,
        args.output,
    )

    # --------------------------------------------------------
    # RUNTIME
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
        "architecture": (
            "FairFace ResNet-34"
        ),

        "dataset": (
            "LFW"
        ),

        "analysis_type": (
            "demographic prediction audit; "
            "no demographic ground truth used"
        ),

        "input_preprocessing": (
            "project five-point alignment "
            "reconstructed from stored landmarks"
        ),

        "checkpoint_sha256": (
            "37F2E74C0E7DA196F8EDD7DAE0C88CFF"
            "DBEF7923BEEBEBCC27381F0AD0FDB2C1"
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

        "device": str(
            device
        ),

        "hardware": (
            hardware
        ),

        "batch_size": int(
            batch_size
        ),

        "seed": int(
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
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print()
    print(
        "LFW prediction audit complete"
    )

    print(
        "========================================="
    )

    print(
        f"Images processed: "
        f"{len(predictions):,}"
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
        "PREDICTED AGE DISTRIBUTION"
    )

    print(
        age_distribution.to_string(
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
        confidence_summary.to_string(
            index=False
        )
    )

    if not repeated_identity_summary.empty:

        print()
        print(
            "REPEATED-IDENTITY PREDICTION CONSISTENCY"
        )

        print(
            repeated_identity_summary.to_string(
                index=False
            )
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "These are FairFace model predictions, "
        "not LFW demographic ground-truth labels."
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
    