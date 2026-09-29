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

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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
    "results/model_audit/fairface_resnet34/utkface_aligned"
)


FAIRFACE_RACE_LABELS = [
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


def harmonise_fairface_prediction(
    fairface_race: str,
) -> str:

    mapping = {
        "White": "White",
        "Black": "Black",

        "East Asian": "Asian",
        "Southeast Asian": "Asian",

        "Indian": "Indian",

        "Latino_Hispanic": "Out_of_scope",
        "Middle Eastern": "Out_of_scope",
    }

    return mapping[
        fairface_race
    ]


# ============================================================
# PREPROCESSING CONFIG
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
            "preprocessing.yaml could not be loaded."
        )

    return config


def detection_compatible_config(
    preprocessing_cfg: dict,
) -> dict:
    """
    src.detection.align_face() expects the older/internal alignment
    key names. The frozen project configuration may use the more
    descriptive names output_size_px and
    canonical_landmark_positions.

    Adapt in memory only. Do not modify the frozen YAML.
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
                "Alignment configuration has neither "
                "'output_size' nor 'output_size_px'."
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
                "Alignment configuration has neither "
                "'canonical_landmark_fractions' nor "
                "'canonical_landmark_positions'."
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
    """
    Convert the Parquet object-array representation into the
    numeric 5 x 2 structure expected by align_face().
    """

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
            f"Expected 5 landmarks, found {len(points)}."
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
                f"Landmark {index} has "
                f"{numeric.size} coordinates."
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

    array = np.asarray(
        normalised,
        dtype=np.float32,
    )

    if array.shape != (
        5,
        2,
    ):
        raise RuntimeError(
            f"Unexpected landmark shape: "
            f"{array.shape}"
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

    alignment_eligible = (
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
        ~alignment_eligible
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
        alignment_eligible
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
# UTKFACE DATASET
# ============================================================

class UTKFaceAlignedDataset(
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

        original_path = str(
            row[
                "original_path"
            ]
        )

        rgb, _ = (
            image_integrity.load_rgb_for_analysis(
                paths.data_abs(
                    original_path
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
                f"Alignment transform unavailable for "
                f"{row['image_id']}."
            )

        if valid_mask is None:
            raise RuntimeError(
                f"Alignment mask unavailable for "
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

    rows = []

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

                fairface_race = (
                    FAIRFACE_RACE_LABELS[
                        race_index
                    ]
                )

                record = {
                    "image_id": (
                        image_id
                    ),

                    "input_source": (
                        "reconstructed_project_alignment"
                    ),

                    "race_pred_fairface": (
                        fairface_race
                    ),

                    "race_pred_harmonised": (
                        harmonise_fairface_prediction(
                            fairface_race
                        )
                    ),

                    "model_race_confidence": float(
                        race_probs[
                            index,
                            race_index,
                        ]
                    ),

                    "gender_pred": (
                        GENDER_LABELS[
                            gender_index
                        ]
                    ),

                    "model_gender_confidence": float(
                        gender_probs[
                            index,
                            gender_index,
                        ]
                    ),

                    "age_pred": (
                        AGE_LABELS[
                            age_index
                        ]
                    ),

                    "model_age_confidence": float(
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
                    FAIRFACE_RACE_LABELS
                ):

                    column = (
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
                        column
                    ] = float(
                        race_probs[
                            index,
                            race_number,
                        ]
                    )

                rows.append(
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
            rows
        ),
        elapsed,
    )


# ============================================================
# METRIC HELPERS
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
# INTERSECTIONAL METRICS
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
        ).astype(float),
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
# GENDER FAIRNESS ACROSS UTKFACE RACE
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
        - valid_tpr.min()
    )

    fpr_difference = float(
        valid_fpr.max()
        - valid_fpr.min()
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
# RACE ONE-VS-REST FAIRNESS ACROSS GENDER
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
                "tpr": tpr,
                "fpr": fpr,
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
# TAXONOMY ANALYSIS
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

    out_of_scope = (
        comparable[
            "race_pred_harmonised"
        ].eq(
            "Out_of_scope"
        )
    )

    others = predictions[
        predictions[
            "race_true_native"
        ].eq(
            "Others"
        )
    ]

    rows = [
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

    return pd.DataFrame(
        rows
    )


# ============================================================
# PREDICTION DISTRIBUTIONS
# ============================================================

def save_prediction_distributions(
    predictions: pd.DataFrame,
    output: Path,
) -> None:

    race_distribution = (
        predictions[
            "race_pred_fairface"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "fairface_predicted_race"
        )
        .reset_index(
            name="n"
        )
    )

    race_distribution[
        "proportion"
    ] = (
        race_distribution[
            "n"
        ]
        /
        len(
            predictions
        )
    )

    race_distribution.to_csv(
        output
        / "fairface_race_prediction_distribution.csv",
        index=False,
    )

    cross = pd.crosstab(
        predictions[
            "race_true_native"
        ],
        predictions[
            "race_pred_fairface"
        ],
    )

    cross.to_csv(
        output
        / "fairface_prediction_by_utk_native_race.csv"
    )


# ============================================================
# CONFUSION MATRIX FIGURES
# ============================================================

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
    ].copy()

    matrix = pd.crosstab(
        frame[
            "race_true_native"
        ],
        frame[
            "race_pred_harmonised"
        ],
    )

    matrix = matrix.reindex(
        index=HARMONISED_RACES,
        columns=PREDICTED_HARMONISED_CLASSES,
        fill_value=0,
    )

    matrix.index.name = (
        "actual"
    )

    matrix.columns.name = (
        "predicted"
    )

    matrix.to_csv(
        output
        / "race_confusion_matrix_harmonised.csv"
    )

    fig, ax = plt.subplots(
        figsize=(
            9,
            7,
        )
    )

    display = ax.imshow(
        matrix.values
    )

    ax.set_xticks(
        range(
            len(
                matrix.columns
            )
        )
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=35,
        ha="right",
    )

    ax.set_yticks(
        range(
            len(
                matrix.index
            )
        )
    )

    ax.set_yticklabels(
        matrix.index
    )

    ax.set_xlabel(
        "Predicted harmonised category"
    )

    ax.set_ylabel(
        "Actual UTKFace category"
    )

    ax.set_title(
        "FairFace ResNet-34 on UTKFace: "
        "Race Confusion Matrix"
    )

    maximum = (
        matrix.values.max()
        if matrix.size
        else 0
    )

    midpoint = (
        maximum / 2
        if maximum
        else 0
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            value = int(
                matrix.iloc[
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


def save_gender_confusion(
    predictions: pd.DataFrame,
    output: Path,
) -> None:

    frame = predictions[
        predictions[
            "gender_true"
        ].notna()
    ].copy()

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
        "FairFace ResNet-34 on UTKFace: "
        "Gender Confusion Matrix"
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
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Cross-dataset FairFace ResNet-34 audit "
            "on deterministically aligned UTKFace images."
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
        "FairFace ResNet-34 → "
        "UTKFace Aligned Audit"
    )

    print(
        "================================"
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

    dataset = UTKFaceAlignedDataset(
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
        "Loading FairFace ResNet-34..."
    )

    model = build_model(
        args.checkpoint,
        device,
    )

    print(
        "Model loaded successfully."
    )

    print()
    print(
        "Reconstructing alignment "
        "and running inference..."
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
        "gender_pred"
    ].isna().any():

        missing = int(
            predictions[
                "gender_pred"
            ].isna().sum()
        )

        raise RuntimeError(
            f"{missing} eligible rows did not "
            "receive predictions."
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
    ].copy()

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
    ].copy()

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

    taxonomy = (
        build_taxonomy_summary(
            predictions
        )
    )

    taxonomy.to_csv(
        args.output
        / "taxonomy_summary.csv",
        index=False,
    )

    save_prediction_distributions(
        predictions,
        args.output,
    )

    save_race_confusion(
        predictions,
        args.output,
    )

    save_gender_confusion(
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
        "architecture": (
            "FairFace ResNet-34"
        ),

        "dataset": (
            "UTKFace"
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

        "device": str(
            device
        ),

        "hardware": (
            hardware
        ),

        "batch_size": int(
            batch_size
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
        "GENDER EQUALIZED-ODDS SUMMARY"
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
    