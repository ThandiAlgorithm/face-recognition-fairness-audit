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

from src import paths


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 26041

DEFAULT_MASTER = Path(
    "metadata/master_annotations.parquet"
)

DEFAULT_CHECKPOINT = Path(
    "models/fairface/res34_fair_align_multi_7_20190809.pt"
)

DEFAULT_OUTPUT = Path(
    "results/model_audit/fairface_resnet34/fairface_validation"
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
# LABEL NORMALISATION
# ============================================================

def normalise_race(value) -> str | None:
    if pd.isna(value):
        return None

    text = (
        str(value)
        .strip()
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )

    text = " ".join(
        text.split()
    )

    mapping = {
        "white": "White",
        "black": "Black",

        "latino hispanic": "Latino_Hispanic",
        "latino/hispanic": "Latino_Hispanic",
        "latino": "Latino_Hispanic",
        "hispanic": "Latino_Hispanic",

        "east asian": "East Asian",
        "east": "East Asian",

        "southeast asian": "Southeast Asian",
        "southeast": "Southeast Asian",

        "indian": "Indian",

        "middle eastern": "Middle Eastern",
        "middle east": "Middle Eastern",
    }

    return mapping.get(text)


def normalise_gender(value) -> str | None:
    if pd.isna(value):
        return None

    text = str(
        value
    ).strip().lower()

    mapping = {
        "male": "Male",
        "m": "Male",
        "female": "Female",
        "f": "Female",
    }

    return mapping.get(text)


# ============================================================
# LOAD FAIRFACE VALIDATION SET
# ============================================================

def load_fairface_validation(
    master_path: Path,
    limit: int | None,
) -> pd.DataFrame:

    master = pd.read_parquet(
        master_path
    )

    dataset_text = (
        master["dataset"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    fairface = master[
        dataset_text.eq(
            "fairface"
        )
    ].copy()

    if fairface.empty:
        raise RuntimeError(
            "No FairFace rows found in the master table."
        )

    split_text = (
        fairface["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    path_text = (
        fairface["original_path"]
        .astype(str)
        .str.replace(
            "\\",
            "/",
            regex=False,
        )
        .str.lower()
    )

    validation_mask = (
        split_text.isin(
            {
                "val",
                "validation",
                "valid",
            }
        )
        |
        path_text.str.contains(
            "/val/",
            regex=False,
        )
    )

    frame = fairface[
        validation_mask
    ].copy()

    if frame.empty:
        print(
            "Available FairFace split values:"
        )

        print(
            fairface["split"]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

        raise RuntimeError(
            "Could not identify the FairFace validation split."
        )

    frame["race_true"] = (
        frame["race"]
        .map(
            normalise_race
        )
    )

    frame["gender_true"] = (
        frame["gender"]
        .map(
            normalise_gender
        )
    )

    bad_race = frame[
        frame["race_true"].isna()
    ]["race"].value_counts(
        dropna=False
    )

    bad_gender = frame[
        frame["gender_true"].isna()
    ]["gender"].value_counts(
        dropna=False
    )

    if not bad_race.empty:
        print()
        print(
            "Unmapped race labels:"
        )
        print(
            bad_race.to_string()
        )

    if not bad_gender.empty:
        print()
        print(
            "Unmapped gender labels:"
        )
        print(
            bad_gender.to_string()
        )

    frame = frame[
        frame["race_true"].notna()
        &
        frame["gender_true"].notna()
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
                "--limit must be greater than zero."
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

class FairFaceAuditDataset(Dataset):

    def __init__(
        self,
        frame: pd.DataFrame,
    ) -> None:

        self.frame = (
            frame.reset_index(
                drop=True
            )
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
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

    def __len__(self) -> int:
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

        image_path = paths.data_abs(
            str(
                row["original_path"]
            )
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            tensor = self.transform(
                image
            )

        return (
            tensor,
            str(
                row["image_id"]
            ),
        )


# ============================================================
# LOAD FAIRFACE MODEL
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

        clean_key = str(key)

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

    state_dict = load_checkpoint(
        checkpoint_path
    )

    model.load_state_dict(
        state_dict,
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
                    race_indices[index]
                )

                gender_index = int(
                    gender_indices[index]
                )

                age_index = int(
                    age_indices[index]
                )

                record = {
                    "image_id": image_id,

                    "race_pred": (
                        RACE_LABELS[
                            race_index
                        ]
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
                processed / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining = (
                total - processed
            )

            eta_seconds = (
                remaining / rate
                if rate > 0
                else 0.0
            )

            if (
                processed == total
                or processed % 500
                < len(image_ids)
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
# OVERALL METRICS
# ============================================================

def build_overall_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for task, labels in [
        (
            "race",
            RACE_LABELS,
        ),
        (
            "gender",
            GENDER_LABELS,
        ),
    ]:

        y_true = predictions[
            f"{task}_true"
        ]

        y_pred = predictions[
            f"{task}_pred"
        ]

        rows.append(
            {
                "task": task,

                "n": len(
                    predictions
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
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def build_classification_report(
    y_true,
    y_pred,
    labels,
) -> pd.DataFrame:

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return pd.DataFrame(
        report
    ).T


# ============================================================
# INTERSECTIONAL METRICS
# ============================================================

def build_intersectional_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    frame = predictions.copy()

    frame[
        "race_correct"
    ] = (
        frame["race_true"]
        == frame["race_pred"]
    )

    frame[
        "gender_correct"
    ] = (
        frame["gender_true"]
        == frame["gender_pred"]
    )

    frame[
        "both_correct"
    ] = (
        frame["race_correct"]
        & frame["gender_correct"]
    )

    result = (
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

    result[
        "reporting_status"
    ] = np.select(
        [
            result["n"] < 30,
            result["n"] < 100,
        ],
        [
            "suppressed",
            "low_confidence",
        ],
        default="reportable",
    )

    return result


# ============================================================
# SPECIFIC INTERSECTIONAL COMPARISONS
# ============================================================

def build_key_comparisons(
    intersectional: pd.DataFrame,
) -> pd.DataFrame:

    requested = [
        (
            "White Male",
            "Black Female",
        ),
        (
            "Black Male",
            "White Male",
        ),
        (
            "White Female",
            "Black Female",
        ),
    ]

    def get_group(
        name: str,
    ):

        race, gender = (
            name.rsplit(
                " ",
                1,
            )
        )

        group = intersectional[
            intersectional[
                "race_true"
            ].eq(
                race
            )
            &
            intersectional[
                "gender_true"
            ].eq(
                gender
            )
        ]

        if group.empty:
            return None

        return group.iloc[0]

    rows = []

    for group_a_name, group_b_name in requested:

        a = get_group(
            group_a_name
        )

        b = get_group(
            group_b_name
        )

        if (
            a is None
            or b is None
        ):
            continue

        rows.append(
            {
                "group_a": (
                    group_a_name
                ),

                "group_b": (
                    group_b_name
                ),

                "group_a_n": int(
                    a["n"]
                ),

                "group_b_n": int(
                    b["n"]
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
# GENDER FAIRNESS ACROSS RACE
# ============================================================

def build_gender_fairness_by_race(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    rows = []

    for race in RACE_LABELS:

        frame = predictions[
            predictions[
                "race_true"
            ].eq(
                race
            )
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
                & predicted_female
            ).sum()
        )

        fn = int(
            (
                actual_female
                & ~predicted_female
            ).sum()
        )

        fp = int(
            (
                actual_male
                & predicted_female
            ).sum()
        )

        tn = int(
            (
                actual_male
                & ~predicted_female
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

        fnr = (
            fn / female_n
            if female_n > 0
            else np.nan
        )

        fpr = (
            fp / male_n
            if male_n > 0
            else np.nan
        )

        tnr = (
            tn / male_n
            if male_n > 0
            else np.nan
        )

        rows.append(
            {
                "race": race,

                "n": len(
                    frame
                ),

                "n_female": female_n,
                "n_male": male_n,

                "true_positive": tp,
                "false_negative": fn,
                "false_positive": fp,
                "true_negative": tn,

                "female_tpr": tpr,
                "female_fnr": fnr,
                "female_fpr": fpr,
                "female_tnr": tnr,
            }
        )

    rates = pd.DataFrame(
        rows
    )

    tpr_values = (
        rates[
            "female_tpr"
        ].dropna()
    )

    fpr_values = (
        rates[
            "female_fpr"
        ].dropna()
    )

    tpr_difference = float(
        tpr_values.max()
        - tpr_values.min()
    )

    fpr_difference = float(
        fpr_values.max()
        - fpr_values.min()
    )

    equalized_odds_difference = float(
        max(
            tpr_difference,
            fpr_difference,
        )
    )

    summary = {
        "task": (
            "gender classification"
        ),

        "positive_class": (
            "Female"
        ),

        "sensitive_feature": (
            "actual race"
        ),

        "tpr_difference": (
            tpr_difference
        ),

        "fpr_difference": (
            fpr_difference
        ),

        "equalized_odds_difference": (
            equalized_odds_difference
        ),
    }

    return (
        rates,
        summary,
    )


# ============================================================
# RACE ONE-VS-REST FAIRNESS BY GENDER
# ============================================================

def build_race_fairness_by_gender(
    predictions: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for target_race in RACE_LABELS:

        group_results = {}

        for gender in GENDER_LABELS:

            frame = predictions[
                predictions[
                    "gender_true"
                ].eq(
                    gender
                )
            ]

            actual_target = (
                frame[
                    "race_true"
                ].eq(
                    target_race
                )
            )

            predicted_target = (
                frame[
                    "race_pred"
                ].eq(
                    target_race
                )
            )

            tp = int(
                (
                    actual_target
                    & predicted_target
                ).sum()
            )

            fn = int(
                (
                    actual_target
                    & ~predicted_target
                ).sum()
            )

            fp = int(
                (
                    ~actual_target
                    & predicted_target
                ).sum()
            )

            tn = int(
                (
                    ~actual_target
                    & ~predicted_target
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

            group_results[
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

        male = group_results[
            "Male"
        ]

        female = group_results[
            "Female"
        ]

        tpr_difference = abs(
            male["tpr"]
            - female["tpr"]
        )

        fpr_difference = abs(
            male["fpr"]
            - female["fpr"]
        )

        eo_difference = max(
            tpr_difference,
            fpr_difference,
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
                    male["tpr"]
                ),

                "female_tpr": (
                    female["tpr"]
                ),

                "tpr_difference": (
                    tpr_difference
                ),

                "male_fpr": (
                    male["fpr"]
                ),

                "female_fpr": (
                    female["fpr"]
                ),

                "fpr_difference": (
                    fpr_difference
                ),

                "equalized_odds_difference": (
                    eo_difference
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
    y_true,
    y_pred,
    labels,
    title: str,
    csv_path: Path,
    png_path: Path,
) -> None:

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    matrix_df.index.name = (
        "actual"
    )

    matrix_df.columns.name = (
        "predicted"
    )

    matrix_df.to_csv(
        csv_path
    )

    figure_size = (
        (10, 8)
        if len(labels) > 2
        else (6, 5)
    )

    fig, ax = plt.subplots(
        figsize=figure_size
    )

    displayed = ax.imshow(
        matrix
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "Actual label"
    )

    ax.set_xticks(
        range(
            len(labels)
        )
    )

    ax.set_yticks(
        range(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        labels
    )

    maximum = (
        matrix.max()
        if matrix.size
        else 0
    )

    threshold = (
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
                str(value),
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    fig.colorbar(
        displayed,
        ax=ax,
        fraction=0.046,
        pad=0.04,
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


def save_normalised_confusion_matrix(
    y_true,
    y_pred,
    labels,
    output_path: Path,
) -> None:

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
    )

    frame = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    frame.index.name = (
        "actual"
    )

    frame.columns.name = (
        "predicted"
    )

    frame.to_csv(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Audit the pretrained FairFace ResNet-34 "
            "race and gender classifier."
        )
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=DEFAULT_MASTER,
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

    if not args.master.exists():
        raise FileNotFoundError(
            f"Master table not found: "
            f"{args.master}"
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{args.checkpoint}"
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
        "FairFace ResNet-34 Model Audit"
    )

    print(
        "================================"
    )

    print(
        f"Checkpoint: "
        f"{args.checkpoint}"
    )

    print(
        f"Device:     "
        f"{device}"
    )

    print(
        f"Hardware:   "
        f"{hardware}"
    )

    print(
        f"Batch size: "
        f"{batch_size}"
    )

    print()

    frame = load_fairface_validation(
        master_path=(
            args.master
        ),
        limit=(
            args.limit
        ),
    )

    print(
        f"Validation images: "
        f"{len(frame):,}"
    )

    print()
    print(
        "Race ground truth:"
    )

    print(
        frame[
            "race_true"
        ]
        .value_counts()
        .reindex(
            RACE_LABELS,
            fill_value=0,
        )
        .to_string()
    )

    print()
    print(
        "Gender ground truth:"
    )

    print(
        frame[
            "gender_true"
        ]
        .value_counts()
        .reindex(
            GENDER_LABELS,
            fill_value=0,
        )
        .to_string()
    )

    dataset = FairFaceAuditDataset(
        frame
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
        checkpoint_path=(
            args.checkpoint
        ),
        device=device,
    )

    print(
        "Model loaded successfully."
    )

    print()
    print(
        "Running inference..."
    )

    inference_results, elapsed = (
        run_inference(
            model=model,
            loader=loader,
            device=device,
        )
    )

    predictions = frame.merge(
        inference_results,
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
            f"{missing} images did not receive predictions."
        )

    predictions[
        "race_correct"
    ] = (
        predictions[
            "race_true"
        ]
        == predictions[
            "race_pred"
        ]
    )

    predictions[
        "gender_correct"
    ] = (
        predictions[
            "gender_true"
        ]
        == predictions[
            "gender_pred"
        ]
    )

    predictions[
        "both_correct"
    ] = (
        predictions[
            "race_correct"
        ]
        &
        predictions[
            "gender_correct"
        ]
    )

    # --------------------------------------------------------
    # SAVE RAW PREDICTIONS
    # --------------------------------------------------------

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
    # CLASS REPORTS
    # --------------------------------------------------------

    race_report = (
        build_classification_report(
            predictions[
                "race_true"
            ],
            predictions[
                "race_pred"
            ],
            RACE_LABELS,
        )
    )

    race_report.to_csv(
        args.output
        / "race_classification_report.csv"
    )

    gender_report = (
        build_classification_report(
            predictions[
                "gender_true"
            ],
            predictions[
                "gender_pred"
            ],
            GENDER_LABELS,
        )
    )

    gender_report.to_csv(
        args.output
        / "gender_classification_report.csv"
    )

    # --------------------------------------------------------
    # INTERSECTIONAL METRICS
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
    # GENDER FAIRNESS BY RACE
    # --------------------------------------------------------

    (
        gender_fairness,
        gender_eo_summary,
    ) = (
        build_gender_fairness_by_race(
            predictions
        )
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
            gender_eo_summary,
            handle,
            indent=2,
        )

    # --------------------------------------------------------
    # RACE FAIRNESS BY GENDER
    # --------------------------------------------------------

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
    # CONFUSION MATRICES
    # --------------------------------------------------------

    save_confusion_matrix(
        predictions[
            "race_true"
        ],
        predictions[
            "race_pred"
        ],
        RACE_LABELS,
        (
            "FairFace ResNet-34: "
            "Race Confusion Matrix"
        ),
        args.output
        / "race_confusion_matrix.csv",
        args.output
        / "race_confusion_matrix.png",
    )

    save_normalised_confusion_matrix(
        predictions[
            "race_true"
        ],
        predictions[
            "race_pred"
        ],
        RACE_LABELS,
        args.output
        / "race_confusion_matrix_normalised.csv",
    )

    save_confusion_matrix(
        predictions[
            "gender_true"
        ],
        predictions[
            "gender_pred"
        ],
        GENDER_LABELS,
        (
            "FairFace ResNet-34: "
            "Gender Confusion Matrix"
        ),
        args.output
        / "gender_confusion_matrix.csv",
        args.output
        / "gender_confusion_matrix.png",
    )

    save_normalised_confusion_matrix(
        predictions[
            "gender_true"
        ],
        predictions[
            "gender_pred"
        ],
        GENDER_LABELS,
        args.output
        / "gender_confusion_matrix_normalised.csv",
    )

    # --------------------------------------------------------
    # RUNTIME
    # --------------------------------------------------------

    throughput = (
        len(predictions)
        / elapsed
        if elapsed > 0
        else np.nan
    )

    runtime = {
        "architecture": (
            "FairFace ResNet-34"
        ),

        "checkpoint": str(
            args.checkpoint
        ),

        "checkpoint_sha256": (
            "37F2E74C0E7DA196F8EDD7DAE0C88CFF"
            "DBEF7923BEEBEBCC27381F0AD0FDB2C1"
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

        "n_images": int(
            len(predictions)
        ),

        "elapsed_seconds": float(
            elapsed
        ),

        "images_per_second": float(
            throughput
        ),

        "seed": (
            SEED
        ),

        "race_classes": (
            RACE_LABELS
        ),

        "gender_classes": (
            GENDER_LABELS
        ),

        "age_classes": (
            AGE_LABELS
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
    # CONSOLE RESULTS
    # --------------------------------------------------------

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
        f"Elapsed:          "
        f"{elapsed:.2f} seconds"
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
        "REQUESTED WHITE/BLACK COMPARISONS"
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
        "GENDER EQUALIZED-ODDS SUMMARY"
    )

    print(
        json.dumps(
            gender_eo_summary,
            indent=2,
        )
    )

    print()
    print(
        "RACE ONE-VS-REST FAIRNESS BY GENDER"
    )

    print(
        race_fairness.to_string(
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

    