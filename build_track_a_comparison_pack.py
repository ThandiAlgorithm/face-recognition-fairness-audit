from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

MASTER = Path(
    "metadata/master_annotations.parquet"
)

FF_FAIRFACE = Path(
    "results/model_audit/fairface_resnet34/"
    "fairface_validation/predictions.csv"
)

DF_FAIRFACE = Path(
    "results/model_audit/deepface/"
    "fairface_validation/predictions.csv"
)

FF_UTK = Path(
    "results/model_audit/fairface_resnet34/"
    "utkface_aligned/predictions.csv"
)

DF_UTK = Path(
    "results/model_audit/deepface/"
    "utkface_aligned/predictions.csv"
)

FF_LFW = Path(
    "results/model_audit/fairface_resnet34/"
    "lfw_predictions/predictions.csv"
)

DF_LFW = Path(
    "results/model_audit/deepface/"
    "lfw_aligned/predictions.csv"
)

OUTPUT = Path(
    "results/model_audit/model_comparison/track_a_final"
)


# ============================================================
# CONSTANTS
# ============================================================

RACE_6 = [
    "White",
    "Black",
    "Asian",
    "Indian",
    "Latino_Hispanic",
    "Middle Eastern",
]

RACE_4 = [
    "White",
    "Black",
    "Asian",
    "Indian",
]

UTK_RACES = [
    "White",
    "Black",
    "Asian",
    "Indian",
    "Others",
]

GENDERS = [
    "Male",
    "Female",
]

MODELS = [
    "FairFace ResNet-34",
    "DeepFace",
]


# ============================================================
# COLUMN DISCOVERY
# ============================================================

RACE_PREDICTION_CANDIDATES = [
    "race_pred_harmonised",
    "race_pred",
    "predicted_race",
    "pred_race",
    "race_prediction",
    "race_predicted",
    "pred_race_label",
    "prediction_race",
]

GENDER_PREDICTION_CANDIDATES = [
    "gender_pred",
    "predicted_gender",
    "pred_gender",
    "gender_prediction",
    "gender_predicted",
    "pred_gender_label",
    "prediction_gender",
]


def find_column(
    frame: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:

    for column in candidates:

        if column in frame.columns:

            return column

    raise RuntimeError(
        f"Could not identify {description}. "
        f"Available columns are:\n"
        f"{list(frame.columns)}"
    )


# ============================================================
# NORMALISATION
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
        "man": "Male",
        "m": "Male",
        "0": "Male",

        "female": "Female",
        "woman": "Female",
        "f": "Female",
        "1": "Female",
    }

    return mapping.get(
        text
    )


def normalise_race_6(
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

        "asian": "Asian",
        "east asian": "Asian",
        "southeast asian": "Asian",
        "south east asian": "Asian",

        "indian": "Indian",

        "latino": "Latino_Hispanic",
        "hispanic": "Latino_Hispanic",
        "latino hispanic": "Latino_Hispanic",

        "middle eastern": "Middle Eastern",
    }

    return mapping.get(
        text
    )


def normalise_race_4_prediction(
    value,
) -> str | None:

    race = normalise_race_6(
        value
    )

    if race is None:

        text = str(
            value
        ).strip().lower()

        if text in {
            "out_of_scope",
            "out of scope",
        }:

            return "Out_of_scope"

        return None

    if race in RACE_4:

        return race

    return "Out_of_scope"


def normalise_utk_truth(
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


# ============================================================
# LOADERS
# ============================================================

def load_master() -> pd.DataFrame:

    master = pd.read_parquet(
        MASTER
    )

    required = [
        "image_id",
        "dataset",
        "subject_id",
        "race",
        "gender",
    ]

    missing = [
        column
        for column in required
        if column not in master.columns
    ]

    if missing:

        raise RuntimeError(
            f"Master table is missing: {missing}"
        )

    return master


def load_prediction_file(
    path: Path,
    model: str,
    dataset: str,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing prediction file:\n{path}"
        )

    frame = pd.read_csv(
        path
    )

    if "image_id" not in frame.columns:

        raise RuntimeError(
            f"{path} contains no image_id column."
        )

    race_column = find_column(
        frame,
        RACE_PREDICTION_CANDIDATES,
        f"{model} {dataset} race prediction column",
    )

    gender_column = find_column(
        frame,
        GENDER_PREDICTION_CANDIDATES,
        f"{model} {dataset} gender prediction column",
    )

    output = frame[
        [
            "image_id",
            race_column,
            gender_column,
        ]
    ].copy()

    output = output.rename(
        columns={
            race_column: "race_pred_raw",
            gender_column: "gender_pred_raw",
        }
    )

    if output[
        "image_id"
    ].duplicated().any():

        raise RuntimeError(
            f"Duplicate image_id values found in "
            f"{model} {dataset} predictions."
        )

    print(
        f"{model:20s} | "
        f"{dataset:10s} | "
        f"{len(output):,} predictions | "
        f"race={race_column} | "
        f"gender={gender_column}"
    )

    return output


# ============================================================
# MERGE TWO MODELS WITH CANONICAL METADATA
# ============================================================

def merge_dataset(
    master: pd.DataFrame,
    dataset_name: str,
    fairface_path: Path,
    deepface_path: Path,
    expected_n: int,
) -> pd.DataFrame:

    dataset_master = master[
        master[
            "dataset"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(
            dataset_name.lower()
        )
    ][
        [
            "image_id",
            "subject_id",
            "race",
            "gender",
        ]
    ].copy()

    fairface = load_prediction_file(
        fairface_path,
        "FairFace ResNet-34",
        dataset_name,
    ).rename(
        columns={
            "race_pred_raw": (
                "fairface_race_raw"
            ),
            "gender_pred_raw": (
                "fairface_gender_raw"
            ),
        }
    )

    deepface = load_prediction_file(
        deepface_path,
        "DeepFace",
        dataset_name,
    ).rename(
        columns={
            "race_pred_raw": (
                "deepface_race_raw"
            ),
            "gender_pred_raw": (
                "deepface_gender_raw"
            ),
        }
    )

    merged = (
        dataset_master
        .merge(
            fairface,
            on="image_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            deepface,
            on="image_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(
        merged
    ) != expected_n:

        raise RuntimeError(
            f"{dataset_name}: expected "
            f"{expected_n:,} common images but found "
            f"{len(merged):,}."
        )

    return merged


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def classification_metrics(
    dataset: str,
    model: str,
    task: str,
    truth: pd.Series,
    prediction: pd.Series,
    labels: list[str],
) -> dict:

    return {
        "dataset": dataset,
        "model": model,
        "task": task,

        "n": int(
            len(
                truth
            )
        ),

        "accuracy": float(
            accuracy_score(
                truth,
                prediction,
            )
        ),

        "precision_macro": float(
            precision_score(
                truth,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),

        "recall_macro": float(
            recall_score(
                truth,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_macro": float(
            f1_score(
                truth,
                prediction,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_weighted": float(
            f1_score(
                truth,
                prediction,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


# ============================================================
# GENDER EQUALIZED ODDS
# ============================================================

def gender_equalized_odds(
    frame: pd.DataFrame,
    dataset: str,
    model: str,
    prediction_column: str,
    race_column: str,
    race_groups: list[str],
) -> tuple[
    pd.DataFrame,
    dict,
]:

    rows = []

    for race in race_groups:

        group = frame[
            frame[
                race_column
            ].eq(
                race
            )
        ].copy()

        if group.empty:

            continue

        actual_female = (
            group[
                "gender_true"
            ].eq(
                "Female"
            )
        )

        actual_male = (
            group[
                "gender_true"
            ].eq(
                "Male"
            )
        )

        predicted_female = (
            group[
                prediction_column
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

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "race": race,

                "n": int(
                    len(
                        group
                    )
                ),

                "n_female": int(
                    positive_n
                ),

                "n_male": int(
                    negative_n
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
        "dataset": dataset,
        "model": model,

        "positive_class": (
            "Female"
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
# FAIRFACE VALIDATION EVALUATION
# ============================================================

def evaluate_fairface(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    frame = frame.copy()

    frame[
        "race_true"
    ] = frame[
        "race"
    ].map(
        normalise_race_6
    )

    frame[
        "gender_true"
    ] = frame[
        "gender"
    ].map(
        normalise_gender
    )

    frame[
        "fairface_race_pred"
    ] = frame[
        "fairface_race_raw"
    ].map(
        normalise_race_6
    )

    frame[
        "deepface_race_pred"
    ] = frame[
        "deepface_race_raw"
    ].map(
        normalise_race_6
    )

    frame[
        "fairface_gender_pred"
    ] = frame[
        "fairface_gender_raw"
    ].map(
        normalise_gender
    )

    frame[
        "deepface_gender_pred"
    ] = frame[
        "deepface_gender_raw"
    ].map(
        normalise_gender
    )

    required = [
        "race_true",
        "gender_true",
        "fairface_race_pred",
        "deepface_race_pred",
        "fairface_gender_pred",
        "deepface_gender_pred",
    ]

    if frame[
        required
    ].isna().any().any():

        problem = frame[
            required
        ].isna().sum()

        raise RuntimeError(
            "FairFace normalization produced missing "
            f"values:\n{problem}"
        )

    rows = []

    for model, race_column, gender_column in [
        (
            "FairFace ResNet-34",
            "fairface_race_pred",
            "fairface_gender_pred",
        ),
        (
            "DeepFace",
            "deepface_race_pred",
            "deepface_gender_pred",
        ),
    ]:

        rows.append(
            classification_metrics(
                "FairFace validation",
                model,
                "race_6class",
                frame[
                    "race_true"
                ],
                frame[
                    race_column
                ],
                RACE_6,
            )
        )

        rows.append(
            classification_metrics(
                "FairFace validation",
                model,
                "gender",
                frame[
                    "gender_true"
                ],
                frame[
                    gender_column
                ],
                GENDERS,
            )
        )

    overall = pd.DataFrame(
        rows
    )

    fairness_tables = []

    eo_rows = []

    for model, prediction_column in [
        (
            "FairFace ResNet-34",
            "fairface_gender_pred",
        ),
        (
            "DeepFace",
            "deepface_gender_pred",
        ),
    ]:

        rates, summary = (
            gender_equalized_odds(
                frame,
                "FairFace validation",
                model,
                prediction_column,
                "race_true",
                RACE_6,
            )
        )

        fairness_tables.append(
            rates
        )

        eo_rows.append(
            summary
        )

    fairness = pd.concat(
        fairness_tables,
        ignore_index=True,
    )

    eo = pd.DataFrame(
        eo_rows
    )

    return (
        overall,
        fairness,
        eo,
    )


# ============================================================
# UTKFACE EVALUATION
# ============================================================

def evaluate_utkface(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    frame = frame.copy()

    frame[
        "race_true_native"
    ] = frame[
        "race"
    ].map(
        normalise_utk_truth
    )

    frame[
        "gender_true"
    ] = frame[
        "gender"
    ].map(
        normalise_gender
    )

    frame[
        "fairface_race_pred"
    ] = frame[
        "fairface_race_raw"
    ].map(
        normalise_race_4_prediction
    )

    frame[
        "deepface_race_pred"
    ] = frame[
        "deepface_race_raw"
    ].map(
        normalise_race_4_prediction
    )

    frame[
        "fairface_gender_pred"
    ] = frame[
        "fairface_gender_raw"
    ].map(
        normalise_gender
    )

    frame[
        "deepface_gender_pred"
    ] = frame[
        "deepface_gender_raw"
    ].map(
        normalise_gender
    )

    gender_valid = frame[
        frame[
            "gender_true"
        ].notna()
    ].copy()

    race_valid = frame[
        frame[
            "race_true_native"
        ].isin(
            RACE_4
        )
    ].copy()

    if len(
        gender_valid
    ) != 24050:

        raise RuntimeError(
            "Expected 24,050 UTKFace rows with usable "
            f"gender labels; found {len(gender_valid):,}."
        )

    if len(
        race_valid
    ) != 22340:

        raise RuntimeError(
            "Expected 22,340 UTKFace race-comparable "
            f"rows; found {len(race_valid):,}."
        )

    rows = []

    for model, race_column, gender_column in [
        (
            "FairFace ResNet-34",
            "fairface_race_pred",
            "fairface_gender_pred",
        ),
        (
            "DeepFace",
            "deepface_race_pred",
            "deepface_gender_pred",
        ),
    ]:

        rows.append(
            classification_metrics(
                "UTKFace aligned",
                model,
                "race_4class",
                race_valid[
                    "race_true_native"
                ],
                race_valid[
                    race_column
                ],
                RACE_4,
            )
        )

        rows.append(
            classification_metrics(
                "UTKFace aligned",
                model,
                "gender",
                gender_valid[
                    "gender_true"
                ],
                gender_valid[
                    gender_column
                ],
                GENDERS,
            )
        )

    overall = pd.DataFrame(
        rows
    )

    fairness_tables = []

    eo_rows = []

    fairness_frame = gender_valid[
        gender_valid[
            "race_true_native"
        ].notna()
    ].copy()

    for model, prediction_column in [
        (
            "FairFace ResNet-34",
            "fairface_gender_pred",
        ),
        (
            "DeepFace",
            "deepface_gender_pred",
        ),
    ]:

        rates, summary = (
            gender_equalized_odds(
                fairness_frame,
                "UTKFace aligned",
                model,
                prediction_column,
                "race_true_native",
                UTK_RACES,
            )
        )

        fairness_tables.append(
            rates
        )

        eo_rows.append(
            summary
        )

    fairness = pd.concat(
        fairness_tables,
        ignore_index=True,
    )

    eo = pd.DataFrame(
        eo_rows
    )

    return (
        overall,
        fairness,
        eo,
    )


# ============================================================
# LFW MODEL AGREEMENT
# ============================================================

def modal_value_and_consistency(
    series: pd.Series,
) -> tuple[
    str,
    float,
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

    modal = candidates[
        0
    ]

    consistency = (
        maximum
        /
        len(
            series
        )
    )

    return (
        modal,
        float(
            consistency
        ),
    )


def evaluate_lfw(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    frame = frame.copy()

    frame[
        "fairface_race_pred"
    ] = frame[
        "fairface_race_raw"
    ].map(
        normalise_race_6
    )

    frame[
        "deepface_race_pred"
    ] = frame[
        "deepface_race_raw"
    ].map(
        normalise_race_6
    )

    frame[
        "fairface_gender_pred"
    ] = frame[
        "fairface_gender_raw"
    ].map(
        normalise_gender
    )

    frame[
        "deepface_gender_pred"
    ] = frame[
        "deepface_gender_raw"
    ].map(
        normalise_gender
    )

    required = [
        "fairface_race_pred",
        "deepface_race_pred",
        "fairface_gender_pred",
        "deepface_gender_pred",
    ]

    if frame[
        required
    ].isna().any().any():

        raise RuntimeError(
            "LFW prediction normalization produced "
            "missing labels."
        )

    # --------------------------------------------------------
    # IMAGE-LEVEL AGREEMENT
    # --------------------------------------------------------

    race_agreement = (
        frame[
            "fairface_race_pred"
        ]
        ==
        frame[
            "deepface_race_pred"
        ]
    )

    gender_agreement = (
        frame[
            "fairface_gender_pred"
        ]
        ==
        frame[
            "deepface_gender_pred"
        ]
    )

    agreement = pd.DataFrame(
        [
            {
                "task": (
                    "race"
                ),

                "n": int(
                    len(
                        frame
                    )
                ),

                "agreement_rate": float(
                    race_agreement.mean()
                ),

                "disagreement_rate": float(
                    1
                    -
                    race_agreement.mean()
                ),

                "cohen_kappa": float(
                    cohen_kappa_score(
                        frame[
                            "fairface_race_pred"
                        ],
                        frame[
                            "deepface_race_pred"
                        ],
                        labels=RACE_6,
                    )
                ),
            },

            {
                "task": (
                    "gender"
                ),

                "n": int(
                    len(
                        frame
                    )
                ),

                "agreement_rate": float(
                    gender_agreement.mean()
                ),

                "disagreement_rate": float(
                    1
                    -
                    gender_agreement.mean()
                ),

                "cohen_kappa": float(
                    cohen_kappa_score(
                        frame[
                            "fairface_gender_pred"
                        ],
                        frame[
                            "deepface_gender_pred"
                        ],
                        labels=GENDERS,
                    )
                ),
            },
        ]
    )

    # --------------------------------------------------------
    # PREDICTED DISTRIBUTIONS
    # --------------------------------------------------------

    distribution_rows = []

    for model, race_column, gender_column in [
        (
            "FairFace ResNet-34",
            "fairface_race_pred",
            "fairface_gender_pred",
        ),
        (
            "DeepFace",
            "deepface_race_pred",
            "deepface_gender_pred",
        ),
    ]:

        for category in RACE_6:

            n = int(
                frame[
                    race_column
                ].eq(
                    category
                ).sum()
            )

            distribution_rows.append(
                {
                    "model": model,
                    "task": "race",
                    "category": category,
                    "n": n,

                    "proportion": float(
                        n
                        /
                        len(
                            frame
                        )
                    ),
                }
            )

        for category in GENDERS:

            n = int(
                frame[
                    gender_column
                ].eq(
                    category
                ).sum()
            )

            distribution_rows.append(
                {
                    "model": model,
                    "task": "gender",
                    "category": category,
                    "n": n,

                    "proportion": float(
                        n
                        /
                        len(
                            frame
                        )
                    ),
                }
            )

    distributions = pd.DataFrame(
        distribution_rows
    )

    # --------------------------------------------------------
    # TOTAL VARIATION DISTANCE BETWEEN MODEL DISTRIBUTIONS
    # --------------------------------------------------------

    distance_rows = []

    for task, categories in [
        (
            "race",
            RACE_6,
        ),
        (
            "gender",
            GENDERS,
        ),
    ]:

        ff = (
            distributions[
                (
                    distributions[
                        "model"
                    ].eq(
                        "FairFace ResNet-34"
                    )
                )
                &
                (
                    distributions[
                        "task"
                    ].eq(
                        task
                    )
                )
            ]
            .set_index(
                "category"
            )
            .reindex(
                categories
            )[
                "proportion"
            ]
        )

        df = (
            distributions[
                (
                    distributions[
                        "model"
                    ].eq(
                        "DeepFace"
                    )
                )
                &
                (
                    distributions[
                        "task"
                    ].eq(
                        task
                    )
                )
            ]
            .set_index(
                "category"
            )
            .reindex(
                categories
            )[
                "proportion"
            ]
        )

        tv = float(
            0.5
            *
            np.abs(
                ff.values
                -
                df.values
            ).sum()
        )

        distance_rows.append(
            {
                "task": task,
                "total_variation_distance": tv,
            }
        )

    distribution_distance = pd.DataFrame(
        distance_rows
    )

    # --------------------------------------------------------
    # CROSS-MODEL RACE MATRIX
    # --------------------------------------------------------

    race_matrix = pd.crosstab(
        frame[
            "fairface_race_pred"
        ],
        frame[
            "deepface_race_pred"
        ],
    )

    race_matrix = race_matrix.reindex(
        index=RACE_6,
        columns=RACE_6,
        fill_value=0,
    )

    race_matrix.index.name = (
        "fairface_resnet34_prediction"
    )

    race_matrix.columns.name = (
        "deepface_prediction"
    )

    # --------------------------------------------------------
    # REPEATED-IDENTITY CONSISTENCY
    # --------------------------------------------------------

    subject_counts = (
        frame[
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

    repeated = frame[
        frame[
            "subject_id"
        ].isin(
            repeated_subjects
        )
    ].copy()

    identity_rows = []

    for subject_id, group in repeated.groupby(
        "subject_id",
        observed=True,
    ):

        (
            ff_race_modal,
            ff_race_consistency,
        ) = modal_value_and_consistency(
            group[
                "fairface_race_pred"
            ]
        )

        (
            df_race_modal,
            df_race_consistency,
        ) = modal_value_and_consistency(
            group[
                "deepface_race_pred"
            ]
        )

        (
            ff_gender_modal,
            ff_gender_consistency,
        ) = modal_value_and_consistency(
            group[
                "fairface_gender_pred"
            ]
        )

        (
            df_gender_modal,
            df_gender_consistency,
        ) = modal_value_and_consistency(
            group[
                "deepface_gender_pred"
            ]
        )

        identity_rows.append(
            {
                "subject_id": (
                    subject_id
                ),

                "n_images": int(
                    len(
                        group
                    )
                ),

                "fairface_modal_race": (
                    ff_race_modal
                ),

                "deepface_modal_race": (
                    df_race_modal
                ),

                "race_modal_agreement": bool(
                    ff_race_modal
                    ==
                    df_race_modal
                ),

                "fairface_race_consistency": (
                    ff_race_consistency
                ),

                "deepface_race_consistency": (
                    df_race_consistency
                ),

                "fairface_modal_gender": (
                    ff_gender_modal
                ),

                "deepface_modal_gender": (
                    df_gender_modal
                ),

                "gender_modal_agreement": bool(
                    ff_gender_modal
                    ==
                    df_gender_modal
                ),

                "fairface_gender_consistency": (
                    ff_gender_consistency
                ),

                "deepface_gender_consistency": (
                    df_gender_consistency
                ),
            }
        )

    identity = pd.DataFrame(
        identity_rows
    )

    identity_summary = pd.DataFrame(
        [
            {
                "model": (
                    "FairFace ResNet-34"
                ),

                "repeated_identities": int(
                    len(
                        identity
                    )
                ),

                "images_in_repeated_identities": int(
                    len(
                        repeated
                    )
                ),

                "mean_race_consistency": float(
                    identity[
                        "fairface_race_consistency"
                    ].mean()
                ),

                "median_race_consistency": float(
                    identity[
                        "fairface_race_consistency"
                    ].median()
                ),

                "mean_gender_consistency": float(
                    identity[
                        "fairface_gender_consistency"
                    ].mean()
                ),

                "median_gender_consistency": float(
                    identity[
                        "fairface_gender_consistency"
                    ].median()
                ),
            },

            {
                "model": (
                    "DeepFace"
                ),

                "repeated_identities": int(
                    len(
                        identity
                    )
                ),

                "images_in_repeated_identities": int(
                    len(
                        repeated
                    )
                ),

                "mean_race_consistency": float(
                    identity[
                        "deepface_race_consistency"
                    ].mean()
                ),

                "median_race_consistency": float(
                    identity[
                        "deepface_race_consistency"
                    ].median()
                ),

                "mean_gender_consistency": float(
                    identity[
                        "deepface_gender_consistency"
                    ].mean()
                ),

                "median_gender_consistency": float(
                    identity[
                        "deepface_gender_consistency"
                    ].median()
                ),
            },
        ]
    )

    modal_summary = pd.DataFrame(
        [
            {
                "task": (
                    "race"
                ),

                "repeated_identities": int(
                    len(
                        identity
                    )
                ),

                "modal_label_agreement_rate": float(
                    identity[
                        "race_modal_agreement"
                    ].mean()
                ),
            },

            {
                "task": (
                    "gender"
                ),

                "repeated_identities": int(
                    len(
                        identity
                    )
                ),

                "modal_label_agreement_rate": float(
                    identity[
                        "gender_modal_agreement"
                    ].mean()
                ),
            },
        ]
    )

    return (
        agreement,
        distributions,
        distribution_distance,
        race_matrix,
        identity,
        identity_summary,
        modal_summary,
    )


# ============================================================
# FIGURES
# ============================================================

def save_performance_figure(
    overall: pd.DataFrame,
    task: str,
    metric: str,
    title: str,
    filename: str,
) -> None:

    frame = overall[
        overall[
            "task"
        ].eq(
            task
        )
    ].copy()

    datasets = [
        "FairFace validation",
        "UTKFace aligned",
    ]

    x = np.arange(
        len(
            datasets
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    for offset, model in zip(
        [
            -width / 2,
            width / 2,
        ],
        MODELS,
    ):

        values = []

        for dataset in datasets:

            value = frame[
                frame[
                    "dataset"
                ].eq(
                    dataset
                )
                &
                frame[
                    "model"
                ].eq(
                    model
                )
            ][
                metric
            ].iloc[
                0
            ]

            values.append(
                value
            )

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model,
        )

        for bar, value in zip(
            bars,
            values,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                value + 0.012,
                f"{value:.1%}",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        datasets
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        metric.replace(
            "_",
            " "
        ).title()
    )

    ax.set_title(
        title
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / filename,
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_eo_figure(
    eo: pd.DataFrame,
) -> None:

    datasets = [
        "FairFace validation",
        "UTKFace aligned",
    ]

    x = np.arange(
        len(
            datasets
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(
            9,
            6,
        )
    )

    for offset, model in zip(
        [
            -width / 2,
            width / 2,
        ],
        MODELS,
    ):

        values = []

        for dataset in datasets:

            value = eo[
                eo[
                    "dataset"
                ].eq(
                    dataset
                )
                &
                eo[
                    "model"
                ].eq(
                    model
                )
            ][
                "equalized_odds_difference"
            ].iloc[
                0
            ]

            values.append(
                value
            )

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model,
        )

        for bar, value in zip(
            bars,
            values,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                value + 0.008,
                f"{value * 100:.1f} pp",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        datasets
    )

    ax.set_ylabel(
        "Equalized-odds difference"
    )

    ax.set_title(
        "Gender Equalized-Odds Difference by Dataset and Model"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "03_gender_equalized_odds_comparison.png",
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_lfw_race_distribution(
    distributions: pd.DataFrame,
) -> None:

    frame = distributions[
        distributions[
            "task"
        ].eq(
            "race"
        )
    ].copy()

    x = np.arange(
        len(
            RACE_6
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )

    for offset, model in zip(
        [
            -width / 2,
            width / 2,
        ],
        MODELS,
    ):

        values = (
            frame[
                frame[
                    "model"
                ].eq(
                    model
                )
            ]
            .set_index(
                "category"
            )
            .reindex(
                RACE_6
            )[
                "proportion"
            ]
            .values
        )

        ax.bar(
            x + offset,
            values,
            width,
            label=model,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        RACE_6,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel(
        "Proportion of LFW images"
    )

    ax.set_title(
        "Model-Assigned Race Distribution on LFW"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "04_lfw_race_distribution_comparison.png",
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_lfw_gender_distribution(
    distributions: pd.DataFrame,
) -> None:

    frame = distributions[
        distributions[
            "task"
        ].eq(
            "gender"
        )
    ].copy()

    x = np.arange(
        len(
            GENDERS
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    for offset, model in zip(
        [
            -width / 2,
            width / 2,
        ],
        MODELS,
    ):

        values = (
            frame[
                frame[
                    "model"
                ].eq(
                    model
                )
            ]
            .set_index(
                "category"
            )
            .reindex(
                GENDERS
            )[
                "proportion"
            ]
            .values
        )

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model,
        )

        for bar, value in zip(
            bars,
            values,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                value + 0.01,
                f"{value:.1%}",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        GENDERS
    )

    ax.set_ylabel(
        "Proportion of LFW images"
    )

    ax.set_title(
        "Model-Assigned Gender Distribution on LFW"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "05_lfw_gender_distribution_comparison.png",
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_lfw_agreement_figure(
    agreement: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(
            7,
            5,
        )
    )

    bars = ax.bar(
        agreement[
            "task"
        ].str.title(),
        agreement[
            "agreement_rate"
        ],
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Prediction agreement rate"
    )

    ax.set_title(
        "FairFace ResNet-34 vs DeepFace Agreement on LFW"
    )

    for bar, value in zip(
        bars,
        agreement[
            "agreement_rate"
        ],
    ):

        ax.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            value + 0.015,
            f"{value:.1%}",
            ha="center",
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "06_lfw_model_agreement.png",
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_lfw_identity_consistency_figure(
    summary: pd.DataFrame,
) -> None:

    x = np.arange(
        2
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    for offset, model in zip(
        [
            -width / 2,
            width / 2,
        ],
        MODELS,
    ):

        row = summary[
            summary[
                "model"
            ].eq(
                model
            )
        ].iloc[
            0
        ]

        values = [
            row[
                "mean_race_consistency"
            ],
            row[
                "mean_gender_consistency"
            ],
        ]

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model,
        )

        for bar, value in zip(
            bars,
            values,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                value + 0.01,
                f"{value:.1%}",
                ha="center",
                fontsize=8,
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Race",
            "Gender",
        ]
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Mean repeated-identity consistency"
    )

    ax.set_title(
        "Repeated-Identity Demographic Stability on LFW"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "07_lfw_identity_consistency_comparison.png",
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def save_lfw_race_matrix(
    matrix: pd.DataFrame,
) -> None:

    values = matrix.values

    fig, ax = plt.subplots(
        figsize=(
            10,
            8,
        )
    )

    display = ax.imshow(
        values
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
        rotation=30,
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
        "DeepFace prediction"
    )

    ax.set_ylabel(
        "FairFace ResNet-34 prediction"
    )

    ax.set_title(
        "Cross-Model Race Assignments on LFW"
    )

    maximum = (
        values.max()
        if values.size
        else 0
    )

    midpoint = (
        maximum / 2
        if maximum
        else 0
    )

    for row in range(
        values.shape[
            0
        ]
    ):

        for column in range(
            values.shape[
                1
            ]
        ):

            value = int(
                values[
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
        OUTPUT
        / "08_lfw_cross_model_race_matrix.png",
        dpi=260,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# KEY FINDINGS TABLE
# ============================================================

def build_key_findings(
    overall: pd.DataFrame,
    eo: pd.DataFrame,
    lfw_agreement: pd.DataFrame,
    distribution_distance: pd.DataFrame,
    identity_summary: pd.DataFrame,
    modal_summary: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for dataset in [
        "FairFace validation",
        "UTKFace aligned",
    ]:

        for task in [
            "gender",
        ]:

            for model in MODELS:

                value = overall[
                    overall[
                        "dataset"
                    ].eq(
                        dataset
                    )
                    &
                    overall[
                        "model"
                    ].eq(
                        model
                    )
                    &
                    overall[
                        "task"
                    ].eq(
                        task
                    )
                ][
                    "accuracy"
                ].iloc[
                    0
                ]

                rows.append(
                    {
                        "domain": (
                            "classification performance"
                        ),

                        "dataset": (
                            dataset
                        ),

                        "measure": (
                            f"{model} gender accuracy"
                        ),

                        "value": float(
                            value
                        ),
                    }
                )

        for model in MODELS:

            value = eo[
                eo[
                    "dataset"
                ].eq(
                    dataset
                )
                &
                eo[
                    "model"
                ].eq(
                    model
                )
            ][
                "equalized_odds_difference"
            ].iloc[
                0
            ]

            rows.append(
                {
                    "domain": (
                        "fairness"
                    ),

                    "dataset": (
                        dataset
                    ),

                    "measure": (
                        f"{model} gender EO difference"
                    ),

                    "value": float(
                        value
                    ),
                }
            )

    for _, row in lfw_agreement.iterrows():

        rows.append(
            {
                "domain": (
                    "measurement sensitivity"
                ),

                "dataset": (
                    "LFW"
                ),

                "measure": (
                    f"{row['task']} image-level "
                    "cross-model agreement"
                ),

                "value": float(
                    row[
                        "agreement_rate"
                    ]
                ),
            }
        )

    for _, row in distribution_distance.iterrows():

        rows.append(
            {
                "domain": (
                    "measurement sensitivity"
                ),

                "dataset": (
                    "LFW"
                ),

                "measure": (
                    f"{row['task']} distribution "
                    "total variation distance"
                ),

                "value": float(
                    row[
                        "total_variation_distance"
                    ]
                ),
            }
        )

    for _, row in modal_summary.iterrows():

        rows.append(
            {
                "domain": (
                    "measurement stability"
                ),

                "dataset": (
                    "LFW"
                ),

                "measure": (
                    f"{row['task']} repeated-identity "
                    "modal cross-model agreement"
                ),

                "value": float(
                    row[
                        "modal_label_agreement_rate"
                    ]
                ),
            }
        )

    for _, row in identity_summary.iterrows():

        rows.append(
            {
                "domain": (
                    "measurement stability"
                ),

                "dataset": (
                    "LFW"
                ),

                "measure": (
                    f"{row['model']} mean "
                    "race consistency"
                ),

                "value": float(
                    row[
                        "mean_race_consistency"
                    ]
                ),
            }
        )

        rows.append(
            {
                "domain": (
                    "measurement stability"
                ),

                "dataset": (
                    "LFW"
                ),

                "measure": (
                    f"{row['model']} mean "
                    "gender consistency"
                ),

                "value": float(
                    row[
                        "mean_gender_consistency"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MARKDOWN SUMMARY
# ============================================================

def write_summary(
    overall: pd.DataFrame,
    eo: pd.DataFrame,
    agreement: pd.DataFrame,
    distribution_distance: pd.DataFrame,
    identity_summary: pd.DataFrame,
    modal_summary: pd.DataFrame,
) -> None:

    def metric(
        dataset: str,
        model: str,
        task: str,
        column: str,
    ) -> float:

        return float(
            overall[
                overall[
                    "dataset"
                ].eq(
                    dataset
                )
                &
                overall[
                    "model"
                ].eq(
                    model
                )
                &
                overall[
                    "task"
                ].eq(
                    task
                )
            ][
                column
            ].iloc[
                0
            ]
        )

    def eo_value(
        dataset: str,
        model: str,
    ) -> float:

        return float(
            eo[
                eo[
                    "dataset"
                ].eq(
                    dataset
                )
                &
                eo[
                    "model"
                ].eq(
                    model
                )
            ][
                "equalized_odds_difference"
            ].iloc[
                0
            ]
        )

    lfw_race_agreement = float(
        agreement[
            agreement[
                "task"
            ].eq(
                "race"
            )
        ][
            "agreement_rate"
        ].iloc[
            0
        ]
    )

    lfw_gender_agreement = float(
        agreement[
            agreement[
                "task"
            ].eq(
                "gender"
            )
        ][
            "agreement_rate"
        ].iloc[
            0
        ]
    )

    race_tv = float(
        distribution_distance[
            distribution_distance[
                "task"
            ].eq(
                "race"
            )
        ][
            "total_variation_distance"
        ].iloc[
            0
        ]
    )

    gender_tv = float(
        distribution_distance[
            distribution_distance[
                "task"
            ].eq(
                "gender"
            )
        ][
            "total_variation_distance"
        ].iloc[
            0
        ]
    )

    race_modal = float(
        modal_summary[
            modal_summary[
                "task"
            ].eq(
                "race"
            )
        ][
            "modal_label_agreement_rate"
        ].iloc[
            0
        ]
    )

    gender_modal = float(
        modal_summary[
            modal_summary[
                "task"
            ].eq(
                "gender"
            )
        ][
            "modal_label_agreement_rate"
        ].iloc[
            0
        ]
    )

    lines = [
        "# Track A — Demographic Attribute Classification Audit",
        "",
        "## Evaluation design",
        "",
        (
            "FairFace ResNet-34 and DeepFace were evaluated "
            "on the same FairFace validation images and the "
            "same deterministically aligned UTKFace images. "
            "Race taxonomies were harmonised before direct "
            "comparison."
        ),
        "",
        (
            "LFW contains no independent demographic ground "
            "truth in the canonical project table. LFW results "
            "therefore describe model-assigned demographic "
            "labels, cross-model agreement, and repeated-identity "
            "stability rather than demographic accuracy."
        ),
        "",
        "## Labelled-dataset results",
        "",
        (
            f"- FairFace validation gender accuracy: "
            f"FairFace ResNet-34 "
            f"{metric('FairFace validation', 'FairFace ResNet-34', 'gender', 'accuracy'):.2%}; "
            f"DeepFace "
            f"{metric('FairFace validation', 'DeepFace', 'gender', 'accuracy'):.2%}."
        ),
        (
            f"- FairFace validation six-class race accuracy: "
            f"FairFace ResNet-34 "
            f"{metric('FairFace validation', 'FairFace ResNet-34', 'race_6class', 'accuracy'):.2%}; "
            f"DeepFace "
            f"{metric('FairFace validation', 'DeepFace', 'race_6class', 'accuracy'):.2%}."
        ),
        (
            f"- FairFace validation gender equalized-odds "
            f"difference: FairFace ResNet-34 "
            f"{eo_value('FairFace validation', 'FairFace ResNet-34') * 100:.2f} pp; "
            f"DeepFace "
            f"{eo_value('FairFace validation', 'DeepFace') * 100:.2f} pp."
        ),
        (
            f"- UTKFace gender accuracy: FairFace ResNet-34 "
            f"{metric('UTKFace aligned', 'FairFace ResNet-34', 'gender', 'accuracy'):.2%}; "
            f"DeepFace "
            f"{metric('UTKFace aligned', 'DeepFace', 'gender', 'accuracy'):.2%}."
        ),
        (
            f"- UTKFace four-class race accuracy: "
            f"FairFace ResNet-34 "
            f"{metric('UTKFace aligned', 'FairFace ResNet-34', 'race_4class', 'accuracy'):.2%}; "
            f"DeepFace "
            f"{metric('UTKFace aligned', 'DeepFace', 'race_4class', 'accuracy'):.2%}."
        ),
        (
            f"- UTKFace gender equalized-odds difference: "
            f"FairFace ResNet-34 "
            f"{eo_value('UTKFace aligned', 'FairFace ResNet-34') * 100:.2f} pp; "
            f"DeepFace "
            f"{eo_value('UTKFace aligned', 'DeepFace') * 100:.2f} pp."
        ),
        "",
        "## LFW measurement sensitivity",
        "",
        (
            f"- Image-level race-label agreement between the "
            f"two demographic classifiers was "
            f"{lfw_race_agreement:.2%}."
        ),
        (
            f"- Image-level gender-label agreement was "
            f"{lfw_gender_agreement:.2%}."
        ),
        (
            f"- Total-variation distance between the two "
            f"model-assigned LFW race distributions was "
            f"{race_tv:.4f}; for gender it was "
            f"{gender_tv:.4f}."
        ),
        (
            f"- Among repeated LFW identities, modal race "
            f"assignments agreed between models for "
            f"{race_modal:.2%} of identities, while modal "
            f"gender assignments agreed for "
            f"{gender_modal:.2%}."
        ),
        "",
        "## Interpretation boundary",
        "",
        (
            "Differences between model outputs demonstrate "
            "measurement sensitivity and performance disparity "
            "under the evaluated datasets and preprocessing "
            "conditions. They do not establish a causal source "
            "of bias, and model-assigned LFW demographic labels "
            "must not be treated as verified demographic ground "
            "truth."
        ),
        "",
    ]

    (
        OUTPUT
        / "track_a_summary.md"
    ).write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Track A Final Comparison Pack"
    )

    print(
        "=============================================="
    )

    master = load_master()

    print()
    print(
        "Loading prediction files..."
    )

    fairface = merge_dataset(
        master,
        "fairface",
        FF_FAIRFACE,
        DF_FAIRFACE,
        10954,
    )

    utkface = merge_dataset(
        master,
        "utkface",
        FF_UTK,
        DF_UTK,
        24054,
    )

    lfw = merge_dataset(
        master,
        "lfw",
        FF_LFW,
        DF_LFW,
        13233,
    )

    print()
    print(
        "Evaluating labelled datasets..."
    )

    (
        ff_overall,
        ff_fairness,
        ff_eo,
    ) = evaluate_fairface(
        fairface
    )

    (
        utk_overall,
        utk_fairness,
        utk_eo,
    ) = evaluate_utkface(
        utkface
    )

    overall = pd.concat(
        [
            ff_overall,
            utk_overall,
        ],
        ignore_index=True,
    )

    fairness = pd.concat(
        [
            ff_fairness,
            utk_fairness,
        ],
        ignore_index=True,
    )

    eo = pd.concat(
        [
            ff_eo,
            utk_eo,
        ],
        ignore_index=True,
    )

    print()
    print(
        "Evaluating LFW cross-model measurement sensitivity..."
    )

    (
        lfw_agreement,
        lfw_distributions,
        lfw_distribution_distance,
        lfw_race_matrix,
        lfw_identity,
        lfw_identity_summary,
        lfw_modal_summary,
    ) = evaluate_lfw(
        lfw
    )

    # --------------------------------------------------------
    # SAVE TABLES
    # --------------------------------------------------------

    overall.to_csv(
        OUTPUT
        / "01_labelled_dataset_model_metrics.csv",
        index=False,
    )

    eo.to_csv(
        OUTPUT
        / "02_gender_equalized_odds_summary.csv",
        index=False,
    )

    fairness.to_csv(
        OUTPUT
        / "03_gender_fairness_by_race.csv",
        index=False,
    )

    lfw_agreement.to_csv(
        OUTPUT
        / "04_lfw_cross_model_agreement.csv",
        index=False,
    )

    lfw_distributions.to_csv(
        OUTPUT
        / "05_lfw_model_assigned_distributions.csv",
        index=False,
    )

    lfw_distribution_distance.to_csv(
        OUTPUT
        / "06_lfw_distribution_distance.csv",
        index=False,
    )

    lfw_race_matrix.to_csv(
        OUTPUT
        / "07_lfw_cross_model_race_matrix.csv"
    )

    lfw_identity.to_csv(
        OUTPUT
        / "08_lfw_repeated_identity_model_comparison.csv",
        index=False,
    )

    lfw_identity_summary.to_csv(
        OUTPUT
        / "09_lfw_identity_consistency_summary.csv",
        index=False,
    )

    lfw_modal_summary.to_csv(
        OUTPUT
        / "10_lfw_identity_modal_agreement.csv",
        index=False,
    )

    key_findings = build_key_findings(
        overall,
        eo,
        lfw_agreement,
        lfw_distribution_distance,
        lfw_identity_summary,
        lfw_modal_summary,
    )

    key_findings.to_csv(
        OUTPUT
        / "11_track_a_key_findings.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    save_performance_figure(
        overall,
        "gender",
        "accuracy",
        (
            "Gender Classification Accuracy "
            "Across Labelled Datasets"
        ),
        "01_gender_accuracy_comparison.png",
    )

    # Race task differs by dataset, so build a common
    # race-only temporary table for plotting accuracy.
    race_plot = overall[
        overall[
            "task"
        ].isin(
            [
                "race_6class",
                "race_4class",
            ]
        )
    ].copy()

    race_plot[
        "task"
    ] = "race"

    save_performance_figure(
        race_plot,
        "race",
        "accuracy",
        (
            "Harmonised Race Classification Accuracy "
            "Across Labelled Datasets"
        ),
        "02_race_accuracy_comparison.png",
    )

    save_eo_figure(
        eo
    )

    save_lfw_race_distribution(
        lfw_distributions
    )

    save_lfw_gender_distribution(
        lfw_distributions
    )

    save_lfw_agreement_figure(
        lfw_agreement
    )

    save_lfw_identity_consistency_figure(
        lfw_identity_summary
    )

    save_lfw_race_matrix(
        lfw_race_matrix
    )

    write_summary(
        overall,
        eo,
        lfw_agreement,
        lfw_distribution_distance,
        lfw_identity_summary,
        lfw_modal_summary,
    )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print()
    print(
        "LABELLED-DATASET PERFORMANCE"
    )

    print()

    print(
        overall[
            [
                "dataset",
                "model",
                "task",
                "n",
                "accuracy",
                "f1_macro",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "GENDER EQUALIZED ODDS"
    )

    print()

    print(
        eo[
            [
                "dataset",
                "model",
                "tpr_difference",
                "fpr_difference",
                "equalized_odds_difference",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "LFW CROSS-MODEL AGREEMENT"
    )

    print()

    print(
        lfw_agreement.to_string(
            index=False
        )
    )

    print()
    print(
        "LFW DISTRIBUTION DISTANCE"
    )

    print()

    print(
        lfw_distribution_distance.to_string(
            index=False
        )
    )

    print()
    print(
        "LFW REPEATED-IDENTITY CONSISTENCY"
    )

    print()

    print(
        lfw_identity_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "LFW REPEATED-IDENTITY MODAL AGREEMENT"
    )

    print()

    print(
        lfw_modal_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Track A comparison pack complete."
    )

    print(
        "Results saved to:"
    )

    print(
        OUTPUT.resolve()
    )


if __name__ == "__main__":
    main()
    