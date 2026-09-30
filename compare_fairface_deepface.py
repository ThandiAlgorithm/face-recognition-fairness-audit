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
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

FAIRFACE_RESULTS = Path(
    "results/model_audit/fairface_resnet34/fairface_validation"
)

DEEPFACE_RESULTS = Path(
    "results/model_audit/deepface/fairface_validation"
)

OUTPUT = Path(
    "results/model_audit/model_comparison/fairface_validation"
)


COMMON_RACES = [
    "White",
    "Black",
    "Asian",
    "Indian",
    "Latino_Hispanic",
    "Middle Eastern",
]

GENDERS = [
    "Male",
    "Female",
]


# ============================================================
# TAXONOMY
# ============================================================

def harmonise_race(
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


# ============================================================
# METRICS
# ============================================================

def metric_row(
    model: str,
    task: str,
    truth,
    prediction,
    labels,
) -> dict:

    return {
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

def gender_fairness(
    frame: pd.DataFrame,
    prediction_column: str,
    model: str,
) -> tuple[pd.DataFrame, dict]:

    rows = []

    for race in COMMON_RACES:

        group = frame[
            frame[
                "race_true_common"
            ].eq(
                race
            )
        ].copy()

        female = (
            group[
                "gender_true"
            ].eq(
                "Female"
            )
        )

        male = (
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
                female
                &
                predicted_female
            ).sum()
        )

        fn = int(
            (
                female
                &
                ~predicted_female
            ).sum()
        )

        fp = int(
            (
                male
                &
                predicted_female
            ).sum()
        )

        tn = int(
            (
                male
                &
                ~predicted_female
            ).sum()
        )

        tpr = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else np.nan
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else np.nan
        )

        rows.append(
            {
                "model": model,
                "race": race,
                "n": len(group),
                "n_female": int(
                    female.sum()
                ),
                "n_male": int(
                    male.sum()
                ),
                "female_tpr": tpr,
                "female_fpr": fpr,
            }
        )

    rates = pd.DataFrame(
        rows
    )

    tpr_gap = float(
        rates[
            "female_tpr"
        ].max()
        -
        rates[
            "female_tpr"
        ].min()
    )

    fpr_gap = float(
        rates[
            "female_fpr"
        ].max()
        -
        rates[
            "female_fpr"
        ].min()
    )

    summary = {
        "model": model,
        "tpr_difference": tpr_gap,
        "fpr_difference": fpr_gap,
        "equalized_odds_difference": max(
            tpr_gap,
            fpr_gap,
        ),
    }

    return (
        rates,
        summary,
    )


# ============================================================
# INTERSECTIONAL COMPARISON
# ============================================================

def intersectional_comparison(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    working = frame.copy()

    working[
        "fairface_race_correct"
    ] = (
        working[
            "race_true_common"
        ]
        ==
        working[
            "fairface_race_pred"
        ]
    )

    working[
        "deepface_race_correct"
    ] = (
        working[
            "race_true_common"
        ]
        ==
        working[
            "deepface_race_pred"
        ]
    )

    working[
        "fairface_gender_correct"
    ] = (
        working[
            "gender_true"
        ]
        ==
        working[
            "fairface_gender_pred"
        ]
    )

    working[
        "deepface_gender_correct"
    ] = (
        working[
            "gender_true"
        ]
        ==
        working[
            "deepface_gender_pred"
        ]
    )

    table = (
        working.groupby(
            [
                "race_true_common",
                "gender_true",
            ],
            observed=True,
        )
        .agg(
            n=(
                "image_id",
                "count",
            ),

            fairface_race_accuracy=(
                "fairface_race_correct",
                "mean",
            ),

            deepface_race_accuracy=(
                "deepface_race_correct",
                "mean",
            ),

            fairface_gender_accuracy=(
                "fairface_gender_correct",
                "mean",
            ),

            deepface_gender_accuracy=(
                "deepface_gender_correct",
                "mean",
            ),
        )
        .reset_index()
    )

    table[
        "race_accuracy_gap_deepface_minus_fairface"
    ] = (
        table[
            "deepface_race_accuracy"
        ]
        -
        table[
            "fairface_race_accuracy"
        ]
    )

    table[
        "gender_accuracy_gap_deepface_minus_fairface"
    ] = (
        table[
            "deepface_gender_accuracy"
        ]
        -
        table[
            "fairface_gender_accuracy"
        ]
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
# MODEL AGREEMENT
# ============================================================

def agreement_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        task,
        truth_column,
        fairface_column,
        deepface_column,
    ) in [
        (
            "race",
            "race_true_common",
            "fairface_race_pred",
            "deepface_race_pred",
        ),

        (
            "gender",
            "gender_true",
            "fairface_gender_pred",
            "deepface_gender_pred",
        ),
    ]:

        ff_correct = (
            frame[
                fairface_column
            ]
            ==
            frame[
                truth_column
            ]
        )

        df_correct = (
            frame[
                deepface_column
            ]
            ==
            frame[
                truth_column
            ]
        )

        model_agreement = (
            frame[
                fairface_column
            ]
            ==
            frame[
                deepface_column
            ]
        )

        rows.append(
            {
                "task": task,

                "n": int(
                    len(
                        frame
                    )
                ),

                "prediction_agreement_rate": float(
                    model_agreement.mean()
                ),

                "both_correct": int(
                    (
                        ff_correct
                        &
                        df_correct
                    ).sum()
                ),

                "fairface_only_correct": int(
                    (
                        ff_correct
                        &
                        ~df_correct
                    ).sum()
                ),

                "deepface_only_correct": int(
                    (
                        ~ff_correct
                        &
                        df_correct
                    ).sum()
                ),

                "neither_correct": int(
                    (
                        ~ff_correct
                        &
                        ~df_correct
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FIGURES
# ============================================================

def performance_figure(
    overall: pd.DataFrame,
) -> None:

    pivot = overall.pivot(
        index="task",
        columns="model",
        values=[
            "accuracy",
            "f1_macro",
        ],
    )

    categories = [
        "Race accuracy",
        "Race macro F1",
        "Gender accuracy",
        "Gender macro F1",
    ]

    fairface = [
        pivot.loc[
            "race_harmonised_6class",
            (
                "accuracy",
                "FairFace ResNet-34",
            ),
        ],

        pivot.loc[
            "race_harmonised_6class",
            (
                "f1_macro",
                "FairFace ResNet-34",
            ),
        ],

        pivot.loc[
            "gender",
            (
                "accuracy",
                "FairFace ResNet-34",
            ),
        ],

        pivot.loc[
            "gender",
            (
                "f1_macro",
                "FairFace ResNet-34",
            ),
        ],
    ]

    deepface = [
        pivot.loc[
            "race_harmonised_6class",
            (
                "accuracy",
                "DeepFace",
            ),
        ],

        pivot.loc[
            "race_harmonised_6class",
            (
                "f1_macro",
                "DeepFace",
            ),
        ],

        pivot.loc[
            "gender",
            (
                "accuracy",
                "DeepFace",
            ),
        ],

        pivot.loc[
            "gender",
            (
                "f1_macro",
                "DeepFace",
            ),
        ],
    ]

    x = np.arange(
        len(
            categories
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(
            11,
            7,
        )
    )

    bars_a = ax.bar(
        x - width / 2,
        fairface,
        width,
        label="FairFace ResNet-34",
    )

    bars_b = ax.bar(
        x + width / 2,
        deepface,
        width,
        label="DeepFace",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        categories
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Demographic Classifier Performance on FairFace Validation"
    )

    ax.legend()

    for bars in [
        bars_a,
        bars_b,
    ]:

        for bar in bars:

            value = (
                bar.get_height()
            )

            ax.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                value + 0.012,
                f"{value:.1%}",
                ha="center",
                fontsize=8,
            )

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "01_model_performance_comparison.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def gender_tpr_figure(
    fairness: pd.DataFrame,
) -> None:

    pivot = fairness.pivot(
        index="race",
        columns="model",
        values="female_tpr",
    )

    pivot = pivot.reindex(
        COMMON_RACES
    )

    x = np.arange(
        len(
            pivot
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )

    ax.bar(
        x - width / 2,
        pivot[
            "FairFace ResNet-34"
        ],
        width,
        label="FairFace ResNet-34",
    )

    ax.bar(
        x + width / 2,
        pivot[
            "DeepFace"
        ],
        width,
        label="DeepFace",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        pivot.index,
        rotation=25,
        ha="right",
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Female true-positive rate"
    )

    ax.set_title(
        "Gender Classification TPR by Race and Model"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "02_gender_female_tpr_by_race.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    fairface = pd.read_csv(
        FAIRFACE_RESULTS
        / "predictions.csv"
    )

    deepface = pd.read_csv(
        DEEPFACE_RESULTS
        / "predictions.csv"
    )

    fairface[
        "race_true_common"
    ] = fairface[
        "race_true"
    ].map(
        harmonise_race
    )

    fairface[
        "fairface_race_pred"
    ] = fairface[
        "race_pred"
    ].map(
        harmonise_race
    )

    fairface[
        "fairface_gender_pred"
    ] = fairface[
        "gender_pred"
    ]

    fairface_subset = fairface[
        [
            "image_id",
            "race_true_common",
            "gender_true",
            "fairface_race_pred",
            "fairface_gender_pred",
        ]
    ].copy()

    deepface_subset = deepface[
        [
            "image_id",
            "race_pred",
            "gender_pred",
        ]
    ].rename(
        columns={
            "race_pred": (
                "deepface_race_pred"
            ),
            "gender_pred": (
                "deepface_gender_pred"
            ),
        }
    )

    merged = fairface_subset.merge(
        deepface_subset,
        on="image_id",
        how="inner",
        validate="one_to_one",
    )

    if len(
        merged
    ) != 10954:

        raise RuntimeError(
            f"Expected 10,954 matched images; "
            f"found {len(merged):,}."
        )

    overall = pd.DataFrame(
        [
            metric_row(
                "FairFace ResNet-34",
                "race_harmonised_6class",
                merged[
                    "race_true_common"
                ],
                merged[
                    "fairface_race_pred"
                ],
                COMMON_RACES,
            ),

            metric_row(
                "DeepFace",
                "race_harmonised_6class",
                merged[
                    "race_true_common"
                ],
                merged[
                    "deepface_race_pred"
                ],
                COMMON_RACES,
            ),

            metric_row(
                "FairFace ResNet-34",
                "gender",
                merged[
                    "gender_true"
                ],
                merged[
                    "fairface_gender_pred"
                ],
                GENDERS,
            ),

            metric_row(
                "DeepFace",
                "gender",
                merged[
                    "gender_true"
                ],
                merged[
                    "deepface_gender_pred"
                ],
                GENDERS,
            ),
        ]
    )

    overall.to_csv(
        OUTPUT
        / "overall_model_comparison.csv",
        index=False,
    )

    (
        fairface_fairness,
        fairface_eo,
    ) = gender_fairness(
        merged,
        "fairface_gender_pred",
        "FairFace ResNet-34",
    )

    (
        deepface_fairness,
        deepface_eo,
    ) = gender_fairness(
        merged,
        "deepface_gender_pred",
        "DeepFace",
    )

    fairness = pd.concat(
        [
            fairface_fairness,
            deepface_fairness,
        ],
        ignore_index=True,
    )

    fairness.to_csv(
        OUTPUT
        / "gender_fairness_model_comparison.csv",
        index=False,
    )

    eo_summary = pd.DataFrame(
        [
            fairface_eo,
            deepface_eo,
        ]
    )

    eo_summary.to_csv(
        OUTPUT
        / "gender_equalized_odds_model_comparison.csv",
        index=False,
    )

    intersections = (
        intersectional_comparison(
            merged
        )
    )

    intersections.to_csv(
        OUTPUT
        / "intersectional_model_comparison.csv",
        index=False,
    )

    agreement = (
        agreement_summary(
            merged
        )
    )

    agreement.to_csv(
        OUTPUT
        / "model_prediction_agreement.csv",
        index=False,
    )

    performance_figure(
        overall
    )

    gender_tpr_figure(
        fairness
    )

    print()
    print(
        "FairFace ResNet-34 vs DeepFace"
    )

    print(
        "=========================================="
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
        "GENDER EQUALIZED ODDS"
    )

    print(
        eo_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "MODEL AGREEMENT"
    )

    print(
        agreement.to_string(
            index=False
        )
    )

    print()
    print(
        "INTERSECTIONAL MODEL COMPARISON"
    )

    print(
        intersections.to_string(
            index=False
        )
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT.resolve()
    )


if __name__ == "__main__":
    main()
    