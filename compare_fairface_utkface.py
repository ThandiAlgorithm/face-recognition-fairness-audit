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


FAIRFACE_DIR = Path(
    "results/model_audit/fairface_resnet34/fairface_validation"
)

UTKFACE_DIR = Path(
    "results/model_audit/fairface_resnet34/utkface_aligned"
)

OUTPUT_DIR = Path(
    "results/model_audit/fairface_resnet34/cross_dataset_comparison"
)


HARMONISED_RACES = [
    "White",
    "Black",
    "Asian",
    "Indian",
]


# ============================================================
# HARMONISATION
# ============================================================

def harmonise_fairface_race(
    value: str,
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

    return mapping.get(
        value,
        "Out_of_scope",
    )


# ============================================================
# FAIRFACE 4-CLASS EVALUATION
# ============================================================

def calculate_fairface_harmonised(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    frame = predictions.copy()

    frame[
        "race_true_harmonised"
    ] = frame[
        "race_true"
    ].map(
        harmonise_fairface_race
    )

    frame[
        "race_pred_harmonised"
    ] = frame[
        "race_pred"
    ].map(
        harmonise_fairface_race
    )

    comparable = frame[
        frame[
            "race_true_harmonised"
        ].isin(
            HARMONISED_RACES
        )
    ].copy()

    y_true = comparable[
        "race_true_harmonised"
    ]

    y_pred = comparable[
        "race_pred_harmonised"
    ]

    metrics = pd.DataFrame(
        [
            {
                "dataset": (
                    "FairFace validation"
                ),

                "task": (
                    "race_harmonised_4class"
                ),

                "n": int(
                    len(
                        comparable
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
                        labels=HARMONISED_RACES,
                        average="macro",
                        zero_division=0,
                    )
                ),

                "recall_macro": float(
                    recall_score(
                        y_true,
                        y_pred,
                        labels=HARMONISED_RACES,
                        average="macro",
                        zero_division=0,
                    )
                ),

                "f1_macro": float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=HARMONISED_RACES,
                        average="macro",
                        zero_division=0,
                    )
                ),

                "f1_weighted": float(
                    f1_score(
                        y_true,
                        y_pred,
                        labels=HARMONISED_RACES,
                        average="weighted",
                        zero_division=0,
                    )
                ),
            }
        ]
    )

    taxonomy = pd.DataFrame(
        [
            {
                "dataset": (
                    "FairFace validation"
                ),

                "comparable_n": int(
                    len(
                        comparable
                    )
                ),

                "out_of_scope_prediction_n": int(
                    comparable[
                        "race_pred_harmonised"
                    ]
                    .eq(
                        "Out_of_scope"
                    )
                    .sum()
                ),

                "out_of_scope_prediction_rate": float(
                    comparable[
                        "race_pred_harmonised"
                    ]
                    .eq(
                        "Out_of_scope"
                    )
                    .mean()
                ),
            }
        ]
    )

    return (
        metrics,
        taxonomy,
    )


# ============================================================
# MAIN COMPARISON
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fairface_predictions = pd.read_csv(
        FAIRFACE_DIR
        / "predictions.csv"
    )

    fairface_overall = pd.read_csv(
        FAIRFACE_DIR
        / "overall_metrics.csv"
    )

    utkface_overall = pd.read_csv(
        UTKFACE_DIR
        / "overall_metrics.csv"
    )

    with open(
        FAIRFACE_DIR
        / "gender_equalized_odds_summary.json",
        "r",
        encoding="utf-8",
    ) as handle:
        fairface_eo = json.load(
            handle
        )

    with open(
        UTKFACE_DIR
        / "gender_equalized_odds_summary.json",
        "r",
        encoding="utf-8",
    ) as handle:
        utkface_eo = json.load(
            handle
        )

    (
        fairface_race4,
        fairface_taxonomy,
    ) = calculate_fairface_harmonised(
        fairface_predictions
    )

    fairface_race4.to_csv(
        OUTPUT_DIR
        / "fairface_harmonised_4class_metrics.csv",
        index=False,
    )

    fairface_taxonomy.to_csv(
        OUTPUT_DIR
        / "fairface_harmonised_taxonomy_summary.csv",
        index=False,
    )

    ff_gender = fairface_overall[
        fairface_overall[
            "task"
        ].eq(
            "gender"
        )
    ].iloc[0]

    utk_gender = utkface_overall[
        utkface_overall[
            "task"
        ].eq(
            "gender_all_utkface"
        )
    ].iloc[0]

    utk_race = utkface_overall[
        utkface_overall[
            "task"
        ].eq(
            "race_harmonised_4class"
        )
    ].iloc[0]

    ff_race = fairface_race4.iloc[
        0
    ]

    comparison = pd.DataFrame(
        [
            {
                "dataset": (
                    "FairFace validation"
                ),

                "gender_accuracy": (
                    ff_gender[
                        "accuracy"
                    ]
                ),

                "gender_macro_f1": (
                    ff_gender[
                        "f1_macro"
                    ]
                ),

                "race_4class_accuracy": (
                    ff_race[
                        "accuracy"
                    ]
                ),

                "race_4class_macro_f1": (
                    ff_race[
                        "f1_macro"
                    ]
                ),

                "gender_equalized_odds_difference": (
                    fairface_eo[
                        "equalized_odds_difference"
                    ]
                ),
            },

            {
                "dataset": (
                    "UTKFace aligned"
                ),

                "gender_accuracy": (
                    utk_gender[
                        "accuracy"
                    ]
                ),

                "gender_macro_f1": (
                    utk_gender[
                        "f1_macro"
                    ]
                ),

                "race_4class_accuracy": (
                    utk_race[
                        "accuracy"
                    ]
                ),

                "race_4class_macro_f1": (
                    utk_race[
                        "f1_macro"
                    ]
                ),

                "gender_equalized_odds_difference": (
                    utkface_eo[
                        "equalized_odds_difference"
                    ]
                ),
            },
        ]
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "cross_dataset_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # FIGURE 1: PERFORMANCE
    # --------------------------------------------------------

    metrics = [
        "gender_accuracy",
        "gender_macro_f1",
        "race_4class_accuracy",
        "race_4class_macro_f1",
    ]

    labels = [
        "Gender\naccuracy",
        "Gender\nmacro F1",
        "Race 4-class\naccuracy",
        "Race 4-class\nmacro F1",
    ]

    x = np.arange(
        len(
            metrics
        )
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(
            11,
            7,
        )
    )

    for dataset_index, (
        _,
        row,
    ) in enumerate(
        comparison.iterrows()
    ):

        values = [
            row[
                metric
            ]
            for metric in metrics
        ]

        positions = (
            x
            + (
                dataset_index
                - 0.5
            )
            * width
        )

        bars = ax.bar(
            positions,
            values,
            width=width,
            label=row[
                "dataset"
            ],
        )

        for bar, value in zip(
            bars,
            values,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                value
                + 0.012,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "FairFace ResNet-34 Cross-Dataset Performance"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "01_cross_dataset_performance.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # --------------------------------------------------------
    # FIGURE 2: EQUALIZED ODDS
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    bars = ax.bar(
        comparison[
            "dataset"
        ],
        comparison[
            "gender_equalized_odds_difference"
        ],
    )

    ax.set_ylabel(
        "Equalized-odds difference"
    )

    ax.set_title(
        "Gender Classification Fairness Across Datasets"
    )

    ax.set_ylim(
        0,
        max(
            0.1,
            comparison[
                "gender_equalized_odds_difference"
            ].max()
            * 1.25,
        ),
    )

    for bar, value in zip(
        bars,
        comparison[
            "gender_equalized_odds_difference"
        ],
    ):

        ax.text(
            bar.get_x()
            + bar.get_width()
            / 2,
            value
            + 0.003,
            f"{value:.1%}",
            ha="center",
        )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "02_gender_equalized_odds_comparison.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print()
    print(
        "Cross-dataset comparison complete"
    )

    print(
        "================================="
    )

    print()
    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        "FairFace harmonised taxonomy:"
    )

    print(
        fairface_taxonomy.to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()
    