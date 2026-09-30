from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src import paths


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = Path(
    "results/model_audit/fairface_resnet34/fairface_validation"
)

PREDICTIONS = RESULTS_DIR / "predictions.csv"
OVERALL = RESULTS_DIR / "overall_metrics.csv"
RACE_REPORT = RESULTS_DIR / "race_classification_report.csv"
GENDER_REPORT = RESULTS_DIR / "gender_classification_report.csv"
INTERSECTIONAL = RESULTS_DIR / "intersectional_metrics.csv"
GENDER_FAIRNESS = RESULTS_DIR / "gender_fairness_by_race.csv"
RACE_FAIRNESS = RESULTS_DIR / "race_fairness_by_gender.csv"

OUTPUT = RESULTS_DIR / "report_pack"
FIGURES = OUTPUT / "figures"
EXAMPLES = OUTPUT / "examples"

RACE_ORDER = [
    "White",
    "Black",
    "Latino_Hispanic",
    "East Asian",
    "Southeast Asian",
    "Indian",
    "Middle Eastern",
]

GENDER_ORDER = [
    "Male",
    "Female",
]

SEED = 26041


# ============================================================
# HELPERS
# ============================================================

def save_figure(
    fig,
    filename: str,
) -> None:

    FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES / filename,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)


def percent_axis(
    ax,
    axis="y",
) -> None:

    from matplotlib.ticker import PercentFormatter

    if axis == "y":
        ax.yaxis.set_major_formatter(
            PercentFormatter(1.0)
        )
    else:
        ax.xaxis.set_major_formatter(
            PercentFormatter(1.0)
        )


# ============================================================
# FIGURE 1 — OVERALL PERFORMANCE
# ============================================================

def plot_overall_performance(
    overall: pd.DataFrame,
) -> None:

    metrics = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]

    labels = [
        "Accuracy",
        "Macro precision",
        "Macro recall",
        "Macro F1",
    ]

    tasks = [
        "race",
        "gender",
    ]

    x = np.arange(
        len(metrics)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for index, task in enumerate(
        tasks
    ):
        row = overall[
            overall["task"].eq(task)
        ].iloc[0]

        values = [
            row[m]
            for m in metrics
        ]

        positions = (
            x
            + (
                index - 0.5
            )
            * width
        )

        bars = ax.bar(
            positions,
            values,
            width=width,
            label=task.capitalize(),
        )

        for bar, value in zip(
            bars,
            values,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height()
                + 0.01,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(
        0,
        1.05,
    )

    percent_axis(ax)

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "FairFace ResNet-34 Overall Classification Performance"
    )

    ax.legend()

    save_figure(
        fig,
        "01_overall_performance.png",
    )


# ============================================================
# FIGURE 2 — RACE CLASS PERFORMANCE
# ============================================================

def plot_race_metrics(
    report: pd.DataFrame,
) -> None:

    available = [
        race
        for race in RACE_ORDER
        if race in report.index
    ]

    subset = report.loc[
        available
    ]

    x = np.arange(
        len(available)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    metrics = [
        (
            "precision",
            "Precision",
        ),
        (
            "recall",
            "Recall",
        ),
        (
            "f1-score",
            "F1",
        ),
    ]

    for index, (
        column,
        label,
    ) in enumerate(metrics):

        values = (
            subset[column]
            .astype(float)
            .values
        )

        ax.bar(
            x
            + (
                index - 1
            )
            * width,
            values,
            width=width,
            label=label,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        available,
        rotation=30,
        ha="right",
    )

    ax.set_ylim(
        0,
        1.05,
    )

    percent_axis(ax)

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Race Classification Performance by Class"
    )

    ax.legend()

    save_figure(
        fig,
        "02_race_precision_recall_f1.png",
    )


# ============================================================
# FIGURE 3 — INTERSECTIONAL RACE ACCURACY
# ============================================================

def plot_intersectional_race_accuracy(
    intersectional: pd.DataFrame,
) -> None:

    pivot = intersectional.pivot(
        index="race_true",
        columns="gender_true",
        values="race_accuracy",
    )

    pivot = pivot.reindex(
        RACE_ORDER
    )

    pivot = pivot.reindex(
        columns=GENDER_ORDER
    )

    matrix = pivot.values

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    image = ax.imshow(
        matrix,
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        range(
            len(GENDER_ORDER)
        )
    )

    ax.set_xticklabels(
        GENDER_ORDER
    )

    ax.set_yticks(
        range(
            len(RACE_ORDER)
        )
    )

    ax.set_yticklabels(
        RACE_ORDER
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            value = matrix[
                row,
                column,
            ]

            if np.isfinite(value):

                ax.text(
                    column,
                    row,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                )

    ax.set_title(
        "Race Classification Accuracy by Race × Gender"
    )

    ax.set_xlabel(
        "Gender"
    )

    ax.set_ylabel(
        "Race"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Race accuracy",
    )

    save_figure(
        fig,
        "03_intersectional_race_accuracy.png",
    )


# ============================================================
# FIGURE 4 — INTERSECTIONAL GENDER ACCURACY
# ============================================================

def plot_intersectional_gender_accuracy(
    intersectional: pd.DataFrame,
) -> None:

    pivot = intersectional.pivot(
        index="race_true",
        columns="gender_true",
        values="gender_accuracy",
    )

    pivot = pivot.reindex(
        RACE_ORDER
    )

    pivot = pivot.reindex(
        columns=GENDER_ORDER
    )

    matrix = pivot.values

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    image = ax.imshow(
        matrix,
        vmin=0.85,
        vmax=1.0,
    )

    ax.set_xticks(
        range(
            len(GENDER_ORDER)
        )
    )

    ax.set_xticklabels(
        GENDER_ORDER
    )

    ax.set_yticks(
        range(
            len(RACE_ORDER)
        )
    )

    ax.set_yticklabels(
        RACE_ORDER
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            value = matrix[
                row,
                column,
            ]

            if np.isfinite(value):

                ax.text(
                    column,
                    row,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                )

    ax.set_title(
        "Gender Classification Accuracy by Race × Gender"
    )

    ax.set_xlabel(
        "Gender"
    )

    ax.set_ylabel(
        "Race"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Gender accuracy",
    )

    save_figure(
        fig,
        "04_intersectional_gender_accuracy.png",
    )


# ============================================================
# FIGURE 5 — GENDER TPR/FPR BY RACE
# ============================================================

def plot_gender_equalized_odds(
    fairness: pd.DataFrame,
) -> None:

    frame = (
        fairness
        .set_index("race")
        .reindex(RACE_ORDER)
        .reset_index()
    )

    x = np.arange(
        len(frame)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    ax.bar(
        x - width / 2,
        frame[
            "female_tpr"
        ],
        width=width,
        label="TPR: actual Female → Female",
    )

    ax.bar(
        x + width / 2,
        frame[
            "female_fpr"
        ],
        width=width,
        label="FPR: actual Male → Female",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        frame["race"],
        rotation=30,
        ha="right",
    )

    ax.set_ylim(
        0,
        1.05,
    )

    percent_axis(ax)

    ax.set_ylabel(
        "Rate"
    )

    ax.set_title(
        "Gender Classification Error Rates Across Race"
    )

    ax.legend()

    save_figure(
        fig,
        "05_gender_equalized_odds_by_race.png",
    )


# ============================================================
# FIGURE 6 — RACE FAIRNESS GAPS
# ============================================================

def plot_race_gender_gaps(
    fairness: pd.DataFrame,
) -> None:

    frame = (
        fairness
        .set_index(
            "target_race"
        )
        .reindex(
            RACE_ORDER
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    bars = ax.barh(
        frame[
            "target_race"
        ],
        frame[
            "equalized_odds_difference"
        ],
    )

    ax.set_xlim(
        0,
        max(
            0.3,
            frame[
                "equalized_odds_difference"
            ].max()
            * 1.15,
        ),
    )

    percent_axis(
        ax,
        axis="x",
    )

    ax.set_xlabel(
        "Equalized-odds difference"
    )

    ax.set_title(
        "Race One-vs-Rest Performance Gap Across Gender"
    )

    for bar, value in zip(
        bars,
        frame[
            "equalized_odds_difference"
        ],
    ):

        ax.text(
            value + 0.005,
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
        )

    save_figure(
        fig,
        "06_race_equalized_odds_by_gender.png",
    )


# ============================================================
# FIGURE 7 — CONFIDENCE BY INTERSECTION
# ============================================================

def plot_confidence_by_group(
    intersectional: pd.DataFrame,
) -> None:

    frame = intersectional.copy()

    frame[
        "group"
    ] = (
        frame[
            "race_true"
        ]
        + " | "
        + frame[
            "gender_true"
        ]
    )

    frame = frame.sort_values(
        "mean_model_race_confidence"
    )

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    ax.barh(
        frame[
            "group"
        ],
        frame[
            "mean_model_race_confidence"
        ],
    )

    ax.set_xlim(
        0,
        1,
    )

    percent_axis(
        ax,
        axis="x",
    )

    ax.set_xlabel(
        "Mean confidence"
    )

    ax.set_title(
        "Mean Race-Prediction Confidence by Intersection"
    )

    save_figure(
        fig,
        "07_race_confidence_by_intersection.png",
    )


# ============================================================
# FIGURE 8 — CORRECT VS INCORRECT CONFIDENCE
# ============================================================

def plot_error_confidence(
    predictions: pd.DataFrame,
) -> None:

    correct = predictions[
        predictions[
            "race_correct"
        ].eq(True)
    ][
        "model_race_confidence"
    ]

    incorrect = predictions[
        predictions[
            "race_correct"
        ].eq(False)
    ][
        "model_race_confidence"
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        correct,
        bins=20,
        alpha=0.6,
        label="Correct race prediction",
        density=True,
    )

    ax.hist(
        incorrect,
        bins=20,
        alpha=0.6,
        label="Incorrect race prediction",
        density=True,
    )

    ax.set_xlabel(
        "Model race confidence"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Confidence Distribution for Correct and Incorrect Race Predictions"
    )

    ax.legend()

    save_figure(
        fig,
        "08_race_confidence_correct_vs_error.png",
    )


# ============================================================
# EXAMPLE GRIDS
# ============================================================

def create_example_grid(
    frame: pd.DataFrame,
    filename: str,
    title: str,
    n: int = 12,
) -> None:

    if frame.empty:
        return

    sample_n = min(
        n,
        len(frame),
    )

    sample = frame.sample(
        n=sample_n,
        random_state=SEED,
    )

    columns = 4

    rows = int(
        np.ceil(
            sample_n
            / columns
        )
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            12,
            rows * 4,
        ),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for ax in axes:
        ax.axis(
            "off"
        )

    for ax, (
        _,
        row,
    ) in zip(
        axes,
        sample.iterrows(),
    ):

        image_path = (
            paths.data_abs(
                str(
                    row[
                        "original_path"
                    ]
                )
            )
        )

        with Image.open(
            image_path
        ) as image:

            image = image.convert(
                "RGB"
            )

            ax.imshow(
                image
            )

        ax.axis(
            "off"
        )

        subtitle = (
            f"True: {row['race_true']} | "
            f"{row['gender_true']}\n"
            f"Pred: {row['race_pred']} | "
            f"{row['gender_pred']}\n"
            f"Race conf: "
            f"{row['model_race_confidence']:.2f}"
        )

        ax.set_title(
            subtitle,
            fontsize=8,
        )

    fig.suptitle(
        title,
        fontsize=16,
    )

    EXAMPLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.tight_layout()

    fig.savefig(
        EXAMPLES / filename,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_examples(
    predictions: pd.DataFrame,
) -> None:

    create_example_grid(
        predictions[
            predictions[
                "race_correct"
            ].eq(True)
            &
            predictions[
                "gender_correct"
            ].eq(True)
        ],
        "01_correct_predictions.png",
        "Examples: Correct Race and Gender Predictions",
    )

    create_example_grid(
        predictions[
            predictions[
                "race_correct"
            ].eq(False)
        ],
        "02_race_errors.png",
        "Examples: Race Classification Errors",
    )

    create_example_grid(
        predictions[
            predictions[
                "gender_correct"
            ].eq(False)
        ],
        "03_gender_errors.png",
        "Examples: Gender Classification Errors",
    )

    key = predictions[
        (
            predictions[
                "race_true"
            ].eq(
                "White"
            )
            &
            predictions[
                "gender_true"
            ].eq(
                "Male"
            )
        )
        |
        (
            predictions[
                "race_true"
            ].eq(
                "White"
            )
            &
            predictions[
                "gender_true"
            ].eq(
                "Female"
            )
        )
        |
        (
            predictions[
                "race_true"
            ].eq(
                "Black"
            )
            &
            predictions[
                "gender_true"
            ].eq(
                "Male"
            )
        )
        |
        (
            predictions[
                "race_true"
            ].eq(
                "Black"
            )
            &
            predictions[
                "gender_true"
            ].eq(
                "Female"
            )
        )
    ]

    create_example_grid(
        key,
        "04_white_black_intersection_examples.png",
        "Examples from White/Black Gender Intersections",
        n=16,
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

def create_key_findings(
    overall: pd.DataFrame,
    intersectional: pd.DataFrame,
    gender_fairness: pd.DataFrame,
    race_fairness: pd.DataFrame,
) -> pd.DataFrame:

    race_row = overall[
        overall[
            "task"
        ].eq(
            "race"
        )
    ].iloc[0]

    gender_row = overall[
        overall[
            "task"
        ].eq(
            "gender"
        )
    ].iloc[0]

    best_race = intersectional.loc[
        intersectional[
            "race_accuracy"
        ].idxmax()
    ]

    worst_race = intersectional.loc[
        intersectional[
            "race_accuracy"
        ].idxmin()
    ]

    largest_gap = race_fairness.loc[
        race_fairness[
            "equalized_odds_difference"
        ].idxmax()
    ]

    gender_tpr_gap = (
        gender_fairness[
            "female_tpr"
        ].max()
        -
        gender_fairness[
            "female_tpr"
        ].min()
    )

    gender_fpr_gap = (
        gender_fairness[
            "female_fpr"
        ].max()
        -
        gender_fairness[
            "female_fpr"
        ].min()
    )

    rows = [
        {
            "finding": (
                "Overall race accuracy"
            ),
            "value": (
                race_row[
                    "accuracy"
                ]
            ),
        },
        {
            "finding": (
                "Overall race macro F1"
            ),
            "value": (
                race_row[
                    "f1_macro"
                ]
            ),
        },
        {
            "finding": (
                "Overall gender accuracy"
            ),
            "value": (
                gender_row[
                    "accuracy"
                ]
            ),
        },
        {
            "finding": (
                "Overall gender macro F1"
            ),
            "value": (
                gender_row[
                    "f1_macro"
                ]
            ),
        },
        {
            "finding": (
                "Best race×gender race accuracy"
            ),
            "value": (
                best_race[
                    "race_accuracy"
                ]
            ),
            "group": (
                f"{best_race['race_true']} "
                f"{best_race['gender_true']}"
            ),
        },
        {
            "finding": (
                "Lowest race×gender race accuracy"
            ),
            "value": (
                worst_race[
                    "race_accuracy"
                ]
            ),
            "group": (
                f"{worst_race['race_true']} "
                f"{worst_race['gender_true']}"
            ),
        },
        {
            "finding": (
                "Intersectional race-accuracy range"
            ),
            "value": (
                best_race[
                    "race_accuracy"
                ]
                -
                worst_race[
                    "race_accuracy"
                ]
            ),
        },
        {
            "finding": (
                "Gender-classification TPR range across race"
            ),
            "value": (
                gender_tpr_gap
            ),
        },
        {
            "finding": (
                "Gender-classification FPR range across race"
            ),
            "value": (
                gender_fpr_gap
            ),
        },
        {
            "finding": (
                "Largest race one-vs-rest gender disparity"
            ),
            "value": (
                largest_gap[
                    "equalized_odds_difference"
                ]
            ),
            "group": (
                largest_gap[
                    "target_race"
                ]
            ),
        },
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    required = [
        PREDICTIONS,
        OVERALL,
        RACE_REPORT,
        GENDER_REPORT,
        INTERSECTIONAL,
        GENDER_FAIRNESS,
        RACE_FAIRNESS,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing result files:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(
        PREDICTIONS
    )

    overall = pd.read_csv(
        OVERALL
    )

    race_report = pd.read_csv(
        RACE_REPORT,
        index_col=0,
    )

    gender_report = pd.read_csv(
        GENDER_REPORT,
        index_col=0,
    )

    intersectional = pd.read_csv(
        INTERSECTIONAL
    )

    gender_fairness = pd.read_csv(
        GENDER_FAIRNESS
    )

    race_fairness = pd.read_csv(
        RACE_FAIRNESS
    )

    plot_overall_performance(
        overall
    )

    plot_race_metrics(
        race_report
    )

    plot_intersectional_race_accuracy(
        intersectional
    )

    plot_intersectional_gender_accuracy(
        intersectional
    )

    plot_gender_equalized_odds(
        gender_fairness
    )

    plot_race_gender_gaps(
        race_fairness
    )

    plot_confidence_by_group(
        intersectional
    )

    plot_error_confidence(
        predictions
    )

    create_examples(
        predictions
    )

    key_findings = create_key_findings(
        overall,
        intersectional,
        gender_fairness,
        race_fairness,
    )

    key_findings.to_csv(
        OUTPUT
        / "key_findings.csv",
        index=False,
    )

    print()
    print(
        "FairFace report pack complete"
    )

    print(
        "================================"
    )

    print(
        f"Figures:  "
        f"{FIGURES.resolve()}"
    )

    print(
        f"Examples: "
        f"{EXAMPLES.resolve()}"
    )

    print()

    print(
        key_findings.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
    