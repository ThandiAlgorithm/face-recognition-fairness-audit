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
    "results/model_audit/fairface_resnet34/lfw_predictions"
)

PREDICTIONS_PATH = (
    RESULTS_DIR / "predictions.csv"
)

RACE_DISTRIBUTION_PATH = (
    RESULTS_DIR / "predicted_race_distribution.csv"
)

GENDER_DISTRIBUTION_PATH = (
    RESULTS_DIR / "predicted_gender_distribution.csv"
)

AGE_DISTRIBUTION_PATH = (
    RESULTS_DIR / "predicted_age_distribution.csv"
)

INTERSECTIONS_PATH = (
    RESULTS_DIR / "predicted_race_gender_intersections.csv"
)

CONFIDENCE_PATH = (
    RESULTS_DIR / "confidence_summary.csv"
)

IDENTITY_PATH = (
    RESULTS_DIR / "identity_prediction_summary.csv"
)

REPEATED_IDENTITY_PATH = (
    RESULTS_DIR / "repeated_identity_consistency_summary.csv"
)

IDENTITY_DEPTH_PATH = (
    RESULTS_DIR / "identity_depth_by_modal_predicted_race.csv"
)


REPORT_DIR = (
    RESULTS_DIR / "report_pack"
)

FIGURES_DIR = (
    REPORT_DIR / "figures"
)

EXAMPLES_DIR = (
    REPORT_DIR / "examples"
)

TABLES_DIR = (
    REPORT_DIR / "tables"
)


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

AGE_ORDER = [
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

SEED = 26041


# ============================================================
# HELPERS
# ============================================================

def ensure_directories() -> None:

    for directory in [
        REPORT_DIR,
        FIGURES_DIR,
        EXAMPLES_DIR,
        TABLES_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_figure(
    fig,
    filename: str,
) -> None:

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / filename,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


def safe_percent(
    value,
) -> str:

    if pd.isna(value):
        return "N/A"

    return (
        f"{float(value):.1%}"
    )


# ============================================================
# FIGURE 1 — PREDICTED RACE DISTRIBUTION
# ============================================================

def plot_race_distribution(
    race_distribution: pd.DataFrame,
) -> None:

    frame = (
        race_distribution
        .set_index(
            "predicted_race"
        )
        .reindex(
            RACE_ORDER
        )
        .reset_index()
    )

    frame[
        "n"
    ] = frame[
        "n"
    ].fillna(
        0
    )

    frame[
        "proportion"
    ] = frame[
        "proportion"
    ].fillna(
        0
    )

    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )

    bars = ax.bar(
        frame[
            "predicted_race"
        ],
        frame[
            "n"
        ],
    )

    ax.set_title(
        "LFW: FairFace ResNet-34 Predicted Race Distribution"
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.set_xlabel(
        "Predicted race category"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    for bar, proportion in zip(
        bars,
        frame[
            "proportion"
        ],
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{proportion:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    save_figure(
        fig,
        "01_predicted_race_distribution.png",
    )


# ============================================================
# FIGURE 2 — PREDICTED GENDER DISTRIBUTION
# ============================================================

def plot_gender_distribution(
    gender_distribution: pd.DataFrame,
) -> None:

    frame = (
        gender_distribution
        .set_index(
            "predicted_gender"
        )
        .reindex(
            GENDER_ORDER
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(
            7,
            6,
        )
    )

    bars = ax.bar(
        frame[
            "predicted_gender"
        ],
        frame[
            "n"
        ],
    )

    ax.set_title(
        "LFW: FairFace ResNet-34 Predicted Gender Distribution"
    )

    ax.set_ylabel(
        "Number of images"
    )

    for bar, proportion in zip(
        bars,
        frame[
            "proportion"
        ],
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{proportion:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_figure(
        fig,
        "02_predicted_gender_distribution.png",
    )


# ============================================================
# FIGURE 3 — PREDICTED AGE DISTRIBUTION
# ============================================================

def plot_age_distribution(
    age_distribution: pd.DataFrame,
) -> None:

    frame = (
        age_distribution
        .set_index(
            "predicted_age_band"
        )
        .reindex(
            AGE_ORDER
        )
        .reset_index()
    )

    frame[
        "n"
    ] = frame[
        "n"
    ].fillna(
        0
    )

    frame[
        "proportion"
    ] = frame[
        "proportion"
    ].fillna(
        0
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    bars = ax.bar(
        frame[
            "predicted_age_band"
        ],
        frame[
            "n"
        ],
    )

    ax.set_title(
        "LFW: FairFace ResNet-34 Predicted Age-Band Distribution"
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.set_xlabel(
        "Predicted age band"
    )

    for bar, proportion in zip(
        bars,
        frame[
            "proportion"
        ],
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{proportion:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(
        fig,
        "03_predicted_age_distribution.png",
    )


# ============================================================
# FIGURE 4 — RACE × GENDER HEATMAP
# ============================================================

def plot_intersection_heatmap(
    intersections: pd.DataFrame,
) -> None:

    pivot = intersections.pivot(
        index="predicted_race",
        columns="predicted_gender",
        values="n",
    )

    pivot = pivot.reindex(
        index=RACE_ORDER,
        columns=GENDER_ORDER,
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

    image = ax.imshow(
        matrix
    )

    ax.set_xticks(
        range(
            len(
                GENDER_ORDER
            )
        )
    )

    ax.set_xticklabels(
        GENDER_ORDER
    )

    ax.set_yticks(
        range(
            len(
                RACE_ORDER
            )
        )
    )

    ax.set_yticklabels(
        RACE_ORDER
    )

    ax.set_xlabel(
        "Predicted gender"
    )

    ax.set_ylabel(
        "Predicted race"
    )

    ax.set_title(
        "LFW: Predicted Race × Gender Distribution"
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

            count = int(
                matrix[
                    row,
                    column,
                ]
            )

            ax.text(
                column,
                row,
                str(
                    count
                ),
                ha="center",
                va="center",
                color=(
                    "white"
                    if count > midpoint
                    else "black"
                ),
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Image count",
    )

    save_figure(
        fig,
        "04_predicted_race_gender_heatmap.png",
    )


# ============================================================
# FIGURE 5 — RACE INTERSECTION PROPORTIONS
# ============================================================

def plot_intersection_proportions(
    intersections: pd.DataFrame,
) -> None:

    frame = intersections.copy()

    frame[
        "group"
    ] = (
        frame[
            "predicted_race"
        ]
        + " | "
        + frame[
            "predicted_gender"
        ]
    )

    frame = frame.sort_values(
        "proportion"
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            9,
        )
    )

    bars = ax.barh(
        frame[
            "group"
        ],
        frame[
            "proportion"
        ],
    )

    ax.set_xlabel(
        "Proportion of LFW images"
    )

    ax.set_title(
        "LFW: Predicted Race × Gender Intersection Proportions"
    )

    for bar, value in zip(
        bars,
        frame[
            "proportion"
        ],
    ):

        ax.text(
            value + 0.002,
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.1%}",
            va="center",
            fontsize=8,
        )

    save_figure(
        fig,
        "05_intersectional_proportions.png",
    )


# ============================================================
# FIGURE 6 — MODEL CONFIDENCE
# ============================================================

def plot_confidence_summary(
    confidence: pd.DataFrame,
) -> None:

    frame = confidence.copy()

    fig, ax = plt.subplots(
        figsize=(
            8,
            6,
        )
    )

    bars = ax.bar(
        frame[
            "task"
        ],
        frame[
            "mean_confidence"
        ],
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_ylabel(
        "Mean model confidence"
    )

    ax.set_title(
        "LFW: Mean FairFace Prediction Confidence"
    )

    for bar, value in zip(
        bars,
        frame[
            "mean_confidence"
        ],
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.015,
            f"{value:.1%}",
            ha="center",
        )

    save_figure(
        fig,
        "06_mean_prediction_confidence.png",
    )


# ============================================================
# FIGURE 7 — CONFIDENCE DISTRIBUTIONS
# ============================================================

def plot_confidence_distributions(
    predictions: pd.DataFrame,
) -> None:

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    ax.hist(
        predictions[
            "predicted_race_confidence"
        ],
        bins=30,
        alpha=0.6,
        density=True,
        label="Race",
    )

    ax.hist(
        predictions[
            "predicted_gender_confidence"
        ],
        bins=30,
        alpha=0.6,
        density=True,
        label="Gender",
    )

    ax.hist(
        predictions[
            "predicted_age_confidence"
        ],
        bins=30,
        alpha=0.6,
        density=True,
        label="Age",
    )

    ax.set_xlabel(
        "Prediction confidence"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "LFW: Confidence Distribution by Attribute"
    )

    ax.legend()

    save_figure(
        fig,
        "07_confidence_distributions.png",
    )


# ============================================================
# FIGURE 8 — RACE CONFIDENCE BY PREDICTED CLASS
# ============================================================

def plot_race_confidence_by_class(
    predictions: pd.DataFrame,
) -> None:

    data = []

    labels = []

    for race in RACE_ORDER:

        values = predictions[
            predictions[
                "predicted_race"
            ].eq(
                race
            )
        ][
            "predicted_race_confidence"
        ]

        if len(
            values
        ) > 0:

            data.append(
                values.values
            )

            labels.append(
                race
            )

    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )

    ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
    )

    ax.set_ylim(
        0,
        1.02,
    )

    ax.set_ylabel(
        "Prediction confidence"
    )

    ax.set_title(
        "LFW: Race-Prediction Confidence by Predicted Category"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    save_figure(
        fig,
        "08_race_confidence_by_predicted_class.png",
    )


# ============================================================
# FIGURE 9 — REPEATED IDENTITY CONSISTENCY
# ============================================================

def plot_identity_consistency(
    identity_summary: pd.DataFrame,
) -> None:

    repeated = identity_summary[
        identity_summary[
            "n_images"
        ] >= 2
    ].copy()

    fig, ax = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    ax.hist(
        repeated[
            "race_prediction_consistency"
        ],
        bins=20,
        alpha=0.6,
        label="Race",
    )

    ax.hist(
        repeated[
            "gender_prediction_consistency"
        ],
        bins=20,
        alpha=0.6,
        label="Gender",
    )

    ax.hist(
        repeated[
            "age_prediction_consistency"
        ],
        bins=20,
        alpha=0.6,
        label="Age",
    )

    ax.set_xlabel(
        "Within-identity prediction consistency"
    )

    ax.set_ylabel(
        "Number of repeated identities"
    )

    ax.set_title(
        "LFW: Prediction Consistency Across Images of the Same Identity"
    )

    ax.legend()

    save_figure(
        fig,
        "09_repeated_identity_consistency.png",
    )


# ============================================================
# FIGURE 10 — CONSISTENCY VS IDENTITY DEPTH
# ============================================================

def plot_consistency_vs_depth(
    identity_summary: pd.DataFrame,
) -> None:

    repeated = identity_summary[
        identity_summary[
            "n_images"
        ] >= 2
    ].copy()

    repeated[
        "depth_band"
    ] = pd.cut(
        repeated[
            "n_images"
        ],
        bins=[
            1,
            2,
            5,
            10,
            20,
            50,
            np.inf,
        ],
        labels=[
            "2",
            "3–5",
            "6–10",
            "11–20",
            "21–50",
            "51+",
        ],
    )

    summary = (
        repeated.groupby(
            "depth_band",
            observed=True,
        )
        .agg(
            n_identities=(
                "subject_id",
                "count",
            ),

            mean_race_consistency=(
                "race_prediction_consistency",
                "mean",
            ),

            mean_gender_consistency=(
                "gender_prediction_consistency",
                "mean",
            ),

            mean_age_consistency=(
                "age_prediction_consistency",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        TABLES_DIR
        / "identity_consistency_by_depth.csv",
        index=False,
    )

    x = np.arange(
        len(
            summary
        )
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(
            11,
            6,
        )
    )

    ax.bar(
        x - width,
        summary[
            "mean_race_consistency"
        ],
        width=width,
        label="Race",
    )

    ax.bar(
        x,
        summary[
            "mean_gender_consistency"
        ],
        width=width,
        label="Gender",
    )

    ax.bar(
        x + width,
        summary[
            "mean_age_consistency"
        ],
        width=width,
        label="Age",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        summary[
            "depth_band"
        ].astype(str)
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_xlabel(
        "Number of images per identity"
    )

    ax.set_ylabel(
        "Mean consistency"
    )

    ax.set_title(
        "LFW: Prediction Consistency vs Identity Depth"
    )

    ax.legend()

    save_figure(
        fig,
        "10_consistency_vs_identity_depth.png",
    )


# ============================================================
# FIGURE 11 — IDENTITY DEPTH BY PREDICTED RACE
# ============================================================

def plot_identity_depth_by_race(
    identity_depth: pd.DataFrame,
) -> None:

    frame = (
        identity_depth
        .set_index(
            "modal_predicted_race"
        )
        .reindex(
            RACE_ORDER
        )
        .dropna(
            how="all"
        )
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(
            12,
            7,
        )
    )

    bars = ax.bar(
        frame[
            "modal_predicted_race"
        ],
        frame[
            "mean_images_per_identity"
        ],
    )

    ax.set_ylabel(
        "Mean images per identity"
    )

    ax.set_xlabel(
        "Modal predicted race"
    )

    ax.set_title(
        "LFW: Mean Identity Depth by Modal Predicted Race"
    )

    ax.tick_params(
        axis="x",
        rotation=30,
    )

    for bar, value in zip(
        bars,
        frame[
            "mean_images_per_identity"
        ],
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_figure(
        fig,
        "11_identity_depth_by_predicted_race.png",
    )


# ============================================================
# EXAMPLE GRID
# ============================================================

def create_example_grid(
    frame: pd.DataFrame,
    filename: str,
    title: str,
    n: int = 12,
    sort_column: str | None = None,
    ascending: bool = True,
) -> None:

    if frame.empty:
        return

    working = frame.copy()

    if (
        sort_column is not None
        and sort_column
        in working.columns
    ):

        working = working.sort_values(
            sort_column,
            ascending=ascending,
        )

        sample = working.head(
            min(
                n,
                len(
                    working
                ),
            )
        )

    else:

        sample = working.sample(
            n=min(
                n,
                len(
                    working
                ),
            ),
            random_state=SEED,
        )

    columns = 4

    rows = int(
        np.ceil(
            len(
                sample
            )
            / columns
        )
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            13,
            rows * 4,
        ),
    )

    axes = np.asarray(
        axes
    ).reshape(
        -1
    )

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

        image_path = paths.data_abs(
            str(
                row[
                    "original_path"
                ]
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

        title_text = (
            f"Race: {row['predicted_race']} "
            f"({row['predicted_race_confidence']:.2f})\n"
            f"Gender: {row['predicted_gender']} "
            f"({row['predicted_gender_confidence']:.2f})\n"
            f"Age: {row['predicted_age_band']} "
            f"({row['predicted_age_confidence']:.2f})"
        )

        ax.set_title(
            title_text,
            fontsize=8,
        )

    fig.suptitle(
        title,
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        EXAMPLES_DIR / filename,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# EXAMPLES BY RACE
# ============================================================

def create_race_examples(
    predictions: pd.DataFrame,
) -> None:

    for race in RACE_ORDER:

        frame = predictions[
            predictions[
                "predicted_race"
            ].eq(
                race
            )
        ]

        if frame.empty:
            continue

        safe_name = (
            race
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

        create_example_grid(
            frame,
            filename=(
                f"race_{safe_name}_high_confidence.png"
            ),
            title=(
                f"LFW Examples Predicted as {race}"
            ),
            n=8,
            sort_column=(
                "predicted_race_confidence"
            ),
            ascending=False,
        )


# ============================================================
# LOW-CONFIDENCE EXAMPLES
# ============================================================

def create_low_confidence_examples(
    predictions: pd.DataFrame,
) -> None:

    create_example_grid(
        predictions,
        "low_confidence_race_predictions.png",
        "LFW: Lowest-Confidence Race Predictions",
        n=12,
        sort_column=(
            "predicted_race_confidence"
        ),
        ascending=True,
    )

    create_example_grid(
        predictions,
        "low_confidence_gender_predictions.png",
        "LFW: Lowest-Confidence Gender Predictions",
        n=12,
        sort_column=(
            "predicted_gender_confidence"
        ),
        ascending=True,
    )

    create_example_grid(
        predictions,
        "low_confidence_age_predictions.png",
        "LFW: Lowest-Confidence Age Predictions",
        n=12,
        sort_column=(
            "predicted_age_confidence"
        ),
        ascending=True,
    )


# ============================================================
# INCONSISTENT IDENTITY EXAMPLES
# ============================================================

def create_inconsistent_identity_examples(
    predictions: pd.DataFrame,
    identity_summary: pd.DataFrame,
) -> None:

    repeated = identity_summary[
        identity_summary[
            "n_images"
        ] >= 2
    ].copy()

    repeated = repeated.sort_values(
        [
            "race_prediction_consistency",
            "gender_prediction_consistency",
        ]
    )

    candidates = repeated.head(
        6
    )

    if candidates.empty:
        return

    selected = predictions[
        predictions[
            "subject_id"
        ].isin(
            candidates[
                "subject_id"
            ]
        )
    ].copy()

    subjects = list(
        candidates[
            "subject_id"
        ]
    )

    max_images = 5

    fig, axes = plt.subplots(
        len(subjects),
        max_images,
        figsize=(
            15,
            3.5 * len(
                subjects
            ),
        ),
    )

    if len(
        subjects
    ) == 1:

        axes = np.asarray(
            [
                axes
            ]
        )

    for row_number, subject in enumerate(
        subjects
    ):

        group = selected[
            selected[
                "subject_id"
            ].eq(
                subject
            )
        ].head(
            max_images
        )

        for column in range(
            max_images
        ):

            ax = axes[
                row_number,
                column,
            ]

            ax.axis(
                "off"
            )

            if column >= len(
                group
            ):
                continue

            row = group.iloc[
                column
            ]

            image_path = paths.data_abs(
                str(
                    row[
                        "original_path"
                    ]
                )
            )

            with Image.open(
                image_path
            ) as image:

                ax.imshow(
                    image.convert(
                        "RGB"
                    )
                )

            ax.set_title(
                (
                    f"{row['predicted_race']}\n"
                    f"{row['predicted_gender']}\n"
                    f"{row['predicted_age_band']}"
                ),
                fontsize=8,
            )

            ax.axis(
                "off"
            )

        axes[
            row_number,
            0,
        ].set_ylabel(
            str(
                subject
            ),
            fontsize=8,
        )

    fig.suptitle(
        "LFW: Examples of Within-Identity Prediction Variation",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        EXAMPLES_DIR
        / "within_identity_prediction_variation.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# KEY FINDINGS TABLE
# ============================================================

def build_key_findings(
    predictions: pd.DataFrame,
    race_distribution: pd.DataFrame,
    gender_distribution: pd.DataFrame,
    confidence: pd.DataFrame,
    repeated_summary: pd.DataFrame,
) -> pd.DataFrame:

    dominant_race = (
        race_distribution
        .sort_values(
            "n",
            ascending=False,
        )
        .iloc[
            0
        ]
    )

    dominant_gender = (
        gender_distribution
        .sort_values(
            "n",
            ascending=False,
        )
        .iloc[
            0
        ]
    )

    race_conf = confidence[
        confidence[
            "task"
        ].eq(
            "race"
        )
    ].iloc[
        0
    ]

    gender_conf = confidence[
        confidence[
            "task"
        ].eq(
            "gender"
        )
    ].iloc[
        0
    ]

    age_conf = confidence[
        confidence[
            "task"
        ].eq(
            "age"
        )
    ].iloc[
        0
    ]

    rows = [
        {
            "finding": (
                "LFW images processed"
            ),
            "value": int(
                len(
                    predictions
                )
            ),
            "group": "",
        },

        {
            "finding": (
                "Largest predicted race category"
            ),
            "value": float(
                dominant_race[
                    "proportion"
                ]
            ),
            "group": (
                dominant_race[
                    "predicted_race"
                ]
            ),
        },

        {
            "finding": (
                "Largest predicted gender category"
            ),
            "value": float(
                dominant_gender[
                    "proportion"
                ]
            ),
            "group": (
                dominant_gender[
                    "predicted_gender"
                ]
            ),
        },

        {
            "finding": (
                "Mean race confidence"
            ),
            "value": float(
                race_conf[
                    "mean_confidence"
                ]
            ),
            "group": "",
        },

        {
            "finding": (
                "Mean gender confidence"
            ),
            "value": float(
                gender_conf[
                    "mean_confidence"
                ]
            ),
            "group": "",
        },

        {
            "finding": (
                "Mean age confidence"
            ),
            "value": float(
                age_conf[
                    "mean_confidence"
                ]
            ),
            "group": "",
        },
    ]

    if not repeated_summary.empty:

        row = repeated_summary.iloc[
            0
        ]

        rows.extend(
            [
                {
                    "finding": (
                        "Repeated identities"
                    ),
                    "value": int(
                        row[
                            "n_repeated_identities"
                        ]
                    ),
                    "group": "",
                },

                {
                    "finding": (
                        "Mean repeated-identity race consistency"
                    ),
                    "value": float(
                        row[
                            "mean_race_prediction_consistency"
                        ]
                    ),
                    "group": "",
                },

                {
                    "finding": (
                        "Mean repeated-identity gender consistency"
                    ),
                    "value": float(
                        row[
                            "mean_gender_prediction_consistency"
                        ]
                    ),
                    "group": "",
                },

                {
                    "finding": (
                        "Mean repeated-identity age consistency"
                    ),
                    "value": float(
                        row[
                            "mean_age_prediction_consistency"
                        ]
                    ),
                    "group": "",
                },
            ]
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# WEBSITE SUMMARY TABLE
# ============================================================

def build_website_summary(
    key_findings: pd.DataFrame,
) -> pd.DataFrame:

    frame = key_findings.copy()

    frame[
        "display_value"
    ] = frame.apply(
        lambda row: (
            f"{row['value']:.1%}"
            if (
                isinstance(
                    row[
                        "value"
                    ],
                    (float, np.floating),
                )
                and 0 <= row[
                    "value"
                ] <= 1
            )
            else str(
                row[
                    "value"
                ]
            )
        ),
        axis=1,
    )

    return frame[
        [
            "finding",
            "display_value",
            "group",
        ]
    ]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    ensure_directories()

    required = [
        PREDICTIONS_PATH,
        RACE_DISTRIBUTION_PATH,
        GENDER_DISTRIBUTION_PATH,
        AGE_DISTRIBUTION_PATH,
        INTERSECTIONS_PATH,
        CONFIDENCE_PATH,
        IDENTITY_PATH,
        REPEATED_IDENTITY_PATH,
        IDENTITY_DEPTH_PATH,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing required result files:\n"
            + "\n".join(
                str(
                    path
                )
                for path in missing
            )
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    race_distribution = pd.read_csv(
        RACE_DISTRIBUTION_PATH
    )

    gender_distribution = pd.read_csv(
        GENDER_DISTRIBUTION_PATH
    )

    age_distribution = pd.read_csv(
        AGE_DISTRIBUTION_PATH
    )

    intersections = pd.read_csv(
        INTERSECTIONS_PATH
    )

    confidence = pd.read_csv(
        CONFIDENCE_PATH
    )

    identity_summary = pd.read_csv(
        IDENTITY_PATH
    )

    repeated_summary = pd.read_csv(
        REPEATED_IDENTITY_PATH
    )

    identity_depth = pd.read_csv(
        IDENTITY_DEPTH_PATH
    )

    # --------------------------------------------------------
    # FIGURES
    # --------------------------------------------------------

    plot_race_distribution(
        race_distribution
    )

    plot_gender_distribution(
        gender_distribution
    )

    plot_age_distribution(
        age_distribution
    )

    plot_intersection_heatmap(
        intersections
    )

    plot_intersection_proportions(
        intersections
    )

    plot_confidence_summary(
        confidence
    )

    plot_confidence_distributions(
        predictions
    )

    plot_race_confidence_by_class(
        predictions
    )

    plot_identity_consistency(
        identity_summary
    )

    plot_consistency_vs_depth(
        identity_summary
    )

    plot_identity_depth_by_race(
        identity_depth
    )

    # --------------------------------------------------------
    # EXAMPLE IMAGES
    # --------------------------------------------------------

    create_race_examples(
        predictions
    )

    create_low_confidence_examples(
        predictions
    )

    create_inconsistent_identity_examples(
        predictions,
        identity_summary,
    )

    # --------------------------------------------------------
    # KEY TABLES
    # --------------------------------------------------------

    key_findings = build_key_findings(
        predictions,
        race_distribution,
        gender_distribution,
        confidence,
        repeated_summary,
    )

    key_findings.to_csv(
        TABLES_DIR
        / "lfw_key_findings.csv",
        index=False,
    )

    website_summary = build_website_summary(
        key_findings
    )

    website_summary.to_csv(
        TABLES_DIR
        / "lfw_website_summary.csv",
        index=False,
    )

    intersections.to_csv(
        TABLES_DIR
        / "lfw_predicted_intersections.csv",
        index=False,
    )

    # --------------------------------------------------------
    # CONSOLE SUMMARY
    # --------------------------------------------------------

    print()
    print(
        "LFW FairFace report pack complete"
    )

    print(
        "========================================"
    )

    print()
    print(
        "KEY FINDINGS"
    )

    print(
        key_findings.to_string(
            index=False
        )
    )

    print()
    print(
        f"Figures:"
    )

    print(
        FIGURES_DIR.resolve()
    )

    print()
    print(
        f"Examples:"
    )

    print(
        EXAMPLES_DIR.resolve()
    )

    print()
    print(
        f"Tables:"
    )

    print(
        TABLES_DIR.resolve()
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "All race, gender and age categories in this "
        "LFW pack are FairFace model predictions, "
        "not independently verified LFW demographic labels."
    )


if __name__ == "__main__":
    main()
    