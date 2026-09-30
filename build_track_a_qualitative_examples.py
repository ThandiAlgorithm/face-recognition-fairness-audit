from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src import detection
from src import image_integrity
from src import paths


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 26041

MASTER = Path(
    "metadata/master_annotations.parquet"
)

PREPROCESSING = Path(
    "configs/preprocessing.yaml"
)

OUTPUT = Path(
    "results/model_audit/model_comparison/"
    "track_a_final/qualitative_examples"
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


RACE_6 = [
    "White",
    "Black",
    "Asian",
    "Indian",
    "Latino_Hispanic",
    "Middle Eastern",
]

GENDERS = [
    "Female",
    "Male",
]


RACE_PREDICTION_CANDIDATES = [
    "race_pred_harmonised",
    "race_pred",
    "predicted_race",
    "pred_race",
]

GENDER_PREDICTION_CANDIDATES = [
    "gender_pred",
    "predicted_gender",
    "pred_gender",
]


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


def normalise_race(
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

        "others": "Others",
        "other": "Others",
    }

    return mapping.get(
        text
    )


# ============================================================
# COLUMN DISCOVERY
# ============================================================

def find_column(
    columns: list[str],
    candidates: list[str],
    description: str,
) -> str:

    for column in candidates:

        if column in columns:
            return column

    raise RuntimeError(
        f"Could not locate {description}. "
        f"Available columns: {columns}"
    )


def load_predictions(
    file_path: Path,
    prefix: str,
) -> pd.DataFrame:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n{file_path}"
        )

    header = pd.read_csv(
        file_path,
        nrows=0,
    )

    columns = list(
        header.columns
    )

    race_column = find_column(
        columns,
        RACE_PREDICTION_CANDIDATES,
        f"{prefix} race prediction",
    )

    gender_column = find_column(
        columns,
        GENDER_PREDICTION_CANDIDATES,
        f"{prefix} gender prediction",
    )

    frame = pd.read_csv(
        file_path,
        usecols=[
            "image_id",
            race_column,
            gender_column,
        ],
        low_memory=False,
    )

    frame = frame.rename(
        columns={
            race_column: (
                f"{prefix}_race_raw"
            ),

            gender_column: (
                f"{prefix}_gender_raw"
            ),
        }
    )

    frame[
        f"{prefix}_race"
    ] = frame[
        f"{prefix}_race_raw"
    ].map(
        normalise_race
    )

    frame[
        f"{prefix}_gender"
    ] = frame[
        f"{prefix}_gender_raw"
    ].map(
        normalise_gender
    )

    return frame


# ============================================================
# ALIGNMENT CONFIGURATION
# ============================================================

def load_alignment_config() -> dict:

    with open(
        PREPROCESSING,
        "r",
        encoding="utf-8",
    ) as handle:

        config = yaml.safe_load(
            handle
        )

    alignment = dict(
        config.get(
            "alignment",
            {},
        )
    )

    if "output_size" not in alignment:

        alignment[
            "output_size"
        ] = alignment[
            "output_size_px"
        ]

    if (
        "canonical_landmark_fractions"
        not in alignment
    ):

        alignment[
            "canonical_landmark_fractions"
        ] = alignment[
            "canonical_landmark_positions"
        ]

    config[
        "alignment"
    ] = alignment

    return config


def normalise_landmarks(
    value,
) -> list[list[float]]:

    points = np.asarray(
        value,
        dtype=object,
    ).reshape(-1)

    if len(points) != 5:

        raise RuntimeError(
            f"Expected 5 landmarks; found "
            f"{len(points)}."
        )

    output = []

    for point in points:

        coordinates = np.asarray(
            point,
            dtype=np.float32,
        ).reshape(-1)

        if len(coordinates) != 2:

            raise RuntimeError(
                "Invalid landmark coordinates."
            )

        output.append(
            [
                float(
                    coordinates[0]
                ),
                float(
                    coordinates[1]
                ),
            ]
        )

    return output


# ============================================================
# IMAGE LOADING
# ============================================================

def load_display_image(
    row: pd.Series,
    dataset: str,
    alignment_config: dict,
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

    # FairFace images are already the prepared dataset faces
    # used by the FairFace validation experiment.
    if dataset == "fairface":

        return rgb

    # UTKFace and LFW were evaluated using the project's
    # deterministic five-point alignment reconstructed from
    # the stored landmarks. Display that same model input.
    landmarks = normalise_landmarks(
        row[
            "landmarks"
        ]
    )

    (
        aligned_rgb,
        transform_matrix,
        valid_mask,
    ) = detection.align_face(
        rgb,
        landmarks,
        alignment_config,
    )

    if aligned_rgb is None:

        raise RuntimeError(
            f"Could not align "
            f"{row['image_id']}."
        )

    if transform_matrix is None:

        raise RuntimeError(
            f"Missing transform for "
            f"{row['image_id']}."
        )

    if valid_mask is None:

        raise RuntimeError(
            f"Missing valid mask for "
            f"{row['image_id']}."
        )

    return aligned_rgb


# ============================================================
# DATASET MERGE
# ============================================================

def build_dataset_frame(
    master: pd.DataFrame,
    dataset: str,
    ff_path: Path,
    df_path: Path,
) -> pd.DataFrame:

    frame = master[
        master[
            "dataset"
        ]
        .astype(str)
        .str.lower()
        .eq(
            dataset
        )
    ].copy()

    fairface = load_predictions(
        ff_path,
        "fairface",
    )

    deepface = load_predictions(
        df_path,
        "deepface",
    )

    frame = (
        frame
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

    frame[
        "race_true"
    ] = frame[
        "race"
    ].map(
        normalise_race
    )

    frame[
        "gender_true"
    ] = frame[
        "gender"
    ].map(
        normalise_gender
    )

    frame[
        "race_agreement"
    ] = (
        frame[
            "fairface_race"
        ]
        ==
        frame[
            "deepface_race"
        ]
    )

    frame[
        "gender_agreement"
    ] = (
        frame[
            "fairface_gender"
        ]
        ==
        frame[
            "deepface_gender"
        ]
    )

    return (
        frame
        .sort_values(
            "image_id"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# DETERMINISTIC SAMPLING
# ============================================================

def deterministic_sample(
    frame: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:

    if len(frame) <= n:

        return frame.copy()

    return (
        frame.sample(
            n=n,
            random_state=seed,
        )
        .sort_values(
            "image_id"
        )
        .copy()
    )


def stratified_sample(
    frame: pd.DataFrame,
    group_columns: list[str],
    total_n: int,
    seed: int,
) -> pd.DataFrame:

    groups = list(
        frame.groupby(
            group_columns,
            dropna=False,
            observed=True,
        )
    )

    groups = sorted(
        groups,
        key=lambda item: str(
            item[0]
        ),
    )

    if not groups:

        raise RuntimeError(
            "No groups available for sampling."
        )

    base = (
        total_n
        //
        len(groups)
    )

    remainder = (
        total_n
        %
        len(groups)
    )

    selected = []

    for index, (
        group_key,
        group,
    ) in enumerate(
        groups
    ):

        target = (
            base
            +
            (
                1
                if index < remainder
                else 0
            )
        )

        target = min(
            target,
            len(group),
        )

        if target == 0:

            continue

        sampled = group.sample(
            n=target,
            random_state=(
                seed
                +
                index
            ),
        ).copy()

        sampled[
            "selection_reason"
        ] = (
            "stratified_"
            + "_".join(
                str(value)
                for value
                in (
                    group_key
                    if isinstance(
                        group_key,
                        tuple,
                    )
                    else (
                        group_key,
                    )
                )
            )
        )

        selected.append(
            sampled
        )

    output = pd.concat(
        selected,
        ignore_index=True,
    )

    # If a small group prevented us reaching total_n,
    # fill deterministically from the remaining population.
    if len(output) < total_n:

        remaining = frame[
            ~frame[
                "image_id"
            ].isin(
                output[
                    "image_id"
                ]
            )
        ]

        fill_n = min(
            total_n
            -
            len(output),
            len(remaining),
        )

        if fill_n > 0:

            fill = remaining.sample(
                n=fill_n,
                random_state=seed + 999,
            ).copy()

            fill[
                "selection_reason"
            ] = (
                "stratified_fill"
            )

            output = pd.concat(
                [
                    output,
                    fill,
                ],
                ignore_index=True,
            )

    return (
        output
        .sort_values(
            "image_id"
        )
        .head(
            total_n
        )
        .reset_index(
            drop=True
        )
    )


def select_lfw_diagnostic_sample(
    frame: pd.DataFrame,
    total_n: int,
) -> pd.DataFrame:

    if total_n != 30:

        # For arbitrary requested sample sizes,
        # fall back to deterministic sampling.
        output = deterministic_sample(
            frame,
            total_n,
            SEED,
        )

        output[
            "selection_reason"
        ] = (
            "deterministic_sample"
        )

        return output

    # 10 examples where race differs but gender agrees.
    race_disagreement = frame[
        (
            ~frame[
                "race_agreement"
            ]
        )
        &
        (
            frame[
                "gender_agreement"
            ]
        )
    ]

    part_a = deterministic_sample(
        race_disagreement,
        10,
        SEED,
    )

    part_a[
        "selection_reason"
    ] = (
        "race_disagreement"
    )

    used = set(
        part_a[
            "image_id"
        ]
    )

    # 10 examples where gender differs.
    gender_disagreement = frame[
        (
            ~frame[
                "gender_agreement"
            ]
        )
        &
        (
            ~frame[
                "image_id"
            ].isin(
                used
            )
        )
    ]

    part_b = deterministic_sample(
        gender_disagreement,
        10,
        SEED + 1,
    )

    part_b[
        "selection_reason"
    ] = (
        "gender_disagreement"
    )

    used.update(
        part_b[
            "image_id"
        ]
    )

    # 10 examples where both models agree.
    full_agreement = frame[
        (
            frame[
                "race_agreement"
            ]
        )
        &
        (
            frame[
                "gender_agreement"
            ]
        )
        &
        (
            ~frame[
                "image_id"
            ].isin(
                used
            )
        )
    ]

    part_c = deterministic_sample(
        full_agreement,
        10,
        SEED + 2,
    )

    part_c[
        "selection_reason"
    ] = (
        "race_and_gender_agreement"
    )

    output = pd.concat(
        [
            part_a,
            part_b,
            part_c,
        ],
        ignore_index=True,
    )

    if len(output) != 30:

        raise RuntimeError(
            "Could not construct the requested "
            "30-image LFW diagnostic sample."
        )

    return output.reset_index(
        drop=True
    )


def select_sample(
    frame: pd.DataFrame,
    dataset: str,
    n: int,
    strategy: str,
) -> pd.DataFrame:

    if strategy == "first":

        output = (
            frame
            .sort_values(
                "image_id"
            )
            .head(
                n
            )
            .copy()
        )

        output[
            "selection_reason"
        ] = (
            "first_by_image_id"
        )

        return output.reset_index(
            drop=True
        )

    if dataset == "fairface":

        labelled = frame[
            frame[
                "race_true"
            ].notna()
            &
            frame[
                "gender_true"
            ].notna()
        ].copy()

        return stratified_sample(
            labelled,
            [
                "race_true",
                "gender_true",
            ],
            n,
            SEED,
        )

    if dataset == "utkface":

        labelled = frame[
            frame[
                "race_true"
            ].notna()
            &
            frame[
                "gender_true"
            ].notna()
        ].copy()

        return stratified_sample(
            labelled,
            [
                "race_true",
                "gender_true",
            ],
            n,
            SEED + 100,
        )

    if dataset == "lfw":

        return select_lfw_diagnostic_sample(
            frame,
            n,
        )

    raise ValueError(
        f"Unsupported dataset: {dataset}"
    )


# ============================================================
# CAPTIONS
# ============================================================

def correctness_text(
    truth,
    prediction,
) -> str:

    if (
        truth is None
        or pd.isna(
            truth
        )
    ):

        return ""

    return (
        "OK"
        if truth == prediction
        else "ERR"
    )


def caption_for_row(
    row: pd.Series,
    dataset: str,
) -> str:

    if dataset in {
        "fairface",
        "utkface",
    }:

        ff_race_status = correctness_text(
            row[
                "race_true"
            ],
            row[
                "fairface_race"
            ],
        )

        ff_gender_status = correctness_text(
            row[
                "gender_true"
            ],
            row[
                "fairface_gender"
            ],
        )

        df_race_status = correctness_text(
            row[
                "race_true"
            ],
            row[
                "deepface_race"
            ],
        )

        df_gender_status = correctness_text(
            row[
                "gender_true"
            ],
            row[
                "deepface_gender"
            ],
        )

        return (
            f"ID: {row['image_id']}\n"
            f"Truth: {row['race_true']} | "
            f"{row['gender_true']}\n"
            f"FairFace: {row['fairface_race']} "
            f"[{ff_race_status}] | "
            f"{row['fairface_gender']} "
            f"[{ff_gender_status}]\n"
            f"DeepFace: {row['deepface_race']} "
            f"[{df_race_status}] | "
            f"{row['deepface_gender']} "
            f"[{df_gender_status}]"
        )

    return (
        f"ID: {row['image_id']}\n"
        f"FairFace: {row['fairface_race']} | "
        f"{row['fairface_gender']}\n"
        f"DeepFace: {row['deepface_race']} | "
        f"{row['deepface_gender']}\n"
        f"Agree — race: "
        f"{'YES' if row['race_agreement'] else 'NO'} | "
        f"gender: "
        f"{'YES' if row['gender_agreement'] else 'NO'}"
    )


# ============================================================
# FILE-SAFE ID
# ============================================================

def safe_image_key(
    image_id: str,
) -> str:

    digest = hashlib.sha256(
        str(
            image_id
        ).encode(
            "utf-8"
        )
    ).hexdigest()[
        :10
    ]

    return digest


# ============================================================
# INDIVIDUAL PANELS
# ============================================================

def save_individual_panel(
    image: np.ndarray,
    row: pd.Series,
    dataset: str,
    index: int,
    output_dir: Path,
) -> Path:

    fig, ax = plt.subplots(
        figsize=(
            7,
            7,
        )
    )

    ax.imshow(
        image
    )

    ax.axis(
        "off"
    )

    ax.set_title(
        caption_for_row(
            row,
            dataset,
        ),
        fontsize=10,
        loc="left",
        pad=12,
    )

    fig.tight_layout()

    file_name = (
        f"{index:02d}_"
        f"{safe_image_key(row['image_id'])}.png"
    )

    path = (
        output_dir
        / file_name
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


# ============================================================
# CONTACT SHEETS
# ============================================================

def save_contact_sheet(
    items: list[
        tuple[
            np.ndarray,
            pd.Series,
        ]
    ],
    dataset: str,
    page_number: int,
    output_dir: Path,
) -> Path:

    # 10 examples per page:
    # 5 columns × 2 rows.
    columns = 5
    rows = 2

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            24,
            11,
        ),
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    for axis in axes:

        axis.axis(
            "off"
        )

    for axis, (
        image,
        row,
    ) in zip(
        axes,
        items,
    ):

        axis.imshow(
            image
        )

        axis.axis(
            "off"
        )

        axis.set_title(
            caption_for_row(
                row,
                dataset,
            ),
            fontsize=8,
            loc="left",
            pad=8,
        )

    dataset_title = {
        "fairface": (
            "FairFace Validation"
        ),
        "utkface": (
            "UTKFace Aligned"
        ),
        "lfw": (
            "LFW Prediction-Only"
        ),
    }[
        dataset
    ]

    fig.suptitle(
        (
            f"Track A Qualitative Audit — "
            f"{dataset_title} — "
            f"Page {page_number}"
        ),
        fontsize=16,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    path = (
        output_dir
        / (
            f"contact_sheet_"
            f"{page_number:02d}.png"
        )
    )

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


# ============================================================
# MANIFEST
# ============================================================

def build_manifest_row(
    row: pd.Series,
    dataset: str,
    sample_index: int,
    panel_path: Path,
) -> dict:

    output = {
        "dataset": dataset,
        "sample_index": (
            sample_index
        ),
        "image_id": (
            row[
                "image_id"
            ]
        ),
        "selection_reason": (
            row[
                "selection_reason"
            ]
        ),
        "fairface_race": (
            row[
                "fairface_race"
            ]
        ),
        "fairface_gender": (
            row[
                "fairface_gender"
            ]
        ),
        "deepface_race": (
            row[
                "deepface_race"
            ]
        ),
        "deepface_gender": (
            row[
                "deepface_gender"
            ]
        ),
        "race_model_agreement": bool(
            row[
                "race_agreement"
            ]
        ),
        "gender_model_agreement": bool(
            row[
                "gender_agreement"
            ]
        ),
        "panel_path": str(
            panel_path
        ),
    }

    if dataset in {
        "fairface",
        "utkface",
    }:

        output[
            "ground_truth_race"
        ] = row[
            "race_true"
        ]

        output[
            "ground_truth_gender"
        ] = row[
            "gender_true"
        ]

        output[
            "fairface_race_correct"
        ] = bool(
            row[
                "fairface_race"
            ]
            ==
            row[
                "race_true"
            ]
        )

        output[
            "fairface_gender_correct"
        ] = bool(
            row[
                "fairface_gender"
            ]
            ==
            row[
                "gender_true"
            ]
        )

        output[
            "deepface_race_correct"
        ] = bool(
            row[
                "deepface_race"
            ]
            ==
            row[
                "race_true"
            ]
        )

        output[
            "deepface_gender_correct"
        ] = bool(
            row[
                "deepface_gender"
            ]
            ==
            row[
                "gender_true"
            ]
        )

    else:

        output[
            "ground_truth_race"
        ] = None

        output[
            "ground_truth_gender"
        ] = None

        output[
            "fairface_race_correct"
        ] = None

        output[
            "fairface_gender_correct"
        ] = None

        output[
            "deepface_race_correct"
        ] = None

        output[
            "deepface_gender_correct"
        ] = None

    return output


# ============================================================
# DATASET PACK
# ============================================================

def build_dataset_pack(
    frame: pd.DataFrame,
    dataset: str,
    alignment_config: dict,
    n: int,
    strategy: str,
) -> pd.DataFrame:

    sample = select_sample(
        frame,
        dataset,
        n,
        strategy,
    )

    dataset_output = (
        OUTPUT
        / dataset
    )

    individual_output = (
        dataset_output
        / "individual"
    )

    individual_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        f"{dataset.upper()}: "
        f"selected {len(sample)} examples"
    )

    rendered = []

    manifest_rows = []

    for position, (
        _,
        row,
    ) in enumerate(
        sample.iterrows(),
        start=1,
    ):

        image = load_display_image(
            row,
            dataset,
            alignment_config,
        )

        panel_path = (
            save_individual_panel(
                image,
                row,
                dataset,
                position,
                individual_output,
            )
        )

        rendered.append(
            (
                image,
                row,
            )
        )

        manifest_rows.append(
            build_manifest_row(
                row,
                dataset,
                position,
                panel_path,
            )
        )

        print(
            f"  rendered "
            f"{position:02d}/"
            f"{len(sample):02d}"
        )

    # Three contact sheets of 10 examples.
    page_size = 10

    for start in range(
        0,
        len(rendered),
        page_size,
    ):

        page_number = (
            start
            //
            page_size
            +
            1
        )

        save_contact_sheet(
            rendered[
                start:
                start
                +
                page_size
            ],
            dataset,
            page_number,
            dataset_output,
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        dataset_output
        / "sample_manifest.csv",
        index=False,
    )

    return manifest


# ============================================================
# SUMMARY
# ============================================================

def write_readme(
    manifests: pd.DataFrame,
    strategy: str,
    n: int,
) -> None:

    lines = [
        "# Track A Qualitative Example Pack",
        "",
        (
            f"Selection strategy: `{strategy}`. "
            f"Target sample size: {n} images per dataset."
        ),
        "",
        (
            "The same selected image is shown with both "
            "FairFace ResNet-34 and DeepFace demographic "
            "predictions. FairFace and UTKFace panels also "
            "show dataset ground-truth labels."
        ),
        "",
        (
            "LFW panels are prediction-only because the "
            "canonical project master table contains no "
            "independent LFW demographic ground truth. "
            "The displayed race and gender values are "
            "model-assigned labels, not verified attributes."
        ),
        "",
        (
            "These examples are intended for qualitative "
            "inspection and error analysis. They do not "
            "replace the full-dataset quantitative metrics."
        ),
        "",
        "## Files",
        "",
        (
            "- Each dataset folder contains three contact "
            "sheets of ten examples."
        ),
        (
            "- Each `individual/` folder contains one labelled "
            "PNG per selected image."
        ),
        (
            "- Each dataset contains `sample_manifest.csv`."
        ),
        (
            "- `all_samples_manifest.csv` combines all datasets."
        ),
        "",
    ]

    (
        OUTPUT
        / "README.md"
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

    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic qualitative examples "
            "for the Track A demographic model audit."
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help=(
            "Number of examples per dataset. "
            "Default: 30."
        ),
    )

    parser.add_argument(
        "--strategy",
        choices=[
            "audit",
            "first",
        ],
        default="audit",
        help=(
            "'audit' uses stratified/diagnostic "
            "sampling. 'first' selects the first "
            "images by stable image_id."
        ),
    )

    args = parser.parse_args()

    if args.n <= 0:

        raise ValueError(
            "--n must be positive."
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Track A Qualitative Example Pack"
    )

    print(
        "===================================="
    )

    print(
        f"Sample size per dataset: "
        f"{args.n}"
    )

    print(
        f"Selection strategy: "
        f"{args.strategy}"
    )

    print()

    master = pd.read_parquet(
        MASTER
    )

    alignment_config = (
        load_alignment_config()
    )

    print(
        "Loading model outputs..."
    )

    fairface = build_dataset_frame(
        master,
        "fairface",
        FF_FAIRFACE,
        DF_FAIRFACE,
    )

    utkface = build_dataset_frame(
        master,
        "utkface",
        FF_UTK,
        DF_UTK,
    )

    lfw = build_dataset_frame(
        master,
        "lfw",
        FF_LFW,
        DF_LFW,
    )

    print(
        f"FairFace common images: "
        f"{len(fairface):,}"
    )

    print(
        f"UTKFace common images: "
        f"{len(utkface):,}"
    )

    print(
        f"LFW common images: "
        f"{len(lfw):,}"
    )

    manifests = []

    manifests.append(
        build_dataset_pack(
            fairface,
            "fairface",
            alignment_config,
            args.n,
            args.strategy,
        )
    )

    manifests.append(
        build_dataset_pack(
            utkface,
            "utkface",
            alignment_config,
            args.n,
            args.strategy,
        )
    )

    manifests.append(
        build_dataset_pack(
            lfw,
            "lfw",
            alignment_config,
            args.n,
            args.strategy,
        )
    )

    combined = pd.concat(
        manifests,
        ignore_index=True,
    )

    combined.to_csv(
        OUTPUT
        / "all_samples_manifest.csv",
        index=False,
    )

    write_readme(
        combined,
        args.strategy,
        args.n,
    )

    print()
    print(
        "QUALITATIVE PACK COMPLETE"
    )

    print(
        "===================================="
    )

    print(
        f"Total examples: "
        f"{len(combined):,}"
    )

    print()

    for dataset in [
        "fairface",
        "utkface",
        "lfw",
    ]:

        subset = combined[
            combined[
                "dataset"
            ].eq(
                dataset
            )
        ]

        print(
            f"{dataset:10s}: "
            f"{len(subset):2d} examples"
        )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT.resolve()
    )

    print()
    print(
        "Open each dataset folder and start "
        "with contact_sheet_01.png."
    )


if __name__ == "__main__":
    main()
    