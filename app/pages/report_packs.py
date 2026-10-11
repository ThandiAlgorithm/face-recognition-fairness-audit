from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui import (
    evidence_note,
    metric_row,
    page_header,
    research_card,
    section_header,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "model_audit"
)

COMPARISON_ROOT = (
    MODEL_ROOT
    / "model_comparison"
    / "track_a_final"
)

QUALITATIVE_ROOT = (
    COMPARISON_ROOT
    / "qualitative_examples"
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_csv(
    path_string: str,
) -> pd.DataFrame:

    path = Path(
        path_string
    )

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        low_memory=False,
    )


def model_path(
    value: str,
) -> Path:

    return (
        MODEL_ROOT
        / value
    )


def available_data_files(
    roots: list[Path],
) -> list[Path]:

    files = []

    for root in roots:

        if not root.exists():
            continue

        files.extend(
            root.rglob(
                "*.csv"
            )
        )

        files.extend(
            root.rglob(
                "*.json"
            )
        )

    return sorted(
        set(files),
        key=lambda item: (
            str(item).lower()
        ),
    )


def show_result_file(
    path: Path,
) -> None:

    suffix = (
        path.suffix
        .lower()
    )

    if suffix == ".csv":

        frame = load_csv(
            str(path)
        )

        st.caption(
            f"{len(frame):,} rows"
        )

        st.dataframe(
            frame,
            hide_index=True,
            use_container_width=True,
            height=520,
        )

    elif suffix == ".json":

        try:

            st.json(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:

            st.warning(
                "The JSON file could not be displayed."
            )


def show_figure(
    path: Path,
    title: str,
    description: str,
) -> None:

    with st.container(
        border=True
    ):

        st.markdown(
            f"### {title}"
        )

        if path.exists():

            st.image(
                str(path),
                use_container_width=True,
            )

        else:

            st.info(
                "Figure is not available "
                "in the current environment."
            )

        st.write(
            description
        )


# ============================================================
# EVIDENCE COLLECTIONS
# ============================================================

EVIDENCE = {
    "Demographic measurement summary": {
        "description": (
            "Consolidated comparison of FairFace ResNet-34 and "
            "DeepFace across FairFace, UTKFace and LFW."
        ),
        "roots": [
            model_path(
                "model_comparison/"
                "track_a_final"
            )
        ],
        "figures": [
            (
                model_path(
                    "model_comparison/"
                    "track_a_final/"
                    "01_gender_accuracy_comparison.png"
                ),
                "Gender classification performance",
                (
                    "Comparison of gender-classification accuracy on "
                    "the independently labelled FairFace and UTKFace "
                    "evaluation sets."
                ),
            ),
            (
                model_path(
                    "model_comparison/"
                    "track_a_final/"
                    "02_race_accuracy_comparison.png"
                ),
                "Race classification performance",
                (
                    "Within-dataset comparison of harmonised race "
                    "classification accuracy."
                ),
            ),
            (
                model_path(
                    "model_comparison/"
                    "track_a_final/"
                    "03_gender_equalized_odds_comparison.png"
                ),
                "Gender equalized-odds difference",
                (
                    "Measured variation in female true-positive and "
                    "false-positive rates across labelled race groups."
                ),
            ),
            (
                model_path(
                    "model_comparison/"
                    "track_a_final/"
                    "06_lfw_model_agreement.png"
                ),
                "Demographic-model agreement on LFW",
                (
                    "Direct comparison of demographic assignments made "
                    "by the two classifiers on the same LFW images."
                ),
            ),
            (
                model_path(
                    "model_comparison/"
                    "track_a_final/"
                    "07_lfw_identity_consistency_comparison.png"
                ),
                "Repeated-identity consistency",
                (
                    "Stability of demographic assignments across "
                    "multiple photographs of the same LFW identity."
                ),
            ),
        ],
    },

    "FairFace model on FairFace": {
        "description": (
            "Held-out FairFace validation evidence for the "
            "FairFace ResNet-34 demographic classifier."
        ),
        "roots": [
            model_path(
                "fairface_resnet34/"
                "fairface_validation"
            )
        ],
        "figures": [
            (
                model_path(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "race_confusion_matrix.png"
                ),
                "Race confusion matrix",
                (
                    "Native FairFace race-category classification "
                    "behaviour."
                ),
            ),
            (
                model_path(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "gender_confusion_matrix.png"
                ),
                "Gender confusion matrix",
                (
                    "Male/Female classification behaviour on held-out "
                    "FairFace images."
                ),
            ),
            (
                model_path(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "report_pack/figures/"
                    "03_intersectional_race_accuracy.png"
                ),
                "Intersectional race accuracy",
                (
                    "Race-classification performance after separating "
                    "samples by demographic intersection."
                ),
            ),
            (
                model_path(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "report_pack/figures/"
                    "05_gender_equalized_odds_by_race.png"
                ),
                "Gender error behaviour by race",
                (
                    "Female TPR and FPR across labelled race groups."
                ),
            ),
        ],
    },

    "FairFace model on UTKFace": {
        "description": (
            "Cross-dataset demographic evaluation of FairFace "
            "ResNet-34 on aligned UTKFace images."
        ),
        "roots": [
            model_path(
                "fairface_resnet34/"
                "utkface_aligned"
            )
        ],
        "figures": [
            (
                model_path(
                    "fairface_resnet34/"
                    "utkface_aligned/"
                    "race_confusion_matrix_harmonised.png"
                ),
                "Harmonised race confusion matrix",
                (
                    "Four-class comparison using categories compatible "
                    "between UTKFace and the classifier."
                ),
            ),
            (
                model_path(
                    "fairface_resnet34/"
                    "utkface_aligned/"
                    "gender_confusion_matrix.png"
                ),
                "Gender confusion matrix",
                (
                    "Gender-classification behaviour following "
                    "cross-dataset transfer."
                ),
            ),
        ],
    },

    "DeepFace on FairFace": {
        "description": (
            "DeepFace demographic classification on the same held-out "
            "FairFace validation population."
        ),
        "roots": [
            model_path(
                "deepface/"
                "fairface_validation"
            )
        ],
        "figures": [
            (
                model_path(
                    "deepface/"
                    "fairface_validation/"
                    "race_confusion_matrix.png"
                ),
                "Race confusion matrix",
                (
                    "DeepFace classification behaviour in the "
                    "harmonised FairFace race evaluation."
                ),
            ),
            (
                model_path(
                    "deepface/"
                    "fairface_validation/"
                    "gender_confusion_matrix.png"
                ),
                "Gender confusion matrix",
                (
                    "DeepFace gender errors on the same FairFace "
                    "validation images."
                ),
            ),
        ],
    },

    "DeepFace on UTKFace": {
        "description": (
            "DeepFace demographic classification on the aligned "
            "UTKFace evaluation population."
        ),
        "roots": [
            model_path(
                "deepface/"
                "utkface_aligned"
            )
        ],
        "figures": [
            (
                model_path(
                    "deepface/"
                    "utkface_aligned/"
                    "race_confusion_matrix_harmonised.png"
                ),
                "Harmonised race confusion matrix",
                (
                    "Four-class UTKFace race evaluation."
                ),
            ),
            (
                model_path(
                    "deepface/"
                    "utkface_aligned/"
                    "gender_confusion_matrix.png"
                ),
                "Gender confusion matrix",
                (
                    "DeepFace gender behaviour on UTKFace."
                ),
            ),
        ],
    },

    "LFW demographic sensitivity": {
        "description": (
            "Prediction-only demographic evidence for LFW. These "
            "outputs measure annotation sensitivity rather than "
            "demographic accuracy."
        ),
        "roots": [
            model_path(
                "fairface_resnet34/"
                "lfw_predictions"
            ),
            model_path(
                "deepface/"
                "lfw_aligned"
            ),
        ],
        "figures": [
            (
                model_path(
                    "model_comparison/"
                    "track_a_final/"
                    "04_lfw_race_distribution_comparison.png"
                ),
                "Model-assigned race composition",
                (
                    "Comparison of inferred LFW race composition under "
                    "the two annotation models."
                ),
            ),
            (
                model_path(
                    "model_comparison/"
                    "track_a_final/"
                    "05_lfw_gender_distribution_comparison.png"
                ),
                "Model-assigned gender composition",
                (
                    "Comparison of inferred LFW gender composition."
                ),
            ),
            (
                model_path(
                    "model_comparison/"
                    "track_a_final/"
                    "08_lfw_cross_model_race_matrix.png"
                ),
                "Cross-model race assignment matrix",
                (
                    "Agreement and disagreement between race categories "
                    "assigned by the two models."
                ),
            ),
        ],
    },
}


# ============================================================
# HEADER
# ============================================================

page_header(
    "Data & Methods",
    (
        "Technical evidence, experimental artefacts and underlying "
        "result tables supporting the analyses presented throughout "
        "the website."
    ),
    "Technical reference",
)


# ============================================================
# OVERVIEW
# ============================================================

section_header(
    "Evidence archive",
    (
        "This section preserves detailed outputs without placing every "
        "diagnostic figure on the main presentation pages."
    ),
)

metric_row(
    [
        (
            "Source datasets",
            "3",
        ),
        (
            "Demographic models",
            "2",
        ),
        (
            "Recognition reference",
            "ArcFace",
        ),
        (
            "Final dataset",
            "HarmoFace-95K",
        ),
    ]
)

st.info(
    """
    The main research pages present the most important findings.
    This section provides access to the supporting figures, tables and
    diagnostic outputs for technical review and reproducibility.
    """
)

st.divider()


# ============================================================
# EVIDENCE BROWSER
# ============================================================

section_header(
    "Result explorer"
)

selected_name = st.selectbox(
    "Evidence collection",
    list(
        EVIDENCE.keys()
    ),
)

collection = EVIDENCE[
    selected_name
]

st.write(
    collection[
        "description"
    ]
)

figures_available = [
    item
    for item
    in collection[
        "figures"
    ]
    if item[0].exists()
]

files = available_data_files(
    collection[
        "roots"
    ]
)

metric_row(
    [
        (
            "Figures available",
            str(
                len(
                    figures_available
                )
            ),
        ),
        (
            "Tables / summaries",
            str(
                len(
                    files
                )
            ),
        ),
    ]
)

st.divider()


# ============================================================
# FIGURES
# ============================================================

if figures_available:

    section_header(
        "Figures"
    )

    figure_names = {
        title: (
            path,
            description,
        )
        for (
            path,
            title,
            description,
        )
        in figures_available
    }

    selected_figure = st.selectbox(
        "Figure",
        list(
            figure_names.keys()
        ),
    )

    path, description = (
        figure_names[
            selected_figure
        ]
    )

    show_figure(
        path,
        selected_figure,
        description,
    )


# ============================================================
# TABLES
# ============================================================

if files:

    section_header(
        "Underlying results"
    )

    options = {
        str(
            path.relative_to(
                MODEL_ROOT
            )
        ): path
        for path in files
    }

    selected_file_name = st.selectbox(
        "Result table or summary",
        list(
            options.keys()
        ),
    )

    selected_file = options[
        selected_file_name
    ]

    show_result_file(
        selected_file
    )

st.divider()


# ============================================================
# QUALITATIVE EVIDENCE
# ============================================================

section_header(
    "Qualitative image review",
    (
        "Deterministic samples support manual inspection of model "
        "agreement and disagreement."
    ),
)

dataset_name = st.segmented_control(
    "Dataset",
    [
        "FairFace",
        "UTKFace",
        "LFW",
    ],
    default="FairFace",
)

dataset_key = {
    "FairFace": "fairface",
    "UTKFace": "utkface",
    "LFW": "lfw",
}[
    dataset_name
]

dataset_root = (
    QUALITATIVE_ROOT
    / dataset_key
)

manifest_path = (
    dataset_root
    / "sample_manifest.csv"
)

manifest = (
    load_csv(
        str(
            manifest_path
        )
    )
    if manifest_path.exists()
    else pd.DataFrame()
)

view_mode = st.radio(
    "View",
    [
        "Contact sheet",
        "Manifest",
    ],
    horizontal=True,
)

if view_mode == "Contact sheet":

    page = st.radio(
        "Page",
        [
            1,
            2,
            3,
        ],
        horizontal=True,
    )

    sheet = (
        dataset_root
        / f"contact_sheet_{page:02d}.png"
    )

    if sheet.exists():

        st.image(
            str(sheet),
            use_container_width=True,
            caption=(
                f"{dataset_name} qualitative sample"
            ),
        )

    else:

        st.info(
            "Contact sheet not available."
        )

else:

    if manifest.empty:

        st.info(
            "Qualitative manifest not available."
        )

    else:

        st.dataframe(
            manifest,
            hide_index=True,
            use_container_width=True,
            height=600,
        )

st.caption(
    """
    Qualitative samples support error analysis and illustration.
    They are not replacements for the full-dataset quantitative
    results.
    """
)
