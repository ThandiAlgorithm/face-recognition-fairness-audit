from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_AUDIT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "model_audit"
)

TRACK_A_ROOT = (
    MODEL_AUDIT_ROOT
    / "model_comparison"
    / "track_a_final"
)

FAIRFACE_COMPARISON_ROOT = (
    MODEL_AUDIT_ROOT
    / "model_comparison"
    / "fairface_validation"
)

QUALITATIVE_ROOT = (
    TRACK_A_ROOT
    / "qualitative_examples"
)


# ============================================================
# GENERIC LOADERS
# ============================================================

@st.cache_data(show_spinner=False)
def _load_csv(path_string: str) -> pd.DataFrame:
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def load_csv(path: Path) -> pd.DataFrame:
    return _load_csv(
        str(path)
    )


# ============================================================
# TRACK A — FINAL COMPARISON
# ============================================================

def load_labelled_dataset_metrics() -> pd.DataFrame:
    return load_csv(
        TRACK_A_ROOT
        / "01_labelled_dataset_model_metrics.csv"
    )


def load_gender_equalized_odds() -> pd.DataFrame:
    return load_csv(
        TRACK_A_ROOT
        / "02_gender_equalized_odds_summary.csv"
    )


def load_gender_fairness_by_race() -> pd.DataFrame:
    return load_csv(
        TRACK_A_ROOT
        / "03_gender_fairness_by_race.csv"
    )


def load_lfw_cross_model_agreement() -> pd.DataFrame:
    return load_csv(
        TRACK_A_ROOT
        / "04_lfw_cross_model_agreement.csv"
    )


def load_lfw_model_assigned_distributions() -> pd.DataFrame:
    return load_csv(
        TRACK_A_ROOT
        / "05_lfw_model_assigned_distributions.csv"
    )


def load_lfw_distribution_distance() -> pd.DataFrame:
    return load_csv(
        TRACK_A_ROOT
        / "06_lfw_distribution_distance.csv"
    )


def load_lfw_cross_model_race_matrix() -> pd.DataFrame:
    return load_csv(
        TRACK_A_ROOT
        / "07_lfw_cross_model_race_matrix.csv"
    )


def load_lfw_identity_comparison() -> pd.DataFrame:
    return load_csv(
        TRACK_A_ROOT
        / "08_lfw_repeated_identity_model_comparison.csv"
    )


def load_lfw_identity_consistency() -> pd.DataFrame:
    return load_csv(
        TRACK_A_ROOT
        / "09_lfw_identity_consistency_summary.csv"
    )


def load_lfw_modal_agreement() -> pd.DataFrame:
    return load_csv(
        TRACK_A_ROOT
        / "10_lfw_identity_modal_agreement.csv"
    )


def load_track_a_key_findings() -> pd.DataFrame:
    return load_csv(
        TRACK_A_ROOT
        / "11_track_a_key_findings.csv"
    )


# ============================================================
# QUALITATIVE AUDIT
# ============================================================

def load_qualitative_manifest(
    dataset: str | None = None,
) -> pd.DataFrame:

    if dataset is None:
        path = (
            QUALITATIVE_ROOT
            / "all_samples_manifest.csv"
        )
    else:
        path = (
            QUALITATIVE_ROOT
            / dataset
            / "sample_manifest.csv"
        )

    return load_csv(
        path
    )


def qualitative_contact_sheet(
    dataset: str,
    page: int,
) -> Path:

    return (
        QUALITATIVE_ROOT
        / dataset
        / f"contact_sheet_{page:02d}.png"
    )


# ============================================================
# FIGURE PATHS
# ============================================================

def track_a_figure(
    filename: str,
) -> Path:

    return (
        TRACK_A_ROOT
        / filename
    )


def fairface_comparison_figure(
    filename: str,
) -> Path:

    return (
        FAIRFACE_COMPARISON_ROOT
        / filename
    )


def model_audit_artifact(
    *parts: str,
) -> Path:

    return MODEL_AUDIT_ROOT.joinpath(
        *parts
    )
