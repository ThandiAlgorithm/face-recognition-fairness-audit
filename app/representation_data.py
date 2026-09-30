from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REP_ROOT = PROJECT_ROOT / "results" / "representation"


@st.cache_data(show_spinner=False)
def load_rep_csv(issue: str, filename: str) -> pd.DataFrame:
    path = REP_ROOT / issue / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


# ============================================================
# REP-001 — image-level representation
# ============================================================

def load_rep001_dataset_summary() -> pd.DataFrame:
    return load_rep_csv(
        "REP-001",
        "rep001_dataset_summary.csv",
    )


def load_rep001_gender() -> pd.DataFrame:
    return load_rep_csv(
        "REP-001",
        "rep001_gender_composition.csv",
    )


def load_rep001_age() -> pd.DataFrame:
    return load_rep_csv(
        "REP-001",
        "rep001_age_band_harmonised_composition.csv",
    )


def load_rep001_race() -> pd.DataFrame:
    return load_rep_csv(
        "REP-001",
        "rep001_race_composition.csv",
    )


def load_rep001_availability() -> pd.DataFrame:
    return load_rep_csv(
        "REP-001",
        "rep001_demographic_availability.csv",
    )


# ============================================================
# REP-002 — identity availability
# ============================================================

def load_rep002_identity_support() -> pd.DataFrame:
    return load_rep_csv(
        "REP-002",
        "rep002_identity_support.csv",
    )


# ============================================================
# REP-003 — identity depth
# ============================================================

def load_rep003_identity_summary() -> pd.DataFrame:
    return load_rep_csv(
        "REP-003",
        "rep003_identity_depth_summary.csv",
    )


def load_rep003_identity_distribution() -> pd.DataFrame:
    return load_rep_csv(
        "REP-003",
        "rep003_identity_depth_distribution.csv",
    )


# ============================================================
# REP-004 — intersectional representation
# ============================================================

def load_rep004_intersections() -> pd.DataFrame:
    return load_rep_csv(
        "REP-004",
        "rep004_intersectional_composition.csv",
    )


def load_rep004_status_summary() -> pd.DataFrame:
    return load_rep_csv(
        "REP-004",
        "rep004_reporting_status_summary.csv",
    )


# ============================================================
# REP-005 — target prevalence / base rates
# ============================================================

def load_rep005_overall() -> pd.DataFrame:
    return load_rep_csv(
        "REP-005",
        "rep005_overall_base_rate.csv",
    )


def load_rep005_race() -> pd.DataFrame:
    return load_rep_csv(
        "REP-005",
        "rep005_base_rate_by_race.csv",
    )


def load_rep005_age() -> pd.DataFrame:
    return load_rep_csv(
        "REP-005",
        "rep005_base_rate_by_age.csv",
    )


# ============================================================
# REP-006 — missingness
# ============================================================

def load_rep006_missingness() -> pd.DataFrame:
    return load_rep_csv(
        "REP-006",
        "rep006_label_missingness.csv",
    )


def load_rep006_patterns() -> pd.DataFrame:
    return load_rep_csv(
        "REP-006",
        "rep006_missingness_patterns.csv",
    )


def load_rep006_affected_analyses() -> pd.DataFrame:
    return load_rep_csv(
        "REP-006",
        "rep006_affected_analyses.csv",
    )
