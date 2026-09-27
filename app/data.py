from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MASTER_PATH = PROJECT_ROOT / "metadata" / "master_annotations.parquet"
DIAGNOSTICS_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "detection_alignment_diagnostics.parquet"
)
RESULTS_REGISTRY_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "results_registry.csv"
)


@st.cache_data(show_spinner=False)
def load_master() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Master table not found: {MASTER_PATH}"
        )

    return pd.read_parquet(MASTER_PATH)


@st.cache_data(show_spinner=False)
def load_detection_diagnostics() -> pd.DataFrame:
    if not DIAGNOSTICS_PATH.exists():
        return pd.DataFrame()

    return pd.read_parquet(DIAGNOSTICS_PATH)


@st.cache_data(show_spinner=False)
def load_results_registry() -> pd.DataFrame:
    if not RESULTS_REGISTRY_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(RESULTS_REGISTRY_PATH)


def dataset_label(value: str) -> str:
    labels = {
        "lfw": "LFW",
        "fairface": "FairFace",
        "utkface": "UTKFace",
    }

    return labels.get(str(value).lower(), str(value))


def category_counts(
    series: pd.Series,
    missing_label: str = "Unavailable",
) -> pd.DataFrame:
    values = (
        series.astype("string")
        .fillna(missing_label)
        .replace("<NA>", missing_label)
    )

    result = (
        values.value_counts(dropna=False)
        .rename_axis("Category")
        .reset_index(name="Count")
    )

    total = result["Count"].sum()

    result["Percentage"] = (
        result["Count"] / total * 100
        if total
        else 0
    )

    return result
    