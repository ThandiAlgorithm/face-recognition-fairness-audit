from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from model_audit_data import load_csv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASE1_ROOT = (
    PROJECT_ROOT
    / "results"
    / "phase1_audit"
)

MEA_ROOT = PHASE1_ROOT / "measurement_audit"
VIS_ROOT = PHASE1_ROOT / "visual_audit"
SHIFT_ROOT = PHASE1_ROOT / "shift_audit"
LFW_ROOT = PHASE1_ROOT / "lfw_audit"
DETECTION_FIGURES_ROOT = PHASE1_ROOT / "detection_audit" / "figures"
ALIGNMENT_QA_ROOT = PHASE1_ROOT / "alignment_qa"
COMBINED_ROOT = PHASE1_ROOT / "combined"

METADATA_ROOT = PROJECT_ROOT / "metadata"


# ============================================================
# GENERIC LOADERS (json/text/parquet siblings of model_audit_data.load_csv)
# ============================================================

@st.cache_data(show_spinner=False)
def _load_json(path_string: str) -> dict | None:
    path = Path(path_string)

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict | None:
    return _load_json(
        str(path)
    )


@st.cache_data(show_spinner=False)
def _load_text(path_string: str) -> str | None:
    path = Path(path_string)

    if not path.exists():
        return None

    return path.read_text(encoding="utf-8")


def load_text(path: Path) -> str | None:
    return _load_text(
        str(path)
    )


@st.cache_data(show_spinner=False)
def _load_parquet(path_string: str) -> pd.DataFrame:
    path = Path(path_string)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def load_parquet(path: Path) -> pd.DataFrame:
    return _load_parquet(
        str(path)
    )


# ============================================================
# MEA — measurement audit (skin-tone reliability)
# ============================================================

def load_mea_status_counts() -> pd.DataFrame:
    return load_csv(MEA_ROOT / "tables" / "MEA_status_counts.csv")


def load_mea_summary_by_dataset() -> pd.DataFrame:
    return load_csv(MEA_ROOT / "tables" / "MEA_summary_by_dataset.csv")


def load_mea_subgroup_summary() -> pd.DataFrame:
    return load_csv(MEA_ROOT / "tables" / "MEA_subgroup_summary.csv")


def load_mea_subgroup_omnibus() -> pd.DataFrame:
    return load_csv(MEA_ROOT / "tables" / "MEA_subgroup_omnibus.csv")


def load_mea_subgroup_pairwise() -> pd.DataFrame:
    return load_csv(MEA_ROOT / "tables" / "MEA_subgroup_pairwise.csv")


def load_mea_reliability_agreement() -> pd.DataFrame:
    return load_csv(MEA_ROOT / "tables" / "MEA_reliability_agreement.csv")


def load_mea_boundary_sensitivity() -> pd.DataFrame:
    return load_csv(MEA_ROOT / "tables" / "MEA_boundary_sensitivity.csv")


def load_mea_skin_group_distribution() -> pd.DataFrame:
    return load_csv(MEA_ROOT / "tables" / "MEA_skin_group_distribution.csv")


def load_mea_roi_failure_by_group() -> pd.DataFrame:
    return load_csv(MEA_ROOT / "tables" / "MEA_roi_failure_by_group.csv")


def load_mea_roi_failure_tests() -> pd.DataFrame:
    return load_csv(MEA_ROOT / "tables" / "MEA_roi_failure_tests.csv")


def load_mea_method() -> dict | None:
    return load_json(MEA_ROOT / "metadata" / "MEA_method.json")


def load_mea_limitations_text() -> str | None:
    return load_text(MEA_ROOT / "MEA_limitations_draft.md")


def mea_figure(filename: str) -> Path:
    return MEA_ROOT / "figures" / filename


def mea_qa_image(filename: str) -> Path:
    return MEA_ROOT / "qa" / filename


# ============================================================
# VIS — visual quality audit
# ============================================================

def load_vis_status_counts() -> pd.DataFrame:
    return load_csv(VIS_ROOT / "tables" / "VIS_status_counts.csv")


def load_vis_summary_by_dataset() -> pd.DataFrame:
    return load_csv(VIS_ROOT / "tables" / "VIS_summary_by_dataset.csv")


def load_vis_subgroup_summary() -> pd.DataFrame:
    return load_csv(VIS_ROOT / "tables" / "VIS_subgroup_summary.csv")


def load_vis_subgroup_omnibus() -> pd.DataFrame:
    return load_csv(VIS_ROOT / "tables" / "VIS_subgroup_omnibus.csv")


def load_vis_subgroup_pairwise() -> pd.DataFrame:
    return load_csv(VIS_ROOT / "tables" / "VIS_subgroup_pairwise.csv")


def load_vis_qc_flag_counts() -> pd.DataFrame:
    return load_csv(VIS_ROOT / "tables" / "VIS_qc_flag_counts.csv")


def load_vis_brightness_vs_skin() -> pd.DataFrame:
    return load_csv(VIS_ROOT / "tables" / "VIS_brightness_vs_skin_L.csv")


def vis_figure(filename: str) -> Path:
    return VIS_ROOT / "figures" / filename


def vis_qa_image(filename: str) -> Path:
    return VIS_ROOT / "qa" / filename


# ============================================================
# Brightness x skin tone — combined MEA/VIS view
# ============================================================

def load_brightness_vs_skin_tone() -> pd.DataFrame:
    return load_parquet(COMBINED_ROOT / "brightness_vs_skin_tone.parquet")


# ============================================================
# SHIFT — cross-dataset comparison
# ============================================================

def load_shift_analysis_set_counts() -> pd.DataFrame:
    return load_csv(SHIFT_ROOT / "tables" / "SHIFT_analysis_set_counts.csv")


def load_shift_categorical_distributions() -> pd.DataFrame:
    return load_csv(SHIFT_ROOT / "tables" / "SHIFT_categorical_distributions.csv")


def load_shift_categorical_tests() -> pd.DataFrame:
    return load_csv(SHIFT_ROOT / "tables" / "SHIFT_categorical_tests.csv")


def load_shift_continuous_summary() -> pd.DataFrame:
    return load_csv(SHIFT_ROOT / "tables" / "SHIFT_continuous_summary.csv")


def load_shift_continuous_tests() -> pd.DataFrame:
    return load_csv(SHIFT_ROOT / "tables" / "SHIFT_continuous_tests.csv")


def load_shift_effective_balance_summary() -> pd.DataFrame:
    return load_csv(SHIFT_ROOT / "tables" / "SHIFT_effective_balance_summary.csv")


def load_shift_effective_balance_tests() -> pd.DataFrame:
    return load_csv(SHIFT_ROOT / "tables" / "SHIFT_effective_balance_tests.csv")


def load_shift_identity_depth() -> pd.DataFrame:
    return load_csv(SHIFT_ROOT / "tables" / "SHIFT_identity_depth.csv")


def load_shift_ita_within_native_race() -> pd.DataFrame:
    return load_csv(SHIFT_ROOT / "tables" / "SHIFT_ita_within_native_race.csv")


def load_shift_nominal_balance() -> pd.DataFrame:
    return load_csv(SHIFT_ROOT / "tables" / "SHIFT_nominal_balance.csv")


def load_shift_frozen_axes() -> dict | None:
    return load_json(SHIFT_ROOT / "metadata" / "SHIFT_frozen_axes.json")


def load_shift_limitations_text() -> str | None:
    return load_text(SHIFT_ROOT / "SHIFT_interpretation_limits_draft.md")


def shift_figure(filename: str) -> Path:
    return SHIFT_ROOT / "figures" / filename


# ============================================================
# LFW — pair protocol audit
# ============================================================

def load_lfw_composition_images_vs_pairs() -> pd.DataFrame:
    return load_csv(LFW_ROOT / "tables" / "LFW_composition_images_vs_pairs.csv")


def load_lfw_composition_tests() -> pd.DataFrame:
    return load_csv(LFW_ROOT / "tables" / "LFW_composition_tests.csv")


def load_lfw_genuine_visual_difference_by_skin_group() -> pd.DataFrame:
    return load_csv(LFW_ROOT / "tables" / "LFW_genuine_visual_difference_by_skin_group.csv")


def load_lfw_genuine_visual_difference_tests() -> pd.DataFrame:
    return load_csv(LFW_ROOT / "tables" / "LFW_genuine_visual_difference_tests.csv")


def load_lfw_impostor_same_group_vs_random() -> pd.DataFrame:
    return load_csv(LFW_ROOT / "tables" / "LFW_impostor_same_group_vs_random.csv")


def load_lfw_impostor_yoking_counts() -> pd.DataFrame:
    return load_csv(LFW_ROOT / "tables" / "LFW_impostor_yoking_counts.csv")


def load_lfw_pair_file_checks() -> pd.DataFrame:
    return load_csv(LFW_ROOT / "tables" / "LFW_pair_file_checks.csv")


def load_lfw_pair_resolution_counts() -> pd.DataFrame:
    return load_csv(LFW_ROOT / "tables" / "LFW_pair_resolution_counts.csv")


def load_lfw_visual_difference_median_by_yoking() -> pd.DataFrame:
    return load_csv(LFW_ROOT / "tables" / "LFW_visual_difference_median_by_yoking.csv")


def load_lfw_status() -> dict | None:
    return load_json(LFW_ROOT / "metadata" / "LFW_status.json")


def load_lfw_protocol_rules() -> dict | None:
    return load_json(LFW_ROOT / "metadata" / "LFW_protocol_rules.json")


# ============================================================
# Detection audit (tables already tracked at repo-root metadata/,
# following the same precedent app/data.py sets)
# ============================================================

def load_detection_integrity_by_dataset() -> pd.DataFrame:
    return load_csv(METADATA_ROOT / "detection_integrity_by_dataset.csv")


def load_detection_scores() -> pd.DataFrame:
    return load_csv(METADATA_ROOT / "detection_scores.csv")


def load_confidence_summary() -> pd.DataFrame:
    return load_csv(METADATA_ROOT / "confidence_summary.csv")


def load_confidence_histogram() -> pd.DataFrame:
    return load_csv(METADATA_ROOT / "confidence_histogram.csv")


def load_subgroup_detection_failures() -> pd.DataFrame:
    return load_csv(METADATA_ROOT / "subgroup_detection_failures.csv")


def load_alignment_failures_by_subgroup() -> pd.DataFrame:
    return load_csv(METADATA_ROOT / "alignment_failures_by_subgroup.csv")


def load_detector_limitations_text() -> str | None:
    return load_text(METADATA_ROOT / "detector_limitations_draft.md")


def detection_figure(filename: str) -> Path:
    return DETECTION_FIGURES_ROOT / filename


# ============================================================
# Phase 0 alignment QA
# ============================================================

def load_alignment_qa_manifest() -> pd.DataFrame:
    return load_csv(ALIGNMENT_QA_ROOT / "samples_manifest.csv")


def alignment_qa_sample(filename: str) -> Path:
    return ALIGNMENT_QA_ROOT / "samples" / filename


def alignment_qa_artifact(*parts: str) -> Path:
    return ALIGNMENT_QA_ROOT.joinpath(*parts)


# ============================================================
# GENERIC ESCAPE HATCH
# ============================================================

def phase1_audit_artifact(*parts: str) -> Path:
    return PHASE1_ROOT.joinpath(*parts)
