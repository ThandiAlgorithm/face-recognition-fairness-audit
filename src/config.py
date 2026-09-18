from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src import paths


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise RuntimeError(f"Config must contain a mapping: {path}")

    return data


def require_key(mapping: dict, key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing config key '{key}' in {context}")
    return mapping[key]


def load_gov002() -> dict:
    path = paths.CONFIGS_DIR / "GOV-002_categories.yaml"
    data = load_yaml(path)

    required = [
        "gender",
        "age_bands_fine",
        "age_bands_coarse",
        "ita_groups",
        "skin_binary",
        "race",
    ]

    for key in required:
        require_key(data, key, "GOV-002")

    return data


def load_gov006() -> dict:
    return load_yaml(
        paths.CONFIGS_DIR / "GOV-006_preprocessing.yaml"
    )


def detector_config(gov006: dict) -> dict:
    section = require_key(gov006, "detector", "GOV-006")

    required = [
        "confidence_threshold",
        "allow_upscaling",
        "primary_face_rule",
        "preserve_all_detections",
    ]

    for key in required:
        require_key(section, key, "GOV-006 detector")

    return section


def alignment_config(gov006: dict) -> dict:
    section = require_key(gov006, "alignment", "GOV-006")

    required = [
        "output_size",
        "canonical_landmark_fractions",
    ]

    for key in required:
        require_key(section, key, "GOV-006 alignment")

    points = section["canonical_landmark_fractions"]

    if not isinstance(points, list) or len(points) != 5:
        raise ValueError(
            "GOV-006 alignment canonical_landmark_fractions "
            "must contain five points."
        )

    return section


def skin_config(gov006: dict) -> dict:
    section = require_key(gov006, "skin_measurement", "GOV-006")

    required = [
        "roi_patches",
        "roi_skin_filter",
        "roi_min_valid_pixels",
        "ita_guard",
    ]

    for key in required:
        require_key(section, key, "GOV-006 skin_measurement")

    require_key(
        section["ita_guard"],
        "b_star_min_abs",
        "GOV-006 ita_guard",
    )

    require_key(
        section["roi_skin_filter"],
        "cb_range",
        "GOV-006 roi_skin_filter",
    )

    require_key(
        section["roi_skin_filter"],
        "cr_range",
        "GOV-006 roi_skin_filter",
    )

    return section
