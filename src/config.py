from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src import paths


CONFIG_FILES = {
    "categories": "categories.yaml",
    "preprocessing": "preprocessing.yaml",
    "models": "models.yaml",
    "exclusions": "exclusions.yaml",
    "statistics": "statistics.yaml",
    "reporting": "reporting.yaml",
}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Config must contain a mapping: {path}")
    return data


def load_config(name: str) -> dict:
    if name not in CONFIG_FILES:
        raise KeyError(f"Unknown config: {name}")
    return load_yaml(paths.CONFIGS_DIR / CONFIG_FILES[name])


def load_categories() -> dict:
    return load_config("categories")


def load_preprocessing() -> dict:
    return load_config("preprocessing")


def load_models() -> dict:
    return load_config("models")


def load_exclusions() -> dict:
    return load_config("exclusions")


def load_statistics() -> dict:
    return load_config("statistics")


def load_reporting() -> dict:
    return load_config("reporting")


def require_key(mapping: dict, key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing config key '{key}' in {context}")
    return mapping[key]


def validate_preprocessing(cfg: dict) -> None:
    for key in [
        "datasets",
        "integrity",
        "colour",
        "greyscale",
        "detector",
        "alignment",
        "skin_measurement",
        "splits",
        "duplicates",
    ]:
        require_key(cfg, key, "preprocessing.yaml")

    for dataset in ["lfw", "fairface", "utkface"]:
        item = require_key(cfg["datasets"], dataset, "preprocessing datasets")
        for key in [
            "dataset_version",
            "variant",
            "image_roots",
            "source_preprocessed",
            "project_detection_required",
            "project_alignment_required",
        ]:
            require_key(item, key, f"preprocessing datasets.{dataset}")

    detector = cfg["detector"]
    for key in [
        "name",
        "implementation",
        "confidence_threshold",
        "nms_threshold",
        "top_k",
        "keep_top_k",
        "resize",
        "preserve_all_detections",
        "primary_face_rule",
    ]:
        require_key(detector, key, "preprocessing detector")

    if detector["implementation"] != "project_native":
        raise RuntimeError(
            "The current frozen preprocessing config expects project_native "
            "(see docs/protocol_amendments/AMEND-20260919-01.md). "
            "Use GOV-010 before changing the RetinaFace implementation again."
        )

    points = cfg["alignment"].get("canonical_landmark_fractions")
    if not isinstance(points, list) or len(points) != 5:
        raise RuntimeError("Alignment must contain five canonical landmark points.")


def validate_categories(cfg: dict) -> None:
    for key in [
        "gender",
        "age_bands",
        "ita_groups",
        "skin_binary",
        "race",
    ]:
        require_key(cfg, key, "categories.yaml")
