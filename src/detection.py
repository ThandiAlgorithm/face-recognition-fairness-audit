
from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

# Check TensorFlow availability before setting the flag.
_TENSORFLOW_IMPORTED_TOO_EARLY = "tensorflow" in sys.modules
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def _check_retinaface_runtime() -> None:
    if _TENSORFLOW_IMPORTED_TOO_EARLY:
        raise RuntimeError(
            "tensorflow was imported before src.detection set "
            "TF_USE_LEGACY_KERAS=1. restart the kernel and import "
            "src.detection before anything else touches tensorflow."
        )


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"

CANONICAL_LANDMARK_FRACTIONS = [
    (0.35, 0.35),
    (0.65, 0.35),
    (0.50, 0.52),
    (0.38, 0.72),
    (0.62, 0.72),
]
ALIGN_OUTPUT_SIZE = 224

# RetinaFace needs a lower threshold for cropped faces.
DEFAULT_THRESHOLD = 0.5


def build_pilot_sample(canonical_images_df: pd.DataFrame, seed: int, n_per_dataset: int = 50) -> pd.DataFrame:
    parts = []
    for dataset, group in canonical_images_df.groupby("dataset"):
        n = min(n_per_dataset, len(group))
        parts.append(group.sample(n=n, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def get_model_info() -> dict:
    return {
        "package": "retina-face",
        "package_version": package_version("retina-face"),
        "backend": "tensorflow",
        "tensorflow_version": package_version("tensorflow"),
        "tf_keras_version": package_version("tf-keras"),
        "legacy_keras_enabled": os.environ.get("TF_USE_LEGACY_KERAS") == "1",
        "threshold": DEFAULT_THRESHOLD,
        "allow_upscaling": True,
    }


def detect_faces(image_path: Path, threshold: float = DEFAULT_THRESHOLD) -> dict:
    _check_retinaface_runtime()
    from retinaface import RetinaFace

    result = {
        "path": str(image_path),
        "inference_success": False,
        "inference_error": None,
        "number_of_faces": 0,
        "detections": [],
    }
    try:
        raw = RetinaFace.detect_faces(str(image_path), threshold=threshold)
        if not isinstance(raw, dict):
            raw = {}
        for face in raw.values():
            x1, y1, x2, y2 = [float(v) for v in face["facial_area"]]
            lm = face["landmarks"]
            kps = [
                [float(lm["left_eye"][0]), float(lm["left_eye"][1])],
                [float(lm["right_eye"][0]), float(lm["right_eye"][1])],
                [float(lm["nose"][0]), float(lm["nose"][1])],
                [float(lm["mouth_left"][0]), float(lm["mouth_left"][1])],
                [float(lm["mouth_right"][0]), float(lm["mouth_right"][1])],
            ]
            result["detections"].append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "kps": kps,
                    "det_score": float(face["score"]),
                }
            )
        result["number_of_faces"] = len(result["detections"])
        result["inference_success"] = True
    except Exception as exc:
        result["inference_error"] = f"{type(exc).__name__}: {exc}"
    return result


def select_primary_face(detections: list[dict]) -> Optional[dict]:
    if not detections:
        return None

    def bbox_area(d):
        x1, y1, x2, y2 = d["bbox"]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return max(detections, key=lambda d: (bbox_area(d), d["det_score"]))


def align_face(image_bgr: np.ndarray, kps: list[list[float]], output_size: int = ALIGN_OUTPUT_SIZE) -> Optional[np.ndarray]:
    src = np.array(kps, dtype=np.float32)
    dst = np.array(
        [(fx * output_size, fy * output_size) for fx, fy in CANONICAL_LANDMARK_FRACTIONS],
        dtype=np.float32,
    )
    transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if transform is None:
        return None
    return cv2.warpAffine(image_bgr, transform, (output_size, output_size), flags=cv2.INTER_LINEAR)
