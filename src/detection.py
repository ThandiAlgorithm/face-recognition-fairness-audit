from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

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


def _runtime_defaults() -> tuple[float, bool, int, list]:
    from src import config

    gov006 = config.load_gov006()
    detector_cfg = config.detector_config(gov006)
    alignment_cfg = config.alignment_config(gov006)

    return (
        float(detector_cfg["confidence_threshold"]),
        bool(detector_cfg["allow_upscaling"]),
        int(alignment_cfg["output_size"]),
        alignment_cfg["canonical_landmark_fractions"],
    )


def build_retinaface_model():
    _check_retinaface_runtime()
    from retinaface import RetinaFace

    return RetinaFace.build_model()


def get_model_info(threshold: Optional[float] = None, allow_upscaling: Optional[bool] = None) -> dict:
    default_threshold, default_upscaling, _, _ = _runtime_defaults()

    if threshold is None:
        threshold = default_threshold
    if allow_upscaling is None:
        allow_upscaling = default_upscaling

    return {
        "package": "retina-face",
        "package_version": package_version("retina-face"),
        "backend": "tensorflow",
        "tensorflow_version": package_version("tensorflow"),
        "tf_keras_version": package_version("tf-keras"),
        "legacy_keras_enabled": os.environ.get("TF_USE_LEGACY_KERAS") == "1",
        "threshold": float(threshold),
        "allow_upscaling": bool(allow_upscaling),
    }


def detect_faces(
    image_path: Path,
    model=None,
    threshold: Optional[float] = None,
    allow_upscaling: Optional[bool] = None,
) -> dict:
    _check_retinaface_runtime()
    from retinaface import RetinaFace

    default_threshold, default_upscaling, _, _ = _runtime_defaults()

    if threshold is None:
        threshold = default_threshold
    if allow_upscaling is None:
        allow_upscaling = default_upscaling
    if model is None:
        model = build_retinaface_model()

    result = {
        "path": str(image_path),
        "inference_success": False,
        "inference_error": None,
        "number_of_faces": 0,
        "detections": [],
    }

    try:
        raw = RetinaFace.detect_faces(
            img_path=str(image_path),
            threshold=float(threshold),
            model=model,
            allow_upscaling=bool(allow_upscaling),
        )

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


def align_face(
    image_rgb: np.ndarray,
    keypoints: list[list[float]],
    output_size: Optional[int] = None,
    canonical_landmark_fractions: Optional[list] = None,
    return_transform: bool = False,
):
    _, _, default_size, default_points = _runtime_defaults()

    if output_size is None:
        output_size = default_size
    if canonical_landmark_fractions is None:
        canonical_landmark_fractions = default_points

    source = np.array(keypoints, dtype=np.float32)
    destination = np.array(
        [
            [float(fx) * output_size, float(fy) * output_size]
            for fx, fy in canonical_landmark_fractions
        ],
        dtype=np.float32,
    )

    transform, _ = cv2.estimateAffinePartial2D(source, destination, method=cv2.LMEDS)

    if transform is None:
        return (None, None) if return_transform else None

    aligned = cv2.warpAffine(
        image_rgb,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
    )

    if return_transform:
        return aligned, transform

    return aligned
