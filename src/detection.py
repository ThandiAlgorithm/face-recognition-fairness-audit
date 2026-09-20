from __future__ import annotations

import hashlib
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from src.retinaface import (
    MNET_CONFIG,
    PriorBox,
    RetinaFace,
    decode_boxes,
    decode_landmarks,
)


EXPECTED_CHECKPOINT_SHA256 = (
    "2979b33ffafda5d74b6948cd7a5b9a7a"
    "62f62b949cef24e95fd15d2883a65220"
)

_MODEL = None


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _checkpoint_path() -> Path:
    return (
        Path.home()
        / ".cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "mobilenet0.25_Final.pth"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def build_retinaface_model():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    checkpoint_path = _checkpoint_path()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"RetinaFace checkpoint not found: {checkpoint_path}"
        )

    actual_sha256 = _sha256_file(checkpoint_path)

    if actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "RetinaFace checkpoint SHA256 mismatch. "
            f"Expected {EXPECTED_CHECKPOINT_SHA256}, "
            f"found {actual_sha256}."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "RetinaFace checkpoint is not a state dictionary."
        )

    model = RetinaFace()

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    model.eval()
    model.to("cpu")

    _MODEL = model

    return _MODEL


def _detector_resize(
    image_bgr: np.ndarray,
    detector_cfg: dict,
) -> tuple[np.ndarray, float]:
    height, width = image_bgr.shape[:2]

    resize = float(
        detector_cfg.get(
            "resize",
            1.0,
        )
    )

    max_size = int(
        detector_cfg.get(
            "max_size",
            -1,
        )
    )

    allow_upscaling = bool(
        detector_cfg.get(
            "allow_upscaling",
            True,
        )
    )

    if resize <= 0:
        raise ValueError(
            "Detector resize must be greater than zero."
        )

    if not allow_upscaling:
        resize = min(
            resize,
            1.0,
        )

    if max_size > 0:
        largest_dimension = max(
            height,
            width,
        )

        if largest_dimension * resize > max_size:
            resize = (
                float(max_size)
                / float(largest_dimension)
            )

    if abs(resize - 1.0) < 1e-12:
        return image_bgr, 1.0

    resized = cv2.resize(
        image_bgr,
        None,
        fx=resize,
        fy=resize,
        interpolation=cv2.INTER_LINEAR,
    )

    return resized, resize


def _prepare_input(
    image_bgr: np.ndarray,
) -> torch.Tensor:
    image = image_bgr.astype(
        np.float32,
        copy=True,
    )

    image -= np.asarray(
        [104.0, 117.0, 123.0],
        dtype=np.float32,
    )

    image = image.transpose(
        2,
        0,
        1,
    )

    image = np.ascontiguousarray(
        image
    )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    return tensor


def detect_faces(
    image_bgr: np.ndarray,
    preprocessing_cfg: dict,
    model=None,
) -> dict:
    detector_cfg = preprocessing_cfg["detector"]

    threshold = float(
        detector_cfg[
            "confidence_threshold"
        ]
    )

    nms_threshold = float(
        detector_cfg.get(
            "nms_threshold",
            0.4,
        )
    )

    top_k = int(
        detector_cfg.get(
            "top_k",
            5000,
        )
    )

    keep_top_k = int(
        detector_cfg.get(
            "keep_top_k",
            750,
        )
    )

    result = {
        "inference_success": False,
        "inference_error": None,
        "n_detections": 0,
        "detections": [],
        "inference_ms": None,
        "resize_factor": None,
    }

    start = time.perf_counter()

    try:
        if (
            not isinstance(
                image_bgr,
                np.ndarray,
            )
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
        ):
            raise ValueError(
                "RetinaFace input must be an HxWx3 BGR numpy array."
            )

        if model is None:
            model = build_retinaface_model()

        resized_bgr, resize_factor = _detector_resize(
            image_bgr,
            detector_cfg,
        )

        result["resize_factor"] = float(
            resize_factor
        )

        input_height, input_width = (
            resized_bgr.shape[:2]
        )

        tensor = _prepare_input(
            resized_bgr
        )

        with torch.inference_mode():
            (
                locations,
                confidences,
                landmark_locations,
            ) = model(
                tensor
            )

        priors = PriorBox(
            image_size=(
                input_height,
                input_width,
            ),
            config=MNET_CONFIG,
        ).generate(
            device="cpu"
        )

        locations = locations[0]
        confidences = confidences[0]
        landmark_locations = (
            landmark_locations[0]
        )

        if locations.shape[0] != priors.shape[0]:
            raise RuntimeError(
                "RetinaFace prior count does not match "
                "the number of model predictions. "
                f"Predictions={locations.shape[0]}, "
                f"priors={priors.shape[0]}."
            )

        boxes = decode_boxes(
            locations,
            priors,
            MNET_CONFIG["variance"],
        )

        landmarks = decode_landmarks(
            landmark_locations,
            priors,
            MNET_CONFIG["variance"],
        )

        box_scale = torch.tensor(
            [
                input_width,
                input_height,
                input_width,
                input_height,
            ],
            dtype=torch.float32,
        )

        landmark_scale = torch.tensor(
            [
                input_width,
                input_height,
            ]
            * 5,
            dtype=torch.float32,
        )

        boxes = (
            boxes
            * box_scale
            / resize_factor
        )

        landmarks = (
            landmarks
            * landmark_scale
            / resize_factor
        )

        scores = confidences[:, 1]

        keep = torch.where(
            scores > threshold
        )[0]

        boxes = boxes[keep]
        landmarks = landmarks[keep]
        scores = scores[keep]

        if scores.numel() > 0:
            order = torch.argsort(
                scores,
                descending=True,
            )

            if top_k > 0:
                order = order[
                    :top_k
                ]

            boxes = boxes[order]
            landmarks = landmarks[order]
            scores = scores[order]

            keep_after_nms = nms(
                boxes,
                scores,
                nms_threshold,
            )

            if keep_top_k > 0:
                keep_after_nms = (
                    keep_after_nms[
                        :keep_top_k
                    ]
                )

            boxes = boxes[
                keep_after_nms
            ]

            landmarks = landmarks[
                keep_after_nms
            ]

            scores = scores[
                keep_after_nms
            ]

        detections = []

        for index in range(
            len(scores)
        ):
            bbox = [
                float(value)
                for value
                in boxes[
                    index
                ].tolist()
            ]

            flat_landmarks = (
                landmarks[
                    index
                ].tolist()
            )

            points = [
                [
                    float(
                        flat_landmarks[
                            point_index * 2
                        ]
                    ),
                    float(
                        flat_landmarks[
                            point_index * 2 + 1
                        ]
                    ),
                ]
                for point_index
                in range(5)
            ]

            detections.append(
                {
                    "bbox": bbox,
                    "score": float(
                        scores[
                            index
                        ].item()
                    ),
                    "landmarks": points,
                }
            )

        result["detections"] = (
            detections
        )

        result["n_detections"] = len(
            detections
        )

        result[
            "inference_success"
        ] = True

    except Exception as exc:
        result["inference_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    result["inference_ms"] = (
        time.perf_counter()
        - start
    ) * 1000.0

    return result


def select_primary_face(
    detections: list[dict],
    preprocessing_cfg: Optional[dict] = None,
) -> Optional[dict]:
    if not detections:
        return None

    def bbox_area(
        detection: dict,
    ) -> float:
        x1, y1, x2, y2 = [
            float(value)
            for value
            in detection["bbox"]
        ]

        return (
            max(
                0.0,
                x2 - x1,
            )
            * max(
                0.0,
                y2 - y1,
            )
        )

    return max(
        detections,
        key=lambda detection: (
            bbox_area(
                detection
            ),
            float(
                detection["score"]
            ),
        ),
    )


def align_face(
    image_rgb: np.ndarray,
    landmarks: list[list[float]],
    preprocessing_cfg: dict,
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    alignment_cfg = (
        preprocessing_cfg[
            "alignment"
        ]
    )

    output_size = int(
        alignment_cfg[
            "output_size"
        ]
    )

    canonical_points = (
        alignment_cfg[
            "canonical_landmark_fractions"
        ]
    )

    source = np.asarray(
        landmarks,
        dtype=np.float32,
    )

    if source.shape != (
        5,
        2,
    ):
        raise ValueError(
            "Alignment requires exactly five "
            "2D facial landmarks."
        )

    destination = np.asarray(
        [
            [
                float(x_fraction)
                * output_size,
                float(y_fraction)
                * output_size,
            ]
            for (
                x_fraction,
                y_fraction,
            )
            in canonical_points
        ],
        dtype=np.float32,
    )

    transform, _ = (
        cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.LMEDS,
        )
    )

    if transform is None:
        return (
            None,
            None,
            None,
        )

    aligned = cv2.warpAffine(
        image_rgb,
        transform,
        (
            output_size,
            output_size,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    source_mask = np.ones(
        image_rgb.shape[:2],
        dtype=np.uint8,
    )

    valid_mask = cv2.warpAffine(
        source_mask,
        transform,
        (
            output_size,
            output_size,
        ),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)

    return (
        aligned,
        transform,
        valid_mask,
    )


def get_model_info(
    preprocessing_cfg: dict,
) -> dict:
    detector_cfg = preprocessing_cfg[
        "detector"
    ]

    checkpoint_path = (
        _checkpoint_path()
    )

    return {
        "name": "RetinaFace",
        "framework": "pytorch",
        "implementation": "project_native",
        "backbone": "mobilenet0.25",
        "torch_version": package_version(
            "torch"
        ),
        "torchvision_version": package_version(
            "torchvision"
        ),
        "device": "cpu",
        "weights_file": (
            checkpoint_path.name
        ),
        "weights_path": str(
            checkpoint_path
        ),
        "weights_sha256": (
            EXPECTED_CHECKPOINT_SHA256
        ),
        "confidence_threshold": float(
            detector_cfg[
                "confidence_threshold"
            ]
        ),
        "nms_threshold": float(
            detector_cfg.get(
                "nms_threshold",
                0.4,
            )
        ),
        "top_k": int(
            detector_cfg.get(
                "top_k",
                5000,
            )
        ),
        "keep_top_k": int(
            detector_cfg.get(
                "keep_top_k",
                750,
            )
        ),
        "resize": float(
            detector_cfg.get(
                "resize",
                1.0,
            )
        ),
        "max_size": int(
            detector_cfg.get(
                "max_size",
                -1,
            )
        ),
        "alignment_output_size": int(
            preprocessing_cfg[
                "alignment"
            ][
                "output_size"
            ]
        ),
    }
