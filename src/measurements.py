
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
from skimage.color import rgb2lab

# BT 601 luma weights for RGB values.
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114])


def brightness(rgb: np.ndarray) -> float:
    luma = rgb[..., :3].astype(np.float64) @ LUMA_WEIGHTS
    return float(luma.mean())


def contrast(rgb: np.ndarray) -> float:
    luma = rgb[..., :3].astype(np.float64) @ LUMA_WEIGHTS
    return float(luma.std())


def sharpness(rgb: np.ndarray) -> float:
    grey = cv2.cvtColor(rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def face_size(bbox: list[float], image_width: int, image_height: int) -> dict:
    x1, y1, x2, y2 = bbox
    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = image_width * image_height
    return {
        "face_bbox_area_px": bbox_area,
        "face_area_ratio": bbox_area / image_area if image_area > 0 else None,
    }


def extract_skin_roi_pixels(aligned_rgb: np.ndarray, roi_patches: dict, skin_filter: dict) -> np.ndarray:
    h, w = aligned_rgb.shape[:2]
    valid_pixels = []

    ycbcr = cv2.cvtColor(aligned_rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    cb = ycbcr[..., 2].astype(np.int16)
    cr = ycbcr[..., 1].astype(np.int16)
    cb_lo, cb_hi = skin_filter["cb_range"]
    cr_lo, cr_hi = skin_filter["cr_range"]
    skin_mask_full = (cb >= cb_lo) & (cb <= cb_hi) & (cr >= cr_lo) & (cr <= cr_hi)

    for patch in roi_patches.values():
        x_lo, x_hi = patch["x"]
        y_lo, y_hi = patch["y"]
        px1, px2 = int(x_lo * w), int(x_hi * w)
        py1, py2 = int(y_lo * h), int(y_hi * h)

        region_rgb = aligned_rgb[py1:py2, px1:px2, :3]
        region_mask = skin_mask_full[py1:py2, px1:px2]
        valid_pixels.append(region_rgb[region_mask])

    if not valid_pixels:
        return np.empty((0, 3), dtype=np.uint8)
    return np.concatenate(valid_pixels, axis=0)


def cielab_median(rgb_pixels: np.ndarray) -> Optional[tuple[float, float, float]]:
    if len(rgb_pixels) == 0:
        return None
    lab = rgb2lab((rgb_pixels / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    return float(np.median(lab[:, 0])), float(np.median(lab[:, 1])), float(np.median(lab[:, 2]))


def compute_ita(l_star: float, b_star: float, b_star_min_abs: float) -> dict:
    if abs(b_star) < b_star_min_abs:
        return {"ita": None, "ita_undefined": True}
    ita = math.atan((l_star - 50.0) / b_star) * 180.0 / math.pi
    return {"ita": ita, "ita_undefined": False}


def compute_hue(a_star: float, b_star: float) -> float:
    return math.degrees(math.atan2(b_star, a_star)) % 360.0
