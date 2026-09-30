from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def compute_ita(L_star: float, b_star: float, b_star_min_abs: float = 1.0) -> tuple[float | None, bool]:
    """Compute Individual Typology Angle in degrees.

    Returns (value, undefined_flag). The project treats |b*| below the guard as undefined.
    """
    if not np.isfinite(L_star) or not np.isfinite(b_star) or abs(float(b_star)) < float(b_star_min_abs):
        return None, True
    value = math.degrees(math.atan((float(L_star) - 50.0) / float(b_star)))
    return float(value), False


def compute_hue(a_star: float, b_star: float) -> float | None:
    if not np.isfinite(a_star) or not np.isfinite(b_star):
        return None
    return float(math.degrees(math.atan2(float(b_star), float(a_star))) % 360.0)


def _roi_mask(h: int, w: int, patches: dict) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for spec in patches.values():
        x1, x2 = sorted(float(v) for v in spec["x"])
        y1, y2 = sorted(float(v) for v in spec["y"])
        xa = max(0, min(w, int(round(x1 * w))))
        xb = max(0, min(w, int(round(x2 * w))))
        ya = max(0, min(h, int(round(y1 * h))))
        yb = max(0, min(h, int(round(y2 * h))))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = True
    return mask


def extract_skin_roi(rgb: np.ndarray, preprocessing: dict, valid_mask: np.ndarray | None = None) -> dict:
    """Extract fixed relative face patches and apply the frozen YCrCb skin-pixel filter."""
    sm = preprocessing["skin_measurement"]
    h, w = rgb.shape[:2]
    roi = _roi_mask(h, w, sm["roi_patches"])
    if valid_mask is not None:
        roi &= valid_mask.astype(bool)
    total = int(roi.sum())

    filt = sm.get("roi_skin_filter", {})
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cb = ycrcb[..., 2]
    cr = ycrcb[..., 1]
    cb_lo, cb_hi = filt.get("cb_range", [0, 255])
    cr_lo, cr_hi = filt.get("cr_range", [0, 255])
    skin = (cb >= float(cb_lo)) & (cb <= float(cb_hi)) & (cr >= float(cr_lo)) & (cr <= float(cr_hi))
    use = roi & skin
    pixels = rgb[use]
    valid_n = int(use.sum())
    return {
        "pixels": pixels,
        "roi_mask": roi,
        "skin_mask": use,
        "total_roi_pixels": total,
        "valid_pixel_count": valid_n,
        "valid_fraction": float(valid_n / total) if total else 0.0,
    }


def face_size(bbox: list[float] | tuple[float, ...], image_width: int, image_height: int) -> dict:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1c, x2c = max(0.0, x1), min(float(image_width), x2)
    y1c, y2c = max(0.0, y1), min(float(image_height), y2)
    bw, bh = max(0.0, x2c - x1c), max(0.0, y2c - y1c)
    area = bw * bh
    denom = max(1.0, float(image_width) * float(image_height))
    return {
        "face_px_area": float(area),
        "face_area_ratio": float(area / denom),
        "bbox_width_px": float(bw),
        "bbox_height_px": float(bh),
    }
