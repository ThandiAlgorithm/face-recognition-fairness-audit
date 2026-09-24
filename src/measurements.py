from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
from skimage.color import rgb2lab


def _lab(rgb: np.ndarray) -> np.ndarray:
    return rgb2lab(rgb[..., :3].astype(np.float64) / 255.0)


def brightness(rgb: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> float:
    l_star = _lab(rgb)[..., 0]
    values = l_star[valid_mask] if valid_mask is not None else l_star.reshape(-1)
    return float(values.mean())


def contrast(rgb: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> float:
    l_star = _lab(rgb)[..., 0]
    values = l_star[valid_mask] if valid_mask is not None else l_star.reshape(-1)
    return float(values.std())


def sharpness(rgb: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> float:
    grey = cv2.cvtColor(rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(grey, cv2.CV_64F)
    values = laplacian[valid_mask] if valid_mask is not None else laplacian.reshape(-1)
    return float(values.var())


def face_size(bbox: list[float], image_width: int, image_height: int) -> dict:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    width = float(image_width)
    height = float(image_height)
    x1 = min(max(x1, 0.0), width)
    x2 = min(max(x2, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    y2 = min(max(y2, 0.0), height)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = width * height
    return {
        "face_px_area": float(area),
        "face_area_ratio": float(area / image_area) if image_area > 0 else None,
    }


def _clip_fraction_pair(values: list[float]) -> tuple[float, float]:
    lower = min(1.0, max(0.0, float(values[0])))
    upper = min(1.0, max(0.0, float(values[1])))
    return (lower, upper) if lower <= upper else (upper, lower)


def extract_skin_roi(
    aligned_rgb: np.ndarray,
    preprocessing_cfg: dict,
    valid_mask: Optional[np.ndarray] = None,
) -> dict:
    skin_cfg = preprocessing_cfg["skin_measurement"]
    patches = skin_cfg["roi_patches"]
    filter_cfg = skin_cfg["roi_skin_filter"]
    height, width = aligned_rgb.shape[:2]

    ycrcb = cv2.cvtColor(aligned_rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[..., 1].astype(np.int16)
    cb = ycrcb[..., 2].astype(np.int16)
    cb_low, cb_high = filter_cfg["cb_range"]
    cr_low, cr_high = filter_cfg["cr_range"]
    skin_mask = (
        (cb >= int(cb_low))
        & (cb <= int(cb_high))
        & (cr >= int(cr_low))
        & (cr <= int(cr_high))
    )
    if valid_mask is not None:
        skin_mask &= valid_mask

    pixels = []
    total_roi_pixels = 0
    valid_pixel_count = 0

    for patch in patches.values():
        x_low, x_high = _clip_fraction_pair(patch["x"])
        y_low, y_high = _clip_fraction_pair(patch["y"])
        x1 = min(max(int(round(x_low * width)), 0), width)
        x2 = min(max(int(round(x_high * width)), 0), width)
        y1 = min(max(int(round(y_low * height)), 0), height)
        y2 = min(max(int(round(y_high * height)), 0), height)
        if x2 <= x1 or y2 <= y1:
            continue

        region_rgb = aligned_rgb[y1:y2, x1:x2, :3]
        region_mask = skin_mask[y1:y2, x1:x2]
        if valid_mask is None:
            total_roi_pixels += int(region_mask.size)
        else:
            total_roi_pixels += int(valid_mask[y1:y2, x1:x2].sum())
        selected = region_rgb[region_mask]
        valid_pixel_count += int(len(selected))
        if len(selected):
            pixels.append(selected)

    combined = np.concatenate(pixels, axis=0) if pixels else np.empty((0, 3), dtype=np.uint8)
    valid_fraction = valid_pixel_count / total_roi_pixels if total_roi_pixels > 0 else 0.0
    return {
        "pixels": combined,
        "total_roi_pixels": int(total_roi_pixels),
        "valid_pixel_count": int(valid_pixel_count),
        "valid_fraction": float(valid_fraction),
    }


def cielab_median(rgb_pixels: np.ndarray) -> Optional[tuple[float, float, float]]:
    if len(rgb_pixels) == 0:
        return None
    lab = rgb2lab((rgb_pixels.astype(np.float64) / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    return (
        float(np.median(lab[:, 0])),
        float(np.median(lab[:, 1])),
        float(np.median(lab[:, 2])),
    )


def compute_ita(l_star: float, b_star: float, tolerance: float) -> tuple[Optional[float], bool]:
    if abs(float(b_star)) < float(tolerance):
        return None, True
    value = math.degrees(math.atan((float(l_star) - 50.0) / float(b_star)))
    return float(value), False


def compute_hue(a_star: float, b_star: float) -> float:
    return float(math.degrees(math.atan2(float(b_star), float(a_star))) % 360.0)


def measure_aligned_face(
    aligned_rgb: np.ndarray,
    preprocessing_cfg: dict,
    is_greyscale: bool,
    valid_mask: Optional[np.ndarray],
    bbox: Optional[list[float]],
    original_width: Optional[int],
    original_height: Optional[int],
) -> tuple[dict, dict]:
    result = {
        "brightness": brightness(aligned_rgb, valid_mask),
        "contrast": contrast(aligned_rgb, valid_mask),
        "sharpness": sharpness(aligned_rgb, valid_mask),
        "face_px_area": None,
        "face_area_ratio": None,
        "yaw": None,
        "pitch": None,
        "roll": None,
        "skin_roi_valid": False,
        "L_star": None,
        "a_star": None,
        "b_star": None,
        "ITA": None,
        "hue_angle": None,
        "ita_group": None,
        "ita_ordinal": None,
        "fitz_classifier": None,
        "fitz_classifier_conf": None,
        "fitz_ordinal": None,
        "skin_group_ita": None,
        "skin_group_fitz": None,
        "ita_undefined": False,
    }
    diagnostics = {
        "skin_roi_total_pixel_count": None,
        "skin_roi_valid_pixel_count": None,
        "skin_valid_fraction": None,
        "skin_invalid_reason": None,
    }

    if bbox is not None and original_width is not None and original_height is not None:
        result.update(face_size(bbox, original_width, original_height))

    if is_greyscale:
        diagnostics["skin_invalid_reason"] = "greyscale_no_chroma"
        return result, diagnostics

    skin = extract_skin_roi(aligned_rgb, preprocessing_cfg, valid_mask)
    diagnostics["skin_roi_total_pixel_count"] = skin["total_roi_pixels"]
    diagnostics["skin_roi_valid_pixel_count"] = skin["valid_pixel_count"]
    diagnostics["skin_valid_fraction"] = skin["valid_fraction"]

    minimum = int(preprocessing_cfg["skin_measurement"]["roi_min_valid_pixels"])
    if skin["valid_pixel_count"] < minimum:
        diagnostics["skin_invalid_reason"] = "invalid_skin_roi"
        return result, diagnostics

    lab = cielab_median(skin["pixels"])
    if lab is None:
        diagnostics["skin_invalid_reason"] = "invalid_skin_roi"
        return result, diagnostics

    result["skin_roi_valid"] = True
    result["L_star"], result["a_star"], result["b_star"] = lab
    tolerance = float(preprocessing_cfg["skin_measurement"]["ita_guard"]["b_star_min_abs"])
    result["ITA"], result["ita_undefined"] = compute_ita(
        result["L_star"],
        result["b_star"],
        tolerance,
    )
    result["hue_angle"] = compute_hue(result["a_star"], result["b_star"])
    if result["ita_undefined"]:
        diagnostics["skin_invalid_reason"] = "ita_undefined"
    return result, diagnostics
