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
    image_area = int(image_width) * int(image_height)
    ratio = bbox_area / image_area if image_area > 0 else None
    return {
        "face_bbox_area_px": bbox_area,
        "face_area_ratio": ratio,
    }


def _clip_fraction_range(lower: float, upper: float) -> tuple[float, float]:
    clipped_lower = min(1.0, max(0.0, float(lower)))
    clipped_upper = min(1.0, max(0.0, float(upper)))
    if clipped_upper < clipped_lower:
        clipped_lower, clipped_upper = clipped_upper, clipped_lower
    return clipped_lower, clipped_upper


def extract_skin_roi(aligned_rgb: np.ndarray, roi_patches: dict, skin_filter: dict) -> dict:
    height, width = aligned_rgb.shape[:2]

    ycrcb = cv2.cvtColor(aligned_rgb[..., :3].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    cb = ycrcb[..., 2].astype(np.int16)
    cr = ycrcb[..., 1].astype(np.int16)

    cb_low, cb_high = skin_filter["cb_range"]
    cr_low, cr_high = skin_filter["cr_range"]
    skin_mask = (cb >= cb_low) & (cb <= cb_high) & (cr >= cr_low) & (cr <= cr_high)

    valid_chunks = []
    total_roi_pixels = 0

    for patch in roi_patches.values():
        x_low, x_high = _clip_fraction_range(patch["x"][0], patch["x"][1])
        y_low, y_high = _clip_fraction_range(patch["y"][0], patch["y"][1])

        x1 = min(max(int(round(x_low * width)), 0), width)
        x2 = min(max(int(round(x_high * width)), 0), width)
        y1 = min(max(int(round(y_low * height)), 0), height)
        y2 = min(max(int(round(y_high * height)), 0), height)

        if x2 <= x1 or y2 <= y1:
            continue

        region_rgb = aligned_rgb[y1:y2, x1:x2, :3]
        region_mask = skin_mask[y1:y2, x1:x2]

        total_roi_pixels += int(region_mask.size)

        selected = region_rgb[region_mask]
        if len(selected):
            valid_chunks.append(selected)

    if valid_chunks:
        valid_pixels = np.concatenate(valid_chunks, axis=0)
    else:
        valid_pixels = np.empty((0, 3), dtype=np.uint8)

    valid_count = int(len(valid_pixels))
    valid_fraction = valid_count / total_roi_pixels if total_roi_pixels > 0 else 0.0

    return {
        "pixels": valid_pixels,
        "total_roi_pixels": total_roi_pixels,
        "valid_pixel_count": valid_count,
        "valid_fraction": float(valid_fraction),
    }


def extract_skin_roi_pixels(aligned_rgb: np.ndarray, roi_patches: dict, skin_filter: dict) -> np.ndarray:
    return extract_skin_roi(aligned_rgb, roi_patches, skin_filter)["pixels"]


def cielab_median(rgb_pixels: np.ndarray) -> Optional[tuple[float, float, float]]:
    if len(rgb_pixels) == 0:
        return None
    lab = rgb2lab((rgb_pixels.astype(np.float64) / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    return float(np.median(lab[:, 0])), float(np.median(lab[:, 1])), float(np.median(lab[:, 2]))


def compute_ita(l_star: float, b_star: float, b_star_min_abs: float) -> dict:
    if abs(float(b_star)) < float(b_star_min_abs):
        return {"ita": None, "ita_undefined": True}
    ita = math.degrees(math.atan((float(l_star) - 50.0) / float(b_star)))
    return {"ita": float(ita), "ita_undefined": False}


def compute_hue(a_star: float, b_star: float) -> float:
    return float(math.degrees(math.atan2(float(b_star), float(a_star))) % 360.0)


def measure_face_crop(
    rgb: np.ndarray,
    source_achromatic: bool,
    roi_patches: dict,
    skin_filter: dict,
    roi_min_valid_pixels: int,
    b_star_min_abs: float,
    bbox: Optional[list[float]] = None,
    original_width: Optional[int] = None,
    original_height: Optional[int] = None,
) -> dict:
    row = {
        "brightness": brightness(rgb),
        "contrast": contrast(rgb),
        "sharpness": sharpness(rgb),
        "face_bbox_area_px": None,
        "face_area_ratio": None,
        "yaw": None,
        "pitch": None,
        "roll": None,
        "source_achromatic": bool(source_achromatic),
        "skin_roi_total_pixel_count": None,
        "skin_roi_valid_pixel_count": None,
        "skin_valid_fraction": None,
        "skin_roi_valid": False,
        "skin_roi_invalid_reason": None,
        "L_star": None,
        "a_star": None,
        "b_star": None,
        "ITA": None,
        "ita_undefined": None,
        "hue_angle": None,
    }

    if bbox is not None and original_width is not None and original_height is not None:
        row.update(face_size(bbox, int(original_width), int(original_height)))

    if source_achromatic:
        row["skin_roi_invalid_reason"] = "source image is achromatic"
        return row

    skin = extract_skin_roi(rgb, roi_patches, skin_filter)

    row["skin_roi_total_pixel_count"] = skin["total_roi_pixels"]
    row["skin_roi_valid_pixel_count"] = skin["valid_pixel_count"]
    row["skin_valid_fraction"] = skin["valid_fraction"]

    if skin["valid_pixel_count"] < int(roi_min_valid_pixels):
        row["skin_roi_invalid_reason"] = "too few valid skin pixels"
        return row

    row["skin_roi_valid"] = True

    lab = cielab_median(skin["pixels"])
    if lab is None:
        row["skin_roi_valid"] = False
        row["skin_roi_invalid_reason"] = "cielab conversion returned no value"
        return row

    row["L_star"], row["a_star"], row["b_star"] = lab

    ita_result = compute_ita(row["L_star"], row["b_star"], b_star_min_abs)
    row["ITA"] = ita_result["ita"]
    row["ita_undefined"] = ita_result["ita_undefined"]

    row["hue_angle"] = compute_hue(row["a_star"], row["b_star"])

    return row
