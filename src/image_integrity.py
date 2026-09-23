from __future__ import annotations

import time
import warnings
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageCms, ImageOps
from skimage.color import rgb2lab

from src import paths

EXIF_ORIENTATION_TAG = 274
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


def _exif_orientation(image: Image.Image) -> Optional[int]:
    try:
        exif = image.getexif()
    except Exception:
        return None
    if not exif:
        return None
    return exif.get(EXIF_ORIENTATION_TAG)


def _jpeg_structurally_complete(path: Path) -> Optional[bool]:
    try:
        with open(path, "rb") as handle:
            head = handle.read(2)
            if head != JPEG_SOI:
                return None
            handle.seek(-2, 2)
            return handle.read(2) == JPEG_EOI
    except OSError:
        return False


def _profile_name(profile_bytes: bytes) -> Optional[str]:
    try:
        profile = ImageCms.ImageCmsProfile(BytesIO(profile_bytes))
        return str(ImageCms.getProfileName(profile)).strip() or "embedded_icc"
    except Exception:
        return "embedded_icc_unreadable_name"


def _sample_lab_chroma(rgb: np.ndarray, max_pixels: int, seed: int) -> dict:
    pixels = rgb.reshape(-1, 3)
    if len(pixels) > max_pixels:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(pixels), size=max_pixels, replace=False)
        pixels = pixels[indices]
    lab = rgb2lab((pixels.astype(np.float64) / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    return {
        "mean_a_star": float(lab[:, 1].mean()),
        "mean_b_star": float(lab[:, 2].mean()),
        "mean_abs_a_star": float(np.abs(lab[:, 1]).mean()),
        "mean_abs_b_star": float(np.abs(lab[:, 2]).mean()),
    }


def load_rgb_for_analysis(path: Path) -> tuple[np.ndarray, dict]:
    with Image.open(path) as image:
        original_mode = image.mode
        original_size = image.size
        original_format = image.format
        exif_orientation = _exif_orientation(image)
        icc_bytes = image.info.get("icc_profile")
        colour_profile = _profile_name(icc_bytes) if icc_bytes else None

        oriented = ImageOps.exif_transpose(image)
        exif_applied = exif_orientation not in {None, 1}
        colour_profile_assumed = icc_bytes is None
        colour_conversion = "assumed_srgb"
        colour_conversion_success = True
        colour_conversion_error = None

        if icc_bytes:
            try:
                source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_bytes))
                target_profile = ImageCms.createProfile("sRGB")
                converted = ImageCms.profileToProfile(
                    oriented.convert("RGB"),
                    source_profile,
                    target_profile,
                    outputMode="RGB",
                )
                colour_conversion = "embedded_profile_to_srgb"
            except Exception as exc:
                converted = oriented.convert("RGB")
                colour_conversion = "profile_conversion_failed"
                colour_conversion_success = False
                colour_conversion_error = f"{type(exc).__name__}: {exc}"
        else:
            converted = oriented.convert("RGB")

        converted.load()
        rgb = np.asarray(converted, dtype=np.uint8).copy()

    metadata = {
        "original_mode": original_mode,
        "header_width_px": int(original_size[0]),
        "header_height_px": int(original_size[1]),
        "format": original_format,
        "exif_orientation": exif_orientation,
        "exif_applied": bool(exif_applied),
        "colour_profile": colour_profile,
        "colour_profile_assumed": bool(colour_profile_assumed),
        "colour_conversion": colour_conversion,
        "colour_conversion_success": bool(colour_conversion_success),
        "colour_conversion_error": colour_conversion_error,
    }
    return rgb, metadata


def inspect_image(path: Path, preprocessing_cfg: dict) -> dict:
    start = time.perf_counter()
    row = {
        "original_path": paths.data_rel(path),
        "decode_success": False,
        "decode_error": None,
        "width_px": None,
        "height_px": None,
        "original_mode": None,
        "format": None,
        "colour_profile": None,
        "colour_profile_assumed": None,
        "exif_orientation": None,
        "exif_applied": None,
        "colour_conversion": None,
        "colour_conversion_success": None,
        "colour_conversion_error": None,
        "is_greyscale": None,
        "jpeg_structurally_complete": None,
        "decompression_bomb_warning": False,
        "mean_a_star": None,
        "mean_b_star": None,
        "decode_ms": None,
    }

    complete = _jpeg_structurally_complete(path)
    row["jpeg_structurally_complete"] = complete
    if complete is False:
        row["decode_error"] = "TruncatedJPEG: missing end marker"
        row["decode_ms"] = (time.perf_counter() - start) * 1000.0
        return row

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", Image.DecompressionBombWarning)
            rgb, info = load_rgb_for_analysis(path)
            row["decompression_bomb_warning"] = any(
                issubclass(item.category, Image.DecompressionBombWarning)
                for item in caught
            )

        row.update(info)
        row["height_px"], row["width_px"] = [int(value) for value in rgb.shape[:2]]

        greyscale_cfg = preprocessing_cfg["greyscale"]
        chroma = _sample_lab_chroma(
            rgb,
            int(greyscale_cfg.get("sample_max_pixels", 10000)),
            int(greyscale_cfg.get("sample_seed", 26041)),
        )
        row["mean_a_star"] = chroma["mean_a_star"]
        row["mean_b_star"] = chroma["mean_b_star"]

        tolerance = float(greyscale_cfg["chroma_tolerance"])
        mode_is_l = row["original_mode"] == "L"
        chroma_near_zero = (
            abs(chroma["mean_a_star"]) <= tolerance
            and abs(chroma["mean_b_star"]) <= tolerance
        )
        row["is_greyscale"] = bool(mode_is_l or chroma_near_zero)
        row["decode_success"] = True

    except Exception as exc:
        row["decode_error"] = f"{type(exc).__name__}: {exc}"

    row["decode_ms"] = (time.perf_counter() - start) * 1000.0
    return row


def inspect_manifest_images(
    manifest: pd.DataFrame,
    preprocessing_cfg: dict,
    checkpoint_path: Path,
    checkpoint_every: int = 2000,
) -> pd.DataFrame:
    images = manifest[
        manifest["file_role"].eq(
            "image_candidate"
        )
    ].copy()

    expected = set(
        images["original_path"].astype(str)
    )

    rows = []
    completed = set()

    if checkpoint_path.exists():
        prior = pd.read_parquet(
            checkpoint_path
        )

        prior = prior[
            prior["original_path"]
            .astype(str)
            .isin(expected)
        ].copy()

        rows = prior.to_dict(
            "records"
        )

        completed = set(
            prior["original_path"]
            .astype(str)
        )

    pending = images[
        ~images["original_path"].isin(
            completed
        )
    ]

    for count, original_path in enumerate(
        pending["original_path"],
        start=1,
    ):
        rows.append(
            inspect_image(
                paths.data_abs(
                    original_path
                ),
                preprocessing_cfg,
            )
        )

        if count % checkpoint_every == 0:
            pd.DataFrame(
                rows
            ).to_parquet(
                checkpoint_path,
                index=False,
            )

            print(
                f"Integrity checkpoint: "
                f"{len(rows)} rows"
            )

    output = pd.DataFrame(
        rows
    )

    output.to_parquet(
        checkpoint_path,
        index=False,
    )

    produced = set(
        output["original_path"]
        .astype(str)
    )

    missing = expected - produced
    extra = produced - expected

    if missing:
        raise RuntimeError(
            f"Integrity output is missing "
            f"{len(missing)} image candidates."
        )

    if extra:
        raise RuntimeError(
            f"Integrity output contains "
            f"{len(extra)} stale rows."
        )

    return output
