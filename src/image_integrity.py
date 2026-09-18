
from __future__ import annotations

import time
import warnings
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageCms, ImageOps

from src import paths

try:
    from skimage.color import rgb2lab

    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False

EXIF_ORIENTATION_TAG = 274

# Chroma threshold for skin measurements.
ACHROMATIC_CHROMA_TOLERANCE = 3.0

# Minimum coloured pixel fraction.
ACHROMATIC_MIN_FRACTION_BELOW_TOL = 0.99

ACHROMATIC_SAMPLE_MAX_PIXELS = 10_000
ACHROMATIC_SAMPLE_SEED = 26041


def _exif_orientation(img: Image.Image) -> Optional[int]:
    try:
        exif = img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    return exif.get(EXIF_ORIENTATION_TAG)


def _lab_chroma_stats(arr: np.ndarray) -> dict:
    empty = {
        "mean_a_star": None,
        "mean_b_star": None,
        "mean_chroma": None,
        "p95_chroma": None,
        "max_chroma": None,
        "frac_pixels_below_chroma_tol": None,
        "achromatic_rgb": None,
    }
    if not _HAVE_SKIMAGE:
        return empty

    rgb = arr[..., :3].reshape(-1, 3)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    if len(rgb) > ACHROMATIC_SAMPLE_MAX_PIXELS:
        rng = np.random.default_rng(ACHROMATIC_SAMPLE_SEED)
        idx = rng.choice(len(rgb), size=ACHROMATIC_SAMPLE_MAX_PIXELS, replace=False)
        rgb = rgb[idx]

    lab = rgb2lab((rgb / 255.0).reshape(-1, 1, 3)).reshape(-1, 3)
    chroma = np.sqrt(lab[:, 1] ** 2 + lab[:, 2] ** 2)
    frac_below = float((chroma < ACHROMATIC_CHROMA_TOLERANCE).mean())
    return {
        "mean_a_star": float(lab[:, 1].mean()),
        "mean_b_star": float(lab[:, 2].mean()),
        "mean_chroma": float(chroma.mean()),
        "p95_chroma": float(np.percentile(chroma, 95)),
        "max_chroma": float(chroma.max()),
        "frac_pixels_below_chroma_tol": frac_below,
        "achromatic_rgb": bool(frac_below >= ACHROMATIC_MIN_FRACTION_BELOW_TOL),
    }


JPEG_SUFFIXES = {".jpg", ".jpeg"}
JPEG_SOI, JPEG_EOI = b"\xff\xd8", b"\xff\xd9"


def _jpeg_looks_complete(path: Path) -> Optional[bool]:
    if path.suffix.lower() not in JPEG_SUFFIXES:
        return None
    try:
        with open(path, "rb") as fh:
            head = fh.read(2)
            if len(head) < 2:
                return False
            fh.seek(-2, 2)
            return head == JPEG_SOI and fh.read(2) == JPEG_EOI
    except OSError:
        return False


def load_rgb_for_analysis(path: Path) -> tuple[np.ndarray, dict]:
    with Image.open(path) as image:
        original_mode = image.mode
        exif_orientation = _exif_orientation(image)
        icc_profile = image.info.get("icc_profile")

        oriented = ImageOps.exif_transpose(image)
        color_conversion = "assumed_srgb_no_profile"
        color_conversion_error = None

        if icc_profile:
            try:
                source_profile = ImageCms.ImageCmsProfile(BytesIO(icc_profile))
                destination_profile = ImageCms.createProfile("sRGB")
                converted = ImageCms.profileToProfile(
                    oriented.convert("RGB"),
                    source_profile,
                    destination_profile,
                    outputMode="RGB",
                )
                color_conversion = "embedded_profile_to_srgb"
            except Exception as exc:
                converted = oriented.convert("RGB")
                color_conversion = "profile_conversion_failed_assumed_srgb"
                color_conversion_error = f"{type(exc).__name__}: {exc}"
        else:
            converted = oriented.convert("RGB")

        converted.load()
        array = np.asarray(converted).copy()

    metadata = {
        "original_mode": original_mode,
        "exif_orientation": exif_orientation,
        "icc_profile_present": bool(icc_profile),
        "color_conversion": color_conversion,
        "color_conversion_error": color_conversion_error,
    }

    return array, metadata


def decode_and_measure_image(path: Path) -> dict:
    try:
        stored_path = paths.to_rel(path)
    except ValueError:
        stored_path = str(path)

    row = {
        "path": stored_path,
        "decode_success": False,
        "decode_error": None,
        "header_width": None,
        "header_height": None,
        "decoded_width": None,
        "decoded_height": None,
        "decoded_channels": None,
        "array_shape": None,
        "header_decoded_size_match": None,
        "original_mode": None,
        "exif_orientation": None,
        "icc_profile_present": None,
        "pixel_mean": None,
        "pixel_std": None,
        "achromatic_rgb": None,
        "mean_a_star": None,
        "mean_b_star": None,
        "mean_chroma": None,
        "p95_chroma": None,
        "max_chroma": None,
        "frac_pixels_below_chroma_tol": None,
        "jpeg_structurally_complete": None,
        "decompression_bomb_warning": False,
        "decode_ms": None,
    }

    start = time.perf_counter()

    # Reject JPEGs without an end marker before loading.
    complete = _jpeg_looks_complete(path)
    row["jpeg_structurally_complete"] = complete
    if complete is False:
        row["decode_error"] = "TruncatedJPEG: end-of-image marker missing, not decoded"
        row["decode_ms"] = (time.perf_counter() - start) * 1000.0
        return row

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", Image.DecompressionBombWarning)

            with Image.open(path) as img:
                row["header_width"], row["header_height"] = img.size
                row["original_mode"] = img.mode
                row["exif_orientation"] = _exif_orientation(img)
                row["icc_profile_present"] = "icc_profile" in img.info

                oriented = ImageOps.exif_transpose(img)
                try:
                    oriented.load()  # force full pixel decode -- this is the point
                    arr = np.asarray(oriented)
                finally:
                    if oriented is not img:
                        oriented.close()

            row["decompression_bomb_warning"] = any(
                issubclass(w.category, Image.DecompressionBombWarning) for w in caught
            )

        row["array_shape"] = list(arr.shape)
        if arr.ndim == 2:
            row["decoded_height"], row["decoded_width"] = arr.shape
            row["decoded_channels"] = 1
        else:
            row["decoded_height"], row["decoded_width"] = arr.shape[:2]
            row["decoded_channels"] = arr.shape[2]

        header_wh = (row["header_width"], row["header_height"])
        decoded_wh = (row["decoded_width"], row["decoded_height"])
        # EXIF rotation can swap width and height.
        row["header_decoded_size_match"] = decoded_wh in (header_wh, header_wh[::-1])

        # Exclude alpha from image statistics.
        colour = arr[..., :3] if arr.ndim == 3 else arr
        row["pixel_mean"] = float(colour.mean())
        row["pixel_std"] = float(colour.std())

        if arr.ndim == 3 and arr.shape[2] >= 3:
            row.update(_lab_chroma_stats(arr))
        else:
            # Single channel images have no chroma.
            row["achromatic_rgb"] = True

        row["decode_success"] = True
        del arr  # explicit: nothing decoded should outlive this function call

    except Exception as exc:
        row["decode_error"] = f"{type(exc).__name__}: {exc}"

    row["decode_ms"] = (time.perf_counter() - start) * 1000.0
    return row


def process_image_batch(
    image_paths: list[Path],
    checkpoint_path: Path,
    checkpoint_every: int = 5000,
) -> pd.DataFrame:
    def _rel(p) -> str:
        try:
            return paths.to_rel(p)
        except ValueError:
            return str(p)

    rows: list[dict] = []
    already_done: set[str] = set()

    if checkpoint_path.exists():
        existing = pd.read_parquet(checkpoint_path)
        rows = existing.to_dict("records")
        already_done = set(existing["path"])  # already relative, per decode_and_measure_image
        print(f"Resuming from checkpoint: {len(already_done)} images already processed.")

    pending = [p for p in image_paths if _rel(p) not in already_done]
    print(f"{len(pending)} images to decode (of {len(image_paths)} total).")

    for i, path in enumerate(pending, start=1):
        rows.append(decode_and_measure_image(path))
        if i % checkpoint_every == 0:
            pd.DataFrame(rows).to_parquet(checkpoint_path, index=False)
            print(f"  checkpoint: {i}/{len(pending)} new images decoded ({len(rows)} total saved)")

    out = pd.DataFrame(rows)
    out.to_parquet(checkpoint_path, index=False)
    print(f"Done. {len(rows)} total rows saved to {checkpoint_path}")

    # Confirm all selected files were processed.
    requested = {_rel(p) for p in image_paths}
    produced = set(out["path"])
    missing = requested - produced
    if missing:
        raise RuntimeError(
            f"{len(missing)} requested paths absent from output. "
            f"First few: {sorted(missing)[:5]}"
        )

    return out
