from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps


def load_rgb_for_analysis(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load an image deterministically for analysis.

    Applies EXIF orientation and, when possible, converts an embedded ICC profile to sRGB.
    Returns an RGB uint8 array plus lightweight provenance metadata.
    """
    p = Path(path)
    with Image.open(p) as im0:
        im = ImageOps.exif_transpose(im0)
        had_icc = bool(im.info.get("icc_profile"))
        icc_converted = False
        if had_icc:
            try:
                src = ImageCms.ImageCmsProfile(BytesIO(im.info["icc_profile"]))
                dst = ImageCms.createProfile("sRGB")
                im = ImageCms.profileToProfile(im, src, dst, outputMode="RGB")
                icc_converted = True
            except Exception:
                im = im.convert("RGB")
        else:
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    return arr, {
        "path": str(p),
        "exif_orientation_applied": True,
        "embedded_icc_present": had_icc,
        "icc_converted_to_srgb": icc_converted,
        "mode": "RGB",
    }
