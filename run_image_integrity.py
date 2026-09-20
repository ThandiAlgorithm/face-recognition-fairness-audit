from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, image_integrity, paths


def main() -> None:
    paths.ensure_dirs()
    preprocessing = config.load_preprocessing()
    manifest = pd.read_parquet(paths.METADATA_DIR / "file_manifest.parquet")
    checkpoint = paths.CACHE_DIR / "image_integrity.parquet"

    integrity = image_integrity.inspect_manifest_images(
        manifest,
        preprocessing,
        checkpoint,
    )
    output_path = paths.METADATA_DIR / "image_integrity.parquet"
    integrity.to_parquet(output_path, index=False)

    summary = {
        "image_candidates": int(len(integrity)),
        "decode_success": int(integrity["decode_success"].sum()),
        "decode_failure": int((~integrity["decode_success"]).sum()),
        "is_greyscale": int(integrity["is_greyscale"].fillna(False).sum()),
        "exif_applied": int(integrity["exif_applied"].fillna(False).sum()),
        "colour_profile_assumed": int(integrity["colour_profile_assumed"].fillna(False).sum()),
        "colour_conversion_failure": int((integrity["colour_conversion_success"] == False).sum()),
    }
    report_path = paths.DATA_PREP_OUTPUTS_DIR / "reports" / "integrity_summary.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if summary["colour_conversion_failure"] > 0:
        print("Colour conversion failures exist. Review them before closing DATA-003.")

    print(json.dumps(summary, indent=2))
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
