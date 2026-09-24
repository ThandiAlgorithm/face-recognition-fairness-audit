from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, lfw_protocol, paths


def main() -> None:
    paths.ensure_dirs()
    preprocessing = config.load_preprocessing()
    master = pd.read_parquet(paths.METADATA_DIR / "master_annotations.parquet")
    lfw_cfg = preprocessing["datasets"]["lfw"]
    pairs_path = paths.DATA_DIR / "LFW" / "pairs.csv"
    image_root = paths.DATA_DIR / lfw_cfg["image_roots"][0]

    people_check = lfw_protocol.validate_people_counts(
        paths.DATA_DIR / "LFW" / "people.csv",
        image_root,
    )
    people_check.to_csv(
        paths.DATA_PREP_OUTPUTS_DIR / "tables" / "lfw_people_validation.csv",
        index=False,
    )
    if not people_check["folder_exists"].all() or not people_check["count_matches"].all():
        raise RuntimeError("LFW people.csv validation failed.")

    pairs = lfw_protocol.build_lfw_pairs(master, pairs_path)
    output_path = paths.METADATA_DIR / "lfw_pairs.parquet"
    pairs.to_parquet(output_path, index=False)

    fold_summary = pairs.groupby(["fold", "is_genuine"]).size().unstack(fill_value=0)
    fold_summary.to_csv(paths.DATA_PREP_OUTPUTS_DIR / "tables" / "lfw_view2_fold_counts.csv")
    print(fold_summary)
    print("Saved:", output_path)
    print("LFW View 2 is generated as a sealed reporting table. Do not use it for development tuning.")


if __name__ == "__main__":
    main()
