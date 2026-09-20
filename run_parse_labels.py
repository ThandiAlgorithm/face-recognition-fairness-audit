from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, metadata_parsers, paths


def main() -> None:
    paths.ensure_dirs()
    preprocessing = config.load_preprocessing()
    categories = config.load_categories()
    config.validate_categories(categories)

    manifest = pd.read_parquet(paths.METADATA_DIR / "file_manifest.parquet")
    labels = metadata_parsers.build_label_table(manifest, preprocessing, categories)
    labels_path = paths.METADATA_DIR / "parsed_labels.parquet"
    labels.to_parquet(labels_path, index=False)

    summary = (
        labels.assign(parse_ok=~labels["parse_failed"])
        .groupby("split", dropna=False)["parse_ok"]
        .agg(["count", "sum"])
        .reset_index()
    )
    summary_path = paths.DATA_PREP_OUTPUTS_DIR / "tables" / "label_parse_summary.csv"
    summary.to_csv(summary_path, index=False)

    review_path = paths.METADATA_DIR / "source_review.yaml"
    if not review_path.exists():
        raise FileNotFoundError(f"Missing human source review file: {review_path}")
    with open(review_path, "r", encoding="utf-8") as handle:
        review = yaml.safe_load(handle)

    review_complete = True
    for dataset in ["lfw", "fairface", "utkface"]:
        item = review.get(dataset, {})
        required = [
            item.get("official_source_reviewed") is True,
            item.get("errata_reviewed") is True,
            item.get("identity_metadata_reviewed") is True,
        ]
        if not all(required):
            review_complete = False

    report = {
        "rows": int(len(labels)),
        "parse_failed": int(labels["parse_failed"].sum()),
        "source_review_complete": review_complete,
    }
    report_path = paths.DATA_PREP_OUTPUTS_DIR / "reports" / "labels_and_source_review.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not review_complete:
        print("DATA-005 remains open until metadata/source_review.yaml is reviewed and completed.")


if __name__ == "__main__":
    main()
