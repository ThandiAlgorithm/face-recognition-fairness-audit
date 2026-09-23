from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config, data_inventory, paths


def main() -> None:
    paths.ensure_dirs()
    preprocessing = config.load_preprocessing()
    config.validate_preprocessing(preprocessing)

    manifest = data_inventory.build_file_manifest(preprocessing)
    manifest_path = paths.METADATA_DIR / "file_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)

    counts = data_inventory.dataset_count_report(manifest)
    counts_path = paths.DATA_PREP_OUTPUTS_DIR / "tables" / "canonical_source_counts.csv"
    counts.to_csv(counts_path, index=False)

    archive_checks = data_inventory.check_archives(preprocessing)
    archive_path = paths.DATA_PREP_OUTPUTS_DIR / "tables" / "archive_checks.csv"
    archive_checks.to_csv(archive_path, index=False)

    variant_report = []
    for dataset, dataset_cfg in preprocessing["datasets"].items():
        warnings = data_inventory.validate_dataset_variant(dataset, dataset_cfg)
        variant_report.append(
            {
                "dataset": dataset,
                "dataset_version": dataset_cfg["dataset_version"],
                "variant": dataset_cfg["variant"],
                "image_roots": dataset_cfg["image_roots"],
                "source_preprocessed": dataset_cfg["source_preprocessed"],
                "project_detection_required": dataset_cfg["project_detection_required"],
                "warnings": warnings,
            }
        )
    report_path = paths.DATA_PREP_OUTPUTS_DIR / "reports" / "dataset_variant_report.json"
    report_path.write_text(json.dumps(variant_report, indent=2), encoding="utf-8")

    print(counts.to_string(index=False))
    print()
    if archive_checks.empty:
        print("No archives are configured. DATA-002 cannot close.")
    else:
        print(archive_checks.to_string(index=False))
        if not archive_checks["checksum_matches"].all():
            print()
            print("DATA-002 is not closed. Fill the expected SHA-256 values and rerun.")

    print()
    print("Saved:", manifest_path)
    print("Saved:", counts_path)
    print("Saved:", archive_path)


if __name__ == "__main__":
    main()
