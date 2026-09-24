from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import paths
from src.detection_integrity import (
    DetectionAuditInputs,
    build_detection_integrity_table,
    build_subgroup_failure_rows,
    build_confidence_rows,
    build_alignment_failure_rows,
    build_provenance_notes,
    build_terminal_state_table,
    prepare_detection_frame,
)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Phase 0 input missing: {path}"
        )
    return path


def main() -> None:
    paths.ensure_dirs()

    master_path = _require(
        paths.METADATA_DIR / "master_annotations.parquet"
    )
    manifest_path = _require(
        paths.METADATA_DIR / "file_manifest.parquet"
    )
    diagnostics_path = _require(
        paths.METADATA_DIR
        / "detection_alignment_diagnostics.parquet"
    )

    master = pd.read_parquet(master_path)
    manifest = pd.read_parquet(manifest_path)
    diagnostics = pd.read_parquet(diagnostics_path)

    frame = prepare_detection_frame(
        DetectionAuditInputs(
            master=master,
            manifest=manifest,
            diagnostics=diagnostics,
        )
    )

    # Reconcile before writing any analysis outputs.
    integrity_table = build_detection_integrity_table(frame)

    expected = {
        "fairface": 97698,
        "lfw": 13233,
        "utkface": 24108,
    }

    actual = dict(
        zip(
            integrity_table["dataset"],
            integrity_table["canonical_images"],
        )
    )

    if actual != expected:
        raise RuntimeError(
            f"Unexpected canonical populations: {actual}"
        )

    totals = {
        "entered_detector": 37326,
        "zero_detection": 39,
        "inference_error": 0,
        "aligned_success": 37287,
    }

    for column, expected_count in totals.items():
        actual_count = int(integrity_table[column].sum())

        if actual_count != expected_count:
            raise RuntimeError(
                f"Detection integrity {column}: "
                f"expected {expected_count}, got {actual_count}"
            )

    terminal = build_terminal_state_table(frame)
    subgroup_failures = build_subgroup_failure_rows(frame)

    confidence_summary, confidence_values = (
        build_confidence_rows(frame)
    )

    alignment_failures = build_alignment_failure_rows(frame)
    provenance = build_provenance_notes(frame)

    output_dir = (
        paths.DATA_PREP_OUTPUTS_DIR
        / "detection_integrity"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "detection_integrity_by_dataset.csv": integrity_table,
        "detection_terminal_states.csv": terminal,
        "subgroup_detection_failure_analysis_ready.csv": subgroup_failures,
        "confidence_summary.csv": confidence_summary,
        "alignment_failure_analysis_ready.csv": alignment_failures,
        "detection_provenance.csv": provenance,
    }

    for filename, table in outputs.items():
        table.to_csv(
            output_dir / filename,
            index=False,
        )

    confidence_values.to_parquet(
        output_dir / "confidence_values.parquet",
        index=False,
    )

    status = {
        "canonical_rows": int(len(frame)),
        "datasets": sorted(
            frame["dataset"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "detection_integrity": {
            "status": "reconciliation_generated",
            "note": "Cross-checked against Phase 0.",
        },
        "subgroup_failures": {
            "status": "descriptive_counts_generated",
            "pending": [
                "STAT-001 confidence intervals and tests",
                "minimum-cell reporting rules",
            ],
        },
        "confidence": {
            "status": "confidence_distributions_generated",
            "pending": [
                "approved effect-size procedures",
                "planned multiple-comparison corrections",
            ],
        },
        "alignment": {
            "status": "descriptive_counts_generated",
            "pending": [
                "approved statistical procedures",
                "alignment QA interpretation",
            ],
        },
        "failure_review": {
            "status": "draft_template",
            "pending": [
                "review the detection integrity, subgroup, confidence and alignment tables",
            ],
        },
    }

    status_path = output_dir / "detection_audit_run_status.json"

    status_path.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )

    print("\nDETECTION INTEGRITY")
    print(integrity_table.to_string(index=False))

    print("\nTERMINAL STATES")
    print(terminal.to_string(index=False))

    print("\nOutputs saved to:", output_dir)

    for filename in outputs:
        print(" -", filename)

    print(" - confidence_values.parquet")
    print(" - detection_audit_run_status.json")

    print(
        "\nThese are descriptive outputs, "
        "not final statistical conclusions."
    )


if __name__ == "__main__":
    main()
