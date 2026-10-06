from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ERROR_ROOT = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "reference_a"
    / "error_audit"
)

REPORT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "reference_a"
    / "report_pack"
)

AUDIT_PATH = (
    ERROR_ROOT
    / "reference_a_error_audit.csv"
)

SUMMARY_PATH = (
    ERROR_ROOT
    / "reference_a_error_summary.csv"
)

OUTPUT_CASES = (
    REPORT_ROOT
    / "error_cases_sanitized.csv"
)

OUTPUT_SUMMARY = (
    REPORT_ROOT
    / "error_summary.csv"
)


def main() -> None:

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing error audit:\n{AUDIT_PATH}"
        )

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing error summary:\n{SUMMARY_PATH}"
        )

    audit = pd.read_csv(
        AUDIT_PATH,
        low_memory=False,
    )

    summary = pd.read_csv(
        SUMMARY_PATH,
    )

    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    wanted = [
        "pair_index",
        "outcome",
        "protocol_label",
        "model_decision",
        "cosine_similarity",
        "automatic_flags",
        "has_multi_face_signal",
        "has_alignment_signal",
        "has_detection_signal",
        "image1_n_detections",
        "image2_n_detections",
        "image1_multi_detection",
        "image2_multi_detection",
        "image1_det_confidence",
        "image2_det_confidence",
        "image1_alignment_valid",
        "image2_alignment_valid",
        "image1_brightness",
        "image2_brightness",
        "image1_contrast",
        "image2_contrast",
        "image1_sharpness",
        "image2_sharpness",
        "image1_face_area_ratio",
        "image2_face_area_ratio",
    ]

    available = [
        column
        for column in wanted
        if column in audit.columns
    ]

    safe = audit[
        available
    ].copy()

    safe = safe.sort_values(
        [
            "outcome",
            "cosine_similarity",
        ],
        ascending=[
            True,
            True,
        ],
    )

    safe.to_csv(
        OUTPUT_CASES,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "REFERENCE A ERROR REPORT PACK"
    )

    print(
        "=" * 72
    )

    print(
        f"Error cases: {len(safe)}"
    )

    print(
        "Absolute local image paths removed."
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_CASES
    )

    print(
        OUTPUT_SUMMARY
    )


if __name__ == "__main__":
    main()
    