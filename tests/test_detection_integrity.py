from __future__ import annotations

import pandas as pd
import pytest

from src.detection_integrity import (
    DetectionAuditInputs,
    build_detection_integrity_table,
    build_subgroup_failure_rows,
    build_confidence_rows,
    build_alignment_failure_rows,
    prepare_detection_frame,
)


def _fixtures():
    master = pd.DataFrame(
        [
            {
                "image_id": "lfw-a",
                "dataset": "lfw",
                "original_path": "LFW/a.jpg",
                "subject_id": "a",
                "corrupt": False,
                "gender": "female",
                "gender_source": "derived",
                "race": "unknown",
                "race_source": "unavailable",
                "age_band": "unknown",
                "det_confidence": 0.99,
                "n_detections": 1,
                "no_detection": False,
                "multi_detection": False,
                "bbox": [1, 1, 20, 20],
                "landmarks": [
                    [2, 2],
                    [3, 2],
                    [2.5, 3],
                    [2, 4],
                    [3, 4],
                ],
                "alignment_valid": True,
            },
            {
                "image_id": "lfw-b",
                "dataset": "lfw",
                "original_path": "LFW/b.jpg",
                "subject_id": "b",
                "corrupt": False,
                "gender": "male",
                "gender_source": "derived",
                "race": "unknown",
                "race_source": "unavailable",
                "age_band": "unknown",
                "det_confidence": None,
                "n_detections": 0,
                "no_detection": True,
                "multi_detection": False,
                "bbox": None,
                "landmarks": None,
                "alignment_valid": False,
            },
            {
                "image_id": "fairface-a",
                "dataset": "fairface",
                "original_path": "FairFace/a.jpg",
                "subject_id": None,
                "corrupt": False,
                "gender": "female",
                "gender_source": "native",
                "race": "Black",
                "race_source": "native",
                "age_band": "20-29",
                "det_confidence": None,
                "n_detections": None,
                "no_detection": None,
                "multi_detection": None,
                "bbox": None,
                "landmarks": None,
                "alignment_valid": True,
            },
        ]
    )

    manifest = pd.DataFrame(
        [
            {
                "file_role": "image_candidate",
                "original_path": "LFW/a.jpg",
                "source_preprocessed": False,
                "project_detection_required": True,
                "project_alignment_required": True,
            },
            {
                "file_role": "image_candidate",
                "original_path": "LFW/b.jpg",
                "source_preprocessed": False,
                "project_detection_required": True,
                "project_alignment_required": True,
            },
            {
                "file_role": "image_candidate",
                "original_path": "FairFace/a.jpg",
                "source_preprocessed": True,
                "project_detection_required": False,
                "project_alignment_required": False,
            },
        ]
    )

    diagnostics = pd.DataFrame(
        [
            {
                "image_id": "lfw-a",
                "inference_success": True,
                "inference_error": None,
            },
            {
                "image_id": "lfw-b",
                "inference_success": True,
                "inference_error": None,
            },
            {
                "image_id": "fairface-a",
                "inference_success": None,
                "inference_error": None,
            },
        ]
    )

    return master, manifest, diagnostics


def test_detection_integrity_reconciles():
    master, manifest, diagnostics = _fixtures()

    frame = prepare_detection_frame(
        DetectionAuditInputs(
            master,
            manifest,
            diagnostics,
        )
    )

    table = build_detection_integrity_table(frame)

    lfw = table.loc[
        table["dataset"].eq("lfw")
    ].iloc[0]

    assert lfw["entered_detector"] == 2
    assert lfw["detected_ge1"] == 1
    assert lfw["zero_detection"] == 1
    assert lfw["aligned_success"] == 1

    fairface = table.loc[
        table["dataset"].eq("fairface")
    ].iloc[0]

    assert fairface["entered_detector"] == 0

    assert (
        fairface[
            "publisher_preprocessed_not_project_detected"
        ]
        == 1
    )


def test_diagnostics_overlap_with_completed_master():
    master, manifest, diagnostics = _fixtures()

    master["inference_success"] = [
        True,
        True,
        None,
    ]

    master["inference_error"] = [
        None,
        None,
        None,
    ]

    frame = prepare_detection_frame(
        DetectionAuditInputs(
            master,
            manifest,
            diagnostics,
        )
    )

    assert "inference_success" in frame.columns
    assert "inference_success_x" not in frame.columns

    value = frame.loc[
        frame["image_id"].eq("lfw-a"),
        "inference_success",
    ].iloc[0]

    assert value == True


def test_disagreeing_diagnostics_fail_loudly():
    master, manifest, diagnostics = _fixtures()

    master["inference_success"] = [
        False,
        True,
        None,
    ]

    with pytest.raises(
        RuntimeError,
        match="disagrees with diagnostics",
    ):
        prepare_detection_frame(
            DetectionAuditInputs(
                master,
                manifest,
                diagnostics,
            )
        )


def test_missing_diagnostics_fail_loudly():
    master, manifest, diagnostics = _fixtures()

    with pytest.raises(
        RuntimeError,
        match="identical image IDs",
    ):
        prepare_detection_frame(
            DetectionAuditInputs(
                master,
                manifest,
                diagnostics.iloc[:-1],
            )
        )


def test_fairface_excluded_from_detector_denominators():
    master, manifest, diagnostics = _fixtures()

    frame = prepare_detection_frame(
        DetectionAuditInputs(
            master,
            manifest,
            diagnostics,
        )
    )

    subgroup_failures = build_subgroup_failure_rows(frame)

    confidence_summary, confidence_values = (
        build_confidence_rows(frame)
    )

    alignment_failures = build_alignment_failure_rows(frame)

    for table in [
        subgroup_failures,
        confidence_summary,
        confidence_values,
        alignment_failures,
    ]:
        assert "fairface" not in set(table["dataset"])
