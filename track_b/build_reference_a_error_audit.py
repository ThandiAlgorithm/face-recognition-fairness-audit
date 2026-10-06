from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAIR_SCORES_PATH = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "arcface_reference_aligned"
    / "verification"
    / "pair_scores.csv"
)

MASTER_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "master_annotations.parquet"
)

PHASE1_PATH = (
    PROJECT_ROOT
    / "data"
    / "derived"
    / "cache"
    / "detection_alignment_measurement.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "reference_a"
    / "error_audit"
)

AUDIT_PATH = (
    OUTPUT_DIR
    / "reference_a_error_audit.csv"
)

REVIEW_PATH = (
    OUTPUT_DIR
    / "reference_a_error_review.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "reference_a_error_summary.csv"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "reference_a_error_audit_metadata.json"
)


# ============================================================
# HELPERS
# ============================================================

def canonical_lfw_path(value) -> str:
    path = (
        str(value)
        .strip()
        .replace("\\", "/")
    )

    lower = path.lower()

    marker = "/data/lfw/"

    if marker in lower:
        index = lower.index(marker)

        path = path[
            index
            + len("/data/"):
        ]

    else:
        parts = path.split("/")

        for i, part in enumerate(parts):
            if part.lower() == "lfw":
                path = "/".join(
                    parts[i:]
                )
                break

    return path.lower()


def resolve_project_path(value) -> str:
    if pd.isna(value):
        return ""

    path = Path(
        str(value)
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return str(
        path.resolve()
    )


def bool_value(value) -> bool:
    if pd.isna(value):
        return False

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    text = (
        str(value)
        .strip()
        .lower()
    )

    return text in {
        "true",
        "1",
        "yes",
    }


def finite_or_nan(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


# ============================================================
# LOAD
# ============================================================

def load_inputs():
    for path in [
        PAIR_SCORES_PATH,
        MASTER_PATH,
        PHASE1_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file:\n{path}"
            )

    scores = pd.read_csv(
        PAIR_SCORES_PATH,
        low_memory=False,
    )

    master = pd.read_parquet(
        MASTER_PATH
    )

    phase1 = pd.read_parquet(
        PHASE1_PATH
    )

    return (
        scores,
        master,
        phase1,
    )


# ============================================================
# MASTER LOOKUP
# ============================================================

def prepare_master(
    master: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "dataset",
        "image_id",
        "original_path",
        "aligned_path",
    }

    missing = (
        required
        - set(
            master.columns
        )
    )

    if missing:
        raise KeyError(
            "Master table missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    lfw = master[
        master[
            "dataset"
        ]
        .astype(str)
        .str.lower()
        .eq("lfw")
    ].copy()

    lfw[
        "path_key"
    ] = lfw[
        "original_path"
    ].map(
        canonical_lfw_path
    )

    if lfw[
        "path_key"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate canonical LFW paths "
            "found in master table."
        )

    return lfw[
        [
            "path_key",
            "image_id",
            "original_path",
            "aligned_path",
        ]
    ].copy()


# ============================================================
# PHASE 1 LOOKUP
# ============================================================

def prepare_phase1(
    phase1: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "image_id",
        "det_confidence",
        "n_detections",
        "no_detection",
        "multi_detection",
        "alignment_valid",
        "brightness",
        "contrast",
        "sharpness",
        "face_px_area",
        "face_area_ratio",
        "yaw",
        "pitch",
        "roll",
        "ITA",
        "ita_group",
        "skin_group_ita",
    }

    missing = (
        required
        - set(
            phase1.columns
        )
    )

    if missing:
        raise KeyError(
            "Phase 1 cache missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    columns = [
        "image_id",
        "det_confidence",
        "n_detections",
        "no_detection",
        "multi_detection",
        "alignment_valid",
        "brightness",
        "contrast",
        "sharpness",
        "face_px_area",
        "face_area_ratio",
        "yaw",
        "pitch",
        "roll",
        "ITA",
        "ita_group",
        "skin_group_ita",
    ]

    lookup = phase1[
        columns
    ].copy()

    if lookup[
        "image_id"
    ].duplicated().any():
        duplicate_count = int(
            lookup[
                "image_id"
            ].duplicated().sum()
        )

        raise RuntimeError(
            "Phase 1 cache contains duplicate "
            f"image_id values: {duplicate_count}"
        )

    return lookup


# ============================================================
# ERROR PAIRS
# ============================================================

def select_test_errors(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "split",
        "image1_path",
        "image2_path",
        "is_genuine",
        "predicted_genuine",
        "cosine_similarity",
    }

    missing = (
        required
        - set(
            scores.columns
        )
    )

    if missing:
        raise KeyError(
            "Pair-score file missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    test = scores[
        scores[
            "split"
        ]
        .astype(str)
        .str.lower()
        .eq("test")
    ].copy()

    test[
        "is_genuine"
    ] = test[
        "is_genuine"
    ].map(
        bool_value
    )

    test[
        "predicted_genuine"
    ] = test[
        "predicted_genuine"
    ].map(
        bool_value
    )

    errors = test[
        test[
            "is_genuine"
        ]
        != test[
            "predicted_genuine"
        ]
    ].copy()

    errors = errors.reset_index(
        names="pair_index"
    )

    errors[
        "outcome"
    ] = np.where(
        errors[
            "is_genuine"
        ],
        "FN",
        "FP",
    )

    errors[
        "protocol_label"
    ] = np.where(
        errors[
            "is_genuine"
        ],
        "Genuine",
        "Impostor",
    )

    errors[
        "model_decision"
    ] = np.where(
        errors[
            "predicted_genuine"
        ],
        "Genuine",
        "Impostor",
    )

    return errors


# ============================================================
# ADD IMAGE METADATA
# ============================================================

def attach_image_metadata(
    errors: pd.DataFrame,
    master: pd.DataFrame,
    phase1: pd.DataFrame,
) -> pd.DataFrame:

    master_lookup = master.set_index(
        "path_key"
    )

    phase1_lookup = phase1.set_index(
        "image_id"
    )

    records = []

    phase_fields = [
        "det_confidence",
        "n_detections",
        "no_detection",
        "multi_detection",
        "alignment_valid",
        "brightness",
        "contrast",
        "sharpness",
        "face_px_area",
        "face_area_ratio",
        "yaw",
        "pitch",
        "roll",
        "ITA",
        "ita_group",
        "skin_group_ita",
    ]

    for row in errors.itertuples(
        index=False
    ):
        record = {
            "pair_index": (
                row.pair_index
            ),
            "outcome": (
                row.outcome
            ),
            "protocol_label": (
                row.protocol_label
            ),
            "model_decision": (
                row.model_decision
            ),
            "cosine_similarity": (
                finite_or_nan(
                    row.cosine_similarity
                )
            ),
        }

        for side in [
            1,
            2,
        ]:
            raw_path = getattr(
                row,
                f"image{side}_path",
            )

            key = canonical_lfw_path(
                raw_path
            )

            record[
                f"image{side}_raw_path"
            ] = str(
                raw_path
            )

            record[
                f"image{side}_identity_folder"
            ] = Path(
                str(
                    raw_path
                )
            ).parent.name

            if key not in (
                master_lookup.index
            ):
                record[
                    f"image{side}_master_found"
                ] = False

                continue

            m = master_lookup.loc[
                key
            ]

            image_id = m[
                "image_id"
            ]

            record[
                f"image{side}_master_found"
            ] = True

            record[
                f"image{side}_image_id"
            ] = image_id

            record[
                f"image{side}_original_path"
            ] = m[
                "original_path"
            ]

            record[
                f"image{side}_aligned_path"
            ] = resolve_project_path(
                m[
                    "aligned_path"
                ]
            )

            if image_id not in (
                phase1_lookup.index
            ):
                record[
                    f"image{side}_phase1_found"
                ] = False

                continue

            p = phase1_lookup.loc[
                image_id
            ]

            record[
                f"image{side}_phase1_found"
            ] = True

            for field in (
                phase_fields
            ):
                record[
                    f"image{side}_{field}"
                ] = p[
                    field
                ]

        record[
            "same_identity_folder"
        ] = (
            record.get(
                "image1_identity_folder"
            )
            ==
            record.get(
                "image2_identity_folder"
            )
        )

        records.append(
            record
        )

    return pd.DataFrame(
        records
    )


# ============================================================
# AUTOMATIC SIGNALS
# ============================================================

def add_flags(
    audit: pd.DataFrame,
) -> pd.DataFrame:
    out = audit.copy()

    def row_flags(
        row,
    ) -> str:
        flags = []

        for side in [
            1,
            2,
        ]:

            n = row.get(
                f"image{side}_n_detections"
            )

            multi = row.get(
                f"image{side}_multi_detection"
            )

            no_detection = row.get(
                f"image{side}_no_detection"
            )

            alignment_valid = row.get(
                f"image{side}_alignment_valid"
            )

            try:
                if (
                    pd.notna(n)
                    and float(n) > 1
                ):
                    flags.append(
                        f"image{side}:multiple_faces"
                    )
            except Exception:
                pass

            if bool_value(
                multi
            ):
                flag = (
                    f"image{side}:multi_detection"
                )

                if flag not in flags:
                    flags.append(
                        flag
                    )

            if bool_value(
                no_detection
            ):
                flags.append(
                    f"image{side}:no_detection"
                )

            if (
                pd.notna(
                    alignment_valid
                )
                and not bool_value(
                    alignment_valid
                )
            ):
                flags.append(
                    f"image{side}:alignment_invalid"
                )

        if not bool(
            row.get(
                "same_identity_folder",
                True,
            )
        ) and (
            row.get(
                "protocol_label"
            )
            == "Genuine"
        ):
            flags.append(
                "pair:identity_folder_mismatch"
            )

        if not flags:
            return "none"

        return "; ".join(
            flags
        )

    out[
        "automatic_flags"
    ] = out.apply(
        row_flags,
        axis=1,
    )

    out[
        "has_multi_face_signal"
    ] = out[
        "automatic_flags"
    ].str.contains(
        "multiple_faces|multi_detection",
        regex=True,
        na=False,
    )

    out[
        "has_alignment_signal"
    ] = out[
        "automatic_flags"
    ].str.contains(
        "alignment_invalid",
        regex=False,
        na=False,
    )

    out[
        "has_detection_signal"
    ] = out[
        "automatic_flags"
    ].str.contains(
        "no_detection",
        regex=False,
        na=False,
    )

    return out


# ============================================================
# REVIEW TEMPLATE
# ============================================================

def make_review_table(
    audit: pd.DataFrame,
) -> pd.DataFrame:

    review = audit.copy()

    review[
        "qa_primary_category"
    ] = ""

    review[
        "qa_image_side"
    ] = ""

    review[
        "qa_notes"
    ] = ""

    review[
        "reviewed"
    ] = False

    return review


# ============================================================
# SUMMARY
# ============================================================

def make_summary(
    audit: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for outcome in [
        "FN",
        "FP",
        "ALL",
    ]:

        if outcome == "ALL":
            frame = audit

        else:
            frame = audit[
                audit[
                    "outcome"
                ].eq(
                    outcome
                )
            ]

        rows.append(
            {
                "outcome": outcome,
                "n_errors": int(
                    len(frame)
                ),
                "pairs_with_multi_face_signal": int(
                    frame[
                        "has_multi_face_signal"
                    ].sum()
                ),
                "pairs_with_alignment_signal": int(
                    frame[
                        "has_alignment_signal"
                    ].sum()
                ),
                "pairs_with_detection_signal": int(
                    frame[
                        "has_detection_signal"
                    ].sum()
                ),
                "mean_similarity": (
                    float(
                        frame[
                            "cosine_similarity"
                        ].mean()
                    )
                    if not frame.empty
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    scores, master, phase1 = (
        load_inputs()
    )

    master = prepare_master(
        master
    )

    phase1 = prepare_phase1(
        phase1
    )

    errors = select_test_errors(
        scores
    )

    audit = attach_image_metadata(
        errors,
        master,
        phase1,
    )

    audit = add_flags(
        audit
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )

    review = make_review_table(
        audit
    )

    review.to_csv(
        REVIEW_PATH,
        index=False,
    )

    summary = make_summary(
        audit
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "model_id": (
            "ReferenceA_ArcFace"
        ),
        "evaluation_split": (
            "LFW View 1 test"
        ),
        "error_definition": (
            "pair decision differs from "
            "LFW protocol label"
        ),
        "expected_test_errors": {
            "false_non_matches": 22,
            "false_matches": 10,
            "total": 32,
        },
        "qa_categories": [
            "probable_wrong_face_selection",
            "multi_face_but_target_selected",
            "alignment_issue",
            "extreme_pose_or_occlusion",
            "low_image_quality",
            "recognition_like_error",
            "unclear",
        ],
        "interpretation_rule": (
            "Automatic flags identify evidence "
            "requiring review; they do not prove "
            "the causal source of an error."
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    fn_count = int(
        (
            audit[
                "outcome"
            ]
            == "FN"
        ).sum()
    )

    fp_count = int(
        (
            audit[
                "outcome"
            ]
            == "FP"
        ).sum()
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "REFERENCE A ERROR AUDIT"
    )

    print(
        "=" * 72
    )

    print(
        f"Test errors: {len(audit)}"
    )

    print(
        f"False non-matches: {fn_count}"
    )

    print(
        f"False matches:     {fp_count}"
    )

    print(
        "\nAutomatic preprocessing signals:"
    )

    print(
        "  Multi-face signal: "
        f"{int(audit['has_multi_face_signal'].sum())}"
    )

    print(
        "  Alignment signal:  "
        f"{int(audit['has_alignment_signal'].sum())}"
    )

    print(
        "  Detection signal:  "
        f"{int(audit['has_detection_signal'].sum())}"
    )

    print(
        "\nSummary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        AUDIT_PATH
    )

    print(
        REVIEW_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        METADATA_PATH
    )

    print(
        "\nImportant:"
    )

    print(
        "Automatic flags are QA signals only. "
        "Manual review is required before assigning "
        "a causal error category."
    )


if __name__ == "__main__":
    main()
    