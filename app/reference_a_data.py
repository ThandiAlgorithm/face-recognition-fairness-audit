from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

REPORT_PACK = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "reference_a"
    / "report_pack"
)

LOCAL_PAIR_SCORES = (
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


def load_csv(
    name: str,
) -> pd.DataFrame:

    path = (
        REPORT_PACK
        / name
    )

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        low_memory=False,
    )


def load_summary() -> dict:

    path = (
        REPORT_PACK
        / "reference_a_summary.json"
    )

    if not path.exists():
        return {}

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_verification_metrics():
    return load_csv(
        "verification_metrics.csv"
    )


def load_score_distributions():
    return load_csv(
        "score_distributions.csv"
    )


def load_safe_pair_scores():
    return load_csv(
        "pair_scores_sanitized.csv"
    )


def load_roc():
    return load_csv(
        "roc_test.csv"
    )


def load_confusion():
    return load_csv(
        "confusion_matrix_percent.csv"
    )


def load_subgroup_fairness():
    return load_csv(
        "subgroup_fairness.csv"
    )


def load_pair_coverage():
    return load_csv(
        "pair_demographic_coverage.csv"
    )


def load_image_coverage():
    return load_csv(
        "image_demographic_coverage.csv"
    )


def load_disparities():
    return load_csv(
        "fairness_disparity_summary.csv"
    )


def load_bootstrap_overall():
    return load_csv(
        "bootstrap_overall.csv"
    )


def load_bootstrap_subgroups():
    return load_csv(
        "bootstrap_subgroups.csv"
    )


def load_preprocessing_comparison():
    return load_csv(
        "preprocessing_comparison.csv"
    )


def load_qualitative_examples():
    return load_csv(
        "qualitative_examples.csv"
    )


def load_error_cases():
    return load_csv(
        "error_cases_sanitized.csv"
    )


def load_error_summary():
    return load_csv(
        "error_summary.csv"
    )


def load_local_pair_scores():

    if not LOCAL_PAIR_SCORES.exists():
        return pd.DataFrame()

    return pd.read_csv(
        LOCAL_PAIR_SCORES,
        low_memory=False,
    )


def canonical_lfw_path(
    value,
) -> str:

    path = (
        str(value)
        .strip()
        .replace("\\", "/")
    )

    lower = path.lower()

    marker = "/data/lfw/"

    if marker in lower:

        index = lower.index(
            marker
        )

        path = path[
            index
            + len("/data/"):
        ]

    else:

        parts = path.split("/")

        for i, part in enumerate(
            parts
        ):
            if part.lower() == "lfw":

                path = "/".join(
                    parts[i:]
                )

                break

    return path.lower()


def load_alignment_lookup() -> dict:

    if not MASTER_PATH.exists():
        return {}

    master = pd.read_parquet(
        MASTER_PATH
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

    lookup = {}

    for row in lfw.itertuples():

        if pd.isna(
            row.aligned_path
        ):
            continue

        aligned = Path(
            str(
                row.aligned_path
            )
        )

        if not aligned.is_absolute():

            aligned = (
                PROJECT_ROOT
                / aligned
            )

        lookup[
            row.path_key
        ] = aligned

    return lookup
