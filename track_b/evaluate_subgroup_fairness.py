from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCORES = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "arcface_reference"
    / "verification"
    / "arcface_view1_pair_scores.csv"
)

DERIVED_MASTER = (
    PROJECT_ROOT
    / "metadata"
    / "master_annotations_lfw_demographics_derived.parquet"
)

UNKNOWN_VALUES = {
    "",
    "unknown",
    "unavailable",
    "none",
    "nan",
    "na",
    "n/a",
}


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LFW verification fairness by gender, "
            "derived race, and race x gender."
        )
    )

    parser.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_SCORES,
        help="Verification pair-score CSV.",
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="R0_ArcFace",
        help="Model identifier.",
    )

    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="test",
        help=(
            "Pair split to evaluate. "
            "Official reporting should use 'test'."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional custom output directory.",
    )

    return parser.parse_args()


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
            index + len("/data/"):
        ]

    else:
        parts = path.split("/")

        for i, part in enumerate(parts):
            if part.lower() == "lfw":
                path = "/".join(parts[i:])
                break

    return path.lower()


def clean_group(value) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()

    if (
        not text
        or text.lower() in UNKNOWN_VALUES
    ):
        return None

    return text


def combine_intersection(
    race,
    gender,
) -> str | None:
    race = clean_group(race)
    gender = clean_group(gender)

    if race is None or gender is None:
        return None

    return f"{race} × {gender}"


def safe_divide(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return np.nan

    return float(
        numerator / denominator
    )


# ============================================================
# LOAD DEMOGRAPHICS
# ============================================================

def load_demographic_lookup() -> pd.DataFrame:
    if not DERIVED_MASTER.exists():
        raise FileNotFoundError(
            f"Missing demographic master:\n{DERIVED_MASTER}"
        )

    master = pd.read_parquet(
        DERIVED_MASTER
    )

    required = {
        "dataset",
        "original_path",
        "race",
        "gender",
    }

    missing = required - set(
        master.columns
    )

    if missing:
        raise KeyError(
            "Missing columns: "
            + ", ".join(sorted(missing))
        )

    lfw = master[
        master["dataset"]
        .astype(str)
        .str.lower()
        .eq("lfw")
    ].copy()

    lfw["path_key"] = (
        lfw["original_path"]
        .map(canonical_lfw_path)
    )

    lfw["race_clean"] = (
        lfw["race"]
        .map(clean_group)
    )

    lfw["gender_clean"] = (
        lfw["gender"]
        .map(clean_group)
    )

    lfw["intersection_clean"] = [
        combine_intersection(race, gender)
        for race, gender in zip(
            lfw["race_clean"],
            lfw["gender_clean"],
        )
    ]

    if lfw["path_key"].duplicated().any():
        raise RuntimeError(
            "Duplicate canonical LFW paths "
            "found in demographic master."
        )

    return lfw[
        [
            "path_key",
            "race_clean",
            "gender_clean",
            "intersection_clean",
        ]
    ].copy()


# ============================================================
# LOAD SCORES
# ============================================================

def load_pair_scores(
    path: Path,
    split: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing score file:\n{path}"
        )

    pairs = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "image1_path",
        "image2_path",
        "is_genuine",
        "predicted_genuine",
        "split",
    }

    missing = required - set(
        pairs.columns
    )

    if missing:
        raise KeyError(
            "Score file missing columns: "
            + ", ".join(sorted(missing))
        )

    if split != "all":
        pairs = pairs[
            pairs["split"]
            .astype(str)
            .str.lower()
            .eq(split)
        ].copy()

    pairs["image1_key"] = (
        pairs["image1_path"]
        .map(canonical_lfw_path)
    )

    pairs["image2_key"] = (
        pairs["image2_path"]
        .map(canonical_lfw_path)
    )

    return pairs.reset_index(
        drop=True
    )


# ============================================================
# ATTACH DEMOGRAPHICS
# ============================================================

def attach_demographics(
    pairs: pd.DataFrame,
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    indexed = lookup.set_index(
        "path_key"
    )

    out = pairs.copy()

    for side in [1, 2]:
        key = f"image{side}_key"

        out[f"race{side}"] = (
            out[key].map(
                indexed["race_clean"]
            )
        )

        out[f"gender{side}"] = (
            out[key].map(
                indexed["gender_clean"]
            )
        )

        out[f"intersection{side}"] = (
            out[key].map(
                indexed[
                    "intersection_clean"
                ]
            )
        )

    return out


# ============================================================
# COVERAGE
# ============================================================

def image_level_coverage(
    lookup: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    total = len(lookup)

    for column, name in [
        ("gender_clean", "gender"),
        ("race_clean", "race"),
        (
            "intersection_clean",
            "intersection",
        ),
    ]:
        known = int(
            lookup[column]
            .notna()
            .sum()
        )

        rows.append(
            {
                "attribute": name,
                "n_images": total,
                "known": known,
                "unknown_or_missing": (
                    total - known
                ),
                "coverage_pct": (
                    known / total
                ),
            }
        )

    return pd.DataFrame(rows)


def pair_coverage(
    frame: pd.DataFrame,
    attribute: str,
) -> dict:
    col1 = f"{attribute}1"
    col2 = f"{attribute}2"

    both_known = (
        frame[col1].notna()
        & frame[col2].notna()
    )

    same = (
        both_known
        & frame[col1].eq(
            frame[col2]
        )
    )

    different = (
        both_known
        & ~frame[col1].eq(
            frame[col2]
        )
    )

    genuine = frame[
        "is_genuine"
    ].astype(bool)

    impostor = ~genuine

    return {
        "attribute": attribute,
        "total_pairs": len(frame),
        "both_labels_known": int(
            both_known.sum()
        ),
        "both_labels_known_pct": float(
            both_known.mean()
        ),
        "any_unknown_or_missing": int(
            (~both_known).sum()
        ),
        "genuine_same_known_group": int(
            (
                genuine
                & same
            ).sum()
        ),
        "genuine_conflicting_known_labels": int(
            (
                genuine
                & different
            ).sum()
        ),
        "impostor_within_group": int(
            (
                impostor
                & same
            ).sum()
        ),
        "impostor_cross_group": int(
            (
                impostor
                & different
            ).sum()
        ),
    }


# ============================================================
# SUBGROUP METRICS
# ============================================================

def evaluate_attribute(
    frame: pd.DataFrame,
    attribute: str,
    model_id: str,
    split_name: str,
) -> pd.DataFrame:
    col1 = f"{attribute}1"
    col2 = f"{attribute}2"

    eligible = frame[
        frame[col1].notna()
        & frame[col2].notna()
        & frame[col1].eq(
            frame[col2]
        )
    ].copy()

    rows = []

    for group_name, group in eligible.groupby(
        col1
    ):
        actual = group[
            "is_genuine"
        ].astype(bool)

        predicted = group[
            "predicted_genuine"
        ].astype(bool)

        tp = int(
            (
                actual
                & predicted
            ).sum()
        )

        fn = int(
            (
                actual
                & ~predicted
            ).sum()
        )

        fp = int(
            (
                ~actual
                & predicted
            ).sum()
        )

        tn = int(
            (
                ~actual
                & ~predicted
            ).sum()
        )

        positives = tp + fn
        negatives = tn + fp
        total = positives + negatives

        tpr = safe_divide(
            tp,
            positives,
        )

        fnmr = safe_divide(
            fn,
            positives,
        )

        fmr = safe_divide(
            fp,
            negatives,
        )

        tnr = safe_divide(
            tn,
            negatives,
        )

        if (
            not np.isnan(tpr)
            and not np.isnan(tnr)
        ):
            balanced_accuracy = (
                tpr + tnr
            ) / 2.0
        else:
            balanced_accuracy = np.nan

        rows.append(
            {
                "model_id": model_id,
                "split": split_name,
                "attribute": attribute,
                "group": group_name,
                "n_pairs": total,
                "n_genuine": positives,
                "n_impostor": negatives,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "TPR_TAR": tpr,
                "FMR_FPR": fmr,
                "FNMR_FNR": fnmr,
                "specificity_TNR": tnr,
                "accuracy": safe_divide(
                    tp + tn,
                    total,
                ),
                "balanced_accuracy": (
                    balanced_accuracy
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CROSS-GROUP IMPOSTORS
# ============================================================

def cross_group_impostor_fmr(
    frame: pd.DataFrame,
    attribute: str,
    model_id: str,
    split_name: str,
) -> pd.DataFrame:
    col1 = f"{attribute}1"
    col2 = f"{attribute}2"

    cross = frame[
        (~frame["is_genuine"].astype(bool))
        & frame[col1].notna()
        & frame[col2].notna()
        & ~frame[col1].eq(
            frame[col2]
        )
    ].copy()

    if cross.empty:
        return pd.DataFrame()

    cross["group_pair"] = cross.apply(
        lambda row: " ↔ ".join(
            sorted(
                [
                    str(row[col1]),
                    str(row[col2]),
                ]
            )
        ),
        axis=1,
    )

    rows = []

    for name, group in cross.groupby(
        "group_pair"
    ):
        n = len(group)

        false_matches = int(
            group[
                "predicted_genuine"
            ]
            .astype(bool)
            .sum()
        )

        rows.append(
            {
                "model_id": model_id,
                "split": split_name,
                "attribute": attribute,
                "group_pair": name,
                "n_impostor_pairs": n,
                "false_matches": (
                    false_matches
                ),
                "FMR_FPR": safe_divide(
                    false_matches,
                    n,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# DISPARITY SUMMARY
# ============================================================

def disparity_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if metrics.empty:
        return pd.DataFrame()

    for (
        model_id,
        split_name,
        attribute,
    ), group in metrics.groupby(
        [
            "model_id",
            "split",
            "attribute",
        ]
    ):
        row = {
            "model_id": model_id,
            "split": split_name,
            "attribute": attribute,
            "n_groups": len(group),
        }

        for metric in [
            "TPR_TAR",
            "FMR_FPR",
            "FNMR_FNR",
        ]:
            values = (
                group[metric]
                .dropna()
            )

            if values.empty:
                min_value = np.nan
                max_value = np.nan
                gap = np.nan

            else:
                min_value = float(
                    values.min()
                )

                max_value = float(
                    values.max()
                )

                gap = (
                    max_value
                    - min_value
                )

            row[
                f"{metric}_min"
            ] = min_value

            row[
                f"{metric}_max"
            ] = max_value

            row[
                f"{metric}_gap"
            ] = gap

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    scores_path = (
        args.scores
        if args.scores.is_absolute()
        else PROJECT_ROOT
        / args.scores
    )

    split_name = (
        args.split.lower()
    )

    if args.output_dir is None:
        output_dir = (
            scores_path.parent
            / "fairness"
            / f"view1_{split_name}"
        )
    else:
        output_dir = (
            args.output_dir
            if args.output_dir.is_absolute()
            else PROJECT_ROOT
            / args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    lookup = (
        load_demographic_lookup()
    )

    pairs = load_pair_scores(
        scores_path,
        split_name,
    )

    pairs = attach_demographics(
        pairs,
        lookup,
    )

    # Verify every image path joined to
    # an actual LFW master row.
    lookup_keys = set(
        lookup["path_key"]
    )

    unresolved1 = int(
        (
            ~pairs["image1_key"]
            .isin(lookup_keys)
        ).sum()
    )

    unresolved2 = int(
        (
            ~pairs["image2_key"]
            .isin(lookup_keys)
        ).sum()
    )

    if unresolved1 or unresolved2:
        raise RuntimeError(
            "Unresolved LFW paths: "
            f"image1={unresolved1}, "
            f"image2={unresolved2}"
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TRACK B DEMOGRAPHIC FAIRNESS EVALUATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Model: {args.model_id}"
    )

    print(
        f"Evaluation split: {split_name}"
    )

    print(
        f"Pairs loaded: {len(pairs):,}"
    )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    image_coverage = (
        image_level_coverage(
            lookup
        )
    )

    coverage = pd.DataFrame(
        [
            pair_coverage(
                pairs,
                "gender",
            ),
            pair_coverage(
                pairs,
                "race",
            ),
            pair_coverage(
                pairs,
                "intersection",
            ),
        ]
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metric_frames = []

    cross_frames = []

    for attribute in [
        "gender",
        "race",
        "intersection",
    ]:
        result = evaluate_attribute(
            pairs,
            attribute,
            args.model_id,
            split_name,
        )

        if not result.empty:
            metric_frames.append(
                result
            )

        cross = (
            cross_group_impostor_fmr(
                pairs,
                attribute,
                args.model_id,
                split_name,
            )
        )

        if not cross.empty:
            cross_frames.append(
                cross
            )

    subgroup_metrics = (
        pd.concat(
            metric_frames,
            ignore_index=True,
        )
        if metric_frames
        else pd.DataFrame()
    )

    cross_group = (
        pd.concat(
            cross_frames,
            ignore_index=True,
        )
        if cross_frames
        else pd.DataFrame()
    )

    disparities = (
        disparity_summary(
            subgroup_metrics
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    pairs.to_csv(
        output_dir
        / "pair_scores_with_demographics.csv",
        index=False,
    )

    image_coverage.to_csv(
        output_dir
        / "image_demographic_coverage.csv",
        index=False,
    )

    coverage.to_csv(
        output_dir
        / "pair_demographic_coverage.csv",
        index=False,
    )

    subgroup_metrics.to_csv(
        output_dir
        / "subgroup_fairness.csv",
        index=False,
    )

    cross_group.to_csv(
        output_dir
        / "cross_group_impostor_fmr.csv",
        index=False,
    )

    disparities.to_csv(
        output_dir
        / "fairness_disparity_summary.csv",
        index=False,
    )

    metadata = {
        "model_id": args.model_id,
        "split": split_name,
        "official_reporting_split": (
            split_name == "test"
        ),
        "demographic_source": (
            "externally derived LFW annotations"
        ),
        "race_is_ground_truth": False,
        "primary_fmr_definition": (
            "within-group impostor pairs"
        ),
        "cross_group_impostors": (
            "reported separately"
        ),
        "genuine_group_rule": (
            "both images have same known label"
        ),
        "unknown_policy": (
            "excluded from known-group metric "
            "and retained in coverage reporting"
        ),
        "view2_used": False,
    }

    (
        output_dir
        / "fairness_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        "\nImage-level demographic coverage:"
    )

    print(
        image_coverage.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print(
        "\nPair-level coverage:"
    )

    print(
        coverage.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print(
        "\nSubgroup metric rows:",
        len(subgroup_metrics),
    )

    print(
        "Cross-group impostor rows:",
        len(cross_group),
    )

    print(
        "\nOutput directory:"
    )

    print(
        output_dir
    )

    if args.model_id == "R0_ArcFace":
        print(
            "\nNOTE:"
        )

        print(
            "Current ArcFace raw-input results remain "
            "diagnostic only until aligned inputs are used."
        )


if __name__ == "__main__":
    main()
    