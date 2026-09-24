from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="metadata/near_duplicate_review.csv",
    )

    parser.add_argument(
        "--strong-output",
        default="metadata/near_duplicate_review_strong.csv",
    )

    parser.add_argument(
        "--weak-output",
        default="metadata/near_duplicate_review_weak_sample.csv",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help=(
            "Number of unresolved weak crop-only pairs "
            "to sample for human review."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=26041,
        help=(
            "Deterministic sampling seed. "
            "Record this value in the amendment."
        ),
    )

    return parser.parse_args()


def canonical_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def main():
    args = parse_args()

    input_path = Path(args.input)
    strong_path = Path(args.strong_output)
    weak_path = Path(args.weak_output)

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    df = pd.read_csv(
        input_path
    )

    required = {
        "original_path_a",
        "original_path_b",
        "dataset_a",
        "dataset_b",
        "cross_dataset",
        "phash_distance",
        "crop_matching_regions",
        "crop_distance_sum",
        "candidate_method",
        "review_status",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    # Work only with within-dataset pairs. Cross-dataset review is separate.
    cross_mask = (
        df["cross_dataset"]
        .map(canonical_bool)
    )

    within = df[
        ~cross_mask
    ].copy()

    # Only unresolved pairs need new review work. Previously reviewed
    # decisions stay in the original near_duplicate_review.csv.
    unresolved = within[
        within["review_status"]
        .fillna("candidate")
        .astype(str)
        .eq("candidate")
    ].copy()

    unresolved[
        "crop_matching_regions"
    ] = pd.to_numeric(
        unresolved[
            "crop_matching_regions"
        ],
        errors="coerce",
    )

    # Strong-evidence stratum, reviewed exhaustively: phash candidates,
    # phash+crop candidates, and crop-only candidates with 3+ matching
    # regions. No pair is automatically confirmed or rejected.
    strong_mask = (
        unresolved[
            "candidate_method"
        ].isin(
            [
                "phash",
                "phash+crop",
            ]
        )
        |
        (
            unresolved[
                "candidate_method"
            ].eq("crop")
            &
            unresolved[
                "crop_matching_regions"
            ].ge(3)
        )
    )

    strong = unresolved[
        strong_mask
    ].copy()

    strong[
        "review_stratum"
    ] = "strong_exhaustive"

    # Sort strongest-looking candidates first.
    method_rank = {
        "phash+crop": 0,
        "phash": 1,
        "crop": 2,
    }

    strong[
        "_method_rank"
    ] = (
        strong[
            "candidate_method"
        ]
        .map(method_rank)
        .fillna(9)
    )

    strong[
        "_phash_sort"
    ] = pd.to_numeric(
        strong[
            "phash_distance"
        ],
        errors="coerce",
    ).fillna(999999)

    strong[
        "_crop_sum_sort"
    ] = pd.to_numeric(
        strong[
            "crop_distance_sum"
        ],
        errors="coerce",
    ).fillna(999999)

    strong = strong.sort_values(
        by=[
            "_method_rank",
            "_phash_sort",
            "crop_matching_regions",
            "_crop_sum_sort",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
        kind="stable",
    )

    strong = strong.drop(
        columns=[
            "_method_rank",
            "_phash_sort",
            "_crop_sum_sort",
        ]
    )

    # Weak-evidence stratum: crop-only, exactly 2 matching regions.
    # Not automatically labelled false_alarm, sampled deterministically
    # for human review instead.
    weak_population = unresolved[
        (
            unresolved[
                "candidate_method"
            ].eq("crop")
        )
        &
        (
            unresolved[
                "crop_matching_regions"
            ].eq(2)
        )
    ].copy()

    sample_n = min(
        args.sample_size,
        len(weak_population),
    )

    if sample_n > 0:
        weak_sample = (
            weak_population.sample(
                n=sample_n,
                random_state=args.seed,
                replace=False,
            )
            .copy()
        )
    else:
        weak_sample = (
            weak_population.copy()
        )

    weak_sample[
        "review_stratum"
    ] = "weak_random_sample"

    weak_sample[
        "sampling_seed"
    ] = args.seed

    weak_sample[
        "weak_population_size"
    ] = len(
        weak_population
    )

    weak_sample[
        "requested_sample_size"
    ] = args.sample_size

    # Save
    strong_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weak_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    strong.to_csv(
        strong_path,
        index=False,
    )

    weak_sample.to_csv(
        weak_path,
        index=False,
    )

    # Audit summary
    print(
        "Total pair rows:",
        len(df),
    )

    print(
        "Within-dataset pairs:",
        len(within),
    )

    print(
        "Already reviewed within-dataset pairs:",
        len(within) - len(unresolved),
    )

    print(
        "Unresolved within-dataset pairs:",
        len(unresolved),
    )

    print()

    print(
        "Strong unresolved pairs "
        "(exhaustive review):",
        len(strong),
    )

    print(
        "Weak crop-only, 2-region "
        "population:",
        len(weak_population),
    )

    print(
        "Weak random sample:",
        len(weak_sample),
    )

    print(
        "Sampling seed:",
        args.seed,
    )

    print()

    accounted = (
        len(strong)
        + len(weak_population)
    )

    print(
        "Unresolved pairs classified "
        "into strong/weak strata:",
        accounted,
    )

    print(
        "Unclassified unresolved pairs:",
        len(unresolved) - accounted,
    )

    print()

    print(
        "Saved strong review file:",
        strong_path,
    )

    print(
        "Saved weak sample file:",
        weak_path,
    )


if __name__ == "__main__":
    main()
