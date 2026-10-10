from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "final_mvp_candidate"
    / "splits"
    / "final_mvp_with_splits.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "final_mvp_candidate"
    / "frozen_training"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FROZEN_MANIFEST = (
    OUTPUT_DIR
    / "training_manifest_frozen.csv"
)

EXCLUDED_FILE = (
    OUTPUT_DIR
    / "excluded_missing_or_invalid_paths.csv"
)


EXPECTED_LABELS = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino",
]


def resolve_path(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    path = Path(text)

    if path.is_absolute():
        return path

    return ROOT / path


def check_paths(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    resolved = []
    exists = []

    for value in df["image_path"]:
        path = resolve_path(value)

        if path is None:
            resolved.append("")
            exists.append(False)
            continue

        resolved.append(
            str(path)
        )

        exists.append(
            path.exists()
            and path.is_file()
        )

    df["resolved_image_path"] = resolved
    df["image_file_exists"] = exists

    return df


def validate_structure(
    df: pd.DataFrame,
) -> None:

    duplicate_records = df.duplicated(
        subset=[
            "source_dataset",
            "image_id",
        ]
    ).sum()

    if duplicate_records:
        raise RuntimeError(
            f"Duplicate source/image records: "
            f"{duplicate_records}"
        )

    labels = set(
        df[
            "candidate_demographic_label"
        ].dropna().unique()
    )

    unexpected = (
        labels
        - set(EXPECTED_LABELS)
    )

    if unexpected:
        raise RuntimeError(
            f"Unexpected labels: "
            f"{sorted(unexpected)}"
        )

    invalid_splits = (
        set(
            df["split"]
            .dropna()
            .unique()
        )
        - {
            "train",
            "val",
            "test",
        }
    )

    if invalid_splits:
        raise RuntimeError(
            f"Unexpected split names: "
            f"{sorted(invalid_splits)}"
        )


def check_lfw_identity_leakage(
    df: pd.DataFrame,
) -> int:

    if "identity_id" not in df.columns:
        return 0

    lfw = df[
        (
            df["source_dataset"]
            .astype(str)
            .str.lower()
            == "lfw"
        )
        & df["identity_id"].notna()
    ]

    if lfw.empty:
        return 0

    counts = (
        lfw.groupby(
            "identity_id"
        )["split"]
        .nunique()
    )

    return int(
        (
            counts > 1
        ).sum()
    )


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:

    print("=" * 72)
    print("FREEZE FINAL TRAINING MANIFEST")
    print("=" * 72)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    print(
        f"\nInput records: {len(df):,}"
    )

    validate_structure(
        df
    )

    df = check_paths(
        df
    )

    invalid = df[
        ~df[
            "image_file_exists"
        ]
    ].copy()

    valid = df[
        df[
            "image_file_exists"
        ]
    ].copy()

    print(
        f"Valid image files: "
        f"{len(valid):,}"
    )

    print(
        f"Excluded records: "
        f"{len(invalid):,}"
    )

    invalid.to_csv(
        EXCLUDED_FILE,
        index=False,
    )

    # Keep the resolved absolute path for the
    # training loader.
    valid["image_path_original"] = (
        valid["image_path"]
    )

    valid["image_path"] = (
        valid[
            "resolved_image_path"
        ]
    )

    valid = valid.drop(
        columns=[
            "resolved_image_path",
            "image_file_exists",
        ]
    )

    leakage = (
        check_lfw_identity_leakage(
            valid
        )
    )

    if leakage:
        raise RuntimeError(
            "LFW identity leakage detected "
            f"after filtering: {leakage}"
        )

    valid.to_csv(
        FROZEN_MANIFEST,
        index=False,
    )

    manifest_hash = sha256_file(
        FROZEN_MANIFEST
    )

    split_summary = (
        valid["split"]
        .value_counts()
        .reindex(
            [
                "train",
                "val",
                "test",
            ]
        )
        .rename_axis("split")
        .reset_index(
            name="count"
        )
    )

    split_summary[
        "percentage"
    ] = (
        split_summary["count"]
        / len(valid)
        * 100
    )

    demographic_summary = (
        pd.crosstab(
            valid[
                "candidate_demographic_label"
            ],
            valid["split"],
        )
        .reindex(
            EXPECTED_LABELS,
            fill_value=0,
        )
    )

    split_summary.to_csv(
        OUTPUT_DIR
        / "frozen_split_summary.csv",
        index=False,
    )

    demographic_summary.to_csv(
        OUTPUT_DIR
        / "frozen_demographic_split_counts.csv"
    )

    metadata = {
        "input_manifest": str(INPUT),
        "frozen_manifest": str(
            FROZEN_MANIFEST
        ),
        "input_records": int(
            len(df)
        ),
        "frozen_records": int(
            len(valid)
        ),
        "excluded_records": int(
            len(invalid)
        ),
        "lfw_identity_leakage": int(
            leakage
        ),
        "sha256": manifest_hash,
    }

    (
        OUTPUT_DIR
        / "frozen_manifest_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n")
    print("=" * 72)
    print("FINAL SPLIT SUMMARY")
    print("=" * 72)

    print(
        split_summary.to_string(
            index=False,
            formatters={
                "percentage":
                    lambda x:
                    f"{x:.2f}%"
            },
        )
    )

    print("\n")
    print("=" * 72)
    print("DEMOGRAPHIC COUNTS")
    print("=" * 72)

    print(
        demographic_summary.to_string()
    )

    print("\nLFW identity leakage:")
    print(leakage)

    print("\nSHA256:")
    print(manifest_hash)

    print("\nFrozen manifest:")
    print(FROZEN_MANIFEST)

    print("\nExcluded records:")
    print(EXCLUDED_FILE)

    print(
        "\nNo source image was "
        "deleted, copied or modified."
    )


if __name__ == "__main__":
    main()
    