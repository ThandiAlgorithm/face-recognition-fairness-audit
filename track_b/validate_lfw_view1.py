from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "LFW"

# Your local LFW archive currently has:
# data/LFW/lfw/lfw/<person>/<person>_0001.jpg
LFW_IMAGE_ROOT = DATA_ROOT / "lfw" / "lfw"

PAIR_FILES = {
    ("train", "genuine"): DATA_ROOT / "matchpairsDevTrain.csv",
    ("train", "impostor"): DATA_ROOT / "mismatchpairsDevTrain.csv",
    ("test", "genuine"): DATA_ROOT / "matchpairsDevTest.csv",
    ("test", "impostor"): DATA_ROOT / "mismatchpairsDevTest.csv",
}

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "arcface_reference"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# IMAGE RESOLUTION
# ============================================================

def lfw_image_path(
    person: str,
    image_number: int,
) -> Path:
    """
    Convert an LFW identity + image number into the expected image path.

    Example:
        Aaron_Peirsol, 1
        ->
        Aaron_Peirsol/Aaron_Peirsol_0001.jpg
    """

    filename = (
        f"{person}_{int(image_number):04d}.jpg"
    )

    return (
        LFW_IMAGE_ROOT
        / person
        / filename
    )


# ============================================================
# PAIR LOADERS
# ============================================================

def load_genuine_pairs(
    path: Path,
    split: str,
) -> pd.DataFrame:

    df = pd.read_csv(
        path
    )

    required = {
        "name",
        "imagenum1",
        "imagenum2",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    out = pd.DataFrame(
        {
            "split": split,
            "pair_type": "genuine",
            "is_genuine": True,
            "person1": df["name"].astype(str),
            "image_number1": df["imagenum1"].astype(int),
            "person2": df["name"].astype(str),
            "image_number2": df["imagenum2"].astype(int),
        }
    )

    return out


def load_impostor_pairs(
    path: Path,
    split: str,
) -> pd.DataFrame:

    df = pd.read_csv(
        path
    )

    # pandas automatically renames the second duplicate "name"
    # column to "name.1".
    required = {
        "name",
        "imagenum1",
        "name.1",
        "imagenum2",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    out = pd.DataFrame(
        {
            "split": split,
            "pair_type": "impostor",
            "is_genuine": False,
            "person1": df["name"].astype(str),
            "image_number1": df["imagenum1"].astype(int),
            "person2": df["name.1"].astype(str),
            "image_number2": df["imagenum2"].astype(int),
        }
    )

    return out


# ============================================================
# VALIDATION
# ============================================================

def resolve_pairs(
    pairs: pd.DataFrame,
) -> pd.DataFrame:

    pairs = pairs.copy()

    pairs["image1_path"] = [
        str(
            lfw_image_path(
                person,
                number,
            )
        )
        for person, number
        in zip(
            pairs["person1"],
            pairs["image_number1"],
        )
    ]

    pairs["image2_path"] = [
        str(
            lfw_image_path(
                person,
                number,
            )
        )
        for person, number
        in zip(
            pairs["person2"],
            pairs["image_number2"],
        )
    ]

    pairs["image1_exists"] = [
        Path(path).exists()
        for path
        in pairs["image1_path"]
    ]

    pairs["image2_exists"] = [
        Path(path).exists()
        for path
        in pairs["image2_path"]
    ]

    pairs["pair_resolved"] = (
        pairs["image1_exists"]
        & pairs["image2_exists"]
    )

    return pairs


def print_summary(
    pairs: pd.DataFrame,
) -> None:

    print(
        "\n"
        + "=" * 68
    )

    print(
        "LFW VIEW 1 PAIR VALIDATION"
    )

    print(
        "=" * 68
    )

    summary = (
        pairs
        .groupby(
            [
                "split",
                "pair_type",
            ],
            dropna=False,
        )
        .agg(
            n_pairs=(
                "pair_type",
                "size",
            ),
            n_resolved=(
                "pair_resolved",
                "sum",
            ),
        )
        .reset_index()
    )

    summary[
        "n_unresolved"
    ] = (
        summary["n_pairs"]
        - summary["n_resolved"]
    )

    print(
        "\nPair counts:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nTotal pairs:",
        f"{len(pairs):,}",
    )

    print(
        "Resolved pairs:",
        f"{int(pairs['pair_resolved'].sum()):,}",
    )

    print(
        "Unresolved pairs:",
        f"{int((~pairs['pair_resolved']).sum()):,}",
    )

    unique_images = pd.unique(
        pd.concat(
            [
                pairs["image1_path"],
                pairs["image2_path"],
            ],
            ignore_index=True,
        )
    )

    print(
        "Unique images required:",
        f"{len(unique_images):,}",
    )

    genuine = pairs[
        pairs["is_genuine"]
    ]

    impostor = pairs[
        ~pairs["is_genuine"]
    ]

    bad_genuine = genuine[
        genuine["person1"]
        != genuine["person2"]
    ]

    bad_impostor = impostor[
        impostor["person1"]
        == impostor["person2"]
    ]

    print(
        "\nProtocol integrity:"
    )

    print(
        "Invalid genuine identity pairs:",
        len(bad_genuine),
    )

    print(
        "Invalid impostor identity pairs:",
        len(bad_impostor),
    )

    if (
        pairs["pair_resolved"].all()
        and len(bad_genuine) == 0
        and len(bad_impostor) == 0
    ):
        print(
            "\nPASS: View 1 pair definitions "
            "resolve correctly."
        )

    else:
        print(
            "\nFAIL: investigate unresolved "
            "or invalid pairs before recognition."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not LFW_IMAGE_ROOT.exists():
        raise FileNotFoundError(
            "LFW image root not found:\n"
            f"{LFW_IMAGE_ROOT}"
        )

    frames = []

    for (
        split,
        pair_type,
    ), path in PAIR_FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing pair file: {path}"
            )

        if pair_type == "genuine":

            frame = load_genuine_pairs(
                path,
                split,
            )

        else:

            frame = load_impostor_pairs(
                path,
                split,
            )

        frames.append(
            frame
        )

    pairs = pd.concat(
        frames,
        ignore_index=True,
    )

    pairs = resolve_pairs(
        pairs
    )

    # Deterministic pair ID for future joins.
    pairs.insert(
        0,
        "pair_id",
        [
            f"LFW-V1-{i:05d}"
            for i
            in range(
                1,
                len(pairs) + 1,
            )
        ],
    )

    output_path = (
        OUTPUT_ROOT
        / "lfw_view1_pairs_resolved.csv"
    )

    pairs.to_csv(
        output_path,
        index=False,
    )

    print_summary(
        pairs
    )

    print(
        "\nSaved:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()
    