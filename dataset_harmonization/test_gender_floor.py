from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "mvp_volume_aware"
    / "mvp_training_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "mvp_volume_aware"
    / "gender_floor_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


GROUPS = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino",
]

MIN_SHARE = 0.35


def max_size_with_gender_floor(
    female_n: int,
    male_n: int,
    min_share: float,
) -> int:
    """
    Largest subset possible without oversampling such that
    female and male each represent at least min_share.
    """

    total = female_n + male_n

    if total == 0:
        return 0

    female_share = female_n / total
    male_share = male_n / total

    if (
        female_share >= min_share
        and male_share >= min_share
    ):
        return total

    minority = min(
        female_n,
        male_n,
    )

    return int(
        minority / min_share
    )


def main():

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    df["gender_clean"] = (
        df["gender_label"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    rows = []

    for group in GROUPS:

        g = df[
            df[
                "candidate_demographic_label"
            ]
            == group
        ]

        female = int(
            (
                g["gender_clean"]
                == "female"
            ).sum()
        )

        male = int(
            (
                g["gender_clean"]
                == "male"
            ).sum()
        )

        known = female + male

        candidate_n = (
            max_size_with_gender_floor(
                female,
                male,
                MIN_SHARE,
            )
        )

        removed = (
            known - candidate_n
        )

        retention = (
            candidate_n
            / known
            * 100
            if known
            else 0
        )

        rows.append(
            {
                "demographic_group": group,
                "current_known_gender_n": known,
                "female_n": female,
                "male_n": male,
                "female_pct": (
                    female / known * 100
                    if known
                    else 0
                ),
                "male_pct": (
                    male / known * 100
                    if known
                    else 0
                ),
                "max_n_at_35_65": candidate_n,
                "images_removed": removed,
                "retention_pct": retention,
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        OUTPUT_DIR
        / "gender_floor_feasibility.csv",
        index=False,
    )

    print("=" * 72)
    print("35/65 GENDER FLOOR FEASIBILITY")
    print("=" * 72)

    print(
        result.to_string(
            index=False,
            formatters={
                "female_pct":
                    lambda x: f"{x:.1f}%",
                "male_pct":
                    lambda x: f"{x:.1f}%",
                "retention_pct":
                    lambda x: f"{x:.1f}%",
            },
        )
    )

    print("\nCurrent total:")
    print(
        f"{result['current_known_gender_n'].sum():,}"
    )

    print("\nMaximum total under 35/65 floor:")
    print(
        f"{result['max_n_at_35_65'].sum():,}"
    )

    print("\nImages removed:")
    print(
        f"{result['images_removed'].sum():,}"
    )

    print("\nOutput:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
    