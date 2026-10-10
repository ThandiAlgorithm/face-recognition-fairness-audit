from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "Harmonized_Face_Dataset_v1_MATERIALIZED"
)

METADATA_DIR = DATASET_DIR / "metadata"

RECORDS_FILE = (
    METADATA_DIR / "records.csv"
)

BACKUP_FILE = (
    METADATA_DIR
    / "records_before_filename_enrichment.csv"
)

LABELED_RECORDS_FILE = (
    METADATA_DIR
    / "records_labeled.csv"
)

QA_DIR = DATASET_DIR / "qa"

CONTACT_DIR = QA_DIR / "contact_sheets"

QA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONTACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SOURCE_ABBREVIATIONS = {
    "fairface": "FF",
    "utkface": "UTK",
    "lfw": "LFW",
}


CLASS_TO_INDEX = {
    "White": 0,
    "Black": 1,
    "Indian": 2,
    "East Asian": 3,
    "Southeast Asian": 4,
    "Middle Eastern": 5,
    "Latino": 6,
}


EXPECTED_SOURCE_MAPPING = {
    "White": "White",
    "Black": "Black",
    "Indian": "Indian",
    "East Asian": "East Asian",
    "Southeast Asian": "Southeast Asian",
    "Middle Eastern": "Middle Eastern",
    "Latino_Hispanic": "Latino",
}


def safe_text(value) -> str:

    if pd.isna(value):
        return ""

    return str(value).strip()


def filename_safe(value: str) -> str:

    value = safe_text(value)

    if not value:
        return "NA"

    value = value.replace(" ", "-")

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "",
        value,
    )

    return value or "NA"


def gender_code(value) -> str:

    value = safe_text(
        value
    ).lower()

    if value == "female":
        return "F"

    if value == "male":
        return "M"

    return "U"


def age_code(row) -> str:

    if "age_group" in row.index:

        group = safe_text(
            row["age_group"]
        )

        if (
            group
            and group.lower()
            not in {
                "unknown",
                "missing",
                "nan",
                "none",
            }
        ):
            return filename_safe(
                group
            )

    if "age_value" in row.index:

        value = row[
            "age_value"
        ]

        if pd.notna(value):

            try:
                return (
                    "age"
                    + str(
                        int(
                            float(value)
                        )
                    )
                )

            except Exception:
                pass

    return "ageNA"


def label_filename(
    row,
    current_path: Path,
) -> str:

    label = filename_safe(
        row[
            "candidate_demographic_label"
        ]
    )

    source = SOURCE_ABBREVIATIONS.get(
        safe_text(
            row[
                "source_dataset"
            ]
        ).lower(),
        "SRC",
    )

    gender = gender_code(
        row.get(
            "gender_label"
        )
    )

    age = age_code(
        row
    )

    record_id = filename_safe(
        row[
            "record_id"
        ]
    )

    extension = (
        current_path
        .suffix
        .lower()
    )

    return (
        f"{label}"
        f"__{source}"
        f"__{gender}"
        f"__{age}"
        f"__{record_id}"
        f"{extension}"
    )


def source_mapping_status(
    row,
) -> str:

    source_label = safe_text(
        row.get(
            "race_label"
        )
    )

    harmonized = safe_text(
        row[
            "candidate_demographic_label"
        ]
    )

    if not source_label:
        return "NOT_CHECKABLE"

    expected = (
        EXPECTED_SOURCE_MAPPING.get(
            source_label
        )
    )

    if expected is None:

        # Some source labels may already use
        # the harmonised wording.
        if source_label == harmonized:
            return "CONSISTENT"

        return "NOT_CHECKABLE"

    if expected == harmonized:
        return "CONSISTENT"

    return "MAPPING_MISMATCH"


def rename_images(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    if (
        "original_materialized_relative_path"
        not in result.columns
    ):
        result[
            "original_materialized_relative_path"
        ] = result[
            "materialized_relative_path"
        ]

    new_paths = []

    total = len(
        result
    )

    for i, (
        index,
        row,
    ) in enumerate(
        result.iterrows(),
        start=1,
    ):

        relative = Path(
            row[
                "materialized_relative_path"
            ]
        )

        old_path = (
            DATASET_DIR
            / relative
        )

        if not old_path.exists():

            raise FileNotFoundError(
                f"Missing image:\n"
                f"{old_path}"
            )

        new_name = label_filename(
            row,
            old_path,
        )

        new_path = (
            old_path.parent
            / new_name
        )

        if (
            old_path
            != new_path
        ):

            if (
                new_path.exists()
                and new_path
                != old_path
            ):
                raise FileExistsError(
                    f"Filename collision:\n"
                    f"{new_path}"
                )

            old_path.rename(
                new_path
            )

        new_relative = (
            new_path.relative_to(
                DATASET_DIR
            )
        )

        new_paths.append(
            str(
                new_relative
            ).replace(
                "\\",
                "/",
            )
        )

        if (
            i % 5000 == 0
            or i == total
        ):
            print(
                f"  renamed "
                f"{i:,}/"
                f"{total:,}"
            )

    result[
        "materialized_relative_path"
    ] = new_paths

    result[
        "class_index"
    ] = result[
        "candidate_demographic_label"
    ].map(
        CLASS_TO_INDEX
    )

    result[
        "source_abbreviation"
    ] = (
        result[
            "source_dataset"
        ]
        .astype(str)
        .str.lower()
        .map(
            SOURCE_ABBREVIATIONS
        )
        .fillna(
            "SRC"
        )
    )

    result[
        "source_mapping_status"
    ] = result.apply(
        source_mapping_status,
        axis=1,
    )

    return result


def create_mapping_audit(
    df: pd.DataFrame,
):

    columns = [
        column
        for column in [
            "source_dataset",
            "race_label",
            "candidate_demographic_label",
            "source_mapping_status",
        ]
        if column in df.columns
    ]

    audit = (
        df.groupby(
            columns,
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False,
        )
    )

    audit.to_csv(
        QA_DIR
        / "source_label_mapping_audit.csv",
        index=False,
    )

    return audit


def create_source_counts(
    df: pd.DataFrame,
):

    table = pd.crosstab(
        df[
            "candidate_demographic_label"
        ],
        df[
            "source_dataset"
        ],
    )

    table.to_csv(
        QA_DIR
        / "label_by_source_counts.csv"
    )


def create_gender_counts(
    df: pd.DataFrame,
):

    if "gender_label" not in df.columns:
        return

    table = pd.crosstab(
        df[
            "candidate_demographic_label"
        ],
        df[
            "gender_label"
        ],
    )

    table.to_csv(
        QA_DIR
        / "label_by_gender_counts.csv"
    )


def font(size):

    try:
        return ImageFont.truetype(
            "arial.ttf",
            size,
        )

    except Exception:
        return ImageFont.load_default()


def make_contact_sheet(
    df: pd.DataFrame,
    label: str,
    n=36,
):

    subset = df[
        df[
            "candidate_demographic_label"
        ]
        == label
    ]

    if subset.empty:
        return

    n = min(
        n,
        len(subset),
    )

    sample = subset.sample(
        n=n,
        random_state=2026,
    )

    columns = 6

    tile_width = 190
    image_height = 155
    caption_height = 55

    tile_height = (
        image_height
        + caption_height
    )

    rows = (
        n
        + columns
        - 1
    ) // columns

    header = 70

    canvas = Image.new(
        "RGB",
        (
            columns
            * tile_width,
            header
            + rows
            * tile_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (15, 12),
        (
            f"QA sample — "
            f"{label}"
        ),
        fill="black",
        font=font(24),
    )

    draw.text(
        (15, 42),
        (
            "For provenance / image-quality review. "
            "Do not infer or overwrite race from appearance."
        ),
        fill="black",
        font=font(12),
    )

    for position, (
        _,
        row,
    ) in enumerate(
        sample.iterrows()
    ):

        col = (
            position
            % columns
        )

        r = (
            position
            // columns
        )

        x = (
            col
            * tile_width
        )

        y = (
            header
            + r
            * tile_height
        )

        path = (
            DATASET_DIR
            / Path(
                row[
                    "materialized_relative_path"
                ]
            )
        )

        try:

            with Image.open(
                path
            ) as image:

                image = (
                    image
                    .convert(
                        "RGB"
                    )
                )

                image.thumbnail(
                    (
                        tile_width - 10,
                        image_height - 10,
                    )
                )

                px = (
                    x
                    + (
                        tile_width
                        - image.width
                    )
                    // 2
                )

                py = (
                    y
                    + (
                        image_height
                        - image.height
                    )
                    // 2
                )

                canvas.paste(
                    image,
                    (
                        px,
                        py,
                    )
                )

        except Exception:

            draw.text(
                (
                    x + 5,
                    y + 5,
                ),
                "IMAGE ERROR",
                fill="black",
                font=font(12),
            )

        source = (
            SOURCE_ABBREVIATIONS.get(
                safe_text(
                    row[
                        "source_dataset"
                    ]
                ).lower(),
                "SRC",
            )
        )

        gender = gender_code(
            row.get(
                "gender_label"
            )
        )

        age = age_code(
            row
        )

        record_id = safe_text(
            row[
                "record_id"
            ]
        )[:8]

        caption = (
            f"{source} | "
            f"{gender} | "
            f"{age}\n"
            f"id:{record_id}"
        )

        draw.text(
            (
                x + 5,
                y
                + image_height
                + 2,
            ),
            caption,
            fill="black",
            font=font(11),
        )

    filename = (
        label
        .lower()
        .replace(
            " ",
            "_",
        )
        + "_qa.jpg"
    )

    canvas.save(
        CONTACT_DIR
        / filename,
        quality=92,
    )


def create_contact_sheets(
    df,
):

    for label in (
        CLASS_TO_INDEX
    ):

        print(
            f"Creating QA sheet: "
            f"{label}"
        )

        make_contact_sheet(
            df,
            label,
        )


def main():

    print(
        "=" * 72
    )

    print(
        "ENRICH MATERIALIZED DATASET LABELS"
    )

    print(
        "=" * 72
    )

    df = pd.read_csv(
        RECORDS_FILE,
        low_memory=False,
    )

    print(
        f"\nRecords: "
        f"{len(df):,}"
    )

    if not BACKUP_FILE.exists():

        shutil.copy2(
            RECORDS_FILE,
            BACKUP_FILE,
        )

        print(
            "\nMetadata backup created:"
        )

        print(
            BACKUP_FILE
        )

    print(
        "\nRenaming copied images..."
    )

    df = rename_images(
        df
    )

    df.to_csv(
        RECORDS_FILE,
        index=False,
    )

    df.to_csv(
        LABELED_RECORDS_FILE,
        index=False,
    )

    (
        METADATA_DIR
        / "class_to_index.json"
    ).write_text(
        json.dumps(
            CLASS_TO_INDEX,
            indent=2,
        ),
        encoding="utf-8",
    )

    audit = create_mapping_audit(
        df
    )

    create_source_counts(
        df
    )

    create_gender_counts(
        df
    )

    create_contact_sheets(
        df
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "SOURCE / HARMONIZED LABEL AUDIT"
    )

    print(
        "=" * 72
    )

    print(
        audit.to_string(
            index=False
        )
    )

    mismatches = int(
        (
            df[
                "source_mapping_status"
            ]
            == "MAPPING_MISMATCH"
        ).sum()
    )

    print(
        f"\nMapping mismatches: "
        f"{mismatches:,}"
    )

    print(
        "\nUpdated metadata:"
    )

    print(
        RECORDS_FILE
    )

    print(
        "\nQA contact sheets:"
    )

    print(
        CONTACT_DIR
    )

    print(
        "\nTraining image pixels "
        "were NOT modified."
    )


if __name__ == "__main__":
    main()
    