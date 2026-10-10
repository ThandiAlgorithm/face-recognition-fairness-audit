from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "final_mvp_candidate"
    / "frozen_training"
    / "training_manifest_frozen.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "harmonized_dataset"
    / "FINAL_HARMONIZED_DATASET"
)

LABEL_DIR = OUTPUT_DIR / "labels"
PREVIEW_DIR = OUTPUT_DIR / "previews"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DATASET_NAME = (
    "Harmonized Diversity-Aware Face Dataset"
)

VERSION = "1.0"

SEED = 2026

LABEL_ORDER = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino",
]


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def stable_record_id(row):
    token = (
        f"{row.get('source_dataset', '')}|"
        f"{row.get('image_id', '')}"
    )

    digest = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    return digest[:16]


def load_manifest():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Frozen manifest not found:\n{INPUT}"
        )

    return pd.read_csv(
        INPUT,
        low_memory=False,
    )


# ============================================================
# FINAL PUBLIC-FACING MANIFEST
# ============================================================

def build_final_manifest(
    df: pd.DataFrame,
) -> pd.DataFrame:

    final = pd.DataFrame()

    final["record_id"] = df.apply(
        stable_record_id,
        axis=1,
    )

    final["image_path"] = df[
        "image_path"
    ]

    if "image_path_original" in df.columns:
        final[
            "source_image_path"
        ] = df[
            "image_path_original"
        ]

    final["source_dataset"] = df[
        "source_dataset"
    ]

    final["source_image_id"] = df[
        "image_id"
    ]

    final[
        "demographic_label"
    ] = df[
        "candidate_demographic_label"
    ]

    optional_columns = {
        "race_label":
            "source_demographic_label",

        "race_source":
            "demographic_label_source",

        "mapping_reason":
            "harmonization_rule",

        "gender_label":
            "gender_label",

        "gender_source":
            "gender_label_source",

        "age_value":
            "age_value",

        "age_group":
            "age_group",

        "ita_value":
            "ita_value",

        "ita_group":
            "ita_group",

        "identity_id":
            "identity_id",
    }

    for source_col, output_col in (
        optional_columns.items()
    ):
        if source_col in df.columns:
            final[
                output_col
            ] = df[
                source_col
            ]

    return final


# ============================================================
# SUMMARY TABLES
# ============================================================

def build_demographic_summary(
    df: pd.DataFrame,
):

    summary = (
        df[
            "demographic_label"
        ]
        .value_counts()
        .reindex(
            LABEL_ORDER
        )
        .rename_axis(
            "demographic_label"
        )
        .reset_index(
            name="count"
        )
    )

    summary["percentage"] = (
        summary["count"]
        / len(df)
        * 100
    )

    return summary


def build_source_summary(
    df: pd.DataFrame,
):

    table = pd.crosstab(
        df[
            "demographic_label"
        ],
        df[
            "source_dataset"
        ],
    )

    table = table.reindex(
        LABEL_ORDER,
        fill_value=0,
    )

    table["total"] = (
        table.sum(axis=1)
    )

    return table.reset_index()


def build_gender_summary(
    df: pd.DataFrame,
):

    if "gender_label" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()

    temp["gender_label"] = (
        temp["gender_label"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    counts = pd.crosstab(
        temp[
            "demographic_label"
        ],
        temp[
            "gender_label"
        ],
    )

    counts = counts.reindex(
        LABEL_ORDER,
        fill_value=0,
    )

    percentages = (
        counts
        .div(
            counts.sum(axis=1),
            axis=0,
        )
        * 100
    )

    percentages = (
        percentages
        .add_suffix("_percent")
    )

    result = pd.concat(
        [
            counts,
            percentages,
        ],
        axis=1,
    )

    return result.reset_index()


def build_ita_summary(
    df: pd.DataFrame,
):

    if "ita_group" not in df.columns:
        return pd.DataFrame()

    temp = df.copy()

    temp["ita_group"] = (
        temp["ita_group"]
        .fillna("missing")
        .astype(str)
        .str.lower()
    )

    counts = pd.crosstab(
        temp[
            "demographic_label"
        ],
        temp[
            "ita_group"
        ],
    )

    counts = counts.reindex(
        LABEL_ORDER,
        fill_value=0,
    )

    percentages = (
        counts
        .div(
            counts.sum(axis=1),
            axis=0,
        )
        * 100
    )

    percentages = (
        percentages
        .add_suffix("_percent")
    )

    result = pd.concat(
        [
            counts,
            percentages,
        ],
        axis=1,
    )

    return result.reset_index()


# ============================================================
# LABEL-SPECIFIC MANIFESTS
# ============================================================

def save_label_manifests(
    df: pd.DataFrame,
):

    for label in LABEL_ORDER:

        subset = df[
            df[
                "demographic_label"
            ]
            == label
        ].copy()

        filename = (
            label
            .lower()
            .replace(" ", "_")
            + ".csv"
        )

        subset.to_csv(
            LABEL_DIR / filename,
            index=False,
        )


# ============================================================
# VISUAL PREVIEW CONTACT SHEETS
# ============================================================

def create_contact_sheet(
    df: pd.DataFrame,
    label: str,
    n_images: int = 20,
):

    subset = df[
        df[
            "demographic_label"
        ]
        == label
    ].copy()

    if subset.empty:
        return

    sample_n = min(
        n_images,
        len(subset),
    )

    sample = subset.sample(
        n=sample_n,
        random_state=SEED,
    )

    thumb_size = 180
    columns = 5

    rows = (
        sample_n
        + columns
        - 1
    ) // columns

    title_height = 70

    canvas = Image.new(
        "RGB",
        (
            columns * thumb_size,
            rows * thumb_size
            + title_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        canvas
    )

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            24,
        )

        small_font = (
            ImageFont.truetype(
                "arial.ttf",
                12,
            )
        )

    except Exception:
        font = ImageFont.load_default()
        small_font = (
            ImageFont.load_default()
        )

    draw.text(
        (15, 12),
        f"{label} — sample records",
        fill="black",
        font=font,
    )

    draw.text(
        (15, 42),
        (
            "Labels are inherited/harmonized "
            "from source metadata; "
            "not assigned by visual inspection."
        ),
        fill="black",
        font=small_font,
    )

    for position, (
        _,
        row,
    ) in enumerate(
        sample.iterrows()
    ):

        image_path = Path(
            clean_text(
                row[
                    "image_path"
                ]
            )
        )

        x = (
            position
            % columns
        ) * thumb_size

        y = (
            position
            // columns
        ) * thumb_size
        + title_height

        try:
            with Image.open(
                image_path
            ) as image:

                image = (
                    image
                    .convert("RGB")
                )

                image.thumbnail(
                    (
                        thumb_size - 10,
                        thumb_size - 10,
                    )
                )

                tile = Image.new(
                    "RGB",
                    (
                        thumb_size,
                        thumb_size,
                    ),
                    "white",
                )

                paste_x = (
                    thumb_size
                    - image.width
                ) // 2

                paste_y = (
                    thumb_size
                    - image.height
                ) // 2

                tile.paste(
                    image,
                    (
                        paste_x,
                        paste_y,
                    ),
                )

                canvas.paste(
                    tile,
                    (x, y),
                )

        except Exception:
            draw.rectangle(
                (
                    x,
                    y,
                    x + thumb_size,
                    y + thumb_size,
                ),
                outline="black",
            )

            draw.text(
                (
                    x + 10,
                    y + 10,
                ),
                "Image unavailable",
                fill="black",
                font=small_font,
            )

    filename = (
        label
        .lower()
        .replace(" ", "_")
        + "_preview.jpg"
    )

    canvas.save(
        PREVIEW_DIR / filename,
        quality=92,
    )


def create_all_previews(
    df: pd.DataFrame,
):

    for label in LABEL_ORDER:
        print(
            f"Creating preview: "
            f"{label}"
        )

        create_contact_sheet(
            df,
            label,
        )


# ============================================================
# DATASET CARD
# ============================================================

def write_dataset_card(
    df: pd.DataFrame,
    demographic_summary:
        pd.DataFrame,
):

    counts = dict(
        zip(
            demographic_summary[
                "demographic_label"
            ],
            demographic_summary[
                "count"
            ],
        )
    )

    largest = max(
        counts.values()
    )

    smallest = min(
        counts.values()
    )

    ratio = (
        largest / smallest
    )

    text = f"""# {DATASET_NAME}

Version: {VERSION}

## Overview

This dataset is a harmonized, diversity-aware combination of
FairFace, UTKFace and Labeled Faces in the Wild (LFW).

The final manifest contains **{len(df):,} valid image records**.

The source images are not altered. The final dataset is represented
through a reproducible manifest that preserves the location and
provenance of every image.

## Final demographic taxonomy

The dataset uses seven explicit demographic categories:

1. White
2. Black
3. Indian
4. East Asian
5. Southeast Asian
6. Middle Eastern
7. Latino

## Final composition

"""

    for label in LABEL_ORDER:
        n = counts[
            label
        ]

        percentage = (
            n
            / len(df)
            * 100
        )

        text += (
            f"- {label}: "
            f"{n:,} "
            f"({percentage:.2f}%)\n"
        )

    text += f"""

The largest-to-smallest demographic group ratio is
**{ratio:.2f}:1**.

## Construction principles

The dataset was created using the following principles:

- Source images were not relabelled through visual inspection.
- Original source labels and provenance are preserved.
- No minority demographic group was duplicated to create balance.
- Large groups were selectively reduced rather than minority
  groups being repeatedly oversampled.
- A 35/65 gender representation floor was applied where required.
- Skin tone was analysed separately using ITA rather than being
  treated as equivalent to demographic category.
- Image brightness, contrast and sharpness were audited separately.
- Records with invalid or missing image files were excluded from
  the final manifest.
- Known LFW identities were checked to prevent duplication across
  dataset partitions during development.

## Source-label harmonization

FairFace labels were retained at their available level of detail.

UTKFace and LFW records labelled only as broad `Asian` were not
forced into East Asian or Southeast Asian categories and therefore
were not included in this seven-class final product.

`Other` and `Unknown` records were also preserved during development
but excluded from this explicit-label final dataset.

## Important interpretation

These demographic labels are dataset annotations and should not be
treated as objective biological categories.

Race, ethnicity, geographic population and measured skin tone are
not interchangeable concepts.

The dataset is intended for research into representation, dataset
bias and computer-vision fairness.

## Files

- `harmonized_face_dataset_manifest.csv`
  - canonical final dataset manifest

- `labels/`
  - one CSV manifest for each demographic category

- `demographic_summary.csv`
  - final demographic counts and percentages

- `source_summary.csv`
  - contribution of each source dataset

- `gender_summary.csv`
  - gender composition by demographic category

- `ita_summary.csv`
  - ITA composition by demographic category

- `previews/`
  - visual sample sheets for inspection

- `label_schema.json`
  - machine-readable taxonomy definition

## Provenance

Source datasets:

- FairFace
- UTKFace
- Labeled Faces in the Wild (LFW)

The harmonized manifest preserves source dataset information for
every record.

## Limitations

Some final demographic categories originate predominantly or
entirely from FairFace because the other source datasets do not
provide equivalent fine-grained labels.

Gender labels use the binary categories available in the source
metadata and should not be interpreted as a complete representation
of gender identity.

ITA measurements have missing values for a small proportion of
records and are affected by imaging and illumination conditions.

The dataset should therefore be described as **harmonized,
diversity-aware and representation-balanced**, rather than as a
perfectly fair or universally representative dataset.
"""

    (
        OUTPUT_DIR
        / "DATASET_CARD.md"
    ).write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# LABEL SCHEMA
# ============================================================

def write_label_schema():

    schema = {
        "dataset_name":
            DATASET_NAME,

        "version":
            VERSION,

        "primary_label":
            "demographic_label",

        "labels":
            LABEL_ORDER,

        "source_datasets": [
            "fairface",
            "utkface",
            "lfw",
        ],

        "notes": {
            "demographic_label":
                (
                    "Harmonized from source metadata. "
                    "No appearance-based manual "
                    "reclassification was performed."
                ),

            "gender_label":
                (
                    "Preserved from available "
                    "source/derived metadata."
                ),

            "ita_value":
                (
                    "Skin-tone measurement retained "
                    "separately from demographic label."
                ),

            "ita_group":
                (
                    "Derived skin-tone grouping; "
                    "not equivalent to race or ethnicity."
                ),
        },
    }

    (
        OUTPUT_DIR
        / "label_schema.json"
    ).write_text(
        json.dumps(
            schema,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# README
# ============================================================

def write_readme():

    text = f"""# {DATASET_NAME}

The final dataset product is contained in this directory.

Start with:

1. `DATASET_CARD.md`
2. `harmonized_face_dataset_manifest.csv`
3. `demographic_summary.csv`
4. `previews/`

The manifest is the canonical dataset.

Images remain in their original source locations and are referenced
by `image_path`. This avoids unnecessary image duplication and
preserves provenance.
"""

    (
        OUTPUT_DIR
        / "README.md"
    ).write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("CREATE FINAL HARMONIZED DATASET PRODUCT")
    print("=" * 72)

    source = load_manifest()

    final = build_final_manifest(
        source
    )

    # Safety / integrity checks
    duplicates = final.duplicated(
        subset=[
            "source_dataset",
            "source_image_id",
        ]
    ).sum()

    if duplicates:
        raise RuntimeError(
            f"Duplicate source records: "
            f"{duplicates}"
        )

    invalid_labels = (
        set(
            final[
                "demographic_label"
            ].unique()
        )
        - set(
            LABEL_ORDER
        )
    )

    if invalid_labels:
        raise RuntimeError(
            f"Unexpected labels: "
            f"{invalid_labels}"
        )

    # --------------------------------------------------------
    # Save canonical manifest
    # --------------------------------------------------------

    final.to_csv(
        OUTPUT_DIR
        / "harmonized_face_dataset_manifest.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    demographic_summary = (
        build_demographic_summary(
            final
        )
    )

    source_summary = (
        build_source_summary(
            final
        )
    )

    gender_summary = (
        build_gender_summary(
            final
        )
    )

    ita_summary = (
        build_ita_summary(
            final
        )
    )

    demographic_summary.to_csv(
        OUTPUT_DIR
        / "demographic_summary.csv",
        index=False,
    )

    source_summary.to_csv(
        OUTPUT_DIR
        / "source_summary.csv",
        index=False,
    )

    if not gender_summary.empty:
        gender_summary.to_csv(
            OUTPUT_DIR
            / "gender_summary.csv",
            index=False,
        )

    if not ita_summary.empty:
        ita_summary.to_csv(
            OUTPUT_DIR
            / "ita_summary.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Per-label manifests
    # --------------------------------------------------------

    save_label_manifests(
        final
    )

    # --------------------------------------------------------
    # Documentation
    # --------------------------------------------------------

    write_label_schema()

    write_dataset_card(
        final,
        demographic_summary,
    )

    write_readme()

    # --------------------------------------------------------
    # Visual previews
    # --------------------------------------------------------

    create_all_previews(
        final
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("FINAL DATASET")
    print("=" * 72)

    print(
        f"\nDataset records: "
        f"{len(final):,}"
    )

    print("\nFinal labels:")

    print(
        demographic_summary.to_string(
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
    print("SOURCE CONTRIBUTION")
    print("=" * 72)

    print(
        source_summary.to_string(
            index=False
        )
    )

    print("\nFinal product:")
    print(OUTPUT_DIR)

    print(
        "\nNo model was trained."
    )

    print(
        "No source image was modified."
    )


if __name__ == "__main__":
    main()
    