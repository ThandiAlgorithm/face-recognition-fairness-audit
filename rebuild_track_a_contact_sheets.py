from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(
    "results/model_audit/model_comparison/"
    "track_a_final/qualitative_examples"
)

DATASETS = {
    "fairface": "FairFace Validation",
    "utkface": "UTKFace Aligned",
    "lfw": "LFW Prediction-Only",
}

IMAGES_PER_PAGE = 10

COLUMNS = 5
ROWS = 2


# ============================================================
# LOAD INDIVIDUAL PANELS
# ============================================================

def get_individual_panels(
    dataset: str,
) -> list[Path]:

    folder = (
        BASE
        / dataset
        / "individual"
    )

    if not folder.exists():

        raise FileNotFoundError(
            f"Individual-panel folder not found:\n"
            f"{folder}"
        )

    panels = sorted(
        folder.glob("*.png")
    )

    if len(panels) != 30:

        raise RuntimeError(
            f"Expected 30 individual panels for "
            f"{dataset}, found {len(panels)}."
        )

    return panels


# ============================================================
# CONTACT SHEET
# ============================================================

def build_contact_sheet(
    dataset: str,
    dataset_title: str,
    panel_paths: list[Path],
    page_number: int,
) -> Path:

    # Each input PNG already contains:
    #
    #   white caption region
    #   +
    #   photograph
    #
    # We therefore do NOT draw another subplot title.
    # This avoids text overlapping the second image row.

    fig, axes = plt.subplots(
        ROWS,
        COLUMNS,
        figsize=(
            25,
            14,
        ),
        facecolor="white",
    )

    axes = axes.flatten()

    for axis in axes:

        axis.set_facecolor(
            "white"
        )

        axis.axis(
            "off"
        )

    for axis, panel_path in zip(
        axes,
        panel_paths,
    ):

        with Image.open(
            panel_path
        ) as image:

            rgb = image.convert(
                "RGB"
            )

            axis.imshow(
                rgb
            )

        axis.axis(
            "off"
        )

    fig.suptitle(
        (
            "Track A Qualitative Audit — "
            f"{dataset_title} — "
            f"Page {page_number}"
        ),
        fontsize=20,
        y=0.985,
    )

    # Deliberately leave space between the two rows.
    fig.subplots_adjust(
        left=0.015,
        right=0.985,
        bottom=0.02,
        top=0.94,
        wspace=0.055,
        hspace=0.16,
    )

    output_path = (
        BASE
        / dataset
        / f"contact_sheet_{page_number:02d}.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.18,
    )

    plt.close(
        fig
    )

    return output_path


# ============================================================
# DATASET
# ============================================================

def rebuild_dataset(
    dataset: str,
    dataset_title: str,
) -> None:

    panels = get_individual_panels(
        dataset
    )

    print()
    print(
        f"{dataset.upper()}"
    )

    print(
        "-" * len(
            dataset
        )
    )

    for start in range(
        0,
        len(panels),
        IMAGES_PER_PAGE,
    ):

        page_number = (
            start
            // IMAGES_PER_PAGE
            + 1
        )

        page_panels = panels[
            start:
            start + IMAGES_PER_PAGE
        ]

        output = build_contact_sheet(
            dataset,
            dataset_title,
            page_panels,
            page_number,
        )

        print(
            f"Rebuilt: {output}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "Rebuilding Track A Contact Sheets"
    )

    print(
        "================================="
    )

    for (
        dataset,
        dataset_title,
    ) in DATASETS.items():

        rebuild_dataset(
            dataset,
            dataset_title,
        )

    print()
    print(
        "CONTACT-SHEET REBUILD COMPLETE"
    )

    print(
        "================================="
    )

    print(
        "Nine contact sheets overwritten."
    )

    print()
    print(
        "Open:"
    )

    print(
        BASE.resolve()
    )


if __name__ == "__main__":
    main()
    