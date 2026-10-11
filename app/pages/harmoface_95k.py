from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ui import (
    evidence_note,
    metric_row,
    page_header,
    research_card,
    section_header,
)


# ============================================================
# PATHS
# ============================================================

REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RESULTS_ROOT = (
    REPO_ROOT
    / "results"
    / "harmonized_dataset"
)

FINAL_ROOT = (
    RESULTS_ROOT
    / "FINAL_HARMONIZED_DATASET"
)

MANIFEST_PATH = (
    FINAL_ROOT
    / "harmonized_face_dataset_manifest.csv"
)

FROZEN_PATH = (
    RESULTS_ROOT
    / "final_mvp_candidate"
    / "frozen_training"
    / "training_manifest_frozen.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

WITS_NAVY = "#002B5C"
WITS_BLUE = "#0067B1"
WITS_GOLD = "#D8AA19"

CLASS_ORDER = [
    "White",
    "Black",
    "Indian",
    "East Asian",
    "Southeast Asian",
    "Middle Eastern",
    "Latino",
]

CLASS_COUNTS_FALLBACK = {
    "White": 14981,
    "Black": 14986,
    "Indian": 14989,
    "East Asian": 13837,
    "Southeast Asian": 12210,
    "Middle Eastern": 9265,
    "Latino": 14990,
}

SOURCE_COUNTS_FALLBACK = {
    "FairFace": 80034,
    "UTKFace": 10948,
    "LFW": 4276,
}


# ============================================================
# HELPERS
# ============================================================

@st.cache_data(
    show_spinner=False
)
def load_csv(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        low_memory=False,
    )


def first_existing_column(
    frame: pd.DataFrame,
    choices: list[str],
) -> str | None:

    for column in choices:

        if column in frame.columns:
            return column

    return None


def pretty_label(
    value,
) -> str:

    text = (
        str(value)
        .strip()
        .replace("_", " ")
    )

    replacements = {
        "East asian": "East Asian",
        "Southeast asian": "Southeast Asian",
        "South east asian": "Southeast Asian",
        "Middle eastern": "Middle Eastern",
    }

    return replacements.get(
        text,
        text,
    )


def pretty_source(
    value,
) -> str:

    text = (
        str(value)
        .strip()
        .lower()
    )

    mapping = {
        "fairface": "FairFace",
        "fair_face": "FairFace",
        "utkface": "UTKFace",
        "utk_face": "UTKFace",
        "lfw": "LFW",
        "labeled faces in the wild": "LFW",
    }

    return mapping.get(
        text,
        str(value),
    )


def composition_from_manifest(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    column = first_existing_column(
        frame,
        [
            "demographic_label",
            "candidate_demographic_label",
            "harmonized_demographic_label",
            "harmonized_label",
            "race_label",
            "label",
        ],
    )

    if (
        frame.empty
        or column is None
    ):

        output = pd.DataFrame(
            {
                "Demographic class": (
                    list(
                        CLASS_COUNTS_FALLBACK.keys()
                    )
                ),
                "Images": (
                    list(
                        CLASS_COUNTS_FALLBACK.values()
                    )
                ),
            }
        )

    else:

        values = (
            frame[column]
            .fillna("Unknown")
            .map(pretty_label)
            .value_counts()
        )

        output = (
            values
            .rename_axis(
                "Demographic class"
            )
            .reset_index(
                name="Images"
            )
        )

    output["Share (%)"] = (
        100
        * output["Images"]
        / output["Images"].sum()
    ).round(2)

    order_map = {
        label: index
        for index, label
        in enumerate(
            CLASS_ORDER
        )
    }

    output["_order"] = (
        output[
            "Demographic class"
        ]
        .map(order_map)
        .fillna(999)
    )

    output = (
        output
        .sort_values(
            "_order"
        )
        .drop(
            columns="_order"
        )
        .reset_index(
            drop=True
        )
    )

    return output


def source_from_manifest(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    column = first_existing_column(
        frame,
        [
            "source_dataset",
            "dataset",
            "source",
            "dataset_source",
        ],
    )

    if (
        frame.empty
        or column is None
    ):

        output = pd.DataFrame(
            {
                "Source dataset": (
                    list(
                        SOURCE_COUNTS_FALLBACK.keys()
                    )
                ),
                "Images": (
                    list(
                        SOURCE_COUNTS_FALLBACK.values()
                    )
                ),
            }
        )

    else:

        values = (
            frame[column]
            .fillna("Unknown")
            .map(pretty_source)
            .value_counts()
        )

        output = (
            values
            .rename_axis(
                "Source dataset"
            )
            .reset_index(
                name="Images"
            )
        )

    output["Share (%)"] = (
        100
        * output["Images"]
        / output["Images"].sum()
    ).round(2)

    return output


# ============================================================
# DATA
# ============================================================

manifest = load_csv(
    MANIFEST_PATH
)

frozen = load_csv(
    FROZEN_PATH
)

demographic = (
    composition_from_manifest(
        manifest
    )
)

source_table = (
    source_from_manifest(
        manifest
    )
)

total_records = (
    len(manifest)
    if not manifest.empty
    else 95_258
)


# ============================================================
# SPLIT COUNTS
# ============================================================

if (
    not frozen.empty
    and "split" in frozen.columns
):

    split_counts = (
        frozen["split"]
        .astype(str)
        .str.lower()
        .value_counts()
    )

    train_n = int(
        split_counts.get(
            "train",
            0,
        )
    )

    val_n = int(
        split_counts.get(
            "val",
            split_counts.get(
                "validation",
                0,
            ),
        )
    )

    test_n = int(
        split_counts.get(
            "test",
            0,
        )
    )

else:

    train_n = 76_395
    val_n = 9_544
    test_n = 9_319


# ============================================================
# HEADER
# ============================================================

page_header(
    "HarmoFace-95K",
    (
        "A harmonised, diversity-aware and provenance-preserving "
        "face dataset constructed from FairFace, UTKFace and LFW."
    ),
    "Dataset contribution",
)


# ============================================================
# HEADLINE
# ============================================================

metric_row(
    [
        (
            "Images",
            f"{total_records:,}",
        ),
        (
            "Demographic classes",
            "7",
        ),
        (
            "Source datasets",
            "3",
        ),
        (
            "Largest / smallest class",
            "1.62 : 1",
        ),
    ]
)

st.write(
    """
    HarmoFace-95K was developed from the combined dataset audit to
    reduce majority dominance while retaining useful data volume,
    source provenance and measurable diversity.

    The final taxonomy contains **White, Black, Indian, East Asian,
    Southeast Asian, Middle Eastern and Latino** records.
    """
)

st.info(
    """
    HarmoFace-95K is described as harmonised and diversity-aware rather
    than perfectly balanced or universally representative. The dataset
    retains the limitations and annotation uncertainty of its source
    data.
    """
)

st.divider()


# ============================================================
# MAIN TABS
# ============================================================

(
    composition_tab,
    provenance_tab,
    construction_tab,
    diversity_tab,
    limitations_tab,
) = st.tabs(
    [
        "Composition",
        "Provenance",
        "Construction",
        "Diversity",
        "Limitations",
    ]
)


# ============================================================
# COMPOSITION
# ============================================================

with composition_tab:

    section_header(
        "Final demographic composition",
        (
            "The selected balancing strategy reduces majority "
            "dominance without duplicating minority records."
        ),
    )

    fig = px.bar(
        demographic,
        x="Demographic class",
        y="Images",
        text="Images",
        color="Demographic class",
        category_orders={
            "Demographic class": (
                CLASS_ORDER
            )
        },
        title=(
            "HarmoFace-95K demographic composition"
        ),
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )

    fig.update_layout(
        showlegend=False,
        height=470,
        xaxis_title="",
        yaxis_title="Images",
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        demographic,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Share (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
        },
    )

    st.caption(
        """
        The class counts describe the engineered dataset distribution.
        They do not imply that demographic categories are biological
        ground truth or that these proportions represent the global
        population.
        """
    )

    st.divider()

    section_header(
        "Frozen dataset split"
    )

    metric_row(
        [
            (
                "Training",
                f"{train_n:,}",
            ),
            (
                "Validation",
                f"{val_n:,}",
            ),
            (
                "Test",
                f"{test_n:,}",
            ),
        ]
    )

    split_frame = pd.DataFrame(
        {
            "Split": [
                "Train",
                "Validation",
                "Test",
            ],
            "Images": [
                train_n,
                val_n,
                test_n,
            ],
        }
    )

    fig = px.bar(
        split_frame,
        x="Split",
        y="Images",
        text="Images",
        color="Split",
        title="Frozen train / validation / test split",
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )

    fig.update_layout(
        showlegend=False,
        height=370,
        xaxis_title="",
        yaxis_title="Images",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.success(
        """
        Known LFW identities were checked for cross-split leakage.
        No LFW identity leakage was detected in the frozen split.
        """
    )


# ============================================================
# PROVENANCE
# ============================================================

with provenance_tab:

    section_header(
        "Source provenance",
        (
            "Every final record retains the dataset from which it "
            "originated."
        ),
    )

    left, right = st.columns(
        [
            1,
            1,
        ]
    )

    with left:

        fig = px.pie(
            source_table,
            names="Source dataset",
            values="Images",
            hole=0.55,
            title=(
                "Contribution by source dataset"
            ),
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        fig.update_layout(
            height=430,
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:

        st.dataframe(
            source_table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Share (%)": (
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    )
                ),
            },
        )

        st.markdown(
            """
            **Final source mix**

            - FairFace: approximately **84.02%**
            - UTKFace: approximately **11.49%**
            - LFW: approximately **4.49%**
            """
        )

    st.info(
        """
        Provenance is retained because source dataset and demographic
        category are not independent. Differences in acquisition,
        annotation, image quality and preprocessing can remain visible
        after datasets are merged.
        """
    )

    st.divider()

    section_header(
        "Source dependence by class"
    )

    st.warning(
        """
        East Asian, Southeast Asian, Middle Eastern and Latino records
        in the final dataset originate entirely from FairFace. These
        classes therefore remain source-dependent even though the
        overall demographic distribution is substantially more balanced.
        """
    )

    source_by_class = pd.DataFrame(
        {
            "Demographic class": [
                "White",
                "Black",
                "Indian",
                "East Asian",
                "Southeast Asian",
                "Middle Eastern",
                "Latino",
            ],
            "FairFace": [
                7266,
                10980,
                11486,
                13837,
                12210,
                9265,
                14990,
            ],
            "LFW": [
                3734,
                382,
                160,
                0,
                0,
                0,
                0,
            ],
            "UTKFace": [
                3981,
                3624,
                3343,
                0,
                0,
                0,
                0,
            ],
        }
    )

    source_long = (
        source_by_class
        .melt(
            id_vars=(
                "Demographic class"
            ),
            var_name=(
                "Source dataset"
            ),
            value_name="Images",
        )
    )

    fig = px.bar(
        source_long,
        x="Demographic class",
        y="Images",
        color="Source dataset",
        barmode="stack",
        category_orders={
            "Demographic class": (
                CLASS_ORDER
            )
        },
        title=(
            "Source contribution within each demographic class"
        ),
    )

    fig.update_layout(
        height=470,
        xaxis_title="",
        yaxis_title="Images",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# CONSTRUCTION
# ============================================================

with construction_tab:

    section_header(
        "Dataset construction",
        (
            "The final dataset is the result of explicit label, "
            "balancing, diversity and integrity decisions."
        ),
    )

    pipeline = st.columns(4)

    with pipeline[0]:

        research_card(
            "Source inventory",
            (
                "135,039 records from FairFace, UTKFace and LFW were "
                "inventoried for labels, paths, identities and "
                "available image measurements."
            ),
            "01",
        )

    with pipeline[1]:

        research_card(
            "Label harmonisation",
            (
                "Source labels were mapped conservatively into seven "
                "explicit categories while broad or unresolved labels "
                "were kept outside the final taxonomy."
            ),
            "02",
        )

    with pipeline[2]:

        research_card(
            "Volume-aware balancing",
            (
                "Alternative balancing strategies were compared before "
                "selecting a method that reduced majority dominance "
                "without duplicating minority records."
            ),
            "03",
        )

    with pipeline[3]:

        research_card(
            "Integrity validation",
            (
                "Path checks, demographic audits and split-integrity "
                "checks produced the final frozen 95,258-image dataset."
            ),
            "04",
        )

    st.divider()

    section_header(
        "Why strict parity was not selected"
    )

    strategy = pd.DataFrame(
        {
            "Strategy": [
                "Strict parity · all classes",
                "Strict parity · explicit classes",
                "Volume-aware cap",
                "Final validated dataset",
            ],
            "Records": [
                35632,
                72975,
                96432,
                95258,
            ],
            "Retention (%)": [
                26.39,
                54.04,
                71.41,
                70.54,
            ],
        }
    )

    fig = px.bar(
        strategy,
        x="Strategy",
        y="Records",
        text="Records",
        color="Strategy",
        title=(
            "Data retained under alternative construction strategies"
        ),
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )

    fig.update_layout(
        showlegend=False,
        height=450,
        xaxis_title="",
        yaxis_title="Records",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        strategy,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Retention (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
        },
    )

    st.info(
        """
        Strict equal-count balancing would have removed a substantial
        amount of usable data. The selected strategy instead preserves
        more data while constraining the largest demographic classes.
        """
    )

    st.divider()

    section_header(
        "Final validation"
    )

    metric_row(
        [
            (
                "Candidate after gender check",
                "95,272",
            ),
            (
                "Invalid paths removed",
                "14",
            ),
            (
                "Final valid images",
                "95,258",
            ),
            (
                "LFW identity leakage",
                "0",
            ),
        ]
    )


# ============================================================
# DIVERSITY
# ============================================================

with diversity_tab:

    section_header(
        "Diversity beyond demographic counts",
        (
            "Image count parity is not treated as the complete "
            "definition of dataset balance."
        ),
    )

    metric_row(
        [
            (
                "ITA coverage",
                "95.2%",
            ),
            (
                "Exact age coverage",
                "11.4%",
            ),
            (
                "Face-area coverage",
                "15.8%",
            ),
            (
                "Gender floor",
                "35 / 65",
            ),
        ]
    )

    left, right = st.columns(2)

    with left:

        evidence_note(
            "Skin tone remains separate",
            (
                "ITA is retained as an image-derived skin-tone "
                "measurement rather than being treated as equivalent "
                "to race or ethnicity."
            ),
        )

    with right:

        evidence_note(
            "No ITA quota",
            (
                "Exact ITA parity was rejected because it would remove "
                "large amounts of valid data and would imply a stronger "
                "relationship between demographic category and skin "
                "tone than the evidence supports."
            ),
        )

    st.divider()

    section_header(
        "Gender representation"
    )

    st.write(
        """
        A project-specific soft **35/65 gender floor** was used as a
        representation safeguard rather than as a claim that every
        demographic group must contain exactly equal gender counts.

        Only the Middle Eastern candidate group required adjustment.
        The adjustment reduced male-labelled records without
        oversampling female-labelled records.
        """
    )

    st.divider()

    section_header(
        "Image conditions"
    )

    st.write(
        """
        Brightness, contrast, sharpness and available face-scale
        measurements were audited alongside demographic composition.

        These variables were documented rather than artificially forced
        to identical distributions because doing so would remove valid
        data and potentially conceal meaningful source-dataset
        differences.
        """
    )

    st.info(
        """
        The objective is not to erase every measurable difference
        between groups. The objective is to make those differences
        visible, measurable and reproducible.
        """
    )


# ============================================================
# LIMITATIONS
# ============================================================

with limitations_tab:

    section_header(
        "Limitations",
        (
            "HarmoFace-95K reduces several representation problems but "
            "does not remove all uncertainty or source bias."
        ),
    )

    limitations = [
        (
            "Not globally representative",
            (
                "The final class distribution is an engineered research "
                "distribution, not an estimate of world population "
                "demographics."
            ),
        ),
        (
            "Labels remain annotations",
            (
                "Demographic categories originate from source labels or "
                "conservative harmonisation rules. They are not "
                "objective biological truth."
            ),
        ),
        (
            "Source concentration remains",
            (
                "Four final demographic classes are sourced entirely "
                "from FairFace, limiting source diversity within those "
                "categories."
            ),
        ),
        (
            "Image domains differ",
            (
                "FairFace, UTKFace and LFW differ in acquisition "
                "conditions, image quality, identity structure and "
                "annotation history."
            ),
        ),
        (
            "Age is incomplete",
            (
                "Exact age is available for only a minority of the "
                "combined records and was therefore not used as a hard "
                "balancing constraint."
            ),
        ),
        (
            "Licensing remains source-dependent",
            (
                "Redistribution of the underlying images must respect "
                "the terms and licences of the original datasets."
            ),
        ),
    ]

    for title, description in limitations:

        with st.container(
            border=True
        ):

            st.markdown(
                f"### {title}"
            )

            st.write(
                description
            )

    st.divider()

    section_header(
        "Appropriate use"
    )

    st.success(
        """
        HarmoFace-95K is intended as a documented, training-ready
        research dataset for controlled experimentation in demographic
        classification, facial-recognition evaluation and mitigation.
        """
    )

    st.warning(
        """
        The dataset should not be described as perfectly fair,
        universally representative or free of demographic measurement
        uncertainty.
        """
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

st.divider()

section_header(
    "Reproducibility",
    (
        "The final dataset is frozen so that later experiments can be "
        "repeated using the same images and partitions."
    ),
)

r1, r2, r3 = st.columns(3)

with r1:

    research_card(
        "Frozen manifest",
        (
            "The canonical manifest records each final image, its "
            "harmonised label, source provenance and assigned split."
        ),
        "MANIFEST",
    )

with r2:

    research_card(
        "Materialised images",
        (
            "Physical image folders are organised by train, validation "
            "and test split, then by demographic class."
        ),
        "IMAGES",
    )

with r3:

    research_card(
        "Audit metadata",
        (
            "Supporting metadata preserves source labels and additional "
            "QA information without modifying image pixels."
        ),
        "METADATA",
    )

with st.expander(
    "Dataset folder structure"
):

    st.code(
        """
HarmoFace-95K/
├── images/
│   ├── train/
│   │   ├── Black/
│   │   ├── East_Asian/
│   │   ├── Indian/
│   │   ├── Latino/
│   │   ├── Middle_Eastern/
│   │   ├── Southeast_Asian/
│   │   └── White/
│   ├── val/
│   └── test/
└── metadata/
    ├── records.csv
    ├── training_manifest_frozen.csv
    ├── class_to_index.json
    └── materialization_summary.json
        """,
        language=None,
    )

st.caption(
    """
    HarmoFace-95K · University of the Witwatersrand ·
    School of Electrical and Information Engineering
    """
)