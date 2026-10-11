from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data import (
    dataset_label,
    load_master,
)
from representation_data import (
    load_rep001_age,
    load_rep001_availability,
    load_rep001_dataset_summary,
    load_rep001_gender,
    load_rep001_race,
    load_rep002_identity_support,
    load_rep003_identity_distribution,
    load_rep003_identity_summary,
    load_rep004_intersections,
    load_rep004_status_summary,
    load_rep005_age,
    load_rep005_overall,
    load_rep005_race,
    load_rep006_affected_analyses,
    load_rep006_missingness,
    load_rep006_patterns,
)
from phase1_audit_data import (
    alignment_qa_artifact,
    alignment_qa_sample,
    detection_figure,
    load_alignment_failures_by_subgroup,
    load_alignment_qa_manifest,
    load_brightness_vs_skin_tone,
    load_confidence_histogram,
    load_confidence_summary,
    load_detection_integrity_by_dataset,
    load_detector_limitations_text,
    load_lfw_composition_images_vs_pairs,
    load_lfw_composition_tests,
    load_lfw_genuine_visual_difference_by_skin_group,
    load_lfw_genuine_visual_difference_tests,
    load_lfw_impostor_same_group_vs_random,
    load_lfw_impostor_yoking_counts,
    load_lfw_pair_file_checks,
    load_lfw_pair_resolution_counts,
    load_lfw_protocol_rules,
    load_lfw_status,
    load_lfw_visual_difference_median_by_yoking,
    load_mea_boundary_sensitivity,
    load_mea_limitations_text,
    load_mea_method,
    load_mea_reliability_agreement,
    load_mea_roi_failure_by_group,
    load_mea_roi_failure_tests,
    load_mea_skin_group_distribution,
    load_mea_status_counts,
    load_mea_subgroup_omnibus,
    load_mea_subgroup_pairwise,
    load_mea_subgroup_summary,
    load_mea_summary_by_dataset,
    load_shift_analysis_set_counts,
    load_shift_categorical_distributions,
    load_shift_categorical_tests,
    load_shift_continuous_summary,
    load_shift_continuous_tests,
    load_shift_effective_balance_summary,
    load_shift_effective_balance_tests,
    load_shift_frozen_axes,
    load_shift_identity_depth,
    load_shift_ita_within_native_race,
    load_shift_limitations_text,
    load_shift_nominal_balance,
    load_subgroup_detection_failures,
    load_vis_brightness_vs_skin,
    load_vis_qc_flag_counts,
    load_vis_status_counts,
    load_vis_subgroup_omnibus,
    load_vis_subgroup_pairwise,
    load_vis_subgroup_summary,
    load_vis_summary_by_dataset,
    mea_figure,
    mea_qa_image,
    shift_figure,
    vis_figure,
    vis_qa_image,
)
from ui import (
    evidence_note,
    metric_row,
    page_header,
    research_card,
    section_header,
)


# ============================================================
# VISUAL CONSTANTS
# ============================================================

WITS_NAVY = "#002B5C"
WITS_BLUE = "#0067B1"
WITS_GOLD = "#D8AA19"
LIGHT_BLUE = "#5DADE2"
GREY = "#9AA5B1"

DATASET_COLOURS = {
    "LFW": WITS_NAVY,
    "FairFace": WITS_BLUE,
    "UTKFace": WITS_GOLD,
    "lfw": WITS_NAVY,
    "fairface": WITS_BLUE,
    "utkface": WITS_GOLD,
}

PALETTE = [
    WITS_BLUE,
    WITS_GOLD,
    WITS_NAVY,
    LIGHT_BLUE,
    "#6C8EBF",
    "#A67C00",
    "#4C6A92",
    "#80B9DD",
]


# ============================================================
# HELPERS
# ============================================================

def pretty_dataset(value: str) -> str:
    names = {
        "lfw": "LFW",
        "fairface": "FairFace",
        "utkface": "UTKFace",
    }

    return names.get(
        str(value).lower(),
        str(value),
    )


def bool_count(
    frame: pd.DataFrame,
    column: str,
) -> int:

    if column not in frame.columns:
        return 0

    return int(
        frame[column]
        .eq(True)
        .sum()
    )


def show_table(
    frame: pd.DataFrame,
    empty_message: str = "Data not available.",
    height: int | str | None = None,
) -> None:

    if frame.empty:
        st.info(
            empty_message
        )
        return

    dataframe_options = {
        "hide_index": True,
        "use_container_width": True,
    }

    if height is not None:
        dataframe_options["height"] = height

    st.dataframe(
        frame,
        **dataframe_options,
    )
    


def show_image(
    path,
    caption: str | None = None,
) -> None:

    if path.exists():
        st.image(
            str(path),
            caption=caption,
            use_container_width=True,
        )
    else:
        st.info(
            f"Figure not available: {path.name}"
        )


def image_grid(
    paths: list,
    columns: int = 4,
) -> None:

    for start in range(
        0,
        len(paths),
        columns,
    ):

        row_paths = paths[
            start:start + columns
        ]

        cols = st.columns(
            columns
        )

        for col, path in zip(
            cols,
            row_paths,
        ):

            with col:

                if path.exists():

                    st.image(
                        str(path),
                        use_container_width=True,
                        caption=(
                            path.stem
                            .replace(
                                ".aligned",
                                "",
                            )
                        ),
                    )


def prepare_dataset_names(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    output = frame.copy()

    if "dataset" in output.columns:

        output["Dataset"] = (
            output["dataset"]
            .map(pretty_dataset)
        )

    return output


# ============================================================
# LOAD DATA
# ============================================================

try:

    master = (
        load_master()
        .copy()
    )

except Exception as exc:

    st.error(
        f"Could not load the master audit table: {exc}"
    )

    st.stop()


master["Dataset"] = (
    master["dataset"]
    .map(dataset_label)
)

dataset_summary = (
    load_rep001_dataset_summary()
)

gender_data = (
    load_rep001_gender()
)

age_data = (
    load_rep001_age()
)

race_data = (
    load_rep001_race()
)

availability_data = (
    load_rep001_availability()
)

identity_support = (
    load_rep002_identity_support()
)

identity_summary = (
    load_rep003_identity_summary()
)

identity_distribution = (
    load_rep003_identity_distribution()
)

intersection_data = (
    load_rep004_intersections()
)

intersection_status = (
    load_rep004_status_summary()
)

base_rate_overall = (
    load_rep005_overall()
)

base_rate_race = (
    load_rep005_race()
)

base_rate_age = (
    load_rep005_age()
)

missingness = (
    load_rep006_missingness()
)

missingness_patterns = (
    load_rep006_patterns()
)

affected_analyses = (
    load_rep006_affected_analyses()
)


# ============================================================
# HEADER
# ============================================================

page_header(
    "Dataset Audit",
    (
        "A forensic assessment of representation, identity structure, "
        "data integrity, image conditions and measurement reliability "
        "across FairFace, UTKFace and LFW."
    ),
    "Dataset evidence",
)


# ============================================================
# SUMMARY
# ============================================================

metric_row(
    [
        (
            "Images audited",
            f"{len(master):,}",
        ),
        (
            "Datasets",
            f"{master['dataset'].nunique():,}",
        ),
        (
            "LFW identities",
            (
                f"{master.loc[master['dataset'].eq('lfw'), 'subject_id'].nunique():,}"
            ),
        ),
        (
            "Valid skin ROIs",
            f"{bool_count(master, 'skin_roi_valid'):,}",
        ),
    ]
)

st.write(
    """
    The dataset audit is designed to establish what is represented in
    the data before model performance is interpreted. Representation,
    image quality, preprocessing behaviour and measurement reliability
    are treated as possible contributors to later performance
    differences rather than as fairness conclusions by themselves.
    """
)

st.divider()


# ============================================================
# SELECTED FINDINGS
# ============================================================

section_header(
    "Selected findings",
    "The strongest dataset-level observations from the combined audit.",
)

f1, f2, f3 = st.columns(3)

with f1:

    research_card(
        "Dataset dominance",
        (
            "FairFace contributes 97,698 images, approximately 72.35% "
            "of the combined 135,039-image audit population."
        ),
        "Finding 01",
    )

with f2:

    research_card(
        "LFW identity imbalance",
        (
            "LFW contains 5,749 identities, but 4,069 identities "
            "(70.78%) appear only once while a small number contribute "
            "many images."
        ),
        "Finding 02",
    )

with f3:

    research_card(
        "Label availability differs",
        (
            "FairFace and UTKFace contain native demographic labels. "
            "LFW demographic information used in this project is "
            "derived and therefore carries additional provenance and "
            "measurement uncertainty."
        ),
        "Finding 03",
    )

st.info(
    """
    Dataset balance is not only a question of image counts. Identity
    depth, label availability, visual conditions and measurement
    reliability can all affect how a fairness audit should be
    interpreted.
    """
)

st.divider()


# ============================================================
# MAIN TABS
# ============================================================

(
    representation_tab,
    integrity_tab,
    detection_tab,
    visual_tab,
    measurement_tab,
    shift_tab,
    protocol_tab,
) = st.tabs(
    [
        "Representation",
        "Data Integrity",
        "Detection & Alignment",
        "Visual Conditions",
        "Skin-tone Measurement",
        "Cross-dataset Shift",
        "LFW Verification Protocol",
    ]
)


# ============================================================
# REPRESENTATION
# ============================================================

with representation_tab:

    section_header(
        "Representation",
        (
            "Demographic composition, identity depth, subgroup support "
            "and label availability."
        ),
    )

    (
        composition_subtab,
        identity_subtab,
        intersection_subtab,
        prevalence_subtab,
        missingness_subtab,
    ) = st.tabs(
        [
            "Composition",
            "Identity structure",
            "Intersectional support",
            "Target prevalence",
            "Missingness",
        ]
    )


    # --------------------------------------------------------
    # COMPOSITION
    # --------------------------------------------------------

    with composition_subtab:

        prepared_summary = (
            prepare_dataset_names(
                dataset_summary
            )
        )

        if not prepared_summary.empty:

            fig = px.bar(
                prepared_summary,
                x="Dataset",
                y="n_images",
                text="n_images",
                color="Dataset",
                color_discrete_map=DATASET_COLOURS,
                title="Combined dataset composition",
            )

            fig.update_layout(
                showlegend=False,
                height=390,
                xaxis_title="",
                yaxis_title="Images",
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20,
                ),
            )

            fig.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        selected_dataset = st.segmented_control(
            "Inspect dataset",
            [
                "FairFace",
                "UTKFace",
                "LFW",
            ],
            default="FairFace",
        )

        key = {
            "FairFace": "fairface",
            "UTKFace": "utkface",
            "LFW": "lfw",
        }[
            selected_dataset
        ]

        left, right = st.columns(2)

        with left:

            gender = gender_data[
                gender_data[
                    "dataset"
                ].eq(key)
            ]

            if not gender.empty:

                fig = px.bar(
                    gender,
                    x="category",
                    y="n_images",
                    color="category",
                    color_discrete_sequence=PALETTE,
                    title="Gender composition",
                )

                fig.update_layout(
                    showlegend=False,
                    height=360,
                    xaxis_title="",
                    yaxis_title="Images",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        with right:

            age = age_data[
                age_data[
                    "dataset"
                ].eq(key)
            ]

            if not age.empty:

                fig = px.bar(
                    age,
                    x="category",
                    y="n_images",
                    color="category",
                    color_discrete_sequence=PALETTE,
                    title="Age-band composition",
                )

                fig.update_layout(
                    showlegend=False,
                    height=360,
                    xaxis_title="",
                    yaxis_title="Images",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        race = race_data[
            race_data[
                "dataset"
            ].eq(key)
        ]

        if not race.empty:

            fig = px.bar(
                race,
                x="category",
                y="n_images",
                color="category",
                color_discrete_sequence=PALETTE,
                title="Race / ethnicity label composition",
            )

            fig.update_layout(
                showlegend=False,
                height=410,
                xaxis_title="",
                yaxis_title="Images",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        availability = availability_data[
            availability_data[
                "dataset"
            ].eq(key)
        ].copy()

        if not availability.empty:

            with st.expander(
                "Label availability",
                expanded=False,
            ):

                display = availability[
                    [
                        "measure",
                        "n_total",
                        "n_available",
                        "n_missing_or_unknown",
                        "pct_available",
                        "pct_missing_or_unknown",
                    ]
                ].copy()

                display.columns = [
                    "Measure",
                    "Total",
                    "Available",
                    "Missing / unknown",
                    "Available (%)",
                    "Missing / unknown (%)",
                ]

                show_table(
                    display
                )


    # --------------------------------------------------------
    # IDENTITY STRUCTURE
    # --------------------------------------------------------

    with identity_subtab:

        support = prepare_dataset_names(
            identity_support
        )

        if not support.empty:

            display = support[
                [
                    "Dataset",
                    "n_images",
                    "n_unique_identities",
                    "n_singleton_identities",
                    "n_repeated_identities",
                    "max_images_per_identity",
                    "subject_id_available",
                ]
            ].copy()

            display.columns = [
                "Dataset",
                "Images",
                "Unique identities",
                "Singleton identities",
                "Repeated identities",
                "Maximum images / identity",
                "Identity labels available",
            ]

            show_table(
                display
            )

        lfw_summary = (
            identity_summary[
                identity_summary[
                    "dataset"
                ].eq("lfw")
            ]
        )

        if not lfw_summary.empty:

            row = (
                lfw_summary
                .iloc[0]
            )

            metric_row(
                [
                    (
                        "LFW identities",
                        f"{int(row['n_identities']):,}",
                    ),
                    (
                        "Singleton identities",
                        f"{row['pct_singleton_identities']:.2f}%",
                    ),
                    (
                        "Median images / identity",
                        f"{row['median_images_per_identity']:.0f}",
                    ),
                    (
                        "Maximum images / identity",
                        f"{int(row['max_images_per_identity']):,}",
                    ),
                ]
            )

        depth = identity_distribution[
            identity_distribution[
                "dataset"
            ].eq("lfw")
        ].copy()

        if not depth.empty:

            fig = px.scatter(
                depth,
                x="n_images_per_identity",
                y="n_identities",
                size="images_contributed",
                hover_data=[
                    "pct_identities",
                    "pct_images_contributed",
                ],
                color_discrete_sequence=[
                    WITS_BLUE
                ],
                title="LFW identity-depth distribution",
            )

            fig.update_xaxes(
                type="log",
                title="Images per identity (log scale)",
            )

            fig.update_yaxes(
                title="Number of identities",
            )

            fig.update_layout(
                height=430,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.info(
            """
            LFW has a pronounced identity long tail. Image-level counts
            therefore should not be interpreted as uniform identity
            coverage.
            """
        )


    # --------------------------------------------------------
    # INTERSECTIONAL SUPPORT
    # --------------------------------------------------------

    with intersection_subtab:

        status = prepare_dataset_names(
            intersection_status
        )

        if not status.empty:

            melted = status.melt(
                id_vars=[
                    "Dataset",
                    "intersection",
                ],
                value_vars=[
                    "n_reportable_cells",
                    "n_low_confidence_cells",
                    "n_suppressed_cells",
                ],
                var_name="Status",
                value_name="Cells",
            )

            melted["Status"] = (
                melted["Status"]
                .replace(
                    {
                        "n_reportable_cells": "Reportable",
                        "n_low_confidence_cells": "Low confidence",
                        "n_suppressed_cells": "Suppressed",
                    }
                )
            )

            fig = px.bar(
                melted,
                x="intersection",
                y="Cells",
                color="Status",
                facet_col="Dataset",
                barmode="stack",
                color_discrete_map={
                    "Reportable": WITS_BLUE,
                    "Low confidence": WITS_GOLD,
                    "Suppressed": GREY,
                },
                title="Intersectional subgroup support",
            )

            fig.update_layout(
                height=470,
                xaxis_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.caption(
            """
            Reporting rule used in the audit: n ≥ 100 reportable;
            30–99 low confidence; n < 30 suppressed.
            """
        )

        selected = st.selectbox(
            "Inspect dataset",
            [
                "FairFace",
                "UTKFace",
                "LFW",
            ],
            key="intersection_dataset",
        )

        selected_key = {
            "FairFace": "fairface",
            "UTKFace": "utkface",
            "LFW": "lfw",
        }[
            selected
        ]

        detail = intersection_data[
            intersection_data[
                "dataset"
            ].eq(selected_key)
        ].copy()

        if not detail.empty:

            show_table(
                detail[
                    [
                        "intersection",
                        "cell_label",
                        "n_images",
                        "percentage_reported",
                        "reporting_status",
                    ]
                ].sort_values(
                    [
                        "intersection",
                        "n_images",
                    ],
                    ascending=[
                        True,
                        False,
                    ],
                ),
                height=450,
            )


    # --------------------------------------------------------
    # TARGET PREVALENCE
    # --------------------------------------------------------

    with prevalence_subtab:

        overall = prepare_dataset_names(
            base_rate_overall
        )

        available = overall[
            overall[
                "base_rate_pct_reported"
            ].notna()
        ].copy()

        if not available.empty:

            fig = px.bar(
                available,
                x="Dataset",
                y="base_rate_pct_reported",
                text="base_rate_pct_reported",
                color="Dataset",
                color_discrete_map=DATASET_COLOURS,
                title="Female prevalence among eligible images",
            )

            fig.update_layout(
                showlegend=False,
                height=380,
                yaxis_title="Female prevalence (%)",
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.caption(
            """
            The prevalence values describe label composition. They are
            not recognition accuracy or fairness metrics.
            """
        )

        subgroup_choice = st.radio(
            "Subgroup breakdown",
            [
                "Race",
                "Age",
            ],
            horizontal=True,
        )

        if subgroup_choice == "Race":

            subgroup = prepare_dataset_names(
                base_rate_race
            )

        else:

            subgroup = prepare_dataset_names(
                base_rate_age
            )

        subgroup = subgroup[
            subgroup[
                "base_rate_pct_reported"
            ].notna()
        ]

        if not subgroup.empty:

            fig = px.bar(
                subgroup,
                x="subgroup",
                y="base_rate_pct_reported",
                color="Dataset",
                barmode="group",
                color_discrete_map=DATASET_COLOURS,
                title=(
                    f"Female prevalence by "
                    f"{subgroup_choice.lower()}"
                ),
            )

            fig.update_layout(
                height=410,
                xaxis_title="",
                yaxis_title="Female prevalence (%)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    # --------------------------------------------------------
    # MISSINGNESS
    # --------------------------------------------------------

    with missingness_subtab:

        miss = prepare_dataset_names(
            missingness
        )

        if not miss.empty:

            heatmap = (
                miss.pivot(
                    index="label",
                    columns="Dataset",
                    values="pct_missing_or_unknown",
                )
            )

            fig = px.imshow(
                heatmap,
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale=[
                    [
                        0.0,
                        "#EAF3FA",
                    ],
                    [
                        0.5,
                        WITS_GOLD,
                    ],
                    [
                        1.0,
                        WITS_NAVY,
                    ],
                ],
                title="Missing or unknown labels (%)",
            )

            fig.update_layout(
                height=400,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            with st.expander(
                "Detailed missingness table"
            ):

                show_table(
                    miss,
                    height=450,
                )

        st.warning(
            """
            Label availability determines which fairness questions can
            be answered. Missing or derived demographic information
            should not be treated as equivalent to independently
            verified ground truth.
            """
        )

        if not missingness_patterns.empty:

            with st.expander(
                "Missingness patterns"
            ):

                show_table(
                    prepare_dataset_names(
                        missingness_patterns
                    )
                )

        if not affected_analyses.empty:

            with st.expander(
                "Analyses affected by label availability"
            ):

                show_table(
                    prepare_dataset_names(
                        affected_analyses
                    ),
                    height=450,
                )


# ============================================================
# DATA INTEGRITY
# ============================================================

with integrity_tab:

    section_header(
        "Data integrity",
        (
            "Structural checks for corrupt records, label parsing and "
            "duplicate candidates."
        ),
    )

    rows = []

    for dataset_key, frame in master.groupby(
        "dataset"
    ):

        rows.append(
            {
                "Dataset": pretty_dataset(
                    dataset_key
                ),
                "Images": len(frame),
                "Corrupt": bool_count(
                    frame,
                    "corrupt",
                ),
                "Label parse failures": bool_count(
                    frame,
                    "parse_failed",
                ),
                "Rows in exact-duplicate group": int(
                    frame[
                        "exact_dup_group"
                    ].notna()
                    .sum()
                ),
                "Exact-duplicate groups": int(
                    frame[
                        "exact_dup_group"
                    ].dropna()
                    .nunique()
                ),
                "Confirmed near-duplicate rows": int(
                    frame[
                        "near_dup_group"
                    ].notna()
                    .sum()
                ),
                "Cross-dataset duplicate flags": bool_count(
                    frame,
                    "cross_dataset_dup",
                ),
            }
        )

    show_table(
        pd.DataFrame(
            rows
        )
    )

    left, right = st.columns(2)

    with left:

        if "near_dup_status" in master.columns:

            status = (
                master[
                    "near_dup_status"
                ]
                .astype("string")
                .fillna(
                    "Not applicable"
                )
                .value_counts()
                .rename_axis(
                    "Status"
                )
                .reset_index(
                    name="Images"
                )
            )

            fig = px.bar(
                status,
                x="Status",
                y="Images",
                color_discrete_sequence=[
                    WITS_BLUE
                ],
                title="Near-duplicate status",
            )

            fig.update_layout(
                height=360,
                xaxis_title="",
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with right:

        if (
            "cross_dataset_dup_status"
            in master.columns
        ):

            status = (
                master[
                    "cross_dataset_dup_status"
                ]
                .astype("string")
                .fillna(
                    "Not applicable"
                )
                .value_counts()
                .rename_axis(
                    "Status"
                )
                .reset_index(
                    name="Images"
                )
            )

            fig = px.bar(
                status,
                x="Status",
                y="Images",
                color_discrete_sequence=[
                    WITS_GOLD
                ],
                title=(
                    "Cross-dataset duplicate status"
                ),
            )

            fig.update_layout(
                height=360,
                xaxis_title="",
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.info(
        """
        Duplicate screening protects later experiments from artificially
        inflated diversity and possible evaluation leakage. Candidate
        duplicate status is not equivalent to a manually confirmed
        duplicate.
        """
    )


# ============================================================
# DETECTION & ALIGNMENT
# ============================================================

with detection_tab:

    section_header(
        "Detection & alignment",
        (
            "Preprocessing reliability is evaluated before recognition "
            "results are interpreted."
        ),
    )

    rows = []

    for dataset_key, frame in master.groupby(
        "dataset"
    ):

        alignment_known = int(
            frame[
                "alignment_valid"
            ]
            .notna()
            .sum()
        )

        alignment_valid = bool_count(
            frame,
            "alignment_valid",
        )

        confidence = (
            frame[
                "det_confidence"
            ]
            .dropna()
        )

        rows.append(
            {
                "Dataset": pretty_dataset(
                    dataset_key
                ),
                "Images": len(frame),
                "No detection": bool_count(
                    frame,
                    "no_detection",
                ),
                "Multiple detections": bool_count(
                    frame,
                    "multi_detection",
                ),
                "Median detection confidence": (
                    float(
                        confidence.median()
                    )
                    if not confidence.empty
                    else None
                ),
                "Alignment success (%)": (
                    100
                    * alignment_valid
                    / alignment_known
                    if alignment_known
                    else None
                ),
            }
        )

    detection_summary = (
        pd.DataFrame(
            rows
        )
    )

    show_table(
        detection_summary
    )

    plot_data = (
        detection_summary[
            [
                "Dataset",
                "Alignment success (%)",
            ]
        ]
        .dropna()
    )

    if not plot_data.empty:

        fig = px.bar(
            plot_data,
            x="Dataset",
            y="Alignment success (%)",
            color="Dataset",
            text="Alignment success (%)",
            color_discrete_map=DATASET_COLOURS,
            title="Alignment success",
        )

        fig.update_yaxes(
            range=[
                0,
                100,
            ]
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )

        fig.update_layout(
            showlegend=False,
            height=380,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.info(
        """
        A recognition error may originate before the recognition model
        if a face is missed, the wrong face is selected or the crop is
        poorly aligned.
        """
    )

    with st.expander(
        "Detailed detector evidence"
    ):

        show_table(
            load_detection_integrity_by_dataset()
        )

        show_table(
            load_confidence_summary()
        )

        show_table(
            load_subgroup_detection_failures()
        )

        show_table(
            load_alignment_failures_by_subgroup()
        )

    with st.expander(
        "Detection and alignment figures"
    ):

        figure_files = [
            "alignment_outcomes.png",
            "confidence_distributions.png",
            "detector_stage_funnel.png",
            "failure_examples_contact_sheet.png",
            "utkface_age_band_confidence.png",
            "utkface_age_band_failure_rates.png",
            "utkface_gender_confidence.png",
            "utkface_gender_failure_rates.png",
            "utkface_race_confidence.png",
            "utkface_race_failure_rates.png",
            "visual_review_categories.png",
        ]

        selected = st.selectbox(
            "Figure",
            figure_files,
            key="detection_figure",
        )

        show_image(
            detection_figure(
                selected
            )
        )

    with st.expander(
        "Alignment QA examples"
    ):

        qa_manifest = (
            load_alignment_qa_manifest()
        )

        if not qa_manifest.empty:

            selected_dataset = (
                st.selectbox(
                    "Dataset",
                    sorted(
                        qa_manifest[
                            "dataset"
                        ]
                        .unique()
                    ),
                    key="alignment_dataset",
                )
            )

            subset = qa_manifest[
                qa_manifest[
                    "dataset"
                ].eq(
                    selected_dataset
                )
            ]

            paths = [
                alignment_qa_sample(
                    name
                )
                for name
                in subset[
                    "filename"
                ]
            ]

            image_grid(
                paths,
                columns=4,
            )

        show_image(
            alignment_qa_artifact(
                "contact_sheets",
                "alignment_contact_sheet.png",
            )
        )

    limitations = (
        load_detector_limitations_text()
    )

    if limitations:

        with st.expander(
            "Detection limitations"
        ):

            st.markdown(
                limitations
            )


# ============================================================
# VISUAL CONDITIONS
# ============================================================

with visual_tab:

    section_header(
        "Visual conditions",
        (
            "Lighting, sharpness, scale and pose are evaluated as "
            "potential confounders."
        ),
    )

    visual_metrics = {
        "Brightness": "brightness",
        "Contrast": "contrast",
        "Sharpness": "sharpness",
        "Face-area ratio": "face_area_ratio",
        "Yaw": "yaw",
        "Pitch": "pitch",
        "Roll": "roll",
    }

    selected_label = (
        st.selectbox(
            "Metric",
            list(
                visual_metrics.keys()
            ),
            key="visual_metric",
        )
    )

    metric = visual_metrics[
        selected_label
    ]

    values = (
        master[
            [
                "Dataset",
                metric,
            ]
        ]
        .dropna()
    )

    if not values.empty:

        if len(values) > 30000:

            plot_values = values.sample(
                30000,
                random_state=26041,
            )

        else:

            plot_values = values

        fig = px.box(
            plot_values,
            x="Dataset",
            y=metric,
            color="Dataset",
            points=False,
            color_discrete_map=DATASET_COLOURS,
            title=selected_label,
        )

        fig.update_layout(
            showlegend=False,
            height=430,
            xaxis_title="",
            yaxis_title=selected_label,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.warning(
        """
        Differences in lighting, pose or image quality can confound
        subgroup comparisons. Recognition disparities should therefore
        not automatically be attributed to demographic representation
        alone.
        """
    )

    with st.expander(
        "Detailed visual-quality evidence"
    ):

        show_table(
            load_vis_status_counts()
        )

        show_table(
            load_vis_summary_by_dataset()
        )

        show_table(
            load_vis_subgroup_summary()
        )

        show_table(
            load_vis_subgroup_omnibus()
        )

        show_table(
            load_vis_qc_flag_counts()
        )

    with st.expander(
        "Visual-quality figures"
    ):

        figure_kind = st.radio(
            "Figure type",
            [
                "Histogram",
                "Boxplot by skin-tone group",
            ],
            horizontal=True,
            key="visual_figure_kind",
        )

        metric_choice = st.selectbox(
            "Measurement",
            [
                "brightness_L_mean",
                "contrast_L_sd",
                "face_area_ratio",
                "face_short_side_px",
                "illumination_asymmetry",
                "pitch_deg",
                "roll_deg",
                "sharpness_lapvar",
                "yaw_deg",
            ],
            key="visual_phase_metric",
        )

        if figure_kind == "Histogram":

            filename = (
                f"VIS_hist_"
                f"{metric_choice}.png"
            )

        else:

            filename = (
                f"VIS_box_"
                f"{metric_choice}"
                f"_by_skin_group_ita.png"
            )

        show_image(
            vis_figure(
                filename
            )
        )

    brightness_skin = (
        load_brightness_vs_skin_tone()
    )

    if not brightness_skin.empty:

        with st.expander(
            "Brightness and skin-tone measurement"
        ):

            sample = (
                brightness_skin
                .sample(
                    n=min(
                        8000,
                        len(
                            brightness_skin
                        ),
                    ),
                    random_state=26041,
                )
            )

            fig = px.scatter(
                sample,
                x="brightness_L_mean",
                y="skin_L_star",
                color="skin_group_ita",
                facet_col="dataset",
                opacity=0.35,
                height=430,
                labels={
                    "brightness_L_mean": (
                        "Image brightness"
                    ),
                    "skin_L_star": (
                        "Measured skin lightness"
                    ),
                    "skin_group_ita": (
                        "ITA group"
                    ),
                },
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            show_table(
                load_vis_brightness_vs_skin()
            )

            st.caption(
                """
                A relationship between image brightness and measured
                skin lightness does not by itself prove a measurement
                error. The purpose of this analysis is to expose a
                possible imaging confound rather than assume it away.
                """
            )


# ============================================================
# SKIN-TONE MEASUREMENT
# ============================================================

with measurement_tab:

    section_header(
        "Skin-tone measurement",
        (
            "Continuous image-derived measurements are analysed "
            "separately from demographic categories."
        ),
    )

    metric_row(
        [
            (
                "Valid ROI",
                f"{bool_count(master, 'skin_roi_valid'):,}",
            ),
            (
                "Invalid ROI",
                (
                    f"{int(master['skin_roi_valid'].eq(False).sum()):,}"
                ),
            ),
            (
                "ITA undefined",
                f"{bool_count(master, 'ita_undefined'):,}",
            ),
            (
                "Classifier outputs",
                (
                    f"{int(master['fitz_classifier'].notna().sum()):,}"
                ),
            ),
        ]
    )

    measurements = {
        "L* lightness": "L_star",
        "ITA": "ITA",
        "Hue angle": "hue_angle",
        "a*": "a_star",
        "b*": "b_star",
    }

    selected = st.selectbox(
        "Continuous measurement",
        list(
            measurements.keys()
        ),
        key="skin_measurement",
    )

    column = measurements[
        selected
    ]

    measurement_values = (
        master[
            [
                "Dataset",
                column,
            ]
        ]
        .dropna()
    )

    if not measurement_values.empty:

        plot_values = (
            measurement_values
            .sample(
                min(
                    40000,
                    len(
                        measurement_values
                    ),
                ),
                random_state=26041,
            )
        )

        fig = px.histogram(
            plot_values,
            x=column,
            color="Dataset",
            nbins=60,
            opacity=0.6,
            color_discrete_map=DATASET_COLOURS,
            title=selected,
        )

        fig.update_layout(
            barmode="overlay",
            height=410,
            yaxis_title="Images",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.info(
        """
        Demographic category and skin-tone measurement are not treated
        as interchangeable variables. ITA is retained as a separate
        image-derived measurement.
        """
    )

    with st.expander(
        "Measurement reliability"
    ):

        show_table(
            load_mea_status_counts()
        )

        show_table(
            load_mea_summary_by_dataset()
        )

        show_table(
            load_mea_reliability_agreement()
        )

        show_table(
            load_mea_boundary_sensitivity()
        )

        show_table(
            load_mea_roi_failure_by_group()
        )

    distribution = (
        load_mea_skin_group_distribution()
    )

    if not distribution.empty:

        if {
            "dataset",
            "category",
            "pct",
        }.issubset(
            distribution.columns
        ):

            fig = px.bar(
                distribution,
                x="category",
                y="pct",
                color="dataset",
                barmode="group",
                color_discrete_map=DATASET_COLOURS,
                title="ITA group distribution",
            )

            fig.update_layout(
                height=400,
                xaxis_title="",
                yaxis_title="Share of images (%)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    with st.expander(
        "Measurement QA figures"
    ):

        metric_choice = st.selectbox(
            "Figure",
            [
                "ITA",
                "skin_L_star",
                "skin_chroma",
                "skin_hue_deg",
            ],
            key="measurement_figure",
        )

        show_image(
            mea_figure(
                f"MEA_hist_"
                f"{metric_choice}.png"
            )
        )

        show_image(
            mea_qa_image(
                "MEA_roi_overlay_sample.png"
            ),
            "Skin ROI QA sample",
        )

    limitations = (
        load_mea_limitations_text()
    )

    if limitations:

        with st.expander(
            "Measurement limitations"
        ):

            st.markdown(
                limitations
            )


# ============================================================
# CROSS-DATASET SHIFT
# ============================================================

with shift_tab:

    section_header(
        "Cross-dataset shift",
        (
            "The three datasets are compared to determine whether they "
            "represent equivalent demographic and image-condition "
            "distributions."
        ),
    )

    categorical = (
        load_shift_categorical_distributions()
    )

    if not categorical.empty:

        if {
            "dataset",
            "variable",
            "category",
            "pct",
        }.issubset(
            categorical.columns
        ):

            variable = st.selectbox(
                "Variable",
                sorted(
                    categorical[
                        "variable"
                    ].unique()
                ),
                key="shift_variable",
            )

            subset = categorical[
                categorical[
                    "variable"
                ].eq(variable)
            ]

            fig = px.bar(
                subset,
                x="category",
                y="pct",
                color="dataset",
                barmode="group",
                color_discrete_map=DATASET_COLOURS,
                title=(
                    f"{variable} distribution "
                    f"by dataset"
                ),
            )

            fig.update_layout(
                height=400,
                xaxis_title="",
                yaxis_title="Share of images (%)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.warning(
        """
        A combined dataset can remain source-dependent even after
        demographic counts are improved. Cross-dataset shift is
        therefore documented rather than assumed to disappear after
        merging.
        """
    )

    with st.expander(
        "Detailed shift analysis"
    ):

        show_table(
            load_shift_analysis_set_counts()
        )

        show_table(
            load_shift_categorical_tests()
        )

        show_table(
            load_shift_continuous_summary()
        )

        show_table(
            load_shift_continuous_tests()
        )

        show_table(
            load_shift_nominal_balance()
        )

        show_table(
            load_shift_effective_balance_summary()
        )

        show_table(
            load_shift_effective_balance_tests()
        )

        show_table(
            load_shift_identity_depth()
        )

        show_table(
            load_shift_ita_within_native_race()
        )

    show_image(
        shift_figure(
            "SHIFT_skin_group_by_dataset.png"
        )
    )

    limitations = (
        load_shift_limitations_text()
    )

    if limitations:

        with st.expander(
            "Interpretation limits"
        ):

            st.markdown(
                limitations
            )


# ============================================================
# LFW VERIFICATION PROTOCOL
# ============================================================

with protocol_tab:

    section_header(
        "LFW verification protocol",
        (
            "Pair-list integrity and pair composition are audited "
            "before recognition performance is interpreted."
        ),
    )

    show_table(
        load_lfw_pair_file_checks()
    )

    st.subheader(
        "Pair resolution"
    )

    show_table(
        load_lfw_pair_resolution_counts()
    )

    yoking = (
        load_lfw_impostor_yoking_counts()
    )

    if not yoking.empty:

        fig = px.bar(
            yoking,
            x="view",
            y="pct",
            color="yoking",
            barmode="stack",
            facet_col="split",
            labels={
                "pct": (
                    "Share of impostor pairs (%)"
                ),
                "view": "View",
                "yoking": "Pair category",
            },
            title=(
                "Impostor-pair skin-tone composition"
            ),
            height=420,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with st.expander(
        "Detailed protocol evidence"
    ):

        show_table(
            load_lfw_composition_images_vs_pairs()
        )

        show_table(
            load_lfw_composition_tests()
        )

        show_table(
            load_lfw_impostor_same_group_vs_random()
        )

        show_table(
            load_lfw_genuine_visual_difference_by_skin_group()
        )

        show_table(
            load_lfw_genuine_visual_difference_tests()
        )

        show_table(
            load_lfw_visual_difference_median_by_yoking()
        )

    protocol_status = (
        load_lfw_status()
    )

    protocol_rules = (
        load_lfw_protocol_rules()
    )

    if protocol_status or protocol_rules:

        with st.expander(
            "Protocol configuration"
        ):

            if protocol_status:
                st.json(
                    protocol_status
                )

            if protocol_rules:
                st.json(
                    protocol_rules
                )
