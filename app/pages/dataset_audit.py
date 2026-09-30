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
    apply_wits_theme,
    blue_callout,
    gold_callout,
    hero,
    section_label,
)


# ============================================================
# STYLE
# ============================================================

apply_wits_theme()

WITS_BLUE = "#0067B1"
WITS_NAVY = "#002B5C"
WITS_GOLD = "#F4C542"
LIGHT_BLUE = "#5DADE2"
GREY = "#9AA5B1"

DATASET_COLOURS = {
    "LFW": WITS_NAVY,
    "FairFace": WITS_BLUE,
    "UTKFace": WITS_GOLD,
}

DATASET_LABELS = {
    "lfw": "LFW",
    "fairface": "FairFace",
    "utkface": "UTKFace",
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
# LOAD DATA
# ============================================================

try:
    master = load_master().copy()
except Exception as exc:
    st.error(f"Could not load master audit table: {exc}")
    st.stop()

master["Dataset"] = master["dataset"].map(dataset_label)

rep001_summary = load_rep001_dataset_summary()
rep001_gender = load_rep001_gender()
rep001_age = load_rep001_age()
rep001_race = load_rep001_race()
rep001_availability = load_rep001_availability()

rep002_identity = load_rep002_identity_support()

rep003_summary = load_rep003_identity_summary()
rep003_distribution = load_rep003_identity_distribution()

rep004_intersections = load_rep004_intersections()
rep004_status = load_rep004_status_summary()

rep005_overall = load_rep005_overall()
rep005_race = load_rep005_race()
rep005_age = load_rep005_age()

rep006_missingness = load_rep006_missingness()
rep006_patterns = load_rep006_patterns()
rep006_affected = load_rep006_affected_analyses()


# ============================================================
# HELPERS
# ============================================================

def pretty_dataset(value: str) -> str:
    return DATASET_LABELS.get(
        str(value).lower(),
        str(value),
    )


def bool_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0

    return int(
        frame[column]
        .eq(True)
        .sum()
    )


def phase1_table(df: pd.DataFrame, empty_message: str = "Not available.") -> None:
    if df.empty:
        st.info(empty_message)
    else:
        st.dataframe(df, hide_index=True, use_container_width=True)


def show_phase1_image(path, caption: str | None = None) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not available: {path.name}")


def phase1_image_grid(paths: list, columns: int = 5) -> None:
    rows = [paths[i:i + columns] for i in range(0, len(paths), columns)]
    for row in rows:
        cols = st.columns(columns)
        for col, path in zip(cols, row):
            with col:
                if path.exists():
                    st.image(str(path), use_container_width=True, caption=path.stem.replace(".aligned", ""))


def format_pct(value) -> str:
    if pd.isna(value):
        return "Unavailable"

    return f"{float(value):.2f}%"


def prepare_rep_dataset_names(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()

    if "dataset" in output.columns:
        output["Dataset"] = (
            output["dataset"]
            .map(pretty_dataset)
        )

    return output


def source_caption(issue: str, filename: str) -> None:
    st.caption(
        f"Source: results/representation/{issue}/{filename}"
    )


# ============================================================
# PAGE HEADER
# ============================================================

hero(
    "Forensic Dataset Audit",
    """
    Phase 1 examines the datasets and preprocessing pipeline before
    evaluating recognition performance. The objective is to identify
    plausible sources of bias, measurement uncertainty and technical
    failure that can later be tested against model error.
    """,
    eyebrow="Stage 1 · Evidence before model evaluation",
)

gold_callout(
    """
    <strong>The audit is not the final fairness result.</strong>
    Representation imbalance, image-quality differences and measurement
    limitations identify hypotheses. The baseline recognition experiment
    must determine whether those conditions actually correspond to
    unequal false-match or false-non-match rates.
    """
)


# ============================================================
# PROJECT-LEVEL SUMMARY
# ============================================================

section_label("Current audit evidence")

top_1, top_2, top_3, top_4 = st.columns(4)

top_1.metric(
    "Images audited",
    f"{len(master):,}",
)

top_2.metric(
    "Datasets",
    f"{master['dataset'].nunique():,}",
)

top_3.metric(
    "LFW identities",
    f"{master.loc[master['dataset'].eq('lfw'), 'subject_id'].nunique():,}",
)

top_4.metric(
    "Valid Phase 0 skin ROIs",
    f"{bool_count(master, 'skin_roi_valid'):,}",
)


# ============================================================
# MAIN TABS
# ============================================================

(
    overview_tab,
    representation_tab,
    integrity_tab,
    detection_tab,
    visual_tab,
    measurement_tab,
    shift_tab,
    lfw_protocol_tab,
) = st.tabs(
    [
        "Overview",
        "Representation",
        "Integrity & Duplicates",
        "Detection & Alignment",
        "Visual Conditions",
        "Measurement",
        "Cross-dataset shift",
        "LFW protocol",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    section_label("Phase 1 map")
    st.subheader("Where the project currently stands")

    audit_matrix = pd.DataFrame(
        {
            "Audit layer": [
                "Representation",
                "Image / label integrity",
                "Duplicate screening",
                "Detection & alignment",
                "Visual conditions",
                "Skin measurement",
                "Baseline recognition",
                "Model fairness audit",
                "Mitigation",
            ],
            "Status": [
                "Complete",
                "Available",
                "Available",
                "Available",
                "Available",
                "Validation in progress",
                "Next stage",
                "Pending baseline",
                "Pending fairness result",
            ],
            "Primary evidence": [
                "REP-001 to REP-006",
                "Canonical Phase 0 metadata",
                "Duplicate metadata and review records",
                "Detection/alignment diagnostics",
                "Master annotation table",
                "Phase 0 values + MEA-001 validation",
                "Not yet generated",
                "Not yet generated",
                "Not yet selected",
            ],
        }
    )

    st.dataframe(
        audit_matrix,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    section_label("Dataset composition")
    st.subheader("135,039 images across three datasets")

    dataset_summary = prepare_rep_dataset_names(
        rep001_summary
    )

    if not dataset_summary.empty:

        fig = px.bar(
            dataset_summary,
            x="Dataset",
            y="n_images",
            text="n_images",
            color="Dataset",
            color_discrete_map=DATASET_COLOURS,
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            yaxis_title="Images",
            height=380,
        )

        fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        summary_display = dataset_summary[
            [
                "Dataset",
                "n_images",
                "pct_all_images",
            ]
        ].copy()

        summary_display.columns = [
            "Dataset",
            "Images",
            "Share of combined audit (%)",
        ]

        st.dataframe(
            summary_display,
            hide_index=True,
            use_container_width=True,
        )

        source_caption(
            "REP-001",
            "rep001_dataset_summary.csv",
        )

    st.divider()

    section_label("Key representation findings")
    st.subheader("What has already been established?")

    finding_1, finding_2 = st.columns(2)

    with finding_1:

        st.markdown(
            """
            ### Dataset composition

            FairFace contributes **97,698 images**, representing
            approximately **72.35%** of the combined three-dataset
            audit population.

            UTKFace contributes **24,108 images** and LFW contributes
            **13,233 images**.

            This combined corpus is therefore not balanced by dataset
            size.
            """
        )

        st.markdown(
            """
            ### LFW identity structure

            LFW contains **5,749 identities**.

            **4,069 identities (70.78%)** are represented by only one
            image, while **1,680 identities (29.22%)** have repeated
            images.

            One identity contributes as many as **530 images**.
            """
        )

    with finding_2:

        st.markdown(
            """
            ### Demographic availability

            FairFace provides complete gender, race and harmonised
            age-band labels in the current audit.

            UTKFace has demographic information for **24,104 of
            24,108 images**, leaving four parsing/missing-label cases.

            Standard LFW metadata in this project does **not** provide
            gender, race or age labels.
            """
        )

        st.markdown(
            """
            ### Important consequence

            The LFW demographic-label gap currently prevents direct
            LFW subgroup fairness claims based on those attributes.

            A defensible demographic-provenance strategy is therefore
            required before LFW model error can be reported by gender,
            race, age or intersectional group.
            """
        )


# ============================================================
# REPRESENTATION
# ============================================================

with representation_tab:

    section_label("REP-001 to REP-006")
    st.subheader("Registered representation audit")

    (
        composition_subtab,
        identity_subtab,
        intersections_subtab,
        base_rate_subtab,
        missingness_subtab,
    ) = st.tabs(
        [
            "Composition",
            "Identity",
            "Intersections",
            "Base Rates",
            "Missingness",
        ]
    )


    # --------------------------------------------------------
    # COMPOSITION
    # --------------------------------------------------------

    with composition_subtab:

        st.subheader("Image-level demographic composition")

        selected_dataset = st.selectbox(
            "Dataset",
            [
                "FairFace",
                "LFW",
                "UTKFace",
            ],
            key="rep_composition_dataset",
        )

        selected_key = {
            "FairFace": "fairface",
            "LFW": "lfw",
            "UTKFace": "utkface",
        }[selected_dataset]

        composition_left, composition_right = st.columns(2)

        with composition_left:

            gender = rep001_gender[
                rep001_gender["dataset"].eq(
                    selected_key
                )
            ].copy()

            if not gender.empty:

                fig = px.bar(
                    gender,
                    x="category",
                    y="n_images",
                    text="n_images",
                    color="category",
                    color_discrete_sequence=PALETTE,
                    title=f"{selected_dataset}: gender composition",
                )

                fig.update_layout(
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Images",
                    height=390,
                )

                fig.update_traces(
                    textposition="outside",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        with composition_right:

            age = rep001_age[
                rep001_age["dataset"].eq(
                    selected_key
                )
            ].copy()

            if not age.empty:

                fig = px.bar(
                    age,
                    x="category",
                    y="n_images",
                    text="n_images",
                    color="category",
                    color_discrete_sequence=PALETTE,
                    title=f"{selected_dataset}: harmonised age bands",
                )

                fig.update_layout(
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Images",
                    height=390,
                )

                fig.update_traces(
                    textposition="outside",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        race = rep001_race[
            rep001_race["dataset"].eq(
                selected_key
            )
        ].copy()

        if not race.empty:

            fig = px.bar(
                race,
                x="category",
                y="n_images",
                text="n_images",
                color="category",
                color_discrete_sequence=PALETTE,
                title=f"{selected_dataset}: native race labels",
            )

            fig.update_layout(
                showlegend=False,
                xaxis_title="",
                yaxis_title="Images",
                height=420,
            )

            fig.update_traces(
                textposition="outside",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        availability = rep001_availability[
            rep001_availability[
                "dataset"
            ].eq(selected_key)
        ].copy()

        if not availability.empty:

            st.subheader("Demographic availability")

            availability_display = availability[
                [
                    "measure",
                    "n_total",
                    "n_available",
                    "n_missing_or_unknown",
                    "pct_available",
                    "pct_missing_or_unknown",
                ]
            ].copy()

            availability_display.columns = [
                "Measure",
                "Total images",
                "Available",
                "Missing / unknown",
                "Available (%)",
                "Missing / unknown (%)",
            ]

            st.dataframe(
                availability_display,
                hide_index=True,
                use_container_width=True,
            )

        if selected_dataset == "LFW":

            gold_callout(
                """
                LFW's native project metadata does not currently
                provide the demographic attributes required for the
                intended subgroup fairness analysis. Missing labels
                must not be silently inferred or treated as known.
                """
            )

        source_caption(
            "REP-001",
            "composition and availability outputs",
        )


    # --------------------------------------------------------
    # IDENTITY
    # --------------------------------------------------------

    with identity_subtab:

        st.subheader("Identity availability across datasets")

        support = prepare_rep_dataset_names(
            rep002_identity
        )

        if not support.empty:

            support_display = support[
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

            support_display.columns = [
                "Dataset",
                "Images",
                "Unique identities",
                "Singleton identities",
                "Repeated identities",
                "Maximum images / identity",
                "Identity labels available",
            ]

            st.dataframe(
                support_display,
                hide_index=True,
                use_container_width=True,
            )

            source_caption(
                "REP-002",
                "rep002_identity_support.csv",
            )

        lfw_summary = rep003_summary[
            rep003_summary[
                "dataset"
            ].eq("lfw")
        ]

        if not lfw_summary.empty:

            row = lfw_summary.iloc[0]

            st.divider()
            st.subheader("LFW identity-depth audit")

            identity_1, identity_2, identity_3, identity_4 = st.columns(4)

            identity_1.metric(
                "Identities",
                f"{int(row['n_identities']):,}",
            )

            identity_2.metric(
                "Singletons",
                f"{int(row['n_singleton_identities']):,}",
                f"{row['pct_singleton_identities']:.2f}%",
            )

            identity_3.metric(
                "Median images / identity",
                f"{row['median_images_per_identity']:.0f}",
            )

            identity_4.metric(
                "Maximum images / identity",
                f"{int(row['max_images_per_identity']):,}",
            )

            st.markdown(
                f"""
                **Depth statistics**

                - Mean: **{row['mean_images_per_identity']:.2f}**
                - 75th percentile: **{row['q75_images_per_identity']:.0f}**
                - 90th percentile: **{row['p90_images_per_identity']:.0f}**
                - 95th percentile: **{row['p95_images_per_identity']:.0f}**
                - 99th percentile: **{row['p99_images_per_identity']:.0f}**
                """
            )

        lfw_depth = rep003_distribution[
            rep003_distribution[
                "dataset"
            ].eq("lfw")
        ].copy()

        if not lfw_depth.empty:

            fig = px.scatter(
                lfw_depth,
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

        blue_callout(
            """
            <strong>Interpretation:</strong> LFW has a pronounced
            identity long tail. More than 70% of identities contribute
            a single image, while a small number contribute many images.
            Therefore image-level counts should not be interpreted as
            uniform identity coverage.
            """
        )

        source_caption(
            "REP-003",
            "identity-depth outputs",
        )


    # --------------------------------------------------------
    # INTERSECTIONS
    # --------------------------------------------------------

    with intersections_subtab:

        st.subheader("Intersectional cell reliability")

        status = prepare_rep_dataset_names(
            rep004_status
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

            status_names = {
                "n_reportable_cells": "Reportable",
                "n_low_confidence_cells": "Low confidence",
                "n_suppressed_cells": "Suppressed",
            }

            melted["Status"] = (
                melted["Status"]
                .map(status_names)
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
                title="Intersectional reporting status",
            )

            fig.update_layout(
                height=470,
                xaxis_title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.markdown(
            """
            **Project reporting thresholds**

            - **n ≥ 100:** reportable
            - **30 ≤ n < 100:** low confidence
            - **n < 30:** suppressed
            """
        )

        selected_intersection_dataset = st.selectbox(
            "Inspect dataset",
            [
                "FairFace",
                "UTKFace",
            ],
            key="intersection_dataset",
        )

        selected_intersection_key = (
            "fairface"
            if selected_intersection_dataset == "FairFace"
            else "utkface"
        )

        detail = rep004_intersections[
            rep004_intersections[
                "dataset"
            ].eq(selected_intersection_key)
        ].copy()

        if not detail.empty:

            status_filter = st.multiselect(
                "Reporting status",
                sorted(
                    detail[
                        "reporting_status"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                ),
                default=sorted(
                    detail[
                        "reporting_status"
                    ]
                    .dropna()
                    .unique()
                    .tolist()
                ),
            )

            if status_filter:
                detail = detail[
                    detail[
                        "reporting_status"
                    ].isin(status_filter)
                ]

            st.dataframe(
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
                hide_index=True,
                use_container_width=True,
                height=480,
            )

        fairface_status = rep004_status[
            rep004_status["dataset"].eq(
                "fairface"
            )
        ]

        utk_status = rep004_status[
            rep004_status["dataset"].eq(
                "utkface"
            )
        ]

        fairface_flagged = int(
            fairface_status[
                "n_low_confidence_cells"
            ].sum()
            + fairface_status[
                "n_suppressed_cells"
            ].sum()
        )

        utk_flagged = int(
            utk_status[
                "n_low_confidence_cells"
            ].sum()
            + utk_status[
                "n_suppressed_cells"
            ].sum()
        )

        gold_callout(
            f"""
            <strong>Finding:</strong> all tested FairFace
            intersectional cells are reportable under the frozen
            thresholds. UTKFace contains <strong>{utk_flagged}
            flagged cells</strong> across the tested intersections,
            while FairFace contains <strong>{fairface_flagged}</strong>.
            Thin cells should not be converted into apparently precise
            subgroup percentages.
            """
        )

        source_caption(
            "REP-004",
            "intersectional composition and reporting-status outputs",
        )


    # --------------------------------------------------------
    # BASE RATES
    # --------------------------------------------------------

    with base_rate_subtab:

        st.subheader("Target prevalence")

        blue_callout(
            """
            REP-005 uses the frozen binary gender target coding:
            <strong>female = 1</strong> and <strong>male = 0</strong>.
            These values describe target prevalence. They are
            <strong>not</strong> recognition accuracy or fairness
            metrics.
            """
        )

        overall = prepare_rep_dataset_names(
            rep005_overall
        )

        available_overall = overall[
            overall[
                "base_rate_pct_reported"
            ].notna()
        ].copy()

        if not available_overall.empty:

            fig = px.bar(
                available_overall,
                x="Dataset",
                y="base_rate_pct_reported",
                text="base_rate_pct_reported",
                color="Dataset",
                color_discrete_map=DATASET_COLOURS,
                title="Female prevalence among eligible images",
            )

            fig.update_yaxes(
                title="Female prevalence (%)",
                range=[
                    0,
                    max(
                        55,
                        available_overall[
                            "base_rate_pct_reported"
                        ].max()
                        + 5,
                    ),
                ],
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
            )

            fig.update_layout(
                showlegend=False,
                height=390,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.dataframe(
            overall[
                [
                    "Dataset",
                    "n",
                    "n_y1",
                    "n_y0",
                    "base_rate_pct_reported",
                    "reporting_status",
                    "analysis_status",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )

        st.markdown(
            """
            **Overall results**

            - FairFace female prevalence: **47.00%**
            - UTKFace female prevalence: **47.80%**
            - LFW: **not available**, because eligible demographic
              labels are unavailable.
            """
        )

        st.divider()

        subgroup_choice = st.radio(
            "Subgroup analysis",
            [
                "Race",
                "Age",
            ],
            horizontal=True,
        )

        if subgroup_choice == "Race":
            subgroup_data = prepare_rep_dataset_names(
                rep005_race
            )
        else:
            subgroup_data = prepare_rep_dataset_names(
                rep005_age
            )

        subgroup_available = subgroup_data[
            subgroup_data[
                "base_rate_pct_reported"
            ].notna()
        ].copy()

        if not subgroup_available.empty:

            fig = px.bar(
                subgroup_available,
                x="subgroup",
                y="base_rate_pct_reported",
                color="Dataset",
                barmode="group",
                color_discrete_map=DATASET_COLOURS,
                title=f"Female prevalence by {subgroup_choice.lower()}",
            )

            fig.update_yaxes(
                title="Female prevalence (%)",
            )

            fig.update_xaxes(
                title="",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        source_caption(
            "REP-005",
            "target coding and base-rate outputs",
        )


    # --------------------------------------------------------
    # MISSINGNESS
    # --------------------------------------------------------

    with missingness_subtab:

        st.subheader("Label availability and missingness")

        missingness = prepare_rep_dataset_names(
            rep006_missingness
        )

        if not missingness.empty:

            missingness_display = missingness[
                [
                    "Dataset",
                    "label",
                    "n_total",
                    "n_available",
                    "n_missing_or_unknown",
                    "pct_available",
                    "pct_missing_or_unknown",
                    "missingness_status",
                ]
            ].copy()

            missingness_display.columns = [
                "Dataset",
                "Label",
                "Total",
                "Available",
                "Missing / unknown",
                "Available (%)",
                "Missing / unknown (%)",
                "Status",
            ]

            st.dataframe(
                missingness_display,
                hide_index=True,
                use_container_width=True,
                height=480,
            )

            heatmap_data = (
                missingness
                .pivot(
                    index="label",
                    columns="Dataset",
                    values="pct_missing_or_unknown",
                )
            )

            fig = px.imshow(
                heatmap_data,
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale=[
                    [0.0, "#EAF3FA"],
                    [0.5, WITS_GOLD],
                    [1.0, WITS_NAVY],
                ],
                title="Missing / unknown labels (%)",
            )

            fig.update_layout(
                height=400,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.markdown(
            """
            **Key findings**

            - FairFace has complete gender, race and harmonised
              age-band labels, but does not provide exact numeric age.
            - LFW has no available gender, race or age labels in the
              current project metadata.
            - UTKFace has **4 rows** with missing demographic labels
              out of 24,108 images.
            """
        )

        gold_callout(
            """
            Missing demographic information is not merely a data-cleaning
            inconvenience. It determines which fairness questions can
            actually be answered. In particular, the current LFW
            metadata cannot support demographic subgroup error analysis
            without an additional defensible label/provenance strategy.
            """
        )

        if not rep006_patterns.empty:

            st.subheader("Missingness patterns")

            patterns = prepare_rep_dataset_names(
                rep006_patterns
            )

            st.dataframe(
                patterns,
                hide_index=True,
                use_container_width=True,
            )

        if not rep006_affected.empty:

            with st.expander(
                "Analyses affected by label availability"
            ):

                affected = prepare_rep_dataset_names(
                    rep006_affected
                )

                st.dataframe(
                    affected,
                    hide_index=True,
                    use_container_width=True,
                    height=500,
                )

        source_caption(
            "REP-006",
            "missingness and affected-analysis outputs",
        )


# ============================================================
# INTEGRITY & DUPLICATES
# ============================================================

with integrity_tab:

    section_label("Canonical Phase 0 metadata")
    st.subheader("Data integrity and duplicate structure")

    integrity_rows = []

    for dataset_key, frame in master.groupby(
        "dataset"
    ):

        integrity_rows.append(
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
                "Rows with exact-duplicate group": int(
                    frame[
                        "exact_dup_group"
                    ].notna().sum()
                ),
                "Exact-duplicate groups": int(
                    frame[
                        "exact_dup_group"
                    ].dropna().nunique()
                ),
                "Rows with confirmed near-duplicate group": int(
                    frame[
                        "near_dup_group"
                    ].notna().sum()
                ),
                "Cross-dataset duplicate flagged": bool_count(
                    frame,
                    "cross_dataset_dup",
                ),
            }
        )

    integrity_summary = pd.DataFrame(
        integrity_rows
    )

    st.dataframe(
        integrity_summary,
        hide_index=True,
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:

        near_status = (
            master[
                "near_dup_status"
            ]
            .astype("string")
            .fillna("Not applicable")
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Images")
        )

        fig = px.bar(
            near_status,
            x="Status",
            y="Images",
            text="Images",
            color_discrete_sequence=[
                WITS_BLUE
            ],
            title="Near-duplicate status",
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            height=390,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:

        cross_status = (
            master[
                "cross_dataset_dup_status"
            ]
            .astype("string")
            .fillna("Not applicable")
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Images")
        )

        fig = px.bar(
            cross_status,
            x="Status",
            y="Images",
            text="Images",
            color_discrete_sequence=[
                WITS_GOLD
            ],
            title="Cross-dataset duplicate status",
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="",
            height=390,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    blue_callout(
        """
        Duplicate screening protects the later experiment from
        artificially inflated diversity and from possible leakage
        between evaluation groups. Candidate duplicate status is not
        equivalent to a human-confirmed duplicate.
        """
    )


# ============================================================
# DETECTION & ALIGNMENT
# ============================================================

with detection_tab:

    section_label("Canonical Phase 0 metadata")
    st.subheader("Face detection and alignment")

    detection_rows = []

    for dataset_key, frame in master.groupby(
        "dataset"
    ):

        alignment_known = int(
            frame[
                "alignment_valid"
            ].notna().sum()
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

        if confidence.empty:
            median_confidence = None
        else:
            median_confidence = float(
                confidence.median()
            )

        if alignment_known:
            success_pct = (
                100.0
                * alignment_valid
                / alignment_known
            )
        else:
            success_pct = None

        detection_rows.append(
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
                "Median detection confidence": median_confidence,
                "Alignment valid": alignment_valid,
                "Alignment success (%)": success_pct,
            }
        )

    detection_summary = pd.DataFrame(
        detection_rows
    )

    st.dataframe(
        detection_summary,
        hide_index=True,
        use_container_width=True,
    )

    plot_detection = (
        detection_summary[
            [
                "Dataset",
                "Alignment success (%)",
            ]
        ]
        .dropna()
    )

    if not plot_detection.empty:

        fig = px.bar(
            plot_detection,
            x="Dataset",
            y="Alignment success (%)",
            text="Alignment success (%)",
            color="Dataset",
            color_discrete_map=DATASET_COLOURS,
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
            height=390,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    confidence_values = (
        master[
            [
                "Dataset",
                "det_confidence",
            ]
        ]
        .dropna()
    )

    if not confidence_values.empty:

        fig = px.histogram(
            confidence_values,
            x="det_confidence",
            color="Dataset",
            nbins=40,
            opacity=0.7,
            color_discrete_map=DATASET_COLOURS,
            title="Detection-confidence distribution",
        )

        fig.update_layout(
            barmode="overlay",
            height=400,
            xaxis_title="Detection confidence",
            yaxis_title="Images",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    blue_callout(
        """
        Detection and alignment are part of the fairness pipeline.
        A recognition error may originate before the recognition
        network if a face is missed, the wrong face is selected, or
        the crop is poorly aligned.
        """
    )

    st.divider()

    section_label("DET-001 to DET-005")
    st.subheader("Detector limitations audit")

    phase1_table(load_detection_integrity_by_dataset(), "Detection integrity table not available.")

    subgroup_failures = load_subgroup_detection_failures()
    if not subgroup_failures.empty:
        dataset_wide = subgroup_failures[subgroup_failures["attribute"].eq("ALL")]
        if not dataset_wide.empty:
            fig = px.bar(
                dataset_wide, x="dataset", y="failure_rate", color="dataset",
                color_discrete_map={k.lower(): v for k, v in DATASET_COLOURS.items()},
                title="Detection failure rate by dataset",
            )
            fig.update_layout(showlegend=False, height=380, yaxis_tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Confidence distribution"):
        phase1_table(load_confidence_summary())
        phase1_table(load_confidence_histogram())

    with st.expander("Subgroup detection failures"):
        phase1_table(subgroup_failures)

    with st.expander("Alignment failures by subgroup"):
        phase1_table(load_alignment_failures_by_subgroup())

    st.markdown("**Detector figures**")
    detection_figure_files = [
        "alignment_outcomes.png", "confidence_distributions.png", "detector_stage_funnel.png",
        "failure_examples_contact_sheet.png", "utkface_age_band_confidence.png",
        "utkface_age_band_failure_rates.png", "utkface_gender_confidence.png",
        "utkface_gender_failure_rates.png", "utkface_race_confidence.png",
        "utkface_race_failure_rates.png", "visual_review_categories.png",
    ]
    chosen_detection_figure = st.selectbox("Figure", detection_figure_files, key="det_figure_choice")
    show_phase1_image(detection_figure(chosen_detection_figure))

    limitations_text = load_detector_limitations_text()
    if limitations_text:
        with st.expander("Detector limitations (draft)"):
            st.markdown(limitations_text)

    st.divider()

    st.subheader("Phase 0 alignment QA")
    st.write(
        "A curated sample of aligned face crops across all three datasets, "
        "plus composite contact sheets and per-stage QA images from Phase 0 "
        "preprocessing. This is a curated sample, not the full 37,000+ "
        "production aligned-crop set."
    )

    qa_manifest = load_alignment_qa_manifest()
    if qa_manifest.empty:
        st.info("Alignment QA manifest not available.")
    else:
        phase1_table(qa_manifest["dataset"].value_counts().rename_axis("dataset").reset_index(name="n"))

        qa_dataset_choice = st.selectbox(
            "Dataset", sorted(qa_manifest["dataset"].unique()), key="alignment_qa_dataset",
        )
        qa_subset = qa_manifest[qa_manifest["dataset"].eq(qa_dataset_choice)]
        qa_paths = [alignment_qa_sample(name) for name in qa_subset["filename"]]
        st.caption(f"Showing {len(qa_paths)} curated {qa_dataset_choice} samples")
        phase1_image_grid(qa_paths, columns=5)

        with st.expander("Full sample list"):
            phase1_table(qa_subset)

    st.markdown("**Composite alignment contact sheets**")
    sheet_choice = st.radio(
        "Contact sheet", ["Full sample", "20-image smoke test"], horizontal=True, key="alignment_sheet_choice",
    )
    sheet_file = "alignment_contact_sheet.png" if sheet_choice == "Full sample" else "alignment_contact_sheet_smoke20.png"
    show_phase1_image(alignment_qa_artifact("contact_sheets", sheet_file))

    qa_col1, qa_col2 = st.columns(2)
    with qa_col1:
        st.markdown("**Detector pilot QA**")
        show_phase1_image(alignment_qa_artifact("detector_pilot", "detection_alignment_qa.png"))
        show_phase1_image(alignment_qa_artifact("detector_pilot", "supplied_crop_qa.png"))
    with qa_col2:
        st.markdown("**Skin-ROI measurement QA**")
        show_phase1_image(alignment_qa_artifact("measurement_examples", "skin_roi_qa.png"))

    st.markdown("**Greyscale / achromatic detection QA**")
    achromatic_cols = st.columns(3)
    achromatic_files = ["achromatic_chroma_distribution.png", "achromatic_sample.png", "mean_rule_only_flagged_sample.png"]
    for col, filename in zip(achromatic_cols, achromatic_files):
        with col:
            show_phase1_image(alignment_qa_artifact("achromatic", filename))


# ============================================================
# VISUAL CONDITIONS
# ============================================================

with visual_tab:

    section_label("Canonical Phase 0 measurements")
    st.subheader("Visual and capture conditions")

    visual_metrics = {
        "Brightness (mean L*)": "brightness",
        "Contrast (L* standard deviation)": "contrast",
        "Sharpness (Laplacian variance)": "sharpness",
        "Face-area ratio": "face_area_ratio",
        "Yaw": "yaw",
        "Pitch": "pitch",
        "Roll": "roll",
    }

    selected_metric_label = st.selectbox(
        "Metric",
        list(
            visual_metrics.keys()
        ),
        key="visual_metric",
    )

    metric = visual_metrics[
        selected_metric_label
    ]

    visual_data = (
        master[
            [
                "Dataset",
                metric,
            ]
        ]
        .dropna()
    )

    if visual_data.empty:

        st.info(
            "No populated values are available for this metric."
        )

    else:

        if len(visual_data) > 30000:

            visual_plot = (
                visual_data
                .sample(
                    30000,
                    random_state=26041,
                )
            )

        else:

            visual_plot = visual_data

        fig = px.box(
            visual_plot,
            x="Dataset",
            y=metric,
            color="Dataset",
            points=False,
            color_discrete_map=DATASET_COLOURS,
            title=selected_metric_label,
        )

        fig.update_layout(
            showlegend=False,
            height=450,
            xaxis_title="",
            yaxis_title=selected_metric_label,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        summary = (
            visual_data
            .groupby("Dataset")[metric]
            .agg(
                n="count",
                median="median",
                mean="mean",
                minimum="min",
                maximum="max",
            )
            .reset_index()
        )

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True,
        )

    gold_callout(
        """
        Visual conditions are potential confounders. If image quality,
        lighting, scale or pose differ systematically between groups,
        later subgroup error differences should not automatically be
        attributed only to demographic representation.
        """
    )

    st.divider()

    section_label("VIS-001")
    st.subheader("Visual quality audit: reliability, QC flags & subgroup tests")

    phase1_table(load_vis_status_counts(), "Status counts not available.")

    st.markdown("**Summary by dataset**")
    phase1_table(load_vis_summary_by_dataset())

    with st.expander("Subgroup summary"):
        phase1_table(load_vis_subgroup_summary())
    with st.expander("Subgroup omnibus tests"):
        phase1_table(load_vis_subgroup_omnibus())
    with st.expander("Subgroup pairwise comparisons"):
        phase1_table(load_vis_subgroup_pairwise())

    st.markdown("**Automated QC flag counts** (statistical outlier flags, not exclusions)")
    phase1_table(load_vis_qc_flag_counts())

    st.markdown("**VIS figures**")
    vis_phase1_metrics = [
        "brightness_L_mean", "contrast_L_sd", "face_area_ratio", "face_short_side_px",
        "illumination_asymmetry", "pitch_deg", "roll_deg", "sharpness_lapvar", "yaw_deg",
    ]
    vis_figure_kind = st.radio(
        "Figure type", ["Histogram", "Boxplot by skin-tone group"], horizontal=True, key="vis_figure_kind",
    )
    vis_metric_choice = st.selectbox("Metric", vis_phase1_metrics, key="vis_phase1_metric")
    if vis_figure_kind == "Histogram":
        show_phase1_image(vis_figure(f"VIS_hist_{vis_metric_choice}.png"))
    else:
        show_phase1_image(vis_figure(f"VIS_box_{vis_metric_choice}_by_skin_group_ita.png"))

    st.markdown("**Extreme-value QA contact sheets**")
    vis_extreme_choice = st.selectbox(
        "Metric", ["brightness_L_mean", "sharpness_lapvar", "yaw_deg"], key="vis_extreme_metric",
    )
    show_phase1_image(vis_qa_image(f"VIS_extremes_{vis_extreme_choice}.png"))

    st.divider()

    section_label("MEA x VIS")
    st.subheader("Does image brightness change which skin-tone group someone is assigned to?")

    blue_callout(
        """
        Skin-tone measurement is derived from pixel values in the same
        photograph whose lighting is measured separately here. If a photo's
        brightness shifts the measured L*/ITA enough to move someone across
        a skin-tone group boundary, that is a <strong>lighting artefact being
        read as a skin-tone difference</strong> — the confound this section is
        designed to surface, not a claim about anyone's actual skin tone.
        """
    )

    combined_brightness = load_brightness_vs_skin_tone()

    if combined_brightness.empty:
        st.info("Combined brightness/skin-tone table not available. Run scripts/sync_phase1_results.py.")
    else:
        st.markdown("**Brightness vs. measured skin lightness**")

        brightness_sample = combined_brightness.sample(n=min(8000, len(combined_brightness)), random_state=26041)
        scatter_fig = px.scatter(
            brightness_sample,
            x="brightness_L_mean",
            y="skin_L_star",
            color="skin_group_ita",
            facet_col="dataset",
            opacity=0.35,
            height=430,
            labels={
                "brightness_L_mean": "Image brightness (L* mean)",
                "skin_L_star": "Measured skin lightness (L*)",
                "skin_group_ita": "Skin-tone group (ITA)",
            },
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

        st.markdown("**Skin-tone group assignment by brightness quartile**")
        st.write(
            "Images are split into four brightness quartiles (Q1 = darkest photos, "
            "Q4 = brightest photos) within each dataset. If skin-tone group shares "
            "shift noticeably across quartiles, brightness is influencing the "
            "assigned group independently of the metric's intended meaning."
        )

        quartile_source = combined_brightness.dropna(subset=["skin_group_ita"]).copy()
        quartile_source["brightness_quartile"] = quartile_source.groupby("dataset")["brightness_L_mean"].transform(
            lambda s: pd.qcut(s, 4, labels=["Q1 (darkest)", "Q2", "Q3", "Q4 (brightest)"], duplicates="drop")
        )
        quartile_share = (
            quartile_source.groupby(["dataset", "brightness_quartile", "skin_group_ita"], observed=True)
            .size().reset_index(name="n")
        )
        quartile_share["pct"] = 100 * quartile_share["n"] / quartile_share.groupby(
            ["dataset", "brightness_quartile"], observed=True
        )["n"].transform("sum")

        brightness_dataset_choice = st.selectbox(
            "Dataset", sorted(quartile_source["dataset"].unique()), key="brightness_quartile_dataset",
        )
        quartile_fig = px.bar(
            quartile_share[quartile_share["dataset"].eq(brightness_dataset_choice)],
            x="brightness_quartile",
            y="pct",
            color="skin_group_ita",
            barmode="stack",
            height=420,
            labels={"brightness_quartile": "Image brightness quartile", "pct": "Share of images (%)", "skin_group_ita": "Skin-tone group (ITA)"},
        )
        st.plotly_chart(quartile_fig, use_container_width=True)

    st.markdown("**Correlation summary** (Spearman rank correlation, brightness vs. skin L*, by dataset)")
    phase1_table(load_vis_brightness_vs_skin())

    gold_callout(
        """
        A significant positive correlation here does not by itself mean the
        skin-tone measurement is invalid — brighter photographs and lighter
        skin can be correlated for real demographic-composition reasons too.
        What this section is for is making the relationship visible so it can
        be checked against subgroup composition, rather than assumed away.
        """
    )


# ============================================================
# MEASUREMENT
# ============================================================

with measurement_tab:

    section_label("MEA-001 validation in progress")
    st.subheader("Skin-colour measurement")

    valid_skin = bool_count(
        master,
        "skin_roi_valid",
    )

    invalid_skin = int(
        master[
            "skin_roi_valid"
        ].eq(False).sum()
    )

    unavailable_skin = int(
        master[
            "skin_roi_valid"
        ].isna().sum()
    )

    ita_undefined = bool_count(
        master,
        "ita_undefined",
    )

    fitz_available = int(
        master[
            "fitz_classifier"
        ].notna().sum()
    )

    measure_1, measure_2, measure_3, measure_4, measure_5 = st.columns(5)

    measure_1.metric(
        "Valid ROI",
        f"{valid_skin:,}",
    )

    measure_2.metric(
        "Invalid ROI",
        f"{invalid_skin:,}",
    )

    measure_3.metric(
        "Unavailable",
        f"{unavailable_skin:,}",
    )

    measure_4.metric(
        "ITA undefined",
        f"{ita_undefined:,}",
    )

    measure_5.metric(
        "Classifier outputs",
        f"{fitz_available:,}",
    )

    gold_callout(
        """
        These values originate from the Phase 0 measurement pipeline.
        The recovered skin-ROI protocol is currently being formally
        validated under MEA-001. The automated Fitzpatrick-classifier
        output is not yet available in the canonical table.
        """
    )

    measurement_metrics = {
        "L* lightness": "L_star",
        "ITA": "ITA",
        "Hue angle": "hue_angle",
        "a*": "a_star",
        "b*": "b_star",
    }

    measurement_label = st.selectbox(
        "Continuous measurement",
        list(
            measurement_metrics.keys()
        ),
        key="measurement_metric",
    )

    measurement_column = (
        measurement_metrics[
            measurement_label
        ]
    )

    measurement_data = (
        master[
            [
                "Dataset",
                measurement_column,
            ]
        ]
        .dropna()
    )

    if not measurement_data.empty:

        if len(measurement_data) > 40000:

            measurement_plot = (
                measurement_data
                .sample(
                    40000,
                    random_state=26041,
                )
            )

        else:

            measurement_plot = (
                measurement_data
            )

        fig = px.histogram(
            measurement_plot,
            x=measurement_column,
            color="Dataset",
            nbins=60,
            opacity=0.6,
            color_discrete_map=DATASET_COLOURS,
            title=measurement_label,
        )

        fig.update_layout(
            barmode="overlay",
            height=420,
            yaxis_title="Images",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        measurement_summary = (
            measurement_data
            .groupby("Dataset")[
                measurement_column
            ]
            .agg(
                n="count",
                median="median",
                mean="mean",
                minimum="min",
                maximum="max",
            )
            .reset_index()
        )

        st.dataframe(
            measurement_summary,
            hide_index=True,
            use_container_width=True,
        )

    st.divider()

    st.subheader("Provisional ITA grouping")

    ita_available = master[
        master[
            "ita_group"
        ].notna()
    ].copy()

    if not ita_available.empty:

        ita_counts = (
            ita_available
            .groupby(
                [
                    "Dataset",
                    "ita_group",
                ],
                dropna=False,
            )
            .size()
            .rename("Images")
            .reset_index()
        )

        fig = px.bar(
            ita_counts,
            x="ita_group",
            y="Images",
            color="Dataset",
            barmode="group",
            color_discrete_map=DATASET_COLOURS,
        )

        fig.update_layout(
            height=420,
            xaxis_title="ITA group",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.caption(
        """
        Continuous L*, ITA and hue remain the primary measurement
        quantities. Discrete skin-tone categories are derived proxies,
        not ground-truth demographic identities.
        """
    )

    st.divider()

    section_label("MEA-001 to MEA-005")
    st.subheader("Skin-tone measurement reliability audit")

    phase1_table(load_mea_status_counts(), "Status counts not available.")

    st.markdown("**Summary by dataset**")
    phase1_table(load_mea_summary_by_dataset())

    with st.expander("Subgroup summary"):
        phase1_table(load_mea_subgroup_summary())
    with st.expander("Subgroup omnibus tests"):
        phase1_table(load_mea_subgroup_omnibus())
    with st.expander("Subgroup pairwise comparisons"):
        phase1_table(load_mea_subgroup_pairwise())

    st.markdown("**Reliability agreement** (Lin's CCC, Bland-Altman bias/limits, kappa)")
    phase1_table(load_mea_reliability_agreement())

    with st.expander("Boundary sensitivity"):
        phase1_table(load_mea_boundary_sensitivity())

    skin_group_distribution = load_mea_skin_group_distribution()
    with st.expander("Skin-tone group distribution"):
        phase1_table(skin_group_distribution)

    if not skin_group_distribution.empty and {"dataset", "category", "pct"}.issubset(skin_group_distribution.columns):
        dist_fig = px.bar(
            skin_group_distribution, x="category", y="pct", color="dataset", barmode="group",
            color_discrete_map={k.lower(): v for k, v in DATASET_COLOURS.items()},
            title="Skin-tone group distribution by dataset",
            labels={"pct": "Share of images (%)", "category": "Skin-tone group"},
        )
        dist_fig.update_layout(height=400)
        st.plotly_chart(dist_fig, use_container_width=True)

    with st.expander("ROI failure by group"):
        phase1_table(load_mea_roi_failure_by_group())
        phase1_table(load_mea_roi_failure_tests())

    st.markdown("**MEA figures**")
    mea_figure_kind = st.radio(
        "Figure type", ["Histogram", "Bland-Altman reliability"], horizontal=True, key="mea_figure_kind",
    )
    if mea_figure_kind == "Histogram":
        mea_hist_metric = st.selectbox(
            "Metric", ["ITA", "skin_L_star", "skin_chroma", "skin_hue_deg"], key="mea_hist_metric",
        )
        show_phase1_image(mea_figure(f"MEA_hist_{mea_hist_metric}.png"))
    else:
        ba_col1, ba_col2 = st.columns(2)
        with ba_col1:
            ba_dataset = st.selectbox("Dataset", ["fairface", "lfw", "utkface"], key="mea_ba_dataset")
        with ba_col2:
            ba_comparison = st.selectbox(
                "Comparison",
                [
                    "rel_ITA_forehead_vs_rel_ITA_left_cheek",
                    "rel_ITA_forehead_vs_rel_ITA_right_cheek",
                    "rel_ITA_left_cheek_vs_rel_ITA_right_cheek",
                    "skin_ITA_deg_vs_phase0_ITA",
                    "skin_ITA_deg_vs_rel_ITA_mean_based",
                ],
                key="mea_ba_comparison",
            )
        show_phase1_image(mea_figure(f"MEA_BA_{ba_dataset}_{ba_comparison}.png"))

    st.markdown("**QA sample**")
    show_phase1_image(mea_qa_image("MEA_roi_overlay_sample.png"), "Skin-tone ROI overlay sample")

    mea_method = load_mea_method()
    if mea_method:
        with st.expander("Methodology"):
            st.json(mea_method)

    mea_limitations = load_mea_limitations_text()
    if mea_limitations:
        with st.expander("Limitations (draft)"):
            st.markdown(mea_limitations)


# ============================================================
# CROSS-DATASET SHIFT
# ============================================================

with shift_tab:

    section_label("SHIFT-001")
    st.subheader("How comparable are FairFace, UTKFace and LFW after measurement?")

    phase1_table(load_shift_analysis_set_counts(), "Analysis set counts not available.")

    categorical_dist = load_shift_categorical_distributions()
    with st.expander("Categorical distributions & tests", expanded=True):
        phase1_table(categorical_dist)
        phase1_table(load_shift_categorical_tests())

    if not categorical_dist.empty and {"dataset", "variable", "category", "pct"}.issubset(categorical_dist.columns):
        shift_variable_choice = st.selectbox(
            "Variable", sorted(categorical_dist["variable"].unique()), key="shift_categorical_variable",
        )
        shift_bar_fig = px.bar(
            categorical_dist[categorical_dist["variable"].eq(shift_variable_choice)],
            x="category", y="pct", color="dataset", barmode="group",
            color_discrete_map={k.lower(): v for k, v in DATASET_COLOURS.items()},
            labels={"pct": "Share of images (%)", "category": shift_variable_choice},
            title=f"{shift_variable_choice} distribution by dataset",
        )
        shift_bar_fig.update_layout(height=400)
        st.plotly_chart(shift_bar_fig, use_container_width=True)

    with st.expander("Continuous summary & tests"):
        phase1_table(load_shift_continuous_summary())
        phase1_table(load_shift_continuous_tests())
    with st.expander("Nominal & effective balance"):
        phase1_table(load_shift_nominal_balance())
        phase1_table(load_shift_effective_balance_summary())
        phase1_table(load_shift_effective_balance_tests())
    with st.expander("Identity depth"):
        phase1_table(load_shift_identity_depth())
    with st.expander("ITA within native race"):
        phase1_table(load_shift_ita_within_native_race())

    st.markdown("**Skin-tone group composition by dataset**")
    show_phase1_image(shift_figure("SHIFT_skin_group_by_dataset.png"))

    shift_axes = load_shift_frozen_axes()
    if shift_axes:
        with st.expander("Frozen comparison axes"):
            st.json(shift_axes)

    shift_limitations = load_shift_limitations_text()
    if shift_limitations:
        with st.expander("Interpretation limits (draft)"):
            st.markdown(shift_limitations)


# ============================================================
# LFW PROTOCOL
# ============================================================

with lfw_protocol_tab:

    section_label("LFW protocol")
    st.subheader("Official pair-list integrity, composition and skin-tone yoking")

    phase1_table(load_lfw_pair_file_checks(), "Pair-file checks not available.")

    st.markdown("**Pair resolution**")
    phase1_table(load_lfw_pair_resolution_counts())

    with st.expander("Composition: images vs. pairs"):
        phase1_table(load_lfw_composition_images_vs_pairs())
        phase1_table(load_lfw_composition_tests())

    st.markdown("**Impostor-pair skin-tone yoking**")
    lfw_yoking = load_lfw_impostor_yoking_counts()
    phase1_table(lfw_yoking)
    if not lfw_yoking.empty:
        lfw_yoking_fig = px.bar(
            lfw_yoking, x="view", y="pct", color="yoking", barmode="stack", facet_col="split",
            labels={"pct": "Share of impostor pairs (%)", "view": "View", "yoking": "Yoking"},
            height=420,
        )
        st.plotly_chart(lfw_yoking_fig, use_container_width=True)

    with st.expander("Impostor same-skin-group vs. random pairing"):
        phase1_table(load_lfw_impostor_same_group_vs_random())

    with st.expander("Genuine-pair visual difference by skin-tone group"):
        phase1_table(load_lfw_genuine_visual_difference_by_skin_group())
        phase1_table(load_lfw_genuine_visual_difference_tests())

    with st.expander("Visual difference, median by yoking category"):
        phase1_table(load_lfw_visual_difference_median_by_yoking())

    lfw_status = load_lfw_status()
    lfw_rules = load_lfw_protocol_rules()
    if lfw_status or lfw_rules:
        with st.expander("Protocol status & rules"):
            if lfw_status:
                st.json(lfw_status)
            if lfw_rules:
                st.json(lfw_rules)
    