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
) = st.tabs(
    [
        "Overview",
        "Representation",
        "Integrity & Duplicates",
        "Detection & Alignment",
        "Visual Conditions",
        "Measurement",
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
    