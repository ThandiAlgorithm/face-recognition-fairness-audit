from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data import (
    category_counts,
    dataset_label,
    load_master,
    load_results_registry,
)
from ui import (
    apply_wits_theme,
    blue_callout,
    gold_callout,
    hero,
    section_label,
)


# ============================================================
# PAGE STYLE
# ============================================================

apply_wits_theme()

WITS_BLUE = "#0067B1"
WITS_NAVY = "#002B5C"
WITS_GOLD = "#F4C542"
LIGHT_BLUE = "#5DADE2"

DATASET_COLOURS = {
    "LFW": WITS_NAVY,
    "FairFace": WITS_BLUE,
    "UTKFace": WITS_GOLD,
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
# LOAD PROJECT DATA
# ============================================================

try:
    master = load_master().copy()
except Exception as exc:
    st.error(f"Could not load master audit table: {exc}")
    st.stop()

master["Dataset"] = master["dataset"].map(dataset_label)

registry = load_results_registry()


# ============================================================
# HELPERS
# ============================================================

def bool_count(frame: pd.DataFrame, column: str) -> int:
    """Count rows explicitly marked True."""

    if column not in frame.columns:
        return 0

    return int(frame[column].eq(True).sum())


def missing_count(frame: pd.DataFrame, column: str) -> int:
    """Count missing values for a column."""

    if column not in frame.columns:
        return len(frame)

    return int(frame[column].isna().sum())


def percentage(value: int, denominator: int) -> str:
    """Return a formatted percentage."""

    if denominator == 0:
        return "—"

    return f"{100.0 * value / denominator:.2f}%"


def distribution_chart(
    frame: pd.DataFrame,
    column: str,
    title: str,
) -> None:
    """Render a categorical distribution."""

    if column not in frame.columns:
        st.info(f"{column} is not available in the current table.")
        return

    counts = category_counts(frame[column])

    if counts.empty:
        st.info("No values are available for this distribution.")
        return

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        text="Count",
        title=title,
        color="Category",
        color_discrete_sequence=PALETTE,
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title="Images",
        margin=dict(l=10, r=10, t=55, b=10),
        height=390,
    )

    fig.update_traces(
        textposition="outside",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# HEADER
# ============================================================

hero(
    "Forensic Dataset Audit",
    """
    A consolidated view of the Phase 1 investigation across
    representation, data integrity, duplicate structure,
    face detection and alignment, visual conditions and
    skin-colour measurement.
    """,
    eyebrow="Stage 1 · Evidence before model evaluation",
)

gold_callout(
    """
    <strong>Why this audit exists:</strong> the purpose is not simply
    to count demographic groups. The audit identifies plausible
    sources of bias before model evaluation so that the baseline
    recognition experiment can test whether those conditions
    propagate into unequal recognition errors.
    """
)


# ============================================================
# TOP-LEVEL METRICS
# ============================================================

section_label("Audit population")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Total images",
    f"{len(master):,}",
)

metric_2.metric(
    "Datasets",
    f"{master['dataset'].nunique():,}",
)

metric_3.metric(
    "LFW identities",
    f"{master.loc[master['dataset'].eq('lfw'), 'subject_id'].nunique():,}",
)

metric_4.metric(
    "Valid skin ROIs",
    f"{bool_count(master, 'skin_roi_valid'):,}",
)


# ============================================================
# MAIN AUDIT TABS
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
# OVERVIEW TAB
# ============================================================

with overview_tab:

    st.subheader("Dataset composition")

    dataset_counts = (
        master["Dataset"]
        .value_counts()
        .rename_axis("Dataset")
        .reset_index(name="Images")
    )

    fig = px.bar(
        dataset_counts,
        x="Dataset",
        y="Images",
        text="Images",
        color="Dataset",
        color_discrete_map=DATASET_COLOURS,
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title="Images",
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    roles = pd.DataFrame(
        {
            "Dataset": [
                "LFW",
                "FairFace",
                "UTKFace",
            ],
            "Images": [
                int(master["dataset"].eq("lfw").sum()),
                int(master["dataset"].eq("fairface").sum()),
                int(master["dataset"].eq("utkface").sum()),
            ],
            "Primary audit role": [
                "Primary facial-verification benchmark",
                "Demographic comparison dataset",
                "Age and demographic comparison dataset",
            ],
        }
    )

    st.dataframe(
        roles,
        hide_index=True,
        use_container_width=True,
    )

    st.divider()

    st.subheader("What does the forensic audit cover?")

    layer_1, layer_2, layer_3 = st.columns(3)

    with layer_1:
        st.markdown(
            """
            ### Representation

            **Who is present?**

            - dataset composition
            - identity representation
            - gender
            - age
            - native race labels
            - intersectional cells
            - missing demographic information
            """
        )

    with layer_2:
        st.markdown(
            """
            ### Technical pipeline

            **What can affect the image before recognition?**

            - corrupt files
            - duplicate images
            - face detection
            - multiple detections
            - alignment
            - image quality
            """
        )

    with layer_3:
        st.markdown(
            """
            ### Measurement

            **What can be measured reliably?**

            - skin-region validity
            - CIELAB values
            - L*
            - ITA
            - hue angle
            - skin-tone proxies
            - measurement uncertainty
            """
        )

    if not registry.empty:

        st.divider()
        st.subheader("Registered project results")

        st.dataframe(
            registry,
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# REPRESENTATION TAB
# ============================================================

with representation_tab:

    st.subheader("Representation audit")

    selected_label = st.selectbox(
        "Select dataset",
        [
            "LFW",
            "FairFace",
            "UTKFace",
        ],
        key="representation_dataset",
    )

    dataset_key_map = {
        "LFW": "lfw",
        "FairFace": "fairface",
        "UTKFace": "utkface",
    }

    selected_key = dataset_key_map[selected_label]

    subset = master[
        master["dataset"].eq(selected_key)
    ].copy()

    rep_1, rep_2, rep_3, rep_4 = st.columns(4)

    rep_1.metric(
        "Images",
        f"{len(subset):,}",
    )

    if subset["subject_id"].notna().any():
        identity_display = f"{subset['subject_id'].nunique():,}"
    else:
        identity_display = "Unavailable"

    rep_2.metric(
        "Identities available",
        identity_display,
    )

    demographic_eligible = bool_count(
        subset,
        "eligible_demographic_audit",
    )

    rep_3.metric(
        "Demographic-audit eligible",
        f"{demographic_eligible:,}",
    )

    rep_4.metric(
        "Demographic exclusions",
        f"{len(subset) - demographic_eligible:,}",
    )

    st.divider()

    chart_left, chart_right = st.columns(2)

    with chart_left:
        distribution_chart(
            subset,
            "gender",
            f"{selected_label}: gender labels",
        )

    with chart_right:
        distribution_chart(
            subset,
            "age_band",
            f"{selected_label}: age bands",
        )

    distribution_chart(
        subset,
        "race",
        f"{selected_label}: native race labels",
    )

    st.divider()

    # --------------------------------------------------------
    # Identity representation
    # --------------------------------------------------------

    st.subheader("Identity representation")

    if subset["subject_id"].notna().any():

        identity_counts = (
            subset["subject_id"]
            .dropna()
            .value_counts()
        )

        singletons = int(
            identity_counts.eq(1).sum()
        )

        repeated = int(
            identity_counts.gt(1).sum()
        )

        identity_1, identity_2, identity_3, identity_4 = st.columns(4)

        identity_1.metric(
            "Identities",
            f"{len(identity_counts):,}",
        )

        identity_2.metric(
            "Singleton identities",
            f"{singletons:,}",
            percentage(
                singletons,
                len(identity_counts),
            ),
        )

        identity_3.metric(
            "Repeated identities",
            f"{repeated:,}",
        )

        identity_4.metric(
            "Maximum images / identity",
            f"{int(identity_counts.max()):,}",
        )

        bins = [
            0,
            1,
            2,
            5,
            10,
            20,
            float("inf"),
        ]

        labels = [
            "1",
            "2",
            "3–5",
            "6–10",
            "11–20",
            "21+",
        ]

        depth_categories = pd.cut(
            identity_counts.to_numpy(),
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        depth_counts = (
            pd.Series(
                depth_categories,
                name="depth_category",
            )
            .value_counts()
            .reindex(
                labels,
                fill_value=0,
            )
        )

        depth = pd.DataFrame(
            {
                "Images per identity": labels,
                "Identities": [
                    int(depth_counts.get(label, 0))
                    for label in labels
                ],
            }
        )

        fig = px.bar(
            depth,
            x="Images per identity",
            y="Identities",
            text="Identities",
            color_discrete_sequence=[
                WITS_BLUE
            ],
            title="Identity-depth distribution",
        )

        fig.update_layout(
            showlegend=False,
            height=390,
            margin=dict(
                l=10,
                r=10,
                t=55,
                b=10,
            ),
        )

        fig.update_traces(
            textposition="outside",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        if selected_key == "lfw":
            blue_callout(
                """
                LFW has a strongly long-tailed identity structure.
                Many identities appear only once while a small number
                contribute substantially more images. Image-level
                representation therefore does not imply equal
                identity-level representation.
                """
            )

    else:

        st.info(
            """
            Stable identity labels are not available for this
            dataset in the current audit.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # Intersectional representation
    # --------------------------------------------------------

    st.subheader("Intersectional representation")

    if "eligible_demographic_audit" in subset.columns:

        eligible_mask = (
            subset["eligible_demographic_audit"]
            .fillna(False)
            .astype(bool)
        )

        eligible = subset.loc[
            eligible_mask
        ].copy()

    else:
        eligible = subset.copy()

    available_dimensions = [
        column
        for column in [
            "gender",
            "race",
            "age_band",
        ]
        if (
            column in eligible.columns
            and eligible[column].notna().any()
        )
    ]

    if len(available_dimensions) >= 2:

        intersection = (
            eligible
            .groupby(
                available_dimensions,
                dropna=False,
            )
            .size()
            .rename("n")
            .reset_index()
        )

        def cell_status(n: int) -> str:
            if n < 30:
                return "Suppressed"

            if n < 100:
                return "Low confidence"

            return "Reportable"

        intersection["Status"] = (
            intersection["n"]
            .apply(cell_status)
        )

        st.dataframe(
            intersection.sort_values(
                "n",
                ascending=False,
            ),
            hide_index=True,
            use_container_width=True,
            height=480,
        )

        status_counts = (
            intersection["Status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Cells")
        )

        fig = px.bar(
            status_counts,
            x="Status",
            y="Cells",
            text="Cells",
            color="Status",
            color_discrete_map={
                "Reportable": WITS_BLUE,
                "Low confidence": WITS_GOLD,
                "Suppressed": "#9AA5B1",
            },
        )

        fig.update_layout(
            showlegend=False,
            height=340,
            xaxis_title="",
        )

        fig.update_traces(
            textposition="outside",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        gold_callout(
            """
            Intersectional cells follow the project thresholds:
            <strong>n &lt; 30</strong> is suppressed,
            <strong>30–99</strong> is treated as low confidence,
            and <strong>n ≥ 100</strong> is reportable.
            """
        )

    else:

        st.info(
            """
            This dataset does not contain enough native demographic
            dimensions for a defensible intersectional
            representation analysis.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------

    st.subheader("Demographic missingness")

    missing_rows = []

    for dataset_key, frame in master.groupby(
        "dataset"
    ):

        missing_rows.append(
            {
                "Dataset": dataset_label(dataset_key),
                "Images": len(frame),
                "Gender unavailable": missing_count(
                    frame,
                    "gender",
                ),
                "Race unavailable": missing_count(
                    frame,
                    "race",
                ),
                "Age unavailable": missing_count(
                    frame,
                    "age_band",
                ),
                "Eligible demographic audit": bool_count(
                    frame,
                    "eligible_demographic_audit",
                ),
            }
        )

    missing_table = pd.DataFrame(
        missing_rows
    )

    st.dataframe(
        missing_table,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# INTEGRITY & DUPLICATES TAB
# ============================================================

with integrity_tab:

    st.subheader(
        "Data integrity and duplicate structure"
    )

    integrity_rows = []

    for dataset_key, frame in master.groupby(
        "dataset"
    ):

        exact_rows = int(
            frame["exact_dup_group"]
            .notna()
            .sum()
        )

        exact_groups = int(
            frame["exact_dup_group"]
            .dropna()
            .nunique()
        )

        near_rows = int(
            frame["near_dup_group"]
            .notna()
            .sum()
        )

        integrity_rows.append(
            {
                "Dataset": dataset_label(
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
                "Rows in exact-duplicate groups": exact_rows,
                "Exact-duplicate groups": exact_groups,
                "Rows in confirmed near-duplicate groups": near_rows,
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

    st.divider()

    duplicate_left, duplicate_right = st.columns(2)

    with duplicate_left:

        st.subheader(
            "Near-duplicate review status"
        )

        near_status = (
            master["near_dup_status"]
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
        )

        fig.update_layout(
            showlegend=False,
            height=380,
            xaxis_title="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with duplicate_right:

        st.subheader(
            "Cross-dataset duplicate status"
        )

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
        )

        fig.update_layout(
            showlegend=False,
            height=380,
            xaxis_title="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    blue_callout(
        """
        Duplicate analysis is important because repeated or
        near-identical images can artificially inflate apparent
        dataset diversity and may introduce leakage between
        evaluation groups if they are not identified.
        """
    )


# ============================================================
# DETECTION & ALIGNMENT TAB
# ============================================================

with detection_tab:

    st.subheader(
        "Face detection and alignment audit"
    )

    detection_rows = []

    for dataset_key, frame in master.groupby(
        "dataset"
    ):

        alignment_known = int(
            frame["alignment_valid"]
            .notna()
            .sum()
        )

        alignment_valid = bool_count(
            frame,
            "alignment_valid",
        )

        confidence = (
            frame["det_confidence"]
            .dropna()
        )

        if confidence.empty:
            median_confidence = None
        else:
            median_confidence = float(
                confidence.median()
            )

        if alignment_known:
            alignment_success = (
                100.0
                * alignment_valid
                / alignment_known
            )
        else:
            alignment_success = None

        detection_rows.append(
            {
                "Dataset": dataset_label(
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
                "Alignment known": alignment_known,
                "Alignment success (%)": alignment_success,
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

    st.divider()

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
            range=[0, 100]
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )

        fig.update_layout(
            showlegend=False,
            height=390,
            xaxis_title="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.subheader(
        "Detection-confidence distribution"
    )

    detection_values = (
        master[
            [
                "Dataset",
                "det_confidence",
            ]
        ]
        .dropna()
    )

    if not detection_values.empty:

        fig = px.histogram(
            detection_values,
            x="det_confidence",
            color="Dataset",
            nbins=40,
            opacity=0.72,
            color_discrete_map=DATASET_COLOURS,
        )

        fig.update_layout(
            barmode="overlay",
            xaxis_title="Detection confidence",
            yaxis_title="Images",
            height=390,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    blue_callout(
        """
        Detection and alignment are audited because performance
        differences can arise before the recognition network itself.
        A face that is not detected or is poorly aligned does not reach
        the recogniser under the same conditions as a successful case.
        """
    )


# ============================================================
# VISUAL CONDITIONS TAB
# ============================================================

with visual_tab:

    st.subheader(
        "Visual and image-condition audit"
    )

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
        "Select visual metric",
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
            "No measurements are currently available for this metric."
        )

    else:

        if len(visual_data) > 30000:

            visual_plot = (
                visual_data
                .sample(
                    n=30000,
                    random_state=26041,
                )
            )

        else:

            visual_plot = (
                visual_data.copy()
            )

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

        visual_summary = (
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
            visual_summary,
            hide_index=True,
            use_container_width=True,
        )

    gold_callout(
        """
        Image-condition differences are possible confounders.
        If one demographic group is systematically darker, blurrier,
        smaller or more difficult to align, later recognition
        disparities cannot automatically be attributed solely to
        identity representation.
        """
    )


# ============================================================
# MEASUREMENT TAB
# ============================================================

with measurement_tab:

    st.subheader(
        "Skin-colour measurement audit"
    )

    valid_skin = bool_count(
        master,
        "skin_roi_valid",
    )

    invalid_skin = int(
        master[
            "skin_roi_valid"
        ]
        .eq(False)
        .sum()
    )

    unavailable_skin = int(
        master[
            "skin_roi_valid"
        ]
        .isna()
        .sum()
    )

    ita_undefined = bool_count(
        master,
        "ita_undefined",
    )

    measure_1, measure_2, measure_3, measure_4 = st.columns(4)

    measure_1.metric(
        "Valid skin ROIs",
        f"{valid_skin:,}",
    )

    measure_2.metric(
        "Invalid skin ROIs",
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

    gold_callout(
        """
        <strong>Validation status:</strong> these are Phase 0
        measurement outputs. The recovered skin-region extraction
        method is currently undergoing formal MEA-001 reproducibility
        and visual validation before confirmatory measurement analysis
        is frozen.
        """
    )

    measurement_metrics = {
        "L* (lightness)": "L_star",
        "ITA": "ITA",
        "Hue angle": "hue_angle",
        "a*": "a_star",
        "b*": "b_star",
    }

    selected_measurement_label = st.selectbox(
        "Select continuous measurement",
        list(
            measurement_metrics.keys()
        ),
        key="measurement_metric",
    )

    measurement_column = (
        measurement_metrics[
            selected_measurement_label
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

    if measurement_data.empty:

        st.info(
            "No values are currently available for this measurement."
        )

    else:

        if len(measurement_data) > 40000:

            measurement_plot = (
                measurement_data
                .sample(
                    n=40000,
                    random_state=26041,
                )
            )

        else:

            measurement_plot = (
                measurement_data.copy()
            )

        fig = px.histogram(
            measurement_plot,
            x=measurement_column,
            color="Dataset",
            nbins=60,
            opacity=0.60,
            color_discrete_map=DATASET_COLOURS,
        )

        fig.update_layout(
            barmode="overlay",
            height=420,
            xaxis_title=selected_measurement_label,
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

    st.subheader(
        "Provisional ITA categories"
    )

    ita_available = master[
        master["ita_group"].notna()
    ].copy()

    if ita_available.empty:

        st.info(
            "No ITA categories are currently available."
        )

    else:

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
            yaxis_title="Images",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.caption(
        """
        Continuous L*, ITA and hue values remain the primary
        measurements. Discrete skin-tone groupings are derived
        analytical categories and should not be interpreted as
        ground-truth demographic identity.
        """
    )
    