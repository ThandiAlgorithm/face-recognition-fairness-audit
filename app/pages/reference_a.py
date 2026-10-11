from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from reference_a_data import (
    canonical_lfw_path,
    load_alignment_lookup,
    load_bootstrap_overall,
    load_bootstrap_subgroups,
    load_confusion,
    load_error_cases,
    load_error_summary,
    load_local_pair_scores,
    load_pair_coverage,
    load_preprocessing_comparison,
    load_qualitative_examples,
    load_roc,
    load_safe_pair_scores,
    load_subgroup_fairness,
    load_summary,
)

from ui import (
    evidence_note,
    finding_card,
    metric_row,
    page_header,
    research_card,
    section_header,
)


# ============================================================
# CONSTANTS
# ============================================================

WITS_BLUE = "#0067B1"
WITS_NAVY = "#002B5C"
WITS_GOLD = "#D8AA19"


# ============================================================
# LOAD DATA
# ============================================================

summary = load_summary()

pair_scores = load_safe_pair_scores()
roc = load_roc()
confusion = load_confusion()
subgroups = load_subgroup_fairness()
pair_coverage = load_pair_coverage()

bootstrap = load_bootstrap_overall()
bootstrap_subgroups = load_bootstrap_subgroups()

preprocessing = load_preprocessing_comparison()
example_manifest = load_qualitative_examples()

error_cases = load_error_cases()
error_summary = load_error_summary()


if not summary:

    st.error(
        """
        Facial-verification results were not found.
        """
    )

    st.stop()


# ============================================================
# HELPERS
# ============================================================

def percent(
    value,
    digits: int = 1,
) -> str:

    try:
        value = float(value)

    except Exception:
        return "Unavailable"

    if not np.isfinite(value):
        return "Unavailable"

    return f"{value * 100:.{digits}f}%"


def bool_value(
    value,
) -> bool:

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(value)

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


def ci_text(
    metric: str,
) -> str:

    if bootstrap.empty:
        return "Unavailable"

    row = bootstrap[
        bootstrap[
            "metric"
        ].eq(metric)
    ]

    if row.empty:
        return "Unavailable"

    row = row.iloc[0]

    return (
        f"{row['ci_lower']:.1%}"
        f"–"
        f"{row['ci_upper']:.1%}"
    )


def show_local_pair_images(
    pair_index: int,
    show_raw: bool = True,
) -> None:

    detailed = load_local_pair_scores()
    lookup = load_alignment_lookup()

    if detailed.empty or not lookup:

        st.info(
            "Local source and aligned images are not available "
            "in this environment."
        )

        return

    if (
        pair_index < 0
        or pair_index >= len(detailed)
    ):

        st.warning(
            "Pair index is outside the available score table."
        )

        return

    pair = detailed.iloc[
        pair_index
    ]

    raw1 = Path(
        str(
            pair[
                "image1_path"
            ]
        )
    )

    raw2 = Path(
        str(
            pair[
                "image2_path"
            ]
        )
    )

    key1 = canonical_lfw_path(
        raw1
    )

    key2 = canonical_lfw_path(
        raw2
    )

    aligned1 = lookup.get(
        key1
    )

    aligned2 = lookup.get(
        key2
    )

    if show_raw:

        st.markdown(
            "#### Original benchmark images"
        )

        left, right = st.columns(2)

        with left:

            if raw1.exists():

                st.image(
                    str(raw1),
                    caption="Source image A",
                    use_container_width=True,
                )

        with right:

            if raw2.exists():

                st.image(
                    str(raw2),
                    caption="Source image B",
                    use_container_width=True,
                )

    st.markdown(
        "#### Aligned model inputs"
    )

    left, right = st.columns(2)

    with left:

        if (
            aligned1 is not None
            and aligned1.exists()
        ):

            st.image(
                str(aligned1),
                caption="Aligned crop A",
                use_container_width=True,
            )

    with right:

        if (
            aligned2 is not None
            and aligned2.exists()
        ):

            st.image(
                str(aligned2),
                caption="Aligned crop B",
                use_container_width=True,
            )


# ============================================================
# HEADER
# ============================================================

page_header(
    "Face Recognition Audit",
    (
        "Evaluation of pretrained ArcFace under a controlled LFW "
        "verification protocol, including overall performance, "
        "subgroup behaviour and end-to-end error analysis."
    ),
    "Facial verification",
)


# ============================================================
# HEADLINE PERFORMANCE
# ============================================================

section_header(
    "Verification performance",
    (
        "The threshold is selected using development pairs and then "
        "held fixed for the test evaluation."
    ),
)

metric_row(
    [
        (
            "Accuracy",
            percent(
                summary[
                    "accuracy"
                ]
            ),
        ),
        (
            "True Accept Rate",
            percent(
                summary[
                    "TPR_TAR"
                ]
            ),
        ),
        (
            "False Match Rate",
            percent(
                summary[
                    "FMR_FPR"
                ]
            ),
        ),
        (
            "False Non-Match Rate",
            percent(
                summary[
                    "FNMR_FNR"
                ]
            ),
        ),
        (
            "ROC AUC",
            f"{summary['ROC_AUC']:.4f}",
        ),
    ]
)

st.caption(
    f"""
    95% bootstrap intervals ·
    Accuracy {ci_text('accuracy')} ·
    TAR {ci_text('TPR_TAR')} ·
    FMR {ci_text('FMR_FPR')} ·
    FNMR {ci_text('FNMR_FNR')}
    """
)

st.divider()


# ============================================================
# KEY FINDINGS
# ============================================================

section_header(
    "Key findings",
    (
        "Overall verification is strong, but aggregate performance "
        "does not describe how errors are distributed."
    ),
)

f1, f2, f3 = st.columns(3)

with f1:

    finding_card(
        "Strong overall verification",
        (
            "The frozen ArcFace pipeline achieved 96.8% test accuracy "
            "with ROC AUC approximately 0.994."
        ),
        (
            "A high aggregate recognition score does not remove the "
            "need for subgroup and error-level analysis."
        ),
    )

with f2:

    finding_card(
        "False non-matches dominate",
        (
            "The test set produced 22 false non-matches and "
            "10 false matches."
        ),
        (
            "Recognition failures are not symmetric. Genuine users "
            "were rejected more often than impostor pairs were accepted."
        ),
    )

with f3:

    finding_card(
        "Preprocessing remains relevant",
        (
            "Twelve of the 32 incorrect verification pairs contained "
            "at least one multi-face detection signal."
        ),
        (
            "Some apparent recognition failures may involve upstream "
            "face-selection or alignment conditions."
        ),
    )


# ============================================================
# MAIN TABS
# ============================================================

(
    performance_tab,
    fairness_tab,
    errors_tab,
    examples_tab,
    methodology_tab,
) = st.tabs(
    [
        "Performance",
        "Subgroup Analysis",
        "Error Analysis",
        "Pair Examples",
        "Method",
    ]
)


# ============================================================
# PERFORMANCE
# ============================================================

with performance_tab:

    section_header(
        "Decision behaviour",
        (
            "Row-normalised percentages show how genuine and impostor "
            "pairs are classified."
        ),
    )

    if not confusion.empty:

        genuine = (
            confusion[
                confusion[
                    "actual_class"
                ].eq(
                    "Genuine"
                )
            ]
            .iloc[0]
        )

        impostor = (
            confusion[
                confusion[
                    "actual_class"
                ].eq(
                    "Impostor"
                )
            ]
            .iloc[0]
        )

        matrix = np.array(
            [
                [
                    genuine[
                        "predicted_genuine_pct"
                    ]
                    * 100,
                    genuine[
                        "predicted_impostor_pct"
                    ]
                    * 100,
                ],
                [
                    impostor[
                        "predicted_genuine_pct"
                    ]
                    * 100,
                    impostor[
                        "predicted_impostor_pct"
                    ]
                    * 100,
                ],
            ]
        )

        labels = np.array(
            [
                [
                    f"{matrix[0, 0]:.1f}%",
                    f"{matrix[0, 1]:.1f}%",
                ],
                [
                    f"{matrix[1, 0]:.1f}%",
                    f"{matrix[1, 1]:.1f}%",
                ],
            ]
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[
                    "Predicted match",
                    "Predicted non-match",
                ],
                y=[
                    "True genuine pair",
                    "True impostor pair",
                ],
                text=labels,
                texttemplate="%{text}",
                textfont=dict(
                    size=18,
                ),
                zmin=0,
                zmax=100,
                colorscale=[
                    [
                        0.0,
                        "#F4F7FA",
                    ],
                    [
                        0.45,
                        "#A8CDE5",
                    ],
                    [
                        1.0,
                        WITS_BLUE,
                    ],
                ],
                colorbar=dict(
                    title="Row %",
                ),
            )
        )

        fig.update_layout(
            title=(
                "Verification confusion matrix "
                "— row-normalised"
            ),
            height=470,
            xaxis_title="Verification decision",
            yaxis_title="Protocol pair type",
            margin=dict(
                l=40,
                r=30,
                t=70,
                b=40,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    metric_row(
        [
            (
                "Genuine accepted",
                "95.6%",
            ),
            (
                "Genuine rejected",
                "4.4%",
            ),
            (
                "Impostor accepted",
                "2.0%",
            ),
            (
                "Impostor rejected",
                "98.0%",
            ),
        ]
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        section_header(
            "ROC curve"
        )

        if not roc.empty:

            fig = px.line(
                roc,
                x="FPR",
                y="TPR",
            )

            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    dash="dash",
                    color="#9AA5B1",
                ),
            )

            fig.update_layout(
                height=430,
                xaxis_title=(
                    "False Match Rate"
                ),
                yaxis_title=(
                    "True Accept Rate"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                f"""
                ROC AUC = {summary['ROC_AUC']:.4f} ·
                EER = {summary['EER']:.2%}
                """
            )

    with right:

        section_header(
            "Similarity distributions"
        )

        if not pair_scores.empty:

            test = (
                pair_scores[
                    pair_scores[
                        "split"
                    ]
                    .astype(str)
                    .str.lower()
                    .eq("test")
                ]
                .copy()
            )

            test[
                "Pair type"
            ] = np.where(
                test[
                    "is_genuine"
                ].astype(bool),
                "Genuine",
                "Impostor",
            )

            fig = px.histogram(
                test,
                x="cosine_similarity",
                color="Pair type",
                nbins=45,
                opacity=0.65,
                barmode="overlay",
                color_discrete_map={
                    "Genuine": WITS_BLUE,
                    "Impostor": WITS_GOLD,
                },
            )

            fig.add_vline(
                x=float(
                    summary[
                        "threshold"
                    ]
                ),
                line_dash="dash",
                line_color=WITS_NAVY,
                annotation_text=(
                    f"Threshold "
                    f"{summary['threshold']:.3f}"
                ),
            )

            fig.update_layout(
                height=430,
                xaxis_title=(
                    "Cosine similarity"
                ),
                yaxis_title=(
                    "Pair count"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    if not preprocessing.empty:

        st.divider()

        section_header(
            "Preprocessing sensitivity",
            (
                "The frozen aligned pipeline is compared with the "
                "earlier raw-input diagnostic."
            ),
        )

        plot = (
            preprocessing.copy()
        )

        plot[
            "Accuracy (%)"
        ] = (
            plot[
                "accuracy"
            ]
            * 100
        )

        plot[
            "AUC (%)"
        ] = (
            plot[
                "ROC_AUC"
            ]
            * 100
        )

        fig = px.bar(
            plot,
            x="pipeline",
            y=[
                "Accuracy (%)",
                "AUC (%)",
            ],
            barmode="group",
        )

        fig.update_layout(
            height=420,
            xaxis_title="",
            yaxis_title="Percent",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            """
            The raw-input run is retained as a preprocessing
            diagnostic. The reported verification result uses the
            frozen project-aligned images.
            """
        )


# ============================================================
# FAIRNESS
# ============================================================

with fairness_tab:

    section_header(
        "Subgroup verification",
        (
            "False-match and false-non-match behaviour are evaluated "
            "where subgroup pair counts are sufficient."
        ),
    )

    st.warning(
        """
        LFW demographic labels used in this analysis are derived
        measurements. Coverage and denominators must therefore be
        considered alongside subgroup error rates.
        """
    )

    if not pair_coverage.empty:

        display = (
            pair_coverage.copy()
        )

        display[
            "Known coverage (%)"
        ] = (
            display[
                "both_labels_known_pct"
            ]
            * 100
        )

        with st.expander(
            "Demographic pair coverage"
        ):

            st.dataframe(
                display[
                    [
                        "attribute",
                        "total_pairs",
                        "both_labels_known",
                        "Known coverage (%)",
                        "genuine_same_known_group",
                        "impostor_within_group",
                        "impostor_cross_group",
                    ]
                ].rename(
                    columns={
                        "attribute": "Attribute",
                        "total_pairs": "Test pairs",
                        "both_labels_known": (
                            "Both labels known"
                        ),
                        "genuine_same_known_group": (
                            "Genuine same-group"
                        ),
                        "impostor_within_group": (
                            "Within-group impostors"
                        ),
                        "impostor_cross_group": (
                            "Cross-group impostors"
                        ),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

    gender = (
        subgroups[
            subgroups[
                "attribute"
            ].eq("gender")
        ]
        .copy()
    )

    if not gender.empty:

        section_header(
            "Gender"
        )

        display = gender[
            [
                "group",
                "n_genuine",
                "n_impostor",
                "TPR_TAR",
                "FMR_FPR",
                "FNMR_FNR",
            ]
        ].copy()

        display[
            "TAR / TPR (%)"
        ] = (
            display[
                "TPR_TAR"
            ]
            * 100
        )

        display[
            "FMR / FPR (%)"
        ] = (
            display[
                "FMR_FPR"
            ]
            * 100
        )

        display[
            "FNMR / FNR (%)"
        ] = (
            display[
                "FNMR_FNR"
            ]
            * 100
        )

        st.dataframe(
            display[
                [
                    "group",
                    "n_genuine",
                    "n_impostor",
                    "TAR / TPR (%)",
                    "FMR / FPR (%)",
                    "FNMR / FNR (%)",
                ]
            ].rename(
                columns={
                    "group": "Group",
                    "n_genuine": (
                        "Genuine pairs"
                    ),
                    "n_impostor": (
                        "Within-group impostors"
                    ),
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        female = (
            gender[
                gender[
                    "group"
                ].eq("female")
            ]
            .iloc[0]
        )

        male = (
            gender[
                gender[
                    "group"
                ].eq("male")
            ]
            .iloc[0]
        )

        gap = abs(
            female[
                "TPR_TAR"
            ]
            - male[
                "TPR_TAR"
            ]
        )

        st.info(
            f"""
            Female genuine-pair TAR was
            **{female['TPR_TAR']:.2%}**
            compared with
            **{male['TPR_TAR']:.2%}**
            for male pairs — a measured difference of
            **{gap * 100:.2f} percentage points**.
            """
        )

    race = (
        subgroups[
            subgroups[
                "attribute"
            ].eq("race")
        ]
        .copy()
    )

    if not race.empty:

        section_header(
            "Derived race"
        )

        race_display = race[
            [
                "group",
                "n_genuine",
                "n_impostor",
                "TPR_TAR",
                "FMR_FPR",
                "FNMR_FNR",
            ]
        ].copy()

        for column in [
            "TPR_TAR",
            "FMR_FPR",
            "FNMR_FNR",
        ]:

            race_display[
                column
            ] *= 100

        st.dataframe(
            race_display.rename(
                columns={
                    "group": "Group",
                    "n_genuine": (
                        "Genuine pairs"
                    ),
                    "n_impostor": (
                        "Within-group impostors"
                    ),
                    "TPR_TAR": (
                        "TAR / TPR (%)"
                    ),
                    "FMR_FPR": (
                        "FMR / FPR (%)"
                    ),
                    "FNMR_FNR": (
                        "FNMR / FNR (%)"
                    ),
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.warning(
            """
            Some race groups contain no within-group impostor pairs in
            the test subset. False Match Rate is therefore not
            estimable for those groups and should not be interpreted
            as zero.
            """
        )

    intersection = (
        subgroups[
            subgroups[
                "attribute"
            ].eq("intersection")
        ]
        .copy()
    )

    if not intersection.empty:

        with st.expander(
            "Race × gender intersections"
        ):

            display = intersection[
                [
                    "group",
                    "n_genuine",
                    "n_impostor",
                    "TPR_TAR",
                    "FMR_FPR",
                    "FNMR_FNR",
                ]
            ].copy()

            for column in [
                "TPR_TAR",
                "FMR_FPR",
                "FNMR_FNR",
            ]:

                display[
                    column
                ] *= 100

            st.dataframe(
                display.rename(
                    columns={
                        "group": "Group",
                        "n_genuine": (
                            "Genuine pairs"
                        ),
                        "n_impostor": (
                            "Within-group impostors"
                        ),
                        "TPR_TAR": (
                            "TAR / TPR (%)"
                        ),
                        "FMR_FPR": (
                            "FMR / FPR (%)"
                        ),
                        "FNMR_FNR": (
                            "FNMR / FNR (%)"
                        ),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )


# ============================================================
# ERROR ANALYSIS
# ============================================================

with errors_tab:

    section_header(
        "End-to-end error analysis",
        (
            "The 32 incorrect verification decisions are inspected "
            "for possible upstream and recognition-related factors."
        ),
    )

    if error_summary.empty:

        st.info(
            "Detailed error-audit results are not available."
        )

    else:

        all_row = (
            error_summary[
                error_summary[
                    "outcome"
                ].eq("ALL")
            ]
            .iloc[0]
        )

        fn_row = (
            error_summary[
                error_summary[
                    "outcome"
                ].eq("FN")
            ]
            .iloc[0]
        )

        fp_row = (
            error_summary[
                error_summary[
                    "outcome"
                ].eq("FP")
            ]
            .iloc[0]
        )

        metric_row(
            [
                (
                    "Total errors",
                    f"{int(all_row['n_errors'])}",
                ),
                (
                    "False non-matches",
                    f"{int(fn_row['n_errors'])}",
                ),
                (
                    "False matches",
                    f"{int(fp_row['n_errors'])}",
                ),
                (
                    "Multi-face signal",
                    (
                        f"{int(all_row['pairs_with_multi_face_signal'])}"
                        " / "
                        f"{int(all_row['n_errors'])}"
                    ),
                ),
            ]
        )

        st.info(
            """
            At least one source image contained a multi-face detection
            signal in 12 of the 32 incorrect verification pairs.
            This does not prove that preprocessing caused the error,
            but it identifies a plausible upstream factor for manual
            inspection.
            """
        )

    if not error_cases.empty:

        with st.expander(
            "View all incorrect pairs"
        ):

            table = error_cases[
                [
                    "pair_index",
                    "outcome",
                    "cosine_similarity",
                    "image1_n_detections",
                    "image2_n_detections",
                    "has_multi_face_signal",
                    "automatic_flags",
                ]
            ].copy()

            st.dataframe(
                table.rename(
                    columns={
                        "pair_index": "Pair",
                        "outcome": "Error type",
                        "cosine_similarity": (
                            "Cosine similarity"
                        ),
                        "image1_n_detections": (
                            "Faces detected A"
                        ),
                        "image2_n_detections": (
                            "Faces detected B"
                        ),
                        "has_multi_face_signal": (
                            "Multi-face signal"
                        ),
                        "automatic_flags": (
                            "Automatic QA flags"
                        ),
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

        section_header(
            "Inspect an error"
        )

        options = (
            error_cases[
                "pair_index"
            ]
            .astype(int)
            .tolist()
        )

        selected_pair = st.selectbox(
            "Verification pair",
            options,
            format_func=lambda pair_index: (
                f"Pair {pair_index}"
            ),
        )

        selected = (
            error_cases[
                error_cases[
                    "pair_index"
                ].eq(
                    selected_pair
                )
            ]
            .iloc[0]
        )

        metric_row(
            [
                (
                    "Protocol label",
                    str(
                        selected[
                            "protocol_label"
                        ]
                    ),
                ),
                (
                    "Model decision",
                    str(
                        selected[
                            "model_decision"
                        ]
                    ),
                ),
                (
                    "Similarity",
                    (
                        f"{selected['cosine_similarity']:.4f}"
                    ),
                ),
                (
                    "Threshold",
                    (
                        f"{summary['threshold']:.4f}"
                    ),
                ),
            ]
        )

        if bool_value(
            selected[
                "has_multi_face_signal"
            ]
        ):

            st.warning(
                """
                This pair contains a multi-face signal. Compare the
                source photograph and aligned crop before attributing
                the error solely to the recognition model.
                """
            )

        show_local_pair_images(
            selected_pair,
            show_raw=True,
        )


# ============================================================
# PAIR EXAMPLES
# ============================================================

with examples_tab:

    section_header(
        "Representative verification decisions",
        (
            "Selected examples show correct and incorrect pair-level "
            "decisions under the frozen threshold."
        ),
    )

    st.caption(
        """
        These examples are qualitative evidence only and do not change
        the quantitative performance estimates.
        """
    )

    if example_manifest.empty:

        st.info(
            "Qualitative example manifest unavailable."
        )

    else:

        outcome_names = {
            "TP": (
                "Correct match"
            ),
            "TN": (
                "Correct rejection"
            ),
            "FP": (
                "False match"
            ),
            "FN": (
                "False non-match"
            ),
        }

        outcome = st.segmented_control(
            "Decision type",
            [
                "TP",
                "TN",
                "FP",
                "FN",
            ],
            format_func=lambda value: (
                outcome_names[
                    value
                ]
            ),
            default="TP",
        )

        subset = (
            example_manifest[
                example_manifest[
                    "outcome"
                ].eq(outcome)
            ]
        )

        for _, row in subset.iterrows():

            pair_index = int(
                row[
                    "pair_index"
                ]
            )

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### Pair {pair_index}"
                )

                metric_row(
                    [
                        (
                            "Similarity",
                            (
                                f"{row['cosine_similarity']:.4f}"
                            ),
                        ),
                        (
                            "Threshold",
                            (
                                f"{row['threshold']:.4f}"
                            ),
                        ),
                        (
                            "Protocol",
                            str(
                                row[
                                    "actual_pair_type"
                                ]
                            ),
                        ),
                        (
                            "Decision",
                            str(
                                row[
                                    "model_decision"
                                ]
                            ),
                        ),
                    ]
                )

                show_local_pair_images(
                    pair_index,
                    show_raw=False,
                )


# ============================================================
# METHOD
# ============================================================

with methodology_tab:

    section_header(
        "Verification protocol",
        (
            "The complete pipeline is frozen before final test "
            "evaluation."
        ),
    )

    stages = st.columns(4)

    with stages[0]:

        research_card(
            "Face detection",
            (
                "RetinaFace locates the face and facial landmarks in "
                "each LFW image."
            ),
            "01",
        )

    with stages[1]:

        research_card(
            "Alignment",
            (
                "Five-point alignment produces the deterministic "
                "face crop used by the recognition model."
            ),
            "02",
        )

    with stages[2]:

        research_card(
            "Embedding",
            (
                "ArcFace converts each aligned face into a "
                "512-dimensional L2-normalised representation."
            ),
            "03",
        )

    with stages[3]:

        research_card(
            "Verification",
            (
                "Cosine similarity is compared with one global "
                "threshold selected using development pairs."
            ),
            "04",
        )

    st.divider()

    metric_row(
        [
            (
                "Unique images",
                f"{summary['unique_images']:,}",
            ),
            (
                "Development pairs",
                f"{summary['train_pairs']:,}",
            ),
            (
                "Test pairs",
                f"{summary['test_pairs']:,}",
            ),
            (
                "Embedding dimension",
                str(
                    summary[
                        "embedding_dimension"
                    ]
                ),
            ),
        ]
    )

    evidence_note(
        "Frozen threshold",
        (
            f"{summary['threshold']:.6f}. "
            "The threshold was selected using development pairs only "
            "and then held fixed for final evaluation."
        ),
    )

    st.info(
        """
        LFW View 2 was not used for preprocessing, threshold selection,
        model selection or the results presented on this website.
        """
    )
    