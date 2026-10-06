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
    apply_wits_theme,
    blue_callout,
    card,
    gold_callout,
    hero,
    section_label,
)


# ============================================================
# STYLE
# ============================================================

apply_wits_theme()

WITS_BLUE = "#0067B1"


# ============================================================
# LOAD
# ============================================================

summary = load_summary()

pair_scores = (
    load_safe_pair_scores()
)

roc = load_roc()

confusion = (
    load_confusion()
)

subgroups = (
    load_subgroup_fairness()
)

pair_coverage = (
    load_pair_coverage()
)

bootstrap = (
    load_bootstrap_overall()
)

bootstrap_subgroups = (
    load_bootstrap_subgroups()
)

preprocessing = (
    load_preprocessing_comparison()
)

example_manifest = (
    load_qualitative_examples()
)

error_cases = (
    load_error_cases()
)

error_summary = (
    load_error_summary()
)


if not summary:

    st.error(
        """
        Reference A report pack was not found.

        Run:

        `python track_b/build_reference_a_report_pack.py`
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
        value = float(
            value
        )
    except Exception:
        return "Unavailable"

    if not np.isfinite(
        value
    ):
        return "Unavailable"

    return (
        f"{value * 100:.{digits}f}%"
    )


def bool_value(
    value,
) -> bool:

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return bool(
            value
        )

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
        return ""

    row = bootstrap[
        bootstrap[
            "metric"
        ].eq(
            metric
        )
    ]

    if row.empty:
        return ""

    row = row.iloc[
        0
    ]

    return (
        f"{row['ci_lower']:.1%}"
        f"–"
        f"{row['ci_upper']:.1%}"
    )


def show_local_pair_images(
    pair_index: int,
    show_raw: bool = True,
) -> None:

    detailed = (
        load_local_pair_scores()
    )

    lookup = (
        load_alignment_lookup()
    )

    if (
        detailed.empty
        or not lookup
    ):

        st.info(
            """
            Local source/aligned images are not available
            in this environment.
            """
        )

        return

    if (
        pair_index < 0
        or pair_index
        >= len(
            detailed
        )
    ):

        st.warning(
            "Pair index is outside the local score table."
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
            "##### Original benchmark images"
        )

        r1, r2 = st.columns(
            2
        )

        with r1:

            if raw1.exists():

                st.image(
                    str(
                        raw1
                    ),
                    caption=(
                        "Source image A"
                    ),
                    use_container_width=True,
                )

        with r2:

            if raw2.exists():

                st.image(
                    str(
                        raw2
                    ),
                    caption=(
                        "Source image B"
                    ),
                    use_container_width=True,
                )

    st.markdown(
        "##### Faces delivered to ArcFace"
    )

    a1, a2 = st.columns(
        2
    )

    with a1:

        if (
            aligned1 is not None
            and aligned1.exists()
        ):

            st.image(
                str(
                    aligned1
                ),
                caption=(
                    "Project-aligned crop A"
                ),
                use_container_width=True,
            )

    with a2:

        if (
            aligned2 is not None
            and aligned2.exists()
        ):

            st.image(
                str(
                    aligned2
                ),
                caption=(
                    "Project-aligned crop B"
                ),
                use_container_width=True,
            )


# ============================================================
# HEADER
# ============================================================

hero(
    "Reference A — Pretrained Recognition Audit",
    """
    ArcFace is evaluated as an existing pretrained
    facial-recognition reference using the project's frozen LFW
    preprocessing and View 1 verification protocol.
    """,
    eyebrow=(
        "Stage 3 · Facial verification"
    ),
)

gold_callout(
    """
    <strong>Reference A is not M0.</strong>
    ArcFace is an external pretrained recognition reference.
    <strong>M0 is the project-trained unmitigated baseline</strong>
    against which M1, M2 and M3 will later be compared.
    """
)


# ============================================================
# HEADLINE
# ============================================================

section_label(
    "Headline result"
)

st.subheader(
    "Project-aligned ArcFace verification"
)

c1, c2, c3, c4, c5 = (
    st.columns(
        5
    )
)

c1.metric(
    "Accuracy",
    percent(
        summary[
            "accuracy"
        ]
    ),
)

c2.metric(
    "TAR / TPR",
    percent(
        summary[
            "TPR_TAR"
        ]
    ),
)

c3.metric(
    "FMR / FPR",
    percent(
        summary[
            "FMR_FPR"
        ]
    ),
)

c4.metric(
    "FNMR / FNR",
    percent(
        summary[
            "FNMR_FNR"
        ]
    ),
)

c5.metric(
    "ROC AUC",
    f"{summary['ROC_AUC']:.4f}",
)

st.caption(
    f"""
    95% bootstrap intervals:
    Accuracy {ci_text('accuracy')} ·
    TAR {ci_text('TPR_TAR')} ·
    FMR {ci_text('FMR_FPR')} ·
    FNMR {ci_text('FNMR_FNR')}
    """
)


# ============================================================
# TABS
# ============================================================

(
    overview_tab,
    verification_tab,
    fairness_tab,
    error_tab,
    examples_tab,
    methodology_tab,
) = st.tabs(
    [
        "Overview",
        "Verification Results",
        "Fairness Analysis",
        "Error Analysis",
        "Pair Examples",
        "Methodology",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    section_label(
        "Experimental design"
    )

    left, right = (
        st.columns(
            2
        )
    )

    with left:

        card(
            "01",
            "4,992 aligned LFW images",
            """
            Every unique image required by LFW View 1 was
            mapped to the frozen Phase 1 project-aligned crop.
            """,
        )

        card(
            "03",
            "512-D ArcFace embeddings",
            """
            One identity embedding was generated per aligned
            image and L2-normalised before cosine scoring.
            """,
        )

    with right:

        card(
            "02",
            "3,200 verification pairs",
            """
            2,200 development pairs selected the threshold;
            1,000 test pairs evaluated the frozen pipeline.
            """,
        )

        card(
            "04",
            "One global threshold",
            f"""
            Development-only threshold:
            **{summary['threshold']:.6f}**.
            No subgroup-specific threshold was used.
            """,
        )

    blue_callout(
        """
        The test set produced <strong>968 correct pair decisions</strong>
        and <strong>32 errors</strong>. Overall accuracy was
        <strong>96.8%</strong>, but the later sections examine whether
        the remaining errors are distributed uniformly across groups
        and visual conditions.
        """
    )

    st.divider()

    section_label(
        "Confidence intervals"
    )

    if not bootstrap.empty:

        display = bootstrap[
            bootstrap[
                "metric"
            ].isin(
                [
                    "accuracy",
                    "balanced_accuracy",
                    "TPR_TAR",
                    "FMR_FPR",
                    "FNMR_FNR",
                ]
            )
        ].copy()

        display[
            "Estimate (%)"
        ] = (
            display[
                "estimate"
            ]
            * 100
        )

        display[
            "Lower 95% CI (%)"
        ] = (
            display[
                "ci_lower"
            ]
            * 100
        )

        display[
            "Upper 95% CI (%)"
        ] = (
            display[
                "ci_upper"
            ]
            * 100
        )

        st.dataframe(
            display[
                [
                    "metric",
                    "Estimate (%)",
                    "Lower 95% CI (%)",
                    "Upper 95% CI (%)",
                ]
            ].rename(
                columns={
                    "metric":
                        "Metric",
                }
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Estimate (%)":
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                "Lower 95% CI (%)":
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
                "Upper 95% CI (%)":
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
            },
        )


# ============================================================
# VERIFICATION
# ============================================================

with verification_tab:

    section_label(
        "Decision behaviour"
    )

    st.subheader(
        "Percentage confusion matrix"
    )

    st.caption(
        """
        Each row is normalised by the protocol pair class.
        Percentages are used instead of raw counts to make the
        false-match and false-non-match behaviour directly readable.
        """
    )

    if not confusion.empty:

        genuine = confusion[
            confusion[
                "actual_class"
            ].eq(
                "Genuine"
            )
        ].iloc[0]

        impostor = confusion[
            confusion[
                "actual_class"
            ].eq(
                "Impostor"
            )
        ].iloc[0]

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
                    f"{matrix[0,0]:.1f}%",
                    f"{matrix[0,1]:.1f}%",
                ],
                [
                    f"{matrix[1,0]:.1f}%",
                    f"{matrix[1,1]:.1f}%",
                ],
            ]
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=[
                    "Predicted genuine",
                    "Predicted impostor",
                ],
                y=[
                    "Protocol genuine",
                    "Protocol impostor",
                ],
                text=labels,
                texttemplate="%{text}",
                zmin=0,
                zmax=100,
                colorscale=[
                    [
                        0,
                        "#F3F6FA",
                    ],
                    [
                        1,
                        WITS_BLUE,
                    ],
                ],
                colorbar=dict(
                    title="Percent"
                ),
            )
        )

        fig.update_layout(
            height=430,
            xaxis_title=(
                "Pipeline decision"
            ),
            yaxis_title=(
                "LFW protocol label"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        """
        **Genuine pairs:** 95.6% correctly accepted and
        4.4% falsely rejected.

        **Impostor pairs:** 98.0% correctly rejected and
        2.0% falsely accepted.
        """
    )

    st.divider()

    left, right = (
        st.columns(
            2
        )
    )

    with left:

        st.subheader(
            "ROC curve"
        )

        if not roc.empty:

            fig = px.line(
                roc,
                x="FPR",
                y="TPR",
                title=(
                    "LFW View 1 test ROC"
                ),
            )

            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(
                    dash="dash"
                ),
            )

            fig.update_layout(
                xaxis_title=(
                    "False Match Rate / FPR"
                ),
                yaxis_title=(
                    "True Accept Rate / TPR"
                ),
                height=430,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                f"""
                AUC = {summary['ROC_AUC']:.4f} ·
                EER = {summary['EER']:.2%}
                """
            )

    with right:

        st.subheader(
            "Similarity distributions"
        )

        if not pair_scores.empty:

            test = pair_scores[
                pair_scores[
                    "split"
                ]
                .astype(str)
                .str.lower()
                .eq("test")
            ].copy()

            test[
                "Protocol pair type"
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
                color=(
                    "Protocol pair type"
                ),
                nbins=45,
                opacity=0.65,
                barmode="overlay",
                title=(
                    "Cosine similarity distributions"
                ),
            )

            fig.add_vline(
                x=float(
                    summary[
                        "threshold"
                    ]
                ),
                line_dash="dash",
                annotation_text=(
                    f"threshold = "
                    f"{summary['threshold']:.3f}"
                ),
            )

            fig.update_layout(
                xaxis_title=(
                    "Cosine similarity"
                ),
                yaxis_title=(
                    "Pair count"
                ),
                height=430,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.divider()

    section_label(
        "Preprocessing validation"
    )

    if not preprocessing.empty:

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
            title=(
                "Raw diagnostic versus "
                "frozen project alignment"
            ),
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Percent",
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    gold_callout(
        """
        The original raw-input run is retained only as a
        <strong>preprocessing diagnostic</strong>. It is not the
        official ArcFace result. Reference A uses the frozen
        project-aligned inputs.
        """
    )


# ============================================================
# FAIRNESS
# ============================================================

with fairness_tab:

    section_label(
        "Verification fairness"
    )

    gold_callout(
        """
        LFW demographic group labels used here are externally
        derived measurements rather than unquestioned demographic
        ground truth. Coverage and denominators are therefore reported
        alongside subgroup rates.
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
                    "attribute":
                        "Attribute",
                    "total_pairs":
                        "Test pairs",
                    "both_labels_known":
                        "Both labels known",
                    "genuine_same_known_group":
                        "Genuine same-group",
                    "impostor_within_group":
                        "Within-group impostors",
                    "impostor_cross_group":
                        "Cross-group impostors",
                }
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Known coverage (%)":
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    ),
            },
        )

    st.divider()

    gender = subgroups[
        subgroups[
            "attribute"
        ].eq(
            "gender"
        )
    ].copy()

    st.subheader(
        "Gender"
    )

    if not gender.empty:

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
                    "n_genuine":
                        "Genuine pairs",
                    "n_impostor":
                        "Within-group impostors",
                    "TPR_TAR":
                        "TAR / TPR (%)",
                    "FMR_FPR":
                        "FMR / FPR (%)",
                    "FNMR_FNR":
                        "FNMR / FNR (%)",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        female = gender[
            gender[
                "group"
            ].eq(
                "female"
            )
        ].iloc[0]

        male = gender[
            gender[
                "group"
            ].eq(
                "male"
            )
        ].iloc[0]

        gap = abs(
            female[
                "TPR_TAR"
            ]
            - male[
                "TPR_TAR"
            ]
        )

        blue_callout(
            f"""
            Female genuine-pair TAR was
            <strong>{female['TPR_TAR']:.2%}</strong>
            compared with
            <strong>{male['TPR_TAR']:.2%}</strong>
            for male pairs, a measured difference of
            <strong>{gap * 100:.2f} percentage points</strong>.
            """
        )

    st.divider()

    race = subgroups[
        subgroups[
            "attribute"
        ].eq(
            "race"
        )
    ].copy()

    st.subheader(
        "Derived race"
    )

    if not race.empty:

        st.dataframe(
            race[
                [
                    "group",
                    "n_genuine",
                    "n_impostor",
                    "TPR_TAR",
                    "FMR_FPR",
                    "FNMR_FNR",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "TPR_TAR":
                    st.column_config.NumberColumn(
                        format="percent"
                    ),
                "FMR_FPR":
                    st.column_config.NumberColumn(
                        format="percent"
                    ),
                "FNMR_FNR":
                    st.column_config.NumberColumn(
                        format="percent"
                    ),
            },
        )

        gold_callout(
            """
            Asian and Black groups have no within-group impostor
            pairs in this test subset. Their FMR is therefore
            <strong>not estimable</strong>. A zero racial FMR gap
            must not be interpreted from these data.
            """
        )

    st.subheader(
        "Race × gender"
    )

    intersection = subgroups[
        subgroups[
            "attribute"
        ].eq(
            "intersection"
        )
    ].copy()

    if not intersection.empty:

        st.dataframe(
            intersection[
                [
                    "group",
                    "n_genuine",
                    "n_impostor",
                    "TPR_TAR",
                    "FMR_FPR",
                    "FNMR_FNR",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "TPR_TAR":
                    st.column_config.NumberColumn(
                        format="percent"
                    ),
                "FMR_FPR":
                    st.column_config.NumberColumn(
                        format="percent"
                    ),
                "FNMR_FNR":
                    st.column_config.NumberColumn(
                        format="percent"
                    ),
            },
        )

    if not bootstrap_subgroups.empty:

        warnings = (
            bootstrap_subgroups[
                bootstrap_subgroups[
                    "interpret_with_caution"
                ].map(
                    bool_value
                )
            ][
                [
                    "attribute",
                    "group",
                    "n_genuine",
                    "n_impostor",
                ]
            ]
            .drop_duplicates()
        )

        st.warning(
            f"""
            {len(warnings)} subgroup definitions were flagged for
            small or absent genuine/impostor samples.
            """
        )


# ============================================================
# ERROR ANALYSIS
# ============================================================

with error_tab:

    section_label(
        "End-to-end error analysis"
    )

    st.subheader(
        "What contributed to the 32 incorrect decisions?"
    )

    if error_summary.empty:

        st.info(
            """
            Error-audit report pack is not available.

            Run:

            `python track_b/build_reference_a_error_report_pack.py`
            """
        )

    else:

        all_row = error_summary[
            error_summary[
                "outcome"
            ].eq(
                "ALL"
            )
        ].iloc[0]

        fn_row = error_summary[
            error_summary[
                "outcome"
            ].eq(
                "FN"
            )
        ].iloc[0]

        fp_row = error_summary[
            error_summary[
                "outcome"
            ].eq(
                "FP"
            )
        ].iloc[0]

        e1, e2, e3, e4 = (
            st.columns(
                4
            )
        )

        e1.metric(
            "Total test errors",
            int(
                all_row[
                    "n_errors"
                ]
            ),
        )

        e2.metric(
            "False non-matches",
            int(
                fn_row[
                    "n_errors"
                ]
            ),
        )

        e3.metric(
            "False matches",
            int(
                fp_row[
                    "n_errors"
                ]
            ),
        )

        e4.metric(
            "Errors with multi-face signal",
            (
                f"{int(all_row['pairs_with_multi_face_signal'])}"
                " / "
                f"{int(all_row['n_errors'])}"
            ),
        )

        blue_callout(
            """
            Phase 1 flagged at least one image as containing
            multiple detected faces in <strong>12 of the 32</strong>
            incorrect verification pairs: 10 false non-matches and
            2 false matches.
            """
        )

        gold_callout(
            """
            <strong>A multi-face signal is not proof of a
            preprocessing failure.</strong> It means multiple candidate
            faces were detected in at least one source image.
            Manual inspection is required to determine whether the
            selected aligned crop represents the LFW protocol identity.
            """
        )

    if not error_cases.empty:

        st.subheader(
            "All 32 incorrect pairs"
        )

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
                    "pair_index":
                        "Pair",
                    "outcome":
                        "Error type",
                    "cosine_similarity":
                        "Cosine similarity",
                    "image1_n_detections":
                        "Faces detected A",
                    "image2_n_detections":
                        "Faces detected B",
                    "has_multi_face_signal":
                        "Multi-face signal",
                    "automatic_flags":
                        "Automatic QA flags",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        st.subheader(
            "Inspect an error"
        )

        options = (
            error_cases[
                "pair_index"
            ]
            .astype(int)
            .tolist()
        )

        def error_label(
            pair_index: int,
        ) -> str:

            row = error_cases[
                error_cases[
                    "pair_index"
                ].eq(
                    pair_index
                )
            ].iloc[0]

            multi = bool_value(
                row[
                    "has_multi_face_signal"
                ]
            )

            suffix = (
                " · multi-face signal"
                if multi
                else ""
            )

            return (
                f"Pair {pair_index} · "
                f"{row['outcome']} · "
                f"score {row['cosine_similarity']:.4f}"
                f"{suffix}"
            )

        selected_pair = st.selectbox(
            "Error pair",
            options,
            format_func=(
                error_label
            ),
        )

        selected = error_cases[
            error_cases[
                "pair_index"
            ].eq(
                selected_pair
            )
        ].iloc[0]

        m1, m2, m3, m4 = (
            st.columns(
                4
            )
        )

        m1.metric(
            "Protocol label",
            selected[
                "protocol_label"
            ],
        )

        m2.metric(
            "Pipeline decision",
            selected[
                "model_decision"
            ],
        )

        m3.metric(
            "Cosine similarity",
            f"{selected['cosine_similarity']:.4f}",
        )

        m4.metric(
            "Frozen threshold",
            f"{summary['threshold']:.4f}",
        )

        if bool_value(
            selected[
                "has_multi_face_signal"
            ]
        ):

            st.warning(
                """
                This pair contains an automatic multi-face signal.
                Compare the original benchmark photograph with the
                aligned crop below before attributing the error solely
                to ArcFace.
                """
            )

        show_local_pair_images(
            selected_pair,
            show_raw=True,
        )

        st.caption(
            """
            The verification score describes the complete frozen
            pipeline. Manual error analysis can identify plausible
            upstream contributors, but the test pair remains in the
            reported 96.8% end-to-end result.
            """
        )


# ============================================================
# QUALITATIVE EXAMPLES
# ============================================================

with examples_tab:

    section_label(
        "Qualitative QA"
    )

    st.subheader(
        "Representative verification decisions"
    )

    gold_callout(
        """
        “Genuine” and “impostor” below refer to the
        <strong>LFW protocol label</strong>. They do not mean that the
        displayed aligned crop was independently verified by a human.
        """
    )

    if example_manifest.empty:

        st.info(
            "Qualitative example manifest unavailable."
        )

    else:

        heading = {
            "TP":
                "Correct match — protocol genuine pair accepted",
            "TN":
                "Correct rejection — protocol impostor pair rejected",
            "FP":
                "Pipeline false match — protocol impostor pair accepted",
            "FN":
                "Pipeline false non-match — protocol genuine pair rejected",
        }

        for outcome in [
            "TP",
            "TN",
            "FP",
            "FN",
        ]:

            st.markdown(
                f"### {heading[outcome]}"
            )

            subset = example_manifest[
                example_manifest[
                    "outcome"
                ].eq(
                    outcome
                )
            ]

            for _, row in subset.iterrows():

                pair_index = int(
                    row[
                        "pair_index"
                    ]
                )

                st.markdown(
                    f"""
                    **Pair {pair_index}** ·
                    similarity **{row['cosine_similarity']:.4f}** ·
                    threshold **{row['threshold']:.4f}** ·
                    protocol label **{row['actual_pair_type']}** ·
                    pipeline decision **{row['model_decision']}**
                    """
                )

                show_local_pair_images(
                    pair_index,
                    show_raw=False,
                )

                st.divider()

    st.caption(
        """
        These examples are qualitative QA only. They are not
        additional estimates of model performance.
        """
    )


# ============================================================
# METHODOLOGY
# ============================================================

with methodology_tab:

    section_label(
        "Frozen protocol"
    )

    st.subheader(
        "Reference A evaluation pipeline"
    )

    st.code(
        """
Original LFW image
        |
        v
RetinaFace detection
        |
        v
Primary-face selection
        |
        v
Five-point project alignment
        |
        v
Frozen aligned crop
        |
        v
ArcFace
        |
        v
512-D embedding
        |
        v
L2 normalisation
        |
        v
Cosine similarity
        |
        +---------------------------+
        |                           |
        v                           v
View 1 development             View 1 test
2,200 pairs                    1,000 pairs
        |                           |
        v                           |
Select threshold                   |
        |                           |
        +--------- freeze ----------+
                                    |
                                    v
                    performance + fairness audit
        """,
        language=None,
    )

    st.markdown(
        f"""
        **Model role:** pretrained recognition reference  
        **Dataset:** LFW  
        **Unique images:** {summary['unique_images']:,}  
        **Development pairs:** {summary['train_pairs']:,}  
        **Test pairs:** {summary['test_pairs']:,}  
        **Embedding dimension:** {summary['embedding_dimension']}  
        **Similarity:** cosine similarity  
        **Frozen threshold:** {summary['threshold']:.6f}  
        **Threshold source:** View 1 development data only  
        **View 2 used:** No
        """
    )

    gold_callout(
        """
        View 2 remains sealed and was not used for preprocessing,
        threshold selection, model selection or any result presented
        on this page.
        """
    )

    st.divider()

    section_label(
        "Interpretation"
    )

    left, right = (
        st.columns(
            2
        )
    )

    with left:

        card(
            "Reference A",
            "Pretrained ArcFace",
            """
            External reference evidence used to validate the
            recognition and evaluation pipeline.
            """,
        )

    with right:

        card(
            "M0",
            "Project-trained baseline",
            """
            Ordinary unmitigated recognition model. M1, M2 and M3
            will be compared against M0 rather than against ArcFace.
            """,
        )

    blue_callout(
        """
        The principal mitigation experiment remains
        <strong>M0 vs M1 vs M2 vs M3</strong>.
        Reference A establishes context and an end-to-end recognition
        benchmark.
        """
    )
    