from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model_audit_data import (
    fairface_comparison_figure,
    load_labelled_dataset_metrics,
    load_lfw_cross_model_agreement,
    load_lfw_identity_consistency,
    load_lfw_modal_agreement,
    load_lfw_model_assigned_distributions,
    load_qualitative_manifest,
    qualitative_contact_sheet,
    track_a_figure,
)

from ui import (
    evidence_note,
    finding_card,
    metric_row,
    page_header,
    section_header,
)


# ============================================================
# CONSTANTS
# ============================================================

WITS_NAVY = "#002B5C"
WITS_BLUE = "#0067B1"
WITS_GOLD = "#D8AA19"
WITS_LIGHT_BLUE = "#DCEAF5"
WITS_PALE_BLUE = "#F4F8FB"

MODEL_COLOURS = {
    "FairFace ResNet-34": WITS_BLUE,
    "DeepFace": WITS_GOLD,
}


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "model_audit"
)


# ============================================================
# HELPERS
# ============================================================

def show_image(
    path: Path,
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


def percentage_table(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    output = frame.copy()

    output["Task"] = (
        output["task"]
        .replace(
            {
                "race_6class": "Race — harmonised 6-class",
                "race_4class": "Race — harmonised 4-class",
                "gender": "Gender",
            }
        )
    )

    output["Dataset"] = (
        output["dataset"]
    )

    output["Model"] = (
        output["model"]
    )

    for source, target in [
        (
            "accuracy",
            "Accuracy (%)",
        ),
        (
            "precision_macro",
            "Macro precision (%)",
        ),
        (
            "recall_macro",
            "Macro recall (%)",
        ),
        (
            "f1_macro",
            "Macro F1 (%)",
        ),
    ]:

        output[target] = (
            output[source]
            * 100
        )

    return output


def clean_class_name(
    value,
) -> str:

    text = (
        str(value)
        .strip()
        .replace("_", " ")
    )

    replacements = {
        "Latino Hispanic": "Latino / Hispanic",
        "Latino hispanic": "Latino / Hispanic",
        "Middle eastern": "Middle Eastern",
        "East asian": "East Asian",
        "Southeast asian": "Southeast Asian",
        "South east asian": "Southeast Asian",
        "Out of scope": "Out of scope",
        "out of scope": "Out of scope",
    }

    return replacements.get(
        text,
        text,
    )


@st.cache_data(
    show_spinner=False
)
def load_confusion_matrix_csv(
    path_string: str,
) -> tuple[pd.DataFrame, list[str]]:

    path = Path(
        path_string
    )

    if not path.exists():
        return pd.DataFrame(), []

    try:

        frame = pd.read_csv(
            path
        )

    except Exception:

        return pd.DataFrame(), []

    if frame.empty:
        return pd.DataFrame(), []

    first_column = (
        frame.columns[0]
    )

    first_values_numeric = (
        pd.to_numeric(
            frame[first_column],
            errors="coerce",
        )
    )

    first_column_looks_like_labels = (
        str(first_column)
        .lower()
        .startswith("unnamed")
        or first_values_numeric
        .isna()
        .any()
    )

    if first_column_looks_like_labels:

        row_labels = (
            frame[first_column]
            .astype(str)
            .map(clean_class_name)
            .tolist()
        )

        matrix = frame.drop(
            columns=[
                first_column
            ]
        )

    else:

        matrix = frame.copy()

        row_labels = [
            clean_class_name(
                value
            )
            for value in matrix.index
        ]

    matrix = (
        matrix
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .fillna(0)
    )

    matrix.columns = [
        clean_class_name(
            column
        )
        for column in matrix.columns
    ]

    if (
        not row_labels
        or len(row_labels)
        != len(matrix)
    ):

        row_labels = [
            clean_class_name(
                column
            )
            for column in matrix.columns[
                :len(matrix)
            ]
        ]

    return (
        matrix,
        row_labels,
    )


def show_percentage_confusion_matrix(
    path: Path,
    title: str,
    taxonomy_note: str | None = None,
) -> None:

    counts, row_labels = (
        load_confusion_matrix_csv(
            str(path)
        )
    )

    if counts.empty:

        st.info(
            "Confusion-matrix data are not available "
            f"for {path.name}."
        )

        return

    row_totals = (
        counts
        .sum(axis=1)
        .replace(
            0,
            pd.NA,
        )
    )

    percentages = (
        counts
        .div(
            row_totals,
            axis=0,
        )
        * 100
    ).fillna(0)

    fig = go.Figure(
        data=go.Heatmap(
            z=percentages.values,
            x=list(
                percentages.columns
            ),
            y=row_labels,
            customdata=counts.values,
            zmin=0,
            zmax=100,
            colorscale=[
                [
                    0.00,
                    "#F7FAFC",
                ],
                [
                    0.10,
                    "#E5F0F7",
                ],
                [
                    0.30,
                    "#B6D5E8",
                ],
                [
                    0.55,
                    "#66A9CF",
                ],
                [
                    0.75,
                    "#2478AD",
                ],
                [
                    1.00,
                    WITS_NAVY,
                ],
            ],
            colorbar=dict(
                title="Row %",
                ticksuffix="%",
                thickness=14,
            ),
            hovertemplate=(
                "<b>True label:</b> %{y}<br>"
                "<b>Predicted label:</b> %{x}<br>"
                "<b>Row share:</b> %{z:.2f}%<br>"
                "<b>Raw count:</b> %{customdata:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    for row_index, true_label in enumerate(
        row_labels
    ):

        for column_index, predicted_label in enumerate(
            percentages.columns
        ):

            value = float(
                percentages.iloc[
                    row_index,
                    column_index,
                ]
            )

            text_colour = (
                "white"
                if value >= 48
                else WITS_NAVY
            )

            fig.add_annotation(
                x=predicted_label,
                y=true_label,
                text=f"{value:.1f}%",
                showarrow=False,
                font=dict(
                    size=14,
                    color=text_colour,
                ),
            )

    matrix_height = max(
        430,
        72 * len(
            row_labels
        ),
    )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(
                size=18,
            ),
        ),
        height=matrix_height,
        xaxis=dict(
            title="Predicted label",
            tickangle=-32,
            side="bottom",
        ),
        yaxis=dict(
            title="True label",
            autorange="reversed",
        ),
        margin=dict(
            l=90,
            r=40,
            t=80,
            b=110,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    if taxonomy_note:

        st.caption(
            taxonomy_note
        )


# ============================================================
# LOAD DATA
# ============================================================

metrics = (
    load_labelled_dataset_metrics()
)

lfw_agreement = (
    load_lfw_cross_model_agreement()
)

lfw_distributions = (
    load_lfw_model_assigned_distributions()
)

lfw_identity = (
    load_lfw_identity_consistency()
)

lfw_modal = (
    load_lfw_modal_agreement()
)


if metrics.empty:

    st.error(
        """
        Demographic-model comparison outputs were not found.
        """
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

page_header(
    "Demographic Measurement",
    (
        "A controlled comparison of FairFace ResNet-34 and DeepFace "
        "as demographic annotation systems across FairFace, UTKFace "
        "and LFW."
    ),
    "Demographic annotation audit",
)


# ============================================================
# SUMMARY
# ============================================================

metric_row(
    [
        (
            "Models evaluated",
            "2",
        ),
        (
            "FairFace validation",
            "10,954",
        ),
        (
            "UTKFace",
            "24,054",
        ),
        (
            "LFW",
            "13,233",
        ),
    ]
)

st.write(
    """
    FairFace and UTKFace provide independent demographic labels and can
    therefore be used to evaluate classification performance. LFW does
    not provide equivalent independent demographic ground truth in the
    canonical project table, so the LFW analysis measures agreement,
    disagreement and repeated-identity stability rather than
    demographic accuracy.
    """
)

st.divider()


# ============================================================
# KEY FINDINGS
# ============================================================

section_header(
    "Key findings",
    (
        "The strongest results from the demographic-measurement "
        "comparison."
    ),
)

f1, f2, f3 = st.columns(3)

with f1:

    finding_card(
        "FairFace validation",
        (
            "Race accuracy was 77.65% for FairFace ResNet-34 and "
            "65.75% for DeepFace. Gender accuracy was 94.39% and "
            "70.17%, respectively."
        ),
        (
            "The demographic classifier itself can materially affect "
            "the apparent performance of an annotation pipeline."
        ),
    )

with f2:

    finding_card(
        "UTKFace domain shift",
        (
            "Race accuracy narrowed to 76.48% versus 75.16%, while "
            "gender accuracy remained separated at 94.35% versus "
            "86.23%."
        ),
        (
            "The size of the model-performance gap depends on the "
            "evaluation dataset."
        ),
    )

with f3:

    finding_card(
        "LFW measurement sensitivity",
        (
            "The models disagreed on 25.29% of race assignments but "
            "only 4.63% of gender assignments on the same 13,233 "
            "images."
        ),
        (
            "Race-based fairness results can be more sensitive to the "
            "choice of annotation model than gender-based results."
        ),
    )

st.divider()


# ============================================================
# TABS
# ============================================================

(
    overview_tab,
    fairface_tab,
    utkface_tab,
    lfw_tab,
    qualitative_tab,
) = st.tabs(
    [
        "Overview",
        "FairFace",
        "UTKFace",
        "LFW Sensitivity",
        "Qualitative Review",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with overview_tab:

    section_header(
        "Performance comparison",
        (
            "Accuracy and class-balanced metrics are reported together "
            "so that majority classes do not dominate interpretation."
        ),
    )

    summary = percentage_table(
        metrics
    )

    st.dataframe(
        summary[
            [
                "Dataset",
                "Model",
                "Task",
                "n",
                "Accuracy (%)",
                "Macro precision (%)",
                "Macro recall (%)",
                "Macro F1 (%)",
            ]
        ].rename(
            columns={
                "n": "Evaluation images",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Accuracy (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
            "Macro precision (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
            "Macro recall (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
            "Macro F1 (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
        },
    )

    left, right = st.columns(2)

    with left:

        show_image(
            track_a_figure(
                "01_gender_accuracy_comparison.png"
            ),
            "Gender classification accuracy",
        )

    with right:

        show_image(
            track_a_figure(
                "02_race_accuracy_comparison.png"
            ),
            "Race classification accuracy",
        )

    st.warning(
        """
        The FairFace race evaluation uses a harmonised six-class
        comparison when the two models are evaluated directly, while
        UTKFace uses a four-class comparable taxonomy. These are valid
        within-dataset comparisons but should not be interpreted as one
        identical race-classification task.
        """
    )

    st.divider()

    section_header(
        "Why macro F1 matters"
    )

    left, right = st.columns(2)

    with left:

        evidence_note(
            "Accuracy",
            (
                "The proportion of all evaluated images classified "
                "correctly. Large classes contribute more strongly to "
                "the overall result."
            ),
        )

    with right:

        evidence_note(
            "Macro F1",
            (
                "F1 is calculated separately for every class and then "
                "averaged equally, making poor performance on smaller "
                "classes more visible."
            ),
        )


# ============================================================
# FAIRFACE
# ============================================================

with fairface_tab:

    section_header(
        "FairFace validation",
        (
            "Both demographic classifiers are evaluated on the same "
            "held-out 10,954-image validation population."
        ),
    )

    fairface_metrics = (
        metrics[
            metrics[
                "dataset"
            ].eq(
                "FairFace validation"
            )
        ]
        .copy()
    )

    display = percentage_table(
        fairface_metrics
    )

    st.dataframe(
        display[
            [
                "Model",
                "Task",
                "n",
                "Accuracy (%)",
                "Macro precision (%)",
                "Macro recall (%)",
                "Macro F1 (%)",
            ]
        ].rename(
            columns={
                "n": "Images",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Accuracy (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
            "Macro precision (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
            "Macro recall (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
            "Macro F1 (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
        },
    )

    metric_row(
        [
            (
                "FairFace model · race",
                "77.65%",
            ),
            (
                "DeepFace · race",
                "65.75%",
            ),
            (
                "FairFace model · gender",
                "94.39%",
            ),
            (
                "DeepFace · gender",
                "70.17%",
            ),
        ]
    )

    left, right = st.columns(2)

    with left:

        show_image(
            fairface_comparison_figure(
                "01_model_performance_comparison.png"
            ),
            "FairFace validation model comparison",
        )

    with right:

        show_image(
            fairface_comparison_figure(
                "02_gender_female_tpr_by_race.png"
            ),
            "Female true-positive rate by labelled race group",
        )

    st.info(
        """
        These are labelled evaluation results because FairFace provides
        demographic annotations for the held-out validation images.
        They describe performance under this dataset, taxonomy and
        preprocessing configuration rather than establishing the
        causal origin of a disparity.
        """
    )

    st.divider()

    section_header(
        "Percentage confusion matrices",
        (
            "Each true-class row is normalised to 100%, allowing "
            "classification behaviour to be compared directly across "
            "classes with different sample sizes."
        ),
    )

    model_choice = st.segmented_control(
        "Model",
        [
            "FairFace ResNet-34",
            "DeepFace",
        ],
        default="FairFace ResNet-34",
        key="fairface_cm_model",
    )

    if model_choice == "FairFace ResNet-34":

        matrix_root = (
            MODEL_RESULTS_ROOT
            / "fairface_resnet34"
            / "fairface_validation"
        )

    else:

        matrix_root = (
            MODEL_RESULTS_ROOT
            / "deepface"
            / "fairface_validation"
        )

    race_path = (
        matrix_root
        / "race_confusion_matrix.csv"
    )

    gender_path = (
        matrix_root
        / "gender_confusion_matrix.csv"
    )

    show_percentage_confusion_matrix(
        race_path,
        (
            f"{model_choice} · FairFace validation · "
            f"race classification"
        ),
        (
            "Taxonomy note: the original FairFace ResNet-34 audit may "
            "show the native seven-category FairFace taxonomy. Direct "
            "FairFace-versus-DeepFace summary metrics use the common "
            "harmonised six-class comparison where required."
        ),
    )

    show_percentage_confusion_matrix(
        gender_path,
        (
            f"{model_choice} · FairFace validation · "
            f"gender classification"
        ),
    )

    st.caption(
        """
        Diagonal percentages are correct classifications. Off-diagonal
        percentages show which alternative labels were assigned to each
        true class. Hover over a cell to see both percentage and raw
        count.
        """
    )


# ============================================================
# UTKFACE
# ============================================================

with utkface_tab:

    section_header(
        "UTKFace",
        (
            "Cross-dataset evaluation tests how both demographic "
            "classifiers behave under a different image population "
            "and label system."
        ),
    )

    st.write(
        """
        **24,054** UTKFace images were processed using the project's
        deterministic alignment pipeline. **24,050** had usable gender
        labels, while **22,340** belonged to the four race categories
        directly comparable between the model and dataset taxonomies.
        """
    )

    utk_metrics = (
        metrics[
            metrics[
                "dataset"
            ].eq(
                "UTKFace aligned"
            )
        ]
        .copy()
    )

    display = percentage_table(
        utk_metrics
    )

    st.dataframe(
        display[
            [
                "Model",
                "Task",
                "n",
                "Accuracy (%)",
                "Macro precision (%)",
                "Macro recall (%)",
                "Macro F1 (%)",
            ]
        ].rename(
            columns={
                "n": "Images",
            }
        ),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Accuracy (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
            "Macro precision (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
            "Macro recall (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
            "Macro F1 (%)": (
                st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            ),
        },
    )

    metric_row(
        [
            (
                "FairFace model · race",
                "76.48%",
            ),
            (
                "DeepFace · race",
                "75.16%",
            ),
            (
                "FairFace model · gender",
                "94.35%",
            ),
            (
                "DeepFace · gender",
                "86.23%",
            ),
        ]
    )

    st.info(
        """
        The race-accuracy gap between the models becomes much smaller
        on UTKFace, while the gender-performance gap remains visible.
        The relative difference between demographic classifiers is
        therefore dependent on the evaluation domain.
        """
    )

    st.divider()

    section_header(
        "Percentage confusion matrices",
        (
            "The harmonised race evaluation retains predictions outside "
            "the common taxonomy as an explicit out-of-scope category."
        ),
    )

    race_left, race_right = (
        st.columns(2)
    )

    with race_left:

        show_percentage_confusion_matrix(
            (
                MODEL_RESULTS_ROOT
                / "fairface_resnet34"
                / "utkface_aligned"
                / "race_confusion_matrix_harmonised.csv"
            ),
            (
                "FairFace ResNet-34 · UTKFace · "
                "harmonised race"
            ),
            (
                "True UTKFace categories are evaluated against the "
                "common comparable race taxonomy."
            ),
        )

    with race_right:

        show_percentage_confusion_matrix(
            (
                MODEL_RESULTS_ROOT
                / "deepface"
                / "utkface_aligned"
                / "race_confusion_matrix_harmonised.csv"
            ),
            (
                "DeepFace · UTKFace · harmonised race"
            ),
            (
                "Predictions outside the comparable four-class taxonomy "
                "remain visible rather than being reassigned."
            ),
        )

    st.markdown(
        "### Gender classification"
    )

    gender_left, gender_right = (
        st.columns(2)
    )

    with gender_left:

        show_percentage_confusion_matrix(
            (
                MODEL_RESULTS_ROOT
                / "fairface_resnet34"
                / "utkface_aligned"
                / "gender_confusion_matrix.csv"
            ),
            (
                "FairFace ResNet-34 · UTKFace · gender"
            ),
        )

    with gender_right:

        show_percentage_confusion_matrix(
            (
                MODEL_RESULTS_ROOT
                / "deepface"
                / "utkface_aligned"
                / "gender_confusion_matrix.csv"
            ),
            (
                "DeepFace · UTKFace · gender"
            ),
        )

    st.caption(
        """
        All matrices shown here are row-normalised. Each true-class row
        therefore totals approximately 100%.
        """
    )


# ============================================================
# LFW
# ============================================================

with lfw_tab:

    section_header(
        "LFW measurement sensitivity",
        (
            "The same 13,233 images are annotated by both demographic "
            "classifiers and their outputs are compared directly."
        ),
    )

    st.warning(
        """
        LFW does not provide independent race or gender ground truth in
        the canonical project table. The results below measure
        cross-model agreement and stability rather than demographic
        classification accuracy.
        """
    )

    if not lfw_agreement.empty:

        race_rows = (
            lfw_agreement[
                lfw_agreement[
                    "task"
                ].eq("race")
            ]
        )

        gender_rows = (
            lfw_agreement[
                lfw_agreement[
                    "task"
                ].eq("gender")
            ]
        )

        if (
            not race_rows.empty
            and not gender_rows.empty
        ):

            race_row = (
                race_rows.iloc[0]
            )

            gender_row = (
                gender_rows.iloc[0]
            )

            metric_row(
                [
                    (
                        "Race agreement",
                        f"{race_row['agreement_rate']:.2%}",
                    ),
                    (
                        "Race disagreement",
                        f"{race_row['disagreement_rate']:.2%}",
                    ),
                    (
                        "Gender agreement",
                        f"{gender_row['agreement_rate']:.2%}",
                    ),
                    (
                        "Gender disagreement",
                        f"{gender_row['disagreement_rate']:.2%}",
                    ),
                ]
            )

            k1, k2 = st.columns(2)

            with k1:

                evidence_note(
                    "Race Cohen's κ",
                    (
                        f"{race_row['cohen_kappa']:.3f}"
                    ),
                )

            with k2:

                evidence_note(
                    "Gender Cohen's κ",
                    (
                        f"{gender_row['cohen_kappa']:.3f}"
                    ),
                )

    left, right = st.columns(2)

    with left:

        show_image(
            track_a_figure(
                "06_lfw_model_agreement.png"
            ),
            "Cross-model demographic agreement",
        )

    with right:

        show_image(
            track_a_figure(
                "08_lfw_cross_model_race_matrix.png"
            ),
            "Cross-model race assignment matrix",
        )

    st.divider()

    section_header(
        "Model-assigned composition",
        (
            "Changing the demographic classifier changes the inferred "
            "composition of the same image population."
        ),
    )

    if not lfw_distributions.empty:

        race_distribution = (
            lfw_distributions[
                lfw_distributions[
                    "task"
                ].eq("race")
            ]
            .copy()
        )

        race_distribution[
            "Proportion (%)"
        ] = (
            race_distribution[
                "proportion"
            ]
            * 100
        )

        fig = px.bar(
            race_distribution,
            x="category",
            y="Proportion (%)",
            color="model",
            barmode="group",
            color_discrete_map=MODEL_COLOURS,
            title=(
                "Model-assigned race composition on LFW"
            ),
        )

        fig.update_layout(
            height=450,
            xaxis_title="Assigned category",
            yaxis_title="Share of images (%)",
            legend_title="Model",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        gender_distribution = (
            lfw_distributions[
                lfw_distributions[
                    "task"
                ].eq("gender")
            ]
            .copy()
        )

        gender_distribution[
            "Proportion (%)"
        ] = (
            gender_distribution[
                "proportion"
            ]
            * 100
        )

        fig = px.bar(
            gender_distribution,
            x="category",
            y="Proportion (%)",
            color="model",
            barmode="group",
            color_discrete_map=MODEL_COLOURS,
            title=(
                "Model-assigned gender composition on LFW"
            ),
        )

        fig.update_layout(
            height=390,
            xaxis_title="Assigned category",
            yaxis_title="Share of images (%)",
            legend_title="Model",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.info(
        """
        The inferred demographic composition of LFW is therefore not
        independent of the annotation system. Fairness analyses that
        use these inferred groups inherit this measurement uncertainty.
        """
    )

    st.divider()

    section_header(
        "Repeated-identity stability",
        (
            "Repeated photographs of the same LFW identity provide a "
            "second test of demographic-label stability."
        ),
    )

    if not lfw_identity.empty:

        consistency = (
            lfw_identity.copy()
        )

        consistency[
            "Race consistency (%)"
        ] = (
            consistency[
                "mean_race_consistency"
            ]
            * 100
        )

        consistency[
            "Gender consistency (%)"
        ] = (
            consistency[
                "mean_gender_consistency"
            ]
            * 100
        )

        st.dataframe(
            consistency[
                [
                    "model",
                    "repeated_identities",
                    "images_in_repeated_identities",
                    "Race consistency (%)",
                    "Gender consistency (%)",
                ]
            ].rename(
                columns={
                    "model": "Model",
                    "repeated_identities": (
                        "Repeated identities"
                    ),
                    "images_in_repeated_identities": (
                        "Images"
                    ),
                }
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Race consistency (%)": (
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    )
                ),
                "Gender consistency (%)": (
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    )
                ),
            },
        )

    if not lfw_modal.empty:

        modal = (
            lfw_modal.copy()
        )

        modal[
            "Modal agreement (%)"
        ] = (
            modal[
                "modal_label_agreement_rate"
            ]
            * 100
        )

        st.dataframe(
            modal[
                [
                    "task",
                    "repeated_identities",
                    "Modal agreement (%)",
                ]
            ].rename(
                columns={
                    "task": "Task",
                    "repeated_identities": (
                        "Repeated identities"
                    ),
                }
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Modal agreement (%)": (
                    st.column_config.NumberColumn(
                        format="%.2f%%"
                    )
                ),
            },
        )

    show_image(
        track_a_figure(
            "07_lfw_identity_consistency_comparison.png"
        )
    )

    st.info(
        """
        Across 1,680 identities with repeated images, the two models
        agreed on the modal race assignment for 75.00% of identities
        and on the modal gender assignment for 96.49%. Race assignment
        is therefore substantially more measurement-sensitive in this
        experiment.
        """
    )


# ============================================================
# QUALITATIVE REVIEW
# ============================================================

with qualitative_tab:

    section_header(
        "Qualitative review",
        (
            "A deterministic image sample supports manual inspection "
            "of demographic-model agreement and disagreement."
        ),
    )

    st.warning(
        """
        The qualitative samples are quality-assurance evidence rather
        than new performance estimates. The LFW sample deliberately
        contains a larger share of model disagreements than the full
        dataset.
        """
    )

    dataset_choice = (
        st.segmented_control(
            "Dataset",
            [
                "FairFace",
                "UTKFace",
                "LFW",
            ],
            default="FairFace",
            key="qualitative_dataset",
        )
    )

    dataset_key = {
        "FairFace": "fairface",
        "UTKFace": "utkface",
        "LFW": "lfw",
    }[
        dataset_choice
    ]

    page = st.radio(
        "Contact-sheet page",
        [
            1,
            2,
            3,
        ],
        horizontal=True,
        key="qualitative_page",
    )

    sheet = qualitative_contact_sheet(
        dataset_key,
        page,
    )

    if sheet.exists():

        st.image(
            str(sheet),
            caption=(
                f"{dataset_choice} qualitative review "
                f"— page {page}"
            ),
            use_container_width=True,
        )

    else:

        st.info(
            "Qualitative contact sheet is not available."
        )

    manifest = (
        load_qualitative_manifest(
            dataset_key
        )
    )

    if not manifest.empty:

        summary = (
            manifest[
                "selection_reason"
            ]
            .value_counts()
            .rename_axis(
                "Selection reason"
            )
            .reset_index(
                name="Images"
            )
        )

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True,
        )

        with st.expander(
            "View image-level qualitative manifest"
        ):

            columns = [
                column
                for column
                in [
                    "sample_index",
                    "image_id",
                    "selection_reason",
                    "ground_truth_race",
                    "ground_truth_gender",
                    "fairface_race",
                    "fairface_gender",
                    "deepface_race",
                    "deepface_gender",
                    "race_model_agreement",
                    "gender_model_agreement",
                ]
                if column
                in manifest.columns
            ]

            st.dataframe(
                manifest[
                    columns
                ],
                hide_index=True,
                use_container_width=True,
                height=470,
            )

    st.caption(
        """
        FairFace and UTKFace examples may be compared with dataset
        demographic labels. LFW examples should be interpreted as
        classifier assignments and model agreement because independent
        demographic ground truth is not available in the project table.
        """
    )
    