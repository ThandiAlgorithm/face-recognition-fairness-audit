from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from model_audit_data import (
    fairface_comparison_figure,
    load_labelled_dataset_metrics,
    load_lfw_cross_model_agreement,
    load_lfw_distribution_distance,
    load_lfw_identity_consistency,
    load_lfw_modal_agreement,
    load_lfw_model_assigned_distributions,
    load_qualitative_manifest,
    model_audit_artifact,
    qualitative_contact_sheet,
    track_a_figure,
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

WITS_NAVY = "#002B5C"
WITS_BLUE = "#0067B1"
WITS_GOLD = "#F4C542"

MODEL_COLOURS = {
    "FairFace ResNet-34": WITS_BLUE,
    "DeepFace": WITS_GOLD,
}


# ============================================================
# HELPERS
# ============================================================

def metric_value(
    frame: pd.DataFrame,
    dataset: str,
    model: str,
    task: str,
    metric: str,
) -> float | None:

    row = frame[
        frame["dataset"].eq(dataset)
        & frame["model"].eq(model)
        & frame["task"].eq(task)
    ]

    if row.empty:
        return None

    return float(
        row.iloc[0][metric]
    )


def pct(
    value: float | None,
) -> str:

    if value is None:
        return "Unavailable"

    return f"{value:.2%}"


def show_image_if_available(
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


# ============================================================
# LOAD DATA
# ============================================================

metrics = load_labelled_dataset_metrics()
lfw_agreement = load_lfw_cross_model_agreement()
lfw_distributions = load_lfw_model_assigned_distributions()
lfw_distance = load_lfw_distribution_distance()
lfw_identity = load_lfw_identity_consistency()
lfw_modal = load_lfw_modal_agreement()
qualitative_manifest = load_qualitative_manifest()


if metrics.empty:
    st.error(
        """
        Track A comparison outputs were not found. Expected results under
        results/model_audit/model_comparison/track_a_final/.
        """
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

hero(
    "Demographic Model Audit",
    """
    Track A compares two demographic attribute classifiers across
    FairFace, UTKFace and LFW. Labelled datasets quantify classifier
    performance; LFW is used to test how strongly inferred demographic
    measurements depend on the annotation model.
    """,
    eyebrow="Stage 2A · Demographic attribute classification",
)

gold_callout(
    """
    <strong>This page is not yet the facial-recognition baseline.</strong>
    FairFace ResNet-34 and DeepFace are being audited here as
    demographic measurement systems. Facial verification is the next
    experimental track.
    """
)


# ============================================================
# TOP SUMMARY
# ============================================================

section_label("Track A status")

st.subheader("Two models · three datasets · one controlled comparison")

s1, s2, s3, s4 = st.columns(4)

s1.metric(
    "Models evaluated",
    "2",
)

s2.metric(
    "FairFace validation",
    "10,954",
)

s3.metric(
    "UTKFace aligned",
    "24,054",
)

s4.metric(
    "LFW",
    "13,233",
)

st.caption(
    """
    FairFace and DeepFace were compared on the same images within each
    dataset. Race taxonomies were harmonised before direct comparison.
    """
)


# ============================================================
# TABS
# ============================================================

(
    summary_tab,
    fairface_tab,
    utkface_tab,
    lfw_tab,
    qualitative_tab,
    verification_tab,
) = st.tabs(
    [
        "Track A Summary",
        "FairFace",
        "UTKFace",
        "LFW Sensitivity",
        "Qualitative Audit",
        "Track B Next",
    ]
)


# ============================================================
# SUMMARY
# ============================================================

with summary_tab:

    section_label("Labelled evaluation")

    st.subheader("Full-dataset classifier performance")

    summary = metrics.copy()

    summary["Task"] = summary["task"].replace(
        {
            "race_6class": "Race — harmonised 6-class",
            "race_4class": "Race — harmonised 4-class",
            "gender": "Gender",
        }
    )

    summary["Dataset"] = summary["dataset"]
    summary["Model"] = summary["model"]

    summary["Accuracy (%)"] = (
        summary["accuracy"]
        * 100
    )

    summary["Macro F1 (%)"] = (
        summary["f1_macro"]
        * 100
    )

    summary["Macro precision (%)"] = (
        summary["precision_macro"]
        * 100
    )

    summary["Macro recall (%)"] = (
        summary["recall_macro"]
        * 100
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
            "Accuracy (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Macro precision (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Macro recall (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Macro F1 (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
        },
    )

    left, right = st.columns(2)

    with left:
        show_image_if_available(
            track_a_figure(
                "01_gender_accuracy_comparison.png"
            )
        )

    with right:
        show_image_if_available(
            track_a_figure(
                "02_race_accuracy_comparison.png"
            )
        )

    gold_callout(
        """
        The race bars use different harmonised evaluation spaces:
        FairFace uses a common <strong>six-class</strong> taxonomy,
        whereas UTKFace uses a comparable <strong>four-class</strong>
        taxonomy. The chart is therefore useful for within-dataset model
        comparison; it should not be interpreted as a single identical
        race task across both datasets.
        """
    )

    section_label("Key performance result")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            ### FairFace validation

            **FairFace ResNet-34**

            - Race accuracy: **77.65%**
            - Race macro F1: **75.24%**
            - Gender accuracy: **94.39%**
            - Gender macro F1: **94.37%**

            **DeepFace**

            - Race accuracy: **65.75%**
            - Race macro F1: **61.53%**
            - Gender accuracy: **70.17%**
            - Gender macro F1: **66.60%**
            """
        )

    with c2:

        st.markdown(
            """
            ### UTKFace

            **FairFace ResNet-34**

            - Race accuracy: **76.48%**
            - Race macro F1: **81.88%**
            - Gender accuracy: **94.35%**
            - Gender macro F1: **94.34%**

            **DeepFace**

            - Race accuracy: **75.16%**
            - Race macro F1: **77.81%**
            - Gender accuracy: **86.23%**
            - Gender macro F1: **86.03%**
            """
        )

    blue_callout(
        """
        Representation adequacy alone does not determine classifier
        performance. The same demographic task can change substantially
        across models and datasets even when subgroup sample sizes are
        large.
        """
    )


# ============================================================
# FAIRFACE
# ============================================================

with fairface_tab:

    section_label("Independent labels")

    st.subheader("FairFace held-out validation set")

    st.write(
        """
        Both classifiers were evaluated on the same **10,954**
        FairFace validation images. Race categories were harmonised to
        a common six-class evaluation space for direct comparison.
        """
    )

    fairface_metrics = metrics[
        metrics["dataset"].eq(
            "FairFace validation"
        )
    ].copy()

    display = fairface_metrics[
        [
            "model",
            "task",
            "n",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
        ]
    ].copy()

    for column in [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]:
        display[column] *= 100

    display = display.rename(
        columns={
            "model": "Model",
            "task": "Task",
            "n": "Images",
            "accuracy": "Accuracy (%)",
            "precision_macro": "Macro precision (%)",
            "recall_macro": "Macro recall (%)",
            "f1_macro": "Macro F1 (%)",
            "f1_weighted": "Weighted F1 (%)",
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Accuracy (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Macro precision (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Macro recall (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Macro F1 (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Weighted F1 (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
        },
    )

    left, right = st.columns(2)

    with left:
        show_image_if_available(
            fairface_comparison_figure(
                "01_model_performance_comparison.png"
            ),
            "FairFace validation model comparison",
        )

    with right:
        show_image_if_available(
            fairface_comparison_figure(
                "02_gender_female_tpr_by_race.png"
            ),
            "Female TPR by race on FairFace validation",
        )

    st.divider()

    st.subheader("Confusion matrices")

    model_choice = st.radio(
        "Model",
        [
            "FairFace ResNet-34",
            "DeepFace",
        ],
        horizontal=True,
        key="fairface_confusion_model",
    )

    if model_choice == "FairFace ResNet-34":
        base = (
            "fairface_resnet34",
            "fairface_validation",
        )
    else:
        base = (
            "deepface",
            "fairface_validation",
        )

    cm_left, cm_right = st.columns(2)

    with cm_left:
        show_image_if_available(
            model_audit_artifact(
                *base,
                "race_confusion_matrix.png",
            ),
            f"{model_choice}: race confusion matrix",
        )

    with cm_right:
        show_image_if_available(
            model_audit_artifact(
                *base,
                "gender_confusion_matrix.png",
            ),
            f"{model_choice}: gender confusion matrix",
        )

    gold_callout(
        """
        These are ground-truth evaluation results because FairFace
        provides demographic labels for the held-out validation images.
        The results describe performance under this dataset and
        preprocessing configuration; they do not establish the causal
        source of any error disparity.
        """
    )


# ============================================================
# UTKFACE
# ============================================================

with utkface_tab:

    section_label("Cross-dataset evaluation")

    st.subheader("UTKFace aligned evaluation")

    st.write(
        """
        **24,054** UTKFace images were processed using the project's
        deterministic five-point alignment. **24,050** had usable gender
        labels, while **22,340** belonged to the four race categories
        directly comparable across the model and dataset taxonomies.
        """
    )

    utk_metrics = metrics[
        metrics["dataset"].eq(
            "UTKFace aligned"
        )
    ].copy()

    display = utk_metrics[
        [
            "model",
            "task",
            "n",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "f1_weighted",
        ]
    ].copy()

    for column in [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]:
        display[column] *= 100

    display = display.rename(
        columns={
            "model": "Model",
            "task": "Task",
            "n": "Images",
            "accuracy": "Accuracy (%)",
            "precision_macro": "Macro precision (%)",
            "recall_macro": "Macro recall (%)",
            "f1_macro": "Macro F1 (%)",
            "f1_weighted": "Weighted F1 (%)",
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Accuracy (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Macro precision (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Macro recall (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Macro F1 (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Weighted F1 (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
        },
    )

    st.markdown(
        """
        ### Cross-dataset result

        Gender performance remained high for FairFace ResNet-34
        (**94.35% accuracy**) while DeepFace decreased to **86.23%**.

        Four-class race accuracy was much closer:
        **76.48%** for FairFace ResNet-34 and **75.16%** for DeepFace.
        However, macro F1 remained separated at **81.88%** and
        **77.81%**, respectively.
        """
    )

    blue_callout(
        """
        Both demographic models showed relatively low UTKFace Indian
        race-classification accuracy despite thousands of Indian-labelled
        examples. The result therefore cannot be explained simply as a
        small-sample problem. Taxonomy, domain, model training data and
        preprocessing remain plausible contributors.
        """
    )

    st.subheader("Race confusion matrices")

    left, right = st.columns(2)

    with left:
        show_image_if_available(
            model_audit_artifact(
                "fairface_resnet34",
                "utkface_aligned",
                "race_confusion_matrix_harmonised.png",
            ),
            "FairFace ResNet-34 — UTKFace",
        )

    with right:
        show_image_if_available(
            model_audit_artifact(
                "deepface",
                "utkface_aligned",
                "race_confusion_matrix_harmonised.png",
            ),
            "DeepFace — UTKFace",
        )

    gold_callout(
        """
        Predictions falling outside the harmonised four-class race
        taxonomy were tracked rather than silently forced into a
        comparable category. Taxonomy incompatibility is therefore part
        of the measurement audit rather than being hidden as an ordinary
        classification error.
        """
    )


# ============================================================
# LFW
# ============================================================

with lfw_tab:

    section_label("Prediction-only analysis")

    st.subheader("Demographic measurement sensitivity on LFW")

    gold_callout(
        """
        LFW does not provide independent race or gender ground truth in
        the canonical project table. The analysis below therefore asks
        whether two demographic classifiers assign the
        <strong>same labels to the same images</strong>. It does not
        identify which classifier is demographically correct.
        """
    )

    if not lfw_agreement.empty:

        race_row = lfw_agreement[
            lfw_agreement["task"].eq(
                "race"
            )
        ].iloc[0]

        gender_row = lfw_agreement[
            lfw_agreement["task"].eq(
                "gender"
            )
        ].iloc[0]

        a, b, c, d = st.columns(4)

        a.metric(
            "Race agreement",
            f"{race_row['agreement_rate']:.2%}",
        )

        b.metric(
            "Race disagreement",
            f"{race_row['disagreement_rate']:.2%}",
        )

        c.metric(
            "Gender agreement",
            f"{gender_row['agreement_rate']:.2%}",
        )

        d.metric(
            "Gender disagreement",
            f"{gender_row['disagreement_rate']:.2%}",
        )

        k1, k2 = st.columns(2)

        k1.metric(
            "Race Cohen's κ",
            f"{race_row['cohen_kappa']:.3f}",
        )

        k2.metric(
            "Gender Cohen's κ",
            f"{gender_row['cohen_kappa']:.3f}",
        )

    left, right = st.columns(2)

    with left:
        show_image_if_available(
            track_a_figure(
                "06_lfw_model_agreement.png"
            )
        )

    with right:
        show_image_if_available(
            track_a_figure(
                "08_lfw_cross_model_race_matrix.png"
            )
        )

    st.divider()

    section_label("Model-assigned composition")

    st.subheader("Same images, different demographic composition")

    if not lfw_distributions.empty:

        race_distribution = lfw_distributions[
            lfw_distributions["task"].eq(
                "race"
            )
        ].copy()

        race_distribution[
            "Proportion (%)"
        ] = (
            race_distribution["proportion"]
            * 100
        )

        fig = px.bar(
            race_distribution,
            x="category",
            y="Proportion (%)",
            color="model",
            barmode="group",
            color_discrete_map=MODEL_COLOURS,
            title="Model-assigned race composition on LFW",
        )

        fig.update_layout(
            xaxis_title="Predicted category",
            legend_title="Model",
            height=460,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        gender_distribution = lfw_distributions[
            lfw_distributions["task"].eq(
                "gender"
            )
        ].copy()

        gender_distribution[
            "Proportion (%)"
        ] = (
            gender_distribution["proportion"]
            * 100
        )

        fig = px.bar(
            gender_distribution,
            x="category",
            y="Proportion (%)",
            color="model",
            barmode="group",
            color_discrete_map=MODEL_COLOURS,
            title="Model-assigned gender composition on LFW",
        )

        fig.update_layout(
            xaxis_title="Predicted category",
            legend_title="Model",
            height=400,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        """
        The aggregate race composition is also model-dependent. For
        example:

        - FairFace ResNet-34 assigned **18.45%** of LFW images to its
          Middle Eastern category, compared with **9.23%** for DeepFace.
        - Latino/Hispanic assignments were **3.76%** for FairFace
          ResNet-34 and **8.01%** for DeepFace.
        - Male assignments were **78.27%** and **81.87%**, respectively.

        These values are classifier outputs, not verified demographic
        attributes.
        """
    )

    st.divider()

    section_label("Repeated identities")

    st.subheader("Does one identity receive stable demographic labels?")

    if not lfw_identity.empty:

        consistency = lfw_identity.copy()

        consistency[
            "Mean race consistency (%)"
        ] = (
            consistency[
                "mean_race_consistency"
            ]
            * 100
        )

        consistency[
            "Mean gender consistency (%)"
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
                    "Mean race consistency (%)",
                    "Mean gender consistency (%)",
                ]
            ].rename(
                columns={
                    "model": "Model",
                    "repeated_identities": "Repeated identities",
                    "images_in_repeated_identities": "Images",
                }
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Mean race consistency (%)": st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
                "Mean gender consistency (%)": st.column_config.NumberColumn(
                    format="%.2f%%"
                ),
            },
        )

    if not lfw_modal.empty:

        modal_display = lfw_modal.copy()

        modal_display[
            "Modal agreement (%)"
        ] = (
            modal_display[
                "modal_label_agreement_rate"
            ]
            * 100
        )

        st.dataframe(
            modal_display[
                [
                    "task",
                    "repeated_identities",
                    "Modal agreement (%)",
                ]
            ].rename(
                columns={
                    "task": "Task",
                    "repeated_identities": "Repeated identities",
                }
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Modal agreement (%)": st.column_config.NumberColumn(
                    format="%.2f%%"
                )
            },
        )

    show_image_if_available(
        track_a_figure(
            "07_lfw_identity_consistency_comparison.png"
        )
    )

    blue_callout(
        """
        Among the 1,680 repeated LFW identities, the models agreed on
        the modal race assignment for <strong>75.00%</strong> of
        identities and on the modal gender assignment for
        <strong>96.49%</strong>. Race assignment is therefore materially
        more sensitive to the demographic classifier than gender
        assignment under this experiment.
        """
    )


# ============================================================
# QUALITATIVE AUDIT
# ============================================================

with qualitative_tab:

    section_label("Manual QA")

    st.subheader("90 deterministic qualitative examples")

    st.write(
        """
        The qualitative audit uses the **same selected image for both
        models**, making prediction differences directly inspectable.

        FairFace and UTKFace examples are stratified across available
        race × gender groups. LFW deliberately includes race
        disagreements, gender disagreements and full agreements.
        """
    )

    gold_callout(
        """
        The 30-image samples are qualitative error-analysis panels,
        <strong>not new performance estimates</strong>. In particular,
        the LFW sample intentionally oversamples disagreements and is not
        representative of the full 13,233-image agreement rate.
        """
    )

    dataset_choice = st.selectbox(
        "Dataset",
        [
            "FairFace",
            "UTKFace",
            "LFW",
        ],
        key="qualitative_dataset",
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
                f"{dataset_choice} qualitative audit "
                f"— page {page} of 3"
            ),
            use_container_width=True,
        )
    else:
        st.warning(
            "Contact sheet not available."
        )

    dataset_manifest = load_qualitative_manifest(
        dataset_key
    )

    if not dataset_manifest.empty:

        st.subheader("Selection manifest")

        selection_summary = (
            dataset_manifest[
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
            selection_summary,
            hide_index=True,
            use_container_width=True,
        )

        with st.expander(
            "View the 30-image audit manifest"
        ):

            manifest_columns = [
                column
                for column in [
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
                in dataset_manifest.columns
            ]

            st.dataframe(
                dataset_manifest[
                    manifest_columns
                ],
                hide_index=True,
                use_container_width=True,
                height=500,
            )

    st.caption(
        """
        Before external public deployment, the redistribution terms of
        the source face datasets should be checked for displayed example
        images. The quantitative audit outputs do not require image
        redistribution.
        """
    )


# ============================================================
# TRACK B
# ============================================================

with verification_tab:

    section_label("Next experiment")

    st.subheader("Track B — facial verification")

    blue_callout(
        """
        Track A establishes that demographic annotation itself can be
        model-dependent. Track B will now test whether a fixed
        face-recognition system produces different verification error
        rates across the resulting groups.
        """
    )

    left, right = st.columns(2)

    with left:
        card(
            "01",
            "Embedding model",
            """
            Convert each aligned LFW face into a fixed-dimensional
            identity embedding using a pretrained recognition model.
            """,
        )

    with right:
        card(
            "02",
            "Verification protocol",
            """
            Compare genuine and impostor pairs using a threshold selected
            on development data and preserved for final evaluation.
            """,
        )

    st.code(
        """
LFW image A                    LFW image B
     |                              |
     v                              v
Aligned face                    Aligned face
     |                              |
     v                              v
Recognition model              Recognition model
     |                              |
     v                              v
Embedding A                    Embedding B
          \\                    /
           \\                  /
            similarity score
                   |
                   v
             fixed threshold
                   |
             +-----+-----+
             |           |
           MATCH      NON-MATCH
        """,
        language=None,
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "ROC / AUC",
        "Pending",
    )

    b.metric(
        "FMR",
        "Pending",
    )

    c.metric(
        "FNMR",
        "Pending",
    )

    d.metric(
        "EER",
        "Pending",
    )

    st.progress(
        0.25
    )

    st.caption(
        """
        Candidate recognition models include ArcFace, FaceNet512 and
        SFace. Embeddings should be generated once per image and reused
        across pair-level analyses.
        """
    )
    