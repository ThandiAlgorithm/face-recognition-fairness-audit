from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui import (
    apply_wits_theme,
    blue_callout,
    gold_callout,
    hero,
    section_label,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "model_audit"
)

TRACK_A_ROOT = (
    MODEL_ROOT
    / "model_comparison"
    / "track_a_final"
)

QUALITATIVE_ROOT = (
    TRACK_A_ROOT
    / "qualitative_examples"
)


# ============================================================
# STYLE
# ============================================================

apply_wits_theme()


# ============================================================
# HELPERS
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(
    path_string: str,
) -> pd.DataFrame:

    path = Path(
        path_string
    )

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        low_memory=False,
    )


def rel(
    value: str,
) -> Path:

    return (
        MODEL_ROOT
        / value
    )


def figure_exists(
    path: Path,
) -> bool:

    return (
        path.exists()
        and path.is_file()
    )


def render_figure(
    path: Path,
    title: str,
    explanation: str,
) -> None:

    st.subheader(
        title
    )

    if figure_exists(
        path
    ):

        st.image(
            str(path),
            use_container_width=True,
        )

    else:

        st.warning(
            f"Figure not found: {path}"
        )

    st.markdown(
        f"""
        **What this shows**

        {explanation}
        """
    )

    try:

        relative = path.relative_to(
            PROJECT_ROOT
        )

        st.caption(
            f"Source: {relative}"
        )

    except ValueError:

        st.caption(
            f"Source: {path}"
        )

    st.divider()


def available_data_files(
    roots: list[Path],
) -> list[Path]:

    files = []

    for root in roots:

        if not root.exists():
            continue

        files.extend(
            root.rglob(
                "*.csv"
            )
        )

        files.extend(
            root.rglob(
                "*.json"
            )
        )

    return sorted(
        set(
            files
        ),
        key=lambda item: str(
            item
        ).lower(),
    )


def render_data_explorer(
    roots: list[Path],
    key_prefix: str,
) -> None:

    files = available_data_files(
        roots
    )

    if not files:

        return

    with st.expander(
        "Inspect the underlying tables and summaries"
    ):

        options = {
            str(
                path.relative_to(
                    MODEL_ROOT
                )
            ): path
            for path in files
        }

        selected_name = st.selectbox(
            "Result file",
            list(
                options.keys()
            ),
            key=(
                f"{key_prefix}_"
                "table_select"
            ),
        )

        selected = options[
            selected_name
        ]

        if (
            selected.suffix.lower()
            == ".csv"
        ):

            frame = load_csv(
                str(
                    selected
                )
            )

            st.caption(
                f"{len(frame):,} rows"
            )

            st.dataframe(
                frame,
                hide_index=True,
                use_container_width=True,
                height=520,
            )

        elif (
            selected.suffix.lower()
            == ".json"
        ):

            st.json(
                selected.read_text(
                    encoding="utf-8"
                )
            )


# ============================================================
# REPORT PACK DEFINITIONS
# ============================================================

PACKS = {
    # --------------------------------------------------------
    # FINAL TRACK A
    # --------------------------------------------------------
    "Track A — Consolidated comparison": {
        "intro": """
        This is the final two-model × three-dataset synthesis. Use this
        section first when presenting the overall Track A result.
        """,
        "talk": """
        Start here. Explain that FairFace and UTKFace provide labelled
        evaluations, whereas LFW is prediction-only because independent
        demographic ground truth is unavailable.
        """,
        "data_roots": [
            rel(
                "model_comparison/"
                "track_a_final"
            )
        ],
        "figures": [
            (
                rel(
                    "model_comparison/"
                    "track_a_final/"
                    "01_gender_accuracy_comparison.png"
                ),
                "Gender classification accuracy",
                """
                Compares gender accuracy for FairFace ResNet-34 and
                DeepFace on the two independently labelled datasets.
                FairFace ResNet-34 remains near 94% on both datasets,
                whereas DeepFace changes from about 70% on FairFace
                validation to about 86% on UTKFace.
                """
            ),
            (
                rel(
                    "model_comparison/"
                    "track_a_final/"
                    "02_race_accuracy_comparison.png"
                ),
                "Harmonised race classification accuracy",
                """
                Compares race-classification accuracy within each
                labelled dataset. FairFace validation uses a common
                six-class taxonomy and UTKFace uses a common four-class
                taxonomy, so the key interpretation is the comparison
                between models within each dataset rather than treating
                both bars as one identical classification task.
                """
            ),
            (
                rel(
                    "model_comparison/"
                    "track_a_final/"
                    "03_gender_equalized_odds_comparison.png"
                ),
                "Gender equalized-odds difference",
                """
                Shows how strongly female true-positive and
                false-positive rates vary across labelled race groups.
                DeepFace exhibits the larger measured disparity on both
                FairFace validation and UTKFace. A lower value does not
                certify that a model is universally fair; it means the
                measured cross-group difference is smaller under this
                experiment.
                """
            ),
            (
                rel(
                    "model_comparison/"
                    "track_a_final/"
                    "04_lfw_race_distribution_comparison.png"
                ),
                "LFW model-assigned race composition",
                """
                Both demographic classifiers are applied to exactly the
                same 13,233 LFW images. The resulting race composition
                changes depending on which classifier supplies the
                demographic labels. These are model-assigned categories,
                not verified demographic ground truth.
                """
            ),
            (
                rel(
                    "model_comparison/"
                    "track_a_final/"
                    "05_lfw_gender_distribution_comparison.png"
                ),
                "LFW model-assigned gender composition",
                """
                The two models produce more similar aggregate gender
                distributions than race distributions, but they are not
                identical. This supports the finding that gender
                measurement is more stable across these two models than
                race measurement.
                """
            ),
            (
                rel(
                    "model_comparison/"
                    "track_a_final/"
                    "06_lfw_model_agreement.png"
                ),
                "Direct model agreement on LFW",
                """
                Measures whether FairFace ResNet-34 and DeepFace assign
                the same demographic category to the same image. Race
                agreement is about 74.7%, while gender agreement is about
                95.4%. The corresponding disagreements are approximately
                25.3% and 4.6%.
                """
            ),
            (
                rel(
                    "model_comparison/"
                    "track_a_final/"
                    "07_lfw_identity_consistency_comparison.png"
                ),
                "Repeated-identity demographic consistency",
                """
                Uses repeated photographs of the same LFW identity to
                ask whether one demographic classifier gives a stable
                category across different photographs. Gender is highly
                stable for both models, while race assignments show more
                variation.
                """
            ),
            (
                rel(
                    "model_comparison/"
                    "track_a_final/"
                    "08_lfw_cross_model_race_matrix.png"
                ),
                "Cross-model race assignment matrix",
                """
                The diagonal contains images where the two demographic
                classifiers assign the same harmonised race category.
                Off-diagonal cells reveal which categories are exchanged
                between the models. This is an agreement matrix, not a
                ground-truth confusion matrix.
                """
            ),
        ],
    },

    # --------------------------------------------------------
    # FAIRFACE MODEL / FAIRFACE DATASET
    # --------------------------------------------------------
    "FairFace ResNet-34 — FairFace validation": {
        "intro": """
        Detailed held-out FairFace validation report for the FairFace
        ResNet-34 demographic classifier.
        """,
        "talk": """
        Use this pack to explain the model's overall accuracy, per-class
        behaviour, intersectional performance, equalized-odds analysis
        and confidence behaviour on independently labelled data.
        """,
        "data_roots": [
            rel(
                "fairface_resnet34/"
                "fairface_validation"
            )
        ],
        "figures": [
            (
                rel(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "race_confusion_matrix.png"
                ),
                "Native FairFace race confusion matrix",
                """
                This matrix uses FairFace's native seven race categories.
                The diagonal represents correct classifications; the
                off-diagonal cells show which labelled groups the model
                confuses. This is different from the later harmonised
                six-class model-comparison evaluation.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "gender_confusion_matrix.png"
                ),
                "Gender confusion matrix",
                """
                Shows correct and incorrect Male/Female classifications.
                Most predictions fall on the diagonal, consistent with
                the model's approximately 94.4% overall gender accuracy.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "report_pack/figures/"
                    "01_overall_performance.png"
                ),
                "Overall FairFace model performance",
                """
                Summarises the main race and gender classifier metrics.
                Accuracy gives overall correctness, while macro F1 gives
                each class equal importance and therefore helps prevent
                large classes from dominating the summary.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "report_pack/figures/"
                    "02_race_precision_recall_f1.png"
                ),
                "Race precision, recall and F1 by class",
                """
                Breaks overall race performance into class-specific
                behaviour. Precision asks how often predictions of a
                class are correct; recall asks how many members of that
                class are successfully found; F1 balances both.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "report_pack/figures/"
                    "03_intersectional_race_accuracy.png"
                ),
                "Intersectional race accuracy",
                """
                Shows race-classification accuracy after separating
                samples by race and gender. It demonstrates why a single
                overall accuracy can hide substantially different
                subgroup performance.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "report_pack/figures/"
                    "04_intersectional_gender_accuracy.png"
                ),
                "Intersectional gender accuracy",
                """
                Shows gender-classification accuracy for each race ×
                gender intersection. The objective is not merely to find
                the lowest bar, but to determine whether performance
                differences are systematic and sufficiently supported by
                sample size.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "report_pack/figures/"
                    "05_gender_equalized_odds_by_race.png"
                ),
                "Gender equalized odds by race",
                """
                Compares female TPR and FPR across the labelled race
                groups. Equalized odds asks whether the classifier's
                error behaviour is similar across groups, not merely
                whether overall accuracy is high.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "report_pack/figures/"
                    "06_race_equalized_odds_by_gender.png"
                ),
                "Race equalized odds by gender",
                """
                Treats each race category as a one-vs-rest classification
                task and compares its TPR/FPR across gender groups. The
                size of the gap varies substantially by race category.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "report_pack/figures/"
                    "07_race_confidence_by_intersection.png"
                ),
                "Race confidence by demographic intersection",
                """
                Compares prediction confidence across race × gender
                intersections. Confidence describes the model's own
                certainty; it is not the same thing as correctness or
                calibration.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "fairface_validation/"
                    "report_pack/figures/"
                    "08_race_confidence_correct_vs_error.png"
                ),
                "Confidence for correct predictions versus errors",
                """
                Tests whether errors tend to occur at lower confidence
                than correct predictions. High-confidence errors are
                especially important because they show that confidence
                cannot automatically be interpreted as reliability.
                """
            ),
        ],
    },

    # --------------------------------------------------------
    # FAIRFACE MODEL / UTKFACE
    # --------------------------------------------------------
    "FairFace ResNet-34 — UTKFace": {
        "intro": """
        Cross-dataset evaluation of FairFace ResNet-34 on deterministically
        aligned UTKFace images.
        """,
        "talk": """
        Use this section to show what happens when the same demographic
        classifier is transferred to a different dataset and a
        harmonised taxonomy is required.
        """,
        "data_roots": [
            rel(
                "fairface_resnet34/"
                "utkface_aligned"
            )
        ],
        "figures": [
            (
                rel(
                    "fairface_resnet34/"
                    "utkface_aligned/"
                    "race_confusion_matrix_harmonised.png"
                ),
                "UTKFace harmonised four-class race confusion matrix",
                """
                Evaluates White, Black, Asian and Indian categories in a
                common four-class space. UTKFace's native Others class is
                not forcibly converted into one of the four comparable
                categories.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "utkface_aligned/"
                    "gender_confusion_matrix.png"
                ),
                "UTKFace gender confusion matrix",
                """
                Shows the model's Male/Female performance on UTKFace.
                Overall gender accuracy remains close to the FairFace
                validation result, although subgroup error rates still
                require separate fairness analysis.
                """
            ),
        ],
    },

    # --------------------------------------------------------
    # FAIRFACE MODEL / LFW
    # --------------------------------------------------------
    "FairFace ResNet-34 — LFW prediction audit": {
        "intro": """
        Detailed prediction-only FairFace ResNet-34 report on all
        13,233 LFW images.
        """,
        "talk": """
        This is one of the most useful packs for the supervisor meeting.
        Keep repeating that these are model-assigned demographic
        measurements, not verified LFW demographic truth.
        """,
        "data_roots": [
            rel(
                "fairface_resnet34/"
                "lfw_predictions"
            )
        ],
        "figures": [
            (
                rel(
                    "fairface_resnet34/"
                    "lfw_predictions/"
                    "report_pack/figures/"
                    "01_predicted_race_distribution.png"
                ),
                "Predicted race distribution on LFW",
                """
                Shows the demographic composition assigned to LFW by
                FairFace ResNet-34. White is the largest model-assigned
                category, while Middle Eastern is the second largest.
                The plot describes classifier output rather than true
                population demographics.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "lfw_predictions/"
                    "report_pack/figures/"
                    "02_predicted_gender_distribution.png"
                ),
                "Predicted gender distribution on LFW",
                """
                FairFace ResNet-34 assigns substantially more LFW images
                to Male than Female. Without independent LFW gender
                labels, this is a measurement result rather than a
                ground-truth representation estimate.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "lfw_predictions/"
                    "report_pack/figures/"
                    "03_predicted_age_distribution.png"
                ),
                "Predicted age distribution on LFW",
                """
                Shows FairFace ResNet-34's model-assigned age bands.
                Age was not included in the DeepFace comparison track, so
                this plot is descriptive for the FairFace model rather
                than a two-model validation.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "lfw_predictions/"
                    "report_pack/figures/"
                    "04_predicted_race_gender_heatmap.png"
                ),
                "Predicted race × gender intersections",
                """
                Breaks the model-assigned LFW composition into
                intersectional race × gender cells. This makes clear that
                aggregate race or gender totals can hide strongly
                unequal intersection sizes.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "lfw_predictions/"
                    "report_pack/figures/"
                    "05_intersectional_proportions.png"
                ),
                "Intersectional proportions",
                """
                Expresses the predicted race × gender cells as shares of
                the LFW image population. These are particularly useful
                when discussing which model-assigned groups would have
                more or less statistical support in a downstream
                verification audit.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "lfw_predictions/"
                    "report_pack/figures/"
                    "06_mean_prediction_confidence.png"
                ),
                "Mean demographic prediction confidence",
                """
                Summarises model confidence for race, gender and age.
                Gender confidence is substantially higher than race and
                age confidence, but confidence must not be treated as
                proof of demographic correctness.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "lfw_predictions/"
                    "report_pack/figures/"
                    "07_confidence_distributions.png"
                ),
                "Prediction confidence distributions",
                """
                Shows the full spread rather than only mean confidence.
                A distribution is more informative because two models or
                attributes can have the same mean while having very
                different tails of uncertain predictions.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "lfw_predictions/"
                    "report_pack/figures/"
                    "08_race_confidence_by_predicted_class.png"
                ),
                "Race confidence by predicted category",
                """
                Tests whether model confidence varies depending on the
                race category assigned. Variation here is relevant
                because a fairness audit based on model-generated labels
                may inherit different levels of uncertainty across
                predicted groups.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "lfw_predictions/"
                    "report_pack/figures/"
                    "09_repeated_identity_consistency.png"
                ),
                "Repeated-identity prediction consistency",
                """
                For identities with multiple photographs, this measures
                how often a model keeps assigning the same demographic
                category. Race and especially age vary more across
                photographs than gender.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "lfw_predictions/"
                    "report_pack/figures/"
                    "10_consistency_vs_identity_depth.png"
                ),
                "Consistency versus number of images per identity",
                """
                Examines whether prediction consistency changes as more
                photographs of the same identity are available.
                Identities with more photographs provide more
                opportunities for changes in pose, lighting and model
                output, so this is descriptive rather than a causal
                relationship.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "lfw_predictions/"
                    "report_pack/figures/"
                    "11_identity_depth_by_predicted_race.png"
                ),
                "Identity depth by modal predicted race",
                """
                Combines LFW's highly uneven identity depth with the
                model's modal race assignment. It helps reveal whether
                some model-assigned groups are disproportionately
                represented by identities with many or few photographs.
                """
            ),
        ],
    },

    # --------------------------------------------------------
    # DEEPFACE / FAIRFACE
    # --------------------------------------------------------
    "DeepFace — FairFace validation": {
        "intro": """
        DeepFace demographic classification evaluated on the same
        10,954 held-out FairFace validation images.
        """,
        "talk": """
        Use this pack alongside the FairFace ResNet-34 pack to show that
        demographic measurement quality is model-dependent even before
        facial recognition is evaluated.
        """,
        "data_roots": [
            rel(
                "deepface/"
                "fairface_validation"
            )
        ],
        "figures": [
            (
                rel(
                    "deepface/"
                    "fairface_validation/"
                    "race_confusion_matrix.png"
                ),
                "DeepFace six-class race confusion matrix",
                """
                Displays class-specific DeepFace race errors in the
                harmonised six-class evaluation. The matrix helps show
                which categories drive the lower overall race accuracy.
                """
            ),
            (
                rel(
                    "deepface/"
                    "fairface_validation/"
                    "gender_confusion_matrix.png"
                ),
                "DeepFace gender confusion matrix",
                """
                Shows a substantially larger gender error burden than
                FairFace ResNet-34 on the same FairFace validation
                images. The subgroup analysis later shows that these
                errors are not evenly distributed.
                """
            ),
        ],
    },

    # --------------------------------------------------------
    # DEEPFACE / UTKFACE
    # --------------------------------------------------------
    "DeepFace — UTKFace": {
        "intro": """
        DeepFace demographic classification evaluated on aligned
        UTKFace.
        """,
        "talk": """
        This pack demonstrates that some DeepFace results improve on a
        different dataset, particularly gender accuracy, while measured
        subgroup disparity remains.
        """,
        "data_roots": [
            rel(
                "deepface/"
                "utkface_aligned"
            )
        ],
        "figures": [
            (
                rel(
                    "deepface/"
                    "utkface_aligned/"
                    "race_confusion_matrix_harmonised.png"
                ),
                "DeepFace UTKFace race confusion matrix",
                """
                Uses the common four-class UTKFace comparison space.
                Overall race accuracy becomes much closer to FairFace
                ResNet-34 than it was on FairFace validation, but
                class-specific behaviour remains uneven.
                """
            ),
            (
                rel(
                    "deepface/"
                    "utkface_aligned/"
                    "gender_confusion_matrix.png"
                ),
                "DeepFace UTKFace gender confusion matrix",
                """
                Gender accuracy improves substantially relative to the
                FairFace validation result. This is evidence that model
                performance depends on evaluation domain rather than
                being a single fixed property of the classifier.
                """
            ),
        ],
    },

    # --------------------------------------------------------
    # DEEPFACE / LFW
    # --------------------------------------------------------
    "DeepFace — LFW prediction audit": {
        "intro": """
        Prediction-only DeepFace demographic audit on the same 13,233
        LFW images.
        """,
        "talk": """
        Put this side-by-side with the FairFace LFW pack. The central
        question is not which model is 'right', but how much the inferred
        demographic composition changes when the annotation method
        changes.
        """,
        "data_roots": [
            rel(
                "deepface/"
                "lfw_aligned"
            )
        ],
        "figures": [
            (
                rel(
                    "deepface/"
                    "lfw_aligned/"
                    "01_predicted_race_distribution.png"
                ),
                "DeepFace predicted race distribution on LFW",
                """
                Shows the LFW race composition assigned by DeepFace.
                Comparison with the FairFace model reveals substantial
                differences for categories such as Middle Eastern and
                Latino/Hispanic.
                """
            ),
            (
                rel(
                    "deepface/"
                    "lfw_aligned/"
                    "02_predicted_gender_distribution.png"
                ),
                "DeepFace predicted gender distribution on LFW",
                """
                Shows DeepFace's aggregate Male/Female assignments.
                Gender composition is more similar between the two
                classifiers than race composition, although it is still
                not identical.
                """
            ),
            (
                rel(
                    "deepface/"
                    "lfw_aligned/"
                    "03_predicted_race_gender_heatmap.png"
                ),
                "DeepFace predicted race × gender intersections",
                """
                Breaks the LFW output into intersectional demographic
                cells. The sizes of these cells matter because downstream
                subgroup performance estimates become less stable when
                a model-assigned group contains few images.
                """
            ),
            (
                rel(
                    "deepface/"
                    "lfw_aligned/"
                    "04_repeated_identity_consistency.png"
                ),
                "DeepFace repeated-identity consistency",
                """
                Measures demographic stability across multiple
                photographs of the same LFW identity. Gender predictions
                are highly stable, while race assignments change more
                often.
                """
            ),
        ],
    },

    # --------------------------------------------------------
    # DIRECT MODEL COMPARISON
    # --------------------------------------------------------
    "FairFace vs DeepFace — FairFace validation": {
        "intro": """
        Direct model-to-model comparison on exactly the same labelled
        FairFace validation images.
        """,
        "talk": """
        This is useful when someone asks whether the performance
        difference could simply come from evaluating different images.
        It cannot here: both models are evaluated on the same held-out
        population.
        """,
        "data_roots": [
            rel(
                "model_comparison/"
                "fairface_validation"
            )
        ],
        "figures": [
            (
                rel(
                    "model_comparison/"
                    "fairface_validation/"
                    "01_model_performance_comparison.png"
                ),
                "Direct demographic model performance comparison",
                """
                Compares common six-class race performance and gender
                performance using identical FairFace validation images.
                FairFace ResNet-34 has higher race and gender performance
                under this evaluation.
                """
            ),
            (
                rel(
                    "model_comparison/"
                    "fairface_validation/"
                    "02_gender_female_tpr_by_race.png"
                ),
                "Female TPR by race and model",
                """
                Displays the intersectional pattern that drives much of
                the gender equalized-odds result. DeepFace female TPR
                changes substantially across labelled race groups.
                """
            ),
        ],
    },

    # --------------------------------------------------------
    # FAIRFACE CROSS DATASET
    # --------------------------------------------------------
    "FairFace ResNet-34 — Cross-dataset comparison": {
        "intro": """
        The same FairFace ResNet-34 model evaluated across FairFace
        validation and UTKFace.
        """,
        "talk": """
        Use this section when discussing domain shift. The model has not
        changed; the evaluation dataset has.
        """,
        "data_roots": [
            rel(
                "fairface_resnet34/"
                "cross_dataset_comparison"
            )
        ],
        "figures": [
            (
                rel(
                    "fairface_resnet34/"
                    "cross_dataset_comparison/"
                    "01_cross_dataset_performance.png"
                ),
                "Cross-dataset performance",
                """
                Shows that FairFace ResNet-34 gender performance remains
                very similar across FairFace validation and UTKFace,
                whereas harmonised race performance decreases on
                UTKFace. This is evidence of dataset/domain sensitivity.
                """
            ),
            (
                rel(
                    "fairface_resnet34/"
                    "cross_dataset_comparison/"
                    "02_gender_equalized_odds_comparison.png"
                ),
                "Cross-dataset equalized-odds comparison",
                """
                Compares the gender equalized-odds difference for the
                same model across datasets. The measured disparity
                changes with the evaluation population, illustrating why
                fairness should not be represented as one universal
                number.
                """
            ),
        ],
    },
}


# ============================================================
# PAGE HEADER
# ============================================================

hero(
    "Evidence & Report Packs",
    """
    Full experimental evidence for supervisor review. This page exposes
    the detailed figures, tables and diagnostic outputs behind the
    summary conclusions presented elsewhere in the dashboard.
    """,
    eyebrow="Research workspace · Full evidence",
)

gold_callout(
    """
    <strong>This page is deliberately detailed.</strong>
    The Model Audit page provides the short presentation summary.
    This evidence library retains the deeper plots needed to inspect,
    challenge and explain the results during the investigation.
    """
)


# ============================================================
# MAIN TABS
# ============================================================

report_tab, qualitative_tab = st.tabs(
    [
        "Report Pack Browser",
        "Qualitative Example Browser",
    ]
)


# ============================================================
# REPORT PACK BROWSER
# ============================================================

with report_tab:

    section_label("Experiment")

    selected_pack_name = st.selectbox(
        "Choose a report pack",
        list(
            PACKS.keys()
        ),
    )

    pack = PACKS[
        selected_pack_name
    ]

    st.header(
        selected_pack_name
    )

    st.write(
        pack[
            "intro"
        ]
    )

    blue_callout(
        f"""
        <strong>Presentation approach:</strong>
        {pack["talk"]}
        """
    )

    available_figures = [
        item
        for item in pack[
            "figures"
        ]
        if figure_exists(
            item[0]
        )
    ]

    metric_1, metric_2 = st.columns(2)

    metric_1.metric(
        "Figures in this pack",
        len(
            available_figures
        ),
    )

    data_files = available_data_files(
        pack[
            "data_roots"
        ]
    )

    metric_2.metric(
        "Tables / summaries",
        len(
            data_files
        ),
    )

    st.divider()

    for (
        path,
        title,
        explanation,
    ) in pack[
        "figures"
    ]:

        render_figure(
            path,
            title,
            explanation,
        )

    render_data_explorer(
        pack[
            "data_roots"
        ],
        selected_pack_name,
    )


# ============================================================
# QUALITATIVE EXAMPLE BROWSER
# ============================================================

with qualitative_tab:

    section_label("Visual inspection")

    st.header(
        "Qualitative audit examples"
    )

    st.write(
        """
        This browser is intended for close inspection of model outputs.
        The contact sheets remain available for overview, while
        individual panels can be enlarged enough for labels to be read
        comfortably during a presentation.
        """
    )

    gold_callout(
        """
        FairFace and UTKFace panels can show dataset demographic labels
        alongside both model predictions. LFW has no independent
        demographic ground truth in this project, so its panels show
        <strong>model assignments and model agreement only</strong>.
        """
    )

    dataset_label = st.selectbox(
        "Dataset",
        [
            "FairFace",
            "UTKFace",
            "LFW",
        ],
        key="report_pack_qual_dataset",
    )

    dataset_key = {
        "FairFace": "fairface",
        "UTKFace": "utkface",
        "LFW": "lfw",
    }[
        dataset_label
    ]

    dataset_root = (
        QUALITATIVE_ROOT
        / dataset_key
    )

    manifest_path = (
        dataset_root
        / "sample_manifest.csv"
    )

    if manifest_path.exists():

        manifest = load_csv(
            str(
                manifest_path
            )
        )

    else:

        manifest = pd.DataFrame()

    mode = st.radio(
        "View",
        [
            "Single enlarged example",
            "Contact sheet",
            "Manifest",
        ],
        horizontal=True,
    )

    # --------------------------------------------------------
    # SINGLE EXAMPLE
    # --------------------------------------------------------

    if (
        mode
        == "Single enlarged example"
    ):

        individual_root = (
            dataset_root
            / "individual"
        )

        individual_files = (
            sorted(
                individual_root.glob(
                    "*.png"
                )
            )
            if individual_root.exists()
            else []
        )

        if individual_files:

            sample_number = st.slider(
                "Example",
                min_value=1,
                max_value=len(
                    individual_files
                ),
                value=1,
            )

            selected_image = (
                individual_files[
                    sample_number - 1
                ]
            )

            left, centre, right = st.columns(
                [
                    1,
                    3,
                    1,
                ]
            )

            with centre:

                st.image(
                    str(
                        selected_image
                    ),
                    use_container_width=True,
                )

            if (
                not manifest.empty
                and len(
                    manifest
                )
                >= sample_number
            ):

                row = manifest.iloc[
                    sample_number - 1
                ]

                st.subheader(
                    "Recorded labels"
                )

                display_items = {
                    "Image ID": row.get(
                        "image_id"
                    ),
                    "Selection reason": row.get(
                        "selection_reason"
                    ),
                    "Ground-truth race": row.get(
                        "ground_truth_race"
                    ),
                    "Ground-truth gender": row.get(
                        "ground_truth_gender"
                    ),
                    "FairFace race": row.get(
                        "fairface_race"
                    ),
                    "FairFace gender": row.get(
                        "fairface_gender"
                    ),
                    "DeepFace race": row.get(
                        "deepface_race"
                    ),
                    "DeepFace gender": row.get(
                        "deepface_gender"
                    ),
                    "Race-model agreement": row.get(
                        "race_model_agreement"
                    ),
                    "Gender-model agreement": row.get(
                        "gender_model_agreement"
                    ),
                }

                display = pd.DataFrame(
                    {
                        "Field": list(
                            display_items.keys()
                        ),
                        "Value": list(
                            display_items.values()
                        ),
                    }
                )

                st.dataframe(
                    display,
                    hide_index=True,
                    use_container_width=True,
                )

        else:

            st.info(
                """
                The tracked repository contains the contact sheets and
                manifests, but the 90 individual panels are intentionally
                kept local rather than committed to Git.

                Restore or regenerate the local individual panels to
                enable this enlarged-example browser.
                """
            )

    # --------------------------------------------------------
    # CONTACT SHEET
    # --------------------------------------------------------

    elif (
        mode
        == "Contact sheet"
    ):

        page = st.radio(
            "Page",
            [
                1,
                2,
                3,
            ],
            horizontal=True,
        )

        sheet = (
            dataset_root
            / f"contact_sheet_{page:02d}.png"
        )

        if sheet.exists():

            st.image(
                str(
                    sheet
                ),
                use_container_width=True,
            )

        else:

            st.warning(
                "Contact sheet not found."
            )

    # --------------------------------------------------------
    # MANIFEST
    # --------------------------------------------------------

    else:

        if manifest.empty:

            st.warning(
                "Sample manifest not found."
            )

        else:

            st.dataframe(
                manifest,
                hide_index=True,
                use_container_width=True,
                height=650,
            )

    st.caption(
        """
        Qualitative examples are for manual error analysis and
        illustration. The 30-image samples are not replacements for the
        full-dataset performance statistics.
        """
    )
    