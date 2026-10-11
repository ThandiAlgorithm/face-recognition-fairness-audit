from __future__ import annotations

import numpy as np
import streamlit as st

from model_audit_data import (
    load_gender_equalized_odds,
    load_lfw_cross_model_agreement,
)

from reference_a_data import (
    load_error_summary,
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
# DATA
# ============================================================

eo = load_gender_equalized_odds()
lfw_agreement = load_lfw_cross_model_agreement()

recognition_summary = load_summary()
recognition_subgroups = load_subgroup_fairness()
error_summary = load_error_summary()


# ============================================================
# HELPERS
# ============================================================

def agreement_value(
    task: str,
    column: str,
) -> float | None:

    if lfw_agreement.empty:
        return None

    row = lfw_agreement[
        lfw_agreement[
            "task"
        ].eq(task)
    ]

    if row.empty:
        return None

    return float(
        row.iloc[0][column]
    )


def eo_value(
    dataset: str,
    model: str,
) -> float | None:

    if eo.empty:
        return None

    row = eo[
        eo[
            "dataset"
        ].eq(dataset)
        & eo[
            "model"
        ].eq(model)
    ]

    if row.empty:
        return None

    return float(
        row.iloc[0][
            "equalized_odds_difference"
        ]
    )


def pct(
    value: float | None,
    fallback: str,
    digits: int = 2,
) -> str:

    if value is None:
        return fallback

    try:

        if not np.isfinite(value):
            return fallback

    except Exception:
        return fallback

    return (
        f"{value * 100:.{digits}f}%"
    )


def percentage_points(
    value: float | None,
    fallback: str,
) -> str:

    if value is None:
        return fallback

    return (
        f"{value * 100:.2f} pp"
    )


# ============================================================
# DERIVED VALUES
# ============================================================

race_disagreement = agreement_value(
    "race",
    "disagreement_rate",
)

gender_disagreement = agreement_value(
    "gender",
    "disagreement_rate",
)

fairface_ff_eo = eo_value(
    "FairFace validation",
    "FairFace ResNet-34",
)

fairface_df_eo = eo_value(
    "FairFace validation",
    "DeepFace",
)

utk_ff_eo = eo_value(
    "UTKFace aligned",
    "FairFace ResNet-34",
)

utk_df_eo = eo_value(
    "UTKFace aligned",
    "DeepFace",
)


recognition_accuracy = (
    float(
        recognition_summary[
            "accuracy"
        ]
    )
    if recognition_summary
    else None
)

recognition_tar = (
    float(
        recognition_summary[
            "TPR_TAR"
        ]
    )
    if recognition_summary
    else None
)

recognition_fmr = (
    float(
        recognition_summary[
            "FMR_FPR"
        ]
    )
    if recognition_summary
    else None
)

recognition_fnmr = (
    float(
        recognition_summary[
            "FNMR_FNR"
        ]
    )
    if recognition_summary
    else None
)


# ============================================================
# HEADER
# ============================================================

page_header(
    "Findings & Implications",
    (
        "The principal technical findings from the dataset audit, "
        "demographic-measurement analysis and facial-verification "
        "experiment, together with their implications for engineering "
        "fairness evaluation."
    ),
    "Research synthesis",
)


# ============================================================
# HEADLINE
# ============================================================

section_header(
    "What the investigation established",
    (
        "Fairness cannot be reduced to one model score or one "
        "demographic balance statistic."
    ),
)

metric_row(
    [
        (
            "LFW race disagreement",
            pct(
                race_disagreement,
                "25.29%",
            ),
        ),
        (
            "LFW gender disagreement",
            pct(
                gender_disagreement,
                "4.63%",
            ),
        ),
        (
            "ArcFace verification accuracy",
            pct(
                recognition_accuracy,
                "96.80%",
            ),
        ),
        (
            "HarmoFace-95K",
            "95,258 images",
        ),
    ]
)

st.info(
    """
    The central result is not that one dataset or one model is simply
    "fair" or "biased". The measured outcome depends on the data,
    demographic annotation method, preprocessing pipeline, recognition
    model, subgroup definition and evaluation protocol.
    """
)

st.divider()


# ============================================================
# FINDING 1
# ============================================================

section_header(
    "1. Demographic labels are themselves a measurement",
)

finding_card(
    "Race annotation was substantially more model-sensitive than gender annotation",
    (
        f"On the same 13,233 LFW images, the two demographic "
        f"classifiers disagreed on "
        f"{pct(race_disagreement, '25.29%')} of race assignments "
        f"but only {pct(gender_disagreement, '4.63%')} of gender "
        f"assignments."
    ),
    (
        "When a dataset does not provide independent demographic ground "
        "truth, the annotation system becomes part of the measurement "
        "pipeline. Fairness results conditioned on those inferred "
        "groups inherit that uncertainty."
    ),
)

left, right = st.columns(2)

with left:

    evidence_note(
        "Race",
        (
            "Cross-model race agreement was approximately 74.7%. "
            "Repeated identities also showed lower race-label stability "
            "than gender-label stability."
        ),
    )

with right:

    evidence_note(
        "Gender",
        (
            "Cross-model gender agreement was approximately 95.4%, "
            "showing substantially greater stability under the two "
            "annotation systems."
        ),
    )

st.divider()


# ============================================================
# FINDING 2
# ============================================================

section_header(
    "2. Model performance is evaluation-domain dependent",
)

finding_card(
    "A demographic classifier does not have one fixed accuracy value",
    (
        "FairFace ResNet-34 and DeepFace were evaluated on both "
        "FairFace and UTKFace. The size of the performance difference "
        "changed materially between datasets."
    ),
    (
        "Benchmark-specific results should not automatically be "
        "generalised to another image population, acquisition process "
        "or label taxonomy."
    ),
)

c1, c2 = st.columns(2)

with c1:

    with st.container(
        border=True
    ):

        st.markdown(
            "### FairFace validation"
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
            ]
        )

        metric_row(
            [
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

with c2:

    with st.container(
        border=True
    ):

        st.markdown(
            "### UTKFace"
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
            ]
        )

        metric_row(
            [
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

st.divider()


# ============================================================
# FINDING 3
# ============================================================

section_header(
    "3. High accuracy does not guarantee equal subgroup behaviour",
)

finding_card(
    "Aggregate classification metrics concealed substantial subgroup differences",
    (
        "Gender equalized-odds differences were consistently larger "
        "for DeepFace than for FairFace ResNet-34 on both independently "
        "labelled evaluation datasets."
    ),
    (
        "Overall accuracy should be accompanied by subgroup TPR, FPR "
        "and class-balanced metrics when a system is evaluated for "
        "demographic fairness."
    ),
)

metric_row(
    [
        (
            "FairFace · FairFace model",
            percentage_points(
                fairface_ff_eo,
                "7.43 pp",
            ),
        ),
        (
            "FairFace · DeepFace",
            percentage_points(
                fairface_df_eo,
                "34.03 pp",
            ),
        ),
        (
            "UTKFace · FairFace model",
            percentage_points(
                utk_ff_eo,
                "6.18 pp",
            ),
        ),
        (
            "UTKFace · DeepFace",
            percentage_points(
                utk_df_eo,
                "19.22 pp",
            ),
        ),
    ]
)

st.caption(
    """
    Equalized-odds difference measures the largest observed difference
    in female TPR or FPR across the labelled race groups in this
    demographic-classification experiment.
    """
)

st.divider()


# ============================================================
# FINDING 4
# ============================================================

section_header(
    "4. A strong recognition model can still contain measurable disparities",
)

finding_card(
    "ArcFace achieved strong overall verification performance on LFW",
    (
        f"The frozen verification pipeline achieved "
        f"{pct(recognition_accuracy, '96.80%')} accuracy, "
        f"{pct(recognition_tar, '95.60%')} True Accept Rate, "
        f"{pct(recognition_fmr, '2.00%')} False Match Rate and "
        f"{pct(recognition_fnmr, '4.40%')} False Non-Match Rate."
    ),
    (
        "Aggregate recognition quality establishes system utility, but "
        "it does not determine whether the remaining errors are "
        "distributed evenly across demographic groups."
    ),
)

gender = (
    recognition_subgroups[
        recognition_subgroups[
            "attribute"
        ].eq("gender")
    ]
    if not recognition_subgroups.empty
    else None
)

if (
    gender is not None
    and not gender.empty
):

    female = gender[
        gender[
            "group"
        ].eq("female")
    ]

    male = gender[
        gender[
            "group"
        ].eq("male")
    ]

    if (
        not female.empty
        and not male.empty
    ):

        female_tar = float(
            female.iloc[0][
                "TPR_TAR"
            ]
        )

        male_tar = float(
            male.iloc[0][
                "TPR_TAR"
            ]
        )

        tar_gap = abs(
            female_tar
            - male_tar
        )

        metric_row(
            [
                (
                    "Female genuine-pair TAR",
                    pct(
                        female_tar,
                        "91.97%",
                    ),
                ),
                (
                    "Male genuine-pair TAR",
                    pct(
                        male_tar,
                        "96.97%",
                    ),
                ),
                (
                    "Measured TAR gap",
                    (
                        f"{tar_gap * 100:.2f} pp"
                    ),
                ),
            ]
        )

        st.warning(
            """
            The subgroup labels used for LFW are derived demographic
            measurements rather than independent demographic ground
            truth. The observed gap is therefore a measured result under
            this annotation procedure, not a universal population claim.
            """
        )

st.divider()


# ============================================================
# FINDING 5
# ============================================================

section_header(
    "5. Recognition errors can originate upstream of the recognition model",
)

finding_card(
    "Twelve of the 32 ArcFace test errors contained a multi-face detection signal",
    (
        "Automatic error analysis showed that some incorrectly "
        "classified verification pairs also contained potentially "
        "challenging preprocessing conditions."
    ),
    (
        "A facial-recognition pipeline should be audited end-to-end. "
        "Detection, face selection, alignment and recognition should "
        "not automatically be treated as independent sources of error."
    ),
)

if not error_summary.empty:

    all_rows = error_summary[
        error_summary[
            "outcome"
        ].eq("ALL")
    ]

    fn_rows = error_summary[
        error_summary[
            "outcome"
        ].eq("FN")
    ]

    fp_rows = error_summary[
        error_summary[
            "outcome"
        ].eq("FP")
    ]

    if (
        not all_rows.empty
        and not fn_rows.empty
        and not fp_rows.empty
    ):

        all_row = all_rows.iloc[0]
        fn_row = fn_rows.iloc[0]
        fp_row = fp_rows.iloc[0]

        metric_row(
            [
                (
                    "Total test errors",
                    str(
                        int(
                            all_row[
                                "n_errors"
                            ]
                        )
                    ),
                ),
                (
                    "False non-matches",
                    str(
                        int(
                            fn_row[
                                "n_errors"
                            ]
                        )
                    ),
                ),
                (
                    "False matches",
                    str(
                        int(
                            fp_row[
                                "n_errors"
                            ]
                        )
                    ),
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

st.caption(
    """
    A multi-face signal identifies a possible upstream contributor; it
    does not by itself prove that preprocessing caused the error.
    """
)

st.divider()


# ============================================================
# FINDING 6
# ============================================================

section_header(
    "6. Equal image counts are not sufficient to define a balanced dataset",
)

finding_card(
    "Strict numerical parity would have discarded a large amount of usable data",
    (
        "The balancing analysis compared strict parity with a "
        "volume-aware approach. HarmoFace-95K ultimately retained "
        "95,258 valid images across seven explicit demographic classes "
        "without duplicating minority records."
    ),
    (
        "Dataset engineering should consider useful volume, source "
        "provenance, identity structure and diversity rather than "
        "treating equal class counts as the complete definition of "
        "fairness."
    ),
)

metric_row(
    [
        (
            "Final images",
            "95,258",
        ),
        (
            "Classes",
            "7",
        ),
        (
            "Largest / smallest class",
            "1.62 : 1",
        ),
        (
            "LFW identity leakage",
            "0",
        ),
    ]
)

st.divider()


# ============================================================
# FINDING 7
# ============================================================

section_header(
    "7. Demographic category and skin tone are not interchangeable",
)

finding_card(
    "ITA was retained as an independent measurement variable",
    (
        "Skin-tone measurements showed substantial within-group "
        "variation. Enforcing exact ITA parity would also have removed "
        "large amounts of otherwise usable data."
    ),
    (
        "Race or ethnicity labels should not be used as direct proxies "
        "for measured skin tone. Both variables can be relevant to a "
        "fairness audit, but they represent different information."
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
    ]
)

st.divider()


# ============================================================
# ENGINEERING IMPLICATIONS
# ============================================================

section_header(
    "Engineering implications",
    (
        "The findings translate into practical requirements for "
        "facial-recognition system development and evaluation."
    ),
)

row1 = st.columns(3)

with row1[0]:

    research_card(
        "Document provenance",
        (
            "Record where images and demographic labels originated so "
            "that dataset and annotation effects remain traceable."
        ),
        "01",
    )

with row1[1]:

    research_card(
        "Separate measurement from truth",
        (
            "Model-generated demographic labels should be reported as "
            "measurements when independent ground truth is unavailable."
        ),
        "02",
    )

with row1[2]:

    research_card(
        "Evaluate subgroups",
        (
            "Overall accuracy should be accompanied by subgroup error "
            "rates and the denominators supporting those estimates."
        ),
        "03",
    )

row2 = st.columns(3)

with row2[0]:

    research_card(
        "Audit preprocessing",
        (
            "Detection, alignment and image conditions should be "
            "examined when recognition errors occur."
        ),
        "04",
    )

with row2[1]:

    research_card(
        "Preserve uncertainty",
        (
            "Thin demographic cells and non-estimable metrics should "
            "remain visibly uncertain rather than being converted into "
            "apparently precise conclusions."
        ),
        "05",
    )

with row2[2]:

    research_card(
        "Test mitigation as a trade-off",
        (
            "A mitigation method should be judged by both subgroup "
            "improvement and its effect on overall system utility."
        ),
        "06",
    )

st.divider()


# ============================================================
# MITIGATION STATUS
# ============================================================

section_header(
    "Mitigation remains an open experimental question",
)

st.info(
    """
    Resampling-based mitigation is being evaluated on the
    dataset-specific trained models. No final mitigation conclusion is
    reported here until the before-and-after model results are complete.
    """
)

left, right = st.columns(2)

with left:

    evidence_note(
        "What must improve",
        (
            "The intervention should reduce a defensible subgroup "
            "performance disparity rather than merely improve a global "
            "accuracy value."
        ),
    )

with right:

    evidence_note(
        "What must be preserved",
        (
            "Any fairness improvement must be interpreted alongside "
            "changes in overall accuracy, balanced performance and other "
            "subgroup error rates."
        ),
    )

st.divider()


# ============================================================
# FINAL CONCLUSION
# ============================================================

section_header(
    "Conclusion",
)

st.markdown(
    """
    ### Fairness is a property of the complete evaluation pipeline

    The investigation shows that fairness measurements can change at
    several points before and during facial recognition.

    **Dataset composition** determines who and what conditions are
    represented.

    **Demographic measurement** determines how images are assigned to
    the groups used in fairness analysis.

    **Preprocessing** determines which facial information is actually
    supplied to the recognition model.

    **Recognition performance** determines how false matches and false
    non-matches are distributed.

    **Mitigation** changes the training distribution or learning
    process and must therefore be evaluated for both benefit and
    trade-off.

    A defensible facial-recognition fairness audit should therefore
    report these stages separately rather than compressing them into a
    single fairness score.
    """
)

st.caption(
    """
    Findings are specific to the datasets, models, demographic
    taxonomies, preprocessing pipeline and evaluation protocols used in
    this investigation.
    """
)
