from __future__ import annotations

import streamlit as st

from model_audit_data import (
    load_lfw_cross_model_agreement,
)
from ui import (
    evidence_note,
    metric_row,
    page_header,
    research_card,
    section_header,
)


# ============================================================
# HELPERS
# ============================================================

def lookup_agreement(
    frame,
    task: str,
) -> float | None:

    if frame.empty:
        return None

    row = frame[
        frame["task"].eq(
            task
        )
    ]

    if row.empty:
        return None

    return float(
        row.iloc[0][
            "agreement_rate"
        ]
    )


def percentage(
    value: float | None,
    fallback: str,
) -> str:

    if value is None:
        return fallback

    return f"{value:.2%}"


# ============================================================
# DATA
# ============================================================

lfw_agreement = (
    load_lfw_cross_model_agreement()
)

race_agreement = lookup_agreement(
    lfw_agreement,
    "race",
)

gender_agreement = lookup_agreement(
    lfw_agreement,
    "gender",
)

race_disagreement = (
    1 - race_agreement
    if race_agreement is not None
    else None
)

gender_disagreement = (
    1 - gender_agreement
    if gender_agreement is not None
    else None
)


# ============================================================
# PAGE HEADER
# ============================================================

page_header(
    "Facial Recognition Fairness Audit",
    (
        "An engineering investigation of dataset representation, "
        "demographic measurement, facial-recognition performance "
        "and mitigation across FairFace, UTKFace and LFW."
    ),
    (
        "University of the Witwatersrand · "
        "School of Electrical and Information Engineering"
    ),
)


# ============================================================
# PROJECT AT A GLANCE
# ============================================================

section_header(
    "Project at a glance",
    (
        "The investigation treats fairness as a system-level "
        "engineering problem rather than a single performance metric."
    ),
)

metric_row(
    [
        (
            "Source records audited",
            "135,039",
        ),
        (
            "Source datasets",
            "3",
        ),
        (
            "HarmoFace-95K",
            "95,258",
        ),
        (
            "Final demographic classes",
            "7",
        ),
    ]
)

st.write(
    """
    The project examines where measurable disparities can enter a
    facial-recognition pipeline and separates those mechanisms into
    distinct layers: the source data, demographic annotation,
    recognition performance and mitigation.
    """
)

st.divider()


# ============================================================
# INVESTIGATION
# ============================================================

section_header(
    "Investigation",
    (
        "Each stage addresses a different source of uncertainty or "
        "performance disparity."
    ),
)

row1 = st.columns(3)

with row1[0]:
    research_card(
        "Dataset Audit",
        (
            "Analyse demographic representation, identity structure, "
            "missing data, duplicates and image characteristics across "
            "FairFace, UTKFace and LFW."
        ),
        "01",
    )

with row1[1]:
    research_card(
        "Demographic Measurement",
        (
            "Evaluate demographic classification performance and test "
            "whether race and gender assignments remain consistent "
            "across annotation models and datasets."
        ),
        "02",
    )

with row1[2]:
    research_card(
        "Face Recognition Audit",
        (
            "Evaluate facial verification and examine false-match and "
            "false-non-match behaviour across demographic and "
            "intersectional groups."
        ),
        "03",
    )

row2 = st.columns(3)

with row2[0]:
    research_card(
        "Mitigation",
        (
            "Compare dataset-specific baseline models with models "
            "trained after resampling-based mitigation and evaluate "
            "performance and subgroup trade-offs."
        ),
        "04",
    )

with row2[1]:
    research_card(
        "HarmoFace-95K",
        (
            "Construct a harmonised, provenance-preserving and "
            "diversity-aware dataset containing 95,258 labelled "
            "face images."
        ),
        "05",
    )

with row2[2]:
    research_card(
        "Engineering Interpretation",
        (
            "Consolidate the statistical evidence into technical "
            "findings, limitations and practical implications for "
            "facial-recognition system development."
        ),
        "06",
    )

st.divider()


# ============================================================
# HEADLINE EVIDENCE
# ============================================================

section_header(
    "Selected evidence",
    (
        "A small number of results summarise why the different "
        "layers of the audit matter."
    ),
)

metric_row(
    [
        (
            "LFW race disagreement",
            percentage(
                race_disagreement,
                "25.29%",
            ),
        ),
        (
            "LFW gender disagreement",
            percentage(
                gender_disagreement,
                "4.63%",
            ),
        ),
        (
            "HarmoFace balance ratio",
            "1.62 : 1",
        ),
    ]
)

left, right = st.columns(2)

with left:
    evidence_note(
        "FairFace validation",
        (
            "FairFace ResNet-34 achieved 77.65% six-class race "
            "accuracy and 94.39% gender accuracy. DeepFace achieved "
            "65.75% race accuracy and 70.17% gender accuracy."
        ),
    )

with right:
    evidence_note(
        "UTKFace",
        (
            "Race accuracy narrowed to 76.48% for FairFace ResNet-34 "
            "and 75.16% for DeepFace, while gender accuracy remained "
            "separated at 94.35% and 86.23%."
        ),
    )

st.info(
    """
    The demographic-measurement results show that the observed
    composition of a dataset and the measured size of demographic
    disparities can depend on the annotation method and evaluation
    domain.
    """
)

st.divider()


# ============================================================
# HARMOFACE
# ============================================================

section_header(
    "Dataset contribution",
    (
        "The dataset engineering work produced a training-ready "
        "harmonised collection."
    ),
)

st.markdown(
    "## HarmoFace-95K"
)

metric_row(
    [
        (
            "Images",
            "95,258",
        ),
        (
            "Training",
            "76,395",
        ),
        (
            "Validation",
            "9,544",
        ),
        (
            "Test",
            "9,319",
        ),
    ]
)

st.write(
    """
    HarmoFace-95K combines audited records from FairFace, UTKFace and
    LFW into seven harmonised demographic categories:

    **White · Black · Indian · East Asian · Southeast Asian ·
    Middle Eastern · Latino**
    """
)

st.write(
    """
    The final construction uses volume-aware sampling rather than
    strict equal-count undersampling. Minority records are not
    duplicated to manufacture balance, source provenance is retained,
    and ITA skin-tone measurements remain separate from demographic
    labels.
    """
)

st.divider()


# ============================================================
# MITIGATION
# ============================================================

section_header(
    "Mitigation",
    (
        "The audit is followed by an intervention designed to address "
        "identified training-distribution problems."
    ),
)

left, right = st.columns(
    [
        1,
        1,
    ]
)

with left:
    with st.container(
        border=True
    ):
        st.markdown(
            "### Baseline"
        )

        st.write(
            """
            Dataset-specific baseline models provide the reference
            performance before mitigation.
            """
        )

with right:
    with st.container(
        border=True
    ):
        st.markdown(
            "### Resampling intervention"
        )

        st.write(
            """
            Corresponding mitigated models are trained after adjusting
            the training distribution and are evaluated using the same
            performance and fairness framework.
            """
        )

st.divider()


# ============================================================
# PROJECT OUTPUTS
# ============================================================

section_header(
    "Engineering outputs",
)

outputs = st.columns(3)

with outputs[0]:
    research_card(
        "Dataset evidence",
        (
            "Reproducible representation, demographic, identity and "
            "image-condition analyses."
        ),
        "DATA",
    )

with outputs[1]:
    research_card(
        "Model evidence",
        (
            "Demographic measurement, recognition and mitigation "
            "performance evaluated quantitatively."
        ),
        "MODELS",
    )

with outputs[2]:
    research_card(
        "HarmoFace-95K",
        (
            "A materialised labelled image dataset with metadata, "
            "provenance and fixed data partitions."
        ),
        "DATASET",
    )

st.divider()

st.caption(
    """
    University of the Witwatersrand ·
    School of Electrical and Information Engineering ·
    Investigation Project
    """
)
