from __future__ import annotations

import pandas as pd
import streamlit as st

from model_audit_data import (
    load_gender_equalized_odds,
    load_labelled_dataset_metrics,
    load_lfw_cross_model_agreement,
)
from ui import (
    apply_wits_theme,
    blue_callout,
    card,
    gold_callout,
    hero,
    section_label,
)


apply_wits_theme()


# ============================================================
# HELPERS
# ============================================================

def lookup_metric(
    frame: pd.DataFrame,
    dataset: str,
    model: str,
    task: str,
    column: str,
) -> float | None:

    if frame.empty:
        return None

    row = frame[
        frame["dataset"].eq(dataset)
        & frame["model"].eq(model)
        & frame["task"].eq(task)
    ]

    if row.empty:
        return None

    return float(
        row.iloc[0][column]
    )


def lookup_agreement(
    frame: pd.DataFrame,
    task: str,
) -> float | None:

    if frame.empty:
        return None

    row = frame[
        frame["task"].eq(task)
    ]

    if row.empty:
        return None

    return float(
        row.iloc[0]["agreement_rate"]
    )


# ============================================================
# DATA
# ============================================================

metrics = load_labelled_dataset_metrics()
eo = load_gender_equalized_odds()
lfw_agreement = load_lfw_cross_model_agreement()

lfw_race_agreement = lookup_agreement(
    lfw_agreement,
    "race",
)

lfw_race_disagreement = (
    1.0 - lfw_race_agreement
    if lfw_race_agreement is not None
    else None
)


# ============================================================
# HEADER
# ============================================================

hero(
    "Facial Recognition Fairness Audit",
    """
    Auditing representation, demographic measurement and model
    performance across LFW, FairFace and UTKFace before testing
    whether disparities propagate into facial verification.
    """,
)

section_label("Investigation structure")

st.subheader("From dataset evidence to recognition fairness")

c1, c2, c3, c4 = st.columns(4)

with c1:
    card(
        "01",
        "Dataset Audit",
        """
        Characterise representation, identity depth, missingness,
        duplicates, image conditions and measurement limitations.
        """,
    )

with c2:
    card(
        "02",
        "Demographic Model Audit",
        """
        Compare FairFace ResNet-34 and DeepFace on labelled datasets
        and test demographic-measurement sensitivity on LFW.
        """,
    )

with c3:
    card(
        "03",
        "Face Verification",
        """
        Evaluate fixed pretrained recognition models under a controlled
        LFW same-person / different-person verification protocol.
        """,
    )

with c4:
    card(
        "04",
        "Mitigation",
        """
        Select an intervention only after the mechanism and largest
        defensible recognition-performance disparity are established.
        """,
    )

st.divider()


# ============================================================
# RESEARCH QUESTION
# ============================================================

section_label("Research question")

st.subheader("The central investigation")

blue_callout(
    """
    <strong>Where do demographic and measurement disparities enter
    the facial-recognition pipeline, how do they affect model
    performance, and do those disparities propagate into facial
    verification?</strong>
    """
)

q1, q2, q3 = st.columns(3)

with q1:
    card(
        "RQ1",
        "What is in the data?",
        """
        Measure representation, missingness, identity structure and
        image-condition differences across the three datasets.
        """,
    )

with q2:
    card(
        "RQ2",
        "How stable is demographic measurement?",
        """
        Determine whether demographic classifiers produce comparable
        performance and assignments across datasets and models.
        """,
    )

with q3:
    card(
        "RQ3",
        "Does it affect recognition?",
        """
        Test whether facial-verification error rates differ across
        defensibly defined demographic and intersectional groups.
        """,
    )

st.divider()


# ============================================================
# CURRENT EVIDENCE
# ============================================================

section_label("Current evidence")

st.subheader("Track A is now complete")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "FairFace validation",
    "10,954 images",
)

m2.metric(
    "UTKFace aligned",
    "24,054 images",
)

m3.metric(
    "LFW compared",
    "13,233 images",
)

m4.metric(
    "LFW race disagreement",
    (
        f"{lfw_race_disagreement:.2%}"
        if lfw_race_disagreement is not None
        else "Unavailable"
    ),
)

gold_callout(
    """
    <strong>Key distinction:</strong> FairFace and UTKFace provide
    independent demographic labels for classifier evaluation. LFW does
    not provide independent race or gender ground truth in the canonical
    project table. LFW is therefore used here to study
    <strong>model-assigned demographic measurement sensitivity</strong>,
    not demographic classification accuracy.
    """
)

st.divider()


# ============================================================
# TRACK A FINDINGS
# ============================================================

section_label("Track A findings")

st.subheader("What the demographic model audit established")

f1, f2 = st.columns(2)

with f1:

    st.markdown(
        """
        ### Labelled-dataset performance

        On **FairFace validation**, FairFace ResNet-34 achieved
        **77.65% six-class race accuracy** and **94.39% gender
        accuracy**, compared with **65.75%** and **70.17%** for
        DeepFace.

        On **UTKFace**, four-class harmonised race accuracy was
        closer between models: **76.48%** for FairFace ResNet-34 and
        **75.16%** for DeepFace. Gender accuracy remained separated at
        **94.35%** and **86.23%**, respectively.
        """
    )

with f2:

    st.markdown(
        """
        ### Measurement sensitivity on LFW

        Across the same **13,233 LFW images**, the classifiers agreed
        on **74.71% of race assignments** and **95.37% of gender
        assignments**.

        This corresponds to **25.29% race disagreement** and
        **4.63% gender disagreement**, demonstrating that inferred
        subgroup composition can depend materially on the demographic
        annotation method.
        """
    )

blue_callout(
    """
    These findings do not establish why a disparity occurs. Training
    data, label taxonomy, preprocessing, domain shift and model design
    can all contribute. The audit therefore separates
    <strong>representation</strong>, <strong>demographic
    measurement</strong> and <strong>recognition performance</strong>
    rather than treating them as the same phenomenon.
    """
)

st.divider()


# ============================================================
# PROGRESS
# ============================================================

section_label("Progress")

st.subheader("Current project position")

a, b, c, d, e = st.columns(5)

a.metric(
    "Representation audit",
    "Complete",
)

b.metric(
    "Demographic model audit",
    "Complete",
)

c.metric(
    "Measurement validation",
    "In progress",
)

d.metric(
    "Face verification",
    "Next",
)

e.metric(
    "Mitigation",
    "Planned",
)

st.caption(
    """
    Track A covers demographic attribute classification and measurement
    sensitivity. Track B will evaluate facial verification and the
    resulting recognition-error disparities.
    """
)
