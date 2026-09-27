import streamlit as st

from ui import (
    apply_wits_theme,
    hero,
    section_label,
    card,
    blue_callout,
)


apply_wits_theme()

hero(
    "Facial Recognition Fairness Audit",
    """
    Investigating where demographic and visual bias enters a
    facial-recognition pipeline, whether those skews translate into
    unequal recognition performance, and whether a technical
    mitigation can reduce the resulting disparity.
    """,
)

section_label("Investigation structure")

st.subheader("From dataset bias to engineering intervention")

c1, c2, c3, c4 = st.columns(4)

with c1:
    card(
        "01",
        "Dataset Audit",
        """
        Identify representation, measurement and image-condition
        differences before evaluating the recogniser.
        """,
    )

with c2:
    card(
        "02",
        "Baseline Recognition",
        """
        Evaluate a fixed facial-recognition system using the
        LFW verification protocol.
        """,
    )

with c3:
    card(
        "03",
        "Fairness Evaluation",
        """
        Compare false matches and false non-matches across
        demographic and intersectional groups.
        """,
    )

with c4:
    card(
        "04",
        "Mitigation",
        """
        Test whether a targeted engineering intervention reduces
        disparity while preserving useful recognition performance.
        """,
    )

st.divider()

section_label("Research question")

st.subheader("The central investigation")

blue_callout(
    """
    <strong>How do demographic and visual skews in facial-recognition
    datasets affect recognition performance, and can mitigation reduce
    the resulting disparities?</strong>
    """
)

section_label("Research questions")

q1, q2, q3 = st.columns(3)

with q1:
    card(
        "RQ1",
        "What bias is present?",
        """
        Characterise demographic representation, identity depth,
        measurement limitations and relevant visual conditions.
        """,
    )

with q2:
    card(
        "RQ2",
        "What does the model do?",
        """
        Determine whether a baseline recognition system produces
        different error rates across the audited groups.
        """,
    )

with q3:
    card(
        "RQ3",
        "Can we reduce the gap?",
        """
        Apply a technically justified mitigation and quantify the
        resulting fairness–performance trade-off.
        """,
    )

st.divider()

section_label("Progress")

st.subheader("Current project position")

a, b, c, d = st.columns(4)

a.metric("Representation audit", "Complete")
b.metric("Measurement audit", "In progress")
c.metric("Baseline model", "Next")
d.metric("Mitigation", "Planned")

st.caption(
    "The dashboard will update as each experimental stage is completed."
)
