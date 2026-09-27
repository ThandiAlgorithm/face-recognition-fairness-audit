import pandas as pd
import streamlit as st

from ui import (
    apply_wits_theme,
    hero,
    section_label,
    card,
    gold_callout,
)


apply_wits_theme()

hero(
    "Fairness Evaluation",
    """
    Moving beyond demographic counts to determine whether the
    recognition system actually makes different kinds or numbers of
    errors for different groups.
    """,
    eyebrow="Stage 3 · Model fairness audit",
)

section_label("Core idea")

st.subheader("The important question changes")

gold_callout(
    """
    The question is no longer only <strong>“Who is represented in the
    dataset?”</strong><br><br>

    It becomes <strong>“Does the recognition system make more mistakes
    for some groups than for others?”</strong>
    """
)

section_label("Primary metrics")

c1, c2, c3, c4 = st.columns(4)

with c1:
    card(
        "FMR",
        "False Match Rate",
        "How often different people are incorrectly accepted as the same person.",
    )

with c2:
    card(
        "FNMR",
        "False Non-Match Rate",
        "How often images of the same person are incorrectly rejected.",
    )

with c3:
    card(
        "TPR",
        "True Positive Rate",
        "How often genuine matches are correctly accepted.",
    )

with c4:
    card(
        "FPR",
        "False Positive Rate",
        "How often impostor comparisons are incorrectly accepted.",
    )

st.divider()

section_label("Planned subgroup analysis")

planned = pd.DataFrame(
    {
        "Dimension": [
            "Gender",
            "Age",
            "Skin tone",
            "Race / available demographic provenance",
            "Intersectional groups",
            "Image conditions",
        ],
        "Question": [
            "Are error rates different between gender groups?",
            "Does verification performance vary with age?",
            "Does measured skin tone relate to verification error?",
            "Which comparisons are defensible from available labels?",
            "Do combined demographic groups show larger disparities?",
            "Can lighting, pose or quality explain part of the gap?",
        ],
        "Status": [
            "Planned",
            "Planned",
            "In preparation",
            "Requires provenance decision",
            "Planned",
            "Planned",
        ],
    }
)

st.dataframe(
    planned,
    hide_index=True,
    use_container_width=True,
)

st.divider()

section_label("Results")

a, b, c = st.columns(3)

a.metric("Largest observed gap", "Pending")
b.metric("Most affected subgroup", "Pending")
c.metric("Equalised-odds comparison", "Pending")

st.caption(
    "These values will be populated automatically from the baseline "
    "experiment outputs."
)
