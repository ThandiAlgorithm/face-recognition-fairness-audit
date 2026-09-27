import pandas as pd
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
    "Bias Mitigation",
    """
    Testing whether a targeted engineering intervention can reduce the
    disparity identified by the baseline experiment while maintaining
    useful face-recognition performance.
    """,
    eyebrow="Stage 4 · Engineering intervention",
)

section_label("Mitigation philosophy")

st.subheader("Diagnose first. Intervene second.")

blue_callout(
    """
    The mitigation method will be chosen in response to the bias
    mechanism identified by the audit and baseline model — not simply
    because a technique is popular.
    """
)

c1, c2, c3 = st.columns(3)

with c1:
    card(
        "01",
        "Identify",
        """
        Locate the largest defensible recognition-performance
        disparity.
        """,
    )

with c2:
    card(
        "02",
        "Intervene",
        """
        Apply one controlled mitigation such as resampling,
        reweighting or an appropriate training intervention.
        """,
    )

with c3:
    card(
        "03",
        "Compare",
        """
        Measure whether fairness improves and quantify the associated
        performance trade-off.
        """,
    )

st.divider()

section_label("Before vs after")

comparison = pd.DataFrame(
    {
        "Metric": [
            "Overall verification accuracy",
            "False Match Rate",
            "False Non-Match Rate",
            "Maximum subgroup gap",
            "Equalised-odds difference",
        ],
        "Baseline": [
            "Pending",
            "Pending",
            "Pending",
            "Pending",
            "Pending",
        ],
        "Mitigated": [
            "Pending",
            "Pending",
            "Pending",
            "Pending",
            "Pending",
        ],
    }
)

st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True,
)

st.subheader("Mitigation status")

st.progress(0.05)

st.caption(
    "Mitigation begins after a defensible baseline disparity has been established."
)
