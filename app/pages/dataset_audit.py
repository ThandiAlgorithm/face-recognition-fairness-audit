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
    "Dataset Audit",
    """
    Understanding what is represented in the data, how reliably it can
    be measured, and which dataset characteristics may later influence
    facial-recognition performance.
    """,
    eyebrow="Stage 1 · Forensic dataset audit",
)

section_label("Audit population")

st.subheader("Three datasets, different roles")

m1, m2, m3, m4 = st.columns(4)

m1.metric("LFW images", "13,233")
m2.metric("LFW identities", "5,749")
m3.metric("FairFace images", "97,698")
m4.metric("UTKFace images", "24,108")

dataset_table = pd.DataFrame(
    {
        "Dataset": ["LFW", "FairFace", "UTKFace"],
        "Images": [13233, 97698, 24108],
        "Role in investigation": [
            "Primary verification benchmark",
            "Demographic comparison dataset",
            "Age and demographic comparison dataset",
        ],
    }
)

st.dataframe(
    dataset_table,
    use_container_width=True,
    hide_index=True,
)

st.subheader("Dataset size")

st.bar_chart(
    dataset_table.set_index("Dataset")[["Images"]],
    height=330,
)

st.divider()

section_label("Audit dimensions")

st.subheader("What are we looking for?")

c1, c2, c3 = st.columns(3)

with c1:
    card(
        "A",
        "Representation",
        """
        Gender, age, available race labels, identities, images per
        identity, intersections and missing demographic information.
        """,
    )

with c2:
    card(
        "B",
        "Visual conditions",
        """
        Brightness, contrast, sharpness, pose, face size, detection
        failures and alignment failures.
        """,
    )

with c3:
    card(
        "C",
        "Measurement validity",
        """
        Skin-region extraction, CIELAB colour values, ITA, hue,
        skin-tone proxies and their limitations.
        """,
    )

st.divider()

section_label("Current findings")

st.subheader("Representation audit")

left, right = st.columns([1, 1])

with left:
    st.markdown(
        """
        #### LFW identity structure

        - **13,233** images
        - **5,749** identities
        - **4,069** identities represented by only one image
        - Long-tailed identity representation
        """
    )

with right:
    st.markdown(
        """
        #### Demographic metadata

        FairFace and UTKFace provide native demographic labels.

        Standard LFW metadata does not provide the complete
        demographic information required for all planned subgroup
        comparisons.
        """
    )

gold_callout(
    """
    <strong>Important:</strong> Dataset imbalance is evidence about the
    data, not proof that the recognition model is unfair. The baseline
    experiment is required to determine whether these differences
    translate into unequal recognition errors.
    """
)

st.caption(
    "Interactive representation figures will be connected to the "
    "project's generated CSV outputs in the next dashboard update."
)
