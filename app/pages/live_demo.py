from __future__ import annotations

import streamlit as st

from ui import (
    page_header,
    research_card,
    section_header,
)


# ============================================================
# HEADER
# ============================================================

page_header(
    "Interactive Face Verification",
    (
        "A demonstration interface showing how two face images can be "
        "compared by a facial-recognition system."
    ),
    "Interactive demonstration",
)


# ============================================================
# CONTEXT
# ============================================================

st.info(
    """
    This demonstration is separate from the controlled LFW research
    experiment. Camera images are intended for temporary demonstration
    only and are not added to the research datasets.
    """
)


# ============================================================
# CAPTURE
# ============================================================

section_header(
    "Capture two images",
    (
        "Use two photographs of the same person or two different "
        "people to demonstrate a verification comparison."
    ),
)

left, right = st.columns(2)

with left:

    with st.container(
        border=True
    ):

        st.markdown(
            "### Image A"
        )

        image_a = (
            st.camera_input(
                "Capture first image",
                key="camera_a",
            )
        )

with right:

    with st.container(
        border=True
    ):

        st.markdown(
            "### Image B"
        )

        image_b = (
            st.camera_input(
                "Capture second image",
                key="camera_b",
            )
        )


# ============================================================
# RESULT
# ============================================================

st.divider()

section_header(
    "Verification result"
)

if (
    image_a is None
    or image_b is None
):

    st.info(
        """
        Capture both images to enable the verification demonstration.
        """
    )

else:

    st.success(
        "Both images have been captured."
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Similarity",
        "—",
    )

    m2.metric(
        "Decision threshold",
        "—",
    )

    m3.metric(
        "Verification result",
        "—",
    )

    st.warning(
        """
        The camera interface is operational, but the recognition model
        has not yet been connected. This page should remain hidden from
        the public navigation until the verification pipeline is active.
        """
    )


# ============================================================
# DEMONSTRATION PURPOSE
# ============================================================

st.divider()

section_header(
    "What the demonstration will show"
)

items = st.columns(3)

with items[0]:

    research_card(
        "Similarity",
        (
            "The model converts both faces into numerical embeddings "
            "and measures how similar those representations are."
        ),
        "01",
    )

with items[1]:

    research_card(
        "Threshold",
        (
            "A fixed decision threshold determines whether the pair is "
            "classified as a match or non-match."
        ),
        "02",
    )

with items[2]:

    research_card(
        "Capture conditions",
        (
            "Visitors can observe how lighting, pose or image quality "
            "can influence the resulting similarity score."
        ),
        "03",
    )

st.caption(
    """
    No captured images should be retained after the demonstration
    session unless explicit consent and an approved data-handling
    process are in place.
    """
)
