import streamlit as st

from ui import (
    apply_wits_theme,
    hero,
    section_label,
    blue_callout,
)


apply_wits_theme()

hero(
    "Live Face Verification",
    """
    An interactive demonstration of facial verification and how capture
    conditions can influence the similarity score produced by a
    recognition system.
    """,
    eyebrow="Open Day · Interactive demonstration",
)

section_label("Capture")

st.subheader("Compare two images")

blue_callout(
    """
    This demonstration is separate from the controlled scientific
    experiment on LFW. Images captured here are intended for temporary
    verification only and will not be added to the research datasets.
    """
)

left, right = st.columns(2)

with left:
    st.markdown("### Image A")
    image_a = st.camera_input(
        "Capture first image",
        key="camera_a",
    )

with right:
    st.markdown("### Image B")
    image_b = st.camera_input(
        "Capture second image",
        key="camera_b",
    )

st.divider()

section_label("Verification result")

if image_a is None or image_b is None:

    st.info(
        "Capture both images to enable the verification pipeline."
    )

else:

    st.success("Both images captured successfully.")

    c1, c2, c3 = st.columns(3)

    c1.metric("Similarity score", "Not connected")
    c2.metric("Decision threshold", "Not connected")
    c3.metric("Verification", "Pending")

    st.warning(
        """
        Webcam capture is working. The face-recognition model has not
        yet been connected to this interface.
        """
    )

st.divider()

section_label("Planned Open Day experiment")

st.write(
    """
    Visitors will eventually be able to compare two images and observe
    how pose, lighting, glasses or other capture conditions change the
    similarity score.

    This provides an intuitive demonstration of why image conditions
    matter in facial-recognition systems.
    """
)
