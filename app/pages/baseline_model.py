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
    "Baseline Face Recognition",
    """
    Establishing the reference recognition performance against which
    demographic error disparities and later mitigation will be
    evaluated.
    """,
    eyebrow="Stage 2 · Baseline experiment",
)

section_label("System architecture")

st.subheader("Detection is not recognition")

left, right = st.columns(2)

with left:
    card(
        "01",
        "Face detector",
        """
        RetinaFace locates the face and facial landmarks. Its job is
        detection and alignment — not deciding identity.
        """,
    )

with right:
    card(
        "02",
        "Face recogniser",
        """
        A recognition network converts each aligned face into a
        numerical embedding used to compare identity similarity.
        """,
    )

st.divider()

section_label("Verification pipeline")

st.subheader("How one LFW pair is evaluated")

st.code(
    """
LFW image A                    LFW image B
     |                              |
     v                              v
Face detection                 Face detection
     |                              |
     v                              v
Face alignment                 Face alignment
     |                              |
     +------------+  +--------------+
                  |  |
                  v  v
          Recognition network
                  |
          Embedding A / B
                  |
                  v
           Similarity score
                  |
                  v
          Fixed threshold
                  |
             +----+----+
             |         |
           MATCH    NON-MATCH
    """,
    language=None,
)

st.divider()

section_label("Planned outputs")

a, b, c, d = st.columns(4)

a.metric("Verification accuracy", "Pending")
b.metric("False Match Rate", "Pending")
c.metric("False Non-Match Rate", "Pending")
d.metric("ROC / AUC", "Pending")

blue_callout(
    """
    The first baseline should use a fixed pretrained recognition model.
    Training or fine-tuning can then be introduced only if required by
    the final mitigation experiment.
    """
)

st.subheader("Experiment status")

st.progress(0.20)

st.caption(
    "Model selection and LFW verification implementation are being prepared."
)
