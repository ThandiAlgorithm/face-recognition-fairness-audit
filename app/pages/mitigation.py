from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import (
    evidence_note,
    finding_card,
    metric_row,
    page_header,
    research_card,
    section_header,
)


# ============================================================
# HEADER
# ============================================================

page_header(
    "Mitigation",
    (
        "A controlled resampling intervention designed to reduce "
        "training-distribution imbalance while preserving useful "
        "recognition performance."
    ),
    "Engineering intervention",
)


# ============================================================
# WHY MITIGATION
# ============================================================

section_header(
    "Why mitigation was required",
    (
        "The intervention follows the evidence produced by the "
        "dataset, demographic-measurement and recognition audits."
    ),
)

left, middle, right = st.columns(3)

with left:

    finding_card(
        "Representation imbalance",
        (
            "The source datasets differ substantially in demographic "
            "composition, identity depth and dataset size."
        ),
        (
            "Training distributions can over-emphasise groups that "
            "are already strongly represented."
        ),
    )

with middle:

    finding_card(
        "Measurement differences",
        (
            "Demographic labels and measured subgroup composition vary "
            "with the annotation model and evaluation dataset."
        ),
        (
            "Mitigation should not assume that demographic categories "
            "are perfectly measured or interchangeable with skin tone."
        ),
    )

with right:

    finding_card(
        "Recognition disparities",
        (
            "Verification performance can differ between demographic "
            "groups even when aggregate recognition accuracy remains "
            "high."
        ),
        (
            "Mitigation must be evaluated using subgroup error rates "
            "as well as overall performance."
        ),
    )

st.divider()


# ============================================================
# INTERVENTION
# ============================================================

section_header(
    "Mitigation strategy",
    (
        "Resampling is applied to each dataset-specific training "
        "pipeline and compared with its original baseline."
    ),
)

st.info(
    """
    The intervention changes the training distribution through
    resampling. The evaluation protocol, model architecture and
    performance metrics should remain fixed so that differences can be
    attributed as cleanly as possible to the mitigation strategy.
    """
)

models = st.columns(3)

with models[0]:

    research_card(
        "FairFace-trained model",
        (
            "Original FairFace training distribution compared with a "
            "resampled FairFace training distribution."
        ),
        "Model 01",
    )

with models[1]:

    research_card(
        "UTKFace-trained model",
        (
            "Original UTKFace training distribution compared with a "
            "resampled UTKFace training distribution."
        ),
        "Model 02",
    )

with models[2]:

    research_card(
        "LFW-trained model",
        (
            "Original LFW training distribution compared with a "
            "resampled LFW training distribution."
        ),
        "Model 03",
    )

st.divider()


# ============================================================
# EXPERIMENTAL DESIGN
# ============================================================

section_header(
    "Controlled comparison"
)

before, intervention, after = st.columns(3)

with before:

    research_card(
        "Baseline",
        (
            "Train the model using the original dataset distribution "
            "and record overall and subgroup performance."
        ),
        "01",
    )

with intervention:

    research_card(
        "Resampling",
        (
            "Modify the training sample distribution while keeping the "
            "recognition task and evaluation protocol unchanged."
        ),
        "02",
    )

with after:

    research_card(
        "Re-evaluation",
        (
            "Repeat the same evaluation and quantify improvements, "
            "remaining disparities and performance trade-offs."
        ),
        "03",
    )

st.divider()


# ============================================================
# METRICS
# ============================================================

section_header(
    "Evaluation criteria",
    (
        "Mitigation is successful only if fairness improvement is "
        "considered together with recognition utility."
    ),
)

metric_cards = st.columns(4)

with metric_cards[0]:

    evidence_note(
        "Overall performance",
        (
            "Accuracy, balanced accuracy and class-balanced "
            "performance."
        ),
    )

with metric_cards[1]:

    evidence_note(
        "False Match Rate",
        (
            "Whether impostor pairs are incorrectly accepted and how "
            "that behaviour changes across groups."
        ),
    )

with metric_cards[2]:

    evidence_note(
        "False Non-Match Rate",
        (
            "Whether genuine pairs are incorrectly rejected and how "
            "that behaviour changes across groups."
        ),
    )

with metric_cards[3]:

    evidence_note(
        "Subgroup disparity",
        (
            "Maximum performance gaps and fairness metrics before and "
            "after mitigation."
        ),
    )

st.divider()


# ============================================================
# MODEL COMPARISON
# ============================================================

section_header(
    "Baseline versus mitigated models",
    (
        "The final comparison will report each dataset-specific model "
        "before and after resampling."
    ),
)

model_tabs = st.tabs(
    [
        "FairFace",
        "UTKFace",
        "LFW",
    ]
)


def model_comparison_panel(
    dataset_name: str,
) -> None:

    st.markdown(
        f"### {dataset_name}-trained model"
    )

    baseline, mitigated = st.columns(2)

    with baseline:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### Baseline"
            )

            st.write(
                """
                Model trained using the original dataset distribution.
                """
            )

    with mitigated:

        with st.container(
            border=True
        ):

            st.markdown(
                "#### After resampling"
            )

            st.write(
                """
                Same modelling pipeline trained using the mitigated
                sample distribution.
                """
            )

    comparison = pd.DataFrame(
        {
            "Metric": [
                "Overall accuracy",
                "Balanced accuracy",
                "Macro F1",
                "False Match Rate",
                "False Non-Match Rate",
                "Maximum subgroup gap",
            ],
            "Baseline": [
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
            ],
            "After resampling": [
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
            ],
        }
    )

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
    )


with model_tabs[0]:
    model_comparison_panel(
        "FairFace"
    )

with model_tabs[1]:
    model_comparison_panel(
        "UTKFace"
    )

with model_tabs[2]:
    model_comparison_panel(
        "LFW"
    )

st.caption(
    """
    Numerical comparison tables will be populated from the final
    mitigation result files. Values are intentionally not inferred or
    reconstructed here.
    """
)

st.divider()


# ============================================================
# INTERPRETATION
# ============================================================

section_header(
    "How mitigation will be judged"
)

left, right = st.columns(2)

with left:

    research_card(
        "Evidence of improvement",
        (
            "A subgroup disparity decreases while overall recognition "
            "performance remains within an acceptable range."
        ),
        "Desired outcome",
    )

with right:

    research_card(
        "Trade-off",
        (
            "A fairness metric improves but overall performance or "
            "another subgroup metric deteriorates materially."
        ),
        "Important outcome",
    )

st.info(
    """
    A mitigation result should not be described as successful simply
    because one fairness metric improves. The full before/after
    performance profile must be considered.
    """
)
