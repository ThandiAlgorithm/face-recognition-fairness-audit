from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from model_audit_data import (
    load_gender_equalized_odds,
    load_gender_fairness_by_race,
)
from ui import (
    apply_wits_theme,
    blue_callout,
    card,
    gold_callout,
    hero,
    section_label,
)


# ============================================================
# STYLE
# ============================================================

apply_wits_theme()

WITS_BLUE = "#0067B1"
WITS_GOLD = "#F4C542"

MODEL_COLOURS = {
    "FairFace ResNet-34": WITS_BLUE,
    "DeepFace": WITS_GOLD,
}


# ============================================================
# LOAD
# ============================================================

eo = load_gender_equalized_odds()
gender_rates = load_gender_fairness_by_race()

if eo.empty:
    st.error(
        """
        Track A fairness outputs were not found under
        results/model_audit/model_comparison/track_a_final/.
        """
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

hero(
    "Fairness Evaluation",
    """
    Quantifying demographic-classification disparities before the
    project proceeds to recognition-level fairness. This separates
    classifier measurement error from the later false-match and
    false-non-match analysis.
    """,
    eyebrow="Stage 2B · Demographic classifier fairness",
)

gold_callout(
    """
    <strong>Two fairness layers must remain separate.</strong>
    The results on this page describe fairness of the
    <strong>demographic attribute classifiers</strong>. They are not yet
    facial-verification FMR/FNMR fairness results.
    """
)


# ============================================================
# METRIC DEFINITIONS
# ============================================================

section_label("Metric")

st.subheader("Gender equalized-odds difference")

blue_callout(
    """
    Gender is treated as the binary classification target with
    <strong>Female</strong> as the positive class. For each labelled
    race group, the audit calculates female TPR and female FPR. The
    equalized-odds difference is the larger of:

    <br><br>
    <strong>max(TPR) − min(TPR)</strong> and
    <strong>max(FPR) − min(FPR)</strong>.

    <br><br>
    A larger value therefore indicates a larger measured difference in
    error behaviour across the evaluated race groups.
    """
)


# ============================================================
# TOP RESULTS
# ============================================================

section_label("Headline result")

st.subheader("Measured disparity across two independently labelled datasets")

def eo_value(
    dataset: str,
    model: str,
) -> float | None:

    row = eo[
        eo["dataset"].eq(dataset)
        & eo["model"].eq(model)
    ]

    if row.empty:
        return None

    return float(
        row.iloc[0][
            "equalized_odds_difference"
        ]
    )


ff_ff = eo_value(
    "FairFace validation",
    "FairFace ResNet-34",
)

ff_df = eo_value(
    "FairFace validation",
    "DeepFace",
)

utk_ff = eo_value(
    "UTKFace aligned",
    "FairFace ResNet-34",
)

utk_df = eo_value(
    "UTKFace aligned",
    "DeepFace",
)

a, b, c, d = st.columns(4)

a.metric(
    "FairFace · FairFace model",
    (
        f"{ff_ff * 100:.2f} pp"
        if ff_ff is not None
        else "Unavailable"
    ),
)

b.metric(
    "FairFace · DeepFace",
    (
        f"{ff_df * 100:.2f} pp"
        if ff_df is not None
        else "Unavailable"
    ),
)

c.metric(
    "UTKFace · FairFace model",
    (
        f"{utk_ff * 100:.2f} pp"
        if utk_ff is not None
        else "Unavailable"
    ),
)

d.metric(
    "UTKFace · DeepFace",
    (
        f"{utk_df * 100:.2f} pp"
        if utk_df is not None
        else "Unavailable"
    ),
)


# ============================================================
# EO FIGURE
# ============================================================

eo_plot = eo.copy()

eo_plot[
    "Equalized-odds difference (pp)"
] = (
    eo_plot[
        "equalized_odds_difference"
    ]
    * 100
)

fig = px.bar(
    eo_plot,
    x="dataset",
    y="Equalized-odds difference (pp)",
    color="model",
    barmode="group",
    text="Equalized-odds difference (pp)",
    color_discrete_map=MODEL_COLOURS,
    title="Gender equalized-odds difference",
)

fig.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
)

fig.update_layout(
    xaxis_title="",
    legend_title="Model",
    height=450,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.markdown(
    """
    ### Interpretation

    On **FairFace validation**, the gender equalized-odds difference was
    **7.43 percentage points** for FairFace ResNet-34 and
    **34.03 percentage points** for DeepFace.

    On **UTKFace**, the corresponding differences were
    **6.18 percentage points** and **19.22 percentage points**.

    The magnitude changes substantially across datasets, but under both
    labelled evaluations DeepFace exhibits a larger measured
    cross-race gender-classification disparity than FairFace ResNet-34.
    """
)

gold_callout(
    """
    This is an empirical comparison under the evaluated datasets and
    preprocessing pipeline. It does not establish whether the disparity
    originates from training-data composition, demographic taxonomy,
    image domain, preprocessing or model architecture.
    """
)


# ============================================================
# GROUP RATES
# ============================================================

section_label("Subgroup rates")

st.subheader("Where does the equalized-odds difference come from?")

dataset_choice = st.radio(
    "Evaluation dataset",
    [
        "FairFace validation",
        "UTKFace aligned",
    ],
    horizontal=True,
)

selected = gender_rates[
    gender_rates["dataset"].eq(
        dataset_choice
    )
].copy()

if not selected.empty:

    selected[
        "Female TPR (%)"
    ] = (
        selected[
            "female_tpr"
        ]
        * 100
    )

    selected[
        "Female FPR (%)"
    ] = (
        selected[
            "female_fpr"
        ]
        * 100
    )

    left, right = st.columns(2)

    with left:

        fig = px.bar(
            selected,
            x="race",
            y="Female TPR (%)",
            color="model",
            barmode="group",
            color_discrete_map=MODEL_COLOURS,
            title=(
                f"{dataset_choice}: female true-positive rate"
            ),
        )

        fig.update_layout(
            xaxis_title="Labelled race group",
            legend_title="Model",
            height=430,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with right:

        fig = px.bar(
            selected,
            x="race",
            y="Female FPR (%)",
            color="model",
            barmode="group",
            color_discrete_map=MODEL_COLOURS,
            title=(
                f"{dataset_choice}: female false-positive rate"
            ),
        )

        fig.update_layout(
            xaxis_title="Labelled race group",
            legend_title="Model",
            height=430,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    display = selected[
        [
            "model",
            "race",
            "n",
            "n_female",
            "n_male",
            "Female TPR (%)",
            "Female FPR (%)",
        ]
    ].rename(
        columns={
            "model": "Model",
            "race": "Race group",
            "n": "Images",
            "n_female": "Female-labelled",
            "n_male": "Male-labelled",
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Female TPR (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
            "Female FPR (%)": st.column_config.NumberColumn(
                format="%.2f%%"
            ),
        },
    )


# ============================================================
# FAIRFACE INTERSECTIONAL RESULT
# ============================================================

section_label("Intersectional evidence")

st.subheader("FairFace validation illustrates why averages are insufficient")

left, right = st.columns(2)

with left:

    card(
        "FairFace ResNet-34",
        "More stable gender performance",
        """
        Gender accuracy remained comparatively high across the native
        FairFace race groups, although race-classification performance
        still varied substantially by demographic intersection.
        """,
    )

with right:

    card(
        "DeepFace",
        "Large female/male performance separation",
        """
        DeepFace showed pronounced gender-classification differences
        between female- and male-labelled samples within several race
        groups, producing the larger equalized-odds difference.
        """,
    )

st.markdown(
    """
    Selected FairFace-validation examples from the full intersectional
    analysis:

    | Intersection | FairFace gender accuracy | DeepFace gender accuracy |
    |---|---:|---:|
    | Black female | 89.83% | 16.64% |
    | Black male | 90.11% | 98.50% |
    | Indian female | 95.41% | 28.44% |
    | Indian male | 95.88% | 99.07% |
    | White female | 93.56% | 50.68% |
    | White male | 94.74% | 96.26% |
    """
)

blue_callout(
    """
    The result does not support a simple statement such as
    “one demographic group is always worst.” The size and direction of
    the disparity depend on the task, demographic intersection, model
    and dataset.
    """
)


# ============================================================
# LFW BOUNDARY
# ============================================================

section_label("LFW fairness boundary")

st.subheader("Why Track A does not claim LFW demographic accuracy")

gold_callout(
    """
    LFW has no independent demographic ground truth in the canonical
    project table. FairFace ResNet-34 and DeepFace labels on LFW are
    therefore <strong>measurements produced by models</strong>.

    Because the two classifiers disagree on approximately
    <strong>25.29% of race assignments</strong>, any future LFW
    recognition-fairness analysis conditioned on inferred race must
    explicitly acknowledge annotation sensitivity.
    """
)

st.write(
    """
    A later verification audit can therefore use model-assigned
    demographic groupings as a sensitivity analysis, but those groupings
    must not be presented as unquestioned demographic truth.
    """
)


# ============================================================
# TRACK B
# ============================================================

st.divider()

section_label("Next fairness layer")

st.subheader("Recognition fairness — Track B")

c1, c2, c3, c4 = st.columns(4)

with c1:
    card(
        "FMR",
        "False Match Rate",
        """
        How often different identities are incorrectly accepted as the
        same person.
        """,
    )

with c2:
    card(
        "FNMR",
        "False Non-Match Rate",
        """
        How often two images of the same identity are incorrectly
        rejected.
        """,
    )

with c3:
    card(
        "TPR / TAR",
        "Genuine acceptance",
        """
        The fraction of genuine same-identity comparisons accepted at
        the frozen verification threshold.
        """,
    )

with c4:
    card(
        "FPR / FAR",
        "Impostor acceptance",
        """
        The fraction of different-identity comparisons incorrectly
        accepted at the frozen threshold.
        """,
    )

st.caption(
    """
    Track B will evaluate genuine and impostor pairs overall and by
    defensibly defined demographic groups. Threshold selection should
    occur on development data rather than the sealed final evaluation
    split.
    """
)
