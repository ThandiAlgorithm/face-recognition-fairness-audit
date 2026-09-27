import streamlit as st


st.set_page_config(
    page_title="Facial Recognition Fairness Audit",
    page_icon="🔎",
    layout="wide",
)


st.title("Facial Recognition Fairness Audit")

st.subheader("Wits EIE Investigation Project")

st.write(
    """
    This project investigates whether demographic and visual skews
    in facial-recognition datasets are associated with unequal
    recognition performance, and whether those disparities can
    be reduced through technical mitigation.
    """
)

st.divider()

st.subheader("Investigation Pipeline")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 1. Dataset Audit")
    st.write(
        """
        What biases, imbalances and measurement issues exist
        in the datasets?
        """
    )

with col2:
    st.markdown("### 2. Baseline Model")
    st.write(
        """
        How well does a baseline facial-recognition system perform?
        """
    )

with col3:
    st.markdown("### 3. Fairness Audit")
    st.write(
        """
        Are recognition errors different across demographic
        and intersectional groups?
        """
    )

with col4:
    st.markdown("### 4. Mitigation")
    st.write(
        """
        Can we reduce those disparities without unacceptable
        loss of recognition performance?
        """
    )

st.divider()

st.subheader("Core Research Question")

st.info(
    """
    How do demographic and visual skews in facial-recognition
    datasets affect recognition performance, and can mitigation
    reduce the resulting disparities?
    """
)

st.subheader("Current Project Status")

st.success("Representation audit completed.")

st.info(
    """
    Measurement audit, baseline model evaluation and fairness
    analysis are currently in progress.
    """
)
