import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Dataset Audit",
    page_icon="📊",
    layout="wide",
)

st.title("Dataset Audit")

st.write(
    """
    This stage examines the datasets before evaluating the
    facial-recognition model.

    The goal is to identify representation, demographic,
    measurement and image-condition differences that could later
    contribute to unequal model performance.
    """
)

st.divider()

st.subheader("Datasets")

col1, col2, col3, col4 = st.columns(4)

col1.metric("LFW images", "13,233")
col2.metric("LFW identities", "5,749")
col3.metric("FairFace images", "97,698")
col4.metric("UTKFace images", "24,108")

st.subheader("Dataset roles")

dataset_table = pd.DataFrame(
    {
        "Dataset": ["LFW", "FairFace", "UTKFace"],
        "Images": [13233, 97698, 24108],
        "Role": [
            "Primary face-verification benchmark",
            "Demographic comparison dataset",
            "Age and demographic comparison dataset",
        ],
    }
)

st.dataframe(
    dataset_table,
    hide_index=True,
    use_container_width=True,
)

st.subheader("Dataset size comparison")

chart_data = dataset_table.set_index("Dataset")[["Images"]]

st.bar_chart(chart_data)

st.divider()

st.subheader("What are we looking for?")

left, right = st.columns(2)

with left:
    st.markdown(
        """
        ### Representation

        - Gender distribution
        - Age distribution
        - Race labels where available
        - Images per identity
        - Intersectional representation
        - Missing demographic labels
        """
    )

with right:
    st.markdown(
        """
        ### Image and measurement conditions

        - Skin-tone measurements
        - Brightness and contrast
        - Blur and sharpness
        - Face size
        - Pose
        - Detection and alignment failures
        """
    )

st.divider()

st.subheader("Current status")

st.success("Representation audit completed.")

st.info(
    """
    Measurement and skin-region validation are currently
    in progress.
    """
)

st.warning(
    """
    Dataset imbalance alone does not prove that a facial-recognition
    model is biased. The next stage tests whether these dataset
    differences correspond to differences in recognition error.
    """
)
