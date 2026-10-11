import streamlit as st


st.set_page_config(
    page_title=(
        "Facial Recognition Fairness Audit"
    ),
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


pages = [
    st.Page(
        "pages/overview.py",
        title="Overview",
        default=True,
    ),
    st.Page(
        "pages/dataset_audit.py",
        title="Dataset Audit",
    ),
    st.Page(
        "pages/baseline_model.py",
        title="Demographic Measurement",
    ),
    st.Page(
        "pages/reference_a.py",
        title="Face Recognition Audit",
    ),
    st.Page(
        "pages/mitigation.py",
        title="Mitigation",
    ),
    st.Page(
        "pages/harmoface_95k.py",
        title="HarmoFace-95K",
    ),
    st.Page(
        "pages/findings.py",
        title="Findings & Implications",
    ),
    st.Page(
        "pages/report_packs.py",
        title="Data & Methods",
    ),
]


navigation = st.navigation(
    pages,
    position="sidebar",
)

navigation.run()
