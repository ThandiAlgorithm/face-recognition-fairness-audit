import streamlit as st

from ui import apply_wits_theme


st.set_page_config(
    page_title="Facial Recognition Fairness Audit",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_wits_theme()


# ---------------------------------------------------------
# Sidebar identity
# ---------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
<div style="padding:0.45rem 0 1.1rem 0;">
<div style="color:#F4C542;font-size:0.76rem;font-weight:800;letter-spacing:0.11em;text-transform:uppercase;">
University of the Witwatersrand
</div>
<div style="color:white;font-size:1.35rem;font-weight:800;line-height:1.2;margin-top:0.45rem;">
Facial Recognition<br>Fairness Audit
</div>
<div style="color:rgba(255,255,255,0.68);font-size:0.78rem;line-height:1.45;margin-top:0.55rem;">
School of Electrical and<br>Information Engineering
</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    

# ---------------------------------------------------------
# Explicit app navigation
# ---------------------------------------------------------

pages = {
    "Investigation": [
        st.Page(
            "pages/overview.py",
            title="Project Overview",
            icon="🏠",
            default=True,
        ),
        st.Page(
            "pages/dataset_audit.py",
            title="Dataset Audit",
            icon="📊",
        ),
        st.Page(
            "pages/baseline_model.py",
            title="Baseline Model",
            icon="🧠",
        ),
        st.Page(
            "pages/fairness_evaluation.py",
            title="Fairness Evaluation",
            icon="⚖️",
        ),
        st.Page(
            "pages/mitigation.py",
            title="Mitigation",
            icon="🛠️",
        ),
    ],
    "Demonstration": [
        st.Page(
            "pages/live_demo.py",
            title="Live Verification Demo",
            icon="📷",
        ),
    ],
}

navigation = st.navigation(
    pages,
    position="sidebar",
)

navigation.run()
