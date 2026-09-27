import streamlit as st


# ---------------------------------------------------------
# Wits-inspired project palette
# ---------------------------------------------------------

NAVY = "#002B5C"
BLUE = "#0067B1"
LIGHT_BLUE = "#EAF3FA"
GOLD = "#F4C542"
PALE_GOLD = "#FFF8DF"

INK = "#102A43"
MUTED = "#627D98"
WHITE = "#FFFFFF"
BACKGROUND = "#F7F9FC"
BORDER = "#D9E2EC"

SUCCESS = "#13795B"
SUCCESS_BG = "#E8F5F0"

WARNING = "#9A6700"
WARNING_BG = "#FFF6D6"


def apply_wits_theme():
    """Apply shared project styling."""

    st.markdown(
        f"""
        <style>

        /* -----------------------------------------------------
           Global page
        ----------------------------------------------------- */

        .stApp {{
            background:
                linear-gradient(
                    180deg,
                    {BACKGROUND} 0%,
                    #FFFFFF 42%,
                    #F8FAFC 100%
                );
        }}

        .block-container {{
            max-width: 1280px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }}

        h1, h2, h3 {{
            color: {NAVY};
            letter-spacing: -0.02em;
        }}

        p {{
            line-height: 1.65;
        }}

        /* -----------------------------------------------------
           Sidebar
        ----------------------------------------------------- */

        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(
                    180deg,
                    {NAVY} 0%,
                    #003E73 55%,
                    #005A9C 100%
                );
        }}

        section[data-testid="stSidebar"] * {{
            color: white;
        }}

        section[data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.20);
        }}

        section[data-testid="stSidebar"] a {{
            text-decoration: none;
        }}

        /* Navigation entries */
        section[data-testid="stSidebar"] [data-testid="stPageLink"] {{
            border-radius: 10px;
            margin-bottom: 4px;
            padding-top: 0.2rem;
            padding-bottom: 0.2rem;
        }}

        section[data-testid="stSidebar"] [data-testid="stPageLink"]:hover {{
            background: rgba(255,255,255,0.10);
        }}

        /* -----------------------------------------------------
           Hero
        ----------------------------------------------------- */

        .wits-hero {{
            background:
                linear-gradient(
                    120deg,
                    {NAVY} 0%,
                    #004B87 58%,
                    {BLUE} 100%
                );
            border-radius: 20px;
            padding: 2.4rem 2.6rem;
            color: white;
            margin-bottom: 1.8rem;
            box-shadow: 0 18px 45px rgba(0,43,92,0.14);
            position: relative;
            overflow: hidden;
        }}

        .wits-hero:after {{
            content: "";
            position: absolute;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            right: -65px;
            top: -95px;
            background: rgba(244,197,66,0.17);
        }}

        .wits-eyebrow {{
            color: {GOLD};
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }}

        .wits-hero-title {{
            color: white;
            font-size: 2.55rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.8rem;
            max-width: 850px;
        }}

        .wits-hero-copy {{
            color: rgba(255,255,255,0.87);
            font-size: 1.04rem;
            line-height: 1.7;
            max-width: 880px;
        }}

        /* -----------------------------------------------------
           Section title
        ----------------------------------------------------- */

        .section-label {{
            display: inline-block;
            color: {BLUE};
            background: {LIGHT_BLUE};
            font-weight: 700;
            font-size: 0.76rem;
            padding: 0.34rem 0.68rem;
            border-radius: 999px;
            margin-bottom: 0.35rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        /* -----------------------------------------------------
           Cards
        ----------------------------------------------------- */

        .wits-card {{
            background: white;
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1.25rem 1.3rem;
            min-height: 175px;
            box-shadow: 0 7px 24px rgba(16,42,67,0.05);
            margin-bottom: 0.75rem;
        }}

        .wits-card-number {{
            color: {GOLD};
            font-size: 0.88rem;
            font-weight: 800;
            letter-spacing: 0.08em;
        }}

        .wits-card-title {{
            color: {NAVY};
            font-size: 1.13rem;
            font-weight: 800;
            margin-top: 0.45rem;
            margin-bottom: 0.5rem;
        }}

        .wits-card-copy {{
            color: {MUTED};
            font-size: 0.91rem;
            line-height: 1.6;
        }}

        /* -----------------------------------------------------
           Callouts
        ----------------------------------------------------- */

        .blue-callout {{
            border-left: 5px solid {BLUE};
            background: {LIGHT_BLUE};
            padding: 1.1rem 1.25rem;
            border-radius: 0 12px 12px 0;
            color: {INK};
            margin: 0.7rem 0 1.4rem 0;
        }}

        .gold-callout {{
            border-left: 5px solid {GOLD};
            background: {PALE_GOLD};
            padding: 1.1rem 1.25rem;
            border-radius: 0 12px 12px 0;
            color: {INK};
            margin: 0.7rem 0 1.4rem 0;
        }}

        .success-card {{
            border-left: 5px solid {SUCCESS};
            background: {SUCCESS_BG};
            padding: 1rem 1.2rem;
            border-radius: 0 12px 12px 0;
            color: {INK};
        }}

        /* -----------------------------------------------------
           Metric cards
        ----------------------------------------------------- */

        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid {BORDER};
            padding: 1rem 1.1rem;
            border-radius: 14px;
            box-shadow: 0 5px 16px rgba(16,42,67,0.05);
        }}

        div[data-testid="stMetricLabel"] {{
            color: {MUTED};
        }}

        div[data-testid="stMetricValue"] {{
            color: {NAVY};
        }}

        /* -----------------------------------------------------
           Buttons
        ----------------------------------------------------- */

        .stButton > button {{
            background: {NAVY};
            color: white;
            border: none;
            border-radius: 9px;
            font-weight: 700;
            padding: 0.55rem 1.15rem;
        }}

        .stButton > button:hover {{
            background: {BLUE};
            color: white;
            border: none;
        }}

        /* -----------------------------------------------------
           Dataframes
        ----------------------------------------------------- */

        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 14px;
            overflow: hidden;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, eyebrow: str = "Wits EIE Investigation Project"):
    st.markdown(
        f"""
        <div class="wits-hero">
            <div class="wits-eyebrow">{eyebrow}</div>
            <div class="wits-hero-title">{title}</div>
            <div class="wits-hero-copy">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str):
    st.markdown(
        f'<div class="section-label">{text}</div>',
        unsafe_allow_html=True,
    )


def card(number: str, title: str, copy: str):
    st.markdown(
        f"""
        <div class="wits-card">
            <div class="wits-card-number">{number}</div>
            <div class="wits-card-title">{title}</div>
            <div class="wits-card-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def blue_callout(text: str):
    st.markdown(
        f'<div class="blue-callout">{text}</div>',
        unsafe_allow_html=True,
    )


def gold_callout(text: str):
    st.markdown(
        f'<div class="gold-callout">{text}</div>',
        unsafe_allow_html=True,
    )
    