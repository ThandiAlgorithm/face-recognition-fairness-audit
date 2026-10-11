from __future__ import annotations

from typing import Iterable

import streamlit as st


def page_header(
    title: str,
    subtitle: str,
    eyebrow: str | None = None,
) -> None:
    """Standard page heading used across the public website."""

    if eyebrow:
        st.caption(eyebrow.upper())

    st.title(title)
    st.markdown(
        f"#### {subtitle}"
    )
    st.divider()


def section_header(
    title: str,
    description: str | None = None,
) -> None:
    """Consistent section heading."""

    st.subheader(title)

    if description:
        st.caption(description)


def metric_row(
    metrics: Iterable[tuple[str, str]],
) -> None:
    """Render a responsive row of Streamlit metric cards."""

    metrics = list(metrics)

    columns = st.columns(
        len(metrics)
    )

    for column, (
        label,
        value,
    ) in zip(
        columns,
        metrics,
    ):
        with column:
            st.metric(
                label,
                value,
            )


def research_card(
    title: str,
    text: str,
    label: str | None = None,
) -> None:
    """Native Streamlit research card."""

    with st.container(
        border=True
    ):
        if label:
            st.caption(
                label.upper()
            )

        st.markdown(
            f"### {title}"
        )

        st.write(
            text
        )


def finding_card(
    title: str,
    result: str,
    implication: str,
) -> None:
    """Display one technical finding and its engineering implication."""

    with st.container(
        border=True
    ):
        st.markdown(
            f"### {title}"
        )

        st.write(
            result
        )

        st.markdown(
            "**Engineering implication**"
        )

        st.write(
            implication
        )


def evidence_note(
    title: str,
    text: str,
) -> None:
    """Small technical note without custom HTML."""

    with st.container(
        border=True
    ):
        st.markdown(
            f"**{title}**"
        )

        st.write(
            text
        )
        
