# telecom_dash/docs.py
from __future__ import annotations

import streamlit as st

# ------- small knowledge base -------

_METRIC_TEXT = {
    "avg_d_kbps": {
        "label": "Download speed",
        "unit": "Mbps",
        "desc": (
            "Median download throughput observed in the tile. "
            "Higher values are better. Shown in **Mbps** (original data in kbps)."
        ),
        "color_rule": "Green = fast (≥ P90), Yellow = typical (~P50–P90), Red = slow (< P50).",
    },
    "avg_u_kbps": {
        "label": "Upload speed",
        "unit": "Mbps",
        "desc": (
            "Median upload throughput observed in the tile. "
            "Higher values are better. Shown in **Mbps** (original data in kbps)."
        ),
        "color_rule": "Green = fast (≥ P90), Yellow = typical (~P50–P90), Red = slow (< P50).",
    },
    "avg_lat_ms": {
        "label": "Latency",
        "unit": "ms",
        "desc": (
            "Median round-trip latency observed in the tile. "
            "Lower values are better. Shown in **milliseconds**."
        ),
        "color_rule": "Green = low (≤ P10), Yellow = typical (~P10–P50), Red = high (> P50).",
    },
}

_GLOSSARY = [
    ("Tile", "A small spatial cell (WKT polygon) aggregating tests in that area."),
    ("Download / Upload", "Median tile-level throughputs from user-initiated tests."),
    ("Latency", "Median round-trip network delay in milliseconds (lower is better)."),
    ("Tests", "Number of test runs contributing to the tile stats."),
    ("Devices", "Distinct devices that ran tests in the tile."),
    ("Tower", "OpenCelliD point representing a cell site location (by radio tech)."),
    ("Tower density", "Count of towers whose coordinates fall inside a tile polygon."),
    ("Anomaly", "Tile flagged when z-score ≤ threshold (latency inverted so high latency is bad)."),
    ("Cluster", "K-means performance segment ranked best→worst (DL↑, UL↑, Latency↓)."),
    ("Quadkey", "Microsoft Bing tile identifier for spatial indexing."),
]

# ------- components -------

def glossary_popover() -> None:
    """Compact popover with key terms. Place near the page title."""
    with st.popover("ℹ️ Glossary / KPI meanings", use_container_width=True):
        st.markdown("### Glossary")
        for term, text in _GLOSSARY:
            st.markdown(f"- **{term}** — {text}")
        st.markdown("---")
        st.markdown("**Color logic**")
        st.markdown(
            "- **Throughput (DL/UL):** green (≥ P90), yellow (P50–P90), red (< P50).  \n"
            "- **Latency:** green (≤ P10), yellow (P10–P50), red (> P50)."
        )
        st.caption("PXX = Percentile of the currently visible tiles (after filters).")


def methodology_expander() -> None:
    """Sidebar expander with data sources & caveats."""
    with st.sidebar.expander("🧪 Data & methodology", expanded=False):
        st.markdown("**Sources**")
        st.markdown(
            "- **Ookla Open Data (Q4 2024)** — tile KPIs (non-commercial terms).\n"
            "- **OpenCelliD** — cell site points for density visualization."
        )
        st.markdown("**Processing**")
        st.markdown(
            "- Tiles filtered by **Min tests** (sidebar) for statistical stability.\n"
            "- **Tower density** computed via point-in-polygon (STRtree).\n"
            "- **Anomalies**: z-score on the selected KPI (latency inverted).\n"
            "- **Clusters**: k-means on DL/UL/Latency; legend ranks best→worst."
        )
        st.markdown("**Caveats**")
        st.markdown(
            "- Open data can be sparse/biased by user behavior and device mix.\n"
            "- Percentile thresholds reflect **current filters**, not all London.\n"
            "- Tower database may include multiple technologies or sectors per site."
        )


def explain_metric(metric: str) -> None:
    """Inline, metric-aware helper under the controls."""
    info = _METRIC_TEXT.get(metric)
    if not info:
        return
    st.caption(
        f"**{info['label']}** — {info['desc']}  \n"
        f"Legend units: **{info['unit']}**. {info['color_rule']}"
    )
