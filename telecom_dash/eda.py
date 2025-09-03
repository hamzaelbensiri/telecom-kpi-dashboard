# telecom_dash/eda.py
# Lightweight EDA utilities for the Streamlit dashboard (Altair + scikit-learn).
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st




# ---------------------------
# Helpers
# ---------------------------

def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Keep only columns present & numeric."""
    keep = []
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() >= max(5, min(20, int(0.05 * len(s)))):  # at least a few valid points
                keep.append(c)
    return keep

def _nice_metric_label(metric: str) -> str:
    return {
        "avg_d_kbps": "Download (kbps)",
        "avg_u_kbps": "Upload (kbps)",
        "avg_lat_ms": "Latency (ms)",
        "tests": "Tests (count)",
        "devices": "Devices (count)",
        "tower_count": "Tower count",
    }.get(metric, metric)

# --- helper meta so titles/units are consistent everywhere ---
def _metric_meta(metric: str):
    if metric == "avg_d_kbps":
        return dict(kind="thr", axis_label="Download (Mbps)", to_plot=lambda s: pd.to_numeric(s, errors="coerce")/1000.0, fmt=".1f")
    if metric == "avg_u_kbps":
        return dict(kind="thr", axis_label="Upload (Mbps)",   to_plot=lambda s: pd.to_numeric(s, errors="coerce")/1000.0, fmt=".1f")
    if metric == "avg_lat_ms":
        return dict(kind="lat", axis_label="Latency (ms)",    to_plot=lambda s: pd.to_numeric(s, errors="coerce"),        fmt=".1f")
    return None


# Histogram (distribution)

def hist_metric(df: pd.DataFrame, metric: str) -> None:
    """Numeric-axis histogram that adapts to the selected KPI (DL/UL in Mbps, Latency in ms)."""
    meta = _metric_meta(metric)
    if meta is None:     # non-performance KPI
        return
    if df is None or df.empty or metric not in df:
        st.info("No data for selected metric.")
        return

    s = meta["to_plot"](df[metric]).dropna()
    if s.empty:
        st.info("Selected metric has no numeric values.")
        return

    # anomalies if present
    anom_mask = df.loc[s.index, "is_anom"].astype(bool) if "is_anom" in df.columns else pd.Series(False, index=s.index)

    # pre-bin for stable bars
    n = len(s)
    bins = int(np.clip(np.sqrt(n), 8, 40))
    counts_all, edges = np.histogram(s, bins=bins)
    counts_bad, _ = np.histogram(s[anom_mask], bins=edges)

    left, right = edges[:-1], edges[1:]
    hist = pd.DataFrame({"left": left, "right": right, "count_all": counts_all, "count_bad": counts_bad})

    x_enc = alt.X(
        "left:Q",
        bin=alt.Bin(binned=True),
        title=meta["axis_label"],
        axis=alt.Axis(format=meta["fmt"], tickCount=8, labelOverlap=True, labelFlush=True),
    )

    base = (
        alt.Chart(hist).mark_bar(opacity=0.8, color="#4e79a7")
        .encode(
            x=x_enc, x2="right:Q",
            y=alt.Y("count_all:Q", title="Count"),
            tooltip=[
                alt.Tooltip("left:Q",  title=f"{meta['axis_label']} (from)",  format=meta["fmt"]),
                alt.Tooltip("right:Q", title=f"{meta['axis_label']} (to)",    format=meta["fmt"]),
                alt.Tooltip("count_all:Q", title="All"),
                alt.Tooltip("count_bad:Q", title="Anomalies"),
            ],
        )
        .properties(title=f"Distribution of {meta['axis_label']}")
    )

    if (hist["count_bad"] > 0).any():
        bad = alt.Chart(hist).mark_bar(opacity=0.55, color="#e15759").encode(x=x_enc, x2="right:Q", y="count_bad:Q")
        chart = base + bad
    else:
        chart = base

    st.altair_chart(chart, use_container_width=True)



def tier_share(df: pd.DataFrame, metric: str, q=(0.2, 0.5, 0.8)) -> None:
    """
    One tier-share panel that adapts to KPI:
      - Download/Upload: higher is better (red→green), thresholds shown in Mbps
      - Latency:         lower is better (green→red), thresholds in ms
    """
    meta = _metric_meta(metric)
    if meta is None:
        return
    if df is None or df.empty or metric not in df:
        st.info("No data for tiers.")
        return

    s_raw = pd.to_numeric(df[metric], errors="coerce").dropna()
    if s_raw.empty:
        st.info("Metric has no numeric values.")
        return

    # quantiles on raw; display thresholds converted via meta
    q1, q2, q3 = s_raw.quantile(list(q))
    disp = (lambda v: v/1000.0) if meta["axis_label"].endswith("(Mbps)") else (lambda v: v)

    if meta["axis_label"].startswith("Latency"):
        labels = ["Excellent", "Good", "Fair", "Poor"]                    # lower is better
        colors = ["#1a9641", "#a6d96a", "#fdae61", "#d7191c"]
    else:
        labels = ["Poor", "Fair", "Good", "Excellent"]                    # higher is better
        colors = ["#d7191c", "#fdae61", "#a6d96a", "#1a9641"]

    cuts = [-np.inf, q1, q2, q3, np.inf]
    tier = pd.cut(s_raw, bins=cuts, labels=labels, include_lowest=True)
    share = (tier.value_counts(normalize=True).reindex(labels, fill_value=0)*100).reset_index()
    share.columns = ["tier","pct"]

    chart = (
        alt.Chart(share).mark_bar()
        .encode(
            x=alt.X("tier:N", sort=labels, title="Tier"),
            y=alt.Y("pct:Q", title="% of tiles"),
            color=alt.Color("tier:N", sort=labels, scale=alt.Scale(domain=labels, range=colors), legend=None),
            tooltip=["tier", alt.Tooltip("pct:Q", format=".1f", title="% of tiles")],
        )
        .properties(title=f"Tier share — {meta['axis_label']}")
    )
    st.altair_chart(chart, use_container_width=True)

    if meta["axis_label"].startswith("Latency"):
        st.caption(
            f"Tiers for {meta['axis_label']} (lower is better): "
            f"Excellent ≤ {disp(q1):{meta['fmt']}}, Good ≤ {disp(q2):{meta['fmt']}}, "
            f"Fair ≤ {disp(q3):{meta['fmt']}}, Poor > {disp(q3):{meta['fmt']}}."
        )
    else:
        st.caption(
            f"Tiers for {meta['axis_label']} (higher is better): "
            f"Poor < {disp(q1):{meta['fmt']}}, Fair {disp(q1):{meta['fmt']}}–{disp(q2):{meta['fmt']}}, "
            f"Good {disp(q2):{meta['fmt']}}–{disp(q3):{meta['fmt']}}, Excellent ≥ {disp(q3):{meta['fmt']}}."
        )
def coverage_vs_perf(df: pd.DataFrame, metric: str) -> None:
    """Scatter + LOESS + binned medians; y-axis adapts to KPI (Mbps for DL/UL, ms for latency)."""
    meta = _metric_meta(metric)
    if meta is None:
        return
    if df is None or df.empty or metric not in df or "tower_count" not in df:
        st.info("Need tower_count and the selected metric.")
        return

    d = df[["tower_count", metric, "tests", "devices"]].dropna().copy()
    if d.empty:
        st.info("No rows for this view.")
        return

    d["y"] = meta["to_plot"](d[metric])
    d = d.dropna(subset=["y"])

    # clip central 98% to reduce leverage
    xlo, xhi = d["tower_count"].quantile([0.01, 0.99])
    ylo, yhi = d["y"].quantile([0.01, 0.99])
    d = d[(d["tower_count"].between(xlo, xhi)) & (d["y"].between(ylo, yhi))]

    pear = d[["tower_count","y"]].corr(method="pearson").iloc[0,1]
    spear = d[["tower_count","y"]].corr(method="spearman").iloc[0,1]

    nbins = int(np.clip(np.sqrt(len(d)), 8, 20))
    try:
        d["_bin"] = pd.qcut(d["tower_count"], q=nbins, duplicates="drop")
        med = d.groupby("_bin", observed=True).agg(x=("tower_count","median"), y=("y","median")).reset_index(drop=True)
    except Exception:
        med = pd.DataFrame(columns=["x","y"])

    base = alt.Chart(d).encode(
        x=alt.X("tower_count:Q", title="Tower count in tile"),
        y=alt.Y("y:Q", title=meta["axis_label"], axis=alt.Axis(format=meta["fmt"]))
    )
    pts = base.mark_circle(size=16, opacity=0.25, color="#4e79a7").encode(
        tooltip=["tower_count", alt.Tooltip("y:Q", title=meta["axis_label"], format=meta["fmt"]), "tests", "devices"]
    )
    smooth = base.transform_loess("tower_count", "y", bandwidth=0.3).mark_line(color="#555", opacity=0.9)
    med_line = alt.Chart(med).mark_line(point=True, color="#1f77b4").encode(x="x:Q", y="y:Q")

    st.altair_chart(pts + smooth + med_line, use_container_width=True)
    st.caption(f"Pearson r = {pear:.3f} • Spearman ρ = {spear:.3f} • n = {len(d):,} (clipped to central 98%)")
