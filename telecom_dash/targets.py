# telecom_dash/targets.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

GOOD_DIR = {
    "avg_d_kbps": "gte",  # higher is better
    "avg_u_kbps": "gte",
    "avg_lat_ms": "lte",  # lower is better
}

LABEL = {
    "avg_d_kbps": "Download",
    "avg_u_kbps": "Upload",
    "avg_lat_ms": "Latency",
}

UNIT = {
    "avg_d_kbps": "Mbps",
    "avg_u_kbps": "Mbps",
    "avg_lat_ms": "ms",
}

def _metric_series(tf: pd.DataFrame, metric: str) -> pd.Series:
    s = pd.to_numeric(tf.get(metric), errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan).dropna()

def _display_value(metric: str, v: float) -> float:
    # show DL/UL in Mbps; latency already ms
    if metric in ("avg_d_kbps", "avg_u_kbps"):
        return v / 1000.0
    return v

def _raw_from_display(metric: str, v_display: float) -> float:
    if metric in ("avg_d_kbps", "avg_u_kbps"):
        return v_display * 1000.0
    return v_display

def targets_panel(tf: pd.DataFrame, metric: str) -> None:
    """
    Metric-aware SLA-style panel:
    - choose threshold aligned to metric & units
    - shows % compliant, failing count
    - optional per-cluster compliance chart if tf['cluster'] exists
    - export failing tiles
    """
    if tf is None or tf.empty or metric not in tf.columns:
        st.info("No data for the selected metric.")
        return

    label = LABEL.get(metric, metric)
    unit = UNIT.get(metric, "")
    direction = GOOD_DIR.get(metric, "gte")

    s = _metric_series(tf, metric)
    if s.empty:
        st.info("No valid values for this metric after filters.")
        return

    # sensible slider range from data (in display units)
    lo, hi = float(s.min()), float(s.max())
    lo_d, hi_d = _display_value(metric, lo), _display_value(metric, hi)

    # defaults (portable and realistic)
    default_map = {"avg_d_kbps": 100.0, "avg_u_kbps": 20.0, "avg_lat_ms": 40.0}
    default_display = default_map.get(metric, (lo_d + hi_d) / 2)

    cols = st.columns([1, 2, 1.2, 2])
    with cols[1]:
        thr_display = st.slider(
            f"{label} target ({unit})",
            min_value=round(lo_d, 1),
            max_value=round(max(hi_d, lo_d + 0.1), 1),
            value=float(default_display),
            step=0.5 if unit == "ms" else 1.0,
            help="Move the threshold to test compliance.",
        )
    thr_raw = _raw_from_display(metric, thr_display)

    # compliance mask
    if direction == "gte":
        ok_mask = tf[metric] >= thr_raw
    else:
        ok_mask = tf[metric] <= thr_raw

    n_total = int(ok_mask.notna().sum())
    n_ok = int(ok_mask.sum())
    n_bad = int((~ok_mask).sum())
    pct_ok = (n_ok / n_total * 100.0) if n_total else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Tiles passing", f"{pct_ok:,.1f} %", help="Share of tiles meeting the target.")
    m2.metric("Failing tiles", f"{n_bad:,}")
    median_display = _display_value(metric, float(s.median()))
    m3.metric(f"Median {label}", f"{median_display:,.1f} {unit}")

    # quick pass/fail bar
    bar_df = pd.DataFrame({
        "status": ["Pass", "Fail"],
        "count": [n_ok, n_bad],
    })
    color_range = ["#2ca25f", "#d73027"]
    st.altair_chart(
        alt.Chart(bar_df).mark_bar().encode(
            x=alt.X("status:N", title=""),
            y=alt.Y("count:Q", title="Tiles"),
            color=alt.Color("status:N", scale=alt.Scale(domain=["Pass","Fail"], range=color_range), legend=None),
            tooltip=["status:N", alt.Tooltip("count:Q", format=",.0f")]
        ).properties(height=160),
        use_container_width=True
    )

    # optional: per-cluster compliance
    if "cluster" in tf.columns:
        df_cl = (
            tf.loc[ok_mask.notna(), ["cluster", metric]]
              .assign(ok=ok_mask[ok_mask.notna()].astype(bool),
                      cluster=lambda d: d["cluster"].astype("Int64"))
              .dropna(subset=["cluster"])
        )
        if not df_cl.empty:
            agg = df_cl.groupby(["cluster", "ok"]).size().rename("n").reset_index()
            # ensure both pass/fail rows exist per cluster
            full = []
            for c in sorted(df_cl["cluster"].dropna().unique()):
                n_c = agg.loc[agg["cluster"] == c, "n"].sum()
                n_ok_c = int(agg[(agg["cluster"] == c) & (agg["ok"] == True)]["n"].sum())
                full.append({"cluster": str(int(c)), "status": "Pass", "count": n_ok_c})
                full.append({"cluster": str(int(c)), "status": "Fail", "count": int(n_c - n_ok_c)})
            comp = pd.DataFrame(full)
            st.altair_chart(
                alt.Chart(comp).mark_bar().encode(
                    x=alt.X("cluster:N", title="Cluster"),
                    y=alt.Y("count:Q", stack="normalize", title="Share"),
                    color=alt.Color("status:N",
                                    scale=alt.Scale(domain=["Pass","Fail"], range=color_range),
                                    legend=alt.Legend(title="Status")),
                    tooltip=["cluster:N","status:N", alt.Tooltip("count:Q", format=",.0f")]
                ).properties(title="Compliance by cluster (share)", height=200),
                use_container_width=True
            )

    # export failing tiles
    fail_df = (
        tf.loc[ok_mask.notna() & (~ok_mask)]
          .assign(threshold_raw=thr_raw,
                  threshold_display=thr_display,
                  unit=unit,
                  direction=direction,
                  metric_name=metric,
                  metric_display=lambda d: d[metric].apply(lambda x: _display_value(metric, x)))
    )
    export_cols = ["tile_x","tile_y","quadkey","tests","devices","tower_count",
                   "metric_name","metric_display","unit","direction","threshold_display"]
    csv = fail_df.reindex(columns=[c for c in export_cols if c in fail_df.columns]).to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Export failing tiles ({len(fail_df):,})",
        data=csv,
        file_name=f"failing_tiles_{metric}.csv",
        mime="text/csv"
    )
