# telecom_dash/viz_panels.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def _format_metric(metric: str, v: float | int | None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if metric in ("avg_d_kbps", "avg_u_kbps"):
        return f"{float(v)/1000.0:.1f} Mbps"
    if metric == "avg_lat_ms":
        return f"{float(v):.1f} ms"
    return f"{v:.0f}"


def kpi_header(tiles: pd.DataFrame | None) -> None:
    """Top-of-page summary KPIs (medians)."""
    c1, c2, c3, c4 = st.columns(4)
    if tiles is None or len(tiles) == 0:
        c1.metric("Tiles (all)", "0")
        c2.metric("Median DL (Mbps)", "-")
        c3.metric("Median UL (Mbps)", "-")
        c4.metric("Median Latency (ms)", "-")
        return

    dl = float(pd.to_numeric(tiles.get("avg_d_kbps"), errors="coerce").median() or np.nan)
    ul = float(pd.to_numeric(tiles.get("avg_u_kbps"), errors="coerce").median() or np.nan)
    lt = float(pd.to_numeric(tiles.get("avg_lat_ms"), errors="coerce").median() or np.nan)

    c1.metric("Tiles (all)", f"{len(tiles):,}")
    c2.metric("Median DL (Mbps)", f"{dl/1000.0:,.1f}" if np.isfinite(dl) else "-")
    c3.metric("Median UL (Mbps)", f"{ul/1000.0:,.1f}" if np.isfinite(ul) else "-")
    c4.metric("Median Latency (ms)", f"{lt:,.1f}" if np.isfinite(lt) else "-")


def export_block(tf: pd.DataFrame | None, color_mode: str, metric: str, export_count: int) -> None:
    """Export current tiles as CSV (bottom/top depending on mode & metric)."""
    if tf is None or len(tf) == 0:
        st.info("Adjust filters to enable export.")
        return

    df = tf.copy()
    if "tower_count" not in df.columns:
        df["tower_count"] = 0

    # Determine sort
    if color_mode == "Performance metric":
        is_latency = (metric == "avg_lat_ms")
        sort_col = metric
        # We export "bottom N" per your UI: speeds low first (ascending), latency high first (descending)
        ascending = True if metric in ("avg_d_kbps", "avg_u_kbps") else False
        label = f"bottom {export_count} by {metric}"
    elif color_mode == "Tower density":
        sort_col = "tower_count"
        ascending = False  # most towers first
        label = f"top {export_count} by tower density"
    elif color_mode == "Performance cluster" and "cluster" in df.columns:
        # Export worst-looking clusters first by simple heuristic (higher latency, lower DL/UL).
        # As a simple approach: sort by cluster then by metric where applicable.
        sort_col = ["cluster", metric] if metric in df.columns else ["cluster"]
        ascending = [True, True] if metric in ("avg_d_kbps", "avg_u_kbps") else [True, False]
        label = f"{export_count} tiles by cluster then {metric}"
    else:
        # Fallback
        sort_col = metric if metric in df.columns else "tower_count"
        ascending = True
        label = f"{export_count} tiles (generic sort)"

    try:
        export_df = df.sort_values(sort_col, ascending=ascending).head(export_count).copy()
    except Exception:
        # In case of mixed types or missing sort column, fallback to index order
        export_df = df.head(export_count).copy()

    # Add metric fields
    export_df["metric_name"] = metric
    export_df["metric_value"] = export_df.get(metric, np.nan)
    export_df["metric_display"] = export_df["metric_value"].apply(
        lambda x: (x/1000.0 if pd.notna(x) else np.nan) if metric in ("avg_d_kbps", "avg_u_kbps")
        else (x if pd.notna(x) else np.nan)
    )
    export_df["metric_unit"] = "Mbps" if metric in ("avg_d_kbps", "avg_u_kbps") else ("ms" if metric == "avg_lat_ms" else "")

    # Columns to export (only keep those that exist)
    preferred_cols = ["tile_x","tile_y","quadkey","tests","devices","tower_count",
                      "metric_name","metric_value","metric_display","metric_unit"]
    cols = [c for c in preferred_cols if c in export_df.columns]
    csv_bytes = export_df[cols].to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"⬇️ Download {label}",
        data=csv_bytes,
        file_name=f"tiles_{color_mode.replace(' ','_').lower()}_{len(export_df)}.csv",
        mime="text/csv"
    )
