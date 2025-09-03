# telecom_dash/cluster_panels.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


def _ramp_palette(n: int) -> list[str]:
    # Green→yellow→red (best→worst)
    base = ["#1a9850", "#66bd63", "#a6d96a", "#d9ef8b",
            "#fee08b", "#fdae61", "#f46d43", "#d73027"]
    if n <= 1:
        return [base[0]]
    idx = np.linspace(0, len(base) - 1, n)
    idx = np.clip(np.round(idx).astype(int), 0, len(base) - 1)
    out, last = [], -1
    for i in idx:
        if i == last and out:
            i = min(i + 1, len(base) - 1)
        out.append(base[i]); last = i
    return out


def _rank_clusters(stats: pd.DataFrame, cluster_ids: list[int]) -> list[int]:
    """
    Rank best→worst using available columns: DL↑, UL↑, Latency↓.
    Returns ordered list of cluster ids present.
    """
    if stats is None or stats.empty:
        return cluster_ids
    df = stats.copy()
    eps = 1e-9
    comps = []
    if "avg_d_kbps" in df:
        x = df["avg_d_kbps"].astype(float); comps.append((x - x.min())/(x.max() - x.min() + eps))
    if "avg_u_kbps" in df:
        x = df["avg_u_kbps"].astype(float); comps.append((x - x.min())/(x.max() - x.min() + eps))
    if "avg_lat_ms" in df:
        x = df["avg_lat_ms"].astype(float); z = (x - x.min())/(x.max() - x.min() + eps); comps.append(1.0 - z)
    score = pd.concat(comps, axis=1).mean(axis=1) if comps else pd.Series(0.0, index=df.index)
    order = score.sort_values(ascending=False).index.tolist()
    order = [int(c) for c in order if int(c) in set(cluster_ids)]
    for c in cluster_ids:
        if c not in order:
            order.append(int(c))
    return order


def cluster_summary(tf: pd.DataFrame, cluster_info: dict | None) -> None:
    """
    Show cluster counts, KPI means, and export.
    Expects tf['cluster'] and cluster_info['stats'] with means in raw units (kbps/ms).
    """
    if tf is None or "cluster" not in tf.columns or cluster_info is None or "stats" not in cluster_info:
        st.info("No clusters to summarize. Turn on clustering in the sidebar.")
        return

    stats = cluster_info["stats"]
    present = sorted(int(c) for c in pd.Series(tf["cluster"]).dropna().unique())
    if not present:
        st.info("No clusters in the current view.")
        return

    # order & colors (best → worst)
    order = _rank_clusters(stats, present)
    ramp = _ramp_palette(len(order))
    color_map = {int(c): ramp[i] for i, c in enumerate(order)}
    order_labels = [str(c) for c in order]  # string domain for Altair

    # counts & share
    cnt = (tf["cluster"].dropna().astype(int).value_counts().reindex(order, fill_value=0))
    total = int(cnt.sum())
    share = (cnt / max(total, 1) * 100.0).round(1)

    # optional: median tower_count from tf (not in stats)
    med_towers = None
    if "tower_count" in tf.columns:
        med_towers = (
            tf.dropna(subset=["cluster"])
              .assign(cluster=lambda d: d["cluster"].astype(int))
              .groupby("cluster")["tower_count"]
              .median()
        )

    # build table rows
    def _safe(stats, col, c):
        return stats[col].loc[c] if (col in stats.columns and c in stats.index) else np.nan

    rows = []
    for c in order:
        dl = _safe(stats, "avg_d_kbps", c)
        ul = _safe(stats, "avg_u_kbps", c)
        lt = _safe(stats, "avg_lat_ms", c)
        mt = med_towers.loc[c] if med_towers is not None and c in med_towers.index else np.nan
        rows.append({
            "cluster": int(c),
            "tiles": int(cnt.loc[c]),
            "share_%": float(share.loc[c]),
            "DL_Mbps": (dl/1000.0) if pd.notna(dl) else np.nan,
            "UL_Mbps": (ul/1000.0) if pd.notna(ul) else np.nan,
            "Latency_ms": float(lt) if pd.notna(lt) else np.nan,
            "Median_towers": float(mt) if pd.notna(mt) else np.nan,
        })
    tbl = pd.DataFrame(rows)

    # --- chart (use string labels to match domain/sort) ---
    tbl_chart = tbl.copy()
    tbl_chart["cluster_label"] = tbl_chart["cluster"].astype(str)

    chart = (
        alt.Chart(tbl_chart)
        .mark_bar()
        .encode(
            x=alt.X("cluster_label:N", sort=order_labels, title="Cluster (best → worst)"),
            y=alt.Y("tiles:Q", title="Tiles"),
            color=alt.Color(
                "cluster_label:N",
                sort=order_labels,
                scale=alt.Scale(domain=order_labels, range=[color_map[int(c)] for c in order]),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("cluster_label:N", title="Cluster"),
                alt.Tooltip("tiles:Q", title="Tiles"),
                alt.Tooltip("share_%:Q", title="Share (%)", format=".1f"),
                alt.Tooltip("DL_Mbps:Q", title="DL (Mbps)", format=".1f"),
                alt.Tooltip("UL_Mbps:Q", title="UL (Mbps)", format=".1f"),
                alt.Tooltip("Latency_ms:Q", title="Latency (ms)", format=".1f"),
                alt.Tooltip("Median_towers:Q", title="Median towers", format=".0f"),
            ],
        )
        .properties(title="Cluster size (best → worst)")
    )
    st.altair_chart(chart, use_container_width=True)

    # Table (keep cluster as int for readability)
    st.dataframe(
        tbl[["cluster", "tiles", "share_%", "DL_Mbps", "UL_Mbps", "Latency_ms", "Median_towers"]],
        use_container_width=True,
        hide_index=True,
    )

    # Export
    csv = tbl.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export cluster summary (CSV)", data=csv, file_name="cluster_summary.csv", mime="text/csv")
