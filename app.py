# app.py
# Telecom KPI Dashboard — London (Real Open Data)
# Streamlit shell that wires together data IO, analytics, geo, and viz modules.

import streamlit as st

from telecom_dash.io_data import load_tiles, load_towers
from telecom_dash.settings import Settings
from telecom_dash.analytics import compute_anomalies, corr_note_for
from telecom_dash.geo_ops import attach_tower_density
from telecom_dash.viz_map import build_map
from telecom_dash.viz_panels import kpi_header, export_block
from telecom_dash.docs import glossary_popover, methodology_expander, explain_metric
from telecom_dash import eda
from telecom_dash.segmentation import compute_clusters, cluster_palette
from telecom_dash.cluster_panels import cluster_summary
from telecom_dash.targets import targets_panel,  LABEL

# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="Telecom KPI Dashboard — London (Real Open Data)", layout="wide")
S = Settings()

# ----------------------------
# Load data (cached inside helpers)
# ----------------------------
tiles = load_tiles(S.tiles_path)
towers = load_towers(S.towers_path, S.bounds)

st.title("📶 Telecom KPI Dashboard — London (Real Open Data)")
glossary_popover()
methodology_expander()  # <-- methodology/caveats in sidebar

st.caption("Tiles: Ookla Open Data (Q4 2024). Towers: OpenCelliD (UK export filtered to London). Non-commercial terms apply.")
if towers is not None:
    st.caption(f"Towers loaded (after cleaning): {len(towers):,}")

# ----------------------------
# Sidebar controls
# ----------------------------
metrics_list = list(S.metrics)  # ensure indexable
default_metric_idx = metrics_list.index("avg_d_kbps") if "avg_d_kbps" in metrics_list else 0

metric = st.sidebar.selectbox("Tile metric (for Performance mode)", metrics_list, index=default_metric_idx)
color_mode = st.sidebar.radio("Color tiles by", list(S.color_modes), index=0)

# inline help for selected metric
explain_metric(metric)

min_tests = 0
if tiles is not None and "tests" in tiles.columns and len(tiles):
    min_tests = st.sidebar.slider(
        "Min tests per tile",
        0, int(tiles["tests"].max()), 20, step=5,
        help="Filter out low-sample tiles for stable stats & faster drawing."
    )

show_tiles = st.sidebar.checkbox("Show tiles (polygons)", value=True)
show_centroids = st.sidebar.checkbox("Show centroids (fast)", value=False)

show_towers = st.sidebar.checkbox("Show towers (points/heatmap)", value=True)
max_towers_points = st.sidebar.slider("Max tower symbols (points mode)", 200, 5000, 1200, 100)
use_cluster = st.sidebar.checkbox("Cluster tower symbols", value=True)
show_heatmap = st.sidebar.checkbox("Show towers heatmap", value=False)
heat_radius = st.sidebar.slider("Heatmap radius", 5, 30, 12, 1)

max_tiles = 1500
if tiles is not None and len(tiles):
    max_tiles = st.sidebar.slider("Max tiles to draw", 200, 4000, 1500, 100)

base = st.sidebar.selectbox("Basemap", ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark Matter"])

with st.sidebar.expander("Anomaly detection (Performance mode)"):
    enable_anom = st.checkbox("Highlight low-performance anomalies", value=True)
    z_thresh = st.slider("Z-score threshold (lower = stricter)", -4.0, -0.5, -2.0, 0.1)
    only_anom = st.checkbox("Show only anomalies", value=False, help="Visible only in Performance mode")

with st.sidebar.expander("Segmentation (clusters)", expanded=False):
    enable_seg = st.checkbox("Enable performance clustering", value=False)
    seg_features = st.multiselect(
        "Cluster on features",
        ["avg_d_kbps", "avg_u_kbps", "avg_lat_ms"],
        default=["avg_d_kbps", "avg_u_kbps", "avg_lat_ms"]
    )
    k_clusters = st.slider("Number of clusters (k)", 3, 8, 4, 1)

export_count = st.sidebar.slider("Export: bottom N tiles", 50, 2000, 200, 50)

# ----------------------------
# Data presence guards
# ----------------------------
if tiles is None and towers is None:
    st.warning(
        "No data yet. Run:\n\n"
        "1) `python scripts/download_ookla_london.py`\n"
        "2) Country CSV → `python scripts/filter_opencellid_country_to_london.py`"
    )
    st.stop()

# ----------------------------
# Filter tiles by tests
# ----------------------------
tf = None
if tiles is not None:
    tf = tiles.copy()
    if "tests" in tf.columns:
        tf = tf[tf["tests"] >= min_tests]
    if len(tf) == 0:
        st.info("No tiles match the current filters.")

# ----------------------------
# Tower density (point-in-polygon), anomalies/quantiles
# ----------------------------
tf = attach_tower_density(tf, towers, max_tiles_for_density=max_tiles, bounds=S.bounds) if tf is not None else None

tf, anom_count, qtiles = compute_anomalies(tf, metric, color_mode, enable_anom, z_thresh, only_anom)

# show anomaly count (new)
if anom_count is not None:
    st.caption(f"🔎 Anomalies flagged (current filters): **{anom_count}**")

# ----------------------------
# Segmentation (clusters)
# ----------------------------
cluster_info = None
if tf is not None and enable_seg:
    tf, info = compute_clusters(tf, tuple(seg_features), k=k_clusters, random_state=0)
    if info:
        info["palette"] = cluster_palette(k_clusters)
        cluster_info = info

# If user selected cluster coloring but no clusters available, hint them (new)
if color_mode == "Performance cluster" and cluster_info is None:
    st.info("Enable clustering in the sidebar (and ensure enough rows) to color tiles by performance clusters.")

# ----------------------------
# Map
# ----------------------------
st.subheader("🗺️ Map")
build_map(
    tf=tf,
    towers=towers,
    base=base,
    color_mode=color_mode,
    metric=metric,
    qtiles=qtiles,
    show_tiles=show_tiles,
    show_centroids=show_centroids,
    show_towers=show_towers,
    use_cluster=use_cluster,
    show_heatmap=show_heatmap,
    max_towers_points=max_towers_points,
    heat_radius=heat_radius,
    cluster_info=cluster_info,
)

if color_mode == "Performance cluster" and cluster_info is not None and tf is not None and "cluster" in tf.columns:
    st.markdown("---")
    st.subheader("🧩 Cluster summary")
    cluster_summary(tf, cluster_info)


# ----------------------------
# Summary + Export
# ----------------------------
st.subheader("📈 Summary")
kpi_header(tiles)

if tf is not None:
    note = corr_note_for(tf, metric)
    if note:
        st.caption(note)

st.markdown("---")
export_block(tf, color_mode, metric, export_count)

# ----------------------------
# EDA (only for performance KPIs)
# ----------------------------

if metric in ("avg_d_kbps", "avg_u_kbps", "avg_lat_ms"):
    st.markdown("---")
    st.subheader("🔍 EDA & Diagnostics")
if metric in ("avg_d_kbps", "avg_u_kbps", "avg_lat_ms"):
    with st.expander("Distribution of selected metric", expanded=False):
        eda.hist_metric(tf, metric)

    # One tier-share that adapts to metric
    title_map = {
        "avg_d_kbps": "Performance tiers — Download",
        "avg_u_kbps": "Performance tiers — Upload",
        "avg_lat_ms": "Latency tiers",
    }
    with st.expander(title_map.get(metric, "Performance tiers"), expanded=False):
        eda.tier_share(tf, metric)

    with st.expander("Coverage vs performance (scatter + trend)", expanded=False):
        eda.coverage_vs_perf(tf, metric)


if metric in ("avg_d_kbps", "avg_u_kbps", "avg_lat_ms"):
    st.markdown("---")
    st.subheader("🎯 Targets & compliance")
    with st.expander(f"Set target for {LABEL.get(metric, metric)}", expanded=False):
        targets_panel(tf, metric)