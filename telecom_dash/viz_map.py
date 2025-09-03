# telecom_dash/viz_map.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

from shapely import wkt
from shapely.geometry import mapping
from branca.element import MacroElement, Template

# ----------------------------
# Helpers
# ----------------------------

def _format_metric(metric: str, v: float | int | None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if metric in ("avg_d_kbps", "avg_u_kbps"):
        return f"{float(v)/1000.0:.1f} Mbps"
    if metric == "avg_lat_ms":
        return f"{float(v):.1f} ms"
    return f"{v:.0f}"

def _color_for_value(v: float | None, qtiles: dict | None, is_latency: bool) -> str:
    if v is None or qtiles is None:
        return "#cccccc"
    q10, q50, q90 = qtiles.get("q10"), qtiles.get("q50"), qtiles.get("q90")
    if any(pd.isna(x) for x in (q10, q50, q90)):
        return "#cccccc"
    if is_latency:
        return "#2ca25f" if v <= q10 else ("#99d8c9" if v <= q50 else "#e5f5f9")
    else:
        return "#2ca25f" if v >= q90 else ("#99d8c9" if v >= q50 else "#e5f5f9")

def _legend_html(q: dict, label: str, invert: bool = False, unit_hint: str = "", metric: str = "") -> str:
    def conv(x: float) -> float:
        if metric in ("avg_d_kbps", "avg_u_kbps"):
            return x / 1000.0
        return x
    q10, q50, q90 = q.get("q10"), q.get("q50"), q.get("q90")
    if any(pd.isna([q10, q50, q90])):
        return ""
    if invert:
        lines = [
            f"<b>{label}</b> {unit_hint} (lower is better)",
            f"<span style='background:#2ca25f'>&nbsp;&nbsp;&nbsp;</span> ≤ {conv(q10):.1f}",
            f"<span style='background:#99d8c9'>&nbsp;&nbsp;&nbsp;</span> {conv(q10):.1f}–{conv(q50):.1f}",
            f"<span style='background:#e5f5f9'>&nbsp;&nbsp;&nbsp;</span> &gt; {conv(q50):.1f}",
        ]
    else:
        lines = [
            f"<b>{label}</b> {unit_hint} (higher is better)",
            f"<span style='background:#e5f5f9'>&nbsp;&nbsp;&nbsp;</span> &lt; {conv(q50):.1f}",
            f"<span style='background:#99d8c9'>&nbsp;&nbsp;&nbsp;</span> {conv(q50):.1f}–{conv(q90):.1f}",
            f"<span style='background:#2ca25f'>&nbsp;&nbsp;&nbsp;</span> ≥ {conv(q90):.1f}",
        ]
    return "<br>".join(lines)

def _add_fixed_box(fmap, inner_html: str, *, left: int = 20, bottom: int = 20, max_width: int | None = None):
    if not inner_html:
        return
    mw = f"max-width:{max_width}px;" if max_width else ""
    template = f"""
    {{% macro html(this, kwargs) %}}
    <div style="
        position:absolute; bottom:{bottom}px; left:{left}px; z-index:9999;
        background:white; padding:8px; border:1px solid #ccc; border-radius:6px;
        font-size:12px; {mw}">
        {inner_html}
    </div>
    {{% endmacro %}}
    """
    box = MacroElement()
    box._template = Template(template)
    fmap.get_root().add_child(box)

def _get_center(tf: pd.DataFrame | None, towers: pd.DataFrame | None):
    if tf is not None and len(tf) and {"tile_y", "tile_x"}.issubset(tf.columns):
        return [float(tf["tile_y"].mean()), float(tf["tile_x"].mean())]
    if towers is not None and len(towers):
        return [float(towers["lat"].mean()), float(towers["lon"].mean())]
    return [51.5074, -0.1278]

# ---------- cluster ranking & ramp coloring ----------

def _ramp_palette(n: int) -> list[str]:
    """Green→yellow→red ramp (best→worst). Samples evenly from an 8-step ColorBrewer RdYlGn reversed."""
    base = ["#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#fee08b", "#fdae61", "#f46d43", "#d73027"]  # green→red
    if n <= 1:
        return [base[0]]
    idx = np.linspace(0, len(base) - 1, n)
    idx = np.clip(np.round(idx).astype(int), 0, len(base) - 1)
    # ensure strictly non-decreasing unique selection
    out, last = [], -1
    for i in idx:
        if i == last and out:
            i = min(i + 1, len(base) - 1)
        out.append(base[i])
        last = i
    return out

def _rank_clusters(stats: pd.DataFrame | None, uniq_ids: list[int]) -> tuple[list[int], dict[int, float]]:
    """
    Compute a simple quality score per cluster:
      score = mean( z01(DL) , z01(UL) , 1 - z01(Lat) ) on available columns
      where z01(x) = (x - min)/(max - min + eps).
    Returns (ordered_ids_best_to_worst, score_map).
    """
    if stats is None or stats.empty:
        return uniq_ids, {int(c): 0.0 for c in uniq_ids}

    df = stats.copy()
    score = None
    eps = 1e-9

    def z01(col, higher_is_better=True):
        x = df[col].astype(float)
        rng = x.max() - x.min()
        z = (x - x.min()) / (rng + eps)
        return z if higher_is_better else (1.0 - z)

    comps = []
    if "avg_d_kbps" in df.columns:
        comps.append(z01("avg_d_kbps", True))
    if "avg_u_kbps" in df.columns:
        comps.append(z01("avg_u_kbps", True))
    if "avg_lat_ms" in df.columns:
        comps.append(z01("avg_lat_ms", False))  # lower latency is better

    if comps:
        score = pd.concat(comps, axis=1).mean(axis=1)
    else:
        score = pd.Series(0.0, index=df.index)

    order = score.sort_values(ascending=False).index.tolist()  # best first
    score_map = {int(k): float(v) for k, v in score.items()}
    # keep only clusters we actually have in tf
    order = [int(c) for c in order if int(c) in set(uniq_ids)]
    # append any missing ids (shouldn't happen)
    for c in uniq_ids:
        if c not in order:
            order.append(int(c))
    return order, score_map

# ----------------------------
# Main entry
# ----------------------------

def build_map(
    tf: pd.DataFrame | None,
    towers: pd.DataFrame | None,
    base: str,
    color_mode: str,
    metric: str,
    qtiles: dict | None,
    show_tiles: bool,
    show_centroids: bool,
    show_towers: bool,
    use_cluster: bool,
    show_heatmap: bool,
    max_towers_points: int,
    heat_radius: int,
    cluster_info: dict | None = None,
) -> None:
    """Render Folium map with robust legends. Cluster mode is best→worst with green→red ramp."""
    center = _get_center(tf, towers)
    fmap = folium.Map(location=center, zoom_start=11, control_scale=True, tiles=None)

    # Basemaps
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap', show=(base == "OpenStreetMap")).add_to(fmap)
    folium.TileLayer('CartoDB positron', name='CartoDB Positron', show=(base == "CartoDB Positron")).add_to(fmap)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark Matter', show=(base == "CartoDB Dark Matter")).add_to(fmap)

    legend_added = False
    colors = {}
    invert = False
    unit_hint = ""

    # ---------------- Tiles layer ----------------
    if show_tiles and tf is not None and len(tf):
        if color_mode == "Performance metric":
            label = metric
            is_latency = (metric == "avg_lat_ms")
            unit_hint = "(Mbps)" if metric in ("avg_d_kbps", "avg_u_kbps") else "(ms)"
            get_val = lambda r: float(r[metric]) if pd.notna(r.get(metric, np.nan)) else None
            if not qtiles:
                try:
                    q10, q50, q90 = tf[metric].quantile([0.10, 0.50, 0.90]).tolist()
                    qtiles = {"q10": q10, "q50": q50, "q90": q90}
                except Exception:
                    qtiles = None
            invert = is_latency

        elif color_mode == "Performance cluster" and ("cluster" in tf.columns):
            label = "cluster"
            is_latency = False
            get_val = lambda r: int(r["cluster"]) if pd.notna(r.get("cluster", np.nan)) else None

            uniq = sorted(int(c) for c in pd.Series(tf["cluster"]).dropna().unique())

            # Rank clusters by quality (best→worst) and assign ramp colors
            stats = (cluster_info or {}).get("stats")
            order, score_map = _rank_clusters(stats, uniq)
            ramp = _ramp_palette(len(order))  # green→red
            colors = {int(c): ramp[i] for i, c in enumerate(order)}

        else:
            # Tower density
            label = "tower_count"
            is_latency = False
            get_val = lambda r: float(r["tower_count"]) if pd.notna(r.get("tower_count", np.nan)) else None
            if "tower_count" in tf.columns:
                try:
                    q10, q50, q90 = tf["tower_count"].quantile([0.10, 0.50, 0.90]).tolist()
                    qtiles = {"q10": q10, "q50": q50, "q90": q90}
                except Exception:
                    qtiles = None

        features = []
        for _, r in tf.iterrows():
            try:
                geom = wkt.loads(r["tile"])
            except Exception:
                continue
            val = get_val(r)
            col = colors.get(val, "#cccccc") if color_mode == "Performance cluster" \
                  else _color_for_value(val, qtiles, is_latency)
            props = {
                "label": label,
                "value": val,
                "value_fmt": (_format_metric(metric, val) if label not in ("tower_count", "cluster")
                              else (f"Cluster {val}" if label == "cluster" else f"{int(val) if val is not None else 0}")),
                "tests": int(r.get("tests", 0)),
                "devices": int(r.get("devices", 0)),
                "towers": int(r.get("tower_count", 0)),
                "color": col,
                "is_anom": bool(r.get("is_anom", False)) if color_mode == "Performance metric" else False,
                "anom": "Yes" if (color_mode == "Performance metric" and bool(r.get("is_anom", False))) else "No",
            }
            features.append({"type": "Feature", "geometry": mapping(geom), "properties": props})

        gj = {"type": "FeatureCollection", "features": features}

        def style_fn(feat):
            is_anom = bool(feat["properties"].get("is_anom", False))
            if color_mode == "Performance metric" and is_anom:
                return {"color": "#d7191c", "weight": 2.0, "fillColor": feat["properties"]["color"], "fillOpacity": 0.35}
            else:
                return {"color": "#555555", "weight": 0.5, "fillColor": feat["properties"]["color"], "fillOpacity": 0.70}

        base_fields = ["label", "value_fmt", "towers", "tests", "devices"]
        base_aliases = ["Color by", "Value", "Towers in tile", "Tests", "Devices"]
        if color_mode == "Performance metric":
            fields = base_fields + ["anom"]; aliases = base_aliases + ["Anomaly"]
        else:
            fields, aliases = base_fields, base_aliases

        folium.GeoJson(
            gj,
            name=f"Tiles ({'metric' if color_mode=='Performance metric' else ('cluster' if color_mode=='Performance cluster' else 'density')})",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True, sticky=False),
            highlight_function=lambda f: {"weight": 2.0, "color": "#000"},
        ).add_to(fmap)

        # Legends
        if color_mode == "Performance cluster" and (cluster_info and "stats" in cluster_info):
            stats = cluster_info["stats"]
            # Use the same order and colors as used for tiles
            uniq = sorted(int(c) for c in pd.Series(tf["cluster"]).dropna().unique())
            order, _score_map = _rank_clusters(stats, uniq)
            ramp = _ramp_palette(len(order))
            color_map = {int(c): ramp[i] for i, c in enumerate(order)}

            rows = []
            for i, c in enumerate(order):
                col = color_map[int(c)]
                dl = stats.get("avg_d_kbps", pd.Series([np.nan])).loc[c] if "avg_d_kbps" in stats else np.nan
                ul = stats.get("avg_u_kbps", pd.Series([np.nan])).loc[c] if "avg_u_kbps" in stats else np.nan
                lt = stats.get("avg_lat_ms", pd.Series([np.nan])).loc[c] if "avg_lat_ms" in stats else np.nan
                dl_s = f"{dl/1000:.1f} Mbps" if pd.notna(dl) else "-"
                ul_s = f"{ul/1000:.1f} Mbps" if pd.notna(ul) else "-"
                lt_s = f"{lt:.1f} ms" if pd.notna(lt) else "-"
                rows.append(
                    f"<div style='margin-bottom:4px'>"
                    f"<span style='display:inline-block;width:12px;height:12px;background:{col};"
                    f"margin-right:6px;border:1px solid #999'></span>"
                    f"<b>Cluster {int(c)}</b>: DL {dl_s} • UL {ul_s} • Lat {lt_s}</div>"
                )
            inner = "<b>Performance clusters</b> <span style='color:#666'>(best → worst)</span><br>" + "".join(rows)
            _add_fixed_box(fmap, inner_html=inner, left=20, bottom=20, max_width=380)
            legend_added = True

        else:
            inner = _legend_html(
                qtiles or {}, label,
                invert=(metric == "avg_lat_ms" and color_mode == "Performance metric"),
                unit_hint=unit_hint, metric=metric
            )
            if inner:
                _add_fixed_box(fmap, inner_html=inner, left=20, bottom=20)
                legend_added = True

    # ---------------- Centroids (optional) ----------------
    if show_centroids and tf is not None and len(tf):
        if color_mode == "Performance metric" and metric in tf.columns:
            try:
                q10, q50, q90 = tf[metric].quantile([0.10, 0.50, 0.90]).tolist()
                cq = {"q10": q10, "q50": q50, "q90": q90}
            except Exception:
                cq = None
            is_latency = (metric == "avg_lat_ms")
            get_val = lambda r: float(r[metric]) if pd.notna(r.get(metric, np.nan)) else None
        elif color_mode == "Tower density" and "tower_count" in tf.columns:
            try:
                q10, q50, q90 = tf["tower_count"].quantile([0.10, 0.50, 0.90]).tolist()
                cq = {"q10": q10, "q50": q50, "q90": q90}
            except Exception:
                cq = None
            is_latency = False
            get_val = lambda r: float(r["tower_count"]) if pd.notna(r.get("tower_count", np.nan)) else None
        else:
            cq = None; is_latency = False; get_val = lambda r: None

        for _, r in tf.iterrows():
            val = get_val(r)
            col = "#666666" if color_mode == "Performance cluster" else _color_for_value(val, cq, is_latency)
            try:
                folium.CircleMarker(
                    location=[r["tile_y"], r["tile_x"]],
                    radius=3, color=col, fill=True, fill_color=col, fill_opacity=0.85, weight=0
                ).add_to(fmap)
            except Exception:
                continue

    # ---------------- Towers layer ----------------
    if show_towers and towers is not None and len(towers):
        if show_heatmap:
            pts = towers[["lat", "lon"]].dropna().values.tolist()
            if len(pts) > 10000:
                pts = pd.DataFrame(pts).sample(10000, random_state=0).values.tolist()
            HeatMap(pts, radius=int(heat_radius), blur=int(heat_radius)).add_to(fmap)
        else:
            tw = towers.sample(min(len(towers), int(max_towers_points)), random_state=0)
            tgt = MarkerCluster(name="Towers (clustered)").add_to(fmap) if use_cluster else fmap
            for _, r in tw.iterrows():
                try:
                    folium.Marker(
                        location=[r["lat"], r["lon"]],
                        icon=folium.Icon(icon="signal", prefix="fa", color="gray"),
                    ).add_to(tgt)
                except Exception:
                    continue

    folium.LayerControl(collapsed=True).add_to(fmap)

   

    st_folium(fmap, width=1100, height=600)
