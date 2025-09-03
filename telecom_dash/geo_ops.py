# telecom_dash/geo_ops.py  (replace the existing helpers + attach_tower_density body)

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from shapely import wkt
from shapely.geometry import Point
from shapely.strtree import STRtree

# ---------- cached helpers (hash-safe & tile-set aware) ----------

def _fingerprint_ids(ids) -> str:
    """Stable md5 of the tile id sequence to key the cache."""
    h = hashlib.md5()
    for i in ids:
        h.update(str(int(i)).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()

@st.cache_resource
def _make_tree(_geoms_tuple: tuple, key: str):
    """
    Build and cache an STRtree for a tuple of shapely geometries.
    'key' participates in the cache hash; '_geoms_tuple' is ignored (unhashable).
    """
    geoms = list(_geoms_tuple)
    return STRtree(geoms)

def _bounds_filter(df: pd.DataFrame, bounds):
    if df is None or df.empty:
        return df
    lat_col = "lat" if "lat" in df.columns else None
    lon_col = "lon" if "lon" in df.columns else None
    if lat_col is None or lon_col is None:
        return df
    try:
        lat_min = getattr(bounds, "LAT_MIN", None) or getattr(bounds, "lat_min", None) or bounds["lat_min"]
        lat_max = getattr(bounds, "LAT_MAX", None) or getattr(bounds, "lat_max", None) or bounds["lat_max"]
        lon_min = getattr(bounds, "LON_MIN", None) or getattr(bounds, "lon_min", None) or bounds["lon_min"]
        lon_max = getattr(bounds, "LON_MAX", None) or getattr(bounds, "lon_max", None) or bounds["lon_max"]
        df = df[df[lat_col].between(lat_min, lat_max) & df[lon_col].between(lon_min, lon_max)]
    except Exception:
        return df
    return df

# ---------- main API ----------

def attach_tower_density(
    tf: pd.DataFrame | None,
    towers: pd.DataFrame | None,
    *,
    max_tiles_for_density: int | None = None,   # kept for API compatibility; ignored
    bounds=None,
) -> pd.DataFrame | None:
    if tf is None or towers is None or tf.empty or towers.empty or "tile" not in tf.columns:
        if tf is not None:
            tf = tf.copy()
            tf["tower_count"] = 0
        return tf

    # Collect polygons
    poly_records = []
    for idx, r in tf.iterrows():
        try:
            geom = wkt.loads(r["tile"])
            if geom is not None:
                poly_records.append((idx, geom))
        except Exception:
            continue

    if not poly_records:
        out = tf.copy()
        out["tower_count"] = 0
        return out

    tile_ids, geoms = zip(*poly_records)
    geoms = list(geoms)

    # Build cached STRtree keyed by the tile-id fingerprint
    tree_key = _fingerprint_ids(tile_ids)
    tree = _make_tree(tuple(geoms), key=tree_key)

    # Shapely 1.x support (map geometry ids to positions)
    geom_id_to_pos = {id(g): i for i, g in enumerate(geoms)}
    pos_to_tileid = dict(enumerate(tile_ids))
    n_geoms = len(geoms)

    # Clean towers
    tw = towers.copy()
    tw = _bounds_filter(tw, bounds)
    if "lat" not in tw.columns or "lon" not in tw.columns:
        tw = tw.rename(columns={"Latitude": "lat", "LAT": "lat", "latitude": "lat",
                                "Longitude": "lon", "LON": "lon", "longitude": "lon"})
    tw["lat"] = pd.to_numeric(tw.get("lat"), errors="coerce")
    tw["lon"] = pd.to_numeric(tw.get("lon"), errors="coerce")
    tw = tw.dropna(subset=["lat", "lon"])
    if tw.empty:
        out = tf.copy()
        out["tower_count"] = 0
        return out

    coords = tw[["lon", "lat"]].to_numpy()
    counts = {tid: 0 for tid in tile_ids}

    # Count towers
    for x, y in coords:
        p = Point(float(x), float(y))
        hits = tree.query(p)  # 2.x → ndarray of idx; 1.x → list of geoms

        if hasattr(hits, "dtype"):  # indices (Shapely 2.x)
            cand_pos = [int(k) for k in np.atleast_1d(hits).tolist()]
        else:  # geometries (Shapely 1.x)
            cand_pos = [geom_id_to_pos.get(id(g), -1) for g in hits]

        for pos in cand_pos:
            if pos < 0 or pos >= n_geoms:  # extra guard against stale/mismatched trees
                continue
            poly = geoms[pos]
            if poly.covers(p):
                counts[pos_to_tileid[pos]] += 1
                break

    out = tf.copy()
    out["tower_count"] = out.index.map(pd.Series(counts)).fillna(0).astype(int)
    return out
