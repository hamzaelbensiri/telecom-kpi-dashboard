# telecom_dash/io_data.py
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from .settings import Bounds


# ----------------------------
# General helpers
# ----------------------------

def _clip_to_bounds(df: pd.DataFrame, bounds: Bounds) -> pd.DataFrame:
    """Return df filtered to lat/lon within bounds if columns exist; otherwise df unchanged."""
    lat_col = "lat" if "lat" in df.columns else None
    lon_col = "lon" if "lon" in df.columns else None
    if lat_col is None or lon_col is None:
        return df
    m = df[lat_col].between(bounds.lat_min, bounds.lat_max) & df[lon_col].between(bounds.lon_min, bounds.lon_max)
    return df.loc[m].copy()


def _normalize_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common variants to lat/lon and coerce to numeric ranges."""
    rename = {
        "Latitude": "lat", "LAT": "lat", "latitude": "lat",
        "Longitude": "lon", "LON": "lon", "longitude": "lon",
    }
    if any(c in df.columns for c in rename):
        df = df.rename(columns=rename)

    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Basic sanity
    if "lat" in df.columns and "lon" in df.columns:
        df = df.dropna(subset=["lat", "lon"])
        df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    return df


# ----------------------------
# Public API (cached)
# ----------------------------

@st.cache_data(show_spinner=False)
def load_tiles(path: str) -> Optional[pd.DataFrame]:
    """
    Load Ookla tiles parquet (if present). Returns None if file missing.
    Expected columns include:
      - 'tile' (WKT polygon), 'tile_x', 'tile_y', 'quadkey'
      - 'avg_d_kbps', 'avg_u_kbps', 'avg_lat_ms', 'tests', 'devices'
    """
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        # fallback if engine issues
        df = pd.read_parquet(path, engine="pyarrow")

    # Ensure numeric KPIs (guard against unexpected dtypes)
    for c in ("avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests", "devices"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # tile_x/tile_y sometimes come as object; coerce to float
    for c in ("tile_x", "tile_y"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_towers(path: str, bounds: Bounds) -> Optional[pd.DataFrame]:
    """
    Load OpenCelliD (or similar) CSV and return cleaned lat/lon within bounds.
    Returns None if file missing.
    """
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df = _normalize_latlon(df)

    if "lat" not in df.columns or "lon" not in df.columns:
        # No usable coordinates
        return pd.DataFrame(columns=["lat", "lon", "radio"])

    # clip to London bbox to avoid strays
    df = _clip_to_bounds(df, bounds)

    # Keep a small, friendly subset of columns if present
    keep = [c for c in ["lat", "lon", "radio", "mcc", "mnc", "lac", "cellid"] if c in df.columns]
    if keep:
        df = df[keep]

    return df.reset_index(drop=True)
