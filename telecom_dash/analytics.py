# telecom_dash/analytics.py
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_quantiles(s: pd.Series, qs=(0.10, 0.50, 0.90)) -> Optional[Dict[str, float]]:
    """Return dict of q10/q50/q90 or None if not computable."""
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    try:
        q10, q50, q90 = s.quantile(qs).tolist()
        return {"q10": float(q10), "q50": float(q50), "q90": float(q90)}
    except Exception:
        return None


def compute_anomalies(
    tf: Optional[pd.DataFrame],
    metric: str,
    color_mode: str,
    enable_anom: bool,
    z_thresh: float,
    only_anom: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[int], Optional[Dict[str, float]]]:
    """
    - Adds anomaly flags (Performance metric mode only).
    - Returns (tf_out, anom_count, qtiles) where qtiles holds q10/q50/q90 for the active
      coloring dimension (metric or tower_count).
    """
    if tf is None or len(tf) == 0:
        return tf, None, None

    tf = tf.copy()

    # Which column defines coloring & quantiles?
    if color_mode == "Performance metric" and metric in tf.columns:
        col = metric
    elif color_mode == "Tower density" and "tower_count" in tf.columns:
        col = "tower_count"
    else:
        col = None  # Performance cluster: legend handled elsewhere

    # Quantiles used by the map legend
    qtiles = _safe_quantiles(tf[col]) if col is not None else None

    anom_count = 0
    if color_mode == "Performance metric" and metric in tf.columns:
        # z-score based anomalies (lower is worse; for latency flip sign)
        s = pd.to_numeric(tf[metric], errors="coerce")
        m = float(s.mean())
        std = float(s.std(ddof=0)) or 1.0
        z = (s - m) / std
        if metric == "avg_lat_ms":
            z = -z  # high latency = bad → flip so "low" z means anomalously bad
        tf["zscore"] = z
        if enable_anom:
            tf["is_anom"] = tf["zscore"] <= float(z_thresh)
        else:
            tf["is_anom"] = False

        anom_count = int(tf["is_anom"].sum())
        if only_anom:
            tf = tf[tf["is_anom"]].copy()
    else:
        # Not in performance mode → clear anomaly flags to avoid stale state
        tf["is_anom"] = False

    return tf, anom_count, qtiles


def corr_note_for(tf: Optional[pd.DataFrame], metric: str) -> str:
    """
    Short note showing correlation between selected metric and tower density.
    Uses Pearson and Spearman on rows with finite values.
    """
    if tf is None or len(tf) == 0:
        return ""

    cols = []
    if metric in tf.columns:
        cols.append(metric)
    if "tower_count" in tf.columns:
        cols.append("tower_count")
    if set(cols) != {metric, "tower_count"}:
        return ""

    d = tf[[metric, "tower_count"]].apply(pd.to_numeric, errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return ""

    try:
        pear = float(d[metric].corr(d["tower_count"], method="pearson"))
    except Exception:
        pear = float("nan")
    try:
        spear = float(d[metric].corr(d["tower_count"], method="spearman"))
    except Exception:
        spear = float("nan")

    n = len(d)
    def _fmt(x: float) -> str:
        return "-" if (x is None or math.isnan(x)) else f"{x:.3f}"

    return f"Pearson r = {_fmt(pear)} • Spearman ρ = {_fmt(spear)} • n = {n:,}"
