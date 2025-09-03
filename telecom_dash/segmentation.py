# telecom_dash/segmentation.py
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _validate_features(tf: pd.DataFrame, features: Iterable[str]) -> Tuple[pd.DataFrame, list[str]]:
    feats = [f for f in features if f in tf.columns]
    if not feats:
        raise ValueError("No valid features found in dataframe for clustering.")
    X = tf[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return X, feats


def cluster_palette(k: int) -> list[str]:
    """
    Legacy palette (not used for ranking; viz_map now uses green→red ramp after ranking).
    Kept for compatibility or external use.
    """
    base = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a",
            "#66a61e", "#e6ab02", "#a6761d", "#666666",
            "#1f78b4", "#b2df8a", "#fb9a99", "#cab2d6"]
    if k <= len(base):
        return base[:k]
    # repeat cyclically if k is larger
    out = []
    for i in range(k):
        out.append(base[i % len(base)])
    return out


def _cluster_stats(tf_with_labels: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Compute per-cluster summary:
      - mean of requested KPI features (kbps/ms as in source)
      - median of tower_count if available
      - tiles count
    Index is cluster label (int).
    """
    cols = list(features)
    extra = []
    if "tower_count" in tf_with_labels.columns:
        extra.append("tower_count")
    use_cols = ["cluster"] + cols + extra
    d = tf_with_labels.dropna(subset=["cluster"])[use_cols].copy()
    if d.empty:
        return pd.DataFrame()

    agg = {c: "mean" for c in cols}
    if "tower_count" in d.columns:
        agg["tower_count"] = "median"
    agg["cluster"] = "count"  # temporary to get counts

    g = d.groupby("cluster", dropna=True).agg(agg).rename(columns={"cluster": "tiles"})
    # ensure integer index
    g.index = g.index.astype(int)
    return g


def compute_clusters(
    tf: pd.DataFrame,
    features: Iterable[str],
    *,
    k: int = 4,
    random_state: int = 0,
) -> tuple[pd.DataFrame, Dict]:
    """
    Run k-means on selected features and return (tf_with_labels, info).
      - Features should be numeric columns; NaNs are dropped for training.
      - Rows with any NaN among features get cluster = NaN (not assigned).
      - info: {'features', 'k', 'model', 'scaler', 'stats'}
    """
    if tf is None or tf.empty:
        return tf, {}

    X, feats = _validate_features(tf, features)

    # rows eligible for training
    mask_train = X.notna().all(axis=1)
    if mask_train.sum() < k:
        # not enough points to form k clusters → assign NaN
        out = tf.copy()
        out["cluster"] = np.nan
        return out, {"features": feats, "k": k, "stats": pd.DataFrame()}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.loc[mask_train, feats])

    km = KMeans(n_clusters=int(k), n_init=10, random_state=random_state)
    labels = km.fit_predict(Xs)

    out = tf.copy()
    out["cluster"] = np.nan
    out.loc[mask_train, "cluster"] = labels.astype(int)

    stats = _cluster_stats(out, feats)

    info = {
        "features": feats,
        "k": int(k),
        "model": km,
        "scaler": scaler,
        "stats": stats,  # used by viz_map + cluster_panels
    }
    return out, info
