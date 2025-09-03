# telecom_dash/settings.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple


@dataclass(frozen=True)
class Bounds:
    """Geographic bounding box (WGS84)."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    # Convenience aliases used elsewhere in the app
    LAT_MIN: float = field(init=False, repr=False)
    LAT_MAX: float = field(init=False, repr=False)
    LON_MIN: float = field(init=False, repr=False)
    LON_MAX: float = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "LAT_MIN", self.lat_min)
        object.__setattr__(self, "LAT_MAX", self.lat_max)
        object.__setattr__(self, "LON_MIN", self.lon_min)
        object.__setattr__(self, "LON_MAX", self.lon_max)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """(lat_min, lat_max, lon_min, lon_max)"""
        return self.lat_min, self.lat_max, self.lon_min, self.lon_max


@dataclass(frozen=True)
class Settings:
    """Central app settings (paths, metrics, UI options)."""

    # Local data files (kept out of Git)
    tiles_path: str = "data/ookla_mobile_london_2024q4.parquet"
    towers_path: str = "data/opencellid_london.csv"

    # Greater London bbox
    bounds: Bounds = Bounds(
        lat_min=51.28, lat_max=51.70,
        lon_min=-0.5103, lon_max=0.334
    )

    # KPI fields available in tiles
    metrics: Tuple[str, ...] = ("avg_d_kbps", "avg_u_kbps", "avg_lat_ms")

    # Map color modes
    color_modes: Tuple[str, ...] = ("Performance metric", "Tower density", "Performance cluster")

    # Misc
    random_state: int = 0

    # Helpful labels/units used in panels
    metric_labels: Tuple[Tuple[str, str], ...] = (
        ("avg_d_kbps", "Download"),
        ("avg_u_kbps", "Upload"),
        ("avg_lat_ms", "Latency"),
    )
    metric_units: Tuple[Tuple[str, str], ...] = (
        ("avg_d_kbps", "Mbps"),
        ("avg_u_kbps", "Mbps"),
        ("avg_lat_ms", "ms"),
    )

    def label_for(self, m: str) -> str:
        return dict(self.metric_labels).get(m, m)

    def unit_for(self, m: str) -> str:
        return dict(self.metric_units).get(m, "")

    def is_throughput(self, m: str) -> bool:
        return m in ("avg_d_kbps", "avg_u_kbps")

    def all_metrics(self) -> Iterable[str]:
        return self.metrics
