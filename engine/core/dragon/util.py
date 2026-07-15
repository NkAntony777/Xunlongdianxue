"""Shared dragon geometry helpers."""
from __future__ import annotations

import numpy as np

from engine.io.dem import DEM


def _m_per_px_dem(dem: DEM) -> tuple[float, float]:
    from engine.core.terrain_analysis import _is_geographic

    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs):
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        return xres * 111_000.0 * cos_lat, yres * 111_000.0
    return float(xres), float(yres)


def _bearing_rc(
    r0: int, c0: int, r1: int, c1: int, mpx: float, mpy: float
) -> float:
    """从 (r0,c0) 指向 (r1,c1) 的方位角，北=0 东=90。"""
    dx = (c1 - c0) * mpx
    dy = (r0 - r1) * mpy  # 行号向下 → 北为 row 减小
    return float((np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0)


def _ang_diff(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


