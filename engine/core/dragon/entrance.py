"""Entrance detection / refine on ridge."""
from __future__ import annotations

from typing import Any

import numpy as np

from engine.io.dem import DEM
from engine.core.dragon.types import RidgeLine
from engine.core.dragon.util import _m_per_px_dem


def refine_entrance_on_ordered(
    dem: DEM,
    ordered_coords: np.ndarray,
    *,
    window: int = 7,
    water_dist: np.ndarray | None = None,
    assume_high_to_low: bool = True,
) -> tuple[tuple[int, int] | None, dict]:
    """在已定向脊（源→入首）上精炼入首点：尾段曲率极大 + 高程急降。

    Args:
        ordered_coords: (N,2) 已从源到入首端排序的脊点
        assume_high_to_low: True 时不再反转；False 时自动高→低
    Returns:
        ((row, col) | None, meta)
    """
    coords = np.asarray(ordered_coords)
    n = len(coords)
    meta = {"method": "curv_drop", "refined": False, "score": 0.0}
    if n < 3:
        return None, meta
    if n < 20:
        # 短脊：较低端点
        e0 = float(dem.data[int(coords[0, 0]), int(coords[0, 1])])
        e1 = float(dem.data[int(coords[-1, 0]), int(coords[-1, 1])])
        i = n - 1
        if assume_high_to_low:
            i = n - 1
        else:
            i = 0 if (np.isfinite(e0) and e0 <= e1) else n - 1
        meta["refined"] = True
        meta["method"] = "short_end"
        return (int(coords[i, 0]), int(coords[i, 1])), meta

    elevs = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        r, c = int(coords[i, 0]), int(coords[i, 1])
        if 0 <= r < dem.data.shape[0] and 0 <= c < dem.data.shape[1]:
            elevs[i] = dem.data[r, c]

    if not assume_high_to_low:
        if np.nanmean(elevs[: max(3, n // 5)]) < np.nanmean(elevs[-max(3, n // 5) :]):
            coords = coords[::-1].copy()
            elevs = elevs[::-1].copy()

    dh = np.diff(elevs, prepend=elevs[0])
    dh = np.where(np.isfinite(dh), dh, 0.0)
    ker = np.ones(max(3, window)) / max(3, window)
    drop_signal = -np.convolve(dh, ker, mode="same")
    d2 = np.diff(dh, prepend=dh[0])
    curv_signal = np.abs(np.convolve(d2, ker, mode="same"))

    # 脊线平面曲率（折角）— 第三判据：入首处常有转向
    plan_curv = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        v1 = coords[i].astype(float) - coords[i - 1].astype(float)
        v2 = coords[i + 1].astype(float) - coords[i].astype(float)
        n1 = float(np.hypot(v1[0], v1[1])) + 1e-9
        n2 = float(np.hypot(v2[0], v2[1])) + 1e-9
        cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        plan_curv[i] = 1.0 - cosv  # 0 直、2 回折

    tail_n = max(8, n // 3)
    start = n - tail_n
    best_i = n - 1
    best_s = -1e18
    for i in range(start, n):
        r, c = int(coords[i, 0]), int(coords[i, 1])
        # 三重：急降 + 高程二阶 + 平面折角
        s = (
            1.0 * float(drop_signal[i])
            + 0.55 * float(curv_signal[i])
            + 1.2 * float(plan_curv[i])
        )
        # 脊线终止：越靠近末端略加分（真正「到头」）
        s += 0.15 * (i - start) / max(tail_n - 1, 1)
        if water_dist is not None and 0 <= r < water_dist.shape[0] and 0 <= c < water_dist.shape[1]:
            dw = float(water_dist[r, c]) if np.isfinite(water_dist[r, c]) else 1e9
            if 40.0 <= dw <= 900.0:
                s += 1.2
            elif dw < 25.0:
                s -= 0.8
        if s > best_s:
            best_s = s
            best_i = i
    meta["refined"] = True
    meta["score"] = float(best_s)
    meta["index"] = int(best_i)
    meta["tail_frac"] = float(best_i / max(n - 1, 1))
    return (int(coords[best_i, 0]), int(coords[best_i, 1])), meta


def find_entrance_on_ridge(
    ridge: RidgeLine,
    dem: DEM,
    *,
    window: int = 7,
    water_dist: np.ndarray | None = None,
) -> tuple[int, int] | None:
    """单条脊上的入首：末端 1/3 内 高程急降 ∩ 曲率极值（+ 近水软加分）。

    调研 §5.6：末端节点 + 局部曲率突变，非简单最低点。
    """
    coords = ridge.coords
    pt, _meta = refine_entrance_on_ordered(
        dem, coords, window=window, water_dist=water_dist, assume_high_to_low=False,
    )
    return pt


def find_entrance_point(
    ridges: list[RidgeLine],
    dem: DEM,
    water=None,
) -> tuple[int, int] | None:
    """全局入首：在主要脊线上选「急降+曲率」最优末端点。"""
    if not ridges:
        return None
    wd = None
    try:
        from engine.core.four_beasts_detect import water_distance_rasters
        if water is not None and not getattr(water, "empty", True):
            wd, _ = water_distance_rasters(dem, water, ban_buffer_m=0.0)
    except Exception:
        wd = None

    best = None
    best_score = -1e18
    for ridge in ridges[:15]:
        pt = find_entrance_on_ridge(ridge, dem, water_dist=wd)
        if pt is None:
            continue
        r, c = pt
        elev = float(dem.data[r, c]) if np.isfinite(dem.data[r, c]) else 0.0
        # 脊越长、落差越大越好
        head = ridge.coords[0]
        head_e = float(dem.data[int(head[0]), int(head[1])]) if np.isfinite(
            dem.data[int(head[0]), int(head[1])]
        ) else elev
        drop = max(0.0, head_e - elev)
        sc = drop * max(ridge.sinuosity, 1.0) * min(ridge.length_m / 500.0, 3.0)
        if sc > best_score:
            best_score = sc
            best = pt
    return best


