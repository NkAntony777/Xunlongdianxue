"""Ridge extraction, vectorize, light mask helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

from engine.io.dem import DEM
from engine.core.dragon.types import RidgeLine
from engine.core.dragon.util import _m_per_px_dem


def extract_ridges(
    flow_acc: np.ndarray,
    dem: DEM,
    min_length_m: float = 50.0,
    smooth_sigma: float = 0.8,
) -> np.ndarray:
    """提取山脊线栅格（分水岭：上游贡献极小）。

    拓扑序正确后，真源/脊 ≈ acc 接近 1；阈值过宽会满图假脊。
    """
    from scipy.ndimage import gaussian_filter, binary_opening

    contributions = np.clip(flow_acc - 1.0, 0, None)
    if smooth_sigma > 0:
        smoothed = gaussian_filter(contributions.astype(np.float64), sigma=smooth_sigma)
    else:
        smoothed = contributions

    # 上游贡献 < 0.5：近似无上游（真分水线）
    ridge = (smoothed < 0.5) & np.isfinite(dem.data)
    ridge = binary_opening(ridge, iterations=1)
    return skeletonize(ridge)


def multi_scale_ridge_mask(
    dem: DEM,
    flow_acc: np.ndarray | None = None,
    *,
    tpi_min: float = 0.6,
) -> np.ndarray:
    """Tier 3：多信号脊带融合。

    - 水文低累积（若提供 flow_acc）
    - TPI 正脊带（特征显著度代理）
    - 断面局部极大（LANDMARK/剖面极值简化）
    """
    from scipy.ndimage import gaussian_filter, binary_opening, maximum_filter

    data = dem.data.astype(np.float64)
    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros(data.shape, dtype=bool)
    fill = np.where(valid, data, float(np.nanmean(data[valid])))

    # TPI
    local = gaussian_filter(fill, sigma=1.0)
    base = gaussian_filter(fill, sigma=5.0)
    tpi = local - base
    tpi_ridge = valid & (tpi >= tpi_min)

    # 断面极大：行/列 3 邻域峰值
    mx_r = maximum_filter(fill, size=(1, 3), mode="nearest")
    mx_c = maximum_filter(fill, size=(3, 1), mode="nearest")
    prof = valid & ((fill >= mx_r - 1e-6) | (fill >= mx_c - 1e-6)) & (tpi > 0.2)

    hydro = np.zeros(data.shape, dtype=bool)
    if flow_acc is not None and flow_acc.shape == data.shape:
        contrib = np.clip(flow_acc - 1.0, 0, None)
        sm = gaussian_filter(contrib.astype(np.float64), sigma=0.8)
        hydro = valid & (sm < 0.5)

    combined = hydro | (tpi_ridge & prof) | (tpi_ridge & hydro)
    if not combined.any():
        combined = tpi_ridge
    combined = binary_opening(combined, iterations=1)
    return skeletonize(combined)


def feature_significance_filter(
    ridges: list[RidgeLine],
    *,
    min_sig_ratio: float = 0.15,
    keep_top: int = 40,
) -> list[RidgeLine]:
    """Tier 3：按特征显著度裁剪弱脊，保留主脉候选。"""
    if not ridges:
        return []
    sigs = np.array([max(r.feature_significance, 1e-9) for r in ridges], dtype=np.float64)
    thr = float(np.nanpercentile(sigs, 100 * (1.0 - min(min_sig_ratio * 3, 0.85))))
    # 至少保留较长的 top
    ranked = sorted(ridges, key=lambda r: -r.feature_significance)
    kept = [r for r in ranked if r.feature_significance >= thr * 0.5 or r.length_m > 400]
    if len(kept) < 3:
        kept = ranked[: min(keep_top, len(ranked))]
    return kept[:keep_top]



def vectorize_ridges(
    ridge_mask: np.ndarray,
    dem: DEM,
    min_length_m: float = 50.0,
) -> list[RidgeLine]:
    """将山脊线栅格矢量化并计算属性。"""
    from skimage.measure import label as ski_label
    from engine.core.terrain_analysis import _is_geographic

    labeled = ski_label(ridge_mask, connectivity=2)
    xres, yres = dem.resolution
    if _is_geographic(dem.crs):
        m_per_unit = 111000.0
    else:
        m_per_unit = 1.0
    min_pixels = int(min_length_m / (min(xres, yres) * m_per_unit))

    ridges: list[RidgeLine] = []
    for region in regionprops(labeled):
        if region.area < min_pixels:
            continue
        coords = region.coords  # (N, 2) (row, col)
        # 计算蜿蜒度
        if len(coords) < 2:
            continue
        actual_length = 0.0
        for i in range(1, len(coords)):
            dr = (coords[i, 0] - coords[i - 1, 0]) * yres * m_per_unit
            dc = (coords[i, 1] - coords[i - 1, 1]) * xres * m_per_unit
            actual_length += np.sqrt(dr ** 2 + dc ** 2)
        ys0 = coords[0, 0] * yres * m_per_unit
        xs0 = coords[0, 1] * xres * m_per_unit
        ys1 = coords[-1, 0] * yres * m_per_unit
        xs1 = coords[-1, 1] * xres * m_per_unit
        straight_length = np.sqrt(
            (ys0 - ys1) ** 2 + (xs0 - xs1) ** 2
        )
        sinuosity = float(actual_length / straight_length) if straight_length > 0 else 1.0

        # 沿线高程统计
        elevs = dem.data[coords[:, 0], coords[:, 1]]
        valid_elevs = elevs[np.isfinite(elevs)]
        if valid_elevs.size == 0:
            continue

        # 特征显著度（粗略：高程均值 × 蜿蜒度 / 长度）
        feature_significance = float(
            np.nanmean(valid_elevs) * sinuosity / max(actual_length, 1)
        )

        ridges.append(
            RidgeLine(
                coords=coords,
                length_m=float(actual_length),
                mean_elevation=float(np.nanmean(valid_elevs)),
                max_elevation=float(np.nanmax(valid_elevs)),
                sinuosity=sinuosity,
                feature_significance=feature_significance,
            )
        )
    return ridges



def light_ridge_mask(
    dem: DEM,
    sigma_local: float = 1.0,
    sigma_base: float = 5.0,
    tpi_min: float = 0.8,
    dilate_px: int = 1,
) -> np.ndarray:
    """轻量脊带：TPI（相对中尺度基底抬升）为正的分水岭带。

    比全量 D8 龙脉快 2 个数量级，适合四象实时路径。
    """
    from scipy.ndimage import gaussian_filter, binary_dilation

    data = dem.data.astype(np.float64)
    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros(data.shape, dtype=bool)
    fill = np.where(valid, data, np.nanmean(data[valid]))
    local = gaussian_filter(fill, sigma=sigma_local)
    base = gaussian_filter(fill, sigma=sigma_base)
    tpi = local - base
    ridge = valid & (tpi >= float(tpi_min))
    if dilate_px > 0:
        ridge = binary_dilation(ridge, iterations=int(dilate_px))
    return ridge


def _ridge_path_fraction(
    ridge_mask: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    n_samples: int = 20,
) -> float:
    """两点间折线落在脊带上的比例（端点邻域略忽略）。"""
    if ridge_mask is None or not np.any(ridge_mask):
        return 0.0
    h, w = ridge_mask.shape
    n = max(6, int(n_samples))
    hit = 0
    tot = 0
    for i in range(1, n - 1):
        t = i / float(n - 1)
        if t < 0.08 or t > 0.92:
            continue
        r = int(round(r0 + t * (r1 - r0)))
        c = int(round(c0 + t * (c1 - c0)))
        if 0 <= r < h and 0 <= c < w:
            tot += 1
            if ridge_mask[r, c]:
                hit += 1
    return float(hit / tot) if tot else 0.0


def _order_ridge_by_dist_to_hole(
    coords: np.ndarray,
    center_row: int,
    center_col: int,
    mpx: float,
    mpy: float,
) -> np.ndarray:
    """按到穴距离排序脊点（近→远），便于从入首向外取父母/少祖。"""
    if coords is None or len(coords) == 0:
        return coords
    dr = (coords[:, 0] - center_row) * mpy
    dc = (coords[:, 1] - center_col) * mpx
    d = np.hypot(dr, dc)
    order = np.argsort(d)
    return coords[order]


