"""Water distance rasters, same-bank, cross-water segments (field layer)."""
from __future__ import annotations

import numpy as np

from engine.io.dem import DEM

WATER_BAN_BUFFER_M = 60.0

def _m_per_px(dem: DEM) -> tuple[float, float]:
    from engine.core.terrain_analysis import _is_geographic
    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs):
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        return xres * 111_000.0 * cos_lat, yres * 111_000.0
    return xres, yres

def water_distance_rasters(
    dem: DEM,
    water=None,
    ban_buffer_m: float = WATER_BAN_BUFFER_M,
) -> tuple[np.ndarray, np.ndarray]:
    """生成 (dist_to_water_m, water_ban_mask)。

    water_ban_mask=True 的像元禁止作为穴心（水面 + 缓冲）。
    无水系时 dist=inf、ban 全 False。
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt

    h, w = dem.data.shape
    dist = np.full((h, w), np.inf, dtype=np.float64)
    ban = np.zeros((h, w), dtype=bool)
    if water is None:
        return dist, ban
    try:
        empty = bool(getattr(water, "empty", True))
    except Exception:
        empty = True
    if empty:
        return dist, ban

    gdf = getattr(water, "gdf", None)
    if gdf is None or gdf.empty:
        return dist, ban

    try:
        if dem.crs is not None and gdf.crs is not None:
            if str(gdf.crs).upper() != str(dem.crs).upper():
                gdf = gdf.to_crs(dem.crs)
    except Exception:
        pass

    try:
        from rasterio import features as rio_features
    except Exception:
        return dist, ban

    shapes = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            shapes.append((geom, 1))
        except Exception:
            continue
    if not shapes:
        return dist, ban

    try:
        water_mask = rio_features.rasterize(
            shapes,
            out_shape=(h, w),
            transform=dem.transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        ).astype(bool)
    except Exception:
        return dist, ban

    if not water_mask.any():
        return dist, ban

    mpx, mpy = _m_per_px(dem)
    # sampling=(row, col) 米/像素
    dist = distance_transform_edt(~water_mask, sampling=(max(mpy, 1e-6), max(mpx, 1e-6)))
    dist = dist.astype(np.float64)

    px = max(1, int(round(float(ban_buffer_m) / max(min(mpx, mpy), 1e-6))))
    ban = binary_dilation(water_mask, iterations=px) if px > 0 else water_mask.copy()
    return dist, ban



def _segment_hits_water(
    water_surface: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    n_samples: int = 24,
    *,
    end_skip: float = 0.12,
) -> bool:
    """两点连线是否穿过水面（不含端点邻域，避免贴岸误判）。

    用于：
      - 软加分「隔水案山/护砂」（朱雀/龙虎）
      - 硬拒「来龙/少祖/玄武隔水」（水界龙止）

    采样数随像素距离自适应，避免窄河道被端点 skip 漏检。
    """
    if water_surface is None or not np.any(water_surface):
        return False
    h, w = water_surface.shape
    if not (0 <= r0 < h and 0 <= c0 < w and 0 <= r1 < h and 0 <= c1 < w):
        return False
    pix_len = float(np.hypot(r1 - r0, c1 - c0))
    # 至少每像素约 1.5 点，上限 128——OSM 河面常仅 1 像元宽
    n = max(int(n_samples), int(np.ceil(pix_len * 1.5)) + 4, 16)
    n = min(n, 128)
    # 端点跳过：约 1–2 像元，避免贴岸假阳性；勿按长比例跳过（会漏窄河）
    skip_px = max(1.0, min(2.0, pix_len * min(float(end_skip), 0.08)))
    lo = skip_px / max(pix_len, 1.0)
    hi = 1.0 - lo
    if lo >= hi:
        lo, hi = 0.08, 0.92
    # 真水面栅格多为单像素河线：中段命中 ≥1 即视为跨水（水界龙止）
    # 旧逻辑要求连续 2 点 → 嘉陵江级单像元河全部漏检
    for i in range(n):
        t = i / float(n - 1) if n > 1 else 0.5
        if t < lo or t > hi:
            continue
        r = int(round(r0 + t * (r1 - r0)))
        c = int(round(c0 + t * (c1 - c0)))
        if 0 <= r < h and 0 <= c < w and bool(water_surface[r, c]):
            return True
    return False


def _ridge_path_crosses_water(
    water_surface: np.ndarray | None,
    coords: np.ndarray | list,
    *,
    min_hits: int = 2,
    sample_stride: int = 1,
) -> bool:
    """脊线折线是否多次踏入水面（龙脉过水 / 脉断）。"""
    if water_surface is None or not np.any(water_surface):
        return False
    h, w = water_surface.shape
    hits = 0
    arr = np.asarray(coords)
    if len(arr) < 2:
        return False
    step = max(1, int(sample_stride))
    for i in range(0, len(arr) - 1, step):
        r0, c0 = int(arr[i, 0]), int(arr[i, 1])
        r1, c1 = int(arr[min(i + step, len(arr) - 1), 0]), int(
            arr[min(i + step, len(arr) - 1), 1]
        )
        if _segment_hits_water(water_surface, r0, c0, r1, c1, n_samples=12, end_skip=0.05):
            hits += 1
            if hits >= min_hits:
                return True
    return False


def same_bank_as_hole(
    water_surface: np.ndarray | None,
    hole_row: int,
    hole_col: int,
    target_row: int,
    target_col: int,
) -> bool:
    """目标与穴是否同岸：优先陆地连通，其次直线不穿水面。

    理论：界水则止 / 水界龙止——来龙祖宗须与穴同岸连续。
    河曲半岛上「绕岸可达」比直线更合理（直线易误判跨水）。
    """
    if water_surface is None or not np.any(water_surface):
        return True
    # 1) 陆地 4-连通：同陆块 = 同岸（寻龙沿陆脊）
    try:
        from scipy.ndimage import label as nd_label
        land = ~water_surface.astype(bool)
        if (
            0 <= hole_row < land.shape[0]
            and 0 <= hole_col < land.shape[1]
            and 0 <= target_row < land.shape[0]
            and 0 <= target_col < land.shape[1]
            and land[hole_row, hole_col]
            and land[target_row, target_col]
        ):
            labeled, nlab = nd_label(land)
            if nlab > 0 and int(labeled[hole_row, hole_col]) == int(
                labeled[target_row, target_col]
            ) and int(labeled[hole_row, hole_col]) > 0:
                return True
            # 不同陆块 → 异岸
            if (
                int(labeled[hole_row, hole_col]) > 0
                and int(labeled[target_row, target_col]) > 0
                and int(labeled[hole_row, hole_col])
                != int(labeled[target_row, target_col])
            ):
                return False
    except Exception:
        pass
    # 2) 回退：直线穿水
    return not _segment_hits_water(
        water_surface, hole_row, hole_col, target_row, target_col,
        n_samples=32, end_skip=0.10,
    )

