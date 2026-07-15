"""四象位置识别 + 主轴线（算法实现，非视觉贴图）。

理论依据：
  - research/02_four_beasts/00_四兽量化规则.md
  - 背山面水：朝向优先指向最近水系；否则背向局地最高砂山
  - 玄武（后靠）：50–500 m 为主，高程应高于穴
  - 朱雀（前案/朝）：200 m–3 km，宜低于靠山（案山不欺主）
  - 青龙（左）/ 白虎（右）：0.5–3 穴距量级，白虎不宜高于青龙
  - 少祖：坐向后方更远的祖山峰，宜高于玄武

坐标约定：
  - facing = **朝向**（人面朝方向，北=0°，东=90°，南=180°，西=270°）
  - 坐向 sit = (facing + 180) % 360（玄武方向）
  - 左青龙 = (facing + 270) % 360，右白虎 = (facing + 90) % 360
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import maximum_filter

from engine.io.dem import DEM


# —— 比例制距离（局尺度 L 归一，非绝对米硬限）——
# L = 多信号特征长度（AOI / 来龙脊 / 前水）；窗 = frac × L，仅像元噪声地板。
# 兼容旧名：下列 *_DIST_M 为 L≈2.2km 时的示意米值，运行时以 beast_distance_windows 为准。
XUANWU_FRAC = (0.12, 0.38)         # 父母山 / 玄武：L 的 12%–38%
SHAOZU_FRAC = (0.42, 1.65)         # 少祖：更远，可超 L
SIDE_FRAC = (0.10, 0.55)           # 龙虎
BAIHU_FRAC = (0.12, 0.48)          # 白虎（略紧于青龙上界）
ZHUQUE_FRAC = (0.12, 1.25)         # 朱雀案朝
# 层级：少祖 / 玄武 距离比（拓扑相对，与米无关）
SHAOZU_XUANWU_DIST_RATIO = 2.0
# 贴身否决：相对 L 的最小比例（非固定米）
XUANWU_MIN_FRAC = 0.10
BAIHU_MIN_FRAC = 0.10
SHAOZU_MIN_FRAC = 0.35
# 仅噪声地板：k × 像元（防 DEM 邻元假峰）
NOISE_FLOOR_CELLS = 3.0
# 玄武高差：下限固定小噪声带；上限随 L 缓变（在 windows 里可重算）
XUANWU_DH_SWEET_M = (25.0, 150.0)
# 旧常量别名（文档/测试兼容；真值运行时覆盖）
XUANWU_DIST_M = (200.0, 700.0)
ZHUQUE_DIST_M = (200.0, 3000.0)
SIDE_DIST_M = (180.0, 1400.0)
BAIHU_DIST_M = (200.0, 1000.0)
SHAOZU_DIST_M = (1000.0, 8000.0)
SECTOR_HALF_BACK = 38.0
SECTOR_HALF_FRONT = 45.0
SECTOR_HALF_SIDE = 45.0
SECTOR_HALF_SHAOZU = 48.0
PEAK_FOOTPRINT_PX = 3
WATER_BAN_BUFFER_M = 60.0
BEAST_WATER_BAN_M = 12.0
CROSS_WATER_BONUS_ZHUQUE = 0.38
CROSS_WATER_BONUS_SIDE = 0.15
CROSS_WATER_PENALTY_BACK = 3.5
REJECT_CROSS_WATER_BACK = True
BAIHU_QL_ELEV_RATIO = 0.85


def noise_floor_m(cell_m: float) -> float:
    """像元噪声地板：仅防邻元假峰，不充当地理尺度。"""
    c = max(float(cell_m), 1.0)
    return float(NOISE_FLOOR_CELLS * c)


def estimate_site_scale_m(
    aoi_half_diag_m: float,
    *,
    cell_m: float = 30.0,
    ridge_length_m: float | None = None,
    water_front_m: float | None = None,
    elev_range_m: float | None = None,
) -> dict[str, float]:
    """估计局尺度 L（米），供比例窗使用。

    信号（可缺省）：
      - L_aoi：图幅半对角 ×0.85（分析区尺度）
      - L_ridge：主来龙源→入首弧长（龙局尺度）
      - L_water：穴→前水距离 ×2.2（明堂/界水尺度）
      - L_relief：高差 ×25（山势粗尺度，弱权重）

    合成：对有效信号取加权几何平均，避免单源极端。
    """
    nf = noise_floor_m(cell_m)
    parts: list[tuple[float, float]] = []  # (value, weight)
    L_aoi = max(nf * 20.0, float(aoi_half_diag_m) * 0.85)
    parts.append((L_aoi, 1.0))
    if ridge_length_m is not None and np.isfinite(ridge_length_m) and ridge_length_m > nf * 10:
        parts.append((float(ridge_length_m), 1.25))
    if water_front_m is not None and np.isfinite(water_front_m) and water_front_m > nf:
        # 明堂特征长 ≈ 前水距的 2–2.5 倍
        parts.append((float(water_front_m) * 2.2, 1.1))
    if elev_range_m is not None and np.isfinite(elev_range_m) and elev_range_m > 5.0:
        parts.append((float(elev_range_m) * 25.0, 0.35))

    # 加权 log 平均
    log_sum = 0.0
    w_sum = 0.0
    for v, w in parts:
        v = max(v, nf * 15.0)
        log_sum += w * float(np.log(v))
        w_sum += w
    L = float(np.exp(log_sum / max(w_sum, 1e-9)))
    # 相对 AOI 软夹：不超出图幅可解释范围太多，也不小于噪声地板团
    L = float(np.clip(L, nf * 25.0, max(L_aoi * 1.35, nf * 40.0)))
    return {
        "L": L,
        "L_aoi": L_aoi,
        "L_ridge": float(ridge_length_m) if ridge_length_m else 0.0,
        "L_water": float(water_front_m) * 2.2 if water_front_m else 0.0,
        "noise_floor_m": nf,
        "cell_m": float(cell_m),
    }


def beast_distance_windows(
    aoi_half_diag_m: float,
    *,
    cell_m: float = 30.0,
    ridge_length_m: float | None = None,
    water_front_m: float | None = None,
    elev_range_m: float | None = None,
    dist_cap_m: float | None = None,
    L: float | None = None,
) -> dict[str, Any]:
    """比例制四象/祖山距离窗：d ∈ [f_lo, f_hi] × L，仅噪声地板。

    不再用 180m/800m 一类绝对硬限（小谷与大河湾尺度差一个数量级）。
    返回值含各窗 (lo, hi) 米（由 L 换算）及 scale 元数据。
    """
    scale = estimate_site_scale_m(
        aoi_half_diag_m,
        cell_m=cell_m,
        ridge_length_m=ridge_length_m,
        water_front_m=water_front_m,
        elev_range_m=elev_range_m,
    )
    if L is not None and np.isfinite(L) and L > 0:
        scale = dict(scale)
        scale["L"] = float(L)
    Ls = float(scale["L"])
    nf = float(scale["noise_floor_m"])
    cap = float(dist_cap_m) if dist_cap_m is not None else Ls * 1.8

    def _win(f_lo: float, f_hi: float, *, min_frac_hard: float = 0.0) -> tuple[float, float]:
        lo = max(nf * 2.0, f_lo * Ls, min_frac_hard * Ls)
        hi = max(lo + nf * 4.0, min(f_hi * Ls, cap))
        return (float(lo), float(hi))

    xw = _win(XUANWU_FRAC[0], XUANWU_FRAC[1], min_frac_hard=XUANWU_MIN_FRAC)
    sz = _win(SHAOZU_FRAC[0], SHAOZU_FRAC[1], min_frac_hard=SHAOZU_MIN_FRAC)
    # 少祖下界至少为玄武下界 × 比例（相对约束）
    sz = (max(sz[0], xw[0] * SHAOZU_XUANWU_DIST_RATIO), sz[1])
    if sz[1] < sz[0] + nf:
        sz = (sz[0], sz[0] + max(nf * 8.0, 0.15 * Ls))
    bh = _win(BAIHU_FRAC[0], BAIHU_FRAC[1], min_frac_hard=BAIHU_MIN_FRAC)
    side = _win(SIDE_FRAC[0], SIDE_FRAC[1], min_frac_hard=BAIHU_MIN_FRAC * 0.9)
    zq = _win(ZHUQUE_FRAC[0], ZHUQUE_FRAC[1])
    return {
        "xuanwu": xw,
        "shaozu": sz,
        "baihu": bh,
        "side": side,
        "zhuque": zq,
        "scale": scale,
        "L": Ls,
        "noise_floor_m": nf,
        # 硬下限 = 窗下界（已是 frac×L），供调用方 hard_min
        "xuanwu_min_hard_m": xw[0],
        "baihu_min_hard_m": bh[0],
        "shaozu_min_hard_m": sz[0],
    }


@dataclass
class BeastPoint:
    """单个砂/山位置 + 可解释指标。"""

    x: float
    y: float
    row: int
    col: int
    elev_m: float
    dist_m: float
    bearing_deg: float
    score: float = 0.0


@dataclass
class FourBeastsPositions:
    """四象位置结果。"""

    shaozu: tuple[float, float] | None  # (x, y)
    xuanwu: tuple[float, float] | None
    zhuque: tuple[float, float] | None
    qinglong: tuple[float, float] | None
    baihu: tuple[float, float] | None
    center: tuple[float, float] | None  # 中心点（参考候选穴）
    facing: float  # 朝向（度，北=0，人面朝方向）
    # 可解释元数据（API / 面板可选用）
    sit: float = 0.0  # 坐向 = facing+180
    facing_method: str = "default"
    meta: dict[str, Any] = field(default_factory=dict)


def _m_per_px(dem: DEM) -> tuple[float, float]:
    """返回 (m_per_px_x, m_per_px_y)。"""
    from engine.core.terrain_analysis import _is_geographic

    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs):
        # 中纬度修正：经度方向 × cos(lat)
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        return xres * 111_000.0 * cos_lat, yres * 111_000.0
    return xres, yres


def _rowcol_to_xy(dem: DEM, r: int, c: int) -> tuple[float, float]:
    return dem.xy(r, c)


def _local_maxima_mask(data: np.ndarray, size: int = 5) -> np.ndarray:
    """局部高程极大值掩膜（排除平台与 nodata）。"""
    from scipy.ndimage import uniform_filter

    valid = np.isfinite(data)
    filled = np.where(valid, data, -np.inf)
    footprint = max(3, size if size % 2 == 1 else size + 1)
    mx = maximum_filter(filled, size=footprint, mode="nearest")
    peaks = valid & (filled == mx)
    # 去掉过平的“假峰”：相对 3×3 邻域平均抬升 < 0.3 m
    local_sum = uniform_filter(np.where(valid, data, 0.0), size=3, mode="nearest")
    local_cnt = uniform_filter(valid.astype(np.float64), size=3, mode="nearest")
    local_mean = local_sum / np.maximum(local_cnt, 1e-9)
    relief = np.where(valid, data - local_mean, 0.0)
    peaks = peaks & (relief >= 0.3)
    return peaks


def _ideal_dist_score(dist_m: float, d_lo: float, d_hi: float) -> float:
    """距离在 [d_lo, d_hi] 内满分，之外高斯衰减。"""
    if dist_m <= 0:
        return 0.0
    if d_lo <= dist_m <= d_hi:
        # 区间中点附近略优
        mid = 0.5 * (d_lo + d_hi)
        span = max(d_hi - d_lo, 1.0)
        return 1.0 - 0.15 * abs(dist_m - mid) / span
    if dist_m < d_lo:
        # 过近
        return max(0.0, 0.55 * dist_m / d_lo)
    # 过远
    over = (dist_m - d_hi) / max(d_hi, 1.0)
    return max(0.0, 0.85 * np.exp(-over * 1.2))


def _dh_sweet_score(rel_m: float, lo: float, hi: float) -> float:
    """相对高差甜区评分：区间内高，过低/过高衰减（玄武父母山）。"""
    if rel_m < 0:
        return -0.6
    if lo <= rel_m <= hi:
        mid = 0.5 * (lo + hi)
        span = max(hi - lo, 1.0)
        return 1.0 - 0.2 * abs(rel_m - mid) / span
    if rel_m < lo:
        return max(-0.3, 0.7 * rel_m / max(lo, 1.0) - 0.2)
    # 过高逼山
    over = (rel_m - hi) / max(hi, 1.0)
    return max(-0.5, 0.7 * np.exp(-over * 1.5) - 0.3)


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


# 得水距离工具仍供 detect 等模块兼容导出
from engine.core.water_model import (  # noqa: E402
    water_get_score,
    water_sha_penalty,
    water_score_from_dist,
)


def _bearing_deg(dx_m: float, dy_m: float) -> float:
    """北=0、东=90 的方位角。dy_m 向北为正。"""
    return float((np.degrees(np.arctan2(dx_m, dy_m)) + 360.0) % 360.0)


def _angle_diff(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


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


def _nearest_water_bearing(
    dem: DEM,
    center_row: int,
    center_col: int,
    water,
) -> tuple[float, float] | None:
    """最近水体方位 + 距离（米）。失败返回 None。"""
    if water is None or getattr(water, "empty", True):
        return None
    try:
        from shapely.ops import nearest_points

        cx_xy, cy_xy = dem.xy(center_row, center_col)
        pt3857 = water._to_3857(cx_xy, cy_xy)
        dists = water.projected_gdf.distance(pt3857)
        if not len(dists) or not np.isfinite(dists.min()) or dists.min() >= 1e7:
            return None
        idx = int(dists.idxmin())
        geom = water.projected_gdf.geometry.iloc[idx]
        _, near = nearest_points(pt3857, geom)
        dx = float(near.x - pt3857.x)
        dy = float(near.y - pt3857.y)
        if abs(dx) + abs(dy) <= 1.0:
            return None
        return _bearing_deg(dx, dy), float(dists.min())
    except Exception:
        return None


def _infer_back_high_az(
    dem: DEM,
    center_row: int,
    center_col: int,
    search_radius_m: float = 2000.0,
) -> tuple[float | None, float]:
    """靠山方位（坐向）与强度分。无可靠靠山时 (None, -1)。"""
    h, w = dem.data.shape
    mpx, mpy = _m_per_px(dem)
    yy, xx = np.mgrid[0:h, 0:w]
    dx_m = (xx - center_col) * mpx
    dy_m = (center_row - yy) * mpy
    dist_m = np.hypot(dx_m, dy_m)
    elev = dem.data
    valid = np.isfinite(elev) & (dist_m >= 40.0) & (dist_m <= search_radius_m)
    if not valid.any():
        return None, -1.0

    cand_elev = float(elev[center_row, center_col])
    rel = np.where(valid, elev - cand_elev, -np.inf)
    high = valid & (rel >= 5.0)
    if not high.any():
        high = valid

    bearing = _bearing_deg_arr(dx_m, dy_m)
    best_back = None
    best_score = -1.0
    for k in range(16):
        center_az = k * 22.5
        half = 22.5
        diff = np.abs(((bearing - center_az + 180.0) % 360.0) - 180.0)
        mask = high & (diff <= half)
        if not mask.any():
            continue
        idxs = np.where(mask)
        scores = []
        for r, c in zip(idxs[0], idxs[1]):
            d = float(dist_m[r, c])
            rh = float(elev[r, c] - cand_elev)
            scores.append(rh * _ideal_dist_score(d, 50.0, 800.0))
        sec_score = float(np.max(scores)) if scores else -1.0
        if sec_score > best_score:
            best_score = sec_score
            best_back = center_az
    return best_back, float(best_score)


def _front_sector_metrics(
    dem: DEM,
    center_row: int,
    center_col: int,
    facing_deg: float,
    *,
    half_deg: float = 50.0,
    r_lo: float = 60.0,
    r_hi: float = 1200.0,
    water_dist: np.ndarray | None = None,
) -> dict[str, float]:
    """前方扇区：开阔度（相对穴偏低）+ 得水（中距水面）。"""
    h, w = dem.data.shape
    mpx, mpy = _m_per_px(dem)
    yy, xx = np.mgrid[0:h, 0:w]
    dx_m = (xx - center_col) * mpx
    dy_m = (center_row - yy) * mpy
    dist_m = np.hypot(dx_m, dy_m)
    bearing = _bearing_deg_arr(dx_m, dy_m)
    ang = np.abs(((bearing - facing_deg + 180.0) % 360.0) - 180.0)
    band = (
        np.isfinite(dem.data)
        & (dist_m >= r_lo)
        & (dist_m <= r_hi)
        & (ang <= half_deg)
    )
    if not band.any():
        return {"openness": 0.0, "water": 0.0, "n": 0.0}

    cand_elev = float(dem.data[center_row, center_col])
    rel = dem.data[band] - cand_elev
    # 前方相对低/平 → 明堂开阔
    openness = float(np.clip(1.0 - np.nanmean(rel) / 40.0, 0.0, 1.5))
    water_sc = 0.0
    if water_dist is not None and water_dist.shape == (h, w):
        wd = water_dist[band]
        finite = np.isfinite(wd)
        if finite.any():
            wdv = wd[finite]
            # 80–800 m 得水甜区；贴水(<40) 与过远均降权
            sweet = (wdv >= 40.0) & (wdv <= 900.0)
            near_frac = float(np.mean(wdv < 30.0))  # 前方大片是水也算面水
            mid_frac = float(np.mean(sweet))
            water_sc = float(np.clip(0.55 * mid_frac + 0.65 * min(near_frac * 2.0, 1.0), 0.0, 1.5))
            # 前方有水但不要求穴贴水
            if float(np.min(wdv)) < 1500.0:
                water_sc = max(water_sc, 0.35 * (1.0 - min(float(np.min(wdv)), 1500.0) / 1500.0))
    return {"openness": openness, "water": water_sc, "n": float(band.sum())}


def infer_facing(
    dem: DEM,
    center_row: int,
    center_col: int,
    water=None,
    search_radius_m: float = 2000.0,
) -> tuple[float, str]:
    """推断朝向（人面朝方向）。

    河湾修正（避免「最近岸把朝向拧反」）：
      1. 靠山方位（坐）与前方明堂/得水联合打分
      2. 最近水体仅作候选；与靠山冲突（>75°）时让位靠山
      3. 优先：背高 + 前方开阔/有水（真正背山面水）
      4. 无信号时默认坐北朝南 facing=180
    """
    h, w = dem.data.shape
    if not (0 <= center_row < h and 0 <= center_col < w):
        return 180.0, "default_south"

    back_az, back_score = _infer_back_high_az(
        dem, center_row, center_col, search_radius_m=search_radius_m,
    )
    face_from_back = (
        (float(back_az) + 180.0) % 360.0 if back_az is not None else None
    )

    nearest = _nearest_water_bearing(dem, center_row, center_col, water)
    nearest_face = nearest[0] if nearest else None
    nearest_dist = nearest[1] if nearest else None

    # 水系距离栅格（0 缓冲表面距离，供扇区得水）
    wd: np.ndarray | None = None
    if water is not None and not getattr(water, "empty", True):
        try:
            wd, _ = water_distance_rasters(dem, water, ban_buffer_m=0.0)
            if not np.isfinite(wd).any():
                wd = None
        except Exception:
            wd = None

    # —— 16 方位候选朝向综合分 ——
    candidates: list[tuple[float, float, str]] = []  # (facing, score, method)

    for k in range(16):
        face = k * 22.5
        met = _front_sector_metrics(
            dem, center_row, center_col, face,
            half_deg=50.0, r_lo=50.0, r_hi=min(search_radius_m, 1500.0),
            water_dist=wd,
        )
        # 该朝向的「背后」应是靠山
        sit = (face + 180.0) % 360.0
        align_back = 0.0
        if back_az is not None:
            align_back = 1.0 - _angle_diff(sit, float(back_az)) / 180.0
        # 与最近水方向一致则小加分（但不主导）
        align_near = 0.0
        if nearest_face is not None:
            align_near = 1.0 - _angle_diff(face, float(nearest_face)) / 180.0

        sc = (
            1.6 * met["openness"]
            + 1.8 * met["water"]
            + 2.2 * max(0.0, align_back) * min(1.0, max(back_score, 0.0) / 80.0)
            + 0.35 * max(0.0, align_near)
        )
        # 前方几乎无样本的扇区降权
        if met["n"] < 20:
            sc *= 0.4
        method = "mingtang_face_water"
        if align_back > 0.75 and met["water"] > 0.15:
            method = "back_high_face_water"
        elif align_back > 0.75 and met["openness"] > 0.5:
            method = "back_to_high_terrain"
        candidates.append((face, sc, method))

    # 显式加入「靠山反方向」「最近水」以便可比
    if face_from_back is not None:
        met_b = _front_sector_metrics(
            dem, center_row, center_col, face_from_back,
            water_dist=wd, r_hi=min(search_radius_m, 1500.0),
        )
        sc_b = (
            2.0 * max(0.0, min(back_score, 120.0) / 80.0)
            + 1.5 * met_b["openness"]
            + 1.6 * met_b["water"]
        )
        candidates.append((face_from_back, sc_b + 0.5, "back_to_high_terrain"))

    if nearest_face is not None:
        met_n = _front_sector_metrics(
            dem, center_row, center_col, float(nearest_face),
            water_dist=wd, r_hi=min(search_radius_m, 1500.0),
        )
        sc_n = 1.0 * met_n["water"] + 0.8 * met_n["openness"] + 0.6
        # 关键：与靠山冲突时重罚「只朝最近岸」
        if face_from_back is not None:
            conflict = _angle_diff(float(nearest_face), face_from_back)
            if conflict > 75.0:
                sc_n -= 2.5 * (conflict / 180.0)
            elif conflict < 40.0:
                sc_n += 1.2  # 最近水与背高一致 → 真·背山面水
        # 最近水过近（割脚向）不额外鼓励
        if nearest_dist is not None and nearest_dist < 40.0:
            sc_n -= 0.4
        candidates.append((float(nearest_face), sc_n, "face_water"))

    if not candidates:
        return 180.0, "default_south"

    # 选最高分
    candidates.sort(key=lambda t: -t[1])
    best_face, best_sc, best_method = candidates[0]

    # 最终护栏：若最优是 face_water 但与靠山反方向差 >90°，强制改靠山
    if (
        best_method == "face_water"
        and face_from_back is not None
        and _angle_diff(best_face, face_from_back) > 90.0
        and back_score > 8.0
    ):
        return float(face_from_back), "back_high_over_nearest_water"

    # 分过低且无水无靠 → 默认南
    if best_sc < 0.35 and face_from_back is None and nearest_face is None:
        return 180.0, "default_south"

    return float(best_face) % 360.0, best_method


def _bearing_deg_arr(dx_m: np.ndarray, dy_m: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(dx_m, dy_m)) + 360.0) % 360.0


def _select_peak_in_sector(
    dem: DEM,
    center_row: int,
    center_col: int,
    direction_deg: float,
    sector_half: float,
    dist_range_m: tuple[float, float],
    peaks_mask: np.ndarray,
    occupied: Iterable[tuple[int, int]] | None = None,
    prefer_higher_than: float | None = None,
    prefer_lower_than: float | None = None,
    min_elev_above_cand: float | None = None,
    max_elev_above_cand: float | None = None,
    max_elev_abs: float | None = None,
    elev_mode: str = "higher",
    dh_sweet: tuple[float, float] | None = None,
    weight_elev: float = 1.0,
    weight_dist: float = 1.0,
    border_margin: int = 3,
    max_dist_cap_m: float | None = None,
    forbid_mask: np.ndarray | None = None,
    water_surface_mask: np.ndarray | None = None,
    cross_water_bonus: float = 0.0,
    cross_water_penalty: float = 0.0,
    reject_cross_water: bool = False,
    viewshed_bonus: float = 0.0,
    out_rejected_cross: list | None = None,
    hard_min_dist_m: float | None = None,
    soft_min_frac: float = 0.35,
) -> BeastPoint | None:
    """在方位扇区 + 距离环内，从局部峰值中选综合最优者。

    硬约束：
      - 避开图幅边缘 border_margin 像素（防止 (0,0) 角点假峰）
      - 距离不超过 max_dist_cap_m（通常为 AOI 半对角）
      - forbid_mask（真水面等）上的峰跳过——不禁止对岸干峰
      - reject_cross_water：穴→峰穿水面则跳过（来龙/少祖/玄武同岸）
      - elev_mode: higher | lower | sweet（sweet 用 dh_sweet 高差窗）

    软偏好：
      - cross_water_bonus>0 且线段穿过 water_surface_mask 时加分（隔水案/护砂）
      - cross_water_penalty>0 且跨水时减分（未硬拒时）
      - viewshed_bonus>0：穴→峰视线开阔加分（朝案有情）
    """
    h, w = dem.data.shape
    mpx, mpy = _m_per_px(dem)
    cand_elev = float(dem.data[center_row, center_col])
    occupied = list(occupied or [])
    margin = max(1, int(border_margin))

    yy, xx = np.mgrid[0:h, 0:w]
    dx_m = (xx - center_col) * mpx
    dy_m = (center_row - yy) * mpy
    dist_m = np.hypot(dx_m, dy_m)
    bearing = _bearing_deg_arr(dx_m, dy_m)
    ang = np.abs(((bearing - direction_deg + 180.0) % 360.0) - 180.0)

    d_lo, d_hi = dist_range_m
    if max_dist_cap_m is not None:
        d_hi = min(d_hi, max_dist_cap_m)
    d_search_hi = min(d_hi * 1.35, max_dist_cap_m if max_dist_cap_m else d_hi * 1.35)
    # 硬下限：少祖等场景禁止用 soft_min_frac 偷近
    d_min_hard = float(hard_min_dist_m) if hard_min_dist_m is not None else max(
        10.0, float(d_lo) * float(soft_min_frac)
    )
    d_min_hard = max(10.0, d_min_hard)

    interior = (
        (yy >= margin) & (yy < h - margin)
        & (xx >= margin) & (xx < w - margin)
    )
    dry = np.ones((h, w), dtype=bool)
    if forbid_mask is not None and forbid_mask.shape == (h, w):
        dry = ~forbid_mask.astype(bool)

    region = (
        peaks_mask
        & interior
        & dry
        & np.isfinite(dem.data)
        & (dist_m >= d_min_hard)
        & (dist_m <= d_search_hi)
        & (ang <= sector_half)
    )
    if not region.any():
        # 回退：扇区内任意高点（仍禁止贴边/真水面/硬下限）
        region = (
            interior
            & dry
            & np.isfinite(dem.data)
            & (dist_m >= d_min_hard)
            & (dist_m <= d_search_hi)
            & (ang <= sector_half)
        )
        if not region.any():
            return None

    has_ws = (
        water_surface_mask is not None
        and water_surface_mask.shape == (h, w)
        and bool(np.any(water_surface_mask))
    )
    use_cross_bonus = cross_water_bonus > 0 and has_ws
    use_cross_pen = (cross_water_penalty > 0 or reject_cross_water) and has_ws

    best: BeastPoint | None = None
    best_s = -1e18
    best_cross: BeastPoint | None = None  # 跨水最优峰 → 可改标朝砂
    best_cross_s = -1e18
    rs, cs = np.where(region)
    for r, c in zip(rs.tolist(), cs.tolist()):
        skip = False
        for or_, oc in occupied:
            if abs(r - or_) <= PEAK_FOOTPRINT_PX and abs(c - oc) <= PEAK_FOOTPRINT_PX:
                skip = True
                break
        if skip:
            continue
        elev = float(dem.data[r, c])
        d = float(dist_m[r, c])
        brg = float(bearing[r, c])
        rel = elev - cand_elev

        if min_elev_above_cand is not None and rel < min_elev_above_cand:
            continue
        if max_elev_above_cand is not None and rel > max_elev_above_cand:
            continue
        if max_elev_abs is not None and elev > max_elev_abs:
            continue

        s_dist = _ideal_dist_score(d, d_lo, d_hi)
        if elev_mode == "sweet" and dh_sweet is not None:
            s_elev = _dh_sweet_score(rel, dh_sweet[0], dh_sweet[1])
        elif elev_mode == "lower":
            # 白虎驯俯 / 朱雀不欺主：中低丘优，过高惩罚
            s_elev = float(np.clip(1.0 - rel / 60.0, -1.0, 1.0))
            if rel < 0:
                s_elev -= 0.3
        else:
            s_elev = float(np.clip(rel / 80.0, -0.5, 1.5))
        s = weight_dist * s_dist + weight_elev * s_elev
        s += 0.15 * (1.0 - _angle_diff(brg, direction_deg) / max(sector_half, 1.0))

        # 图缘/角点重罚（防止少祖玄武飞到 (12,12) 一类假峰）
        edge_prox = min(r, c, h - 1 - r, w - 1 - c)
        if edge_prox < margin:
            continue  # 硬禁：margin 内不选
        if edge_prox < margin + 4:
            s -= 1.6
        elif edge_prox < margin + 8:
            s -= 0.6

        if prefer_higher_than is not None:
            if elev > prefer_higher_than:
                s += 0.25
            else:
                s -= 0.35 * (prefer_higher_than - elev) / 50.0

        if prefer_lower_than is not None:
            if elev <= prefer_lower_than:
                s += 0.35
            else:
                # 白虎抬头：重罚
                s -= 0.9 * (elev - prefer_lower_than) / 40.0

        crosses = False
        if has_ws and (use_cross_bonus or use_cross_pen):
            crosses = _segment_hits_water(
                water_surface_mask, center_row, center_col, r, c
            )

        # 隔水案/护砂：加分
        if use_cross_bonus and crosses:
            s += float(cross_water_bonus)

        # 来龙/少祖/玄武：跨水硬拒或重罚（水界龙止）
        if use_cross_pen and crosses:
            x, y = dem.xy(r, c)
            cand_bp = BeastPoint(
                x=float(x), y=float(y), row=int(r), col=int(c),
                elev_m=elev, dist_m=d, bearing_deg=brg, score=float(s),
            )
            if s > best_cross_s:
                best_cross_s = s
                best_cross = cand_bp
            if reject_cross_water:
                continue
            s -= float(cross_water_penalty)

        # 朝案视线（Viewshed 简化）
        if viewshed_bonus > 0:
            try:
                from engine.core.dragon_vein import sector_viewshed_score
                vs = sector_viewshed_score(
                    dem, center_row, center_col, r, c,
                )
                s += float(viewshed_bonus) * vs
            except Exception:
                pass

        if s > best_s:
            best_s = s
            x, y = dem.xy(r, c)
            best = BeastPoint(
                x=float(x), y=float(y), row=int(r), col=int(c),
                elev_m=elev, dist_m=d, bearing_deg=brg, score=float(s),
            )
    if out_rejected_cross is not None and best_cross is not None:
        out_rejected_cross.append(best_cross)
    return best


def _ridge_point_to_beast(dem: DEM, rp) -> BeastPoint | None:
    """dragon_vein.RidgePoint → BeastPoint。"""
    if rp is None:
        return None
    x, y = dem.xy(int(rp.row), int(rp.col))
    return BeastPoint(
        x=float(x), y=float(y),
        row=int(rp.row), col=int(rp.col),
        elev_m=float(rp.elev_m),
        dist_m=float(rp.dist_m),
        bearing_deg=float(rp.bearing_deg),
        score=float(getattr(rp, "score", 0.0) or 0.0),
    )


def detect_four_beasts(
    dem: DEM,
    center_row: int | None = None,
    center_col: int | None = None,
    facing: float | None = None,
    water=None,
    facing_override: float | None = None,
    dragon_vein=None,
    primary_dragon=None,
    use_incoming_vein: bool = True,
) -> FourBeastsPositions:
    """识别四象位置（峦头：先龙后向，不绑绝对东/南/西/北）。

    Args:
        dem: DEM
        center_row, center_col: 穴位点（候选穴）；None 时用 DEM 中心
        facing: 显式朝向（度）；None 则由主龙/地形推断
        water: 可选 WaterNetwork
        facing_override: 与 facing 同义（兼容旧调用）
        dragon_vein: 全量龙脉结果
        primary_dragon: 主来龙；有则 **坐靠来龙、祖在龙源**
        use_incoming_vein: 是否启用来龙取祖/父（默认开）

    Returns:
        FourBeastsPositions（含 facing_method / meta 可解释字段）
    """
    h, w = dem.data.shape
    if center_row is None:
        center_row = h // 2
    if center_col is None:
        center_col = w // 2
    center_row = int(np.clip(center_row, 0, h - 1))
    center_col = int(np.clip(center_col, 0, w - 1))

    # —— 朝向 / 坐向 ——
    # 峦头：坐靠来龙（向龙源），向为坐之对。绝对东/南/西/北只是结果，不是先验。
    # 优先级：用户指定 > 主来龙定坐 > 明堂/背高推断
    method = "default_south"
    facing_val = 180.0
    sit = 0.0

    if facing_override is not None:
        facing_val = float(facing_override) % 360.0
        method = "user_override"
        sit = (facing_val + 180.0) % 360.0
    elif facing is not None:
        facing_val = float(facing) % 360.0
        method = "user_facing"
        sit = (facing_val + 180.0) % 360.0
    else:
        # 尽量拿到主来龙
        if use_incoming_vein and primary_dragon is None and dragon_vein is not None:
            try:
                from engine.core.dragon_vein import select_primary_dragon
                primary_dragon = select_primary_dragon(
                    dem, water=water, dragon_vein=dragon_vein,
                )
            except Exception:
                primary_dragon = None

        if use_incoming_vein and primary_dragon is not None:
            from engine.core.dragon_vein import _bearing_rc, _m_per_px_dem
            mpx0, mpy0 = _m_per_px_dem(dem)
            # 坐 = 穴 → 龙源（祖来处）；向 = 坐+180（面明堂一侧）
            sit = _bearing_rc(
                center_row, center_col,
                int(primary_dragon.source_row), int(primary_dragon.source_col),
                mpx0, mpy0,
            )
            facing_val = (sit + 180.0) % 360.0
            method = "sit_to_dragon_source"
            # 软标签：前方有水则记 face_water，但绝不因侧向近水改坐改向
            try:
                nw = _nearest_water_bearing(dem, center_row, center_col, water)
                if nw is not None and _angle_diff(nw[0], facing_val) < 70.0:
                    method = "dragon_sit_face_water"
            except Exception:
                pass
        else:
            facing_val, method = infer_facing(
                dem, center_row, center_col, water=water,
            )
            sit = (facing_val + 180.0) % 360.0

    left_dir = (facing_val + 270.0) % 360.0   # 青龙 = 面朝之左
    right_dir = (facing_val + 90.0) % 360.0   # 白虎 = 面朝之右
    front_dir = facing_val
    back_dir = sit

    # 局部峰值（禁边缘：加宽，防图缘假峰）
    peaks = _local_maxima_mask(dem.data, size=5)
    peaks[center_row, center_col] = False
    margin = max(8, min(h, w) // 28)
    peaks[:margin, :] = False
    peaks[-margin:, :] = False
    peaks[:, :margin] = False
    peaks[:, -margin:] = False

    occupied: list[tuple[int, int]] = [(center_row, center_col)]
    cand_elev = float(dem.data[center_row, center_col])

    # 局尺度 L + 比例距离窗（非绝对米硬限）
    mpx, mpy = _m_per_px(dem)
    cell_m = float(max(0.5 * (mpx + mpy), 1.0))
    half_diag_m = 0.5 * float(np.hypot(w * mpx, h * mpy))
    dist_cap = max(noise_floor_m(cell_m) * 30.0, half_diag_m * 0.95)

    # 来龙脊长、前水距 → 参与 L
    ridge_len_m: float | None = None
    if primary_dragon is not None:
        try:
            ordered = getattr(primary_dragon, "ordered_coords", None)
            if ordered is not None and len(ordered) >= 2:
                # 弧长近似
                tot = 0.0
                for i in range(1, len(ordered)):
                    r0, c0 = int(ordered[i - 1][0]), int(ordered[i - 1][1])
                    r1, c1 = int(ordered[i][0]), int(ordered[i][1])
                    tot += float(np.hypot((r1 - r0) * mpy, (c1 - c0) * mpx))
                if tot > cell_m * 5:
                    ridge_len_m = tot
            if ridge_len_m is None:
                lp = getattr(primary_dragon, "length_m", None) or getattr(
                    primary_dragon, "length_proxy_m", None
                )
                if lp is not None and float(lp) > cell_m * 5:
                    ridge_len_m = float(lp)
        except Exception:
            ridge_len_m = None

    water_front_m: float | None = None
    if water is not None and not getattr(water, "empty", True):
        try:
            cx, cy = dem.xy(center_row, center_col)
            d_w = float(water.distance_to_nearest_m(cx, cy))
            if np.isfinite(d_w) and d_w < half_diag_m * 2:
                water_front_m = d_w
        except Exception:
            water_front_m = None

    elev_rng = None
    try:
        finite_e = dem.data[np.isfinite(dem.data)]
        if finite_e.size > 20:
            elev_rng = float(np.nanpercentile(finite_e, 95) - np.nanpercentile(finite_e, 5))
    except Exception:
        elev_rng = None

    _wins = beast_distance_windows(
        half_diag_m,
        cell_m=cell_m,
        ridge_length_m=ridge_len_m,
        water_front_m=water_front_m,
        elev_range_m=elev_rng,
        dist_cap_m=dist_cap,
    )
    L_site = float(_wins["L"])
    XW_DIST = (
        float(_wins["xuanwu"][0]),
        min(float(_wins["xuanwu"][1]), dist_cap),
    )
    SZ_DIST = (
        float(_wins["shaozu"][0]),
        min(float(_wins["shaozu"][1]), dist_cap),
    )
    BH_DIST = (
        float(_wins["baihu"][0]),
        min(float(_wins["baihu"][1]), dist_cap),
    )
    SIDE_DIST = (
        float(_wins["side"][0]),
        min(float(_wins["side"][1]), dist_cap),
    )
    ZQ_DIST = (
        float(_wins["zhuque"][0]),
        min(float(_wins["zhuque"][1]), dist_cap),
    )
    XW_MIN_HARD = float(_wins["xuanwu_min_hard_m"])
    BH_MIN_HARD = float(_wins["baihu_min_hard_m"])
    SZ_MIN_HARD = float(_wins["shaozu_min_hard_m"])
    # 高差甜区上限随 L 缓变（小局收、大局放）
    XW_DH_SWEET = (
        15.0,
        float(np.clip(0.06 * L_site, 40.0, 220.0)),
    )

    # 水体禁选（四象）：仅禁真水面 + 极窄噪声边。
    # 穴心仍用 WATER_BAN_BUFFER_M(60m)；砂点允许对岸近岸案/护砂，视线可跨水。
    _beast_ban_m = float(BEAST_WATER_BAN_M)
    _wd, water_ban = water_distance_rasters(
        dem, water, ban_buffer_m=_beast_ban_m,
    )
    # 真水面（0 缓冲）供「隔水」软加分；勿与穴心宽禁带混用
    _wd0, water_surface = water_distance_rasters(dem, water, ban_buffer_m=0.0)
    if water_surface is None or not np.any(water_surface):
        # 无栅格水面时用极窄 dist 近似表面
        water_surface = np.isfinite(_wd0) & (_wd0 < max(1.0, min(mpx, mpy) * 0.6))
    # 来龙/少祖同岸判定：水面膨胀 1–2 像元（OSM 线河仅 1px 宽，直线易漏）
    water_surface_dragon = water_surface
    try:
        from scipy.ndimage import binary_dilation
        if water_surface is not None and np.any(water_surface):
            # ~1 像元 ≈ 30m COP30；再加 dist 阈值兜底
            water_surface_dragon = binary_dilation(
                water_surface.astype(bool), iterations=2,
            )
            if np.isfinite(_wd0).any():
                water_surface_dragon = water_surface_dragon | (
                    np.isfinite(_wd0) & (_wd0 < max(35.0, min(mpx, mpy) * 1.2))
                )
    except Exception:
        water_surface_dragon = water_surface
    # 双保险：距水 < 极窄缓冲仍禁（防栅格化漏河心）
    if np.isfinite(_wd).any():
        water_ban = water_ban | (np.isfinite(_wd) & (_wd < _beast_ban_m))
    peaks = peaks & (~water_ban)

    sel_kw = dict(
        border_margin=margin,
        max_dist_cap_m=dist_cap,
        forbid_mask=water_ban,
        # 朱雀隔水加分仍用细水面；来龙同岸用膨胀水面（见 sel_back）
        water_surface_mask=water_surface,
    )
    # 玄武/少祖：靠山须同岸（水界龙止硬拒跨水）；朱雀/龙虎可隔水加分
    rejected_cross_peaks: list[BeastPoint] = []
    sel_back = dict(
        sel_kw,
        water_surface_mask=water_surface_dragon,  # 膨胀水面，防单像元河漏检
        cross_water_bonus=0.0,
        cross_water_penalty=float(CROSS_WATER_PENALTY_BACK),
        reject_cross_water=bool(REJECT_CROSS_WATER_BACK),
        out_rejected_cross=rejected_cross_peaks,
    )
    sel_zq = dict(sel_kw, cross_water_bonus=float(CROSS_WATER_BONUS_ZHUQUE))
    sel_side = dict(sel_kw, cross_water_bonus=float(CROSS_WATER_BONUS_SIDE))

    def _reject_water_bp(bp: BeastPoint | None) -> BeastPoint | None:
        """几何兜底：真水面或贴图缘则丢弃。"""
        if bp is None:
            return None
        if bp.row < margin or bp.row >= h - margin or bp.col < margin or bp.col >= w - margin:
            return None
        if water is None or getattr(water, "empty", True):
            return bp
        try:
            if water.intersects(bp.x, bp.y, buffer_m=_beast_ban_m):
                return None
        except Exception:
            pass
        return bp

    def _reject_cross_water_bp(
        bp: BeastPoint | None,
        *,
        role: str = "back",
    ) -> BeastPoint | None:
        """水界龙止：穴→峰穿主水面则否决（来龙/少祖/玄武）。

        被否决的对岸峰记入 rejected_cross_peaks，可供朝砂展示。
        """
        if bp is None:
            return None
        if not REJECT_CROSS_WATER_BACK:
            return bp
        if water_surface_dragon is None or not np.any(water_surface_dragon):
            return bp
        if _segment_hits_water(
            water_surface_dragon, center_row, center_col, bp.row, bp.col,
            n_samples=48, end_skip=0.08,
        ):
            rejected_cross_peaks.append(bp)
            return None
        return bp

    def _too_close(a: BeastPoint | None, b: BeastPoint | None, min_m: float = 80.0) -> bool:
        if a is None or b is None:
            return False
        return float(np.hypot(
            (a.row - b.row) * mpy, (a.col - b.col) * mpx
        )) < min_m

    # ------------------------------------------------------------------
    # 1–2. 玄武 / 少祖
    #   正法：有主来龙 → 坐靠龙源、少祖在源、父母近穴（不先面东再找西峰）
    #   回退：坐向扇区峰
    # ------------------------------------------------------------------
    xw: BeastPoint | None = None
    sz: BeastPoint | None = None
    vein_meta: dict[str, Any] = {"used": False, "method": "none"}
    xw_on_ridge = False
    sz_on_ridge = False

    if use_incoming_vein:
        try:
            from engine.core.dragon_vein import (
                beasts_from_primary_dragon,
                select_incoming_vein,
            )

            vein = None
            if primary_dragon is not None:
                vein = beasts_from_primary_dragon(
                    dem, center_row, center_col, primary_dragon,
                    forbid_mask=water_ban,
                    xuanwu_dist=XW_DIST,
                    shaozu_dist=SZ_DIST,
                    water=water,
                    water_surface=water_surface_dragon,
                )
                # 用主龙精修坐向/朝向（祖峰方向 = 坐）
                pm = vein.meta or {}
                if pm.get("sit_deg") is not None:
                    sit = float(pm["sit_deg"]) % 360.0
                    facing_val = float(pm.get("facing_deg", (sit + 180.0) % 360.0)) % 360.0
                    method = "dragon_source_sit"
                    left_dir = (facing_val + 270.0) % 360.0
                    right_dir = (facing_val + 90.0) % 360.0
                    front_dir = facing_val
                    back_dir = sit
            if vein is None or (vein.xuanwu is None and vein.shaozu is None):
                ridge_lines = None
                ridge_mask_dv = None
                if dragon_vein is not None:
                    ridge_lines = getattr(dragon_vein, "ridge_lines", None) or None
                    ridge_mask_dv = getattr(dragon_vein, "ridge_mask", None)
                vein = select_incoming_vein(
                    dem, center_row, center_col, sit, facing_deg=facing_val,
                    peaks_mask=peaks,
                    forbid_mask=water_ban,
                    water_surface=water_surface_dragon,
                    ridge_lines=ridge_lines,
                    ridge_mask=ridge_mask_dv,
                    xuanwu_dist=XW_DIST,
                    shaozu_dist=SZ_DIST,
                    sector_half=max(SECTOR_HALF_BACK, SECTOR_HALF_SHAOZU),
                )

            vein_meta = {
                "used": True,
                "method": vein.method,
                "score": round(float(vein.score), 3),
                "incoming_azimuth_deg": (
                    round(vein.incoming_azimuth_deg, 1)
                    if vein.incoming_azimuth_deg is not None else None
                ),
                "sit_align_deg": (
                    round(vein.sit_align_deg, 1)
                    if vein.sit_align_deg is not None else None
                ),
                "downhill_ok": bool(vein.downhill_ok),
                "detail": vein.meta,
                "theory": "坐靠来龙；少祖龙源；不绑绝对方位",
            }
            xw = _reject_cross_water_bp(
                _reject_water_bp(_ridge_point_to_beast(dem, vein.xuanwu)),
                role="xuanwu",
            )
            sz = _reject_cross_water_bp(
                _reject_water_bp(_ridge_point_to_beast(dem, vein.shaozu)),
                role="shaozu",
            )
            if xw is not None:
                xw_on_ridge = True
            if sz is not None:
                sz_on_ridge = True
            else:
                vein_meta.setdefault("shaozu_dropped", None)
                if vein.shaozu is not None and sz is None:
                    vein_meta["shaozu_dropped"] = "cross_water_same_bank"
            if xw is None and vein.xuanwu is not None:
                vein_meta["xuanwu_dropped"] = "cross_water_same_bank"
            # 脊上结果仍须过比例硬窗：贴身玄武/伪祖丢弃（相对 L，非绝对米）
            if xw is not None and xw.dist_m < XW_MIN_HARD * 0.90:
                vein_meta["xuanwu_dropped"] = "too_close_hard"
                xw = None
                xw_on_ridge = False
            if xw is not None and sz is not None:
                if (
                    sz.dist_m < SZ_MIN_HARD * 0.85
                    or sz.dist_m < xw.dist_m * SHAOZU_XUANWU_DIST_RATIO
                ):
                    sz = None
                    sz_on_ridge = False
                    vein_meta["shaozu_dropped"] = "not_farther_than_xuanwu"
            # 祖定坐：有少祖则坐向对齐少祖
            if sz is not None and method.startswith("dragon"):
                sit = float(sz.bearing_deg) % 360.0
                facing_val = (sit + 180.0) % 360.0
                left_dir = (facing_val + 270.0) % 360.0
                right_dir = (facing_val + 90.0) % 360.0
                front_dir = facing_val
                back_dir = sit
        except Exception as exc:
            vein_meta = {"used": False, "method": "error", "error": str(exc)}
            xw, sz = None, None

    # 玄武扇区回退：硬 min = 比例窗下界（×L），优先窗中段
    if xw is None:
        xw = _select_peak_in_sector(
            dem, center_row, center_col, back_dir, SECTOR_HALF_BACK,
            XW_DIST, peaks, occupied=occupied,
            min_elev_above_cand=12.0,
            max_elev_above_cand=200.0,
            elev_mode="sweet",
            dh_sweet=XW_DH_SWEET,
            weight_elev=1.2, weight_dist=1.6,
            hard_min_dist_m=XW_MIN_HARD * 0.92,
            soft_min_frac=0.90,
            **sel_back,
        )
        if xw is None:
            xw = _select_peak_in_sector(
                dem, center_row, center_col, back_dir, SECTOR_HALF_BACK + 12,
                (XW_MIN_HARD * 0.85, min(XW_DIST[1] * 1.25, dist_cap)),
                peaks, occupied=occupied,
                min_elev_above_cand=8.0,
                elev_mode="sweet",
                dh_sweet=XW_DH_SWEET,
                weight_elev=1.1, weight_dist=1.4,
                hard_min_dist_m=XW_MIN_HARD * 0.85,
                soft_min_frac=0.90,
                **sel_back,
            )
        if xw is None:
            # 最后一档：仍守比例下限的 75%，禁止邻元贴穴
            _lo_fb = max(XW_MIN_HARD * 0.75, noise_floor_m(cell_m) * 5)
            xw = _select_peak_in_sector(
                dem, center_row, center_col, back_dir, SECTOR_HALF_BACK + 22,
                (_lo_fb, min(XW_DIST[1] * 1.5, dist_cap)), peaks, occupied=occupied,
                min_elev_above_cand=5.0,
                elev_mode="higher",
                weight_elev=1.0, weight_dist=1.2,
                hard_min_dist_m=_lo_fb,
                soft_min_frac=0.95,
                **sel_back,
            )
        xw = _reject_cross_water_bp(_reject_water_bp(xw), role="xuanwu")
        if xw is not None and xw.dist_m < XW_MIN_HARD * 0.75:
            xw = None
        xw_on_ridge = False

    if xw:
        occupied.append((xw.row, xw.col))

    xw_elev = xw.elev_m if xw else cand_elev + 30.0

    # 少祖：远于玄武（比例）、宜更高、贴坐向；禁止贴穴伪祖
    xw_dist = float(xw.dist_m) if xw is not None else max(XW_DIST[0], XW_MIN_HARD)
    sz_hi = float(SZ_DIST[1])
    sz_lo = min(
        sz_hi * 0.95,
        max(float(SZ_DIST[0]), xw_dist * SHAOZU_XUANWU_DIST_RATIO, SZ_MIN_HARD),
    )
    sz_half = min(SECTOR_HALF_SHAOZU, 42.0)
    if sz is not None and xw is not None:
        if (
            sz.dist_m < xw.dist_m * SHAOZU_XUANWU_DIST_RATIO
            or sz.dist_m < SZ_MIN_HARD * 0.80
        ):
            vein_meta["shaozu_dropped"] = "too_close_to_xuanwu_or_hole"
            sz = None
            sz_on_ridge = False
    # 若少祖方位偏离坐向过大（被扫到侧砂），丢弃重选
    if sz is not None and _angle_diff(sz.bearing_deg, back_dir) > 48.0:
        vein_meta["shaozu_dropped"] = "off_sit_axis"
        sz = None
        sz_on_ridge = False
    # 少祖宜高于玄武
    if sz is not None and xw is not None and sz.elev_m < xw.elev_m - 15.0:
        vein_meta["shaozu_dropped"] = "lower_than_xuanwu"
        sz = None
        sz_on_ridge = False
    if sz is None:
        sz = _select_peak_in_sector(
            dem, center_row, center_col, back_dir, sz_half,
            (sz_lo, sz_hi), peaks, occupied=occupied,
            prefer_higher_than=xw_elev,
            min_elev_above_cand=12.0,
            weight_elev=1.45, weight_dist=0.70,
            hard_min_dist_m=sz_lo,
            soft_min_frac=0.95,
            **sel_back,
        )
        if sz is None:
            sz = _select_peak_in_sector(
                dem, center_row, center_col, back_dir, min(58.0, sz_half + 14.0),
                (sz_lo * 0.92, sz_hi),
                peaks, occupied=occupied,
                prefer_higher_than=xw_elev,
                min_elev_above_cand=8.0,
                weight_elev=1.3, weight_dist=0.55,
                hard_min_dist_m=max(sz_lo * 0.90, SZ_MIN_HARD * 0.85),
                soft_min_frac=0.95,
                **sel_back,
            )
        if sz is None:
            # 同岸后半球：宁远、宁高（龙源代理）
            try:
                best_far = None
                best_sc = -1e18
                min_d = max(sz_lo * 0.88, SZ_MIN_HARD * 0.85)
                rs_p, cs_p = np.where(peaks)
                for rr, cc in zip(rs_p.tolist(), cs_p.tolist()):
                    dist = float(np.hypot((rr - center_row) * mpy, (cc - center_col) * mpx))
                    if dist < min_d or dist > sz_hi:
                        continue
                    if not same_bank_as_hole(
                        water_surface_dragon, center_row, center_col, rr, cc,
                    ):
                        continue
                    brg = float(
                        (np.degrees(np.arctan2(
                            (cc - center_col) * mpx,
                            (center_row - rr) * mpy,
                        )) + 360.0) % 360.0
                    )
                    if _angle_diff(brg, back_dir) > 50.0:
                        continue
                    elev = float(dem.data[rr, cc])
                    if elev < max(cand_elev + 8.0, xw_elev - 5.0):
                        continue
                    sc = dist / 450.0 + (elev - cand_elev) / 60.0 + max(0.0, elev - xw_elev) / 40.0
                    if sc > best_sc:
                        best_sc = sc
                        bx, by = dem.xy(rr, cc)
                        best_far = BeastPoint(
                            x=float(bx), y=float(by), row=int(rr), col=int(cc),
                            elev_m=elev, dist_m=dist, bearing_deg=brg, score=float(sc),
                        )
                if best_far is not None:
                    sz = best_far
                    vein_meta["shaozu_from"] = "farthest_same_bank_back_hemisphere"
            except Exception:
                pass
        sz = _reject_cross_water_bp(_reject_water_bp(sz), role="shaozu")
        if sz is not None and xw is not None and sz.dist_m < xw.dist_m * SHAOZU_XUANWU_DIST_RATIO:
            rejected_cross_peaks.append(sz)
            sz = None
            vein_meta["shaozu_dropped"] = "sector_still_too_close"
        sz_on_ridge = False
    # 若与玄武栅格重合，丢弃少祖（宁缺勿贴）
    if sz is not None and xw is not None:
        if abs(sz.row - xw.row) <= PEAK_FOOTPRINT_PX and abs(sz.col - xw.col) <= PEAK_FOOTPRINT_PX:
            sz = None
            sz_on_ridge = False
            vein_meta["shaozu_dropped"] = "coincident_xuanwu"
        elif _too_close(sz, xw, min_m=max(0.12 * L_site, xw.dist_m * 0.55)):
            sz = None
            sz_on_ridge = False
            vein_meta["shaozu_dropped"] = "too_close_xuanwu"
    # 主龙脊上再抢「上游少祖」：源端半段 + 更远 + 更高
    need_ridge_sz = (
        sz is None
        or (xw is not None and sz.dist_m < xw.dist_m * SHAOZU_XUANWU_DIST_RATIO)
        or (sz is not None and xw is not None and sz.elev_m < xw.elev_m - 10.0)
    )
    if need_ridge_sz and primary_dragon is not None:
        try:
            ordered = getattr(primary_dragon, "ordered_coords", None)
            if ordered is not None and len(ordered) >= 5:
                best_i = None
                best_sc = -1e18
                min_d = max(
                    SZ_MIN_HARD * 0.9,
                    (xw.dist_m * SHAOZU_XUANWU_DIST_RATIO) if xw else SZ_DIST[0],
                )
                n_ord = len(ordered)
                # 优先源端 55%（来龙源头）
                for i, rc in enumerate(ordered):
                    rr, cc = int(rc[0]), int(rc[1])
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    src_frac = i / max(n_ord - 1, 1)
                    if src_frac > 0.60:
                        continue  # 近入首段不作少祖
                    if water_ban is not None and water_ban.shape == (h, w) and water_ban[rr, cc]:
                        continue
                    if not same_bank_as_hole(
                        water_surface_dragon, center_row, center_col, rr, cc,
                    ):
                        continue
                    dist = float(np.hypot((rr - center_row) * mpy, (cc - center_col) * mpx))
                    if dist < min_d or dist > sz_hi:
                        continue
                    elev = float(dem.data[rr, cc])
                    if not np.isfinite(elev) or elev < cand_elev + 5.0:
                        continue
                    if xw is not None and elev < xw.elev_m - 8.0:
                        continue
                    src_w = 1.0 - src_frac
                    sc = dist / 600.0 + elev / 90.0 + 2.8 * src_w
                    if xw is not None:
                        sc += max(0.0, elev - xw.elev_m) / 40.0
                    if sc > best_sc:
                        best_sc = sc
                        best_i = (rr, cc, dist, elev)
                # 源端无则放开整脊，仍守距离
                if best_i is None:
                    for i, rc in enumerate(ordered):
                        rr, cc = int(rc[0]), int(rc[1])
                        if not (0 <= rr < h and 0 <= cc < w):
                            continue
                        if water_ban is not None and water_ban.shape == (h, w) and water_ban[rr, cc]:
                            continue
                        if not same_bank_as_hole(
                            water_surface_dragon, center_row, center_col, rr, cc,
                        ):
                            continue
                        dist = float(np.hypot((rr - center_row) * mpy, (cc - center_col) * mpx))
                        if dist < min_d or dist > sz_hi:
                            continue
                        elev = float(dem.data[rr, cc])
                        if not np.isfinite(elev) or elev < cand_elev + 5.0:
                            continue
                        src_w = 1.0 - i / max(n_ord - 1, 1)
                        sc = dist / 700.0 + elev / 100.0 + 1.5 * src_w
                        if sc > best_sc:
                            best_sc = sc
                            best_i = (rr, cc, dist, elev)
                if best_i is not None:
                    rr, cc, dist, elev = best_i
                    bx, by = dem.xy(rr, cc)
                    brg = float(
                        (np.degrees(np.arctan2(
                            (cc - center_col) * mpx,
                            (center_row - rr) * mpy,
                        )) + 360.0) % 360.0
                    )
                    sz = BeastPoint(
                        x=float(bx), y=float(by), row=rr, col=cc,
                        elev_m=elev, dist_m=dist, bearing_deg=brg, score=float(best_sc),
                    )
                    sz_on_ridge = True
                    vein_meta["shaozu_from"] = "primary_ridge_upstream_same_bank"
        except Exception as _e:
            vein_meta["shaozu_upstream_err"] = str(_e)

    if sz:
        occupied.append((sz.row, sz.col))

    # 3. 朱雀——前方案/朝，宜低于靠山；隔水 + 视线开阔（Viewshed）
    sel_zq_vs = dict(sel_zq, viewshed_bonus=0.45)
    zq = _select_peak_in_sector(
        dem, center_row, center_col, front_dir, SECTOR_HALF_FRONT,
        ZQ_DIST,
        peaks, occupied=occupied,
        prefer_lower_than=xw_elev * 0.95 + cand_elev * 0.05,
        elev_mode="lower",
        weight_elev=0.4, weight_dist=1.1, **sel_zq_vs,
    )
    if zq is None:
        zq = _select_peak_in_sector(
            dem, center_row, center_col, front_dir, SECTOR_HALF_FRONT + 10,
            (max(ZQ_DIST[0] * 0.7, noise_floor_m(cell_m) * 5), min(ZQ_DIST[1] * 1.3, dist_cap)),
            peaks, occupied=occupied,
            elev_mode="higher",
            weight_elev=0.2, weight_dist=1.0, **sel_zq_vs,
        )
    zq = _reject_water_bp(zq)
    if zq:
        occupied.append((zq.row, zq.col))

    def _on_spine_axis(bp: BeastPoint | None, *, min_off_deg: float = 42.0) -> bool:
        """是否落在坐-向主轴上（与祖/玄/雀共线）——龙虎不得占主轴。"""
        if bp is None:
            return False
        if _angle_diff(bp.bearing_deg, back_dir) < min_off_deg:
            return True
        if _angle_diff(bp.bearing_deg, front_dir) < min_off_deg:
            return True
        if sz is not None and _angle_diff(bp.bearing_deg, sz.bearing_deg) < min_off_deg:
            return True
        if xw is not None and _angle_diff(bp.bearing_deg, xw.bearing_deg) < min_off_deg:
            return True
        return False

    # 4. 青龙（左）——须在左扇区，禁止与祖/玄共线；忌贴穴
    side_lo, side_hi = float(SIDE_DIST[0]), float(SIDE_DIST[1])
    ql_half = min(SECTOR_HALF_SIDE, 40.0)
    ql = _select_peak_in_sector(
        dem, center_row, center_col, left_dir, ql_half,
        (side_lo, side_hi), peaks, occupied=occupied,
        min_elev_above_cand=0.0,
        elev_mode="higher",
        weight_elev=1.0, weight_dist=1.0,
        hard_min_dist_m=max(side_lo * 0.85, noise_floor_m(cell_m) * 5),
        soft_min_frac=0.90,
        **sel_side,
    )
    ql = _reject_water_bp(ql)
    if ql is not None and _on_spine_axis(ql):
        occupied.append((ql.row, ql.col))
        ql2 = _select_peak_in_sector(
            dem, center_row, center_col, left_dir, ql_half + 8.0,
            (side_lo, side_hi), peaks, occupied=occupied,
            min_elev_above_cand=0.0,
            elev_mode="higher",
            weight_elev=1.0, weight_dist=1.0,
            hard_min_dist_m=max(side_lo * 0.85, noise_floor_m(cell_m) * 5),
            soft_min_frac=0.90,
            **sel_side,
        )
        ql2 = _reject_water_bp(ql2)
        ql = ql2 if (ql2 is not None and not _on_spine_axis(ql2)) else None
        if ql is None:
            vein_meta["qinglong_dropped"] = "collinear_with_shaozu_xuanwu"
    if ql:
        occupied.append((ql.row, ql.col))

    # 5. 白虎（右）——驯俯；硬下限防贴穴；不得与祖玄共线
    bh_lo, bh_hi = float(BH_DIST[0]), float(BH_DIST[1])
    bh_min_hard = max(BH_MIN_HARD, bh_lo * 0.90)
    if ql is not None:
        ql_rel = max(0.0, ql.elev_m - cand_elev)
        max_bh_elev = min(
            ql.elev_m * 0.999,
            cand_elev + BAIHU_QL_ELEV_RATIO * max(ql_rel, 5.0),
        )
        prefer_lt = max_bh_elev
    else:
        max_bh_elev = cand_elev + 40.0
        prefer_lt = max_bh_elev

    bh_half = min(SECTOR_HALF_SIDE, 40.0)
    bh = _select_peak_in_sector(
        dem, center_row, center_col, right_dir, bh_half,
        (bh_lo, bh_hi), peaks, occupied=occupied,
        prefer_lower_than=prefer_lt,
        max_elev_abs=max_bh_elev,
        elev_mode="lower",
        weight_elev=1.1, weight_dist=1.35,
        hard_min_dist_m=bh_min_hard,
        soft_min_frac=0.92,
        **sel_side,
    )
    if bh is None:
        bh = _select_peak_in_sector(
            dem, center_row, center_col, right_dir, bh_half + 10,
            (bh_min_hard, min(side_hi, dist_cap)), peaks, occupied=occupied,
            prefer_lower_than=prefer_lt,
            max_elev_abs=max_bh_elev if ql is not None else None,
            elev_mode="higher",
            weight_elev=0.9, weight_dist=1.15,
            hard_min_dist_m=bh_min_hard * 0.95,
            soft_min_frac=0.92,
            **sel_side,
        )
    bh = _reject_water_bp(bh)
    if bh is not None and bh.dist_m < BH_MIN_HARD * 0.85:
        bh = None
        vein_meta["baihu_dropped"] = "too_close_hard"
    if bh is not None and _on_spine_axis(bh):
        occupied.append((bh.row, bh.col))
        bh2 = _select_peak_in_sector(
            dem, center_row, center_col, right_dir, bh_half + 12,
            (bh_min_hard, min(side_hi, dist_cap)), peaks, occupied=occupied,
            prefer_lower_than=prefer_lt,
            max_elev_abs=max_bh_elev if ql is not None else None,
            elev_mode="higher",
            weight_elev=0.9, weight_dist=1.1,
            hard_min_dist_m=bh_min_hard * 0.95,
            soft_min_frac=0.92,
            **sel_side,
        )
        bh2 = _reject_water_bp(bh2)
        bh = bh2 if (bh2 is not None and not _on_spine_axis(bh2)) else None
    # 白虎不得与玄武/少祖重合：分离阈值 = 0.08L（比例）
    _sep_bh = max(noise_floor_m(cell_m) * 4, 0.08 * L_site)
    if bh is not None and (_too_close(bh, xw, _sep_bh) or _too_close(bh, sz, _sep_bh)):
        if xw:
            occupied.append((xw.row, xw.col))
        if sz:
            occupied.append((sz.row, sz.col))
        bh2 = _select_peak_in_sector(
            dem, center_row, center_col, right_dir, SECTOR_HALF_SIDE + 12,
            (bh_min_hard, min(side_hi, dist_cap)), peaks, occupied=occupied,
            prefer_lower_than=prefer_lt,
            max_elev_abs=max_bh_elev if ql is not None else None,
            elev_mode="higher",
            weight_elev=0.9, weight_dist=1.1,
            hard_min_dist_m=bh_min_hard,
            soft_min_frac=0.92,
            **sel_side,
        )
        bh2 = _reject_water_bp(bh2)
        if (
            bh2 is not None
            and not _too_close(bh2, xw, _sep_bh * 0.9)
            and not _too_close(bh2, sz, _sep_bh * 0.9)
            and bh2.dist_m >= BH_MIN_HARD * 0.85
        ):
            bh = bh2
        else:
            bh = None
    if bh:
        occupied.append((bh.row, bh.col))

    def _xy(bp: BeastPoint | None) -> tuple[float, float] | None:
        return (bp.x, bp.y) if bp else None

    def _bp_meta(
        bp: BeastPoint | None, *, on_ridge: bool | None = None
    ) -> dict[str, Any] | None:
        if bp is None:
            return None
        out = {
            "x": bp.x, "y": bp.y,
            "row": bp.row, "col": bp.col,
            "elev_m": round(bp.elev_m, 2),
            "dist_m": round(bp.dist_m, 1),
            "bearing_deg": round(bp.bearing_deg, 1),
            "score": round(bp.score, 3),
        }
        if on_ridge is not None:
            out["on_ridge"] = bool(on_ridge)
        return out

    center_xy = _rowcol_to_xy(dem, center_row, center_col)

    # 主轴：少祖 → 玄武 → 穴 → 朱雀（方位一致性检查）
    axis_ok = True
    if sz and xw:
        # 少祖应大致在玄武外侧（更远）
        if sz.dist_m < xw.dist_m * 0.9:
            axis_ok = False
        # 共线：少祖与玄武相对穴方位不宜大折
        if _angle_diff(sz.bearing_deg, xw.bearing_deg) > 50.0:
            axis_ok = False

    # 来龙方位：优先 vein；否则从少祖/玄武指向穴（龙气走向）
    # _bearing_deg(dx_east, dy_north)；row 向下增大 → dy_north = (r_from - r_to) * mpy
    incoming_az = vein_meta.get("incoming_azimuth_deg")
    if incoming_az is None:
        src = sz or xw
        if src is not None:
            incoming_az = _bearing_deg(
                (center_col - src.col) * mpx,
                (src.row - center_row) * mpy,
            )
    sit_align = None
    if incoming_az is not None:
        sit_align = round(_angle_diff(float(incoming_az), facing_val), 1)

    # 对岸被拒高峰 → 朝砂候选（隔水可朝不可祖）
    chaoshan_bp = None
    if rejected_cross_peaks:
        # 取距穴较远、较高者
        chaoshan_bp = max(
            rejected_cross_peaks,
            key=lambda p: (p.elev_m, p.dist_m),
        )
        # 若已是朱雀则不重复
        if zq is not None and abs(chaoshan_bp.row - zq.row) <= 2 and abs(
            chaoshan_bp.col - zq.col
        ) <= 2:
            chaoshan_bp = None

    meta = {
        "cand_elev_m": round(cand_elev, 2),
        "facing_deg": round(facing_val, 2),
        "sit_deg": round(sit, 2),
        "facing_method": method,
        "axis_consistent": axis_ok,
        "incoming_vein": vein_meta,
        "incoming_azimuth_deg": (
            round(float(incoming_az), 1) if incoming_az is not None else None
        ),
        "incoming_face_align_deg": sit_align,
        "same_bank_rule": "水界龙止：少祖/玄武须与穴同岸；对岸峰改标朝砂",
        "reject_cross_water_back": bool(REJECT_CROSS_WATER_BACK),
        "chaoshan_across_water": _bp_meta(chaoshan_bp),
        "cross_water_rejected_n": len(rejected_cross_peaks),
        "beasts": {
            "shaozu": _bp_meta(sz, on_ridge=sz_on_ridge if sz else None),
            "xuanwu": _bp_meta(xw, on_ridge=xw_on_ridge if xw else None),
            "zhuque": _bp_meta(zq),
            "qinglong": _bp_meta(ql),
            "baihu": _bp_meta(bh),
        },
        "params_m": {
            "mode": "relative_scale",
            "L_site_m": round(L_site, 1),
            "scale": _wins.get("scale"),
            "xuanwu": XW_DIST,
            "zhuque": ZQ_DIST,
            "side": SIDE_DIST,
            "baihu": BH_DIST,
            "shaozu": SZ_DIST,
            "frac": {
                "xuanwu": XUANWU_FRAC,
                "shaozu": SHAOZU_FRAC,
                "baihu": BAIHU_FRAC,
                "side": SIDE_FRAC,
                "zhuque": ZHUQUE_FRAC,
            },
            "shaozu_xw_ratio": float(SHAOZU_XUANWU_DIST_RATIO),
            "xuanwu_min_hard_m": float(XW_MIN_HARD),
            "baihu_min_hard_m": float(BH_MIN_HARD),
            "shaozu_min_hard_m": float(SZ_MIN_HARD),
            "beast_water_ban_m": _beast_ban_m,
            "cross_water_bonus_zhuque": float(CROSS_WATER_BONUS_ZHUQUE),
            "cross_water_bonus_side": float(CROSS_WATER_BONUS_SIDE),
            "cross_water_penalty_back": float(CROSS_WATER_PENALTY_BACK),
            "aoi_half_diag_m": round(half_diag_m, 1),
            "cell_m": round(cell_m, 2),
        },
    }

    return FourBeastsPositions(
        shaozu=_xy(sz),
        xuanwu=_xy(xw),
        zhuque=_xy(zq),
        qinglong=_xy(ql),
        baihu=_xy(bh),
        center=center_xy,
        facing=float(facing_val),
        sit=float(sit),
        facing_method=method,
        meta=meta,
    )


def _auto_sample_step(h: int, w: int, sample_step: int, max_samples: int | None) -> int:
    """兼容旧 API；生气场路径已全幅矢量化，一般不再依赖采样步长。"""
    step = max(1, int(sample_step))
    if max_samples is None or max_samples <= 0:
        return step
    n = max(1, (h // step) * (w // step))
    if n <= max_samples:
        return step
    s = int(np.ceil(np.sqrt((h * w) / float(max_samples))))
    return max(step, s)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """稳定 sigmoid，避免 overflow。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _water_distance_plateau(
    dist_m: np.ndarray,
    *,
    d_lo: float = 180.0,
    d_hi: float = 900.0,
    d_far: float = 2800.0,
    near_floor: float = 0.22,
    far_floor: float = 0.28,
) -> np.ndarray:
    """得水宽平台：明堂腹地高分，避免贴岸光环。

    - [d_lo, d_hi] → 1.0（有情界水甜区，默认自 ~180m 起）
    - < d_lo：从 near_floor 抬升到 1（贴岸明显低于堂心）
    - > d_hi：缓降到 far_floor
    """
    d = np.asarray(dist_m, dtype=np.float64)
    g = np.full_like(d, far_floor, dtype=np.float64)
    # 近侧肩：ban 外到 d_lo
    near = d < d_lo
    g = np.where(
        near,
        near_floor + (1.0 - near_floor) * np.clip(d / max(d_lo, 1e-6), 0.0, 1.0),
        g,
    )
    # 甜区平台
    g = np.where((d >= d_lo) & (d <= d_hi), 1.0, g)
    # 远侧肩
    mid_far = (d > d_hi) & (d <= d_far)
    t = (d - d_hi) / max(d_far - d_hi, 1e-6)
    g = np.where(mid_far, 1.0 - (1.0 - far_floor) * np.clip(t, 0.0, 1.0), g)
    g = np.where(d > d_far, far_floor, g)
    return np.clip(g, 0.0, 1.0)


def compute_qi_field_layers(
    dem: DEM,
    water=None,
    *,
    tpi_fine_m: float = 80.0,
    tpi_mid_m: float = 400.0,
    enclosure_radius_m: float = 500.0,
    water_opt_m: float = 300.0,
    water_sigma_m: float = 400.0,
    water_lo_m: float = 180.0,
    water_hi_m: float = 900.0,
    water_far_m: float = 2800.0,
    slope_opt_deg: float = 5.0,
    slope_sigma_deg: float = 14.0,
    floor_cangfeng: float = 0.30,
    floor_water: float = 0.18,
    floor_enclosure: float = 0.52,
    floor_stability: float = 0.30,
    mingtang_boost: float = 0.42,
) -> dict[str, np.ndarray]:
    """全矢量生气子场（0–1），方向无关。

    设计目标（对标河湾平坦明堂 / 参考图橙心）::
      - 藏风：中尺度略凹 + **细尺度平台**（大凹中小平）——平台权重大
      - 得水：**宽平台** [water_lo, water_hi]，非 300m 尖峰；贴岸肩低
      - 围合：**开阔平坦优先**（低起伏 + 缓坡），弱化「高墙夹峙」
      - 稳定：0–8° 宜穴，近平给高分
      - 明堂通道：平坦×开阔×有情水距 再乘性抬升（mingtang_boost）
      - 乘性 + 各通道地板，避免单项否决整场

    water_opt_m / water_sigma_m 保留兼容；优先使用 lo/hi/far 平台参数。
    """
    from scipy.ndimage import maximum_filter
    from engine.core.terrain_analysis import (
        compute_slope_aspect,
        tpi as _tpi,
        _is_geographic,
    )

    _ = (water_opt_m, water_sigma_m)  # 兼容旧 weights 键名

    elev = np.asarray(dem.data, dtype=np.float64)
    finite = np.isfinite(elev)
    if not finite.any():
        z = np.zeros_like(elev, dtype=np.float64)
        ban = np.zeros_like(elev, dtype=bool)
        return {
            "cangfeng": z,
            "water": z,
            "enclosure": z,
            "stability": z,
            "qi": z,
            "water_ban": ban,
            "finite": finite,
        }

    slope_arr, _ = compute_slope_aspect(dem)
    tpi_fine = _tpi(dem, radius_m=tpi_fine_m)
    tpi_mid = _tpi(dem, radius_m=tpi_mid_m)
    tpi_fine = np.where(np.isfinite(tpi_fine), tpi_fine, 0.0)
    tpi_mid = np.where(np.isfinite(tpi_mid), tpi_mid, 0.0)

    from scipy.ndimage import uniform_filter

    if _is_geographic(dem.crs):
        m_per_unit = 111000.0
    else:
        m_per_unit = 1.0
    xres_m = abs(dem.resolution[0]) * m_per_unit
    yres_m = abs(dem.resolution[1]) * m_per_unit
    mpp = max(0.5 * (xres_m + yres_m), 1e-6)
    r_px = max(1, int(round(enclosure_radius_m / mpp)))
    # 邻域窗：围合 / 弯内对比
    win = 2 * r_px + 1
    win_small = max(3, 2 * max(1, r_px // 3) + 1)

    # 1) 藏风：中尺度略凹 + 细尺度平台（明堂）；平台权重再提高
    g_basin = _sigmoid(-0.35 * tpi_mid)
    g_platform = np.exp(-(tpi_fine ** 2) / (2.0 * 0.55 ** 2))  # 更贴 |TPI|≈0
    g_cf = 0.25 * g_basin + 0.75 * g_platform
    g_cf = np.clip(np.maximum(g_cf, floor_cangfeng), 0.0, 1.0)

    # 2) 稳定：0–6° 宜穴；近平给高分；陡坡明显衰减
    slope_safe = np.where(np.isfinite(slope_arr), slope_arr, 45.0)
    g_stab = np.exp(
        -((slope_safe - slope_opt_deg) / max(slope_sigma_deg, 1e-6)) ** 2
    )
    g_stab = np.where(slope_safe <= 3.0, np.maximum(g_stab, 0.97), g_stab)
    g_stab = np.where(slope_safe <= 6.0, np.maximum(g_stab, 0.90), g_stab)
    # 陡岸/坡麓（>16°）强压——参考图热力不在山坡
    g_stab = np.where(slope_safe > 16.0, g_stab * 0.38, g_stab)
    g_stab = np.where(slope_safe > 22.0, g_stab * 0.55, g_stab)
    g_stab = np.clip(np.maximum(g_stab, floor_stability), 0.0, 1.0)

    # 3) 得水：宽平台 + 弯内/陆心（距水局部极大 → 半岛心）
    water_dist, water_ban = water_distance_rasters(dem, water)
    has_water = bool(np.isfinite(water_dist).any() and np.any(water_dist < 1e12))
    if has_water:
        d = np.where(np.isfinite(water_dist), water_dist, 1.0e6)
        g_water = _water_distance_plateau(
            d, d_lo=water_lo_m, d_hi=water_hi_m, d_far=water_far_m,
            near_floor=0.20,
        )
        # 贴岸额外肩衰减：禁带外 70–210m 仍压光环，逼热力入堂心
        bank_fade = np.clip((d - 70.0) / 140.0, 0.0, 1.0)
        g_water = g_water * (0.32 + 0.68 * bank_fade)
        # 弯内：距水大于邻域均值 → 离岸、居陆心（河环内侧台地）
        d_land = np.where(water_ban, 0.0, np.clip(d, 0.0, water_hi_m * 1.2))
        d_nb = uniform_filter(d_land, size=win_small, mode="nearest")
        inland_excess = np.clip((d_land - d_nb) / 30.0, 0.0, 1.0)
        # 仅在有情距离带内抬弯内 / 堂心
        in_band = (d_land >= water_lo_m * 0.70) & (d_land <= water_hi_m * 1.2)
        g_inland = np.where(in_band, 0.38 + 0.62 * inland_excess, 0.36)
        g_water = g_water * (0.48 + 0.52 * g_inland)
        g_water = np.where(water_ban, 0.0, np.clip(g_water, 0.0, 1.0))
        g_water = np.where(water_ban, 0.0, np.maximum(g_water, floor_water))
        g_water = np.where(water_ban, 0.0, g_water)
    else:
        g_water = np.full_like(elev, 0.72, dtype=np.float64)
        water_ban = np.zeros_like(elev, dtype=bool)
        d = np.full_like(elev, 500.0)
        in_band = np.ones_like(elev, dtype=bool)

    # 4) 围合：开阔平坦优先（参考图明堂），靠山为辅、弱化高差奖山脚
    fill_val = float(np.nanmedian(elev[finite]))
    elev_f = np.where(finite, elev, fill_val)
    surrounding_max = maximum_filter(elev_f, size=win, mode="nearest")
    surrounding_mean = uniform_filter(elev_f, size=win, mode="nearest")
    relief_max = np.maximum(surrounding_max - elev_f, 0.0)
    # 局部起伏（小窗 std 代理）：明堂腹地应低
    elev_sq = elev_f * elev_f
    local_mean = uniform_filter(elev_f, size=win_small, mode="nearest")
    local_var = np.maximum(
        uniform_filter(elev_sq, size=win_small, mode="nearest") - local_mean ** 2,
        0.0,
    )
    local_std = np.sqrt(local_var)
    g_flat_local = np.exp(-(local_std ** 2) / (2.0 * 6.0 ** 2))  # σ≈6m 内高分
    g_flat_local = np.maximum(g_flat_local, 0.35)

    # 靠山：穴低于邻域平均（背后有高），权重降低——避免热力爬坡
    below_mean = surrounding_mean - elev_f
    g_back = _sigmoid(0.05 * below_mean)
    # 开阔：低 max 高差 + 低局地起伏
    g_open_relief = np.exp(-(np.maximum(relief_max - 18.0, 0.0) ** 2) / (2.0 * 55.0 ** 2))
    g_open_relief = np.maximum(g_open_relief, 0.58)
    g_open_relief = np.where(relief_max > 140.0, g_open_relief * 0.50, g_open_relief)
    g_open = 0.55 * g_open_relief + 0.45 * g_flat_local
    # 合成：开阔平坦为主，靠山为辅；缓坡再抬
    g_enc = 0.32 * g_back + 0.68 * g_open
    g_enc = g_enc * (0.70 + 0.30 * np.exp(-(slope_safe / 12.0) ** 2))
    g_enc = np.clip(np.maximum(g_enc, floor_enclosure), 0.0, 1.0)

    # 5) 明堂通道：平坦平台 × 开阔 × 缓坡 × 有情水距
    g_mt_flat = g_platform * g_flat_local
    g_mt_open = g_open * np.exp(-(slope_safe / 10.0) ** 2)
    if has_water:
        g_mt_water = np.where(in_band, g_water, g_water * 0.75)
    else:
        g_mt_water = g_water
    g_mingtang = np.clip(
        g_mt_flat * (0.40 + 0.60 * g_mt_open) * (0.50 + 0.50 * g_mt_water),
        0.0, 1.0,
    )
    g_mingtang = np.where(water_ban, 0.0, g_mingtang)

    # 乘性融合 + 明堂乘性抬升（参考图：橙心在平坦明堂腹地，非贴岸）
    qi = g_cf * g_water * g_enc * g_stab
    boost = float(np.clip(mingtang_boost, 0.0, 0.60))
    qi = qi * (1.0 - boost + boost * (0.28 + 0.72 * g_mingtang))
    qi = np.where(finite & ~water_ban, qi, 0.0)
    # 平滑：略加大，凝聚单团明堂心（对标参考图大团橙心）
    from scipy.ndimage import gaussian_filter

    qi_s = gaussian_filter(qi, sigma=max(1.0, min(r_px, 10) * 0.50))
    # 保持禁水与无效
    qi = np.where(finite & ~water_ban, qi_s, 0.0)
    # 归一到 [0,1] 保持动态范围
    qmax = float(np.max(qi)) if qi.size else 0.0
    if qmax > 1e-9:
        qi = qi / qmax
    qi = np.clip(qi, 0.0, 1.0)

    return {
        "cangfeng": g_cf.astype(np.float64),
        "water": g_water.astype(np.float64),
        "enclosure": g_enc.astype(np.float64),
        "stability": g_stab.astype(np.float64),
        "mingtang": g_mingtang.astype(np.float64),
        "qi": qi.astype(np.float64),
        "water_ban": water_ban,
        "finite": finite,
    }


def compute_score_grid(
    dem: DEM,
    weights: dict[str, float] | None = None,
    tpi_radius_m: float = 100.0,
    sample_step: int = 4,
    water=None,
    *,
    use_water_form: bool = False,
    max_samples: int | None = 12_000,
    search_radius_m: float = 300.0,
) -> np.ndarray:
    """计算全图生气评分场（热力 + 穴心）。

    生气场（乘性 · 全矢量，方向无关）::

        F(p) = G_藏风(p) × G_得水(p) × G_围合(p) × G_稳定(p)

    - 藏风：中尺度略凹 + 细尺度平台（明堂）
    - 得水：宽距离平台；水面+缓冲硬零
    - 围合：宽高差甜区 + 开阔底分
    - 稳定：缓坡/近平为吉

    输出 0–100（qi×100）；水面/无效像元为 nan。
    穴心 = find_score_peak(场)；四象/少祖在峰值处再 detect。
    """
    _ = (tpi_radius_m, sample_step, use_water_form, max_samples, search_radius_m)
    w = dict(weights or {})
    layers = compute_qi_field_layers(
        dem,
        water,
        tpi_fine_m=float(w.get("tpi_fine_m", 80.0)),
        tpi_mid_m=float(w.get("tpi_mid_m", 400.0)),
        enclosure_radius_m=float(
            w.get("enclosure_radius_m", w.get("search_radius_m", 500.0))
        ),
        water_opt_m=float(w.get("water_opt_m", 300.0)),
        water_sigma_m=float(w.get("water_sigma_m", 400.0)),
        water_lo_m=float(w.get("water_lo_m", 120.0)),
        water_hi_m=float(w.get("water_hi_m", 900.0)),
        water_far_m=float(w.get("water_far_m", 2800.0)),
        slope_opt_deg=float(w.get("slope_opt_deg", 5.0)),
        slope_sigma_deg=float(w.get("slope_sigma_deg", 14.0)),
        floor_cangfeng=float(w.get("floor_cangfeng", 0.32)),
        floor_water=float(w.get("floor_water", 0.22)),
        floor_enclosure=float(w.get("floor_enclosure", 0.48)),
        floor_stability=float(w.get("floor_stability", 0.32)),
    )
    qi = layers["qi"]
    ban = layers["water_ban"]
    finite = layers["finite"]
    score = qi * 100.0
    score = np.where(finite & ~ban, score, np.nan)
    return score.astype(np.float64)


def smooth_score_field(
    score_grid: np.ndarray,
    *,
    smooth_sigma: float | None = None,
) -> tuple[np.ndarray, float]:
    """返回 (平滑后评分场, sigma)。无效区保持 nan。"""
    from scipy.ndimage import gaussian_filter

    h, w = score_grid.shape
    valid = np.isfinite(score_grid)
    if smooth_sigma is None:
        smooth_sigma = max(1.2, min(h, w) / 80.0)
    sigma = float(smooth_sigma)
    if not valid.any():
        return np.full_like(score_grid, np.nan, dtype=np.float64), sigma
    filled = np.where(valid, score_grid.astype(np.float64), 0.0)
    soft = gaussian_filter(filled, sigma=sigma)
    # 边界：无效像元不扩散有效分
    weight = gaussian_filter(valid.astype(np.float64), sigma=sigma)
    soft = soft / np.maximum(weight, 1e-9)
    soft = np.where(valid, soft, np.nan)
    return soft, sigma


def find_score_peak(
    score_grid: np.ndarray,
    *,
    smooth_sigma: float | None = None,
) -> tuple[int, int, float] | None:
    """在评分场上取平滑后的最高点，作为「穴」中心。

    与 render_score_grid 共用 smooth_score_field，保证橙心 ≡ 场评最高点。

    Returns:
        (row, col, smoothed_score) 或 None（全无效）
    """
    if score_grid is None or score_grid.size == 0:
        return None
    soft, _ = smooth_score_field(score_grid, smooth_sigma=smooth_sigma)
    valid = np.isfinite(soft)
    if not valid.any():
        return None
    filled = np.where(valid, soft, -np.inf)
    pr, pc = np.unravel_index(int(np.argmax(filled)), filled.shape)
    pr, pc = int(pr), int(pc)
    return pr, pc, float(soft[pr, pc])


def score_peak_xy(
    dem: DEM,
    score_grid: np.ndarray,
    *,
    smooth_sigma: float | None = None,
) -> tuple[tuple[int, int], list[float], float] | None:
    """评分场峰值 → (row,col) + 世界坐标 [x,y] + 分数。"""
    peak = find_score_peak(score_grid, smooth_sigma=smooth_sigma)
    if peak is None:
        return None
    pr, pc, sc = peak
    try:
        x, y = dem.xy(pr, pc)
    except Exception:
        return None
    return (pr, pc), [float(x), float(y)], float(sc)
