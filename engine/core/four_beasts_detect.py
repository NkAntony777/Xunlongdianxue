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


# —— 米制默认参数（来自调研文档 + 开源对照 research/02_four_beasts/01_*）——
XUANWU_DIST_M = (50.0, 500.0)       # 父母山 / 靠山
XUANWU_DH_SWEET_M = (30.0, 120.0)   # 相对穴高差甜区
ZHUQUE_DIST_M = (200.0, 3000.0)     # 案山–朝山
SIDE_DIST_M = (80.0, 1200.0)        # 龙虎砂（主搜不宜过远）
BAIHU_DIST_M = (80.0, 800.0)        # 白虎更紧：驯俯近砂
SHAOZU_DIST_M = (500.0, 8000.0)     # 少祖更远
SECTOR_HALF_BACK = 40.0             # 玄武扇区半角
SECTOR_HALF_FRONT = 45.0            # 朱雀扇区半角
SECTOR_HALF_SIDE = 50.0             # 龙虎扇区半角
SECTOR_HALF_SHAOZU = 55.0           # 少祖扇区半角
PEAK_FOOTPRINT_PX = 3               # 局部峰值最小间距（像素）
WATER_BAN_BUFFER_M = 60.0           # 水面+缓冲硬禁穴（候选/场评）
# 四象砂点：只禁真水面 + 极窄噪声边。允许对岸近岸案/护砂（视线可跨水）。
# 切勿与穴心 60 m 同宽——宽禁带会删掉参考图常见的跨河岸砂头。
BEAST_WATER_BAN_M = 12.0
CROSS_WATER_BONUS_ZHUQUE = 0.38     # 朱雀：穴→峰线段穿过水面时的软加分（案山隔水）
CROSS_WATER_BONUS_SIDE = 0.15       # 龙虎：隔水护砂轻微加分（非强制）
BAIHU_QL_ELEV_RATIO = 0.85          # 白虎/青龙高度比上限


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
) -> bool:
    """穴心→砂峰线段是否穿过水面（不含端点邻域，避免贴岸误判）。

    用于软加分「隔水案山/护砂」：点仍须在干地，仅判断视线/格局是否跨水。
    """
    if water_surface is None or not np.any(water_surface):
        return False
    h, w = water_surface.shape
    if not (0 <= r0 < h and 0 <= c0 < w and 0 <= r1 < h and 0 <= c1 < w):
        return False
    # 跳过两端各 ~12% 采样，避免穴/峰本身近岸像元被当成「跨水」
    n = max(8, int(n_samples))
    for i in range(2, n - 1):
        t = i / float(n - 1)
        if t < 0.12 or t > 0.88:
            continue
        r = int(round(r0 + t * (r1 - r0)))
        c = int(round(c0 + t * (c1 - c0)))
        if 0 <= r < h and 0 <= c < w and bool(water_surface[r, c]):
            return True
    return False


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
    viewshed_bonus: float = 0.0,
) -> BeastPoint | None:
    """在方位扇区 + 距离环内，从局部峰值中选综合最优者。

    硬约束：
      - 避开图幅边缘 border_margin 像素（防止 (0,0) 角点假峰）
      - 距离不超过 max_dist_cap_m（通常为 AOI 半对角）
      - forbid_mask（真水面等）上的峰跳过——不禁止对岸干峰
      - elev_mode: higher | lower | sweet（sweet 用 dh_sweet 高差窗）

    软偏好：
      - cross_water_bonus>0 且线段穿过 water_surface_mask 时加分（隔水案/护砂）
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
        & (dist_m >= max(10.0, d_lo * 0.35))
        & (dist_m <= d_search_hi)
        & (ang <= sector_half)
    )
    if not region.any():
        # 回退：扇区内任意高点（仍禁止贴边/真水面）
        region = (
            interior
            & dry
            & np.isfinite(dem.data)
            & (dist_m >= max(10.0, d_lo * 0.35))
            & (dist_m <= d_search_hi)
            & (ang <= sector_half)
        )
        if not region.any():
            return None

    use_cross = (
        cross_water_bonus > 0
        and water_surface_mask is not None
        and water_surface_mask.shape == (h, w)
        and bool(np.any(water_surface_mask))
    )

    best: BeastPoint | None = None
    best_s = -1e18
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

        # 隔水砂：视线可跨水，峰点须已在 dry 区
        if use_cross and _segment_hits_water(
            water_surface_mask, center_row, center_col, r, c
        ):
            s += float(cross_water_bonus)

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

    # AOI 半对角：少祖等不得超出图幅可解释范围
    mpx, mpy = _m_per_px(dem)
    half_diag_m = 0.5 * float(np.hypot(w * mpx, h * mpy))
    # 略小于半对角，避免贴边
    dist_cap = max(800.0, half_diag_m * 0.92)

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
    # 双保险：距水 < 极窄缓冲仍禁（防栅格化漏河心）
    if np.isfinite(_wd).any():
        water_ban = water_ban | (np.isfinite(_wd) & (_wd < _beast_ban_m))
    peaks = peaks & (~water_ban)

    sel_kw = dict(
        border_margin=margin,
        max_dist_cap_m=dist_cap,
        forbid_mask=water_ban,
        water_surface_mask=water_surface,
    )
    # 玄武/少祖：靠山宜同岸，不加跨水激励
    sel_back = dict(sel_kw, cross_water_bonus=0.0)
    sel_zq = dict(sel_kw, cross_water_bonus=float(CROSS_WATER_BONUS_ZHUQUE))
    sel_side = dict(sel_kw, cross_water_bonus=float(CROSS_WATER_BONUS_SIDE))

    def _reject_water_bp(bp: BeastPoint | None) -> BeastPoint | None:
        """几何兜底：真水面或贴图缘则丢弃；允许对岸近岸干峰。"""
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
                    xuanwu_dist=XUANWU_DIST_M,
                    shaozu_dist=(SHAOZU_DIST_M[0], min(SHAOZU_DIST_M[1], dist_cap)),
                    water=water,
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
                    ridge_lines=ridge_lines,
                    ridge_mask=ridge_mask_dv,
                    xuanwu_dist=XUANWU_DIST_M,
                    shaozu_dist=(SHAOZU_DIST_M[0], min(SHAOZU_DIST_M[1], dist_cap)),
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
            xw = _reject_water_bp(_ridge_point_to_beast(dem, vein.xuanwu))
            sz = _reject_water_bp(_ridge_point_to_beast(dem, vein.shaozu))
            if xw is not None:
                xw_on_ridge = True
            if sz is not None:
                sz_on_ridge = True
            if xw is not None and sz is not None and sz.dist_m < xw.dist_m * 0.95:
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

    # 玄武扇区回退
    if xw is None:
        xw = _select_peak_in_sector(
            dem, center_row, center_col, back_dir, SECTOR_HALF_BACK,
            XUANWU_DIST_M, peaks, occupied=occupied,
            min_elev_above_cand=12.0,
            max_elev_above_cand=200.0,
            elev_mode="sweet",
            dh_sweet=XUANWU_DH_SWEET_M,
            weight_elev=1.3, weight_dist=1.4, **sel_back,
        )
        if xw is None:
            xw = _select_peak_in_sector(
                dem, center_row, center_col, back_dir, SECTOR_HALF_BACK + 15,
                (40.0, min(900.0, dist_cap)), peaks, occupied=occupied,
                min_elev_above_cand=5.0,
                elev_mode="sweet",
                dh_sweet=(15.0, 160.0),
                weight_elev=1.0, weight_dist=1.1, **sel_back,
            )
        if xw is None:
            xw = _select_peak_in_sector(
                dem, center_row, center_col, back_dir, SECTOR_HALF_BACK + 25,
                (30.0, min(1500.0, dist_cap)), peaks, occupied=occupied,
                min_elev_above_cand=2.0,
                elev_mode="higher",
                weight_elev=0.9, weight_dist=0.9, **sel_back,
            )
        if xw is None:
            xw = _select_peak_in_sector(
                dem, center_row, center_col, back_dir, 70.0,
                (25.0, min(2000.0, dist_cap)), peaks, occupied=occupied,
                min_elev_above_cand=None,
                elev_mode="higher",
                weight_elev=1.0, weight_dist=0.7, **sel_back,
            )
        xw = _reject_water_bp(xw)
        xw_on_ridge = False

    if xw:
        occupied.append((xw.row, xw.col))

    xw_elev = xw.elev_m if xw else cand_elev + 30.0

    # 少祖扇区回退（脊上未取到时）
    sz_hi = min(SHAOZU_DIST_M[1], dist_cap)
    sz_lo = min(SHAOZU_DIST_M[0], sz_hi * 0.4)
    if sz is None:
        sz = _select_peak_in_sector(
            dem, center_row, center_col, back_dir, SECTOR_HALF_SHAOZU,
            (sz_lo, sz_hi), peaks, occupied=occupied,
            prefer_higher_than=xw_elev,
            min_elev_above_cand=10.0,
            weight_elev=1.3, weight_dist=0.7, **sel_back,
        )
        if sz is None:
            sz = _select_peak_in_sector(
                dem, center_row, center_col, back_dir, 70.0,
                (max(300.0, sz_lo * 0.5), sz_hi),
                peaks, occupied=occupied,
                weight_elev=1.0, weight_dist=0.5, **sel_back,
            )
        sz = _reject_water_bp(sz)
        sz_on_ridge = False
    # 若脊上少祖与玄武过近/被占用，再扇区补一次
    if sz is not None and xw is not None:
        if abs(sz.row - xw.row) <= PEAK_FOOTPRINT_PX and abs(sz.col - xw.col) <= PEAK_FOOTPRINT_PX:
            sz = _select_peak_in_sector(
                dem, center_row, center_col, back_dir, SECTOR_HALF_SHAOZU,
                (sz_lo, sz_hi), peaks, occupied=occupied,
                prefer_higher_than=xw_elev,
                min_elev_above_cand=10.0,
                weight_elev=1.3, weight_dist=0.7, **sel_back,
            )
            sz = _reject_water_bp(sz)
            sz_on_ridge = False
    if sz:
        occupied.append((sz.row, sz.col))

    # 3. 朱雀——前方案/朝，宜低于靠山；隔水 + 视线开阔（Viewshed）
    sel_zq_vs = dict(sel_zq, viewshed_bonus=0.45)
    zq = _select_peak_in_sector(
        dem, center_row, center_col, front_dir, SECTOR_HALF_FRONT,
        (ZHUQUE_DIST_M[0], min(ZHUQUE_DIST_M[1], dist_cap)),
        peaks, occupied=occupied,
        prefer_lower_than=xw_elev * 0.95 + cand_elev * 0.05,
        elev_mode="lower",
        weight_elev=0.4, weight_dist=1.1, **sel_zq_vs,
    )
    if zq is None:
        zq = _select_peak_in_sector(
            dem, center_row, center_col, front_dir, SECTOR_HALF_FRONT + 10,
            (100.0, min(4000.0, dist_cap)), peaks, occupied=occupied,
            elev_mode="higher",
            weight_elev=0.2, weight_dist=1.0, **sel_zq_vs,
        )
    zq = _reject_water_bp(zq)
    if zq:
        occupied.append((zq.row, zq.col))

    # 4. 青龙（左）——可高耸有情；允许隔水护砂
    side_hi = min(SIDE_DIST_M[1], dist_cap)
    ql = _select_peak_in_sector(
        dem, center_row, center_col, left_dir, SECTOR_HALF_SIDE,
        (SIDE_DIST_M[0], side_hi), peaks, occupied=occupied,
        min_elev_above_cand=0.0,
        elev_mode="higher",
        weight_elev=1.0, weight_dist=1.0, **sel_side,
    )
    ql = _reject_water_bp(ql)
    if ql:
        occupied.append((ql.row, ql.col))

    # 5. 白虎（右）——驯俯：硬上限 ≤ 0.85×青龙；可隔水，不落水面
    bh_hi = min(BAIHU_DIST_M[1], dist_cap)
    if ql is not None:
        ql_rel = max(0.0, ql.elev_m - cand_elev)
        # 相对穴高差不超过青龙的 0.85；同时绝对高程不高于青龙
        max_bh_elev = min(
            ql.elev_m * 0.999,
            cand_elev + BAIHU_QL_ELEV_RATIO * max(ql_rel, 5.0),
        )
        prefer_lt = max_bh_elev
    else:
        max_bh_elev = cand_elev + 40.0
        prefer_lt = max_bh_elev

    bh = _select_peak_in_sector(
        dem, center_row, center_col, right_dir, SECTOR_HALF_SIDE,
        (BAIHU_DIST_M[0], bh_hi), peaks, occupied=occupied,
        prefer_lower_than=prefer_lt,
        max_elev_abs=max_bh_elev,
        elev_mode="lower",
        weight_elev=1.1, weight_dist=1.2, **sel_side,
    )
    if bh is None:
        # 回退：略放宽距离，仍守青龙高度比
        bh = _select_peak_in_sector(
            dem, center_row, center_col, right_dir, SECTOR_HALF_SIDE + 10,
            (SIDE_DIST_M[0], min(side_hi, 1500.0)), peaks, occupied=occupied,
            prefer_lower_than=prefer_lt,
            max_elev_abs=max_bh_elev if ql is not None else None,
            elev_mode="higher",
            weight_elev=0.9, weight_dist=1.0, **sel_side,
        )
    bh = _reject_water_bp(bh)
    # 白虎不得与玄武/少祖重合或贴在一起（图缘复用假峰）
    if bh is not None and (_too_close(bh, xw, 120.0) or _too_close(bh, sz, 120.0)):
        # 扩大占用后再选
        if xw:
            occupied.append((xw.row, xw.col))
        if sz:
            occupied.append((sz.row, sz.col))
        bh2 = _select_peak_in_sector(
            dem, center_row, center_col, right_dir, SECTOR_HALF_SIDE + 15,
            (SIDE_DIST_M[0], min(side_hi, 1500.0)), peaks, occupied=occupied,
            prefer_lower_than=prefer_lt,
            max_elev_abs=max_bh_elev if ql is not None else None,
            elev_mode="higher",
            weight_elev=0.9, weight_dist=1.0, **sel_side,
        )
        bh2 = _reject_water_bp(bh2)
        if bh2 is not None and not _too_close(bh2, xw, 100.0) and not _too_close(bh2, sz, 100.0):
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
        "beasts": {
            "shaozu": _bp_meta(sz, on_ridge=sz_on_ridge if sz else None),
            "xuanwu": _bp_meta(xw, on_ridge=xw_on_ridge if xw else None),
            "zhuque": _bp_meta(zq),
            "qinglong": _bp_meta(ql),
            "baihu": _bp_meta(bh),
        },
        "params_m": {
            "xuanwu": XUANWU_DIST_M,
            "zhuque": ZHUQUE_DIST_M,
            "side": SIDE_DIST_M,
            "shaozu": SHAOZU_DIST_M,
            "beast_water_ban_m": _beast_ban_m,
            "cross_water_bonus_zhuque": float(CROSS_WATER_BONUS_ZHUQUE),
            "cross_water_bonus_side": float(CROSS_WATER_BONUS_SIDE),
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
    d_lo: float = 120.0,
    d_hi: float = 900.0,
    d_far: float = 2800.0,
    near_floor: float = 0.42,
    far_floor: float = 0.28,
) -> np.ndarray:
    """得水宽平台：明堂腹地高分，避免 300m 尖峰贴岸光环。

    - [d_lo, d_hi] → 1.0（有情界水甜区）
    - < d_lo：从 near_floor 抬升到 1（贴岸不奖满）
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
    water_lo_m: float = 120.0,
    water_hi_m: float = 900.0,
    water_far_m: float = 2800.0,
    slope_opt_deg: float = 5.0,
    slope_sigma_deg: float = 14.0,
    floor_cangfeng: float = 0.32,
    floor_water: float = 0.22,
    floor_enclosure: float = 0.55,
    floor_stability: float = 0.32,
) -> dict[str, np.ndarray]:
    """全矢量生气子场（0–1），方向无关。

    设计目标（对标河湾明堂，避免「贴水高、平地低」）::
      - 藏风：中尺度略凹 + **细尺度平台**（大凹中小平）
      - 得水：**宽平台** [water_lo, water_hi]，非 300m 尖峰
      - 围合：15–100m 高差甜区，**开阔平地有底分**（不强制 60m）
      - 稳定：0–12° 宜穴，**不惩罚近平**
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

    # 1) 藏风：中尺度略凹 + 细尺度平台（明堂）；平台权重更高
    g_basin = _sigmoid(-0.35 * tpi_mid)
    g_platform = np.exp(-(tpi_fine ** 2) / (2.0 * 0.65 ** 2))
    g_cf = 0.32 * g_basin + 0.68 * g_platform
    g_cf = np.clip(np.maximum(g_cf, floor_cangfeng), 0.0, 1.0)

    # 2) 稳定：0–8° 宜穴；陡坡明显衰减（压山脚碎斑）
    slope_safe = np.where(np.isfinite(slope_arr), slope_arr, 45.0)
    g_stab = np.exp(
        -((slope_safe - slope_opt_deg) / max(slope_sigma_deg, 1e-6)) ** 2
    )
    g_stab = np.where(slope_safe <= 4.0, np.maximum(g_stab, 0.94), g_stab)
    # 陡岸/坡麓（>18°）强压
    g_stab = np.where(slope_safe > 18.0, g_stab * 0.45, g_stab)
    g_stab = np.clip(np.maximum(g_stab, floor_stability), 0.0, 1.0)

    # 3) 得水：宽平台 + 弯内/陆心（距水局部极大 → 半岛心）
    water_dist, water_ban = water_distance_rasters(dem, water)
    has_water = bool(np.isfinite(water_dist).any() and np.any(water_dist < 1e12))
    if has_water:
        d = np.where(np.isfinite(water_dist), water_dist, 1.0e6)
        g_water = _water_distance_plateau(
            d, d_lo=water_lo_m, d_hi=water_hi_m, d_far=water_far_m,
        )
        # 弯内：距水大于邻域均值 → 离岸、居陆心（河环内侧台地）
        d_land = np.where(water_ban, 0.0, np.clip(d, 0.0, water_hi_m * 1.2))
        d_nb = uniform_filter(d_land, size=win_small, mode="nearest")
        inland_excess = np.clip((d_land - d_nb) / 40.0, 0.0, 1.0)
        # 仅在有情距离带内抬弯内
        in_band = (d_land >= water_lo_m * 0.7) & (d_land <= water_hi_m * 1.15)
        g_inland = np.where(in_band, 0.55 + 0.45 * inland_excess, 0.50)
        g_water = g_water * (0.62 + 0.38 * g_inland)
        g_water = np.where(water_ban, 0.0, np.clip(g_water, 0.0, 1.0))
        g_water = np.where(water_ban, 0.0, np.maximum(g_water, floor_water))
        g_water = np.where(water_ban, 0.0, g_water)
    else:
        g_water = np.full_like(elev, 0.72, dtype=np.float64)
        water_ban = np.zeros_like(elev, dtype=bool)

    # 4) 围合：单侧靠山（低于邻域均值）+ 开阔底分，弱化 max 高差奖山脚
    fill_val = float(np.nanmedian(elev[finite]))
    elev_f = np.where(finite, elev, fill_val)
    surrounding_max = maximum_filter(elev_f, size=win, mode="nearest")
    surrounding_mean = uniform_filter(elev_f, size=win, mode="nearest")
    relief_max = np.maximum(surrounding_max - elev_f, 0.0)
    # 靠山：穴低于邻域平均（背后有高），sigmoid 缓变
    below_mean = surrounding_mean - elev_f
    g_back = _sigmoid(0.06 * below_mean)  # 低一点有靠
    # 开阔：不要求高墙；低 max 高差给高开阔分
    g_open = np.exp(-(np.maximum(relief_max - 25.0, 0.0) ** 2) / (2.0 * 70.0 ** 2))
    g_open = np.maximum(g_open, 0.62)
    # 逼压：邻域相对过高
    g_open = np.where(relief_max > 160.0, g_open * 0.55, g_open)
    # 合成：靠山 + 开阔；再用缓坡权重压陡麓
    g_enc = 0.45 * g_back + 0.55 * g_open
    g_enc = g_enc * (0.75 + 0.25 * np.exp(-(slope_safe / 16.0) ** 2))
    g_enc = np.clip(np.maximum(g_enc, floor_enclosure), 0.0, 1.0)

    # 乘性融合
    qi = g_cf * g_water * g_enc * g_stab
    qi = np.where(finite & ~water_ban, qi, 0.0)
    # 轻平滑：凝聚单团明堂心（非遮瑕级大模糊）
    from scipy.ndimage import gaussian_filter

    qi_s = gaussian_filter(qi, sigma=max(0.8, min(r_px, 8) * 0.35))
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
