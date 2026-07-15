"""候选穴识别：TPI / TWI / 形态 / 候选点搜索。

参考:
  - 调研报告 04_acupoint/00_穴位判定模型.md
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter, generic_filter

from engine.core.terrain_analysis import compute_slope_aspect, tpi as _tpi
from engine.io.dem import DEM
from engine.utils.helpers import clamp_score


@dataclass
class AcupointCandidate:
    """候选穴对象。"""

    row: int
    col: int
    x: float
    y: float
    elevation: float
    tpi: float
    twi: float
    form_type: str  # 窝/钳/乳/突
    form_score: int
    local_slope: float


def compute_twi(
    dem: DEM,
    flow_acc: np.ndarray,
    radius_m: float = 50.0,
) -> np.ndarray:
    """Topographic Wetness Index = ln(a / tan(slope))。

    a: 上游汇流累积面积（平方米）= flow_acc * cell_area
    """
    xres, yres = dem.resolution
    cell_area = xres * yres
    a = flow_acc * cell_area
    slope, _ = compute_slope_aspect(dem)
    slope_rad = np.deg2rad(np.clip(slope, 0.5, None))  # 平地设 0.5° 下限，避免除零
    twi = np.log(a / np.tan(slope_rad))
    # 边缘 NaN 处理
    twi[~np.isfinite(twi)] = np.nan
    return twi


# TPI 阈值以 30 m DEM 为标定基准；其它分辨率按 scale = cell_m/30 缩放
TPI_REF_CELL_M = 30.0


def tpi_scale_factor(cell_size_m: float | None) -> float:
    """DEM 像元尺度相对 30 m 的自适应因子。

    高分辨率（5 m）→ scale < 1 → 阈值收紧（局部噪声大）
    低分辨率（90 m）→ scale > 1 → 阈值放宽
    限制在 [0.5, 2.5]。
    """
    if cell_size_m is None or not np.isfinite(cell_size_m) or cell_size_m <= 0:
        return 1.0
    return float(np.clip(cell_size_m / TPI_REF_CELL_M, 0.5, 2.5))


def classify_form(
    tpi_value: float,
    local_slope: float = 0.0,
    *,
    cell_size_m: float | None = None,
) -> str:
    """根据 TPI 与局部坡度判读穴位形态（连续分区，无空隙）。

    | TPI 区间（×scale） | 坡度条件 | 形态 |
    |----------|----------|------|
    | < -1.5 | — | 窝穴 |
    | [-1.5, -0.25) | slope>12° | 钳穴 |
    | [-1.5, -0.25) | slope≤12° | 窝穴（浅窝） |
    | [-0.25, 0.3] | — | 平缓 |
    | (0.3, 1.0] | — | 乳穴 |
    | > 1.0 | — | 突穴 |

    cell_size_m: 像元米制边长；None 时按 30 m 标定（向后兼容）。
    """
    s = tpi_scale_factor(cell_size_m)
    t1, t2, t3, t4 = -1.5 * s, -0.25 * s, 0.3 * s, 1.0 * s
    if tpi_value < t1:
        return "窝穴"
    if tpi_value < t2:
        return "钳穴" if local_slope > 12.0 else "窝穴"
    if tpi_value <= t3:
        return "平缓"
    if tpi_value <= t4:
        return "乳穴"
    return "突穴"


def score_form(
    tpi_value: float,
    form: str,
    *,
    cell_size_m: float | None = None,
) -> int:
    """根据形态与 TPI 打分。

    各形态均有"最适 TPI"区间（随 cell_size_m 缩放）：
    - 窝穴: TPI ∈ [-3, -0.5]，最佳约 -1.5（深而不漏）
    - 突穴: TPI ∈ [1, 3]，最佳约 1.5
    - 乳穴: TPI ∈ [0.3, 1]，最佳约 0.5
    - 钳穴: 形态由 TPI + 局部坡度共同决定
    """
    s = tpi_scale_factor(cell_size_m)
    if form == "窝穴":
        if -3 * s <= tpi_value <= -0.5 * s:
            return clamp_score(95 - abs(tpi_value + 1.5 * s) * 5 / max(s, 0.5))
        if tpi_value < -3 * s:
            return clamp_score(80 + (tpi_value + 3 * s) * 5 / max(s, 0.5))
        return clamp_score(70)
    if form == "钳穴":
        return clamp_score(82 + tpi_value * 8 / max(s, 0.5))
    if form == "突穴":
        if 1 * s <= tpi_value <= 3 * s:
            return clamp_score(90 - abs(tpi_value - 1.5 * s) * 6 / max(s, 0.5))
        if tpi_value > 3 * s:
            return clamp_score(75 - (tpi_value - 3 * s) * 8 / max(s, 0.5))
        return clamp_score(70)
    if form == "乳穴":
        if 0.3 * s <= tpi_value <= 1.0 * s:
            return clamp_score(90 - abs(tpi_value - 0.5 * s) * 8 / max(s, 0.5))
        if tpi_value > 1.0 * s:
            return clamp_score(78 - (tpi_value - 1.0 * s) * 10 / max(s, 0.5))
        return clamp_score(70)
    if form == "平缓":
        return clamp_score(88 - abs(tpi_value) * 25 / max(s, 0.5))
    return clamp_score(50 + tpi_value * 20 / max(s, 0.5))


def score_stability(slope_deg: float) -> int:
    """稳定性评分：坡度越低越稳定。"""
    if slope_deg < 5:
        return 95
    if slope_deg < 10:
        return 85
    if slope_deg < 15:
        return 75
    if slope_deg < 20:
        return 65
    if slope_deg < 30:
        return 50
    return 30


def score_openness(
    dem: DEM,
    slope_arr: np.ndarray | None = None,
    forward_sector_deg: float = 180.0,
    radius_m: float = 500.0,
) -> int:
    """明堂开阔度评分：前向扇区平均坡度越低越开阔。

    默认前向为南（180°），可通过参数改变。
    """
    if slope_arr is None:
        slope_arr, _ = compute_slope_aspect(dem)
    h, w = dem.data.shape
    cy, cx = h // 2, w // 2
    from engine.core.terrain_analysis import _is_geographic
    if _is_geographic(dem.crs):
        m_per_unit = 111000.0
    else:
        m_per_unit = 1.0
    yres, xres = dem.resolution[1] * m_per_unit, dem.resolution[0] * m_per_unit
    yy, xx = np.mgrid[0:h, 0:w]
    dx_m = (xx - cx) * xres
    dy_m = (yy - cy) * yres
    dist_m = np.sqrt(dx_m**2 + dy_m**2)
    bearing = (np.degrees(np.arctan2(dx_m, dy_m)) + 360) % 360
    diff = np.abs(((bearing - forward_sector_deg + 180) % 360) - 180)
    region = (diff <= 90) & (dist_m > 50) & (dist_m <= radius_m) & np.isfinite(dem.data)
    if not region.any():
        return 50
    mean_slope = float(np.nanmean(slope_arr[region]))
    return clamp_score(100 - mean_slope * 4)


def filter_candidates_off_water(
    dem: DEM,
    cands: list[AcupointCandidate],
    water,
    *,
    ban_buffer_m: float = 60.0,
) -> list[AcupointCandidate]:
    """剔除落在水面或缓冲带内的候选（与评分场 water_ban 对齐）。

    栅格 ban（rasterize + 膨胀）为主；几何 intersects 为兜底。
    无水系时原样返回。
    """
    if not cands:
        return []
    if water is None or getattr(water, "empty", True):
        return list(cands)

    from engine.core.four_beasts_detect import water_distance_rasters

    _dist, ban = water_distance_rasters(
        dem, water, ban_buffer_m=float(ban_buffer_m),
    )
    h, w = ban.shape
    kept: list[AcupointCandidate] = []
    for c in cands:
        r, col = int(c.row), int(c.col)
        if 0 <= r < h and 0 <= col < w and ban[r, col]:
            continue
        # 几何兜底：点缓冲内与水系相交则禁
        try:
            if water.intersects(c.x, c.y, buffer_m=float(ban_buffer_m)):
                continue
        except Exception:
            pass
        kept.append(c)
    return kept


def search_candidates(
    dem: DEM,
    flow_acc: np.ndarray | None = None,
    tpi_radius_m: float = 100.0,
    tpi_threshold: float = 0.0,
    max_candidates: int = 30,
    step: int = 1,
    water=None,
    *,
    ban_buffer_m: float = 60.0,
    qi_grid: np.ndarray | None = None,
    qi_min_percentile: float = 60.0,
) -> list[AcupointCandidate]:
    """在整个 DEM 上滑动窗口搜索候选穴。

    Args:
        dem: DEM
        flow_acc: 汇流累积栅格（None 时不计算 TWI）
        tpi_radius_m: TPI 邻域半径
        tpi_threshold: TPI 绝对值阈值（|TPI| < 该值被过滤）
        max_candidates: 最大返回数
        step: 步长（栅格）
        water: 可选水系；提供时剔除水面+缓冲带内点
        ban_buffer_m: 水禁缓冲（米），与场评 WATER_BAN_BUFFER 一致
        qi_grid: 可选生气场（0–100）；提供时只在高 qi 分位内出候选
        qi_min_percentile: qi 阈值分位（默认 top 40%：≥P60）

    Returns:
        候选穴列表，按 form/qi 综合分降序
    """
    # E.5：全无效 DEM → 空列表
    if dem is None or dem.data is None or not np.isfinite(dem.data).any():
        return []

    slope_arr, _ = compute_slope_aspect(dem)
    tpi_arr = _tpi(dem, radius_m=tpi_radius_m)
    if flow_acc is not None and np.any(flow_acc > 0):
        twi_arr = compute_twi(dem, flow_acc)
    else:
        twi_arr = np.zeros_like(dem.data)

    h, w = dem.data.shape
    cands: list[AcupointCandidate] = []

    # 局部坡度（在 tpi_radius 上平滑）
    from scipy.ndimage import uniform_filter as uf
    from engine.core.terrain_analysis import _radius_px, _is_geographic

    # P2：TPI 阈值随 DEM 分辨率自适应
    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs):
        cell_m = float(min(xres, yres) * 111000.0)
    else:
        cell_m = float(min(xres, yres))

    px_radius = _radius_px(dem, tpi_radius_m)
    size = 2 * px_radius + 1
    local_slope = uf(slope_arr, size=size, mode="reflect")

    # 预计算水禁掩膜，循环内 O(1) 查询
    water_ban = None
    if water is not None and not getattr(water, "empty", True):
        from engine.core.four_beasts_detect import water_distance_rasters

        _d, water_ban = water_distance_rasters(
            dem, water, ban_buffer_m=float(ban_buffer_m),
        )

    # qi 阈值：只在热力高分干地出候选
    qi_thr = None
    if qi_grid is not None and qi_grid.shape == dem.data.shape:
        if water_ban is not None:
            valid_qi = np.isfinite(qi_grid) & (~water_ban)
        else:
            valid_qi = np.isfinite(qi_grid)
        if valid_qi.any():
            qi_thr = float(np.nanpercentile(qi_grid[valid_qi], qi_min_percentile))

    for r in range(px_radius, h - px_radius, step):
        for c in range(px_radius, w - px_radius, step):
            if not np.isfinite(dem.data[r, c]):
                continue
            if water_ban is not None and water_ban[r, c]:
                continue
            qi_v = None
            if qi_thr is not None:
                q = float(qi_grid[r, c])
                if not np.isfinite(q) or q < qi_thr:
                    continue
                qi_v = q
            tpi_v = float(tpi_arr[r, c])
            thr = tpi_threshold * tpi_scale_factor(cell_m)
            if abs(tpi_v) < thr:
                continue
            form = classify_form(tpi_v, float(local_slope[r, c]), cell_size_m=cell_m)
            form_score = score_form(tpi_v, form, cell_size_m=cell_m)
            twi_v = float(twi_arr[r, c]) if np.isfinite(twi_arr[r, c]) else 0.0
            # TWI 参与：微地形干湿有情（过湿/过干扣分，中等湿润加分）
            # 典型 TWI 约 0–20+；中带 2–10 为吉（得水不积）
            if flow_acc is not None and twi_v > 0:
                if 2.0 <= twi_v <= 10.0:
                    form_score = int(min(100, form_score + 6))
                elif twi_v > 14.0:
                    form_score = int(max(0, form_score - 8))  # 过湿积水
                elif 0 < twi_v < 1.0:
                    form_score = int(max(0, form_score - 3))  # 过干
            # 有 qi 时：排序主看场分，形态为辅（对齐热力）
            if qi_v is not None:
                form_score = int(round(0.30 * form_score + 0.70 * qi_v))
            x, y = dem.xy(r, c)
            cand = AcupointCandidate(
                row=r,
                col=c,
                x=x,
                y=y,
                elevation=float(dem.data[r, c]),
                tpi=tpi_v,
                twi=twi_v,
                form_type=form,
                form_score=form_score,
                local_slope=float(local_slope[r, c]),
            )
            cands.append(cand)

    # 几何兜底再滤一遍（CRS/栅格化失败时）
    if water is not None and not getattr(water, "empty", True):
        cands = filter_candidates_off_water(
            dem, cands, water, ban_buffer_m=ban_buffer_m,
        )

    # 非极大值抑制：相距 < min_dist_m 的只保留 score 最高的
    cands.sort(key=lambda x: -x.form_score)
    kept: list[AcupointCandidate] = []
    min_dist_m = 200.0
    from engine.core.terrain_analysis import _is_geographic

    geographic = _is_geographic(dem.crs)

    def _dist_m(a: AcupointCandidate, b: AcupointCandidate) -> float:
        if geographic:
            dx = (a.x - b.x) * 111_000 * np.cos(np.radians((a.y + b.y) / 2))
            dy = (a.y - b.y) * 111_000
            return float(np.hypot(dx, dy))
        # 投影坐标：单位为米
        return float(np.hypot(a.x - b.x, a.y - b.y))

    for c in cands:
        too_close = False
        for k in kept:
            if _dist_m(c, k) < min_dist_m:
                too_close = True
                break
        if not too_close:
            kept.append(c)
        if len(kept) >= max_candidates:
            break
    return kept
