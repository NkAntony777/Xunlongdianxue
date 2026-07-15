"""晕土（晕界）粗略识别。

传统理据（《葬书·乘气章》）：
  - 穴点核心证据是「晕土」：三色土模糊交界，气温微升，晨雾凝结最重。
  - DEM 代理方案：
      * 穴点在「局部坡度」的最小值带（地形三焦区）
      * 局部曲率近 0（不在尖锐脊/谷）
      * TPI 极小但 TWI 极大（缓坡上的局部最低汇水点）

实现：
  - 三个栅格（TPI + TWI + slope）的局部最小值带交集
  - 对每个候选穴给 0-100 的"晕土可能性"
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter, generic_filter

from engine.io.dem import DEM


@dataclass
class HaloSoil:
    """晕土判读结果。"""

    score: float
    has_min_slope: bool     # 局部坡度是否最小
    has_min_curvature: bool # 局部曲率近 0
    has_max_twi: bool       # 局部 TWI 是否为高值
    soil_temperature_proxy: float  # DEM 代理：穴处局部高程稳定度
    notes: str


def _m_per_px(dem: DEM) -> tuple[float, float]:
    from engine.core.terrain_analysis import _is_geographic

    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs):
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        return xres * 111_000.0 * cos_lat, yres * 111_000.0
    return xres, yres


def score_halo_soil(
    dem: DEM,
    center_row: int,
    center_col: int,
    *,
    search_radius_m: float = 30.0,
    tpi_arr: np.ndarray | None = None,
    twi_arr: np.ndarray | None = None,
) -> HaloSoil:
    """穴点附近晕土可能性打分。

    Args:
        dem: DEM
        center_row, center_col: 穴位
        search_radius_m: 邻域半径（米）
        tpi_arr / twi_arr: 可复用
    """
    h, w = dem.data.shape
    if not (0 <= center_row < h and 0 <= center_col < w):
        return HaloSoil(50.0, False, False, False, 0.0, "中心点越界")

    mpx, mpy = _m_per_px(dem)
    r_px = max(3, int(round(search_radius_m / max(mpx, mpy))))
    r0 = max(0, center_row - r_px)
    r1 = min(h, center_row + r_px + 1)
    c0 = max(0, center_col - r_px)
    c1 = min(w, center_col + r_px + 1)
    sub = dem.data[r0:r1, c0:c1]
    if sub.size == 0:
        return HaloSoil(50.0, False, False, False, 0.0, "邻域为空")

    cand_elev = float(dem.data[center_row, center_col])

    # 1) 局部坡度（图幅级 slope_arr 中取）
    from engine.core.terrain_analysis import compute_slope_aspect
    slope_arr, _ = compute_slope_aspect(dem)
    local_slope_window = uniform_filter(slope_arr, size=2 * r_px + 1, mode="reflect")
    local_at = float(local_slope_window[center_row, center_col])
    has_min_slope = (
        local_at <= float(np.nanpercentile(local_slope_window, 30))
    )

    # 2) 局部曲率（Laplacian 近似）
    blur = uniform_filter(np.where(np.isfinite(sub), sub, cand_elev - 100),
                          size=3, mode="reflect")
    lap = np.pad(blur, 1, mode="edge")
    curvature = (4 * lap[1:-1, 1:-1]
                 - lap[0:-2, 1:-1] - lap[2:, 1:-1]
                 - lap[1:-1, 0:-2] - lap[1:-1, 2:])
    abs_curv = np.abs(curvature)
    curv_center = float(abs_curv[
        min(max(0, center_row - r0), abs_curv.shape[0] - 1),
        min(max(0, center_col - c0), abs_curv.shape[1] - 1),
    ])
    has_min_curv = curv_center <= float(np.nanpercentile(abs_curv, 35))

    # 3) TWI 高值（若传入）
    has_max_twi = False
    if twi_arr is not None:
        local_twi_window = uniform_filter(
            np.where(np.isfinite(twi_arr), twi_arr, -np.inf),
            size=2 * r_px + 1, mode="reflect",
        )
        if (np.isfinite(local_twi_window[center_row, center_col])
                and local_twi_window[center_row, center_col] >= float(
                    np.nanpercentile(local_twi_window[np.isfinite(local_twi_window)], 65))):
            has_max_twi = True

    # 综合打分
    score = 50.0
    note = []
    if has_min_slope:
        score += 18
        note.append("局部坡度最小带")
    if has_min_curv:
        score += 12
        note.append("曲率近 0，缓变带")
    if has_max_twi:
        score += 10
        note.append("TWI 高，水汽聚")
    # 温度代理：DEM 局地稳定度 = 邻域标准差小
    sub_valid = sub[np.isfinite(sub)]
    if sub_valid.size > 0:
        std = float(np.nanstd(sub_valid))
        rel_std = std / max(cand_elev, 1.0)
        if rel_std < 0.05:
            score += 8
            note.append("邻域高程稳定，地温恒")
    score = float(min(100.0, score))

    # 微调：太陡或太偏离 minimum 反而减分
    if not has_min_slope:
        score -= 5
    return HaloSoil(
        score=score,
        has_min_slope=bool(has_min_slope),
        has_min_curvature=bool(has_min_curv),
        has_max_twi=bool(has_max_twi),
        soil_temperature_proxy=float(np.nanstd(sub_valid) if sub_valid.size > 0 else 0.0),
        notes="; ".join(note) if note else "无明显晕土特征",
    )
