"""Yaoxia (narrows) along ridges."""
from __future__ import annotations

from typing import Any

import numpy as np

from engine.io.dem import DEM
from engine.core.dragon.types import RidgeLine


def rel_drop_default(dem: DEM, ridge: "RidgeLine") -> float:
    """估算"蜂腰"判定阈值：脊线最高-最低（米）的 50%，最小 2 m。"""
    elevs = []
    for r, c in ridge.coords:
        if 0 <= r < dem.data.shape[0] and 0 <= c < dem.data.shape[1]:
            e = float(dem.data[r, c])
            if np.isfinite(e):
                elevs.append(e)
    if not elevs:
        return 5.0
    drop = max(elevs) - min(elevs)
    return max(2.0, drop * 0.5)


def find_yaoxia(
    ridges: list[RidgeLine],
    dem: DEM,
    *,
    neck_width_m: float = 60.0,
    min_narrowing_ratio: float = 0.55,
) -> list[dict[str, Any]]:
    """识别「蜂腰鹤膝」过峡点。

    在山脊线（ridge）上寻找"突然变窄"的位置：
      - 该点上下游一段长度内，脊线平均宽度显著小于其他段宽度
      - 峡两侧高低差不悬殊（已"脱煞"），即过峡处坡度缓

    返回 list[{ridge_idx, pos_idx, pos_xy, neck_width_m, ...}]
    """
    yaoxia: list[dict[str, Any]] = []
    if not ridges:
        return yaoxia

    mpx = dem.resolution[0] if dem.crs is None else dem.resolution[0]
    from engine.core.terrain_analysis import _is_geographic
    if _is_geographic(dem.crs):
        m_per_deg = 111000.0
        mpx = mpx * m_per_deg
        mpy = dem.resolution[1] * m_per_deg
    else:
        mpy = dem.resolution[1]
    xres, yres = mpx, mpy

    for r_idx, ridge in enumerate(ridges):
        n = len(ridge.coords)
        if n < 20:
            continue

        # 沿脊线计算每点的"横向脊宽"：从脊线点出发，沿垂直脊方向双侧爬，
        # 找到降高度至 threshold 处的距离 ×2 = 局部"等高线半宽"×2。
        elevs: list[float] = []
        widths_m: list[float] = []   # 横向距离（米）
        window = 5
        max_walk_px = 15  # 最大搜索距离（像素）

        # 脊线局部切向量
        for i in range(n):
            r, c = ridge.coords[i]
            if not (0 <= r < dem.data.shape[0] and 0 <= c < dem.data.shape[1]):
                elevs.append(float("nan"))
                widths_m.append(float("nan"))
                continue
            peak_elev = float(dem.data[r, c])  # 【修复】局部声明
            elevs.append(peak_elev)

            # 切向：用 i ± window 中心差分
            il = max(0, i - 1)
            ir = min(n - 1, i + 1)
            rL, cL = ridge.coords[il]
            rR, cR = ridge.coords[ir]
            tx = (cR - cL) * xres
            ty = (rL - rR) * yres
            t_norm = np.hypot(tx, ty)
            if t_norm < 1e-9:
                widths_m.append(float("nan"))
                continue
            tx /= t_norm
            ty /= t_norm

            # 法向（旋转 90°）
            nx = -ty
            ny = tx

            # 双向爬升至"半山腰"——取全局脊线 50% 高差作为阈值
            target_drop = rel_drop_default(dem, ridge)

            left_dist = 0.0
            right_dist = 0.0
            for d in range(1, max_walk_px + 1):
                px_r = int(round(r - ny * d))
                px_c_l = int(round(c + nx * d))  # 左侧
                px_c_r = int(round(c - nx * d))  # 右侧
                if not (0 <= px_r < dem.data.shape[0]):
                    continue
                if 0 <= px_c_l < dem.data.shape[1]:
                    e_l = float(dem.data[px_r, px_c_l])
                    if e_l < peak_elev - target_drop and left_dist == 0:
                        left_dist = d * (xres + yres) / 2.0
                if 0 <= px_c_r < dem.data.shape[1]:
                    e_r = float(dem.data[px_r, px_c_r])
                    if e_r < peak_elev - target_drop and right_dist == 0:
                        right_dist = d * (xres + yres) / 2.0
                if left_dist and right_dist:
                    break
            # 总横向距离：左右之和（如一侧未找到则用单侧 ×2）
            if left_dist and right_dist:
                widths_m.append(left_dist + right_dist)
            elif left_dist:
                widths_m.append(left_dist * 2.0)
            elif right_dist:
                widths_m.append(right_dist * 2.0)
            else:
                widths_m.append(float(max_walk_px * (xres + yres)))

        widths_arr = np.array(widths_m, dtype=np.float64)
        elev_arr = np.array(elevs, dtype=np.float64)

        if not np.isfinite(widths_arr).any():
            continue

        # 全脊线宽度中位数
        med_w = float(np.nanmedian(widths_arr))
        if med_w <= 0:
            continue

        for i in range(window, n - window):
            local_w = widths_arr[i]
            if not np.isfinite(local_w) or local_w <= 0:
                continue
            ratio = local_w / med_w
            if ratio > min_narrowing_ratio:
                continue
            # 峡点位置（地理坐标）
            r_pt, c_pt = ridge.coords[i]
            x_pt, y_pt = dem.xy(r_pt, c_pt)
            # 峡两侧高点差（不超过 60 m 为宜，过大的说明未"脱煞"）
            left_idx = max(0, i - 30)
            right_idx = min(n - 1, i + 30)
            L_elev = float(np.nanmax(elev_arr[left_idx:i + 1]))
            R_elev = float(np.nanmax(elev_arr[i:right_idx + 1]))
            dh = abs(L_elev - R_elev)
            yaoxia.append({
                "ridge_idx": r_idx,
                "pos_idx": i,
                "row": int(r_pt),
                "col": int(c_pt),
                "x": float(x_pt),
                "y": float(y_pt),
                "neck_width_m": round(float(local_w), 1),
                "median_width_m": round(med_w, 1),
                "narrow_ratio": round(float(ratio), 3),
                "side_relief_diff_m": round(dh, 1),
            })
    return yaoxia


