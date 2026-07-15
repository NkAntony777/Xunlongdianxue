"""Site scale L and fractional beast distance windows.

Moved from four_beasts_detect. Re-exported for compatibility.
"""
from __future__ import annotations

from typing import Any

import numpy as np

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


