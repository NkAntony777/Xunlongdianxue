"""案山「圆净不破碎」、高度型、朝拱型综合评估。

传统理据：
  - 案山：要「圆净、端正、有情」，破碎、嵓岩、侧走、斜飞皆凶
  - 案山高度：宜低于父母山（《葬书》：「案山要高过父母山之上，必主凶」，
    现代多数派：案山 = 父母山的 1/3 ~ 1/2 为吉，过高欺主）
  - 案山拱向：朝穴则为朝案，背穴则为反背

度量：
  1. 破碎度 fragmentation：案山范围内沟谷密度（局部低值计数 / 区域面积）
  2. 圆净度 roundness：4-connected 碎片数 / 掩膜面积，1 越圆净
  3. 高度比 height_ratio：anshan_top / parents_top，> 2/3 衰减，> 1 罚
  4. 朝拱 facing_score：案山外形质心在穴之前方向（z 比 ~ 180°），侧走者衰减

返回 AnshanQuality:
  - score 0-100 总分
  - height_ratio 高
  - fragmentation 0-1，越高越破碎
  - roundness 0-1，越高越圆净
  - is_eligible 是否可作案山
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter, label, gaussian_filter, binary_dilation

from engine.io.dem import DEM


@dataclass
class AnshanQuality:
    score: float
    height_ratio: float
    fragmentation: float
    roundness: float
    top_elev_m: float
    mean_slope: float
    facing_score: float
    is_eligible: bool
    notes: str


# 阈值（米制）
H_RATIO_GOOD = (0.25, 0.55)    # 案山高度/父母山高度的最佳区间
H_RATIO_MAX = 0.85              # 超过此值高度衰减
H_RATIO_FORBID = 1.05           # 超过此值案山"欺主"
FRAG_GOOD_MAX = 0.08           # 破碎度 ≤ 此值优秀
FRAG_BAD_MIN = 0.35            # 破碎度 ≥ 此值凶
ROUND_GOOD_MIN = 0.55          # 圆净度 ≥ 此值优秀
ROUND_BAD_MAX = 0.25           # 圆净度 ≤ 此值破碎
SLOPE_GOOD_MAX = 25.0          # 案山顶部平均坡度上界
SLOPE_BAD_MIN = 38.0           # 过陡为嵓岩


def _m_per_px(dem: DEM) -> tuple[float, float]:
    from engine.core.terrain_analysis import _is_geographic

    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs) and dem.bounds is not None:
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        return xres * 111_000.0 * cos_lat, yres * 111_000.0
    if _is_geographic(dem.crs):
        # 地理坐标但缺 bounds：默认中纬度 cos_lat = 0.79
        return xres * 111_000.0 * 0.79, yres * 111_000.0
    return xres, yres


def _bearing_deg(dx_m: float, dy_m: float) -> float:
    return float((np.degrees(np.arctan2(dx_m, dy_m)) + 360.0) % 360.0)


def score_anshan_quality(
    dem: DEM,
    center_row: int,
    center_col: int,
    anshan_mask: np.ndarray,
    *,
    parents_top_m: float | None = None,
    facing_deg: float = 180.0,
) -> AnshanQuality:
    """评估案山质量。

    Args:
        dem: DEM
        center_row, center_col: 穴位
        anshan_mask: 案山区域掩膜（bool）
        parents_top_m: 父母山顶高程（用于高度比）；None 则用相对穴位为基线
        facing_deg: 朝向，180=南（默认坐北朝南）
    """
    h, w = dem.data.shape
    if not anshan_mask.any():
        return AnshanQuality(
            score=40.0, height_ratio=0.0, fragmentation=1.0,
            roundness=0.0, top_elev_m=float("nan"), mean_slope=0.0,
            facing_score=0.0, is_eligible=False,
            notes="无案山掩膜",
        )

    mpx, mpy = _m_per_px(dem)
    elev = np.where(np.isfinite(dem.data), dem.data, np.nan)
    cand_elev = float(dem.data[center_row, center_col])

    # 案山顶高
    sub = np.where(anshan_mask & np.isfinite(elev), elev, -np.inf)
    top_elev = float(np.nanmax(sub))
    rel_top = top_elev - cand_elev
    if parents_top_m is None:
        parents_top = cand_elev + rel_top * 3.0  # 退而求其次
    else:
        parents_top = float(parents_top_m)
    parents_rel = parents_top - cand_elev
    if parents_rel < 1.0:
        parents_rel = 1.0
    h_ratio = float(rel_top / parents_rel)

    # 破碎度：以"地形位置"低值（鞍/沟）密度估算
    # 局部极低洼点（5% 分位）= 沟谷节点
    blur = gaussian_filter(elev, sigma=3.0)
    valid = np.isfinite(blur) & anshan_mask
    if valid.any():
        local_min = uniform_filter(
            np.where(valid, blur, np.inf), size=5, mode="nearest"
        )
        vals = local_min[valid]
        thr = float(np.nanpercentile(vals, 25))
        gulch_count = int(((local_min <= thr) & valid).sum())
        frag = gulch_count / max(int(valid.sum()), 1)
    else:
        frag = 1.0
    frag = float(np.clip(frag * 4.0, 0.0, 1.0))  # 归一化

    # 圆净度：连通块数 / 总像元（碎片越多越破碎）
    labeled, n = label(anshan_mask)
    area = max(int(anshan_mask.sum()), 1)
    n_segments = int(n) if n else 0
    roundness = float(np.clip(1.0 - (n_segments - 1) / max(area / 50.0, 1.0), 0.0, 1.0))

    # 平均坡度
    from engine.core.terrain_analysis import compute_slope_aspect
    slope_arr, _ = compute_slope_aspect(dem)
    mean_slope = float(np.nanmean(slope_arr[anshan_mask])) if anshan_mask.any() else 0.0

    # 案山朝向一致（案山质心落在穴前向扇区）
    rs, cs = np.where(anshan_mask)
    dx_m = (cs.mean() - center_col) * mpx
    dy_m = (center_row - rs.mean()) * mpy
    centroid_az = _bearing_deg(dx_m, dy_m)
    diff = abs(((centroid_az - facing_deg + 180) % 360) - 180)
    facing_score = float(np.clip(1.0 - diff / 90.0, 0.0, 1.0))

    # 综合分
    score = 60.0
    note = []

    # 高度
    lo, hi = H_RATIO_GOOD
    if lo <= h_ratio <= hi:
        score += 18
        note.append(f"高比 {h_ratio:.2f} 甜区")
    elif h_ratio <= H_RATIO_MAX:
        score += 8
        note.append(f"高比 {h_ratio:.2f} 适中")
    elif h_ratio <= H_RATIO_FORBID:
        score -= 10
        note.append(f"高比 {h_ratio:.2f} 偏高")
    else:
        score -= 28
        note.append(f"高比 {h_ratio:.2f} 欺主凶")

    # 破碎度
    if frag <= FRAG_GOOD_MAX:
        score += 12
        note.append("破碎度低，圆净")
    elif frag <= FRAG_BAD_MIN:
        score -= 6
        note.append("破碎度中等")
    else:
        score -= 22
        note.append("破碎度高，凶")

    # 圆净度
    if roundness >= ROUND_GOOD_MIN:
        score += 8
        note.append("圆净度好")
    elif roundness >= ROUND_BAD_MAX:
        note.append("圆净度一般")
    else:
        score -= 8
        note.append("圆净度差")

    # 坡度
    if mean_slope <= SLOPE_GOOD_MAX:
        score += 6
        note.append("坡度舒缓")
    elif mean_slope >= SLOPE_BAD_MIN:
        score -= 12
        note.append(f"过陡 {mean_slope:.0f}°, 嵓岩嫌疑")

    # 朝拱
    score += 12 * facing_score
    if facing_score < 0.5:
        note.append("侧走/未朝穴")

    score = float(np.clip(score, 0.0, 100.0))
    eligible = bool(
        score >= 55 and h_ratio <= H_RATIO_MAX and frag <= FRAG_BAD_MIN
        and mean_slope < SLOPE_BAD_MIN
    )

    return AnshanQuality(
        score=float(round(score, 1)),
        height_ratio=float(round(h_ratio, 3)),
        fragmentation=float(round(frag, 3)),
        roundness=float(round(roundness, 3)),
        top_elev_m=float(round(top_elev, 2)),
        mean_slope=float(round(mean_slope, 1)),
        facing_score=float(round(facing_score, 3)),
        is_eligible=eligible,
        notes="; ".join(note),
    )
