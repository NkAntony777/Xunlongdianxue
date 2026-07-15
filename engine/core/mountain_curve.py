"""砂形朝抱 / 反背度量。

传统理据：
  - 「青龙蜿蜒拱揖、白虎驯俯朝抱」—— 砂脊凸侧指向穴为反背，凹侧朝向穴为拱抱。
  - 「弯抱有情 vs 反背无情」 —— 影响龙虎是否有情，是否成局。
  - 高度比合适但砂形反背，仍为凶。

判读思路：
  1. 取穴位 -> 砂山的方位扇区主轴线。
  2. 沿砂山 crest（顶脊）采样点列。
  3. 计算采样列相对穴位的「抱向角」 —— crest 在穴侧法向为 +，反侧为 −。
  4. 加权得「朝抱分」 ∈ [0, 100]。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter
from skimage.morphology import skeletonize

from engine.io.dem import DEM


@dataclass
class EmbraceResult:
    """砂形朝抱度量结果。"""

    score: float            # 0-100 朝抱分
    convex_to_acupoint: bool    # True=砂山凸侧朝穴（反背）；False=凹侧朝穴（拱抱）
    mean_bearing_dev_deg: float # 平均法向偏离（度）
    crest_length_m: float       # 砂山 crest 长度
    n_segments: int             # 砂山顶点采样段数
    notes: str


def _m_per_px(dem: DEM) -> tuple[float, float]:
    from engine.core.terrain_analysis import _is_geographic

    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs):
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        return xres * 111_000.0 * cos_lat, yres * 111_000.0
    return xres, yres


def _bearing_deg(dx_m, dy_m):
    """支持数组输入的方位角计算。"""
    arr = np.degrees(np.arctan2(dx_m, dy_m)) + 360.0
    return float(arr % 360.0) if not hasattr(arr, 'shape') else (arr % 360.0)


def _sector_mask(
    aspect_deg: np.ndarray,
    center_deg: float,
    half_width_deg: float,
) -> np.ndarray:
    diff = np.abs(((aspect_deg - center_deg + 180.0) % 360.0) - 180.0)
    return diff <= half_width_deg


def _extract_crest_points(
    dem: DEM,
    center_row: int,
    center_col: int,
    region_mask: np.ndarray,
    *,
    max_points: int = 64,
    smooth_sigma: float = 1.5,
) -> list[tuple[int, int]]:
    """在 region 内提取脊线点列。

    【修复 C.5】原按列等距切片取最大值，对 NE-SW 走向脊取点稀疏到 1-2，
    进入 fallback 返回 55 占位分。新流程：
      1) Gaussian 平滑 DEM；
      2) skeletonize(region_mask) → 单像素宽脊线；
      3) 从距离穴心最近的脊点出发 BFS，沿 8 邻接访问脊线所有点；
      4) 按弧长等距采样 max_points 个点。
    """
    rs, cs = np.where(region_mask)
    if rs.size == 0:
        return []
    h, w = region_mask.shape

    try:
        skel = skeletonize(region_mask)
    except Exception:
        skel = np.zeros_like(region_mask, dtype=bool)

    skel_pts = np.argwhere(skel)
    if len(skel_pts) < 4:
        # 退化：等距列切（兼容旧逻辑）
        cs_min, cs_max = int(cs.min()), int(cs.max())
        bins = max(4, min(max_points, 24, (cs_max - cs_min) + 1))
        pts: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        for k in range(bins):
            col_lo = cs_min + (cs_max - cs_min) * k // max(bins, 1)
            col_hi = cs_min + (cs_max - cs_min) * (k + 1) // max(bins, 1)
            col_hi = max(col_lo + 1, col_hi)
            col_hi = min(col_hi, region_mask.shape[1])
            col_lo = min(col_lo, region_mask.shape[1] - 1)
            sub_mask = region_mask[:, col_lo:col_hi]
            if sub_mask.size == 0 or not sub_mask.any():
                continue
            idx = np.argmax(np.where(sub_mask, dem.data[:, col_lo:col_hi], -np.inf))
            rr, cc = int(idx // sub_mask.shape[1]), int(idx % sub_mask.shape[1])
            pt = (rr, col_lo + cc)
            if pt in used:
                continue
            used.add(pt)
            pts.append(pt)
        return pts

    # 距离穴心最近的脊点
    dists = np.hypot(
        skel_pts[:, 0] - center_row,
        skel_pts[:, 1] - center_col,
    )
    seed = int(np.argmin(dists))
    start = tuple(int(v) for v in skel_pts[seed])
    skel_set = {tuple(int(v) for v in p) for p in skel_pts.tolist()}

    visited: set[tuple[int, int]] = {start}
    chain: list[tuple[int, int]] = [start]
    cur = start
    while len(chain) < max_points * 3:
        # 8 邻接未访问的脊点
        neighbours: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                p = (cur[0] + dr, cur[1] + dc)
                if p in skel_set and p not in visited:
                    neighbours.append(p)
        if not neighbours:
            # 已走完：选距离 cur 最近的未访问脊点重新开始
            unseen = [p for p in skel_set if p not in visited]
            if not unseen:
                break
            # 找到离 chain 末端最近的下一个未访问点
            cur = min(unseen, key=lambda p: np.hypot(p[0]-cur[0], p[1]-cur[1]))
            visited.add(cur)
            chain.append(cur)
            continue
        # 沿"最接近当前方向"的优先
        if len(chain) >= 2:
            prev = chain[-2]
            tdx = cur[0] - prev[0]
            tdy = cur[1] - prev[1]
            def _score(p: tuple[int, int]) -> float:
                dx, dy = p[0] - cur[0], p[1] - cur[1]
                # 投影到当前方向 + 距离惩罚
                d = np.hypot(dx, dy)
                if d == 0:
                    return -1e9
                return (tdx * dx + tdy * dy) / (d * d)
            cur = max(neighbours, key=_score)
        else:
            cur = neighbours[0]
        visited.add(cur)
        chain.append(cur)

    # 沿 chain 等弧长采样 max_points 个点
    if len(chain) < 4:
        return chain
    step = max(1, len(chain) // max_points)
    sampled = [chain[i] for i in range(0, len(chain), step)][:max_points]
    if sampled[-1] != chain[-1]:
        sampled.append(chain[-1])
    return sampled


def measure_embrace(
    dem: DEM,
    center_row: int,
    center_col: int,
    region_mask: np.ndarray,
    *,
    direction_center_deg: float,
    sector_half_deg: float = 60.0,
    inner_radius_m: float = 30.0,
    outer_radius_m: float = 1500.0,
) -> EmbraceResult:
    """度量砂山对穴的朝抱 / 反背。

    Args:
        dem: DEM
        center_row, center_col: 穴位
        region_mask: 砂山占地区域掩膜
        direction_center_deg: 方位扇区中心（左青龙 270 / 右白虎 90 / 前朱雀 180 / 后玄武 0）
        sector_half_deg: 扇区半角
        inner_radius_m: 离穴最近允许距离
        outer_radius_m: 砂山有效远端
    """
    h, w = dem.data.shape
    if not (0 <= center_row < h and 0 <= center_col < w):
        return EmbraceResult(50.0, False, 0.0, 0.0, 0, "中心点越界")

    mpx, mpy = _m_per_px(dem)
    yy, xx = np.mgrid[0:h, 0:w]
    dx_m = (xx - center_col) * mpx
    dy_m = (center_row - yy) * mpy
    dist_m = np.hypot(dx_m, dy_m)
    bearing = _bearing_deg(dx_m, dy_m)
    elev = dem.data
    cand_elev = float(elev[center_row, center_col])

    # 砂山占地区域：扇区内 + 高于穴 + 距离范围
    sector = _sector_mask(bearing, direction_center_deg, sector_half_deg)
    sand_region = (
        region_mask
        & sector
        & (dist_m >= inner_radius_m)
        & (dist_m <= outer_radius_m)
        & np.isfinite(elev)
        & (elev > cand_elev + 2.0)
    )
    if not sand_region.any():
        return EmbraceResult(50.0, False, 0.0, 0.0, 0, "扇区内无砂山")

    pts = _extract_crest_points(dem, center_row, center_col, sand_region)
    if len(pts) < 3:
        # 不到 3 点的不算有效 crest，用替代度量：砂山整体的几何中心方位角
        rs, cs = np.where(sand_region)
        cx_m = float((cs.mean() - center_col) * mpx)
        cy_m = float((center_row - rs.mean()) * mpy)
        az = _bearing_deg(cx_m, cy_m)
        diff = abs(((az - direction_center_deg + 180) % 360) - 180)
        return EmbraceResult(
            55.0 if diff <= sector_half_deg * 0.6 else 35.0,
            diff > sector_half_deg,
            float(diff),
            0.0,
            len(pts),
            "采样不足，使用方位一致性替代",
        )

    # crest 长度
    crest_length = 0.0
    for i in range(1, len(pts)):
        r0p, c0p = pts[i - 1]
        r1p, c1p = pts[i]
        crest_length += np.hypot((c1p - c0p) * mpx, (r1p - r0p) * mpy)

    # 计算 crest 各段凸侧 vs 凹侧朝向穴
    cx_m = float((center_col) * mpx)  # 这两个无用，仿构
    # crest 折线：对每段 i，取 crest[i-1], crest[i], crest[i+1]
    # 局部「拐弯」侧：用 (crest[i] - crest[i-1]) × (crest[i+1] - crest[i]) 的 z 分量符号
    # 再比较「crest[i] -> 穴」向量与凸方向夹角
    v_to_acupoint: list[float] = []  # 每个中点的相对朝向（cos）
    n_pos = 0
    n_neg = 0
    diffs: list[float] = []

    for i in range(1, len(pts) - 1):
        r_prev, c_prev = pts[i - 1]
        r_cur, c_cur = pts[i]
        r_next, c_next = pts[i + 1]
        v1 = np.array([(c_cur - c_prev) * mpx, (r_prev - r_cur) * mpy])
        v2 = np.array([(c_next - c_cur) * mpx, (r_cur - r_next) * mpy])
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])  # z 分量
        # 局部曲率单位向量近似
        if abs(cross) < 1e-6:
            continue
        convex_right = cross > 0
        # 砂顶点指向穴的向量
        rr = (r_cur - center_row) * mpy
        cc = (c_cur - center_col) * mpx
        # 砂中心指向穴（反向）：穴 - 砂
        to_ac = np.array([-cc, -rr])
        if np.linalg.norm(to_ac) < 1e-6:
            continue
        to_ac = to_ac / np.linalg.norm(to_ac)
        # 取 crest 的"凸方向"（凹侧反方向）
        normal = np.array([-v2[1], v2[0]]) if convex_right else np.array([v2[1], -v2[0]])
        nn = np.linalg.norm(normal)
        if nn < 1e-6:
            continue
        normal = normal / nn
        cos_align = float(np.dot(normal, to_ac))  # +1=凸侧朝穴（反背）
        v_to_acupoint.append(cos_align)
        diffs.append(float(np.degrees(np.arccos(np.clip(cos_align, -1, 1)))))
        if cos_align > 0:
            n_pos += 1
        else:
            n_neg += 1

    if not v_to_acupoint:
        return EmbraceResult(50.0, False, 0.0, float(crest_length), len(pts), "无曲率信号")

    avg_cos = float(np.mean(v_to_acupoint))
    convex_to_acupoint = avg_cos > 0
    mean_dev = float(np.mean(diffs)) if diffs else 0.0
    # 朝抱分：凹侧朝穴（avg_cos<0）→ 高分；凸侧朝穴（avg_cos>0）→ 低分
    score = max(0.0, min(100.0, 50.0 - 55.0 * avg_cos))
    # 段数太少减信度
    seg_factor = min(1.0, len(v_to_acupoint) / 5.0)
    score = 50.0 + seg_factor * (score - 50.0)

    note_parts: list[str] = []
    note_parts.append(f"crest 段 {len(v_to_acupoint)} 个有效曲率")
    note_parts.append(f"凸/凹比={n_pos}/{n_neg}")
    if convex_to_acupoint:
        note_parts.append("凸侧朝穴，反背")
    else:
        note_parts.append("凹侧朝穴，拱抱有情")

    return EmbraceResult(
        score=float(round(score, 1)),
        convex_to_acupoint=bool(convex_to_acupoint),
        mean_bearing_dev_deg=float(round(mean_dev, 1)),
        crest_length_m=float(round(crest_length, 1)),
        n_segments=int(len(v_to_acupoint)),
        notes="; ".join(note_parts),
    )
