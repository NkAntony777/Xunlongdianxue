"""五行星体识别：父母山/少祖山形体判读。

传统理据（research/02_four_beasts/00_四兽量化规则 + 峦头理气常识）：
  - 金星：圆净丰满，顶缓坡缓，平面接近正圆    — 最吉，可作父母山
  - 木星：高耸直立，平面长椭圆，长宽比大       — 吉，可作父母山
  - 水星：波浪起伏，平面呈多峰连绵             — 中吉，可作少祖/父母
  - 火星：尖锐瘦削，坡度极陡                   — 大凶，不可作父母山
  - 土星：方正平台，顶平坡陡                   — 中性，多作少祖
  - 廉贞：破碎乱石，火星之变                 — 大凶

判读采用三轴特征：
  1. 平面长宽比 aspect_ratio（最长水平跨度 / 最宽）
  2. 顶曲率 curvature（顶脊线二阶导均方）
  3. 峰数量 peak_count（局部极大值合并后）

输出含「是否可作父母山」布尔，遵循：
  - 金 / 木 / 水 → 父母山用星（吉）
  - 土 → 中性，少祖可用
  - 火 / 廉贞 → 禁作父母山（凶）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter

from engine.io.dem import DEM


@dataclass
class StarBodyResult:
    """五行星体判读结果。"""

    type: str                  # 金/木/水/火/土/不清
    confidence: float          # 0-1
    aspect_ratio: float        # 最长水平跨度 / 最宽
    plan_area_m2: float        # 高于穴的等高线包围面积
    h_relative_m: float        # 相对穴相对高差
    peak_count: int            # 顶脊合并峰数
    mean_top_slope: float      # 顶区段平均坡度（度）
    is_xuanwu_eligible: bool   # 是否可作父母山
    is_shaozu_eligible: bool   # 是否可作少祖
    notes: str


# —— 阈值（米制默认；可经区域调参覆盖）——
ELIGIBLE_FOR_XUANWU = {"\u91d1", "\u6728", "\u6c34"}
ELIGIBLE_FOR_SHAOZU = {"\u91d1", "\u6728", "\u6c34", "\u571f"}

# 平面形态阈值
AR_CIRCLE_MAX = 1.6        # 圆形/椭圆上界
AR_LONG_MIN = 2.6          # 长椭圆起线
AR_HUOS_RATIO_MIN = 3.0    # 火星/木星分界（瘦形）
AR_GIANT = 4.0             # 巨型（连绵水星）起线

# 高度阈值（相对父山平台）
H_MIN_FOR_PEAK_M = 4.0
H_FLAT_TOP_MAX_M = 6.0     # < 6 m 顶部起伏认定为方平台（土/金圆顶）

# 坡度阈值（度）
SLOPE_HUO_MIN = 32.0
SLOPE_JIN_MAX = 22.0
SLOPE_SHUI_MAX = 28.0

# 峰合并距离（米）
PEAK_MERGE_DIST_M = 80.0


def _m_per_px(dem: DEM) -> tuple[float, float]:
    from engine.core.terrain_analysis import _is_geographic

    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs):
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        return xres * 111_000.0 * cos_lat, yres * 111_000.0
    return xres, yres


def _local_peak_indices(
    data_crop: np.ndarray,
    elev_mask: np.ndarray,
    merge_dist_px: int,
) -> list[tuple[int, int]]:
    """在山顶邻域取极大值候选，按合并距离剔除过近峰。"""
    valid = np.isfinite(data_crop) & elev_mask
    if not valid.any():
        return []
    filled = np.where(valid, data_crop, -np.inf)
    # 小窗探测
    size = max(3, min(11, 2 * merge_dist_px + 1))
    mx = maximum_filter(filled, size=size, mode="nearest")
    cand = valid & (filled == mx)
    if not cand.any():
        return []
    rs, cs = np.where(cand)
    order = np.argsort(-filled[rs, cs])  # 最高优先
    kept: list[tuple[int, int]] = []
    used = np.zeros_like(cand)
    for idx in order:
        r, c = int(rs[idx]), int(cs[idx])
        if used[r, c]:
            continue
        kept.append((r, c))
        r_lo = max(0, r - merge_dist_px)
        r_hi = min(cand.shape[0], r + merge_dist_px + 1)
        c_lo = max(0, c - merge_dist_px)
        c_hi = min(cand.shape[1], c + merge_dist_px + 1)
        used[r_lo:r_hi, c_lo:c_hi] = True
    return kept


def _plan_bbox(
    elev_mask: np.ndarray,
    mpx: float,
    mpy: float,
) -> tuple[float, float, float]:
    """平面投影形态度量：主轴长 / 副轴宽 / 面积（米）。

    【修复 G.2】原用 bbox 长宽比对 45° 走向的山脊给出 AR≈1.41 误判为金星。
    新用 PCA 求真实主轴方向与展宽（长椭圆不论方向均能正确判别 AR）。
    """
    rs, cs = np.where(elev_mask)
    if rs.size < 4:
        return 0.0, 0.0, 0.0
    area_m2 = float(elev_mask.sum()) * mpx * mpy
    pts = np.stack([
        (cs - cs.mean()) * mpx,        # 东西方向米
        (rs - rs.mean()) * mpy,        # 南北方向米（行朝下→北为负）
    ], axis=0).astype(np.float64)     # shape (2, N)
    cov = np.cov(pts)
    if cov.shape != (2, 2):
        return 0.0, 0.0, area_m2
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals 升序：eigvals[0]=短半轴方差，eigvals[1]=长半轴方差
    # 主轴展宽 ≈ 2 × 2σ（点云 95%）
    long_side = 2.0 * 2.0 * float(np.sqrt(max(eigvals[1], 0.0)))
    short_side = 2.0 * 2.0 * float(np.sqrt(max(eigvals[0], 0.0)))
    return float(max(long_side, short_side)), float(min(long_side, short_side)), area_m2


def _ridge_curvature(
    data_crop: np.ndarray,
    elev_mask: np.ndarray,
    peaks: list[tuple[int, int]],
) -> float:
    """沿峰脊曲率均方（采样峰的"八邻域二阶差分"）。"""
    if not peaks:
        return 0.0
    h, w = data_crop.shape
    pad = np.pad(data_crop, 1, mode="edge")
    curvatures: list[float] = []
    for r, c in peaks:
        if not (0 <= r < h and 0 <= c < w):
            continue
        # 二阶 Laplacian 等效
        z_c = float(data_crop[r, c])
        z_n = float(pad[r, c])
        z_s = float(pad[r + 2, c + 1])
        z_e = float(pad[r + 1, c + 2])
        z_w = float(pad[r + 1, c])
        z_ne = float(pad[r, c + 2])
        z_nw = float(pad[r, c])
        z_se = float(pad[r + 2, c + 2])
        z_sw = float(pad[r + 2, c])
        lap = (z_n + z_s + z_e + z_w - 4 * z_c) / 4.0
        curvatures.append(abs(lap))
    return float(np.mean(curvatures)) if curvatures else 0.0


def _top_region_mask(
    data: np.ndarray,
    peak_elev: float,
    drop_m: float = 6.0,
) -> np.ndarray:
    """峰顶以下 drop_m 米内的区域。"""
    return np.isfinite(data) & (data >= peak_elev - drop_m)


def classify_star_body(
    dem: DEM,
    center_row: int,
    center_col: int,
    search_radius_m: float = 250.0,
    candidate_elev_m: float | None = None,
) -> StarBodyResult:
    """识别 (center_row, center_col) 附近的一座星体。

    Args:
        dem: DEM
        center_row, center_col: 疑似山体中心
        search_radius_m: 平面搜索半径（米）
        candidate_elev_m: 候选穴相对高程（仅用于「高差」展示）
    """
    h, w = dem.data.shape
    if not (0 <= center_row < h and 0 <= center_col < w):
        return StarBodyResult(
            type="不清", confidence=0.0, aspect_ratio=1.0,
            plan_area_m2=0.0, h_relative_m=0.0, peak_count=0,
            mean_top_slope=0.0, is_xuanwu_eligible=False,
            is_shaozu_eligible=False, notes="中心点越界",
        )
    peak_elev = float(dem.data[center_row, center_col])
    cand_elev = (
        float(candidate_elev_m) if candidate_elev_m is not None
        else float(np.nanmin(dem.data))
    )
    rel_h = peak_elev - cand_elev
    if rel_h < H_MIN_FOR_PEAK_M:
        return StarBodyResult(
            type="不清", confidence=0.1, aspect_ratio=1.0,
            plan_area_m2=0.0, h_relative_m=rel_h, peak_count=0,
            mean_top_slope=0.0, is_xuanwu_eligible=False,
            is_shaozu_eligible=False,
            notes="相对高差不足，非独立山体",
        )

    mpx, mpy = _m_per_px(dem)
    rad_px = max(2, int(round(search_radius_m / max(mpx, mpy))))
    r0 = max(0, center_row - rad_px)
    r1 = min(h, center_row + rad_px + 1)
    c0 = max(0, center_col - rad_px)
    c1 = min(w, center_col + rad_px + 1)
    crop = dem.data[r0:r1, c0:c1].copy()
    if crop.size == 0 or not np.isfinite(crop).any():
        return StarBodyResult(
            type="不清", confidence=0.1, aspect_ratio=1.0,
            plan_area_m2=0.0, h_relative_m=rel_h, peak_count=0,
            mean_top_slope=0.0, is_xuanwu_eligible=False,
            is_shaozu_eligible=False, notes="邻域无有效数据",
        )

    # 平滑后取山体掩膜：高于穴的局部高点
    smooth = gaussian_filter(np.where(np.isfinite(crop), crop, peak_elev - 100),
                             sigma=2.0)
    elev_mask = (
        np.isfinite(crop)
        & (smooth > peak_elev - max(15.0, rel_h * 0.5))
        & (crop > cand_elev + 2.0)
    )
    if not elev_mask.any():
        return StarBodyResult(
            type="不清", confidence=0.1, aspect_ratio=1.0,
            plan_area_m2=0.0, h_relative_m=rel_h, peak_count=0,
            mean_top_slope=0.0, is_xuanwu_eligible=False,
            is_shaozu_eligible=False, notes="山体掩膜为空",
        )

    long_side, short_side, plan_area = _plan_bbox(elev_mask, mpx, mpy)
    ar = long_side / max(short_side, 1e-3)

    # 顶区段：峰以下 6 m 内的范围
    top_mask = _top_region_mask(crop, peak_elev, drop_m=H_FLAT_TOP_MAX_M)
    if not top_mask.any():
        top_mask = elev_mask
    mean_top_slope = 0.0
    if top_mask.any():
        # 利用 dem 的坡度 / 数据不可得就用邻域粗糙度代替
        from scipy.ndimage import uniform_filter
        diff_n = np.abs(crop - np.roll(crop, 1, axis=0)) / max(mpy, 1e-3)
        diff_e = np.abs(crop - np.roll(crop, 1, axis=1)) / max(mpx, 1e-3)
        mean_top_slope = float(np.degrees(
            np.arctan(np.nanmean(np.hypot(diff_n, diff_e)[top_mask]))
        ))

    # 候选峰（合并距离）
    merge_px = max(2, int(round(PEAK_MERGE_DIST_M / max(mpx, mpy))))
    peaks = _local_peak_indices(crop, elev_mask, merge_dist_px=merge_px)
    peak_count = len(peaks)
    ridge_curv = _ridge_curvature(crop, elev_mask, peaks)

    # —— 形态分类 —— #
    type_label = "不清"
    confidence = 0.4
    notes_parts: list[str] = []

    # 火星条件：极度瘦削 + 极陡
    is_huo = (mean_top_slope >= SLOPE_HUO_MIN) and (ar >= AR_HUOS_RATIO_MIN)
    is_jin = (ar < AR_CIRCLE_MAX) and (mean_top_slope <= SLOPE_JIN_MAX)
    is_shuiping = (mean_top_slope <= SLOPE_SHUI_MAX) and (ar >= AR_LONG_MIN)
    is_mu_steep = (
        ar >= AR_HUOS_RATIO_MIN
        and mean_top_slope < SLOPE_HUO_MIN
        and mean_top_slope > SLOPE_JIN_MAX * 0.7   # > 15.4°
        and peak_count <= 1
    )
    is_mu_slim = (
        ar >= AR_HUOS_RATIO_MIN
        and mean_top_slope <= SLOPE_JIN_MAX * 0.7  # <= 15.4°  木星
        and peak_count <= 1
    )
    is_huo_suspect = (
        ar >= AR_HUOS_RATIO_MIN
        and mean_top_slope >= SLOPE_JIN_MAX * 0.7  # >= 15.4°
        and mean_top_slope < SLOPE_HUO_MIN        # < 32°
    )
    is_tu = (
        ar < AR_CIRCLE_MAX
        and (mean_top_slope >= SLOPE_JIN_MAX * 0.6)
        and (peak_count <= 1)
        and (ridge_curv < 0.6)
    )
    is_shui = (
        peak_count >= 2
        and (ar >= AR_LONG_MIN * 0.8)
        and (mean_top_slope <= SLOPE_SHUI_MAX)
    )

    # 优先级：火 > 木 > 金 > 水 > 土
    if is_huo:
        type_label = "火星"
        confidence = 0.85
        notes_parts.append("尖锐瘦削/陡削，凶形")
    elif is_mu_slim:
        type_label = "木星"
        confidence = 0.8
        notes_parts.append("高耸直立但坡缓")
    elif is_huo_suspect:
        type_label = "火星"
        confidence = 0.55
        notes_parts.append("瘦长且偏陡，火形嫌疑")
    elif is_mu_steep:
        type_label = "木星"
        confidence = 0.7
        notes_parts.append("高耸直立")
    elif is_shui:
        type_label = "水星"
        confidence = 0.7
        notes_parts.append("连绵多峰")
    elif is_jin:
        # 再细分：长宽比 < 1.2 圆金，< 1.6 椭圆金
        type_label = "金星"
        confidence = 0.85 if ar < 1.2 else 0.7
        notes_parts.append("圆净丰满")
    elif is_tu:
        type_label = "土星"
        confidence = 0.6
        notes_parts.append("方正平台")
    elif ar >= AR_GIANT and peak_count >= 3:
        type_label = "水星"
        confidence = 0.55
        notes_parts.append("巨型连绵，疑水星")
    else:
        type_label = "不清"
        confidence = 0.3

    # 廉贞：破碎火星变体（峰多 + 起伏频繁 + 顶区曲率均方极高）
    if type_label == "火星" and peak_count >= 3 and ridge_curv >= 1.2:
        type_label = "廉贞"
        confidence = 0.85
        notes_parts.append("破碎乱石")
        is_xuanwu = False
        is_shaozu = False
    else:
        # type_label 是 "金星"/"木星"/... 取首字判定
        base_type_char = type_label[0] if type_label else ""
        is_xuanwu = base_type_char in ELIGIBLE_FOR_XUANWU
        is_shaozu = base_type_char in ELIGIBLE_FOR_SHAOZU

    notes = (
        f"AR={ar:.2f}, topSlope={mean_top_slope:.1f}°, peaks={peak_count}, "
        f"ridgeCurv={ridge_curv:.2f}; " + "; ".join(notes_parts)
    )

    return StarBodyResult(
        type=type_label,
        confidence=float(confidence),
        aspect_ratio=float(ar),
        plan_area_m2=float(plan_area),
        h_relative_m=float(rel_h),
        peak_count=int(peak_count),
        mean_top_slope=float(mean_top_slope),
        is_xuanwu_eligible=bool(is_xuanwu),
        is_shaozu_eligible=bool(is_shaozu),
        notes=notes,
    )


def score_xuanwu_by_star(
    star: StarBodyResult,
    *,
    base_score: float = 70.0,
) -> float:
    """把星体类型折算为父母山分加成（0-100）。

    - 金/木/水 → 提升父母山打分
    - 土 → 中性
    - 火 / 廉贞 → 压制父母山可立度
    """
    if star.type in ("金星", "木星"):
        return min(100.0, base_score + 18 * star.confidence)
    if star.type == "水星":
        return min(100.0, base_score + 8 * star.confidence)
    if star.type == "土星":
        return base_score
    if star.type in ("火星", "廉贞"):
        return max(0.0, base_score - 30 * star.confidence)
    return base_score


def classify_xue_star(
    dem: DEM,
    row: int,
    col: int,
    *,
    search_radius_m: float = 80.0,
    form_hint: str | None = None,
) -> StarBodyResult:
    """穴星本体识别（P2）：在穴位本身做五行星体判读。

    传统「金星开窝、木星出芽、水星浪涌、土星平台、火星忌尖」。
    搜索半径小于父母山（默认 80 m），关注穴心微地形。
    form_hint：窝/钳/乳/突，可修正 notes（不强制改 type）。
    """
    star = classify_star_body(
        dem, row, col, search_radius_m=search_radius_m,
        candidate_elev_m=float(dem.data[row, col]) - 5.0
        if 0 <= row < dem.data.shape[0] and 0 <= col < dem.data.shape[1]
        else None,
    )
    if form_hint:
        # 形态与星体呼应提示
        pairs = {
            ("窝穴", "金星"): "金星开窝",
            ("乳穴", "金星"): "金星垂乳",
            ("突穴", "木星"): "木星出芽",
            ("钳穴", "水星"): "水星开钳",
            ("平缓", "土星"): "土星平台",
        }
        tag = pairs.get((form_hint, star.type))
        if tag:
            star = StarBodyResult(
                type=star.type,
                confidence=min(1.0, star.confidence + 0.1),
                aspect_ratio=star.aspect_ratio,
                plan_area_m2=star.plan_area_m2,
                h_relative_m=star.h_relative_m,
                peak_count=star.peak_count,
                mean_top_slope=star.mean_top_slope,
                is_xuanwu_eligible=star.is_xuanwu_eligible,
                is_shaozu_eligible=star.is_shaozu_eligible,
                notes=f"{star.notes}; 穴星:{tag}",
            )
        else:
            star = StarBodyResult(
                type=star.type,
                confidence=star.confidence,
                aspect_ratio=star.aspect_ratio,
                plan_area_m2=star.plan_area_m2,
                h_relative_m=star.h_relative_m,
                peak_count=star.peak_count,
                mean_top_slope=star.mean_top_slope,
                is_xuanwu_eligible=star.is_xuanwu_eligible,
                is_shaozu_eligible=star.is_shaozu_eligible,
                notes=f"{star.notes}; 穴形={form_hint}",
            )
    return star


def score_xue_star_bonus(star: StarBodyResult) -> tuple[int, str]:
    """穴星对 overall 的固定加减（−8..+8）。"""
    if star.type in ("金星", "木星"):
        return int(round(6 * star.confidence)), f"穴星{star.type}有情"
    if star.type == "水星":
        return int(round(3 * star.confidence)), f"穴星{star.type}"
    if star.type == "土星":
        return 0, "穴星土星中性"
    if star.type in ("火星", "廉贞"):
        return int(round(-8 * star.confidence)), f"穴星{star.type}无情"
    return 0, "穴星未明"
    return base_score * 0.85
