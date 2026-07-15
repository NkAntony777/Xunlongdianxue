"""水系 / 砂山分析。

参考:
  - 调研报告 05_sand_water/00_砂水分形量化.md
  - shanshui-mingtang-fengshui-gis/water_analysis.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from engine.core.anshan_quality import score_anshan_quality as _score_anshan
from engine.io.dem import DEM
from engine.io.rivers import WaterNetwork
from engine.utils.helpers import clamp_score

if TYPE_CHECKING:
    pass


@dataclass
class WaterScore:
    """水系评分结果（得水 / 水煞双通道）。"""

    score: int
    nearest_distance_m: float | None
    direction: str
    intersects: bool
    is_placeholder: bool
    message: str
    get_score: int = 0          # S_得水
    sha_penalty: int = 0        # P_水煞
    form: dict | None = None    # χ 标签


@dataclass
class SandScore:
    """砂山评分结果。"""

    score: int
    back_mountain_height: float
    back_mountain_distance_m: float
    back_mountain_direction: str
    left_peak_count: int
    right_peak_count: int
    front_peak_count: int
    message: str
    # A1-余：朝抱分（None=未评估；已计入 left/right 再融合 overall 的 sand 维）
    embrace_left: float | None = None
    embrace_right: float | None = None


def score_water_relation(
    x: float,
    y: float,
    water: WaterNetwork | None,
    dem: DEM | None = None,
    search_radius_m: float = 5000.0,
) -> WaterScore:
    """计算 (x, y) 处的水系（得水）评分。

    评分规则（数理 §2 双通道）:
      - S_得水：界水有情加分（距离基线 × 形态 Γ）
      - P_水煞：割脚/反弓/直冲罚分
      - score：兼容旧字段的 fused 显示分（非混维单函数）
    """
    from engine.core.water_model import (
        evaluate_water_channels,
        classify_water_form_at_point,
        enrich_form_with_water_curve,
    )

    if water is None or water.empty:
        return WaterScore(
            score=60,
            nearest_distance_m=None,
            direction="未识别",
            intersects=False,
            is_placeholder=True,
            message="未检测到水系数据，得水条件显示中性占位分。",
            get_score=0,
            sha_penalty=0,
            form={},
        )

    try:
        nearest = water.distance_to_nearest_m(x, y)
        direction = water.nearest_direction(x, y)
        intersects = water.intersects(x, y, buffer_m=0)
    except Exception:
        return WaterScore(
            score=60,
            nearest_distance_m=None,
            direction="未识别",
            intersects=False,
            is_placeholder=True,
            message="水系几何异常，得水条件使用中性占位分。",
            get_score=0,
            sha_penalty=0,
            form={},
        )

    if not np.isfinite(nearest):
        nearest = float("inf")

    d_use = 0.0 if intersects else float(nearest)
    # 传 dem：流向按高→低，直冲看是否对准穴
    form = classify_water_form_at_point(x, y, water, dist_m=d_use, dem=dem)
    # A1-水曲：三节弓背/玉带并入 form 软标签 → 再进 Γ_get / P_form（双通道不混维）
    if np.isfinite(d_use) and d_use > 0:
        form = enrich_form_with_water_curve(form, water, x, y, dist_m=d_use)

    # 海拔衰减：z_穴 − z_最近水面（高台少受水势冲，漫滩满煞）
    elev_above = _elev_above_nearest_water(x, y, water, dem)

    ch = evaluate_water_channels(
        d_use,
        banned=bool(intersects),
        form=form,
        elev_above_water_m=elev_above,
    )
    score = clamp_score(ch.fused)
    get_i = clamp_score(ch.get_score)
    sha_i = clamp_score(ch.sha_penalty)
    form = ch.form or form

    elev_note = ""
    if elev_above is not None and np.isfinite(elev_above):
        elev_note = f"，相对水面高差约 {elev_above:.0f} m"
        if elev_above >= 12.0:
            elev_note += "（高台，水煞衰减）"
        elif elev_above <= 3.0:
            elev_note += "（近水位）"

    rush_note = ""
    if float(form.get("rush", 0) or 0) > 0.35:
        cos_f = float(form.get("flow_cos", 0) or 0)
        method = str(form.get("flow_method") or "")
        rush_note = f"，直冲倾向(流向对穴 cos={cos_f:.2f}"
        if method:
            rush_note += f"/{method}"
        rush_note += ")"
    elif float(form.get("side_shoot", 0) or 0) > 0.35:
        rush_note = "，射胁倾向"

    if intersects or (np.isfinite(nearest) and nearest <= 80):
        msg = (
            f"水系过近/相交（约 {0 if intersects else nearest:.0f} m）"
            f"{elev_note}{rush_note}，"
            f"S_得水={get_i}，P_水煞={sha_i}，综合显示 {score}。"
            "宜避割脚、漫滩，不宜直接作穴。"
        )
    else:
        tags = []
        if form.get("jade", 0) > 0.3:
            tags.append("玉带倾向")
        if form.get("reverse_bow", 0) > 0.3:
            tags.append("反弓倾向")
        if form.get("rush", 0) > 0.3:
            tags.append("直冲（流向对穴）")
        if form.get("side_shoot", 0) > 0.3:
            tags.append("射胁")
        if form.get("three_seg_concave", 0) > 0.3:
            tags.append("三节朝抱")
        if form.get("three_seg_convex", 0) > 0.3:
            tags.append("三节反弓")
        tag_s = ("，" + "、".join(tags)) if tags else ""
        if nearest == float("inf"):
            level = "距离未知"
            nearest_disp = -1.0
        elif nearest <= 1000:
            level = "距离适中，界水有情"
            nearest_disp = nearest
        elif nearest <= 3000:
            level = "距离较近"
            nearest_disp = nearest
        elif nearest <= 8000:
            level = "距离偏远，得水较弱"
            nearest_disp = nearest
        else:
            level = "距离较远"
            nearest_disp = nearest
        msg = (
            f"最近水体位于{direction}侧约 {nearest_disp:.0f} m，"
            f"{level}{tag_s}{elev_note}{rush_note}；"
            f"S_得水={get_i}，P_水煞={sha_i}，综合显示 {score}。"
        )

    return WaterScore(
        score=score,
        nearest_distance_m=None if nearest == float("inf") else float(nearest),
        direction=direction,
        intersects=intersects,
        is_placeholder=False,
        message=msg,
        get_score=get_i,
        sha_penalty=sha_i,
        form=form,
    )


def _elev_above_nearest_water(
    x: float,
    y: float,
    water: WaterNetwork | None,
    dem: DEM | None,
) -> float | None:
    """穴高 − 最近水面高；失败返回 None（调用方不衰减）。"""
    if dem is None or water is None or getattr(water, "empty", True):
        return None
    try:
        z_hole = float(dem.sample(x, y))
    except Exception:
        return None
    if not np.isfinite(z_hole):
        return None
    wx, wy = _nearest_water_xy_dem_crs(x, y, water, dem)
    if wx is None:
        return None
    try:
        z_w = float(dem.sample(wx, wy))
    except Exception:
        return None
    if not np.isfinite(z_w):
        return None
    return float(z_hole - z_w)


def _nearest_water_xy_dem_crs(
    x: float,
    y: float,
    water: WaterNetwork,
    dem: DEM,
) -> tuple[float | None, float | None]:
    """最近水系点，尽量落在 DEM 坐标中以便 sample。"""
    try:
        from shapely.geometry import Point
        from shapely.ops import nearest_points

        gdf = getattr(water, "gdf", None)
        if gdf is None or gdf.empty:
            return None, None
        # 优先与 DEM 同 CRS 的 gdf
        pt = Point(float(x), float(y))
        best = None
        best_d = float("inf")
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            try:
                _, on_w = nearest_points(pt, geom)
                d = float(pt.distance(on_w))
                if d < best_d:
                    best_d = d
                    best = on_w
            except Exception:
                continue
        if best is not None:
            return float(best.x), float(best.y)
    except Exception:
        pass
    # 回退：3857 最近点再粗用（同 CRS 时可用）
    try:
        from shapely.ops import nearest_points

        pt = water._to_3857(x, y)
        gdf_p = water.projected_gdf
        if gdf_p is None or gdf_p.empty:
            return None, None
        distances = gdf_p.distance(pt)
        valid = distances.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            return None, None
        idx = valid.idxmin()
        geom = gdf_p.geometry.loc[idx]
        _, on_w = nearest_points(pt, geom)
        return float(on_w.x), float(on_w.y)
    except Exception:
        return None, None


def score_sand_mountain(
    dem: DEM,
    slope_arr: np.ndarray | None = None,
    search_radius_m: float = 500.0,
) -> SandScore:
    """分析候选穴周围砂山（玄武靠山 + 左右青龙白虎 + 案山朝山）。

    Args:
        dem: DEM（中心点为候选穴）
        slope_arr: 预计算坡度
        search_radius_m: 邻域半径
    """
    from scipy.ndimage import maximum_filter

    if slope_arr is None:
        from engine.core.terrain_analysis import compute_slope_aspect

        slope_arr, _ = compute_slope_aspect(dem)

    h, w = dem.data.shape
    cy, cx = h // 2, w // 2
    cand_elev = float(dem.data[cy, cx])
    from engine.core.terrain_analysis import _is_geographic
    if _is_geographic(dem.crs):
        m_per_unit = 111000.0
    else:
        m_per_unit = 1.0
    yres, xres = dem.resolution[1] * m_per_unit, dem.resolution[0] * m_per_unit

    # 局部最大值
    px_radius = max(3, int(round(search_radius_m / max(xres, yres))))
    local_max_size = 2 * px_radius + 1
    local_max = maximum_filter(dem.data, size=local_max_size, mode="reflect")
    is_peak = (dem.data == local_max) & np.isfinite(dem.data) & (dem.data > cand_elev + 5)

    # 计算每个峰相对中心的方向
    yy, xx = np.mgrid[0:h, 0:w]
    dx_m = (xx - cx) * xres
    dy_m = (yy - cy) * yres
    dist_m = np.sqrt(dx_m**2 + dy_m**2)
    bearing = (np.degrees(np.arctan2(dx_m, dy_m)) + 360) % 360

    peak_mask = is_peak & (dist_m > 20) & (dist_m <= search_radius_m * 2)

    # 按方位分扇区统计
    def _in_sector(deg_target: float, half: float = 45.0) -> np.ndarray:
        d = np.abs(((bearing - deg_target + 180) % 360) - 180)
        return d <= half

    back_peak = peak_mask & _in_sector(0, 45)
    left_peak = peak_mask & _in_sector(270, 45)
    right_peak = peak_mask & _in_sector(90, 45)
    front_peak = peak_mask & _in_sector(180, 45)

    # 玄武（后）评分
    if back_peak.any():
        back_heights = dem.data[back_peak]
        back_dists = dist_m[back_peak]
        best_idx = int(np.argmax(back_heights))
        back_height = float(back_heights[best_idx])
        back_dist = float(back_dists[best_idx])
        rel_h = back_height - cand_elev
        # 高度可观（30-200m 最佳）+ 距离适中（50-500m 最佳）
        h_score = 100 - abs(rel_h - 100) / 1.5
        d_score = 100 - abs(back_dist - 250) / 4
        back_score = clamp_score(0.6 * h_score + 0.4 * d_score)
    else:
        back_height = float("nan")
        back_dist = float("nan")
        back_score = 30

    # 左右砂评分：左右各有 1-3 个峰为吉
    def _peak_count_score(n: int) -> int:
        if n == 0:
            return 20
        if n == 1:
            return 75
        if 2 <= n <= 3:
            return 100
        if 4 <= n <= 5:
            return 80
        return 60  # 太多则"砂重"

    left_count = int(left_peak.sum())
    right_count = int(right_peak.sum())
    left_score = _peak_count_score(left_count)
    right_score = _peak_count_score(right_count)

    # 案山（前方 45° 扇区，200-1000m）
    front_near = peak_mask & _in_sector(180, 45) & (dist_m >= 200) & (dist_m <= 1000)
    anshan_count = int(front_near.sum())
    if anshan_count == 0:
        anshan_score = 40
    elif anshan_count == 1:
        anshan_score = 90
    elif anshan_count <= 3:
        anshan_score = 75
    else:
        anshan_score = 50

    # 朝山（前方 45° 扇区，1-5 km）
    front_far = peak_mask & _in_sector(180, 45) & (dist_m > 1000) & (dist_m <= 5000)
    chaoshan_count = int(front_far.sum())
    chaoshan_score = 70 + min(chaoshan_count, 5) * 5

    # —— 案山质量复核：P0-3 圆净/破碎/高度型 ——
    # 仅在案山确实存在时调用
    if front_near.any() and np.isfinite(back_height):
        anshan_q = _score_anshan(
            dem, cy, cx,
            anshan_mask=front_near,
            parents_top_m=float(back_height),
            facing_deg=180.0,
        )
        anshan_score = int(anshan_q.score)

    # —— 砂形朝抱 / 反背（A1-余）：吃进左右砂主路径 ——
    # 传统：青龙蜿蜒拱揖、白虎驯俯朝抱；凸侧朝穴为反背，凹侧朝穴为拱抱。
    # 默认 facing=180（朝南）：左青龙=270、右白虎=90。
    embrace_left_score = None
    embrace_right_score = None
    embrace_notes = ""
    try:
        from engine.core.mountain_curve import measure_embrace as _measure_embrace

        sand_region = (
            np.isfinite(dem.data)
            & (dem.data > cand_elev + 2.0)
            & (dist_m >= 30.0)
            & (dist_m <= search_radius_m * 2.0)
        )
        left_emb = _measure_embrace(
            dem, cy, cx, sand_region, direction_center_deg=270.0,
        )
        right_emb = _measure_embrace(
            dem, cy, cx, sand_region, direction_center_deg=90.0,
        )
        embrace_left_score = float(left_emb.score)
        embrace_right_score = float(right_emb.score)
        # 峰数分 60% + 朝抱分 40%（有砂时才有意义）
        left_score = int(round(0.60 * left_score + 0.40 * left_emb.score))
        right_score = int(round(0.60 * right_score + 0.40 * right_emb.score))
        emb_tags = []
        if left_emb.convex_to_acupoint:
            emb_tags.append("左反背")
        else:
            emb_tags.append("左朝抱")
        if right_emb.convex_to_acupoint:
            emb_tags.append("右反背")
        else:
            emb_tags.append("右朝抱")
        embrace_notes = (
            f"；朝抱 L={left_emb.score:.0f}/R={right_emb.score:.0f}"
            f"（{'/'.join(emb_tags)}）"
        )
    except Exception:
        embrace_notes = ""

    # 综合
    score = clamp_score(
        0.30 * back_score
        + 0.16 * left_score
        + 0.16 * right_score
        + 0.22 * anshan_score
        + 0.16 * chaoshan_score
    )

    return SandScore(
        score=score,
        back_mountain_height=back_height,
        back_mountain_distance_m=back_dist,
        back_mountain_direction="北",
        left_peak_count=left_count,
        right_peak_count=right_count,
        front_peak_count=chaoshan_count,
        message=(
            f"玄武靠山：高 {back_height:.1f} m 距 {back_dist:.0f} m（{back_score}）；"
            f"左砂 {left_count} 个峰（{left_score}）；右砂 {right_count} 个峰（{right_score}）；"
            f"案山 {anshan_count} 个（{anshan_score}）；朝山 {chaoshan_count} 个（{chaoshan_score}）"
            f"{embrace_notes}"
        ),
        embrace_left=embrace_left_score,
        embrace_right=embrace_right_score,
    )
