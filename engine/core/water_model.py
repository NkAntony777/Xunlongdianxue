"""得水 / 水煞双通道数理模型。

规格真源：research/99_summary/03_数理模型_点穴抽象与公式体系.md §2

  S_得水  — 吉项加分（界水有情、玉带、聚等）
  P_水煞  — 凶项罚分（割脚、反弓、直冲等）
  合成仅在 fuse_* 层，禁止用单一 f(d) 混维。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# —— 默认参数（与数理文档 §9 对齐）——
EPS_BAN_M = 40.0
D_CUT_M = 80.0
ALPHA_SHA = 0.45
W_GET = 0.16
V_SHA = 0.16


def _bearing_deg(dx_m: float, dy_m: float) -> float:
    """北=0、东=90 的方位角。dy_m 向北为正。"""
    return float((np.degrees(np.arctan2(dx_m, dy_m)) + 360.0) % 360.0)


@dataclass
class WaterChannels:
    """双通道结果。"""

    get_score: float          # S_得水 ∈ [0,100]
    sha_penalty: float        # P_水煞 ∈ [0,100]
    dist_m: float | None
    hard_ban: bool
    form: dict[str, float]    # 软标签 χ
    fused: float              # 合成后用于兼容旧接口的 0–100「水项」
    message: str = ""


def water_get_baseline(dist_m: float, *, banned: bool = False) -> float:
    """B_get(d)：得水距离基线（不含形态增益）。

    真源：research/99_summary/03_数理模型 §2.4 宽带 50–1000 有情。
    工程细化：**近水而不贴水**——堂心/中距高于贴岸，避免河岸光环。
      0 < d ≤ 50      → 60（界水近，得水弱）
      50 < d ≤ 120    → 70–78（近岸过渡）
      120 < d ≤ 220   → 84–92（入堂）
      220 < d ≤ 700   → 94（堂心有情峰）
      700 < d ≤ 1000  → 88
      1000 < d ≤ 3000 → 78
    更远段为工程外推；割脚近场由 P_水煞另惩。
    """
    if banned or not np.isfinite(dist_m) or dist_m < 0:
        return 0.0
    if dist_m <= 0:
        return 0.0
    if dist_m <= 50:
        return 60.0
    if dist_m <= 120:
        # 50→120：70 → 78
        t = (dist_m - 50.0) / 70.0
        return float(70.0 + 8.0 * t)
    if dist_m <= 220:
        # 120→220：84 → 92
        t = (dist_m - 120.0) / 100.0
        return float(84.0 + 8.0 * t)
    if dist_m <= 700:
        return 94.0
    if dist_m <= 1000:
        # 700→1000：94 → 88
        t = (dist_m - 700.0) / 300.0
        return float(94.0 - 6.0 * t)
    if dist_m <= 3000:
        return 78.0
    if dist_m <= 8000:
        # 3000→8000：78 → 58 线性
        t = (dist_m - 3000.0) / 5000.0
        return float(78.0 - 20.0 * t)
    return max(0.0, 55.0 - min((dist_m - 8000.0) / 3000.0, 12.0))


def water_sha_dist_penalty(dist_m: float, *, banned: bool = False) -> float:
    """P_dist(d)：距离割脚/近场煞。

    贴岸（<150m）加重惩罚，逼候选离岸入堂；200m 外轻煞。
    """
    if banned or not np.isfinite(dist_m) or dist_m < 0:
        return 100.0
    if dist_m <= 0:
        return 100.0
    if dist_m <= 40:
        return 95.0
    if dist_m <= 80:
        return 86.0
    if dist_m <= 120:
        return 62.0
    if dist_m <= 160:
        return 42.0
    if dist_m <= 220:
        return 22.0
    if dist_m <= 1100:
        return 6.0
    return 0.0


def water_get_score(
    dist_m: float,
    *,
    banned: bool = False,
    gamma_get: float = 1.0,
    face_weight: float = 1.0,
) -> float:
    """S_得水 = clip(B_get · Γ_get · (0.5+0.5 C_φ), 0, 100)。"""
    if banned or not np.isfinite(dist_m) or dist_m <= 0:
        return 0.0
    b = water_get_baseline(dist_m, banned=False)
    g = float(np.clip(gamma_get, 0.3, 1.3))
    c = float(np.clip(face_weight, 0.0, 1.0))
    # face_weight=1 → 因子 1.0；0 → 0.5
    face_factor = 0.5 + 0.5 * c
    return float(np.clip(b * g * face_factor, 0.0, 100.0))


def water_sha_penalty(
    dist_m: float,
    *,
    banned: bool = False,
    form_penalty: float = 0.0,
) -> float:
    """P_水煞 = max(P_dist, P_form)。"""
    p_dist = water_sha_dist_penalty(dist_m, banned=banned)
    p_form = float(np.clip(form_penalty, 0.0, 100.0))
    return float(max(p_dist, p_form))


def fuse_water_additive(
    get_s: float,
    sha_p: float,
    *,
    w_get: float = W_GET,
    v_sha: float = V_SHA,
) -> float:
    """U_水 = w_get·S − v_sha·P，再映射到约 0–100 显示用。"""
    raw = w_get * get_s - v_sha * sha_p
    # 映射：中性约 50（无水 get=0 sha=0 → 需在外层处理）
    # 有水时：raw 典型范围约 -16..16 → 压到 0–100
    mid = 50.0 + raw * (50.0 / max(w_get * 86.0, 1e-6))
    return float(np.clip(mid, 0.0, 100.0))


def fuse_field_with_sha(
    positive_score: float,
    sha_p: float,
    *,
    alpha_sha: float = ALPHA_SHA,
) -> float:
    """乘性：F ← (Σ w S) · (1 − α P/100)。"""
    a = float(np.clip(alpha_sha, 0.0, 1.0))
    factor = max(0.25, 1.0 - a * float(sha_p) / 100.0)
    return float(np.clip(positive_score * factor, 0.0, 100.0))


def form_gamma_and_penalty(form: dict[str, float]) -> tuple[float, float]:
    """由软标签 χ 计算 Γ_get 与 P_form。"""
    jade = float(form.get("jade", 0.0))
    pool = float(form.get("pool", 0.0))
    meander = float(form.get("meander", 0.0))
    rev = float(form.get("reverse_bow", 0.0))
    rush = float(form.get("rush", 0.0))
    cut = float(form.get("cut_foot", 0.0))
    leak = float(form.get("leak", 0.0))

    gamma = (
        1.0
        + 0.25 * jade
        + 0.30 * pool
        + 0.15 * meander
        - 0.40 * rev
        - 0.50 * rush
    )
    gamma = float(np.clip(gamma, 0.3, 1.3))

    p_form = 100.0 * max(
        0.90 * rush,
        0.85 * rev,
        0.70 * cut,
        0.60 * leak,
        0.0,
    )
    return gamma, float(np.clip(p_form, 0.0, 100.0))


def _estimate_meander_density(
    line,
    s_star: float,
    L: float,
    dist_w: float,
    *,
    window_m: float = 600.0,
    step_m: float = 40.0,
) -> float:
    """九曲水软标签：单位长度河曲（转角）密度。"""
    if L < step_m * 3 or dist_w <= 0:
        return 0.0
    s0 = max(0.0, s_star - window_m * 0.5)
    s1 = min(L, s_star + window_m * 0.5)
    if s1 - s0 < step_m * 3:
        return 0.0
    headings: list[float] = []
    s = s0
    prev = None
    while s <= s1:
        p = line.interpolate(s)
        if prev is not None:
            dx = float(p.x) - float(prev.x)
            dy = float(p.y) - float(prev.y)
            if dx * dx + dy * dy > 1e-6:
                headings.append(float(np.arctan2(dx, dy)))
            prev = p
        else:
            prev = p
        s += step_m
    if len(headings) < 3:
        return 0.0
    turns = 0
    for a, b in zip(headings[:-1], headings[1:]):
        d = abs(((b - a + np.pi) % (2 * np.pi)) - np.pi)
        if d > np.radians(25.0):
            turns += 1
    density = turns / max((s1 - s0) / 100.0, 1e-3)  # 每百米转角数
    # 0.3–1.2 次/百米为九曲有情
    return float(np.clip(density / 1.0, 0.0, 1.0) * dist_w * 0.85)


def _estimate_pool_soft(
    pt,
    gdf,
    facing: float | None,
    dist_m: float,
    dist_w: float,
    *,
    radius_m: float = 250.0,
) -> float:
    """聚堂水 χ_pool：近域水面/水体边界覆盖的近似面积比。

    无栅格水面时：用线缓冲多边形与前向半圆相交面积 / 半圆面积。
    """
    if dist_w <= 0 or not np.isfinite(dist_m) or dist_m > 2000:
        return 0.0
    try:
        from shapely.geometry import Point, Polygon
        from shapely.ops import unary_union

        # 缓冲河流成条带（近似水面宽）
        bufs = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            try:
                bufs.append(geom.buffer(25.0))
            except Exception:
                continue
        if not bufs:
            return 0.0
        water_poly = unary_union(bufs)
        # 前向扇区半圆（若有 facing）或全圆
        n_seg = 24
        angles = []
        if facing is not None and np.isfinite(facing):
            for k in range(n_seg + 1):
                a = float(facing) - 60.0 + 120.0 * k / n_seg
                angles.append(np.radians(a))
        else:
            angles = [2 * np.pi * k / n_seg for k in range(n_seg + 1)]
        ring = [(float(pt.x), float(pt.y))]
        for a in angles:
            # 3857：x 东 y 北
            ring.append(
                (
                    float(pt.x) + radius_m * np.sin(a),
                    float(pt.y) + radius_m * np.cos(a),
                )
            )
        ring.append((float(pt.x), float(pt.y)))
        sector = Polygon(ring)
        if not sector.is_valid or sector.area < 1.0:
            return 0.0
        inter = water_poly.intersection(sector)
        ratio = float(inter.area / max(sector.area, 1.0))
        # 聚堂：前方面积比 8%–35% 为吉
        if ratio < 0.03:
            return float(np.clip(ratio / 0.03 * 0.3, 0.0, 0.3) * dist_w)
        if ratio <= 0.35:
            return float(np.clip(0.4 + ratio * 1.5, 0.0, 1.0) * dist_w)
        # 过大（湖面过近）略减
        return float(np.clip(0.7 - (ratio - 0.35), 0.2, 0.7) * dist_w)
    except Exception:
        return 0.0


def classify_water_form_at_point(
    x: float,
    y: float,
    water,
    *,
    dist_m: float | None = None,
    facing: float | None = None,
) -> dict[str, float]:
    """估计 χ 标签：玉带 / 反弓 / 割脚 / 直冲 / 聚堂 / 九曲（软值 0–1）。

    基于最近河段三点曲率 + 侧别（数理 §2.2–2.3）+ P2 聚堂/九曲。
    water 为 WaterNetwork；失败时返回空标签。
    """
    form = {
        "jade": 0.0,
        "reverse_bow": 0.0,
        "rush": 0.0,
        "cut_foot": 0.0,
        "pool": 0.0,
        "meander": 0.0,
        "leak": 0.0,
        "side": 0.0,       # ι ∈ {-1,0,1}
        "curvature": 0.0,
    }
    if water is None or getattr(water, "empty", True):
        return form

    if dist_m is None:
        try:
            dist_m = float(water.distance_to_nearest_m(x, y))
        except Exception:
            dist_m = float("inf")

    if not np.isfinite(dist_m):
        return form

    if dist_m <= D_CUT_M:
        form["cut_foot"] = float(np.clip(1.0 - dist_m / D_CUT_M, 0.0, 1.0))

    try:
        from shapely.geometry import Point, LineString, MultiLineString
        from shapely.ops import nearest_points

        pt = water._to_3857(x, y)
        gdf = water.projected_gdf
        if gdf is None or gdf.empty:
            return form
        dists = gdf.distance(pt)
        valid = dists.replace([np.inf, -np.inf], np.nan).dropna()
        if valid.empty:
            return form
        idx = valid.idxmin()
        geom = gdf.geometry.loc[idx]
        if geom is None or geom.is_empty:
            return form

        # 取线几何
        line = None
        if geom.geom_type == "LineString":
            line = geom
        elif geom.geom_type == "MultiLineString":
            # 最近的子线
            best_d, best = 1e18, None
            for g in geom.geoms:
                d = g.distance(pt)
                if d < best_d:
                    best_d, best = d, g
            line = best
        elif geom.geom_type in ("Polygon", "MultiPolygon"):
            # 水面：用边界
            try:
                boundary = geom.boundary
                if boundary.geom_type == "MultiLineString":
                    line = max(boundary.geoms, key=lambda g: g.length)
                else:
                    line = boundary
            except Exception:
                line = None
        if line is None or line.is_empty or line.length < 1e-6:
            return form

        # 最近足点与沿线采样
        _, foot = nearest_points(pt, line)
        # 沿线参数
        s_star = float(line.project(foot))
        L = float(line.length)
        ds = max(30.0, min(L * 0.08, 120.0))  # 米制投影下
        s0 = max(0.0, s_star - ds)
        s1 = min(L, s_star + ds)
        p0 = line.interpolate(s0)
        p1 = foot
        p2 = line.interpolate(s1)
        ax, ay = float(p0.x), float(p0.y)
        bx, by = float(p1.x), float(p1.y)
        cx, cy = float(p2.x), float(p2.y)

        # 侧别：叉积 (p1-p0) × (pt-p1)
        tdx, tdy = bx - ax, by - ay
        rdx, rdy = float(pt.x) - bx, float(pt.y) - by
        cross = tdx * rdy - tdy * rdx
        # 曲率代理：转角
        v1x, v1y = bx - ax, by - ay
        v2x, v2y = cx - bx, cy - by
        n1 = max(np.hypot(v1x, v1y), 1e-9)
        n2 = max(np.hypot(v2x, v2y), 1e-9)
        v1x, v1y = v1x / n1, v1y / n1
        v2x, v2y = v2x / n2, v2y / n2
        # 有向转角 sin
        turn = v1x * v2y - v1y * v2x  # ≈ sin(Δheading)
        form["curvature"] = float(turn)

        # ι：侧别与弯向一致 → 凹侧（玉带），相反 → 反弓
        # turn>0 左转：凹在左侧(cross>0)；turn<0 右转：凹在右侧(cross<0)
        if abs(turn) < 0.08:
            iota = 0.0
        else:
            # 凹侧：cross 与 turn 同号
            if cross * turn > 0:
                iota = 1.0   # 凹侧
            else:
                iota = -1.0  # 凸侧
        form["side"] = iota

        bend_strength = float(np.clip(abs(turn) / 0.5, 0.0, 1.0))
        # 形态距离窗：太远形态意义弱
        dist_w = 1.0
        if dist_m > 2000:
            dist_w = 0.0
        elif dist_m > 800:
            dist_w = float(np.clip(1.0 - (dist_m - 800) / 1200.0, 0.0, 1.0))

        if iota > 0 and bend_strength > 0.15:
            form["jade"] = float(bend_strength * dist_w)
        if iota < 0 and bend_strength > 0.15:
            form["reverse_bow"] = float(bend_strength * dist_w)
        if bend_strength > 0.35:
            form["meander"] = float(0.5 * bend_strength * dist_w)

        # 九曲水：沿河更长窗采样转角次数（P2）
        try:
            form["meander"] = max(
                float(form.get("meander", 0.0)),
                _estimate_meander_density(line, s_star, L, dist_w),
            )
        except Exception:
            pass

        # 聚堂水 χ_pool：前方扇区水面面积比（P2）；无 DEM 时用近距水体缓冲面积代理
        try:
            form["pool"] = max(
                float(form.get("pool", 0.0)),
                _estimate_pool_soft(pt, gdf, facing, dist_m, dist_w),
            )
        except Exception:
            pass

        # 直冲：切向对准点
        # 流向用 (p2-p0)，指向点 (pt-p1)
        fdx, fdy = cx - ax, cy - ay
        fn = max(np.hypot(fdx, fdy), 1e-9)
        fdx, fdy = fdx / fn, fdy / fn
        rnx, rny = rdx / max(np.hypot(rdx, rdy), 1e-9), rdy / max(np.hypot(rdx, rdy), 1e-9)
        # cos：水流指向穴
        cos_to = fdx * rnx + fdy * rny
        if cos_to > 0.85 and dist_m < 1500:
            form["rush"] = float(np.clip((cos_to - 0.85) / 0.15, 0.0, 1.0) * dist_w)

        # 朝向一致性：水在穴之前方扇区 → 抬得水形态；在后方则压低
        if facing is not None and np.isfinite(facing):
            # 从穴(pt) 指向水足点(bx,by) 的方位（3857 下 y 北正近似）
            wb = _bearing_deg(bx - float(pt.x), by - float(pt.y))
            angle_diff = abs(((wb - float(facing) + 180.0) % 360.0) - 180.0)
            if angle_diff < 60:
                form["jade"] = min(1.0, form.get("jade", 0.0) * 1.2 + 0.05)
                form["meander"] = min(1.0, form.get("meander", 0.0) + 0.1)
            elif angle_diff > 120:
                form["jade"] = max(0.0, form.get("jade", 0.0) * 0.65)
                # 水在背后且近：略抬割脚感（不替代 P_dist）
                if dist_m < 200:
                    form["cut_foot"] = max(
                        float(form.get("cut_foot", 0.0)),
                        0.25 * (1.0 - dist_m / 200.0),
                    )

    except Exception:
        return form

    return form


def enrich_form_with_water_curve(
    form: dict[str, float],
    water,
    x: float,
    y: float,
    *,
    dist_m: float,
) -> dict[str, float]:
    """将三节弓背/玉带曲线信号并入 χ 软标签（不替代距离双通道）。

    - three_seg_concave → 抬 jade / meander（得水 Γ 加分）
    - three_seg_convex  → 抬 reverse_bow（水煞形态罚分）
    - 距离自适应权重 × 三节一致性
    原始 form 键保留；新增 three_seg_* / curve_weight 供 geography 展示。
    """
    out = dict(form or {})
    out.setdefault("jade", 0.0)
    out.setdefault("reverse_bow", 0.0)
    out.setdefault("meander", 0.0)
    out["three_seg_concave"] = 0.0
    out["three_seg_convex"] = 0.0
    out["three_seg_consistency"] = 0.0
    out["curve_weight"] = 0.0
    if water is None or getattr(water, "empty", True):
        return out
    if not np.isfinite(dist_m) or dist_m < 5.0 or dist_m > 5000.0:
        return out
    try:
        from engine.core.water_curve import (
            distance_adaptive_form_weight,
            score_water_curve_three_segments,
        )

        three = score_water_curve_three_segments(water, x, y, dist_m=float(dist_m))
        w = float(distance_adaptive_form_weight(float(dist_m)))
        cons = float(three.get("consistency", 0.0))
        c_w = w * (0.5 + 0.5 * cons)
        concave = float(three.get("three_seg_concave", 0.0))
        convex = float(three.get("three_seg_convex", 0.0))
        out["three_seg_concave"] = concave
        out["three_seg_convex"] = convex
        out["three_seg_consistency"] = cons
        out["curve_weight"] = w
        if concave > 0.0 and c_w > 0.0:
            out["jade"] = float(
                min(1.0, max(float(out.get("jade", 0.0)), 0.0) + 0.35 * concave * c_w)
            )
            out["meander"] = float(
                min(1.0, max(float(out.get("meander", 0.0)), 0.0) + 0.20 * concave * c_w)
            )
        if convex > 0.0 and c_w > 0.0:
            out["reverse_bow"] = float(
                min(
                    1.0,
                    max(float(out.get("reverse_bow", 0.0)), 0.0) + 0.40 * convex * c_w,
                )
            )
    except Exception:
        pass
    return out


def evaluate_water_channels(
    dist_m: float,
    *,
    banned: bool = False,
    form: dict[str, float] | None = None,
    facing_weight: float = 1.0,
) -> WaterChannels:
    """完整双通道评估。"""
    form = form or {}
    hard = bool(banned or (np.isfinite(dist_m) and dist_m <= 0))
    gamma, p_form = form_gamma_and_penalty(form)
    # 割脚抬高 cut 标签
    if np.isfinite(dist_m) and dist_m <= D_CUT_M and not hard:
        form = dict(form)
        form["cut_foot"] = max(float(form.get("cut_foot", 0.0)), 1.0 - dist_m / D_CUT_M)
        gamma, p_form = form_gamma_and_penalty(form)

    get_s = water_get_score(
        dist_m if not hard else 0.0,
        banned=hard,
        gamma_get=gamma,
        face_weight=facing_weight,
    )
    sha_p = water_sha_penalty(
        dist_m if not hard else 0.0,
        banned=hard,
        form_penalty=p_form,
    )
    # 兼容旧接口的「水综合显示分」：得水主导、煞惩罚（仍分通道计算）
    fused = float(np.clip(
        get_s * (1.0 - 0.50 * sha_p / 100.0) - 0.20 * sha_p,
        0.0, 100.0,
    ))
    if hard:
        fused = 0.0
        get_s = 0.0
        sha_p = 100.0

    return WaterChannels(
        get_score=get_s,
        sha_penalty=sha_p,
        dist_m=None if not np.isfinite(dist_m) else float(dist_m),
        hard_ban=hard,
        form=form,
        fused=fused,
    )


# —— 向后兼容别名（旧名 water_score_from_dist）——
def water_score_from_dist(dist_m: float, banned: bool = False) -> float:
    """兼容旧调用：返回双通道 fused 显示分（非规范合成入口）。"""
    return evaluate_water_channels(dist_m, banned=banned).fused
