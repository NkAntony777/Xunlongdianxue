"""弓背水三节连贯判别 + 距离自适应形态权重。

传统理据（《水龙经》《地理五诀》）：
  - 「反弓水」「玉带水」不应只看 1 节河段。三节连贯同向才算数：
      * 上节 k-1、中节 k、本节 k+1 都同弯向、反弓概率高
      * 上下游不一致时形态意义弱
  - 距离自适应：
      * 太近河（< 80 m）：形态意义急剧衰减（割脚优先）
      * 中距（80-800 m）：形态权重最大
      * 远河（> 2000 m）：形态权重衰减
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString
from shapely.ops import nearest_points

from engine.io.rivers import WaterNetwork


@dataclass
class RiverSegment:
    """河段描述（每个交媾/合流段）。"""

    coords: np.ndarray        # shape (N, 2)
    mean_turn: float          # 平均有向转角
    length_m: float
    sample_pts_turn: list[float]


def split_river_segments(water: WaterNetwork, *, min_seg_len_m: float = 60.0) -> list[RiverSegment]:
    """将每条 LineString 切分为连续河段，每段最小长度 min_seg_len_m。"""
    segments: list[RiverSegment] = []
    if water.empty:
        return segments
    proj = water.projected_gdf
    for geom in proj.geometry:
        if geom is None or geom.is_empty or geom.geom_type not in ("LineString", "MultiLineString"):
            continue
        lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        for ln in lines:
            if ln.length < min_seg_len_m:
                continue
            coords = np.asarray(ln.coords)
            if coords.shape[0] < 3:
                continue
            # 均匀采样并算每段的有向转角
            pts_turn = []
            for i in range(1, coords.shape[0] - 1):
                v1 = coords[i] - coords[i - 1]
                v2 = coords[i + 1] - coords[i]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                cross = float(v1[0] * v2[1] - v1[1] * v2[0])
                pts_turn.append(cross / (n1 * n2))
            turn = float(np.mean(pts_turn)) if pts_turn else 0.0
            length = float(ln.length)
            segments.append(
                RiverSegment(
                    coords=coords,
                    mean_turn=turn,
                    length_m=length,
                    sample_pts_turn=pts_turn,
                )
            )
    return segments


def score_water_curve_three_segments(
    water: WaterNetwork,
    x: float,
    y: float,
    *,
    dist_m: float,
    forward_look_m: float = 300.0,
    backward_look_m: float = 300.0,
) -> dict[str, float]:
    """根据「前 + 中 + 后」三节判反弓/玉带。

    Returns:
        dict{three_seg_convex, three_seg_concave, consistency}
    """
    empty = {"three_seg_convex": 0.0, "three_seg_concave": 0.0, "consistency": 0.0}
    if water.empty or dist_m > 5000.0 or dist_m < 5.0:
        return empty

    # 找最近河段几何
    try:
        pt = water._to_3857(x, y)
        proj = water.projected_gdf
        dists = proj.distance(pt)
        valid = dists[np.isfinite(dists)]
        if valid.empty:
            return empty
        idx = valid.idxmin()
        line = proj.geometry.iloc[idx]
        if line is None or line.is_empty or line.length < 5.0:
            return empty
        if line.geom_type == "MultiLineString":
            line = max(line.geoms, key=lambda g: g.length)
        # 取最近足点
        _, foot = nearest_points(pt, line)
        s_star = float(line.project(foot))
        L = float(line.length)
    except Exception:
        return empty

    ds_b = max(30.0, min(backward_look_m, L * 0.18))
    ds_f = max(30.0, min(forward_look_m, L * 0.18))
    s_b0 = max(0.0, s_star - ds_b)
    s_b1 = max(0.0, s_star)
    s_f0 = min(L, s_star)
    s_f1 = min(L, s_star + ds_f)

    def _seg_mean_turn(ls: LineString, s0: float, s1: float) -> float:
        if s1 - s0 <= 1.0:
            return 0.0
        samples = 8
        ts = np.linspace(s0, s1, samples)
        coords = [ls.interpolate(float(t)) for t in ts]
        pts_turn = []
        for i in range(1, len(coords) - 1):
            v1 = np.array([coords[i].x - coords[i - 1].x, coords[i].y - coords[i - 1].y])
            v2 = np.array([coords[i + 1].x - coords[i].x, coords[i + 1].y - coords[i].y])
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cross = float(v1[0] * v2[1] - v1[1] * v2[0])
            pts_turn.append(cross / (n1 * n2))
        return float(np.mean(pts_turn)) if pts_turn else 0.0

    turn_back = _seg_mean_turn(line, s_b0, s_b1)
    turn_mid = _seg_mean_turn(line, max(s_b0, s_star - 30.0), min(s_f1, s_star + 30.0))
    turn_fwd = _seg_mean_turn(line, s_f0, s_f1)

    # 三节符号一致 → 一致性高
    turns = [turn_back, turn_mid, turn_fwd]
    signs = np.sign(np.array(turns))
    consistency = float(np.sum(signs == signs[1]) / len(signs)) if abs(signs[1]) > 0 else 0.0

    # 凸与凹
    concave = 0.0
    convex = 0.0
    if abs(turn_mid) > 0.05:
        if turn_mid < 0:
            # 右转（顺时针）= 凹在右侧 → "玉带倾向"
            # 距离自适应：300-1200 m 区间形态意义最大
            if 80.0 <= dist_m <= 1200.0:
                concave = min(1.0, abs(turn_mid) * 5.0)
        else:
            # 左转（逆时针）= 凹在左侧 → 玉带 (侧别再判)
            if 80.0 <= dist_m <= 1200.0:
                concave = min(1.0, abs(turn_mid) * 5.0)
        # 反弓：三节同号且较大
        if consistency >= 0.99 and abs(turn_mid) > 0.2:
            if abs(turn_mid) > abs(turn_back) and abs(turn_mid) > abs(turn_fwd):
                if turn_mid > 0:
                    pass  # 形态已计
        # 但反弓 = 同向弯转 + 凸侧向穴
        # 简化：仅当三节一致性高 + 中节幅度大时扣分（再回到水_model 由 ι 判定）
        if consistency >= 0.99 and abs(turn_mid) > 0.25:
            convex = min(1.0, abs(turn_mid) * 3.0)

    return {
        "three_seg_concave": float(concave),
        "three_seg_convex": float(convex),
        "consistency": float(consistency),
    }


def distance_adaptive_form_weight(dist_m: float) -> float:
    """距离自适应形态权重。

    - < 50 m: 0  (形态被割脚压制)
    - 80-1200 m: 1.0 (形态意义最大)
    - 1200-2000 m: 0.7 线性下降
    - > 2000 m: 0.3 形态意义弱
    """
    if dist_m < 50.0:
        return 0.0
    if dist_m <= 80.0:
        return float((dist_m - 50.0) / 30.0)
    if dist_m <= 1200.0:
        return 1.0
    if dist_m <= 2000.0:
        return float(1.0 - 0.7 * (dist_m - 1200.0) / 800.0 * (1.0 - 0.7))
    return 0.3


def score_multi_segment_concavity(water: WaterNetwork, x: float, y: float) -> float:
    """综合分段曲率 + 距离自适应 → 整体 0-1 凹抱可能性。"""
    if water.empty:
        return 0.0
    try:
        dist = water.distance_to_nearest_m(x, y)
    except Exception:
        return 0.0
    if not np.isfinite(dist) or dist > 5000.0:
        return 0.0
    three = score_water_curve_three_segments(water, x, y, dist_m=dist)
    w = distance_adaptive_form_weight(dist)
    base = max(three.get("three_seg_concave", 0.0),
               three.get("three_seg_convex", 0.0))
    consistency = three.get("consistency", 0.0)
    return float(min(1.0, base * w * (0.5 + 0.5 * consistency)))
