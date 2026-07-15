"""水口识别与龙水交媾点查询。

传统理据：
  - 「水口」= 多条水汇聚 / 离开处。最贵者：来水天门开阔，去水地户紧闭。
  - 「龙水交媾」：两水交汇处为生气最足，水口第一吉地。
  - 「水口砂」：水口两旁之砂山要交牙紧锁，宽开则气散。
  - 「三叉水口」：来龙乘气处与「三水交汇」同位最贵。

实现：
  - find_confluences：在 WaterNetwork 的 projected_gdf 中检测几何交点 / 邻接节点
  - find_water_mouths：检测 GIS 中水系末端/合流点，并对每个口计算
    * 锁紧度：从口两侧最近山头之间的距离 / 水面宽，越紧越吉
    * 是否交媾（>= 2 个水几何汇入）
  - score_mouth_locking：对锁紧度打分
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, nearest_points
from shapely.strtree import STRtree

from engine.io.rivers import WaterNetwork


@dataclass
class ConfluencePoint:
    """水系交汇点（交媾点）。"""

    x: float
    y: float
    n_sources: int            # 汇入的水系数（>= 2 即交媾）
    nearest_acupoint_dist_m: float | None = None


@dataclass
class WaterMouth:
    """水口（合流或末端）位置 + 属性。"""

    x: float
    y: float
    kind: str                  # "confluence" 或 "endpoint" 或 "intersection"
    n_inflows: int             # 入水系条数
    lock_ratio: float          # 水口砂紧锁度（0-1）；越高越紧
    facing_angle_deg: float    # 相对全局最大水体的角度
    is_jiaogou: bool           # 是否为龙水交媾点
    notes: str = ""
    role: str = "unknown"      # "tianmen" 来水 / "dihu" 去水 / "unknown"
    elev_proxy: float | None = None  # 可选高程代理（米）


LOCK_GOOD_MIN = 1.4          # 两侧最近山头距离 / 水面宽 ≥ 此值 = 吉（紧锁）
LOCK_BAD_MAX = 0.6           # < 此值 = 漏气凶
# 地户要求更高锁紧；天门允许更开阔
DIHU_LOCK_GOOD = 1.6
TIANMEN_LOCK_GOOD = 1.1


def _geom_iter_lines(gdf):
    """提取 LineString / MultiLineString 的所有子线。"""
    out = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            out.append(geom)
        elif geom.geom_type == "MultiLineString":
            out.extend(geom.geoms)
    return out


def find_confluences(water: WaterNetwork) -> list[ConfluencePoint]:
    """检测水系交汇点（节点几何接近 < 5 m 视为汇入）。

    返回：所有可能为"合流"的点。
    """
    if water.empty:
        return []
    gdf = water.projected_gdf
    lines = _geom_iter_lines(gdf)
    if len(lines) < 2:
        return []

    # 用空间索引查每条线的端点
    endpoints = []
    for ln in lines:
        coords = list(ln.coords)
        if len(coords) < 2:
            continue
        endpoints.append(Point(coords[0]))
        endpoints.append(Point(coords[-1]))

    if not endpoints:
        return []

    tree = STRtree(endpoints)
    snap_m = 8.0
    counted: set[int] = set()
    confluences: list[ConfluencePoint] = []

    for i, p in enumerate(endpoints):
        if i in counted:
            continue
        # 同位聚合（端点距离 < snap_m）
        cluster_idx = [i]
        near_idx = tree.query(p.buffer(snap_m))
        for j in near_idx:
            if j == i or j in counted:
                continue
            q = endpoints[int(j)]
            if p.distance(q) <= snap_m:
                cluster_idx.append(int(j))
        counted.update(cluster_idx)
        if len(cluster_idx) >= 2:
            cx = float(np.mean([endpoints[k].x for k in cluster_idx]))
            cy = float(np.mean([endpoints[k].y for k in cluster_idx]))
            confluences.append(
                ConfluencePoint(
                    x=cx, y=cy,
                    n_sources=len(cluster_idx),
                    nearest_acupoint_dist_m=None,
                )
            )
    return confluences


def _line_endpoint_distances(water: WaterNetwork, x: float, y: float, snap_m: float = 50.0) -> list[float]:
    """从(x,y)到所有水系 line 上最近点的距离（投影上）。"""
    if water.empty:
        return []
    pt = Point(x, y)
    dists = []
    gdf = water.projected_gdf
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            d = geom.distance(pt)
            if np.isfinite(d):
                dists.append(float(d))
        except Exception:
            continue
    return dists


def _dominant_water_axis(water: WaterNetwork) -> tuple[np.ndarray, np.ndarray] | None:
    """估计水系"主流方向"：最长线段的方向向量。"""
    if water.empty:
        return None
    lines = _geom_iter_lines(water.projected_gdf)
    if not lines:
        return None
    longest = max(lines, key=lambda ln: ln.length)
    if longest.length < 1.0 or len(list(longest.coords)) < 2:
        return None
    coords = list(longest.coords)
    p0 = np.array(coords[0])
    p1 = np.array(coords[-1])
    delta = p1 - p0
    n = np.linalg.norm(delta)
    if n < 1.0:
        return None
    return (p0, p1)


def _endpoint_chain_elev_proxy(line: LineString) -> tuple[float, float]:
    """无 DEM 时：用沿线弧长参数代理「上游高 / 下游低」。

    返回 (start_proxy, end_proxy)：弧长起点=高、终点=低（约定线方向未知，
    真正分治在 classify_mouth_roles 里用全局端点相对位置）。
    """
    return 1.0, 0.0


def classify_mouth_roles(
    mouths: list[WaterMouth],
    water: WaterNetwork,
    *,
    elev_fn=None,
) -> list[WaterMouth]:
    """天门/地户分治（P1/P2）。

    规则（可标定）：
      1. 若提供 elev_fn(x,y)→高程：端点中较高侧为天门（来水），较低为地户（去水）
      2. 否则沿主流轴：轴起点侧偏天门，终点侧偏地户
      3. 合流点默认 unknown（既可来亦可去）；交媾仍保留

    就地更新 mouth.role 并返回同一列表。
    """
    if not mouths:
        return mouths
    dominant = _dominant_water_axis(water)

    # 收集端点型水口的高程/轴位置
    scored: list[tuple[WaterMouth, float]] = []
    for m in mouths:
        if elev_fn is not None:
            try:
                z = float(elev_fn(m.x, m.y))
                if np.isfinite(z):
                    m.elev_proxy = z
                    scored.append((m, z))
                    continue
            except Exception:
                pass
        if dominant is not None:
            p0, p1 = dominant
            # 投影到主流轴 [0,1]
            d = p1 - p0
            n2 = float(np.dot(d, d)) + 1e-12
            t = float(np.dot(np.array([m.x, m.y]) - p0, d) / n2)
            m.elev_proxy = t  # 轴参数代理
            scored.append((m, t))
        else:
            m.role = "unknown"

    if len(scored) >= 2:
        vals = [v for _, v in scored]
        lo, hi = min(vals), max(vals)
        mid = 0.5 * (lo + hi)
        span = hi - lo
        for m, v in scored:
            if span < 1e-6:
                m.role = "unknown"
            elif v >= mid + 0.15 * span:
                m.role = "tianmen"  # 高/上游 → 来水天门
            elif v <= mid - 0.15 * span:
                m.role = "dihu"     # 低/下游 → 去水地户
            else:
                m.role = "unknown"
            role_zh = {"tianmen": "天门", "dihu": "地户", "unknown": "未分"}.get(m.role, "")
            if role_zh and role_zh not in m.notes:
                m.notes = f"{m.notes}; {role_zh}".strip("; ")
    elif len(scored) == 1:
        scored[0][0].role = "unknown"

    # 合流点：若无 role 则保持 unknown
    for m in mouths:
        if not m.role:
            m.role = "unknown"
    return mouths


def find_water_mouths(
    water: WaterNetwork,
    dem_summary: dict | None = None,
    *,
    scan_radius_m: float = 6000.0,
    elev_fn=None,
    classify_roles: bool = True,
) -> list[WaterMouth]:
    """检测水口候选点（合流、线段端点、交点）。

    Args:
        water: WaterNetwork
        dem_summary: 未在本函数使用，保留以便后续接入地形核
        scan_radius_m: 离开"主流区域"的口按此距离为上限，避免远郊虚口
        elev_fn: 可选 (x,y)->elev，用于天门/地户分治
        classify_roles: 是否标注 tianmen/dihu
    """
    confluences = find_confluences(water)
    if water.empty:
        return []

    # 同时收集 line 的所有端点，作为"endpoint"
    lines = _geom_iter_lines(water.projected_gdf)
    endpoints: list[Point] = []
    for ln in lines:
        coords = list(ln.coords)
        if len(coords) >= 2:
            endpoints.append(Point(coords[0]))
            endpoints.append(Point(coords[-1]))

    dominant = _dominant_water_axis(water)
    mouths: list[WaterMouth] = []

    # 主流轴上用做"天门"候选：起点 + 终点 + 合流
    seen: set[tuple[int, int]] = set()

    def _key(p: Point) -> tuple[int, int]:
        return (round(p.x, 1), round(p.y, 1))

    def _register(pt: Point, kind: str, n: int) -> None:
        k = _key(pt)
        if k in seen:
            return
        seen.add(k)
        is_jiao = n >= 2
        facing = 0.0
        if dominant is not None:
            p0, p1 = dominant
            d_main = p1 - p0
            d_to = np.array([pt.x - p0[0], pt.y - p0[1]])
            n1 = np.linalg.norm(d_main)
            n2 = np.linalg.norm(d_to)
            if n1 > 0 and n2 > 0:
                cos_v = float(np.dot(d_main / n1, d_to / n2))
                facing = float(np.degrees(np.arccos(np.clip(cos_v, -1, 1))))
        mouths.append(
            WaterMouth(
                x=float(pt.x), y=float(pt.y),
                kind=kind, n_inflows=n,
                lock_ratio=0.0,
                facing_angle_deg=facing,
                is_jiaogou=is_jiao,
                notes=f"{kind} @({pt.x:.0f},{pt.y:.0f}) sources={n}",
                role="unknown",
            )
        )

    for cp in confluences:
        _register(Point(cp.x, cp.y), "confluence", cp.n_sources)
    for ep in endpoints:
        _register(ep, "endpoint", 1)

    if classify_roles:
        classify_mouth_roles(mouths, water, elev_fn=elev_fn)
    return mouths


def score_mouth_locking(
    water: WaterNetwork,
    mouth: WaterMouth,
    sand_dist_fn,                # 形如 (x, y, bearing_deg) -> 最短山脊距离的函数
    *,
    lock_good: float | None = None,
    lock_bad: float = LOCK_BAD_MAX,
) -> float:
    """为单个水口估算「砂锁紧度」。

    lock_ratio = (左最近砂 + 右最近砂) / 水面宽
        >= lock_good → 1.0
        <= lock_bad → 0.0
        居中线性插值

    天门/地户分治（P1）：
      - 地户（去水）：lock_good 更高（要求更紧）
      - 天门（来水）：lock_good 更低（允许更开）
    """
    role = getattr(mouth, "role", "unknown") or "unknown"
    if lock_good is None:
        if role == "dihu":
            lock_good = DIHU_LOCK_GOOD
        elif role == "tianmen":
            lock_good = TIANMEN_LOCK_GOOD
        else:
            lock_good = LOCK_GOOD_MIN

    if water.empty:
        return 0.5
    gdf = water.projected_gdf
    try:
        pt = Point(mouth.x, mouth.y)
        dists = gdf.distance(pt)
        valid = dists[np.isfinite(dists)]
        if valid.empty:
            return 0.5
        width_m = float(valid.min())
        if width_m < 1.0:
            width_m = 1.0
    except Exception:
        return 0.5

    # 取"水口主轴"方位：沿主流轴方向近似
    dominant = _dominant_water_axis(water)
    if dominant is None:
        return 0.5
    p0, p1 = dominant
    delta = p1 - p0
    bearing = float((np.degrees(np.arctan2(delta[0], delta[1])) + 360.0) % 360.0)
    left_bearing = (bearing - 90.0) % 360.0   # 垂直主流方向左
    right_bearing = (bearing + 90.0) % 360.0  # 垂直主流方向右

    try:
        d_left = float(sand_dist_fn(mouth.x, mouth.y, left_bearing))
        d_right = float(sand_dist_fn(mouth.x, mouth.y, right_bearing))
    except Exception:
        return 0.5

    side_sum = d_left + d_right
    if side_sum <= 0.5:
        return 0.0
    ratio = side_sum / max(width_m, 1.0)
    mouth.lock_ratio = float(ratio)

    if ratio >= lock_good:
        return 1.0
    if ratio <= lock_bad:
        return 0.0
    return float((ratio - lock_bad) / (lock_good - lock_bad))


def best_mouth_for_acupoint(
    water: WaterNetwork,
    acupoint_x: float,
    acupoint_y: float,
    mouths: list[WaterMouth] | None = None,
    *,
    consideration_radius_m: float = 5000.0,
) -> tuple[WaterMouth | None, float]:
    """为穴位找出最近、最相关的水口。

    Returns:
        (WaterMouth, dist_m) 或 (None, inf)
    """
    if water.empty:
        return None, float("inf")
    if mouths is None:
        mouths = find_water_mouths(water)
    if not mouths:
        return None, float("inf")
    best = None
    best_d = float("inf")
    for m in mouths:
        try:
            d = np.hypot(m.x - acupoint_x, m.y - acupoint_y)
        except Exception:
            continue
        if d < best_d and d <= consideration_radius_m:
            best_d = d
            best = m
    return best, float(best_d)


def score_water_mouth_for_candidate(
    mouth: WaterMouth | None,
    lock_ratio: float = 0.5,
) -> tuple[int, str]:
    """候选穴的水口得分加成（0-100）。

    - 龙水交媾（n_inflows >= 2）：高基础分
    - 水口紧锁（lock_ratio ≥ 0.7）：高分
    - 天门宜开 / 地户宜闭（P1 分治奖惩）
    - 远水口（>5 km）：衰减
    """
    if mouth is None:
        return 0, "无相关水口，未加成"
    notes: list[str] = []
    base = 50 if mouth.is_jiaogou else 35
    if mouth.is_jiaogou:
        notes.append(f"龙水交媾点({mouth.n_inflows}源)")
    role = getattr(mouth, "role", "unknown") or "unknown"
    if role == "dihu":
        notes.append("地户(去水)")
        # 地户必须紧：锁紧高奖、漏气重罚
        if lock_ratio >= 0.75:
            notes.append("地户紧闭")
            base += 28
        elif lock_ratio >= 0.5:
            base += 10
        elif lock_ratio <= 0.3:
            notes.append("地户漏气")
            base -= 28
        else:
            base -= 8
    elif role == "tianmen":
        notes.append("天门(来水)")
        # 天门宜开：过紧略减、适度开阔加分
        if lock_ratio <= 0.45:
            notes.append("天门开阔")
            base += 18
        elif lock_ratio <= 0.7:
            base += 8
        else:
            notes.append("天门过窄")
            base -= 5
    else:
        if lock_ratio >= 0.7:
            notes.append("水口紧锁")
            base += 25
        elif lock_ratio <= 0.3:
            notes.append("水口漏")
            base -= 20

    if mouth.kind == "endpoint":
        notes.append("末端型")
        base += 0
    score = max(0, min(100, int(base)))
    return score, "; ".join(notes)
