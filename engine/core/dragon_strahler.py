"""山脊树化合并 + Strahler-like 分级。

调研 Tier 2：
  E: 1=父母/叶, 2=少祖, 3+=太祖
  F: 端点距 < merge_dist_m 且高差 < merge_dh_m 合并
"""
from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.dragon_vein import RidgeLine
from engine.io.dem import DEM


def _m_per_px(dem: DEM) -> tuple[float, float]:
    from engine.core.terrain_analysis import _is_geographic

    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs) and dem.bounds is not None:
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        return xres * 111_000.0 * cos_lat, yres * 111_000.0
    return float(xres), float(yres)


def _end_points(ridge: RidgeLine) -> tuple[tuple[int, int], tuple[int, int]]:
    c = ridge.coords
    return (int(c[0, 0]), int(c[0, 1])), (int(c[-1, 0]), int(c[-1, 1]))


def _elev(dem: DEM, r: int, c: int) -> float:
    if 0 <= r < dem.data.shape[0] and 0 <= c < dem.data.shape[1]:
        e = float(dem.data[r, c])
        return e if np.isfinite(e) else float("nan")
    return float("nan")


def _dist_m(
    a: tuple[int, int], b: tuple[int, int], mpx: float, mpy: float
) -> float:
    return float(np.hypot((a[0] - b[0]) * mpy, (a[1] - b[1]) * mpx))


def merge_ridge_lines(
    ridges: list[RidgeLine],
    dem: DEM,
    *,
    merge_dist_m: float = 50.0,
    merge_dh_m: float = 30.0,
    max_rounds: int = 8,
) -> list[RidgeLine]:
    """端-端距离与高差足够小时合并为更长主脉。"""
    if len(ridges) <= 1:
        return list(ridges)

    mpx, mpy = _m_per_px(dem)
    current = list(ridges)

    for _ in range(max_rounds):
        n = len(current)
        if n <= 1:
            break
        used = [False] * n
        merged: list[RidgeLine] = []
        changed = False

        # 按长度降序，优先挂到长脊上
        order = sorted(range(n), key=lambda i: -current[i].length_m)
        for i in order:
            if used[i]:
                continue
            base = current[i]
            used[i] = True
            b0, b1 = _end_points(base)
            coords = np.asarray(base.coords, dtype=np.int32)

            extended = True
            while extended:
                extended = False
                for j in order:
                    if used[j]:
                        continue
                    other = current[j]
                    o0, o1 = _end_points(other)
                    oc = np.asarray(other.coords, dtype=np.int32)
                    # 四种端点对接
                    pairs = [
                        (b1, o0, False, False),  # base 尾 + other 头
                        (b1, o1, False, True),   # base 尾 + other 尾反
                        (b0, o1, True, False),   # base 头 + other 尾
                        (b0, o0, True, True),    # base 头 + other 头反
                    ]
                    for pe, oe, rev_base, rev_other in pairs:
                        d = _dist_m(pe, oe, mpx, mpy)
                        if d > merge_dist_m:
                            continue
                        e1 = _elev(dem, pe[0], pe[1])
                        e2 = _elev(dem, oe[0], oe[1])
                        if not (np.isfinite(e1) and np.isfinite(e2)):
                            continue
                        if abs(e1 - e2) > merge_dh_m:
                            continue
                        # 拼接
                        c_base = coords[::-1] if rev_base else coords
                        c_oth = oc[::-1] if rev_other else oc
                        # 去重接点
                        if len(c_base) and len(c_oth):
                            if int(c_base[-1, 0]) == int(c_oth[0, 0]) and int(
                                c_base[-1, 1]
                            ) == int(c_oth[0, 1]):
                                c_oth = c_oth[1:]
                        new_coords = np.vstack([c_base, c_oth]) if len(c_oth) else c_base
                        length = float(base.length_m + other.length_m)
                        elevs = dem.data[
                            new_coords[:, 0].clip(0, dem.data.shape[0] - 1),
                            new_coords[:, 1].clip(0, dem.data.shape[1] - 1),
                        ]
                        elevs = elevs[np.isfinite(elevs)]
                        if elevs.size == 0:
                            continue
                        straight = _dist_m(
                            (int(new_coords[0, 0]), int(new_coords[0, 1])),
                            (int(new_coords[-1, 0]), int(new_coords[-1, 1])),
                            mpx, mpy,
                        )
                        sinu = length / max(straight, 1.0)
                        fs = float(np.mean(elevs) * sinu / max(length, 1.0))
                        base = RidgeLine(
                            coords=new_coords.astype(np.int32),
                            length_m=length,
                            mean_elevation=float(np.mean(elevs)),
                            max_elevation=float(np.max(elevs)),
                            sinuosity=float(sinu),
                            feature_significance=fs,
                            strahler_order=max(
                                getattr(base, "strahler_order", 1),
                                getattr(other, "strahler_order", 1),
                            ),
                            parent_idx=None,
                            role=getattr(base, "role", "branch"),
                        )
                        coords = base.coords
                        b0, b1 = _end_points(base)
                        used[j] = True
                        changed = True
                        extended = True
                        break
                    if extended:
                        break

            merged.append(base)

        # 未用到的（理论上 used 全 True）
        for i in range(n):
            if not used[i]:
                merged.append(current[i])
        current = merged
        if not changed:
            break

    return current


def assign_strahler_orders(
    ridges: list[RidgeLine],
    dem: DEM,
    *,
    junction_m: float = 80.0,
) -> list[RidgeLine]:
    """近似 Strahler：叶=1，两同级汇入 +1，否则取 max。

    用端点邻接图近似汇流树：高程较低端为「下游」。
    """
    if not ridges:
        return []

    mpx, mpy = _m_per_px(dem)
    n = len(ridges)
    # 端点列表
    ends: list[list[tuple[int, int]]] = []
    for r in ridges:
        a, b = _end_points(r)
        ends.append([a, b])

    # 邻接：端点接近则连边
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            for ei in ends[i]:
                for ej in ends[j]:
                    if _dist_m(ei, ej, mpx, mpy) <= junction_m:
                        adj[i].append(j)
                        adj[j].append(i)

    # 下游：平均高程更低为下游倾向
    mean_e = [r.mean_elevation for r in ridges]
    # 从叶（度=1 或高程最高的末梢）BFS 赋级
    order = [1] * n
    # 迭代：类似 Strahler 放松
    for _ in range(n + 2):
        changed = False
        for i in range(n):
            # 上游邻居：比 i 更高
            ups = [j for j in adj[i] if mean_e[j] >= mean_e[i] - 1.0]
            if not ups:
                new_o = 1
            else:
                u_orders = [order[j] for j in ups]
                mx = max(u_orders)
                if u_orders.count(mx) >= 2:
                    new_o = mx + 1
                else:
                    new_o = mx
            if new_o != order[i]:
                order[i] = new_o
                changed = True
        if not changed:
            break

    out: list[RidgeLine] = []
    for i, r in enumerate(ridges):
        o = max(1, int(order[i]))
        if o <= 1:
            role = "parent_leaf"  # 父母/叶
        elif o == 2:
            role = "shaozu"
        else:
            role = "taizu"
        out.append(
            RidgeLine(
                coords=r.coords,
                length_m=r.length_m,
                mean_elevation=r.mean_elevation,
                max_elevation=r.max_elevation,
                sinuosity=r.sinuosity,
                feature_significance=r.feature_significance * (1.0 + 0.15 * o),
                strahler_order=o,
                parent_idx=None,
                role=role,
            )
        )
    # parent_idx：指向邻接中更高等级或更长者
    for i in range(n):
        cands = [j for j in adj[i] if order[j] >= order[i]]
        if not cands:
            cands = adj[i]
        if cands:
            parent = max(cands, key=lambda j: (order[j], out[j].length_m))
            # 重建不可变 dataclass
            ri = out[i]
            out[i] = RidgeLine(
                coords=ri.coords,
                length_m=ri.length_m,
                mean_elevation=ri.mean_elevation,
                max_elevation=ri.max_elevation,
                sinuosity=ri.sinuosity,
                feature_significance=ri.feature_significance,
                strahler_order=ri.strahler_order,
                parent_idx=int(parent),
                role=ri.role,
            )
    out.sort(key=lambda x: (-x.strahler_order, -x.length_m))
    return out


def grade_and_merge_ridges(
    ridges: list[RidgeLine],
    dem: DEM,
    *,
    merge_dist_m: float = 50.0,
    merge_dh_m: float = 30.0,
) -> list[RidgeLine]:
    """合并 + Strahler 分级一站式。"""
    merged = merge_ridge_lines(
        ridges, dem, merge_dist_m=merge_dist_m, merge_dh_m=merge_dh_m,
    )
    return assign_strahler_orders(merged, dem)
