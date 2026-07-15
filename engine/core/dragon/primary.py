"""Primary dragon selection and hole reorientation."""
from __future__ import annotations

from typing import Any

import numpy as np

from engine.io.dem import DEM
from engine.core.dragon.types import RidgeLine, PrimaryDragon, DragonVeinResult
from engine.core.dragon.util import _m_per_px_dem, _bearing_rc, _ang_diff
from engine.core.dragon.entrance import (
    refine_entrance_on_ordered,
    find_entrance_on_ridge,
)
from engine.core.dragon.ridge import (
    _order_ridge_by_dist_to_hole,
    _ridge_path_fraction,
)


def _ridge_end_elev(
    dem: DEM, coords: np.ndarray, at_head: bool
) -> float:
    """脊一端邻域均高。"""
    if coords is None or len(coords) < 1:
        return float("-inf")
    n = len(coords)
    k = max(1, n // 7)
    seg = coords[:k] if at_head else coords[-k:]
    es = []
    for r, c in seg:
        r, c = int(r), int(c)
        if 0 <= r < dem.data.shape[0] and 0 <= c < dem.data.shape[1]:
            e = float(dem.data[r, c])
            if np.isfinite(e):
                es.append(e)
    return float(np.mean(es)) if es else float("-inf")


def _ridge_high_low_ends(
    dem: DEM, coords: np.ndarray
) -> tuple[tuple[int, int], tuple[int, int], float, float]:
    """脊两端粗分：更高端暂作源（仅全图初筛；相对穴时必须 reorient）。"""
    if coords is None or len(coords) < 2:
        return (0, 0), (0, 0), 0.0, 0.0
    h0 = (int(coords[0, 0]), int(coords[0, 1]))
    t0 = (int(coords[-1, 0]), int(coords[-1, 1]))
    eh = _ridge_end_elev(dem, coords, True)
    et = _ridge_end_elev(dem, coords, False)
    if eh >= et:
        return h0, t0, eh, et
    return t0, h0, et, eh


def orient_ridge_to_hole(
    dem: DEM,
    coords: np.ndarray,
    center_row: int,
    center_col: int,
    mpx: float,
    mpy: float,
    water_dist: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """相对穴点定向脊：源=坐后/更远更高/离水更远；入首=近穴且更近水。

    解决「南丘更高就把源钉在南」导致少祖落前的问题。
    不绑绝对东/南/西/北。
    """
    if coords is None or len(coords) < 2:
        return coords, {"ok": False}

    coords = np.asarray(coords, dtype=np.int32)
    e0 = (int(coords[0, 0]), int(coords[0, 1]))
    e1 = (int(coords[-1, 0]), int(coords[-1, 1]))
    elev0 = _ridge_end_elev(dem, coords, True)
    elev1 = _ridge_end_elev(dem, coords, False)

    def _d_hole(r: int, c: int) -> float:
        return float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))

    def _d_water(r: int, c: int) -> float:
        if water_dist is None:
            return 500.0
        if 0 <= r < water_dist.shape[0] and 0 <= c < water_dist.shape[1]:
            v = float(water_dist[r, c])
            return v if np.isfinite(v) else 5000.0
        return 5000.0

    d0, d1 = _d_hole(*e0), _d_hole(*e1)
    w0, w1 = _d_water(*e0), _d_water(*e1)

    # 穴→两端方位
    brg0 = _bearing_rc(center_row, center_col, e0[0], e0[1], mpx, mpy)
    brg1 = _bearing_rc(center_row, center_col, e1[0], e1[1], mpx, mpy)
    # 两端是否大致相对（一前一后）
    ends_opposite = _ang_diff(brg0, brg1) > 100.0

    def _src_score(elev: float, d_h: float, d_w: float) -> float:
        # 源：以「距穴远」为主（祖远），高程次之——防近端突峰抢源
        s = d_h / 180.0 + elev / 90.0 + min(d_w, 2500.0) / 2500.0
        return s

    def _ent_score(elev: float, d_h: float, d_w: float) -> float:
        # 入首：近穴、近水甜区、不宜过高欺穴
        s = 2.5 / (1.0 + d_h / 120.0) - elev / 100.0
        if 40.0 <= d_w <= 1000.0:
            s += 2.2
        elif d_w < 30.0:
            s -= 1.5
        else:
            s += max(0.0, 1.0 - (d_w - 1000.0) / 2000.0)
        return s

    # 方案 A：e0 源 e1 入首；方案 B 对调
    sc_a = _src_score(elev0, d0, w0) + _ent_score(elev1, d1, w1)
    sc_b = _src_score(elev1, d1, w1) + _ent_score(elev0, d0, w0)

    # 硬偏好：更远的一端作源（除非远侧明显更低且近侧也远）
    if d0 > d1 * 1.15:
        sc_a += 2.5
        sc_b -= 1.0
    elif d1 > d0 * 1.15:
        sc_b += 2.5
        sc_a -= 1.0

    if ends_opposite:
        sc_a += 0.5
        sc_b += 0.5

    # 落势：源应不低于入首太多
    if elev0 >= elev1 - 15.0:
        sc_a += 0.6
    else:
        sc_a -= 0.8
    if elev1 >= elev0 - 15.0:
        sc_b += 0.6
    else:
        sc_b -= 0.8

    # 近水一端不宜作源
    if w0 < 80.0 and w1 > 150.0:
        sc_a -= 2.0
        sc_b += 0.5
    if w1 < 80.0 and w0 > 150.0:
        sc_b -= 2.0
        sc_a += 0.5

    # 近端突峰：距穴很近却很高 → 禁止作源
    if d0 < max(d1 * 0.55, 200.0) and elev0 > elev1 + 20.0:
        sc_a -= 3.0
    if d1 < max(d0 * 0.55, 200.0) and elev1 > elev0 + 20.0:
        sc_b -= 3.0

    if sc_a >= sc_b:
        ordered = coords
        src, ent = e0, e1
        e_src, e_ent = elev0, elev1
        choice = "head_source"
        conf = sc_a - sc_b
    else:
        ordered = coords[::-1].copy()
        src, ent = e1, e0
        e_src, e_ent = elev1, elev0
        choice = "tail_source"
        conf = sc_b - sc_a

    # 入首精确定位：在有序脊（源→入首端）的尾段用「曲率极大 + 高程急降」
    # 取代单纯端点（报告 06 §2.1）
    ent_refined, refine_meta = refine_entrance_on_ordered(
        dem, ordered, water_dist=water_dist,
    )
    if ent_refined is not None:
        ent = ent_refined
        er, ec = int(ent[0]), int(ent[1])
        if 0 <= er < dem.data.shape[0] and 0 <= ec < dem.data.shape[1]:
            e_ent = float(dem.data[er, ec]) if np.isfinite(dem.data[er, ec]) else e_ent

    flow_az = _bearing_rc(src[0], src[1], ent[0], ent[1], mpx, mpy)
    # 坐：穴看向源
    sit = _bearing_rc(center_row, center_col, src[0], src[1], mpx, mpy)
    facing = (sit + 180.0) % 360.0

    meta = {
        "ok": True,
        "choice": choice,
        "confidence": float(conf),
        "source": src,
        "entrance": ent,
        "source_elev": float(e_src),
        "entrance_elev": float(e_ent),
        "flow_azimuth_deg": float(flow_az),
        "sit_deg": float(sit),
        "facing_deg": float(facing),
        "dist_source_m": float(_d_hole(*src)),
        "dist_entrance_end_m": float(_d_hole(*ent)),
        "ends_opposite": ends_opposite,
        "entrance_refine": refine_meta,
    }
    return ordered, meta


def reorient_primary_to_hole(
    dem: DEM,
    primary: PrimaryDragon,
    center_row: int,
    center_col: int,
    water=None,
) -> PrimaryDragon:
    """把已选主龙相对本穴重定向源/入首（防祖落前）。"""
    mpx, mpy = _m_per_px_dem(dem)
    wd = None
    try:
        from engine.core.four_beasts_detect import water_distance_rasters
        if water is not None and not getattr(water, "empty", True):
            wd, _ = water_distance_rasters(dem, water, ban_buffer_m=0.0)
            if not np.isfinite(wd).any():
                wd = None
    except Exception:
        wd = None

    coords = primary.ordered_coords
    if coords is None or len(coords) < 2:
        # 从原 dragon_vein 取脊
        dv = primary.dragon_vein
        if dv is not None and 0 <= primary.ridge_idx < len(dv.ridge_lines):
            coords = dv.ridge_lines[primary.ridge_idx].coords
        else:
            return primary

    ordered, om = orient_ridge_to_hole(
        dem, coords, center_row, center_col, mpx, mpy, water_dist=wd,
    )
    if not om.get("ok"):
        return primary

    sr, sc_ = om["source"]
    er, ec = om["entrance"]
    ex, ey = dem.xy(int(er), int(ec))
    meta = dict(primary.meta or {})
    meta["reoriented_to_hole"] = True
    meta["orient"] = om
    meta["direction_note"] = "source/entrance relative to hole (not absolute compass)"

    return PrimaryDragon(
        ridge_idx=primary.ridge_idx,
        ordered_coords=ordered,
        entrance_row=int(er),
        entrance_col=int(ec),
        entrance_xy=(float(ex), float(ey)),
        source_row=int(sr),
        source_col=int(sc_),
        flow_azimuth_deg=float(om["flow_azimuth_deg"]),
        sit_deg=float(om["sit_deg"]),
        facing_deg=float(om["facing_deg"]),
        score=float(primary.score),
        method=primary.method + "+hole_orient",
        length_m=primary.length_m,
        downhill_m=float(om["source_elev"] - om["entrance_elev"]),
        dragon_vein=primary.dragon_vein,
        meta=meta,
    )


def select_primary_dragon(
    dem: DEM,
    water=None,
    dragon_vein: DragonVeinResult | None = None,
    *,
    min_length_m: float = 120.0,
    anchor_row: int | None = None,
    anchor_col: int | None = None,
) -> PrimaryDragon | None:
    """选主来龙。

    若给定 anchor（热峰/穴），优先选「脊贴近锚点、源在锚点背后更远更高、
    入首近锚点且近水」的脊；否则全图粗选后再靠 reorient。

    不假定北来南入；方向由落势+相对锚点+得水决定。
    """
    if dragon_vein is None:
        dragon_vein = analyze_dragon_vein(dem, min_length_m=min_length_m)
    ridges = list(dragon_vein.ridge_lines or [])
    if not ridges:
        return None

    mpx, mpy = _m_per_px_dem(dem)
    hh, ww = dem.data.shape
    if anchor_row is None:
        anchor_row = hh // 2
    if anchor_col is None:
        anchor_col = ww // 2
    ar, ac = int(np.clip(anchor_row, 0, hh - 1)), int(np.clip(anchor_col, 0, ww - 1))

    wd = None
    water_surface = None
    try:
        from engine.core.four_beasts_detect import (
            water_distance_rasters,
            _segment_hits_water,
            _ridge_path_crosses_water,
        )

        if water is not None and not getattr(water, "empty", True):
            wd, water_surface = water_distance_rasters(dem, water, ban_buffer_m=0.0)
            if not np.isfinite(wd).any():
                wd = None
            if water_surface is None or not np.any(water_surface):
                if wd is not None:
                    water_surface = np.isfinite(wd) & (
                        wd < max(1.0, min(mpx, mpy) * 0.6)
                    )
            # 膨胀水面：单像元河线选龙时不漏跨水
            if water_surface is not None and np.any(water_surface):
                try:
                    from scipy.ndimage import binary_dilation
                    water_surface = binary_dilation(
                        water_surface.astype(bool), iterations=2,
                    )
                    if wd is not None and np.isfinite(wd).any():
                        water_surface = water_surface | (
                            np.isfinite(wd) & (wd < max(35.0, min(mpx, mpy) * 1.2))
                        )
                except Exception:
                    pass
    except Exception:
        wd = None
        water_surface = None

    water_face_az: float | None = None
    if water is not None and not getattr(water, "empty", True):
        try:
            from engine.core.four_beasts_detect import _nearest_water_bearing

            nw = _nearest_water_bearing(dem, ar, ac, water)
            if nw is not None:
                water_face_az = float(nw[0])
        except Exception:
            water_face_az = None

    best: PrimaryDragon | None = None
    best_sc = -1e18

    for idx, ridge in enumerate(ridges):
        coords = ridge.coords
        if coords is None or len(coords) < 8:
            continue
        if ridge.length_m < min_length_m * 0.5:
            continue

        # 相对锚点定向源/入首
        ordered, om = orient_ridge_to_hole(
            dem, coords, ar, ac, mpx, mpy, water_dist=wd,
        )
        if not om.get("ok"):
            continue
        sr, sc_ = int(om["source"][0]), int(om["source"][1])
        er, ec = int(om["entrance"][0]), int(om["entrance"][1])
        e_src = float(om["source_elev"])
        e_ent = float(om["entrance_elev"])
        downhill = e_src - e_ent
        flow_az = float(om["flow_azimuth_deg"])
        sit = float(om["sit_deg"])
        facing = float(om["facing_deg"])

        # 脊到锚点距离（越近越好 = 此龙服务于该穴/热峰）
        d_ridge = dist_to_ridge_m(ar, ac, ordered, mpx, mpy)
        d_src = float(om.get("dist_source_m", 1e9))
        d_ent_end = float(om.get("dist_entrance_end_m", 1e9))

        # 入首得水
        water_sc = 0.35
        d_w = None
        if wd is not None and 0 <= er < wd.shape[0] and 0 <= ec < wd.shape[1]:
            d_w = float(wd[er, ec]) if np.isfinite(wd[er, ec]) else 1e9
            if d_w < 25.0:
                water_sc = 0.05
            elif 40.0 <= d_w <= 900.0:
                water_sc = 1.3
            elif d_w < 40.0:
                water_sc = 0.4
            else:
                water_sc = max(0.15, 0.7 * np.exp(-(d_w - 900.0) / 1200.0))
        if wd is not None and 0 <= sr < wd.shape[0] and 0 <= sc_ < wd.shape[1]:
            d_ws = float(wd[sr, sc_]) if np.isfinite(wd[sr, sc_]) else 1e9
            if d_ws < 50.0:
                water_sc -= 1.2  # 源贴水：假龙

        face_water_sc = 0.0
        if water_face_az is not None:
            # 龙气（源→入首）宜与「穴→水」大致同向（面水收气）
            face_water_sc = 1.0 - _ang_diff(flow_az, water_face_az) / 180.0

        # —— 水界龙止：源/脊跨主河道 → 重罚（优先同岸来龙）——
        cross_src = False
        path_cross = False
        if water_surface is not None and np.any(water_surface):
            try:
                cross_src = _segment_hits_water(
                    water_surface, ar, ac, sr, sc_,
                    n_samples=28, end_skip=0.10,
                )
                path_cross = _ridge_path_crosses_water(
                    water_surface, ordered, min_hits=2,
                    sample_stride=max(1, len(ordered) // 40),
                )
            except Exception:
                cross_src = False
                path_cross = False
        om = dict(om)
        om["source_across_water"] = bool(cross_src)
        om["ridge_path_crosses_water"] = bool(path_cross)

        # 锚点贴脊 + 源远于入首端（祖远父近）
        near_ridge_sc = max(0.0, 1.0 - d_ridge / 600.0)
        geometry_sc = 0.0
        if d_src > d_ent_end * 1.05:
            geometry_sc += 1.2  # 源更远
        if d_ent_end < 800.0:
            geometry_sc += 0.8  # 入首端靠近锚点
        if downhill > 8.0:
            geometry_sc += 0.6

        length_sc = min(ridge.length_m / 2500.0, 1.3)
        sinu_sc = float(np.clip((ridge.sinuosity - 1.0) / 0.5, 0.0, 1.3))
        down_sc = float(np.clip(downhill / 50.0, -0.5, 1.6))
        edge = min(er, ec, hh - 1 - er, ww - 1 - ec)
        edge_sc = 0.0 if edge < 5 else min(edge / 20.0, 1.0)

        # 置信：相对穴定向置信
        conf = float(om.get("confidence", 0.0))
        # Tier 2：主脉优先（Strahler 高 = 太祖/少祖级）
        so = int(getattr(ridge, "strahler_order", 1) or 1)
        strahler_sc = min(so, 4) / 4.0
        role = getattr(ridge, "role", "branch")
        if role in ("shaozu", "taizu"):
            strahler_sc = max(strahler_sc, 0.75)

        sc = (
            2.8 * near_ridge_sc          # 服务本穴/热峰最重要
            + 1.6 * geometry_sc
            + 1.5 * length_sc
            + 1.0 * sinu_sc
            + 1.4 * max(0.0, down_sc)
            + 1.8 * water_sc
            + 0.8 * max(0.0, face_water_sc)
            + 0.5 * edge_sc
            + 0.4 * min(max(conf, 0.0) / 3.0, 1.0)
            + 0.2 * min(ridge.feature_significance / 50.0, 1.0)
            + 1.2 * strahler_sc          # 真祖级脊加权
        )
        if downhill < 3.0:
            sc -= 0.8
        if d_ridge > 900.0:
            sc -= 1.5  # 离热峰/穴太远的脊降权
        # 界水则止：隔岸来龙大减 / 脊线过水脉断
        if cross_src:
            sc -= 4.5
        if path_cross:
            sc -= 2.5

        if sc > best_sc:
            best_sc = sc
            ex, ey = dem.xy(er, ec)
            best = PrimaryDragon(
                ridge_idx=idx,
                ordered_coords=ordered,
                entrance_row=er,
                entrance_col=ec,
                entrance_xy=(float(ex), float(ey)),
                source_row=sr,
                source_col=sc_,
                flow_azimuth_deg=float(flow_az),
                sit_deg=float(sit),
                facing_deg=float(facing),
                score=float(sc),
                method="primary_anchor_ridge",
                length_m=float(ridge.length_m),
                downhill_m=float(downhill),
                dragon_vein=dragon_vein,
                meta={
                    "sinuosity": float(ridge.sinuosity),
                    "water_score": water_sc,
                    "face_water_align": face_water_sc,
                    "water_face_az": water_face_az,
                    "dist_water_at_entrance_m": d_w,
                    "source_elev": e_src,
                    "entrance_elev": e_ent,
                    "dist_ridge_to_anchor_m": d_ridge,
                    "dist_source_m": d_src,
                    "dist_entrance_end_m": d_ent_end,
                    "anchor": (ar, ac),
                    "orient": om,
                    "direction_note": "select by proximity to anchor + hole-relative source",
                },
            )

    if best is not None:
        best.dragon_vein = dragon_vein
        # 再 reorient 一次保证一致
        best = reorient_primary_to_hole(dem, best, ar, ac, water=water)
    return best


def dist_to_ridge_m(
    row: int,
    col: int,
    ordered_coords: np.ndarray,
    mpx: float,
    mpy: float,
) -> float:
    """点到脊折线的近似最短距离（米）。"""
    if ordered_coords is None or len(ordered_coords) == 0:
        return 1e9
    best = 1e18
    step = max(1, len(ordered_coords) // 80)
    for i in range(0, len(ordered_coords), step):
        r, c = int(ordered_coords[i, 0]), int(ordered_coords[i, 1])
        d = float(np.hypot((r - row) * mpy, (c - col) * mpx))
        if d < best:
            best = d
    return float(best)


def dragon_alignment_score(
    row: int,
    col: int,
    primary: PrimaryDragon,
    mpx: float,
    mpy: float,
    *,
    entrance_sweet_m: tuple[float, float] = (30.0, 1200.0),
    ridge_max_m: float = 800.0,
) -> dict[str, float]:
    """候选相对主来龙对齐分 0–100。

    宽松：只要贴脊/近入首就给高分；不因略偏入首坐标把热峰打到很低。
    """
    er, ec = primary.entrance_row, primary.entrance_col
    d_ent = float(np.hypot((row - er) * mpy, (col - ec) * mpx))
    d_ridge = dist_to_ridge_m(row, col, primary.ordered_coords, mpx, mpy)
    d_src = float(np.hypot(
        (row - primary.source_row) * mpy, (col - primary.source_col) * mpx,
    ))

    # 贴脊：主信号（半岛上沿北来脊的橙心应接近脊）
    if d_ridge <= 120.0:
        s_ridge = 95.0
    elif d_ridge <= 300.0:
        s_ridge = 85.0 - 20.0 * (d_ridge - 120.0) / 180.0
    elif d_ridge <= ridge_max_m:
        s_ridge = 65.0 * (1.0 - (d_ridge - 300.0) / max(ridge_max_m - 300.0, 1.0))
    else:
        s_ridge = max(15.0, 40.0 * np.exp(-(d_ridge - ridge_max_m) / 600.0))

    # 入首邻域：宽甜区（热峰可略离几何入首端点）
    lo, hi = entrance_sweet_m
    if d_ent <= hi:
        s_ent = 90.0 - 25.0 * min(d_ent, hi) / max(hi, 1.0)
    else:
        s_ent = max(20.0, 70.0 * np.exp(-(d_ent - hi) / 800.0))

    # 不宜压在源上（过远骑龙未结）
    if d_src < 150.0 and d_ent > 400.0:
        s_pos = 45.0
    elif d_src > d_ent:
        # 穴比源端更靠近入首侧 = 结穴位合理
        s_pos = 85.0
    else:
        s_pos = 60.0

    # 综合：贴脊权重大，入首距离次之
    total = 0.50 * s_ridge + 0.30 * s_ent + 0.20 * s_pos
    # 底分：有主龙时不要给灾难性低分（避免热峰被龙分打穿）
    total = max(total, 40.0 if d_ridge < 1000.0 else 25.0)

    return {
        "dragon_align": float(np.clip(total, 0.0, 100.0)),
        "dist_entrance_m": d_ent,
        "dist_ridge_m": d_ridge,
        "dist_source_m": d_src,
    }


# ---------------------------------------------------------------------------
# 来龙筛选 + 脊上切少祖/玄武（相对穴；有全量脊时优先）
# ---------------------------------------------------------------------------


def score_ridge_as_incoming(
    dem: DEM,
    ridge_coords: np.ndarray,
    center_row: int,
    center_col: int,
    sit_deg: float,
    facing_deg: float,
    mpx: float,
    mpy: float,
    *,
    max_entrance_m: float = 900.0,
    sector_half: float = 55.0,
) -> dict[str, Any]:
    """评估一条脊是否适合作为本穴来龙。"""
    if ridge_coords is None or len(ridge_coords) < 5:
        return {"score": -1e9, "ok": False}

    cand_elev = float(dem.data[center_row, center_col])
    ordered = _order_ridge_by_dist_to_hole(
        ridge_coords, center_row, center_col, mpx, mpy
    )
    # 入首：距穴最近的脊点
    er, ec = int(ordered[0, 0]), int(ordered[0, 1])
    e_dist = float(
        np.hypot((er - center_row) * mpy, (ec - center_col) * mpx)
    )
    if e_dist > max_entrance_m:
        return {"score": -1e9, "ok": False, "entrance_dist_m": e_dist}

    # 脊点方位：相对穴落在坐向扇区的比例
    in_sector = 0
    n_pts = 0
    elevs = []
    dists = []
    for r, c in ordered:
        r, c = int(r), int(c)
        if not (0 <= r < dem.data.shape[0] and 0 <= c < dem.data.shape[1]):
            continue
        e = float(dem.data[r, c])
        if not np.isfinite(e):
            continue
        d = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
        brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
        n_pts += 1
        elevs.append(e)
        dists.append(d)
        if _ang_diff(brg, sit_deg) <= sector_half:
            in_sector += 1
    if n_pts < 5:
        return {"score": -1e9, "ok": False}

    sector_frac = in_sector / n_pts
    # 远高近低：远端 30% 均高 vs 近端 30%
    k = max(2, n_pts // 3)
    near_e = float(np.mean(elevs[:k]))
    far_e = float(np.mean(elevs[-k:]))
    downhill = far_e - near_e  # >0 势来向穴
    # 龙气走向：最远脊点 → 穴
    fr, fc = int(ordered[-1, 0]), int(ordered[-1, 1])
    flow_az = _bearing_rc(fr, fc, center_row, center_col, mpx, mpy)
    align_face = 1.0 - _ang_diff(flow_az, facing_deg) / 180.0
    align_sit = 1.0 - abs(sector_frac - 1.0)  # 扇区内越多越好

    entrance_score = max(0.0, 1.0 - e_dist / max_entrance_m)
    downhill_score = float(np.clip(downhill / 40.0, -0.5, 1.5))
    length_proxy = float(dists[-1]) if dists else 0.0

    score = (
        2.2 * entrance_score
        + 1.8 * sector_frac
        + 1.5 * max(0.0, downhill_score)
        + 1.4 * align_face
        + 0.4 * min(length_proxy / 2000.0, 1.0)
        + 0.3 * (far_e - cand_elev) / 80.0
    )
    return {
        "score": float(score),
        "ok": sector_frac >= 0.25 and e_dist <= max_entrance_m,
        "entrance": (er, ec),
        "entrance_dist_m": e_dist,
        "sector_frac": sector_frac,
        "downhill_m": downhill,
        "flow_azimuth_deg": flow_az,
        "align_face": align_face,
        "far_elev": far_e,
        "near_elev": near_e,
        "ordered": ordered,
        "length_proxy_m": length_proxy,
    }


