"""Incoming vein and shaozu/xuanwu on ridge."""
from __future__ import annotations

from typing import Any

import numpy as np

from engine.io.dem import DEM
from engine.core.dragon.types import (
    RidgeLine,
    RidgePoint,
    PrimaryDragon,
    IncomingVeinSelection,
    DragonVeinResult,
)
from engine.core.dragon.util import _m_per_px_dem, _bearing_rc, _ang_diff
from engine.core.dragon.primary import (
    orient_ridge_to_hole,
    reorient_primary_to_hole,
    score_ridge_as_incoming,
)
from engine.core.dragon.ridge import (
    light_ridge_mask,
    _order_ridge_by_dist_to_hole,
    _ridge_path_fraction,
)


def pick_xuanwu_shaozu_on_ridge(
    dem: DEM,
    ordered_coords: np.ndarray,
    center_row: int,
    center_col: int,
    sit_deg: float | None,
    mpx: float,
    mpy: float,
    *,
    xuanwu_dist: tuple[float, float] = (200.0, 700.0),
    shaozu_dist: tuple[float, float] = (1000.0, 8000.0),
    xw_dh_sweet: tuple[float, float] = (25.0, 150.0),
    forbid_mask: np.ndarray | None = None,
    water_surface: np.ndarray | None = None,
    sector_half: float | None = 50.0,
    require_sector: bool = True,
    source_first: bool = False,
    reject_cross_water: bool = True,
    shaozu_xw_ratio: float = 2.0,
    xuanwu_min_hard_m: float | None = None,
) -> tuple[RidgePoint | None, RidgePoint | None, dict[str, Any]]:
    """在脊点上切父母山与少祖。

    距离窗由调用方按局尺度 L 换算后传入（比例制）；本函数只做：
      - 窗内 + 高差甜区选父母
      - 少祖 dist ≥ 玄武 × ratio，优先脊源端（拓扑）
    """
    cand_elev = float(dem.data[center_row, center_col])
    h, w = dem.data.shape
    meta: dict[str, Any] = {
        "source_first": source_first,
        "require_sector": require_sector,
        "reject_cross_water": reject_cross_water,
        "cross_water_skipped": 0,
        "shaozu_xw_ratio": shaozu_xw_ratio,
    }
    n = len(ordered_coords) if ordered_coords is not None else 0
    # 硬下限 = 窗下界（调用方已按 L 比例算好）；不再叠加绝对米
    xw_d_lo = float(xuanwu_dist[0])
    if xuanwu_min_hard_m is not None:
        xw_d_lo = max(xw_d_lo, float(xuanwu_min_hard_m))
    xw_d_hi = float(xuanwu_dist[1])
    xuanwu_min_hard_m = xw_d_lo

    def _crosses_water(r: int, c: int) -> bool:
        if not reject_cross_water or water_surface is None:
            return False
        try:
            from engine.core.four_beasts_detect import _segment_hits_water
            return _segment_hits_water(
                water_surface, center_row, center_col, r, c,
                n_samples=28, end_skip=0.10,
            )
        except Exception:
            return False

    def _ok_cell(r: int, c: int) -> bool:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if forbid_mask is not None and forbid_mask.shape == (h, w) and forbid_mask[r, c]:
            return False
        if not bool(np.isfinite(dem.data[r, c])):
            return False
        if _crosses_water(r, c):
            meta["cross_water_skipped"] = int(meta.get("cross_water_skipped", 0)) + 1
            return False
        return True

    def _mk(r: int, c: int, sc: float) -> RidgePoint:
        elev = float(dem.data[r, c])
        dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
        brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
        return RidgePoint(
            row=r, col=c, elev_m=elev, dist_m=dist,
            bearing_deg=brg, score=sc, on_ridge=True,
        )

    def _in_sector(brg: float) -> bool:
        if not require_sector or sit_deg is None or sector_half is None:
            return True
        return _ang_diff(brg, float(sit_deg)) <= float(sector_half)

    # —— 玄武：父母窗硬下限 + 高差甜区；偏好窗中段（不贴身）——
    best_xw: RidgePoint | None = None
    best_xw_s = -1e18
    for r, c in ordered_coords:
        r, c = int(r), int(c)
        if not _ok_cell(r, c):
            continue
        dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
        if dist < xw_d_lo * 0.90 or dist > xw_d_hi * 1.20:
            continue
        brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
        if not _in_sector(brg):
            continue
        elev = float(dem.data[r, c])
        rel = elev - cand_elev
        if rel < 5.0:
            continue
        d_lo, d_hi = xw_d_lo, xw_d_hi
        # 偏好窗中后段（靠山有势，非最近丘）
        mid = 0.55 * d_lo + 0.45 * d_hi
        if d_lo <= dist <= d_hi:
            s_dist = 1.0 - 0.18 * abs(dist - mid) / max(d_hi - d_lo, 1.0)
        elif dist < d_lo:
            s_dist = 0.25 * dist / max(d_lo, 1.0)  # 贴身重罚
        else:
            s_dist = 0.65 * np.exp(-(dist - d_hi) / max(d_hi, 1.0))
        dh_lo, dh_hi = xw_dh_sweet
        if dh_lo <= rel <= dh_hi:
            s_dh = 1.0
        elif rel < dh_lo:
            s_dh = 0.45 * rel / max(dh_lo, 1.0)
        else:
            s_dh = max(-0.3, 0.8 * np.exp(-(rel - dh_hi) / 80.0))
        s_az = 1.0
        if sit_deg is not None and sector_half:
            s_az = 1.0 - _ang_diff(brg, float(sit_deg)) / max(float(sector_half), 1.0)
        s = 1.55 * s_dist + 1.25 * s_dh + 0.45 * s_az + 0.25 * min(rel / 80.0, 1.2)
        if s > best_xw_s:
            best_xw_s = s
            best_xw = _mk(r, c, float(s))

    if best_xw is None and not require_sector:
        # 放宽：脊上「够远够高」点，仍守硬下限
        for r, c in ordered_coords:
            r, c = int(r), int(c)
            if not _ok_cell(r, c):
                continue
            dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
            if dist < xw_d_lo * 0.9 or dist > xw_d_hi * 1.45:
                continue
            elev = float(dem.data[r, c])
            if elev < cand_elev + 3.0:
                continue
            # 略偏中距，忌 1/dist 把点吸到贴身
            mid = 0.55 * xw_d_lo + 0.45 * xw_d_hi
            s = 1.0 - abs(dist - mid) / max(xw_d_hi, 1.0) + (elev - cand_elev) / 70.0
            if dist < xw_d_lo:
                s *= 0.55
            if s > best_xw_s:
                best_xw_s = s
                best_xw = _mk(r, c, float(s))

    if best_xw is None:
        return None, None, {"reason": "no_xuanwu_on_ridge"}
    if best_xw.dist_m < xw_d_lo * 0.85:
        return None, None, {"reason": "xuanwu_too_close", "dist_m": best_xw.dist_m}

    # —— 少祖：来龙上游、更远、宜更高（比例/拓扑，无绝对 800m）——
    best_sz: RidgePoint | None = None
    best_sz_s = -1e18
    sz_lo = max(float(shaozu_dist[0]), best_xw.dist_m * float(shaozu_xw_ratio))
    sz_hi = float(shaozu_dist[1])
    meta["sz_lo_m"] = sz_lo
    meta["xw_dist_m"] = best_xw.dist_m

    def _score_sz(i: int, r: int, c: int, dist: float, elev: float, brg: float) -> float:
        colinear = 1.0 - _ang_diff(brg, best_xw.bearing_deg) / 70.0
        s_elev = float(np.clip((elev - best_xw.elev_m) / 45.0, -0.4, 1.8))
        s_az = 1.0
        if sit_deg is not None:
            s_az = 1.0 - _ang_diff(brg, float(sit_deg)) / 90.0
        s_far = float(np.clip((dist - sz_lo) / max(sz_hi - sz_lo, 500.0), 0.0, 1.0))
        s_src = 0.0
        if source_first and n > 0:
            s_src = 2.2 * (1.0 - i / max(n - 1, 1))
        return (
            0.75 * max(colinear, 0.0)
            + 1.15 * s_elev
            + 0.45 * max(s_az, 0.0)
            + 1.35 * s_far
            + s_src
        )

    # Pass1：源端 55% + 距离/高程硬窗
    for i, (r, c) in enumerate(ordered_coords):
        r, c = int(r), int(c)
        if not _ok_cell(r, c):
            continue
        if source_first and n > 6 and (i / max(n - 1, 1)) > 0.55:
            continue
        dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
        if dist < sz_lo or dist > sz_hi:
            continue
        brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
        if require_sector and not _in_sector(brg):
            if not (source_first and sit_deg is not None
                    and _ang_diff(brg, float(sit_deg)) <= (sector_half or 55) + 35):
                continue
        elev = float(dem.data[r, c])
        if elev < best_xw.elev_m - 8.0:
            continue
        s = _score_sz(i, r, c, dist, elev, brg)
        if s > best_sz_s:
            best_sz_s = s
            best_sz = _mk(r, c, float(s))

    # Pass2：整脊放宽源端限制
    if best_sz is None:
        for i, (r, c) in enumerate(ordered_coords):
            r, c = int(r), int(c)
            if not _ok_cell(r, c):
                continue
            dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
            if dist < sz_lo or dist > sz_hi:
                continue
            brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
            if require_sector and not _in_sector(brg):
                continue
            elev = float(dem.data[r, c])
            if elev < best_xw.elev_m - 12.0:
                continue
            s = _score_sz(i, r, c, dist, elev, brg) - 0.3
            if s > best_sz_s:
                best_sz_s = s
                best_sz = _mk(r, c, float(s))
                meta["shaozu_pass"] = "full_ridge"

    # Pass3：同岸最远够高
    if best_sz is None and n > 0:
        best_rc = None
        best_combo = -1e18
        for i, (r, c) in enumerate(ordered_coords):
            r, c = int(r), int(c)
            if not _ok_cell(r, c):
                continue
            dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
            if dist < sz_lo:
                continue
            elev = float(dem.data[r, c])
            if elev < cand_elev + 5.0:
                continue
            src_w = (1.0 - i / max(n - 1, 1)) if source_first else 0.5
            combo = dist / max(sz_lo, 1.0) + elev / 150.0 + 2.0 * src_w
            if elev >= best_xw.elev_m:
                combo += 0.4
            if combo > best_combo:
                best_combo = combo
                best_rc = (r, c, dist)
        if best_rc is not None:
            best_sz = _mk(best_rc[0], best_rc[1], 0.55)
            meta["shaozu_fallback"] = "farthest_same_bank_upstream"
        else:
            meta["shaozu_fallback"] = "none_upstream_same_bank"

    # 终检：少祖须明显远于玄武（比例）
    if best_sz is not None and best_sz.dist_m < best_xw.dist_m * float(shaozu_xw_ratio):
        meta["shaozu_rejected_too_close"] = True
        best_sz = None

    meta["xuanwu_score"] = best_xw.score
    meta["shaozu_score"] = best_sz.score if best_sz else None
    return best_xw, best_sz, meta


def beasts_from_primary_dragon(
    dem: DEM,
    center_row: int,
    center_col: int,
    primary: PrimaryDragon,
    *,
    forbid_mask: np.ndarray | None = None,
    xuanwu_dist: tuple[float, float] = (200.0, 700.0),
    shaozu_dist: tuple[float, float] = (1000.0, 8000.0),
    water=None,
    water_surface: np.ndarray | None = None,
) -> IncomingVeinSelection:
    """峦头正法：坐靠来龙，少祖=龙源，父母=近穴脊峰。

    先相对本穴 reorient 源/入首，避免南丘更高把祖钉在前。
    水界龙止：源/祖若隔主河道则降权并尽量改取同岸脊点。
    """
    mpx, mpy = _m_per_px_dem(dem)
    primary = reorient_primary_to_hole(
        dem, primary, center_row, center_col, water=water,
    )
    ordered = primary.ordered_coords  # 源 → 入首（已相对穴）
    if ordered is None or len(ordered) < 3:
        return IncomingVeinSelection(
            xuanwu=None, shaozu=None,
            incoming_azimuth_deg=None, sit_align_deg=None,
            downhill_ok=False, method="primary_empty",
            score=-1e9, meta={},
        )

    # 真水面栅格（供同岸判定）；优先用调用方传入的膨胀水面
    _segment_hits_water = None
    _ridge_path_crosses_water = None
    try:
        from engine.core.four_beasts_detect import (
            water_distance_rasters,
            _segment_hits_water,
            _ridge_path_crosses_water,
        )
        if water_surface is None and water is not None and not getattr(water, "empty", True):
            _wd0, water_surface = water_distance_rasters(
                dem, water, ban_buffer_m=0.0,
            )
            if water_surface is None or not np.any(water_surface):
                water_surface = (
                    np.isfinite(_wd0) & (_wd0 < max(1.0, min(mpx, mpy) * 0.6))
                )
            # 本地膨胀，与 detect 一致
            try:
                from scipy.ndimage import binary_dilation
                if water_surface is not None and np.any(water_surface):
                    water_surface = binary_dilation(
                        water_surface.astype(bool), iterations=2,
                    )
            except Exception:
                pass
    except Exception:
        pass

    # 坐向：穴看向龙源；朝向：对向
    sit = float(primary.sit_deg)
    facing = float(primary.facing_deg)
    flow_az = float(primary.flow_azimuth_deg)

    xw, sz, pm = pick_xuanwu_shaozu_on_ridge(
        dem, ordered, center_row, center_col, sit, mpx, mpy,
        xuanwu_dist=xuanwu_dist,
        shaozu_dist=shaozu_dist,
        forbid_mask=forbid_mask,
        water_surface=water_surface,
        sector_half=80.0,
        require_sector=False,
        source_first=True,
        reject_cross_water=True,
    )

    # 脊线本身过水 → 记脉断（仍可返回同岸父母，但降权）
    path_cross = False
    if water_surface is not None:
        try:
            path_cross = _ridge_path_crosses_water(
                water_surface, ordered, min_hits=2, sample_stride=max(1, len(ordered) // 40),
            )
        except Exception:
            path_cross = False
    pm = dict(pm or {})
    pm["ridge_path_crosses_water"] = bool(path_cross)
    if path_cross:
        pm["theory_note"] = "脊线跨水：水界龙止，山龙气脉大减"

    # 源端若隔水：少祖已在 pick 中跳过；再记源跨水标志
    if water_surface is not None and len(ordered) > 0:
        sr0, sc0 = int(ordered[0, 0]), int(ordered[0, 1])
        try:
            if _segment_hits_water(
                water_surface, center_row, center_col, sr0, sc0,
                n_samples=28, end_skip=0.10,
            ):
                pm["source_across_water"] = True
                if sz is None:
                    pm["shaozu_blocked"] = "source_across_water"
        except Exception:
            pass

    # 硬约束：少祖必须在「坐后」半区（与朝向差 > 90°），禁止落在前朱雀半区
    def _is_behind(bp: RidgePoint | None) -> bool:
        if bp is None:
            return False
        # 穴→祖 与 朝向 应接近对向（差≈180°）
        return _ang_diff(bp.bearing_deg, facing) > 90.0

    if sz is not None and not _is_behind(sz):
        # 丢弃前侧假祖，强制取源端（仍须同岸）
        n = len(ordered)
        k = max(2, n // 4)
        best_e = -1e18
        best_rc = None
        for r, c in ordered[:k]:
            r, c = int(r), int(c)
            if forbid_mask is not None and forbid_mask.shape == dem.data.shape:
                if forbid_mask[r, c]:
                    continue
            if not np.isfinite(dem.data[r, c]):
                continue
            # 同岸
            if water_surface is not None:
                try:
                    if _segment_hits_water(
                        water_surface, center_row, center_col, r, c,
                        n_samples=24, end_skip=0.10,
                    ):
                        continue
                except Exception:
                    pass
            elev = float(dem.data[r, c])
            dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
            # 源端少祖：相对 shaozu 窗 / 玄武（调用方已比例化）
            if dist < max(float(shaozu_dist[0]) * 0.75, 1.0):
                continue
            brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
            if _ang_diff(brg, facing) <= 90.0:
                continue  # 仍在前半区
            if elev > best_e:
                best_e = elev
                best_rc = (r, c, dist, brg, elev)
        if best_rc is not None:
            r, c, dist, brg, elev = best_rc
            sz = RidgePoint(
                row=r, col=c, elev_m=elev, dist_m=dist,
                bearing_deg=brg, score=1.0, on_ridge=True,
            )
            pm = dict(pm or {})
            pm["shaozu_forced_behind"] = True
        else:
            sz = None
            pm = dict(pm or {})
            pm["shaozu_rejected_front"] = True

    if sz is not None:
        sit = float(sz.bearing_deg)
        facing = (sit + 180.0) % 360.0
    elif xw is not None and _is_behind(xw):
        sit = float(xw.bearing_deg)
        facing = (sit + 180.0) % 360.0

    downhill_ok = True
    if sz is not None and xw is not None:
        downhill_ok = sz.elev_m >= xw.elev_m - 20.0

    return IncomingVeinSelection(
        xuanwu=xw,
        shaozu=sz,
        incoming_azimuth_deg=flow_az,
        sit_align_deg=_ang_diff(flow_az, facing),
        downhill_ok=downhill_ok,
        method="primary_dragon_classical",
        score=float(primary.score),
        meta={
            "theory": "坐靠来龙；少祖龙源（相对穴后）；禁前侧假祖；水界龙止同岸",
            "sit_deg": sit,
            "facing_deg": facing,
            "primary_ridge_idx": primary.ridge_idx,
            "primary_flow_az": flow_az,
            "reoriented": bool((primary.meta or {}).get("reoriented_to_hole")),
            "orient": (primary.meta or {}).get("orient"),
            "pick": pm,
        },
    )


def select_incoming_vein(
    dem: DEM,
    center_row: int,
    center_col: int,
    sit_deg: float,
    facing_deg: float | None = None,
    *,
    peaks_mask: np.ndarray | None = None,
    forbid_mask: np.ndarray | None = None,
    water_surface: np.ndarray | None = None,
    ridge_lines: list[RidgeLine] | None = None,
    ridge_mask: np.ndarray | None = None,
    xuanwu_dist: tuple[float, float] = (200.0, 700.0),
    shaozu_dist: tuple[float, float] = (1000.0, 8000.0),
    sector_half: float = 55.0,
) -> IncomingVeinSelection:
    """相对穴筛选主来龙，并在脊上切玄武（父母）与少祖。

    优先级：
      1. 已有 ridge_lines（全量龙脉结果）→ 评分为来龙 → 脊上切点
      2. 轻量 TPI 脊带 + 局部峰 → 伪脊链评分 → 切点
      3. 失败则 method=failed，由调用方扇区回退

    龙气走向：少祖→穴 宜接近 facing（由朝向/落势决定，不限定南北）。
    """
    if facing_deg is None:
        facing_deg = (float(sit_deg) + 180.0) % 360.0
    sit_deg = float(sit_deg) % 360.0
    facing_deg = float(facing_deg) % 360.0
    mpx, mpy = _m_per_px_dem(dem)
    h, w = dem.data.shape
    center_row = int(np.clip(center_row, 0, h - 1))
    center_col = int(np.clip(center_col, 0, w - 1))

    # —— 路径 1：矢量化脊线 ——
    if ridge_lines:
        best_sc = -1e18
        best_info = None
        best_idx = -1
        for i, rl in enumerate(ridge_lines):
            info = score_ridge_as_incoming(
                dem, rl.coords, center_row, center_col, sit_deg, facing_deg,
                mpx, mpy, sector_half=sector_half,
            )
            if info.get("ok") and info["score"] > best_sc:
                best_sc = info["score"]
                best_info = info
                best_idx = i
        if best_info is not None:
            ordered = best_info["ordered"]
            xw, sz, pm = pick_xuanwu_shaozu_on_ridge(
                dem, ordered, center_row, center_col, sit_deg, mpx, mpy,
                xuanwu_dist=xuanwu_dist, shaozu_dist=shaozu_dist,
                forbid_mask=forbid_mask,
                water_surface=water_surface,
                sector_half=sector_half,
                reject_cross_water=True,
            )
            flow_az = best_info.get("flow_azimuth_deg")
            if sz is not None:
                flow_az = _bearing_rc(
                    sz.row, sz.col, center_row, center_col, mpx, mpy
                )
            elif xw is not None:
                flow_az = _bearing_rc(
                    xw.row, xw.col, center_row, center_col, mpx, mpy
                )
            downhill_ok = bool(best_info.get("downhill_m", 0) > -5.0)
            sit_align = (
                _ang_diff(flow_az, facing_deg) if flow_az is not None else None
            )
            return IncomingVeinSelection(
                xuanwu=xw,
                shaozu=sz,
                incoming_azimuth_deg=float(flow_az) if flow_az is not None else None,
                sit_align_deg=float(sit_align) if sit_align is not None else None,
                downhill_ok=downhill_ok,
                method="ridge_lines",
                score=float(best_sc),
                meta={
                    "ridge_idx": best_idx,
                    "entrance_dist_m": best_info.get("entrance_dist_m"),
                    "sector_frac": best_info.get("sector_frac"),
                    "downhill_m": best_info.get("downhill_m"),
                    "pick": pm,
                    "same_bank": True,
                },
            )

    # —— 路径 2：轻量 TPI 脊 + 峰 ——
    if ridge_mask is None:
        ridge_mask = light_ridge_mask(dem)
    if peaks_mask is not None and peaks_mask.shape == ridge_mask.shape:
        cand_cells = peaks_mask & ridge_mask
    else:
        # 脊带上的局部高点
        from scipy.ndimage import maximum_filter

        data = dem.data
        valid = np.isfinite(data) & ridge_mask
        filled = np.where(valid, data, -np.inf)
        mx = maximum_filter(filled, size=5, mode="nearest")
        cand_cells = valid & (filled == mx) & (filled > -1e17)

    if forbid_mask is not None and forbid_mask.shape == cand_cells.shape:
        cand_cells = cand_cells & (~forbid_mask.astype(bool))

    rs, cs = np.where(cand_cells)
    if rs.size < 3:
        return IncomingVeinSelection(
            xuanwu=None, shaozu=None,
            incoming_azimuth_deg=None, sit_align_deg=None,
            downhill_ok=False, method="failed_no_ridge_peaks",
            score=-1e9, meta={},
        )

    # 伪脊：坐向扇区内候选峰，按距穴排序（须同岸）
    pts = []
    for r, c in zip(rs.tolist(), cs.tolist()):
        dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
        if dist < 25.0 or dist > max(shaozu_dist[1], 8000.0):
            continue
        brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
        if _ang_diff(brg, sit_deg) > sector_half + 15:
            continue
        if water_surface is not None:
            try:
                from engine.core.four_beasts_detect import _segment_hits_water
                if _segment_hits_water(
                    water_surface, center_row, center_col, r, c,
                    n_samples=24, end_skip=0.10,
                ):
                    continue
            except Exception:
                pass
        elev = float(dem.data[r, c])
        pts.append((dist, r, c, elev, brg))
    if len(pts) < 2:
        return IncomingVeinSelection(
            xuanwu=None, shaozu=None,
            incoming_azimuth_deg=None, sit_align_deg=None,
            downhill_ok=False, method="failed_few_back_peaks",
            score=-1e9, meta={"n_back_peaks": len(pts)},
        )

    pts.sort(key=lambda t: t[0])  # 近→远
    coords = np.array([[p[1], p[2]] for p in pts], dtype=np.int32)

    # 用峰链当「脊」评分
    chain_info = score_ridge_as_incoming(
        dem, coords, center_row, center_col, sit_deg, facing_deg,
        mpx, mpy, sector_half=sector_half, max_entrance_m=1200.0,
    )
    # 共线/脊连通加分：远峰到近峰路径落在 ridge_mask
    path_bonus = 0.0
    if len(pts) >= 2:
        far = pts[-1]
        near = pts[0]
        frac = _ridge_path_fraction(
            ridge_mask, far[1], far[2], near[1], near[2]
        )
        path_bonus = 0.8 * frac
        # 落势：远峰高程 ≥ 近峰
        if far[3] >= near[3] - 5:
            path_bonus += 0.4

    xw, sz, pm = pick_xuanwu_shaozu_on_ridge(
        dem, coords, center_row, center_col, sit_deg, mpx, mpy,
        xuanwu_dist=xuanwu_dist, shaozu_dist=shaozu_dist,
        forbid_mask=forbid_mask,
        water_surface=water_surface,
        sector_half=sector_half,
        reject_cross_water=True,
    )

    # 若少祖与玄武之间脊连通差，降权但仍可用
    if xw and sz:
        frac_xz = _ridge_path_fraction(
            ridge_mask, sz.row, sz.col, xw.row, xw.col
        )
        pm["ridge_frac_sz_xw"] = frac_xz
        if frac_xz < 0.25:
            # 仍保留点，标记弱连通
            pm["weak_ridge_link"] = True

    flow_az = None
    if sz is not None:
        flow_az = _bearing_rc(sz.row, sz.col, center_row, center_col, mpx, mpy)
    elif xw is not None:
        flow_az = _bearing_rc(xw.row, xw.col, center_row, center_col, mpx, mpy)

    base_sc = float(chain_info.get("score", 0.0)) + path_bonus
    if xw is None:
        return IncomingVeinSelection(
            xuanwu=None, shaozu=None,
            incoming_azimuth_deg=flow_az, sit_align_deg=None,
            downhill_ok=False, method="failed_no_xuanwu",
            score=base_sc, meta={"chain": chain_info, "pick": pm},
        )

    downhill_ok = True
    if sz is not None:
        downhill_ok = sz.elev_m >= xw.elev_m - 15.0
    sit_align = _ang_diff(flow_az, facing_deg) if flow_az is not None else None

    return IncomingVeinSelection(
        xuanwu=xw,
        shaozu=sz,
        incoming_azimuth_deg=float(flow_az) if flow_az is not None else None,
        sit_align_deg=float(sit_align) if sit_align is not None else None,
        downhill_ok=downhill_ok,
        method="light_ridge_peaks",
        score=base_sc + (xw.score if xw else 0) * 0.2,
        meta={
            "n_back_peaks": len(pts),
            "path_bonus": path_bonus,
            "chain_downhill_m": chain_info.get("downhill_m"),
            "pick": pm,
        },
    )


