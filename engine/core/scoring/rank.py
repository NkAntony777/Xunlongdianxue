"""Candidate search + rank pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.acupoint import AcupointCandidate
from engine.core.terrain_analysis import compute_slope_aspect, analyze_terrain
from engine.io.dem import DEM
from engine.io.rivers import WaterNetwork
from engine.utils.helpers import clamp_score
from engine.core.scoring.candidate import (
    FusedScore,
    score_candidate,
    _bearing_from_to,
)
from engine.core.scoring.gate import _gate_beasts_for_hole
from engine.core.scoring.weights import DEFAULT_WEIGHTS

def find_and_rank_candidates(
    dem: DEM,
    water: WaterNetwork | None = None,
    top_k: int = 10,
    min_score: int = 50,
    weights: dict[str, float] | None = None,
    *,
    dragon_vein=None,
    primary_dragon=None,
    return_context: bool = False,
    require_beasts: bool = True,
) -> list[FusedScore] | tuple[list[FusedScore], dict[str, Any]]:
    """先定龙再点穴：主来龙入首邻域优先，综合分混入龙对齐。

    Args:
        dem / water / top_k / min_score / weights: 同前
        dragon_vein: 可选已算好的 DragonVeinResult（避免重复 ~2min 全量龙脉）
        primary_dragon: 可选 PrimaryDragon
        return_context: True 时额外返回 {dragon_vein, primary_dragon, qi_grid}
        require_beasts: True 时剔除少祖/四象识别失败、少祖不高于玄武的候选
    """
    from engine.core.acupoint import search_candidates
    from engine.core.dragon_vein import (
        analyze_dragon_vein,
        select_primary_dragon,
        dragon_alignment_score,
        reorient_primary_to_hole,
        _m_per_px_dem,
    )

    # E.5：全 NaN / 无有效像元 → 空结果，不崩溃、不造高分
    if dem is None or dem.data is None or not np.isfinite(dem.data).any():
        empty_ctx = {"dragon_vein": None, "primary_dragon": None, "qi_grid": None}
        return ([], empty_ctx) if return_context else []

    slope_arr, aspect_arr = compute_slope_aspect(dem)
    try:
        terrain = analyze_terrain(dem)
    except ValueError:
        empty_ctx = {"dragon_vein": None, "primary_dragon": None, "qi_grid": None}
        return ([], empty_ctx) if return_context else []

    from engine.core.four_beasts_detect import compute_score_grid, find_score_peak
    from engine.core.four_beasts_detect import WATER_BAN_BUFFER_M

    _ban_m = float(WATER_BAN_BUFFER_M)
    mpx, mpy = _m_per_px_dem(dem)

    # —— Step 1: 生气场（理论最优锚 = 热峰）——
    try:
        qi_grid = compute_score_grid(dem, water=water)
    except Exception:
        qi_grid = None

    peak_rc: tuple[int, int, float] | None = None
    if qi_grid is not None:
        peak = find_score_peak(qi_grid)
        if peak is not None:
            peak_rc = (int(peak[0]), int(peak[1]), float(peak[2]))

    # —— Step 2: 定龙（相对热峰选主脊；全量 D8 可接受数分钟）——
    yaoxia_points: list[dict[str, Any]] = []
    dv = dragon_vein
    primary = primary_dragon
    try:
        if dv is None:
            dv = analyze_dragon_vein(dem, min_length_m=120.0, water=water)
        yaoxia_points = list(getattr(dv, "yaoxia", None) or [])
        ar = peak_rc[0] if peak_rc else dem.data.shape[0] // 2
        ac = peak_rc[1] if peak_rc else dem.data.shape[1] // 2
        # Tier 2 G：热峰 + 入首双信号锚
        from engine.core.dragon_vein import dual_signal_anchor
        ent = getattr(dv, "entrance_point", None) if dv is not None else None
        peak_pt = (peak_rc[0], peak_rc[1]) if peak_rc else None
        anchor = dual_signal_anchor(peak_pt, ent, mpx, mpy)
        if anchor is not None:
            ar, ac = int(anchor[0]), int(anchor[1])
        primary = select_primary_dragon(
            dem, water=water, dragon_vein=dv,
            anchor_row=ar, anchor_col=ac,
        )
    except Exception:
        if primary is None:
            primary = primary_dragon
        yaoxia_points = list(yaoxia_points)

    entrance_xy: tuple[float, float] | None = None
    flow_az: float | None = None
    if primary is not None:
        entrance_xy = primary.entrance_xy
        flow_az = float(primary.flow_azimuth_deg)
    elif dv is not None and getattr(dv, "entrance_xy", None) is not None:
        entrance_xy = (float(dv.entrance_xy[0]), float(dv.entrance_xy[1]))

    def _primary_for_cand(c: AcupointCandidate):
        """相对候选重定向主龙。"""
        if primary is None:
            return None
        try:
            return reorient_primary_to_hole(
                dem, primary, int(c.row), int(c.col), water=water,
            )
        except Exception:
            return primary

    # —— Step 3: 搜穴 ——
    # P0：TWI 参与搜穴——用龙脉汇流累积（若有）
    flow_acc_arr = None
    if dv is not None and getattr(dv, "flow_acc", None) is not None:
        flow_acc_arr = dv.flow_acc

    # 明堂/热力区：步长更密 + NMS 略松，避免河湾橙心无点
    cands = search_candidates(
        dem, flow_acc=flow_acc_arr, tpi_radius_m=100, tpi_threshold=0.0,
        max_candidates=160, step=3, water=water, ban_buffer_m=_ban_m,
        qi_grid=qi_grid, qi_min_percentile=40.0,
        min_dist_m=140.0,
    )
    if len(cands) < 5:
        cands = search_candidates(
            dem, flow_acc=flow_acc_arr, tpi_radius_m=80, tpi_threshold=0.0,
            max_candidates=180, step=2, water=water, ban_buffer_m=_ban_m,
            qi_grid=qi_grid, qi_min_percentile=25.0,
            min_dist_m=120.0,
        )
    if len(cands) < 3:
        cands = search_candidates(
            dem, flow_acc=flow_acc_arr, tpi_radius_m=60, tpi_threshold=0.0,
            max_candidates=180, step=2, water=water, ban_buffer_m=_ban_m,
            qi_grid=qi_grid, qi_min_percentile=10.0,
            min_dist_m=100.0,
        )

    def _long_az_for(c: AcupointCandidate) -> float | None:
        # 来龙方位 = 相对本穴定向后的 源→入首（气向）
        p = _primary_for_cand(c)
        if p is not None:
            return float(p.flow_azimuth_deg)
        if flow_az is not None:
            return flow_az
        if entrance_xy is None:
            return None
        return _bearing_from_to(entrance_xy[0], entrance_xy[1], c.x, c.y)

    def _is_on_water(c: AcupointCandidate) -> bool:
        if water is None or getattr(water, "empty", True):
            return False
        try:
            if water.intersects(c.x, c.y, buffer_m=_ban_m):
                return True
        except Exception:
            pass
        return False

    def _qi_at(c: AcupointCandidate) -> float:
        if qi_grid is None:
            return 50.0
        r, col = int(c.row), int(c.col)
        if 0 <= r < qi_grid.shape[0] and 0 <= col < qi_grid.shape[1]:
            v = float(qi_grid[r, col])
            if np.isfinite(v):
                return v
        return 0.0

    # 与 search_candidates 一致：TPI 阈值随分辨率缩放
    from engine.core.terrain_analysis import _is_geographic as _is_geo_cell
    _xr, _yr = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geo_cell(dem.crs):
        _cell_m = float(min(_xr, _yr) * 111000.0)
    else:
        _cell_m = float(min(_xr, _yr))

    # 与 search_candidates 一致：有 flow_acc 则算 TWI 栅格供强制候选使用
    _twi_arr = None
    if flow_acc_arr is not None and np.any(flow_acc_arr > 0):
        try:
            from engine.core.acupoint import compute_twi
            _twi_arr = compute_twi(dem, flow_acc_arr)
        except Exception:
            _twi_arr = None

    # 水禁栅格：热峰若落在禁带，吸附到邻近干地高 qi
    _water_ban_grid = None
    if water is not None and not getattr(water, "empty", True):
        try:
            from engine.core.four_beasts_detect import water_distance_rasters
            _d_ban, _water_ban_grid = water_distance_rasters(
                dem, water, ban_buffer_m=float(_ban_m),
            )
        except Exception:
            _water_ban_grid = None

    def _snap_dry_high_qi(
        pr: int, pc: int, *, max_m: float = 280.0,
    ) -> tuple[int, int, float]:
        """保证落在干地；优先邻域 qi 最大。返回 (row, col, qi)。"""
        h0, w0 = dem.data.shape
        pr = int(np.clip(pr, 0, h0 - 1))
        pc = int(np.clip(pc, 0, w0 - 1))
        q0 = 0.0
        if qi_grid is not None and np.isfinite(qi_grid[pr, pc]):
            q0 = float(qi_grid[pr, pc])
        on_ban = (
            _water_ban_grid is not None
            and _water_ban_grid.shape == dem.data.shape
            and bool(_water_ban_grid[pr, pc])
        )
        if not on_ban and np.isfinite(dem.data[pr, pc]):
            return pr, pc, q0
        rad = max(2, int(round(max_m / max(min(mpx, mpy), 1.0))))
        best = None
        best_q = -1e18
        for rr in range(max(0, pr - rad), min(h0, pr + rad + 1)):
            for cc in range(max(0, pc - rad), min(w0, pc + rad + 1)):
                if not np.isfinite(dem.data[rr, cc]):
                    continue
                if (
                    _water_ban_grid is not None
                    and _water_ban_grid.shape == dem.data.shape
                    and _water_ban_grid[rr, cc]
                ):
                    continue
                d_m = float(np.hypot((rr - pr) * mpy, (cc - pc) * mpx))
                if d_m > max_m:
                    continue
                q = 0.0
                if qi_grid is not None and np.isfinite(qi_grid[rr, cc]):
                    q = float(qi_grid[rr, cc])
                # 近 + 高 qi
                sc = q - d_m * 0.02
                if sc > best_q:
                    best_q = sc
                    best = (rr, cc, q)
        if best is None:
            return pr, pc, q0
        return int(best[0]), int(best[1]), float(best[2])

    _edge_margin_px = max(4, int(round(120.0 / max(min(mpx, mpy), 1.0))))

    def _is_edge_cell(pr: int, pc: int) -> bool:
        h0, w0 = dem.data.shape
        return (
            pr < _edge_margin_px
            or pc < _edge_margin_px
            or pr >= h0 - _edge_margin_px
            or pc >= w0 - _edge_margin_px
        )

    def _make_cand_at(pr: int, pc: int, form_boost: float = 0.0) -> AcupointCandidate | None:
        from engine.core.acupoint import (
            classify_form, score_form, AcupointCandidate as _AC,
        )
        if not (0 <= pr < dem.data.shape[0] and 0 <= pc < dem.data.shape[1]):
            return None
        if not np.isfinite(dem.data[pr, pc]):
            return None
        # 图缘假点（常堆到 UI 左上角）禁止作为候选
        if _is_edge_cell(pr, pc):
            return None
        tpi_p = float(
            __import__("engine.core.terrain_analysis", fromlist=["tpi"]).tpi(
                dem, radius_m=100
            )[pr, pc]
        )
        if not np.isfinite(tpi_p):
            tpi_p = 0.0
        ls = float(slope_arr[pr, pc]) if np.isfinite(slope_arr[pr, pc]) else 2.0
        ft = classify_form(tpi_p, ls, cell_size_m=_cell_m)
        form_sc = int(score_form(tpi_p, ft, cell_size_m=_cell_m))
        twi_v = 0.0
        if _twi_arr is not None and np.isfinite(_twi_arr[pr, pc]):
            twi_v = float(_twi_arr[pr, pc])
            # 与 search_candidates 相同的 TWI 微调
            if 2.0 <= twi_v <= 10.0:
                form_sc = int(min(100, form_sc + 6))
            elif twi_v > 14.0:
                form_sc = int(max(0, form_sc - 8))
            elif 0 < twi_v < 1.0:
                form_sc = int(max(0, form_sc - 3))
        # 高 form_boost（热峰）时再抬：保证进排序前列
        form_sc = int(round(max(form_sc, form_boost)))
        if form_boost >= 70:
            form_sc = int(min(100, max(form_sc, 88)))
        px, py = dem.xy(pr, pc)
        if not (np.isfinite(px) and np.isfinite(py)):
            return None
        ac = _AC(
            row=pr, col=pc, x=float(px), y=float(py),
            elevation=float(dem.data[pr, pc]),
            tpi=tpi_p, twi=float(twi_v),
            form_type=ft,
            form_score=form_sc,
            local_slope=ls,
        )
        if _is_on_water(ac):
            return None
        return ac

    def _add_cand(c: AcupointCandidate | None, *, min_sep_m: float = 90.0) -> bool:
        if c is None:
            return False
        for i, e in enumerate(cands):
            d = float(np.hypot((e.row - c.row) * mpy, (e.col - c.col) * mpx))
            if d < min_sep_m:
                # 已有邻近点：更高 form 则替换
                if c.form_score > e.form_score:
                    cands[i] = c
                    return True
                return False
        cands.insert(0, c)
        return True

    # 入首邻域强制注入（先龙后穴）；拒绝图缘入首（假龙源/尾常贴边）
    if primary is not None:
        er, ec = primary.entrance_row, primary.entrance_col
        if not _is_edge_cell(er, ec):
            er, ec, _ = _snap_dry_high_qi(er, ec)
            if not _is_edge_cell(er, ec):
                ent_cand = _make_cand_at(er, ec, form_boost=60.0)
                _add_cand(ent_cand, min_sep_m=80.0)
            if qi_grid is not None:
                rad_px = max(3, int(round(600.0 / max(min(mpx, mpy), 1.0))))
                r0 = max(0, er - rad_px)
                r1 = min(qi_grid.shape[0], er + rad_px + 1)
                c0 = max(0, ec - rad_px)
                c1 = min(qi_grid.shape[1], ec + rad_px + 1)
                sub = qi_grid[r0:r1, c0:c1]
                if sub.size and np.isfinite(sub).any():
                    filled = np.where(np.isfinite(sub), sub, -np.inf)
                    # 窗内取 qi 最大且非图缘
                    best_loc = None
                    best_q = -1e18
                    for lr in range(sub.shape[0]):
                        for lc in range(sub.shape[1]):
                            pr, pc = int(r0 + lr), int(c0 + lc)
                            if _is_edge_cell(pr, pc):
                                continue
                            q = float(filled[lr, lc])
                            if q > best_q:
                                best_q = q
                                best_loc = (pr, pc, q)
                    if best_loc is not None:
                        pr, pc, qv = _snap_dry_high_qi(best_loc[0], best_loc[1])
                        if not _is_edge_cell(pr, pc):
                            loc = _make_cand_at(pr, pc, form_boost=max(float(qv), 70.0))
                            _add_cand(loc, min_sep_m=80.0)

    # 热峰 + 明堂高 qi 多种子强制注入（橙心必须有备选）
    peak_cand = None
    if peak_rc is not None:
        pr, pc, psc = peak_rc
        pr, pc, qv = _snap_dry_high_qi(pr, pc, max_m=320.0)
        boost = max(float(psc), float(qv), 85.0)
        peak_cand = _make_cand_at(pr, pc, form_boost=boost)
        if peak_cand is not None:
            # 更新 peak_rc 为吸附后坐标（供 is_qi_peak 匹配）
            peak_rc = (pr, pc, boost)
            _add_cand(peak_cand, min_sep_m=60.0)
            peak_cand = next(
                (c for c in cands if c.row == pr and c.col == pc), peak_cand
            )

    # 明堂高 qi 区「铺点」：网格取局部最高，强制空间分散
    # （禁止全部挤在热峰 200 m 内，右边大片橙心也要有候选）
    if qi_grid is not None and np.isfinite(qi_grid).any():
        try:
            valid = np.isfinite(qi_grid)
            if _water_ban_grid is not None and _water_ban_grid.shape == qi_grid.shape:
                valid = valid & (~_water_ban_grid)
            # 图缘不铺点
            em = _edge_margin_px
            valid[:em, :] = False
            valid[-em:, :] = False
            valid[:, :em] = False
            valid[:, -em:] = False
            q_valid = qi_grid[valid]
            if q_valid.size > 30:
                # 略降分位：覆盖更广的橙色明堂腹地
                q_thr = float(np.nanpercentile(q_valid, 58))
                hot = valid & (qi_grid >= q_thr)
                # 网格边长约 350–450 m：每格最多 1 个最高 qi 干点
                cell_m = 380.0
                cell_px = max(4, int(round(cell_m / max(min(mpx, mpy), 1.0))))
                h0, w0 = qi_grid.shape
                seeds: list[tuple[float, int, int]] = []
                for r0 in range(0, h0, cell_px):
                    for c0 in range(0, w0, cell_px):
                        r1 = min(h0, r0 + cell_px)
                        c1 = min(w0, c0 + cell_px)
                        block = hot[r0:r1, c0:c1]
                        if not block.any():
                            continue
                        sub = np.where(block, qi_grid[r0:r1, c0:c1], -np.inf)
                        li = int(np.argmax(sub))
                        lr, lc = np.unravel_index(li, sub.shape)
                        rr, cc = int(r0 + lr), int(c0 + lc)
                        qv = float(qi_grid[rr, cc])
                        seeds.append((qv, rr, cc))
                # 按 qi 降序；间距 ≥ 280 m 注入（拉开、覆盖橙心）
                seeds.sort(key=lambda t: -t[0])
                n_seed = 0
                peak_r = peak_rc[0] if peak_rc else None
                peak_c = peak_rc[1] if peak_rc else None
                near_peak_n = 0  # 热峰 350 m 内最多 2 个（含峰本身）
                for qv, rr, cc in seeds:
                    if n_seed >= 14:
                        break
                    rr, cc, q2 = _snap_dry_high_qi(rr, cc, max_m=120.0)
                    if _is_edge_cell(rr, cc):
                        continue
                    if peak_r is not None:
                        d_peak = float(np.hypot(
                            (rr - peak_r) * mpy, (cc - peak_c) * mpx,
                        ))
                        if d_peak < 350.0:
                            if near_peak_n >= 2:
                                continue
                            near_peak_n += 1
                    ac = _make_cand_at(rr, cc, form_boost=max(float(q2), float(qv), 72.0))
                    # 更大间距：避免 6/9/C-001 叠在峰上
                    if _add_cand(ac, min_sep_m=280.0):
                        n_seed += 1
        except Exception:
            pass

    # qi 分位：高 qi 保底龙分
    qi_p85, qi_p95 = 70.0, 85.0
    if qi_grid is not None and np.isfinite(qi_grid).any():
        valid_q = qi_grid[np.isfinite(qi_grid)]
        if valid_q.size > 20:
            qi_p85 = float(np.nanpercentile(valid_q, 85))
            qi_p95 = float(np.nanpercentile(valid_q, 95))

    def _fuse_overall(form_sc: float, qv: float, d_align: float) -> int:
        # 理论：橙心（明堂 qi）≈最优 → qi 权更大；龙只作贴脊加分
        # 高 qi 时抬高龙分下限
        da = d_align
        if qv >= qi_p95:
            da = max(da, 82.0)
        elif qv >= qi_p85:
            da = max(da, 70.0)
        elif qv >= 60.0:
            da = max(da, 55.0)
        return clamp_score(0.30 * form_sc + 0.56 * qv + 0.14 * da)

    # 剔除无效坐标 / 图缘点（防止 UI 堆左上角）
    cands = [
        c for c in cands
        if c is not None
        and np.isfinite(c.x) and np.isfinite(c.y)
        and not _is_edge_cell(int(c.row), int(c.col))
    ]

    results: list[FusedScore] = []
    peak_fused: FusedScore | None = None
    for i, c in enumerate(cands):
        if _is_on_water(c):
            continue
        if not (np.isfinite(c.x) and np.isfinite(c.y)):
            continue
        fused = score_candidate(
            dem, c, terrain, water, weights, slope_arr, aspect_arr,
            yaoxia_points=yaoxia_points,
            long_az_deg=_long_az_for(c),
        )
        fused.candidate_id = f"C-{i+1:03d}"
        if water is not None and not getattr(water, "empty", True):
            try:
                if water.intersects(c.x, c.y, buffer_m=0):
                    continue
            except Exception:
                pass
        qv = _qi_at(c)
        d_align = 50.0
        d_meta: dict[str, float] = {}
        p_loc = _primary_for_cand(c)
        if p_loc is not None:
            d_meta = dragon_alignment_score(
                int(c.row), int(c.col), p_loc, mpx, mpy,
            )
            d_align = float(d_meta.get("dragon_align", 50.0))
        fused.overall = _fuse_overall(float(fused.overall), qv, d_align)
        if fused.meta is not None:
            fused.meta["qi_field"] = round(qv, 1)
            fused.meta["dragon_align"] = round(d_align, 1)
            if d_meta:
                fused.meta["dist_entrance_m"] = round(
                    float(d_meta.get("dist_entrance_m", 0)), 1
                )
                fused.meta["dist_ridge_m"] = round(
                    float(d_meta.get("dist_ridge_m", 0)), 1
                )
            p_info = p_loc or primary
            if p_info is not None:
                fused.meta["primary_dragon"] = {
                    "ridge_idx": int(p_info.ridge_idx),
                    "flow_az": round(float(p_info.flow_azimuth_deg), 1),
                    "sit": round(float(p_info.sit_deg), 1),
                    "facing": round(float(p_info.facing_deg), 1),
                    "source_rc": [int(p_info.source_row), int(p_info.source_col)],
                    "entrance_rc": [int(p_info.entrance_row), int(p_info.entrance_col)],
                    "method": getattr(p_info, "method", ""),
                }
                fused.meta["long_az_deg"] = round(float(p_info.flow_azimuth_deg), 1)
            if peak_cand is not None and c.row == peak_cand.row and c.col == peak_cand.col:
                fused.meta["is_qi_peak"] = True
                peak_fused = fused
        if fused.overall >= min_score:
            results.append(fused)

    if not results and cands:
        scored: list[FusedScore] = []
        for i, c in enumerate(cands):
            if _is_on_water(c):
                continue
            fused = score_candidate(
                dem, c, terrain, water, weights, slope_arr, aspect_arr,
                yaoxia_points=yaoxia_points,
                long_az_deg=_long_az_for(c),
            )
            fused.candidate_id = f"C-{i+1:03d}"
            qv = _qi_at(c)
            d_align = 50.0
            p_loc = _primary_for_cand(c)
            if p_loc is not None:
                d_meta = dragon_alignment_score(
                    int(c.row), int(c.col), p_loc, mpx, mpy,
                )
                d_align = float(d_meta.get("dragon_align", 50.0))
            fused.overall = _fuse_overall(float(fused.overall), qv, d_align)
            scored.append(fused)
        scored.sort(key=lambda x: -x.overall)
        results = scored[: max(top_k, 5)]

    results.sort(key=lambda x: -x.overall)

    # —— 四象/少祖门禁：识别失败剔除；少祖高于玄武顺气加分（对标优质候选）——
    # 仅对预排前列做 detect（控时）；不足 top_k 时放宽侧砂数再扫一轮
    if require_beasts and results:
        pool_n = min(len(results), max(top_k * 5, 28))
        pool = results[:pool_n]
        gated: list[FusedScore] = []
        rejected_n = 0
        reject_reasons: dict[str, int] = {}

        def _apply_gate(r: FusedScore, *, min_side: int, elev_hard: bool) -> bool:
            nonlocal rejected_n
            # 找回对应 AcupointCandidate 行号
            crow = ccol = None
            for c in cands:
                if abs(c.x - r.x) < 1e-6 and abs(c.y - r.y) < 1e-6:
                    crow, ccol = int(c.row), int(c.col)
                    break
            if crow is None:
                try:
                    crow, ccol = dem.rowcol(r.x, r.y)
                except Exception:
                    rejected_n += 1
                    reject_reasons["no_rowcol"] = reject_reasons.get("no_rowcol", 0) + 1
                    return False
            p_loc = None
            try:
                # 构造临时 candidate 以 reorient 主龙
                tmp = AcupointCandidate(
                    row=crow, col=ccol, x=r.x, y=r.y,
                    elevation=r.elevation, tpi=0.0, twi=0.0,
                    form_type=r.form_type or "", form_score=0, local_slope=0.0,
                )
                p_loc = _primary_for_cand(tmp)
            except Exception:
                p_loc = primary
            ok, reason, info = _gate_beasts_for_hole(
                dem, crow, ccol, water,
                primary_dragon=p_loc or primary,
                dragon_vein=dv,
                require_shaozu_higher=elev_hard,
                min_side_beasts=min_side,
            )
            if r.meta is None:
                r.meta = {}
            r.meta["beasts_gate"] = info
            r.meta["beasts_gate_reason"] = reason
            if not ok:
                rejected_n += 1
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                return False
            bonus = int(info.get("shaozu_higher_bonus") or 0)
            if bonus:
                r.overall = clamp_score(float(r.overall) + bonus)
                r.scores = dict(r.scores or {})
                r.scores["shaozu_higher_bonus"] = bonus
            return True

        for r in pool:
            if _apply_gate(r, min_side=2, elev_hard=True):
                gated.append(r)
        # 不足：放宽侧砂 ≥1，仍要求祖>玄
        if len(gated) < top_k:
            seen = {(round(g.x, 3), round(g.y, 3)) for g in gated}
            for r in pool:
                key = (round(r.x, 3), round(r.y, 3))
                if key in seen:
                    continue
                if _apply_gate(r, min_side=1, elev_hard=True):
                    gated.append(r)
                    seen.add(key)
                if len(gated) >= top_k:
                    break
        # 仍不足：仅要求祖+玄存在（高程仍硬），侧砂 0
        if len(gated) < max(3, top_k // 2):
            seen = {(round(g.x, 3), round(g.y, 3)) for g in gated}
            for r in pool:
                key = (round(r.x, 3), round(r.y, 3))
                if key in seen:
                    continue
                if _apply_gate(r, min_side=0, elev_hard=True):
                    gated.append(r)
                    seen.add(key)
                if len(gated) >= top_k:
                    break

        if gated:
            gated.sort(key=lambda x: -x.overall)
            results = gated
            if peak_fused is not None:
                # 热峰也须过门禁，否则不强制入榜
                pf_ok = any(
                    abs(g.x - peak_fused.x) < 1e-6 and abs(g.y - peak_fused.y) < 1e-6
                    for g in gated
                )
                if not pf_ok:
                    if _apply_gate(peak_fused, min_side=1, elev_hard=True):
                        results.insert(0, peak_fused)
                        results.sort(key=lambda x: -x.overall)
                    else:
                        peak_fused = None
        # gated 空：保留原 results 但标 dirty（避免整局无穴）
        else:
            for r in results[:top_k]:
                if r.meta is not None:
                    r.meta["beasts_gate_relaxed"] = True
                    r.meta["beasts_gate_reject_summary"] = dict(reject_reasons)

        # 写汇总到首条 meta 便于调试
        if results and results[0].meta is not None:
            results[0].meta["beasts_gate_stats"] = {
                "rejected_n": rejected_n,
                "reasons": reject_reasons,
                "require_beasts": True,
            }

    results.sort(key=lambda x: -x.overall)

    # 热峰强制进入结果池（仅未开门禁或已通过时）
    if peak_fused is not None and not require_beasts:
        in_list = any(
            (getattr(r, "meta") or {}).get("is_qi_peak")
            or (abs(r.x - peak_fused.x) < 1e-6 and abs(r.y - peak_fused.y) < 1e-6)
            for r in results
        )
        if not in_list:
            results = [r for r in results if not (
                abs(r.x - peak_fused.x) < 1e-6 and abs(r.y - peak_fused.y) < 1e-6
            )]
            if peak_fused.overall < 78:
                peak_fused.overall = clamp_score(max(float(peak_fused.overall), 80))
            results.insert(0, peak_fused)
            results.sort(key=lambda x: -x.overall)
    elif peak_fused is not None and require_beasts:
        # 已在 gate 段处理；确保列表中热峰仍可优先
        in_list = any(
            (getattr(r, "meta") or {}).get("is_qi_peak")
            or (abs(r.x - peak_fused.x) < 1e-6 and abs(r.y - peak_fused.y) < 1e-6)
            for r in results
        )
        if in_list:
            # 提到前面再分散
            results.sort(key=lambda x: (
                0 if (getattr(x, "meta") or {}).get("is_qi_peak") else 1,
                -x.overall,
            ))

    # 空间分散 top_k：分数优先，但相邻候选 ≥ ~320 m
    # 避免 6/9/C-001 全挤在场评最高点，明堂右侧空无一穴
    from engine.core.terrain_analysis import _is_geographic as _is_geo_final
    _geo = _is_geo_final(dem.crs)

    def _sep_m(a: FusedScore, b: FusedScore) -> float:
        if _geo:
            mid_lat = (a.y + b.y) / 2.0
            dx = (a.x - b.x) * 111_000.0 * max(0.2, abs(np.cos(np.radians(mid_lat))))
            dy = (a.y - b.y) * 111_000.0
            return float(np.hypot(dx, dy))
        return float(np.hypot(a.x - b.x, a.y - b.y))

    min_sep_final = 320.0
    diverse: list[FusedScore] = []
    # 热峰优先入选
    if peak_fused is not None:
        diverse.append(peak_fused)
    for r in results:
        if peak_fused is not None and (
            (getattr(r, "meta") or {}).get("is_qi_peak")
            or (abs(r.x - peak_fused.x) < 1e-6 and abs(r.y - peak_fused.y) < 1e-6)
        ):
            continue
        if any(_sep_m(r, k) < min_sep_final for k in diverse):
            continue
        diverse.append(r)
        if len(diverse) >= top_k:
            break
    # 若分散后不足 top_k，放宽间距再补
    if len(diverse) < top_k:
        for r in results:
            if any(
                abs(r.x - k.x) < 1e-9 and abs(r.y - k.y) < 1e-9 for k in diverse
            ):
                continue
            if any(_sep_m(r, k) < min_sep_final * 0.55 for k in diverse):
                continue
            diverse.append(r)
            if len(diverse) >= top_k:
                break
    results = diverse

    for i, r in enumerate(results):
        r.rank = i + 1
        r.candidate_id = f"C-{i+1:03d}"
    out = results[:top_k]
    ctx = {
        "dragon_vein": dv,
        "primary_dragon": primary,
        "qi_grid": qi_grid,
        "qi_peak_rowcol": (peak_rc[0], peak_rc[1]) if peak_rc else None,
    }
    if return_context:
        return out, ctx
    return out

