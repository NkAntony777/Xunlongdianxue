"""AOI analysis use-case (orchestration, no HTTP).

Pipeline order (theory: dragon first):
  1) score field (qi)
  2) analyze_dragon_vein (once)
  3) find_and_rank_candidates (shared vein)
  4) hole center = field peak (or top candidate)
  5) detect_four_beasts at hole with primary dragon

Callers: api.routers.layers / CLI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from engine.core.field.qi import compute_score_grid, find_score_peak, score_peak_xy
from engine.core.four_beasts_detect import detect_four_beasts
from engine.core.fengshui_score import find_and_rank_candidates
from engine.core.dragon_vein import analyze_dragon_vein, reorient_primary_to_hole
from engine.io.dem import DEM


@dataclass
class AnalyzeAoiResult:
    dem: DEM
    water: Any = None
    score_grid: Optional[np.ndarray] = None
    peak_rowcol: Optional[tuple[int, int]] = None
    peak_xy: Optional[tuple[float, float]] = None
    peak_score: Optional[float] = None
    four_beasts: Any = None
    four_beasts_meta: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    context: dict = field(default_factory=dict)
    center_row: Optional[int] = None
    center_col: Optional[int] = None
    center_source: str = "dem_center"
    dragon_vein: Any = None
    primary_dragon: Any = None


def analyze_aoi(
    dem: DEM,
    water=None,
    *,
    top_k: int = 10,
    min_score: int = 0,
    sample_step: int = 4,
    require_beasts: bool = True,
    dragon_vein=None,
    primary_dragon=None,
    score_grid: np.ndarray | None = None,
    center_row: int | None = None,
    center_col: int | None = None,
) -> AnalyzeAoiResult:
    """Full AOI analysis without rendering."""
    # 1) score field
    grid = score_grid if score_grid is not None else compute_score_grid(
        dem, sample_step=sample_step, water=water,
    )

    peak_info = score_peak_xy(dem, grid)
    if peak_info is not None:
        (pr, pc), peak_xy, peak_score = peak_info
    else:
        pr = pc = None
        peak_xy = None
        peak_score = None
        peak = find_score_peak(grid)
        if peak is not None:
            pr, pc, peak_score = int(peak[0]), int(peak[1]), float(peak[2])
            try:
                peak_xy = dem.xy(pr, pc)
            except Exception:
                peak_xy = None

    # 2) dragon vein once
    dv = dragon_vein
    if dv is None:
        try:
            dv = analyze_dragon_vein(dem, min_length_m=120.0, water=water)
        except Exception:
            dv = None

    # 3) rank candidates (shares vein)
    ranked, ctx = find_and_rank_candidates(
        dem,
        water=water,
        top_k=top_k,
        min_score=min_score,
        dragon_vein=dv,
        primary_dragon=primary_dragon,
        return_context=True,
        require_beasts=require_beasts,
    )
    dv = (ctx or {}).get("dragon_vein") or dv
    primary = (ctx or {}).get("primary_dragon") or primary_dragon
    qi_peak_rc = (ctx or {}).get("qi_peak_rowcol")
    if qi_peak_rc is not None and pr is None:
        pr, pc = int(qi_peak_rc[0]), int(qi_peak_rc[1])

    # 4) hole center: explicit > field peak > top candidate
    center_source = "dem_center"
    crow = ccol = None
    if center_row is not None and center_col is not None:
        crow = int(np.clip(center_row, 0, dem.height - 1))
        ccol = int(np.clip(center_col, 0, dem.width - 1))
        center_source = "explicit"
    elif pr is not None and pc is not None:
        crow, ccol = int(pr), int(pc)
        center_source = "score_field_peak"
    elif ranked:
        try:
            from rasterio.transform import rowcol
            top = ranked[0]
            r0, c0 = rowcol(dem.transform, top.x, top.y)
            crow, ccol = int(r0), int(c0)
            center_source = "top_candidate"
        except Exception:
            crow, ccol = dem.height // 2, dem.width // 2
            center_source = "dem_center"
    else:
        crow, ccol = dem.height // 2, dem.width // 2
        center_source = "dem_center"

    if primary is not None and crow is not None:
        try:
            primary = reorient_primary_to_hole(
                dem, primary, crow, ccol, water=water,
            )
        except Exception:
            pass

    # 5) four beasts at hole
    fb = None
    fb_meta: dict = {}
    if crow is not None and ccol is not None:
        try:
            fb = detect_four_beasts(
                dem,
                center_row=crow,
                center_col=ccol,
                water=water,
                dragon_vein=dv,
                primary_dragon=primary,
            )
            fb_meta = getattr(fb, "meta", None) or {}
        except Exception as e:
            fb_meta = {"error": str(e)}

    # normalize peak_xy
    pxy = None
    if peak_xy is not None:
        try:
            pxy = (float(peak_xy[0]), float(peak_xy[1]))
        except Exception:
            pxy = None
    if pxy is None and crow is not None:
        try:
            pxy = dem.xy(crow, ccol)
            pxy = (float(pxy[0]), float(pxy[1]))
        except Exception:
            pxy = None

    return AnalyzeAoiResult(
        dem=dem,
        water=water,
        score_grid=grid,
        peak_rowcol=(int(pr), int(pc)) if pr is not None and pc is not None else None,
        peak_xy=pxy,
        peak_score=float(peak_score) if peak_score is not None else None,
        four_beasts=fb,
        four_beasts_meta=fb_meta,
        candidates=list(ranked or []),
        context=dict(ctx or {}),
        center_row=crow,
        center_col=ccol,
        center_source=center_source,
        dragon_vein=dv,
        primary_dragon=primary,
    )


def structured_from_aoi(aoi: AnalyzeAoiResult) -> dict[str, Any]:
    """Build the structured JSON block for /api/layers/* from AnalyzeAoiResult."""
    from engine.core.fengshui_score import _sanitize as _sanitize_floats
    from engine.core.render import (
        four_beasts_geojson,
        candidates_geojson,
        ridges_geojson,
    )

    dem = aoi.dem
    fb = aoi.four_beasts
    results = aoi.candidates
    dv = aoi.dragon_vein
    primary = aoi.primary_dragon

    fb_dict: dict[str, dict[str, float]] = {}
    if fb is not None:
        for k in ("shaozu", "xuanwu", "zhuque", "qinglong", "baihu"):
            v = getattr(fb, k, None)
            if v:
                fb_dict[k] = {"x": float(v[0]), "y": float(v[1])}
                bm = (fb.meta or {}).get("beasts", {}).get(k)
                if isinstance(bm, dict):
                    for mk in ("elev_m", "dist_m", "bearing_deg", "row", "col", "on_ridge"):
                        if mk in bm and bm[mk] is not None:
                            fb_dict[k][mk] = bm[mk]

    center_xy = None
    if fb is not None and fb.center:
        center_xy = {"x": float(fb.center[0]), "y": float(fb.center[1])}
    elif aoi.peak_xy is not None:
        center_xy = {"x": float(aoi.peak_xy[0]), "y": float(aoi.peak_xy[1])}

    facing = float(getattr(fb, "facing", 180.0) or 180.0) if fb else 180.0
    sit = float(getattr(fb, "sit", (facing + 180) % 360) or 0.0) if fb else (facing + 180) % 360

    ridges_geo = {"type": "FeatureCollection", "features": []}
    try:
        ridge_payload = []
        if dv is not None and getattr(dv, "ridge_lines", None):
            order = list(range(len(dv.ridge_lines)))
            if primary is not None and 0 <= primary.ridge_idx < len(order):
                order = [primary.ridge_idx] + [i for i in order if i != primary.ridge_idx]
            for rank, i in enumerate(order[:8], 1):
                r = dv.ridge_lines[i]
                world_coords = []
                for rr, cc in r.coords:
                    try:
                        x, y = dem.xy(int(rr), int(cc))
                        world_coords.append([float(x), float(y)])
                    except Exception:
                        continue
                if len(world_coords) < 2:
                    continue
                ridge_payload.append({
                    "coords": world_coords,
                    "rank": rank,
                    "is_primary": bool(primary and i == primary.ridge_idx),
                })
        ridges_geo = ridges_geojson(ridge_payload)
    except Exception:
        pass

    cands = [
        {
            "id": r.candidate_id,
            "rank": r.rank,
            "x": r.x,
            "y": r.y,
            "overall_score": r.overall,
            "form_type": r.form_type,
            "scores": r.scores,
            "geography": r.geography,
            "meta": getattr(r, "meta", None) or {},
        }
        for r in results
    ]
    cands_geo = candidates_geojson(cands)

    primary_meta = None
    if primary is not None:
        ridge_role = ridge_order = None
        try:
            if dv is not None and 0 <= primary.ridge_idx < len(dv.ridge_lines):
                rr = dv.ridge_lines[primary.ridge_idx]
                ridge_role = getattr(rr, "role", None)
                ridge_order = getattr(rr, "strahler_order", None)
        except Exception:
            pass
        primary_meta = {
            "method": primary.method,
            "score": round(primary.score, 3),
            "flow_azimuth_deg": round(primary.flow_azimuth_deg, 1),
            "sit_deg": round(primary.sit_deg, 1),
            "facing_deg": round(primary.facing_deg, 1),
            "length_m": round(primary.length_m, 1),
            "downhill_m": round(primary.downhill_m, 1),
            "entrance": {
                "row": primary.entrance_row,
                "col": primary.entrance_col,
                "x": primary.entrance_xy[0],
                "y": primary.entrance_xy[1],
            },
            "ridge_idx": primary.ridge_idx,
            "strahler_order": ridge_order,
            "ridge_role": ridge_role,
            "detail": primary.meta,
        }

    qi_peak_rc = aoi.peak_rowcol
    return _sanitize_floats({
        "center": center_xy,
        "center_source": aoi.center_source,
        "center_row": int(aoi.center_row) if aoi.center_row is not None else None,
        "center_col": int(aoi.center_col) if aoi.center_col is not None else None,
        "facing": facing,
        "sit": sit,
        "facing_method": getattr(fb, "facing_method", "") if fb else "",
        "four_beasts": fb_dict,
        "four_beasts_meta": aoi.four_beasts_meta or {},
        "four_beasts_geojson": four_beasts_geojson(fb_dict),
        "ridges_geojson": ridges_geo,
        "candidates": cands,
        "candidates_geojson": cands_geo,
        "primary_dragon": primary_meta,
        "pipeline": "dragon_first",
        "qi_peak": (
            {"row": int(qi_peak_rc[0]), "col": int(qi_peak_rc[1])}
            if qi_peak_rc is not None else None
        ),
    })
