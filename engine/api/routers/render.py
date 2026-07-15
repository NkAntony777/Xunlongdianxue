"""可视化渲染路由。"""
from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from engine.api.schemas.models import AnalyzeRequest
from engine.core.fengshui_score import find_and_rank_candidates
from engine.core.four_beasts_detect import (
    detect_four_beasts,
    compute_score_grid,
    find_score_peak,
)
from engine.core.dragon_vein import analyze_dragon_vein
from engine.core.render import render_dem_overlay, render_combined
from engine.io.dem import clip_dem, load_dem, reproject_dem
from engine.io.rivers import load_water

router = APIRouter()


def _load_dem_with_bbox(req):
    dem = load_dem(req.dem_path)
    if req.bbox:
        dem = clip_dem(dem, (req.bbox.minx, req.bbox.miny, req.bbox.maxx, req.bbox.maxy))
    if str(dem.crs).upper().endswith("4326") or (dem.crs and "GEOG" in str(dem.crs).upper()):
        dem = reproject_dem(dem, "EPSG:3857")
    return dem


@router.post("/dem")
def render_dem(req: AnalyzeRequest):
    """渲染 DEM + 等高线 PNG（base64）。"""
    try:
        dem = _load_dem_with_bbox(req)
        result = render_dem_overlay(
            dem,
            contour_interval=getattr(req, "contour_interval", 10.0),
            colormap="Greys",
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"render failed: {e}")
    return {
        "width": result.width,
        "height": result.height,
        "bbox": list(result.bbox),
        "png_base64": result.png_base64,
        "legend": result.legend,
    }


@router.post("/score-grid")
def render_score_grid(req: AnalyzeRequest, sample_step: int = 4):
    """计算并返回风水评分栅格（PNG）。"""
    import numpy as np
    try:
        dem = _load_dem_with_bbox(req)
        water = load_water(req.water_path) if req.water_path else None
        grid = compute_score_grid(dem, sample_step=sample_step, water=water)
        from engine.core.render import render_score_grid as _render
        result = _render(dem, grid)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"score-grid failed: {e}")
    return {
        "width": result.width,
        "height": result.height,
        "bbox": list(result.bbox),
        "png_base64": result.png_base64,
        "legend": result.legend,
        "score_stats": {
            "min": float(np.nanmin(grid)),
            "max": float(np.nanmax(grid)),
            "mean": float(np.nanmean(grid)),
        },
    }


@router.post("/combined")
def render_combined_view(req: AnalyzeRequest, sample_step: int = 3):
    """综合渲染：DEM + 等高线 + 评分热力 + 四象标点 + 山脊线 + 候选穴。"""
    import numpy as np

    try:
        dem = _load_dem_with_bbox(req)

        water = load_water(req.water_path) if req.water_path else None
        results = find_and_rank_candidates(dem, water, top_k=10, min_score=0)

        # 评分场平滑峰值 = 穴 = 四象中心（与 layers/all 一致）
        grid = compute_score_grid(dem, sample_step=sample_step, water=water)
        peak = find_score_peak(grid)
        if peak is not None:
            crow, ccol, _ = peak
        elif results:
            from rasterio.transform import rowcol
            ref = results[0]
            crow, ccol = rowcol(dem.transform, ref.x, ref.y)
            crow, ccol = int(crow), int(ccol)
            if not (0 <= crow < dem.height and 0 <= ccol < dem.width):
                crow, ccol = dem.height // 2, dem.width // 2
        else:
            crow, ccol = dem.height // 2, dem.width // 2

        # 四象位置（传入水系以背山面水推朝向）
        fb = detect_four_beasts(dem, center_row=crow, center_col=ccol, water=water)
        fb_dict = {}
        if fb.shaozu:
            fb_dict["shaozu"] = {"x": fb.shaozu[0], "y": fb.shaozu[1]}
        if fb.xuanwu:
            fb_dict["xuanwu"] = {"x": fb.xuanwu[0], "y": fb.xuanwu[1]}
        if fb.zhuque:
            fb_dict["zhuque"] = {"x": fb.zhuque[0], "y": fb.zhuque[1]}
        if fb.qinglong:
            fb_dict["qinglong"] = {"x": fb.qinglong[0], "y": fb.qinglong[1]}
        if fb.baihu:
            fb_dict["baihu"] = {"x": fb.baihu[0], "y": fb.baihu[1]}
        if fb.center:
            fb_dict["center"] = {"x": float(fb.center[0]), "y": float(fb.center[1])}

        # 龙脉
        dv = analyze_dragon_vein(dem)
        ridges = [
            {"coords": r.coords.tolist()}
            for r in dv.major_ridges[:5]
        ]

        # 候选穴
        cands = [
            {"id": r.candidate_id, "x": r.x, "y": r.y}
            for r in (results or [])[:10]
        ]

        result = render_combined(
            dem,
            four_beasts=fb_dict,
            score_grid=grid,
            ridges=ridges,
            candidates=cands,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"combined render failed: {e}")
    return {
        "width": result.width,
        "height": result.height,
        "bbox": list(result.bbox),
        "png_base64": result.png_base64,
        "legend": result.legend,
        "four_beasts": fb_dict,
        "n_candidates": len(cands),
        "n_ridges": len(ridges),
    }


@router.post("/four-beasts")
def render_four_beasts(req: AnalyzeRequest):
    """识别四象位置 + 主轴（穴心 = 评分场平滑峰值）。"""
    try:
        dem = _load_dem_with_bbox(req)
        water = load_water(req.water_path) if req.water_path else None
        grid = compute_score_grid(dem, sample_step=4, water=water)
        peak = find_score_peak(grid)
        if peak is not None:
            crow, ccol, peak_score = peak
            source = "score_field_peak"
        else:
            results = find_and_rank_candidates(dem, water, top_k=5, min_score=0)
            if not results:
                raise HTTPException(status_code=400, detail="no candidates")
            from rasterio.transform import rowcol
            ref = results[0]
            crow, ccol = rowcol(dem.transform, ref.x, ref.y)
            crow, ccol = int(crow), int(ccol)
            peak_score = None
            source = "top_candidate"
        fb = detect_four_beasts(
            dem, center_row=int(crow), center_col=int(ccol), water=water,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"four-beasts failed: {e}")

    cx, cy = (float(fb.center[0]), float(fb.center[1])) if fb.center else (None, None)
    out = {
        "center": {"x": cx, "y": cy},
        "center_source": source,
        "peak_score": peak_score,
    }
    for name, pt in [("shaozu", fb.shaozu), ("xuanwu", fb.xuanwu), ("zhuque", fb.zhuque),
                     ("qinglong", fb.qinglong), ("baihu", fb.baihu)]:
        if pt:
            out[name] = {"x": pt[0], "y": pt[1]}
    return out