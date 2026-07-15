"""候选穴搜索路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from engine.api.schemas.models import (
    AnalyzeRequest,
    CandidateItem,
    CandidatesResponse,
    TerrainResult,
)
from engine.core.fengshui_score import find_and_rank_candidates, to_json
from engine.core.terrain_analysis import analyze_terrain
from engine.io.dem import clip_dem, load_dem, reproject_dem
from engine.io.rivers import load_water

router = APIRouter()


@router.post("/search", response_model=CandidatesResponse)
def search(req: AnalyzeRequest):
    """搜索候选穴并按综合分排序。"""
    try:
        dem = load_dem(req.dem_path)
        if req.bbox:
            dem = clip_dem(dem, (req.bbox.minx, req.bbox.miny, req.bbox.maxx, req.bbox.maxy))
        if str(dem.crs).upper().endswith("4326") or (dem.crs and "GEOG" in str(dem.crs).upper()):
            dem = reproject_dem(dem, "EPSG:3857")

        water = load_water(req.water_path) if req.water_path else None
        results = find_and_rank_candidates(
            dem, water, top_k=req.top_k, min_score=req.min_score
        )
        terrain = analyze_terrain(dem)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")

    items = [
        CandidateItem(
            id=r.candidate_id,
            rank=r.rank,
            x=r.x,
            y=r.y,
            elevation_m=r.elevation,
            form_type=r.form_type,
            overall_score=r.overall,
            scores=r.scores,
            geography=r.geography,
            messages=r.messages,
            meta=r.meta or {},
        )
        for r in results
    ] 
    metadata = {
        "dem": req.dem_path,
        "water": req.water_path,
        "bbox": list(dem.bounds),
        "resolution_m": dem.resolution[0],
        "terrain": {
            "mean_elevation": terrain.mean_elevation,
            "relief": terrain.relief,
            "mean_slope": terrain.mean_slope,
            "dominant_aspect": terrain.dominant_aspect,
            "terrain_position": terrain.terrain_position,
        },
    }
    return CandidatesResponse(metadata=metadata, candidates=items)


@router.post("/geojson")
def search_geojson(req: AnalyzeRequest):
    """返回候选穴 GeoJSON FeatureCollection。"""
    try:
        dem = load_dem(req.dem_path)
        if req.bbox:
            dem = clip_dem(dem, (req.bbox.minx, req.bbox.miny, req.bbox.maxx, req.bbox.maxy))
        if str(dem.crs).upper().endswith("4326") or (dem.crs and "GEOG" in str(dem.crs).upper()):
            dem = reproject_dem(dem, "EPSG:3857")
        water = load_water(req.water_path) if req.water_path else None
        results = find_and_rank_candidates(dem, water, top_k=req.top_k, min_score=req.min_score)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")

    from engine.core.fengshui_score import to_geojson

    return to_geojson(results)
