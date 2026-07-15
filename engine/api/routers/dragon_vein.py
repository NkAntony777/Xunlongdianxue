"""龙脉识别路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from engine.api.schemas.models import (
    AnalyzeRequest,
    DragonVeinResponse,
    RidgeLineItem,
)
from engine.core.dragon_vein import analyze_dragon_vein
from engine.io.dem import clip_dem, load_dem, reproject_dem

router = APIRouter()


@router.post("/extract", response_model=DragonVeinResponse)
def extract(req: AnalyzeRequest):
    """提取山脊线 + 龙脉分级 + 入首点定位。"""
    try:
        dem = load_dem(req.dem_path)
        if req.bbox:
            dem = clip_dem(dem, (req.bbox.minx, req.bbox.miny, req.bbox.maxx, req.bbox.maxy))
        if str(dem.crs).upper().endswith("4326") or (dem.crs and "GEOG" in str(dem.crs).upper()):
            dem = reproject_dem(dem, "EPSG:3857")
        dv = analyze_dragon_vein(dem)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"extract failed: {e}")

    major = [
        RidgeLineItem(
            length_m=r.length_m,
            mean_elevation=r.mean_elevation,
            max_elevation=r.max_elevation,
            sinuosity=r.sinuosity,
            feature_significance=r.feature_significance,
            coords=r.coords.tolist(),
        )
        for r in dv.major_ridges[:20]
    ]
    return DragonVeinResponse(
        n_ridges=len(dv.ridge_lines),
        n_major=len(dv.major_ridges),
        entrance_xy=list(dv.entrance_xy) if dv.entrance_xy else None,
        major_ridges=major,
        n_yaoxia=len(dv.yaoxia or []),
    )
