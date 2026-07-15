"""地形分析路由。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from engine.api.schemas.models import AnalyzeRequest, TerrainResult
from engine.core.terrain_analysis import analyze_terrain
from engine.io.dem import clip_dem, load_dem, reproject_dem

router = APIRouter()


@router.post("/analyze", response_model=TerrainResult)
def analyze(req: AnalyzeRequest):
    """加载 DEM 并分析区域级地形指标。

    若 CRS 为 EPSG:4326，自动重投影到 EPSG:3857 以便距离计算。
    """
    try:
        dem = load_dem(req.dem_path)
        if req.bbox:
            dem = clip_dem(dem, (req.bbox.minx, req.bbox.miny, req.bbox.maxx, req.bbox.maxy))
        # 距离计算需要投影坐标
        if str(dem.crs).upper().endswith("4326") or (dem.crs and "GEOG" in str(dem.crs).upper()):
            dem = reproject_dem(dem, "EPSG:3857")
        m = analyze_terrain(dem)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"DEM not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"analyze failed: {e}")

    return TerrainResult(
        bbox=list(dem.bounds),
        resolution_m=dem.resolution[0],
        mean_elevation=m.mean_elevation,
        max_elevation=m.max_elevation,
        min_elevation=m.min_elevation,
        relief=m.relief,
        mean_slope=m.mean_slope,
        max_slope=m.max_slope,
        dominant_aspect=m.dominant_aspect,
        aspect_degree=m.aspect_degree,
        terrain_position=m.terrain_position,
        terrain_roughness=m.terrain_roughness,
    )
