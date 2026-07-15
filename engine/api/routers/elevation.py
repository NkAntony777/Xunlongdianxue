"""高程拉取: POST /api/elevation/fetch"""
from __future__ import annotations

import base64
import io
import math
from typing import Optional

import numpy as np
import rasterio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine.core.aoi_limits import (
    MIN_RADIUS_KM, MAX_RADIUS_KM, DEFAULT_RADIUS_KM,
    validate_radius_km, radius_quality,
)
from engine.io.elevation_api import fetch_dem_for_analysis

elevation_router = APIRouter()


class ElevationFetchRequest(BaseModel):
    lon: float = Field(..., ge=-180, le=180, description="中心经度")
    lat: float = Field(..., ge=-85, le=85, description="中心纬度")
    radius_km: float = Field(
        DEFAULT_RADIUS_KM,
        description=f"分析半径 km（允许 {MIN_RADIUS_KM}–{MAX_RADIUS_KM}）",
    )
    resolution_m: Optional[float] = Field(None, ge=10, le=2000, description="米/像素, 默认自动")
    target_crs: str = Field("EPSG:3857", description="目标 CRS")


@elevation_router.post("/fetch", summary="按 bbox 拉 ESRI World Elevation GeoTIFF")
def elevation_fetch(req: ElevationFetchRequest):
    """拉一个 bbox 范围的 DEM, 返回 GeoTIFF base64 + 摘要."""
    try:
        radius_km = validate_radius_km(req.radius_km)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        dem = fetch_dem_for_analysis(
            req.lon, req.lat, radius_km,
            resolution_m=req.resolution_m,
            target_crs=req.target_crs,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ESRI DEM 拉取失败: {e}")

    buf = io.BytesIO()
    profile = {
        "driver": "GTiff",
        "height": dem.height,
        "width": dem.width,
        "count": 1,
        "dtype": "float32",
        "crs": dem.crs,
        "transform": dem.transform,
        "nodata": dem.nodata if dem.nodata is not None else -9999.0,
    }
    with rasterio.open(buf, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(dem.data), dem.data, -9999.0).astype("float32"), 1)
    tif_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "lon": req.lon, "lat": req.lat, "radius_km": radius_km,
        "radius_quality": radius_quality(radius_km),
        "shape": [dem.height, dem.width],
        "bounds": list(dem.bounds),
        "bbox_lonlat": [
            req.lon - radius_km / (111.32 * math.cos(math.radians(req.lat))),
            req.lat - radius_km / 111.32,
            req.lon + radius_km / (111.32 * math.cos(math.radians(req.lat))),
            req.lat + radius_km / 111.32,
        ],
        "crs": str(dem.crs),
        "resolution_m": float(abs(dem.resolution[0])),
        "elevation_min": float(np.nanmin(dem.data)) if np.isfinite(dem.data).any() else None,
        "elevation_max": float(np.nanmax(dem.data)) if np.isfinite(dem.data).any() else None,
        "elevation_mean": float(np.nanmean(dem.data)) if np.isfinite(dem.data).any() else None,
        "dem_source": "terrarium",  # 主源；内部可 fallback esri
        "geotiff_base64": tif_b64,
    }
