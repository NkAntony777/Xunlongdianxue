"""水系拉取: POST /api/water/fetch"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine.core.aoi_limits import DEFAULT_RADIUS_KM, validate_radius_km
from engine.io.elevation_api import bbox_from_center
from engine.io.overpass import fetch_water_for_analysis

water_router = APIRouter()


class WaterFetchRequest(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-85, le=85)
    radius_km: float = Field(DEFAULT_RADIUS_KM)
    # 前端落盘用 4326，避免 GeoJSON 无 CRS 时被误当成经纬度/投影混用
    target_crs: str = Field("EPSG:4326")


@water_router.post("/fetch", summary="按 bbox 拉 OSM 水系 GeoJSON")
def water_fetch(req: WaterFetchRequest):
    """按 bbox 拉水系。Overpass 失败时降级为空图层 + warning，不硬中断分析。

    返回的 features 默认 EPSG:4326，便于落盘后再由 layers 按 DEM CRS 重投影。
    """
    try:
        radius_km = validate_radius_km(req.radius_km)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    bbox = bbox_from_center(req.lon, req.lat, radius_km)
    warning = None
    target = req.target_crs or "EPSG:4326"
    try:
        gdf = fetch_water_for_analysis(
            req.lon, req.lat, radius_km, target_crs=target,
        )
    except Exception as e:
        try:
            gdf = fetch_water_for_analysis(
                req.lon, req.lat, radius_km,
                target_crs=target,
                allow_empty_on_error=True,
            )
            warning = f"水系服务不稳定，已降级：{e}"
        except Exception as e2:
            return {
                "count": 0,
                "features": [],
                "bbox_lonlat": list(bbox),
                "crs": "EPSG:4326",
                "warning": f"水系拉取失败，已跳过：{e2}",
                "degraded": True,
            }

    if gdf is None or gdf.empty:
        return {
            "count": 0,
            "features": [],
            "bbox_lonlat": list(bbox),
            "crs": "EPSG:4326",
            "warning": warning or "范围内未检索到水系要素",
            "degraded": bool(warning),
        }

    # 强制清理无效几何
    try:
        from engine.io.rivers import _clean_gdf
        gdf = _clean_gdf(gdf)
    except Exception:
        pass

    gj = json.loads(gdf.to_json())
    return {
        "count": int(len(gdf)),
        "bbox_lonlat": list(bbox),
        "crs": str(gdf.crs) if gdf.crs else target,
        "features": gj.get("features", []),
        "warning": warning,
        "degraded": bool(warning),
    }
