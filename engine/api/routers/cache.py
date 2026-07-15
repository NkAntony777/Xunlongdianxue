"""缓存与临时文件: /api/cache/*"""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from engine.io.elevation_api import (
    ping as ping_dem,
    _MEM_CACHE as _DEM_CACHE,
)
from engine.io.overpass import (
    ping as ping_water,
    _MEM_CACHE as _WATER_CACHE,
)
from engine.api.routers._tmp_store import TMP_DIR

cache_router = APIRouter()


@cache_router.get("/info", summary="缓存状态")
def cache_info():
    """返回 DEM / 水系内存缓存状态."""
    return {
        "dem": {
            "entries": len(_DEM_CACHE),
            "max": 16,
            "keys": list(_DEM_CACHE.keys()),
        },
        "water": {
            "entries": len(_WATER_CACHE),
            "max": 16,
            "keys": list(_WATER_CACHE.keys()),
        },
        "services": {
            "esri_dem": ping_dem(),
            "overpass_water": ping_water(),
        },
    }


@cache_router.post("/clear", summary="清空内存缓存")
def cache_clear():
    _DEM_CACHE.clear()
    _WATER_CACHE.clear()
    n = 0
    for p in TMP_DIR.glob("*"):
        try:
            p.unlink()
            n += 1
        except Exception:
            pass
    return {"cleared": True, "dem": 0, "water": 0, "tmp_files": n}


@cache_router.post("/save_tmp", summary="保存上传的 GeoTIFF 临时文件")
async def save_tmp(request: Request):
    """接收前端传来的 GeoTIFF 二进制, 存为临时文件, 返回路径."""
    body = await request.body()
    if not body or len(body) < 100:
        raise HTTPException(status_code=400, detail="文件内容为空或太小")
    name = f"dem_{int(time.time()*1000)}.tif"
    path = TMP_DIR / name
    path.write_bytes(body)
    return {"path": str(path), "size": len(body)}


@cache_router.post("/save_text", summary="保存上传的文本 (GeoJSON) 临时文件")
async def save_text(request: Request):
    """接收前端传来的 GeoJSON 文本, 存为临时文件, 返回路径."""
    body = await request.body()
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON 解析失败: {e}")
    if data.get("content"):
        data = json.loads(data["content"])
    if data.get("type") != "FeatureCollection":
        raise HTTPException(status_code=400, detail="必须是 FeatureCollection")
    name = f"water_{int(time.time()*1000)}.geojson"
    path = TMP_DIR / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return {"path": str(path), "size": path.stat().st_size}
