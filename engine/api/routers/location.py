"""兼容入口：在线数据相关路由的统一 re-export。

实际实现已拆到：
  - search.py      /api/location/search
  - elevation.py   /api/elevation/fetch
  - water.py       /api/water/fetch
  - aoi.py         /api/aoi/*
  - cache.py       /api/cache/*
"""
from __future__ import annotations

from engine.api.routers.search import search_router
from engine.api.routers.elevation import elevation_router
from engine.api.routers.water import water_router
from engine.api.routers.cache import cache_router
from engine.api.routers.aoi import aoi_router

__all__ = [
    "search_router",
    "elevation_router",
    "water_router",
    "cache_router",
    "aoi_router",
]
