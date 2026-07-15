"""FastAPI 应用主入口。"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("xunlong.api")

# 默认数据路径
_ROOT = Path(__file__).resolve().parents[2]
_DEM_CANDIDATES = [
    _ROOT / "data" / "langzhong_cop30.tif",
    _ROOT / "data" / "langzhong_dem.tif",
]
_WATER_CANDIDATES = [
    _ROOT / "data" / "langzhong_rivers_osm.geojson",
    _ROOT / "data" / "langzhong_rivers.geojson",
]
DEM_DEFAULT = os.environ.get(
    "XUNLONG_DEM",
    str(next((p for p in _DEM_CANDIDATES if p.exists()), _DEM_CANDIDATES[-1])),
)
WATER_DEFAULT = os.environ.get(
    "XUNLONG_WATER",
    str(next((p for p in _WATER_CANDIDATES if p.exists()), _WATER_CANDIDATES[-1])),
)
LLM_API_KEY = os.environ.get("XUNLONG_LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("XUNLONG_LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("XUNLONG_LLM_MODEL", "deepseek-chat")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Xunlong Engine starting ...")
    log.info(f"Default DEM: {DEM_DEFAULT}")
    log.info(f"Default Water: {WATER_DEFAULT}")
    yield
    log.info("Xunlong Engine shutting down.")


app = FastAPI(
    title="Xunlong Engine API",
    description="寻龙点穴地形分析引擎 - 后端 API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由
from engine.api.routers import (
    analysis, candidates, dragon_vein, report, llm, render, layers,  # noqa: E402
)
from engine.api.routers.search import search_router
from engine.api.routers.elevation import elevation_router
from engine.api.routers.water import water_router
from engine.api.routers.cache import cache_router
from engine.api.routers.aoi import aoi_router

app.include_router(analysis.router, prefix="/api/terrain", tags=["terrain"])
app.include_router(candidates.router, prefix="/api/candidates", tags=["candidates"])
app.include_router(dragon_vein.router, prefix="/api/dragon-vein", tags=["dragon-vein"])
app.include_router(report.router, prefix="/api/report", tags=["report"])
app.include_router(llm.router, prefix="/api/llm", tags=["llm"])
app.include_router(render.router, prefix="/api/render", tags=["render"])
app.include_router(layers.router, prefix="/api/layers", tags=["layers"])
app.include_router(search_router,   prefix="/api/location",  tags=["location"])
app.include_router(elevation_router, prefix="/api/elevation", tags=["elevation"])
app.include_router(water_router,    prefix="/api/water",     tags=["water"])
app.include_router(cache_router,    prefix="/api/cache",     tags=["cache"])
app.include_router(aoi_router,      prefix="/api/aoi",       tags=["aoi"])


# 健康检查
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "xunlong-engine",
        "version": "0.2.0",
        "llm_configured": bool(LLM_API_KEY),
    }


# 元信息
@app.get("/api/info")
def info():
    return {
        "name": "Xunlong Engine",
        "version": "0.2.0",
        "default_dem": DEM_DEFAULT,
        "default_water": WATER_DEFAULT,
        "llm": {
            "base_url": LLM_BASE_URL,
            "model": LLM_MODEL,
            "configured": bool(LLM_API_KEY),
        },
        "endpoints": [
            "GET  /api/health",
            "GET  /api/info",
            "POST /api/terrain/analyze",
            "POST /api/candidates/search",
            "POST /api/candidates/geojson",
            "POST /api/dragon-vein/extract",
            "POST /api/report/generate",
            "POST /api/llm/interpret",
            "POST /api/render/dem",
            "POST /api/render/score-grid",
            "POST /api/render/four-beasts",
            "POST /api/render/combined",
            "GET  /api/layers/all",
            "GET  /api/layers/basemap",
            "GET  /api/layers/water",
            "GET  /api/layers/water-influence",
            "GET  /api/layers/contours",
            "GET  /api/layers/score",
            "GET  /api/layers/four-beasts",
            "GET  /api/layers/candidates",
            "GET  /api/layers/ridges",
        ],
    }


# 静态前端
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    # /static/* 与 /assets/* 均可访问，避免 index 从 / 加载时相对路径 404
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    assets_dir = FRONTEND_DIR / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def index():
        idx = FRONTEND_DIR / "index.html"
        if idx.exists():
            return HTMLResponse(idx.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Xunlong Engine</h1><p>Frontend not found</p>")
