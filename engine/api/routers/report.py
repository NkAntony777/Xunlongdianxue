"""报告生成路由。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from engine.api.schemas.models import AnalyzeRequest, CandidatesResponse

router = APIRouter()

REPORT_DIR = Path("./reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/generate")
def generate(req: AnalyzeRequest):
    """生成 JSON 报告文件，保存到 ./reports/ 目录并返回路径。"""
    try:
        from engine.core.fengshui_score import find_and_rank_candidates
        from engine.core.terrain_analysis import analyze_terrain
        from engine.io.dem import clip_dem, load_dem, reproject_dem
        from engine.io.rivers import load_water

        dem = load_dem(req.dem_path)
        if req.bbox:
            dem = clip_dem(dem, (req.bbox.minx, req.bbox.miny, req.bbox.maxx, req.bbox.maxy))
        if str(dem.crs).upper().endswith("4326") or (dem.crs and "GEOG" in str(dem.crs).upper()):
            dem = reproject_dem(dem, "EPSG:3857")
        water = load_water(req.water_path) if req.water_path else None
        results = find_and_rank_candidates(dem, water, top_k=req.top_k, min_score=req.min_score)
        terrain = analyze_terrain(dem)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generate failed: {e}")

    from engine.core.fengshui_score import to_json

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
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
    report = to_json(results, metadata=metadata)

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"report_{ts}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "report_path": str(out),
        "metadata": metadata,
        "n_candidates": len(report["candidates"]),
    }


@router.get("/download/{filename}")
def download(filename: str):
    """下载已生成的报告。"""
    p = REPORT_DIR / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(p, media_type="application/json", filename=filename)
