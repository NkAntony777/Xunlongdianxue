"""分图层可视化 API。

为前端提供"可叠加"的图层端点：
  /api/layers/basemap          底图（高程 / 坡度）
  /api/layers/water            水系
  /api/layers/water-influence  水煞影响带
  /api/layers/contours         等高线（PNG + GeoJSON）
  /api/layers/score            风水评分热力（紫→橙）
  /api/layers/buildings        可建城实际覆盖片区（占位）
  /api/layers/ridges           龙脉山脊线 GeoJSON
  /api/layers/four-beasts      四象 GeoJSON
  /api/layers/candidates       候选穴 GeoJSON
  /api/layers/all              一次调用返回所有图层（前端首次加载）
"""
from __future__ import annotations

import base64
import json
import math
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

from engine.api.schemas.models import AnalyzeRequest, BBox
from engine.core.fengshui_score import find_and_rank_candidates
from engine.core.fengshui_score import _sanitize as _sanitize_floats
from engine.core.four_beasts_detect import (
    detect_four_beasts,
    compute_score_grid,
    find_score_peak,
    score_peak_xy,
)
from engine.core.dragon_vein import analyze_dragon_vein
from engine.core.render import (
    render_basemap,
    render_slope_basemap,
    render_contours,
    render_water,
    render_water_influence,
    render_buildings,
    render_buildable_from_dem,
    render_score_grid,
    four_beasts_geojson,
    candidates_geojson,
    ridges_geojson,
)
from engine.io.dem import clip_dem, load_dem, reproject_dem
from engine.io.rivers import load_water

router = APIRouter()


def _load_dem_with_bbox(req: AnalyzeRequest, reproject_metric: bool = False):
    """加载并按 bbox 裁剪。

    默认保留原 CRS（地理坐标下图幅接近正方形，与参考图一致）。
    仅当 reproject_metric=True 时才转到 EPSG:3857。
    """
    dem = load_dem(req.dem_path)
    if req.bbox:
        dem = clip_dem(dem, (req.bbox.minx, req.bbox.miny, req.bbox.maxx, req.bbox.maxy))
    if reproject_metric and (
        str(dem.crs).upper().endswith("4326")
        or (dem.crs and "GEOG" in str(dem.crs).upper())
    ):
        try:
            dem = reproject_dem(dem, "EPSG:3857")
        except Exception:
            pass
    return dem


def _pick_representative_candidate(dem, water, top_k: int):
    try:
        results = find_and_rank_candidates(dem, water, top_k=max(10, top_k), min_score=0)
    except Exception:
        results = []
    if not results:
        return None
    return results[0]


def _load_water_for_dem(req: AnalyzeRequest, dem):
    """加载水系并重投影到 DEM CRS，返回 (WaterNetwork, GeoDataFrame 用于渲染)。"""
    from engine.io.rivers import WaterNetwork, _clean_gdf
    if not req.water_path:
        return None, None
    try:
        water_net = load_water(req.water_path)
    except Exception:
        return None, None
    gdf = water_net.gdf if water_net is not None else None
    if gdf is not None and not gdf.empty and dem.crs is not None:
        try:
            if str(gdf.crs).upper() != str(dem.crs).upper():
                gdf = gdf.to_crs(dem.crs)
            gdf = _clean_gdf(gdf)
        except Exception:
            try:
                import pyproj
                from shapely.ops import transform as shp_transform
                t = pyproj.Transformer.from_crs(
                    gdf.crs, dem.crs, always_xy=True
                ).transform
                gdf = gdf.copy()
                gdf["geometry"] = gdf["geometry"].apply(
                    lambda g: shp_transform(t, g) if g is not None else g
                )
                gdf = gdf.set_crs(dem.crs)
                gdf = _clean_gdf(gdf)
            except Exception:
                gdf = None
    if gdf is None or gdf.empty:
        return None, gdf
    try:
        water = WaterNetwork(gdf=gdf)
    except Exception:
        return None, gdf
    return water, gdf


def _resolve_acupoint_center(
    dem,
    water,
    top_k: int,
    *,
    score_grid=None,
    center_row: int | None = None,
    center_col: int | None = None,
) -> tuple[int, int, str]:
    """解析穴心 (row,col) 与来源。

    优先级（对齐参考图）：
      1. 显式 center_row/col
      2. 评分场平滑最高点（场评最高点 = 穴）
      3. 候选穴第一名
      4. DEM 几何中心
    """
    if center_row is not None and center_col is not None:
        r = int(np.clip(center_row, 0, dem.height - 1))
        c = int(np.clip(center_col, 0, dem.width - 1))
        return r, c, "explicit"

    if score_grid is not None:
        peak = find_score_peak(score_grid)
        if peak is not None:
            pr, pc, _ = peak
            return int(pr), int(pc), "score_field_peak"

    ref = _pick_representative_candidate(dem, water, top_k)
    if ref is not None:
        from rasterio.transform import rowcol
        try:
            crow, ccol = rowcol(dem.transform, ref.x, ref.y)
            crow, ccol = int(crow), int(ccol)
            if 0 <= crow < dem.height and 0 <= ccol < dem.width:
                return crow, ccol, "top_candidate"
        except Exception:
            pass

    return dem.height // 2, dem.width // 2, "dem_center"


def _all_structured_layers(
    req: AnalyzeRequest,
    dem,
    water,
    *,
    score_grid=None,
    center_row: int | None = None,
    center_col: int | None = None,
) -> dict[str, Any]:
    """结构化图层：委托 pipeline.analyze_aoi（先龙后穴后四象）。

    默认穴心 = 场评热峰；显式 center_* 仍可覆盖。
    """
    from engine.pipeline.analyze_aoi import analyze_aoi, structured_from_aoi

    aoi = analyze_aoi(
        dem,
        water,
        top_k=req.top_k,
        min_score=0,
        score_grid=score_grid,
        center_row=center_row,
        center_col=center_col,
        require_beasts=True,
    )
    return structured_from_aoi(aoi)


# ============ 端点 ============

@router.get("/basemap")
def layer_basemap(
    dem_path: str = Query(...),
    water_path: Optional[str] = Query(None),
    mode: str = Query("elevation", description="elevation | slope"),
    bbox_minx: Optional[float] = Query(None),
    bbox_miny: Optional[float] = Query(None),
    bbox_maxx: Optional[float] = Query(None),
    bbox_maxy: Optional[float] = Query(None),
    top_k: int = Query(10, ge=1, le=100),
):
    """底图：高程晕渲 或 坡度底图。"""
    req = AnalyzeRequest(
        dem_path=dem_path,
        water_path=water_path,
        top_k=top_k,
        min_score=0,
        bbox=BBox(
            minx=bbox_minx, miny=bbox_miny,
            maxx=bbox_maxx, maxy=bbox_maxy,
        ) if all(v is not None for v in [bbox_minx, bbox_miny, bbox_maxx, bbox_maxy]) else None,
    )
    try:
        dem = _load_dem_with_bbox(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"basemap failed: {e}")

    if mode == "slope":
        result = render_slope_basemap(dem)
    else:
        result = render_basemap(dem)
    return {
        "width": result.width,
        "height": result.height,
        "bbox": list(result.bbox),
        "png_base64": result.png_base64,
        "legend": result.legend,
        "mode": mode,
    }


@router.get("/water")
def layer_water(
    dem_path: str = Query(...),
    water_path: Optional[str] = Query(None),
    bbox_minx: Optional[float] = Query(None),
    bbox_miny: Optional[float] = Query(None),
    bbox_maxx: Optional[float] = Query(None),
    bbox_maxy: Optional[float] = Query(None),
    top_k: int = Query(10, ge=1, le=100),
):
    """水系图层。"""
    req = AnalyzeRequest(
        dem_path=dem_path,
        water_path=water_path,
        top_k=top_k,
        min_score=0,
        bbox=BBox(
            minx=bbox_minx, miny=bbox_miny,
            maxx=bbox_maxx, maxy=bbox_maxy,
        ) if all(v is not None for v in [bbox_minx, bbox_miny, bbox_maxx, bbox_maxy]) else None,
    )
    try:
        dem = _load_dem_with_bbox(req)
        water_gdf = load_water(req.water_path).gdf if req.water_path else None
        # 重投影到 DEM CRS
        if water_gdf is not None and not water_gdf.empty and dem.crs is not None:
            try:
                import pyproj
                from shapely.ops import transform as shp_transform
                t = pyproj.Transformer.from_crs(
                    water_gdf.crs, dem.crs, always_xy=True
                ).transform
                water_gdf = shp_transform(t, water_gdf)
            except Exception:
                pass
        result = render_water(water_gdf, dem.bounds)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"water failed: {e}")
    return {
        "bbox": list(result.bbox),
        "png_base64": result.png_base64,
        "legend": result.legend,
    }


@router.get("/water-influence")
def layer_water_influence(
    dem_path: str = Query(...),
    water_path: Optional[str] = Query(None),
    bbox_minx: Optional[float] = Query(None),
    bbox_miny: Optional[float] = Query(None),
    bbox_maxx: Optional[float] = Query(None),
    bbox_maxy: Optional[float] = Query(None),
    buffer_m: float = Query(1500),
    top_k: int = Query(10, ge=1, le=100),
):
    """水煞影响带图层。"""
    req = AnalyzeRequest(
        dem_path=dem_path,
        water_path=water_path,
        top_k=top_k,
        min_score=0,
        bbox=BBox(
            minx=bbox_minx, miny=bbox_miny,
            maxx=bbox_maxx, maxy=bbox_maxy,
        ) if all(v is not None for v in [bbox_minx, bbox_miny, bbox_maxx, bbox_maxy]) else None,
    )
    try:
        dem = _load_dem_with_bbox(req)
        water_gdf = load_water(req.water_path).gdf if req.water_path else None
        if water_gdf is not None and not water_gdf.empty and dem.crs is not None:
            try:
                import pyproj
                from shapely.ops import transform as shp_transform
                t = pyproj.Transformer.from_crs(
                    water_gdf.crs, dem.crs, always_xy=True
                ).transform
                water_gdf = shp_transform(t, water_gdf)
            except Exception:
                pass
        result = render_water_influence(water_gdf, dem.bounds, buffer_m=buffer_m)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"water-influence failed: {e}")
    return {
        "bbox": list(result.bbox),
        "png_base64": result.png_base64,
        "legend": result.legend,
    }


@router.get("/contours")
def layer_contours(
    dem_path: str = Query(...),
    water_path: Optional[str] = Query(None),
    bbox_minx: Optional[float] = Query(None),
    bbox_miny: Optional[float] = Query(None),
    bbox_maxx: Optional[float] = Query(None),
    bbox_maxy: Optional[float] = Query(None),
    interval: float = Query(30.0, gt=0),
    top_k: int = Query(10, ge=1, le=100),
):
    """等高线图层。"""
    req = AnalyzeRequest(
        dem_path=dem_path,
        water_path=water_path,
        top_k=top_k,
        min_score=0,
        bbox=BBox(
            minx=bbox_minx, miny=bbox_miny,
            maxx=bbox_maxx, maxy=bbox_maxy,
        ) if all(v is not None for v in [bbox_minx, bbox_miny, bbox_maxx, bbox_maxy]) else None,
    )
    try:
        dem = _load_dem_with_bbox(req)
        result = render_contours(dem, contour_interval=interval)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"contours failed: {e}")
    return {
        "width": result.width,
        "height": result.height,
        "bbox": list(result.bbox),
        "png_base64": result.png_base64,
        "geojson": result.geojson,
        "legend": result.legend,
    }


@router.get("/score")
def layer_score(
    dem_path: str = Query(...),
    water_path: Optional[str] = Query(None),
    bbox_minx: Optional[float] = Query(None),
    bbox_miny: Optional[float] = Query(None),
    bbox_maxx: Optional[float] = Query(None),
    bbox_maxy: Optional[float] = Query(None),
    sample_step: int = Query(4, ge=1, le=10),
    top_k: int = Query(10, ge=1, le=100),
):
    """风水评分热力（紫→橙）。"""
    req = AnalyzeRequest(
        dem_path=dem_path,
        water_path=water_path,
        top_k=top_k,
        min_score=0,
        bbox=BBox(
            minx=bbox_minx, miny=bbox_miny,
            maxx=bbox_maxx, maxy=bbox_maxy,
        ) if all(v is not None for v in [bbox_minx, bbox_miny, bbox_maxx, bbox_maxy]) else None,
    )
    try:
        dem = _load_dem_with_bbox(req)
        water, _wg = _load_water_for_dem(req, dem)
        grid = compute_score_grid(dem, sample_step=sample_step, water=water)
        result = render_score_grid(dem, grid)
        stats = {
            "min": float(np.nanmin(grid)) if np.isfinite(grid).any() else None,
            "max": float(np.nanmax(grid)) if np.isfinite(grid).any() else None,
            "mean": float(np.nanmean(grid)) if np.isfinite(grid).any() else None,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"score failed: {e}")
    return {
        "bbox": list(result.bbox),
        "png_base64": result.png_base64,
        "legend": result.legend,
        "stats": stats,
    }


def _serialize_four_beasts(fb) -> dict[str, Any]:
    """FourBeastsPositions → API 字典。"""
    fb_dict: dict[str, dict[str, float]] = {}
    for k in ("shaozu", "xuanwu", "zhuque", "qinglong", "baihu"):
        v = getattr(fb, k, None)
        if v:
            fb_dict[k] = {"x": float(v[0]), "y": float(v[1])}
            bm = (fb.meta or {}).get("beasts", {}).get(k)
            if isinstance(bm, dict):
                for mk in ("elev_m", "dist_m", "bearing_deg", "row", "col", "on_ridge", "score"):
                    if mk in bm and bm[mk] is not None:
                        fb_dict[k][mk] = bm[mk]
    center = {"x": float(fb.center[0]), "y": float(fb.center[1])} if fb.center else None
    facing = float(fb.facing)
    return {
        "center": center,
        "facing": facing,
        "sit": float(getattr(fb, "sit", (facing + 180) % 360)),
        "facing_method": getattr(fb, "facing_method", ""),
        "four_beasts": fb_dict,
        "meta": getattr(fb, "meta", {}) or {},
        "geojson": four_beasts_geojson(fb_dict),
    }


@router.get("/four-beasts")
def layer_four_beasts(
    dem_path: str = Query(...),
    water_path: Optional[str] = Query(None),
    bbox_minx: Optional[float] = Query(None),
    bbox_miny: Optional[float] = Query(None),
    bbox_maxx: Optional[float] = Query(None),
    bbox_maxy: Optional[float] = Query(None),
    top_k: int = Query(10, ge=1, le=100),
    center_x: Optional[float] = Query(None, description="穴心世界坐标 x（与 DEM CRS 一致）"),
    center_y: Optional[float] = Query(None, description="穴心世界坐标 y"),
    center_row: Optional[int] = Query(None, description="穴心栅格行"),
    center_col: Optional[int] = Query(None, description="穴心栅格列"),
):
    """四象识别。可指定穴心（候选点点击）；未指定时用场评峰值。"""
    req = AnalyzeRequest(
        dem_path=dem_path,
        water_path=water_path,
        top_k=top_k,
        min_score=0,
        bbox=BBox(
            minx=bbox_minx, miny=bbox_miny,
            maxx=bbox_maxx, maxy=bbox_maxy,
        ) if all(v is not None for v in [bbox_minx, bbox_miny, bbox_maxx, bbox_maxy]) else None,
    )
    try:
        from engine.core.dragon_vein import analyze_dragon_vein, select_primary_dragon

        dem = _load_dem_with_bbox(req)
        water, _water_gdf = _load_water_for_dem(req, dem)

        # 主来龙（与 layer_all 同理：坐靠来龙，不绑绝对方位）
        dv = None
        primary = None
        try:
            dv = analyze_dragon_vein(dem, min_length_m=120.0, water=water)
            primary = select_primary_dragon(dem, water=water, dragon_vein=dv)
        except Exception:
            dv, primary = None, None

        center_source = "score_field_peak"
        if center_row is not None and center_col is not None:
            crow = int(np.clip(center_row, 0, dem.height - 1))
            ccol = int(np.clip(center_col, 0, dem.width - 1))
            center_source = "explicit_rowcol"
        elif center_x is not None and center_y is not None:
            from rasterio.transform import rowcol
            crow, ccol = rowcol(dem.transform, float(center_x), float(center_y))
            crow = int(np.clip(int(crow), 0, dem.height - 1))
            ccol = int(np.clip(int(ccol), 0, dem.width - 1))
            center_source = "explicit_xy"
        else:
            try:
                grid = compute_score_grid(dem, sample_step=4, water=water)
            except Exception:
                grid = None
            crow, ccol, center_source = _resolve_acupoint_center(
                dem, water, req.top_k, score_grid=grid,
            )

        fb = detect_four_beasts(
            dem, center_row=crow, center_col=ccol, water=water,
            dragon_vein=dv, primary_dragon=primary,
        )
        out = _serialize_four_beasts(fb)
        out["center_source"] = center_source
        out["center_row"] = int(crow)
        out["center_col"] = int(ccol)
        if primary is not None:
            out["primary_dragon"] = {
                "flow_azimuth_deg": round(primary.flow_azimuth_deg, 1),
                "sit_deg": round(primary.sit_deg, 1),
                "ridge_idx": primary.ridge_idx,
                "method": primary.method,
            }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"four-beasts failed: {e}")
    return _sanitize_floats(out)


@router.get("/candidates")
def layer_candidates(
    dem_path: str = Query(...),
    water_path: Optional[str] = Query(None),
    bbox_minx: Optional[float] = Query(None),
    bbox_miny: Optional[float] = Query(None),
    bbox_maxx: Optional[float] = Query(None),
    bbox_maxy: Optional[float] = Query(None),
    top_k: int = Query(10, ge=1, le=100),
    min_score: int = Query(0, ge=0, le=100),
):
    """候选穴 GeoJSON。"""
    req = AnalyzeRequest(
        dem_path=dem_path,
        water_path=water_path,
        top_k=top_k,
        min_score=min_score,
        bbox=BBox(
            minx=bbox_minx, miny=bbox_miny,
            maxx=bbox_maxx, maxy=bbox_maxy,
        ) if all(v is not None for v in [bbox_minx, bbox_miny, bbox_maxx, bbox_maxy]) else None,
    )
    try:
        dem = _load_dem_with_bbox(req)
        water, _water_gdf = _load_water_for_dem(req, dem)
        results = find_and_rank_candidates(dem, water, top_k=top_k, min_score=min_score)
        cands = [
            {
                "id": r.candidate_id,
                "rank": r.rank,
                "x": r.x,
                "y": r.y,
                "overall_score": r.overall,
                "form_type": r.form_type,
                "scores": r.scores,
                "geography": r.geography,
            }
            for r in results
        ]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"candidates failed: {e}")
    return _sanitize_floats({
        "candidates": cands,
        "geojson": candidates_geojson(cands),
    })


@router.get("/ridges")
def layer_ridges(
    dem_path: str = Query(...),
    water_path: Optional[str] = Query(None),
    bbox_minx: Optional[float] = Query(None),
    bbox_miny: Optional[float] = Query(None),
    bbox_maxx: Optional[float] = Query(None),
    bbox_maxy: Optional[float] = Query(None),
    top_k: int = Query(10, ge=1, le=100),
):
    """龙脉 GeoJSON。"""
    req = AnalyzeRequest(
        dem_path=dem_path,
        water_path=water_path,
        top_k=top_k,
        min_score=0,
        bbox=BBox(
            minx=bbox_minx, miny=bbox_miny,
            maxx=bbox_maxx, maxy=bbox_maxy,
        ) if all(v is not None for v in [bbox_minx, bbox_miny, bbox_maxx, bbox_maxy]) else None,
    )
    try:
        dem = _load_dem_with_bbox(req)
        dv = analyze_dragon_vein(dem)
        ridges = [
            {
                "coords": r.coords.tolist(),
                "rank": i + 1,
            }
            for i, r in enumerate(dv.major_ridges[:5])
        ]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ridges failed: {e}")
    return {
        "ridges": ridges,
        "geojson": ridges_geojson(ridges),
    }


@router.get("/all")
def layer_all(
    dem_path: str = Query(...),
    water_path: Optional[str] = Query(None),
    bbox_minx: Optional[float] = Query(None),
    bbox_miny: Optional[float] = Query(None),
    bbox_maxx: Optional[float] = Query(None),
    bbox_maxy: Optional[float] = Query(None),
    mode: str = Query("elevation"),
    contour_interval: float = Query(30.0),
    sample_step: int = Query(4),
    influence_buffer_m: float = Query(1500.0),
    top_k: int = Query(10),
    min_score: int = Query(0),
):
    """一次调用返回所有图层（PNG base64 + GeoJSON），前端首次加载时使用。"""
    req = AnalyzeRequest(
        dem_path=dem_path,
        water_path=water_path,
        top_k=top_k,
        min_score=min_score,
        bbox=BBox(
            minx=bbox_minx, miny=bbox_miny,
            maxx=bbox_maxx, maxy=bbox_maxy,
        ) if all(v is not None for v in [bbox_minx, bbox_miny, bbox_maxx, bbox_maxy]) else None,
    )

    try:
        dem = _load_dem_with_bbox(req)
        water, water_gdf = _load_water_for_dem(req, dem)

        # 1. 底图
        if mode == "slope":
            basemap = render_slope_basemap(dem)
        else:
            basemap = render_basemap(dem)

        # 2. 水煞影响带
        wi = render_water_influence(water_gdf, dem.bounds, buffer_m=influence_buffer_m)

        # 3. 水系
        w = render_water(water_gdf, dem.bounds)

        # 4. 评分热力 → 平滑峰值即穴心（参考图：场域最高权重 = 穴）
        grid = compute_score_grid(dem, sample_step=sample_step, water=water)
        peak_info = score_peak_xy(dem, grid)
        if peak_info is not None:
            (peak_row, peak_col), peak_xy, peak_score = peak_info
        else:
            peak_row = peak_col = None
            peak_xy = None
            peak_score = None
        sg = render_score_grid(dem, grid)

        # 5. 等高线
        cs = render_contours(dem, contour_interval=contour_interval)

        # 6. 可建城片区（DEM 缓坡规则）
        buildings = render_buildable_from_dem(dem, water_gdf=water_gdf)

        # 7. 结构化：四象 / 主轴 以评分峰值穴为中心
        structured = _all_structured_layers(
            req, dem, water,
            score_grid=grid,
            center_row=peak_row,
            center_col=peak_col,
        )

        # 场评最高点 ≡ 四象穴心（强制同一点）
        if structured.get("center"):
            peak_xy = [
                float(structured["center"]["x"]),
                float(structured["center"]["y"]),
            ]
            if sg.legend is not None:
                sg.legend["peak_xy"] = peak_xy
                if peak_score is not None:
                    sg.legend["peak_score"] = peak_score

        # DEM 真实宽高（米）
        from engine.core.terrain_analysis import _is_geographic
        bw = float(dem.bounds[2] - dem.bounds[0])
        bh = float(dem.bounds[3] - dem.bounds[1])
        if _is_geographic(dem.crs):
            # 中纬度近似
            mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
            width_m = bw * 111_000.0 * max(0.2, abs(np.cos(np.radians(mid_lat))))
            height_m = bh * 111_000.0
            res_m = float(min(dem.resolution)) * 111_000.0
        else:
            width_m, height_m = bw, bh
            res_m = float(min(abs(dem.resolution[0]), abs(dem.resolution[1])))

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"all layers failed: {e}")

    # 必须整包 sanitize：legend / structured / peak 常含 numpy 标量，
    # 否则 FastAPI jsonable_encoder 报 numpy.int32 not iterable → 图层渲染失败
    return _sanitize_floats({
        "bbox": [float(x) for x in dem.bounds],
        "dem": {
            "width": int(dem.width),
            "height": int(dem.height),
            "vmin": float(np.nanmin(dem.data)),
            "vmax": float(np.nanmax(dem.data)),
            "resolution_m": float(res_m),
            "width_m": float(width_m),
            "height_m": float(height_m),
            "crs": str(dem.crs) if dem.crs else None,
        },
        "basemap": {
            "png_base64": basemap.png_base64,
            "legend": basemap.legend,
            "mode": mode,
        },
        "water": {
            "png_base64": w.png_base64,
            "legend": w.legend,
        },
        "water_influence": {
            "png_base64": wi.png_base64,
            "legend": wi.legend,
        },
        "buildings": {
            "png_base64": buildings.png_base64,
            "legend": buildings.legend,
        },
        "score": {
            "png_base64": sg.png_base64,
            "legend": sg.legend,
            "peak_xy": peak_xy,
        },
        "contours": {
            "png_base64": cs.png_base64,
            "geojson": cs.geojson,
            "legend": cs.legend,
        },
        "structured": structured,
    })
