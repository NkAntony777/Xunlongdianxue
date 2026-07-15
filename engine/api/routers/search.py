"""地点搜索路由: POST /api/location/search"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

search_router = APIRouter()

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, description="搜索关键字, 支持中英文")
    limit: int = Field(5, ge=1, le=20)
    countrycodes: Optional[str] = Field("cn", description="国家代码限制 (e.g. 'cn')")


@search_router.post("/search", summary="地点搜索 (Nominatim)")
def location_search(req: SearchRequest):
    """Nominatim 地点搜索 → 返回 [{name, lat, lon, bbox, type, importance}]."""
    params = {
        "q": req.query,
        "format": "json",
        "limit": str(req.limit),
        "addressdetails": "0",
    }
    if req.countrycodes:
        params["countrycodes"] = req.countrycodes
    url = NOMINATIM_URL + "?" + urllib.parse.urlencode(params)
    headers = {
        "User-Agent": "XunlongEngine/0.2 (FengShui research; +https://github.com/xunlong)",
        "Accept-Language": "zh-CN,zh,en",
    }
    try:
        req_http = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req_http, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Nominatim 失败: {e}")

    results = []
    for r in raw:
        bb = r.get("boundingbox", [])
        bbox = None
        if len(bb) == 4:
            try:
                bbox = [float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1])]
            except Exception:
                bbox = None
        try:
            results.append({
                "name": r.get("display_name", ""),
                "short_name": r.get("name", req.query),
                "lat": float(r.get("lat", 0)),
                "lon": float(r.get("lon", 0)),
                "type": r.get("type", ""),
                "class": r.get("class", ""),
                "importance": r.get("importance", 0),
                "bbox": bbox,
            })
        except Exception:
            continue
    return {"query": req.query, "count": len(results), "results": results}
