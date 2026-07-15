"""分析区约束: /api/aoi/*"""
from __future__ import annotations

from fastapi import APIRouter

from engine.core.aoi_limits import (
    validate_radius_km, radius_quality, aoi_limits_payload,
)

aoi_router = APIRouter()


@aoi_router.get("/limits", summary="分析区半径约束")
def aoi_limits():
    """返回前端圈选所需的最小/最大/推荐半径与说明。"""
    return aoi_limits_payload()


@aoi_router.post("/validate", summary="校验圈选半径")
def aoi_validate(body: dict):
    r = body.get("radius_km")
    try:
        rr = validate_radius_km(r)
    except ValueError as e:
        return {"ok": False, "radius_km": r, "quality": "invalid", "detail": str(e)}
    q = radius_quality(rr)
    lim = aoi_limits_payload()
    return {
        "ok": True,
        "radius_km": rr,
        "quality": q,
        "detail": {
            "ok": "尺度合适",
            "small": f"可用但偏小，建议 ≥ {lim['recommended_min_km']} km",
            "large": f"可用但偏大，建议 ≤ {lim['recommended_max_km']} km",
        }.get(q, ""),
    }
