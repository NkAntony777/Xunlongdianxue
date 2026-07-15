"""Serialize fused scores / sanitize JSON."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from engine.core.scoring.candidate import FusedScore

def to_geojson(results: list[FusedScore]) -> dict[str, Any]:
    """输出候选穴为 GeoJSON FeatureCollection。"""
    features = []
    for r in results:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r.x, r.y]},
            "properties": {
                "id": r.candidate_id,
                "rank": r.rank,
                "overall_score": r.overall,
                "elevation_m": r.elevation,
                "form_type": r.form_type,
                "scores": r.scores,
                "geography": r.geography,
                "messages": r.messages,
            },
        })
    return _sanitize({"type": "FeatureCollection", "features": features})


def to_json(results: list[FusedScore], metadata: dict | None = None) -> dict[str, Any]:
    """输出 JSON 报告。"""
    out = {
        "metadata": metadata or {},
        "candidates": [
            {
                "id": r.candidate_id,
                "rank": r.rank,
                "x": r.x,
                "y": r.y,
                "elevation_m": r.elevation,
                "form_type": r.form_type,
                "overall_score": r.overall,
                "scores": r.scores,
                "geography": r.geography,
                "messages": r.messages,
                "meta": r.meta or {},
            }
            for r in results
        ],
    }
    return _sanitize(out)


def _sanitize(obj):
    """递归清洗 JSON：inf/NaN→None，numpy 标量/数组→原生类型，dict 键也转原生。

    修复：FastAPI jsonable_encoder 遇 numpy.int32 键/值会 500
    （'numpy.int32' object is not iterable）。
    """
    import math

    def _key(k):
        if isinstance(k, (np.integer,)):
            return int(k)
        if isinstance(k, (np.floating,)):
            v = float(k)
            if math.isnan(v) or math.isinf(v):
                return str(k)
            # JSON 对象键最终须为 str；先转 Python 数再由 encoder 处理
            return int(v) if v == int(v) else v
        if isinstance(k, (bytes, bytearray)):
            return k.decode("utf-8", errors="replace")
        if isinstance(k, np.bool_):
            return bool(k)
        return k

    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {_key(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int,)):
        # 排除 bool（bool 是 int 子类，上面已处理）
        return int(obj)
    # set / 其它可迭代但非 str
    if isinstance(obj, set):
        return [_sanitize(v) for v in obj]
    return obj
