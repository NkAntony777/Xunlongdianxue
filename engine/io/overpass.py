"""OSM Overpass 水系获取器（多端点 + 重试 + 降级）。

公共 Overpass 经常限流 / 403 / 超时，策略：
  1. 多端点轮询
  2. 浏览器式 UA + Accept
  3. 失败退避重试
  4. 简化查询再试一次
  5. 仍失败则返回空图层（由上层决定是否硬失败）

返回: GeoDataFrame, CRS=EPSG:4326.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Tuple

import geopandas as gpd
from shapely.geometry import LineString, Polygon

from engine.io.elevation_api import bbox_from_center

log = logging.getLogger("xunlong.overpass")

# 更像真实客户端，降低部分镜像的 403
USER_AGENT = (
    "Mozilla/5.0 (compatible; XunlongEngine/0.3; "
    "+https://github.com/xunlong; research GIS)"
)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]


def _build_query(bbox: Tuple[float, float, float, float], simple: bool = False) -> str:
    """构造 Overpass QL。simple=True 时只拉 river/canal/water，减轻负载。"""
    minx, miny, maxx, maxy = bbox
    bbox_str = f"{miny},{minx},{maxy},{maxx}"  # south, west, north, east
    if simple:
        return f"""
[out:json][timeout:40];
(
  way["waterway"~"^(river|canal)$"]({bbox_str});
  way["natural"="water"]({bbox_str});
  relation["natural"="water"]({bbox_str});
);
out geom;
""".strip()
    return f"""
[out:json][timeout:40];
(
  way["waterway"~"^(river|stream|canal|drain|riverbank)$"]({bbox_str});
  relation["waterway"~"^(river|stream|canal)$"]({bbox_str});
  way["natural"="water"]({bbox_str});
  relation["natural"="water"]({bbox_str});
);
out geom;
""".strip()


def _http_post_overpass(url: str, query: str, timeout: float) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": "https://overpass-turbo.eu",
            "Referer": "https://overpass-turbo.eu/",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}")
        raw = resp.read().decode("utf-8", errors="replace")
    # 部分镜像返回 HTML 错误页
    if raw.lstrip().startswith("<"):
        raise RuntimeError("HTML error page instead of JSON")
    payload = json.loads(raw)
    if isinstance(payload, dict) and payload.get("remark") and "error" in str(payload.get("remark")).lower():
        raise RuntimeError(payload.get("remark"))
    return payload


def _http_get_overpass(url: str, query: str, timeout: float) -> dict:
    """部分端点对 GET 更友好。"""
    full = url + "?" + urllib.parse.urlencode({"data": query})
    req = urllib.request.Request(
        full,
        method="GET",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Referer": "https://overpass-turbo.eu/",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}")
        raw = resp.read().decode("utf-8", errors="replace")
    if raw.lstrip().startswith("<"):
        raise RuntimeError("HTML error page instead of JSON")
    return json.loads(raw)


def _fetch_overpass(query: str, timeout: float = 45.0) -> dict:
    """多端点 × (POST, GET) × 轻量重试。"""
    errors: list[str] = []
    for url in OVERPASS_ENDPOINTS:
        for method_name, fn in (("POST", _http_post_overpass), ("GET", _http_get_overpass)):
            for attempt in range(2):
                try:
                    t0 = time.time()
                    data = fn(url, query, timeout=timeout)
                    n = len(data.get("elements", [])) if isinstance(data, dict) else 0
                    log.info(
                        "Overpass ok %s %s elements=%s in %.1fs",
                        method_name, url, n, time.time() - t0,
                    )
                    return data
                except urllib.error.HTTPError as e:
                    msg = f"{method_name} {url} HTTP {e.code}"
                    errors.append(msg)
                    log.warning("Overpass fail: %s", msg)
                    # 403/429 换端点，不重试同一端点
                    if e.code in (403, 429, 502, 503, 504):
                        break
                    time.sleep(0.4 * (attempt + 1))
                except Exception as e:
                    msg = f"{method_name} {url} {type(e).__name__}: {e}"
                    errors.append(msg)
                    log.warning("Overpass fail: %s", msg)
                    time.sleep(0.3 * (attempt + 1))
    raise RuntimeError(
        "Overpass failed on all endpoints: " + ("; ".join(errors[-6:]) if errors else "unknown")
    )


def _elements_to_gdf(data: dict) -> gpd.GeoDataFrame:
    """把 Overpass JSON 元素转 GeoDataFrame。"""
    rows = []
    for el in data.get("elements", []):
        if el.get("type") not in ("way", "relation"):
            continue
        geom = el.get("geometry")
        if not geom:
            continue
        coords = [(p["lon"], p["lat"]) for p in geom]
        if len(coords) < 2:
            continue
        if el["type"] == "way":
            if coords[0] == coords[-1] and len(coords) >= 4:
                shp = Polygon(coords)
            else:
                shp = LineString(coords)
        else:
            try:
                if coords[0] == coords[-1] and len(coords) >= 4:
                    shp = Polygon(coords)
                else:
                    shp = LineString(coords)
            except Exception:
                continue
        tags = el.get("tags", {}) or {}
        rows.append({
            "osm_id": el.get("id"),
            "name": tags.get("name", ""),
            "waterway": tags.get("waterway", ""),
            "natural": tags.get("natural", ""),
            "geometry": shp,
        })
    if not rows:
        return gpd.GeoDataFrame(
            columns=["osm_id", "name", "waterway", "natural", "geometry"],
            crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


# ============== 内存 + 磁盘缓存 ==============

_MEM_CACHE: dict[str, Tuple[float, gpd.GeoDataFrame]] = {}
_MEM_CACHE_MAX = 16
_DISK_CACHE_TTL_S = 6 * 3600  # 6h：Overpass 抖动时复用近期成功结果


def _disk_cache_path(key: str):
    """项目 temp 下的水系磁盘缓存路径。"""
    from pathlib import Path
    import hashlib
    import tempfile

    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    base = Path(tempfile.gettempdir()) / "xunlong_water_cache"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return base / f"{h}.geojson"


def _load_disk_cache(key: str) -> Optional[gpd.GeoDataFrame]:
    from pathlib import Path

    p = _disk_cache_path(key)
    try:
        if not p.is_file():
            return None
        age = time.time() - p.stat().st_mtime
        if age > _DISK_CACHE_TTL_S:
            return None
        gdf = gpd.read_file(p)
        if gdf is None or gdf.empty:
            return None
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        log.info("water disk-cache hit age=%.0fs n=%s", age, len(gdf))
        return gdf
    except Exception as e:
        log.warning("water disk-cache load fail: %s", e)
        return None


def _save_disk_cache(key: str, gdf: gpd.GeoDataFrame) -> None:
    if gdf is None or gdf.empty:
        return
    try:
        p = _disk_cache_path(key)
        gdf.to_file(p, driver="GeoJSON")
    except Exception as e:
        log.warning("water disk-cache save fail: %s", e)


def fetch_water(
    bbox: Tuple[float, float, float, float],
    timeout: float = 55.0,
    use_cache: bool = True,
    allow_empty_on_error: bool = False,
) -> gpd.GeoDataFrame:
    """按 bbox 拉水系 GeoDataFrame (EPSG:4326).

    allow_empty_on_error=True 时，全部失败返回空 gdf 而非抛错。
    成功结果写内存 + 磁盘缓存；全失败时优先回退磁盘缓存（减偶发空白层）。
    """
    key = f"water:{round(bbox[0], 4)},{round(bbox[1], 4)},{round(bbox[2], 4)},{round(bbox[3], 4)}"
    if use_cache and key in _MEM_CACHE:
        ts, gdf = _MEM_CACHE[key]
        del _MEM_CACHE[key]
        _MEM_CACHE[key] = (ts, gdf)
        return gdf

    last_err: Exception | None = None
    for simple in (False, True):
        try:
            query = _build_query(bbox, simple=simple)
            data = _fetch_overpass(query, timeout=timeout)
            gdf = _elements_to_gdf(data)
            # 完整查询空结果时再试 simple（部分镜像截断大响应）
            if gdf.empty and not simple:
                log.warning("fetch_water full query empty, try simple")
                continue
            if use_cache:
                _MEM_CACHE[key] = (time.time(), gdf)
                if len(_MEM_CACHE) > _MEM_CACHE_MAX:
                    oldest = min(_MEM_CACHE, key=lambda k: _MEM_CACHE[k][0])
                    _MEM_CACHE.pop(oldest, None)
                if not gdf.empty:
                    _save_disk_cache(key, gdf)
            return gdf
        except Exception as e:
            last_err = e
            log.warning("fetch_water attempt simple=%s failed: %s", simple, e)
            continue

    # Overpass 全失败：磁盘缓存兜底（避免 UI 偶发无水系）
    if use_cache:
        cached = _load_disk_cache(key)
        if cached is not None and not cached.empty:
            log.warning("fetch_water using disk cache after Overpass fail: %s", last_err)
            _MEM_CACHE[key] = (time.time(), cached)
            # 标记属性供上层 warning（GeoDataFrame attrs）
            try:
                cached = cached.copy()
                cached.attrs["from_disk_cache"] = True
                cached.attrs["cache_error"] = str(last_err)[:200] if last_err else ""
            except Exception:
                pass
            return cached

    if allow_empty_on_error:
        log.error("fetch_water giving empty result: %s", last_err)
        return gpd.GeoDataFrame(
            columns=["osm_id", "name", "waterway", "natural", "geometry"],
            crs="EPSG:4326",
        )
    raise RuntimeError(f"Overpass failed on all endpoints: {last_err}")


def fetch_water_for_analysis(
    center_lon: float,
    center_lat: float,
    radius_km: float,
    target_crs: str = "EPSG:3857",
    **kwargs,
) -> gpd.GeoDataFrame:
    """从中心 + 半径拉水系, 重投影到目标 CRS."""
    bbox = bbox_from_center(center_lon, center_lat, radius_km)
    gdf = fetch_water(bbox, **kwargs)
    if gdf.empty or str(gdf.crs).upper() != target_crs.upper():
        try:
            gdf = gdf.to_crs(target_crs) if not gdf.empty else gdf
        except Exception:
            pass
    return gdf


def ping() -> dict:
    """健康检查: 拉阆中小块水系."""
    try:
        gdf = fetch_water(
            bbox=(105.80, 31.50, 105.90, 31.60),
            timeout=25.0,
            use_cache=False,
        )
        return {
            "ok": True,
            "count": int(len(gdf)),
            "columns": list(gdf.columns),
            "sample_names": gdf["name"].head(5).tolist() if "name" in gdf.columns else [],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ping":
        print(json.dumps(ping(), indent=2, ensure_ascii=False))
    else:
        gdf = fetch_water_for_analysis(105.85, 31.55, radius_km=8)
        print(f"rows: {len(gdf)}, crs: {gdf.crs}")
