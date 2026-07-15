"""在线 DEM 获取器。

主数据源：AWS Terrain Tiles (Mapzen Terrarium 编码，SRTM 系，免费无 key)
备用：ESRI WorldElevation3D Terrain3D（已知部分区域绝对高程偏差大，仅 fallback）

Terrarium 解码: elev = R*256 + G + B/256 - 32768 (米)
"""
from __future__ import annotations

import io
import json
import logging
import math
import time
import urllib.parse
import urllib.request
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image

from engine.io.dem import DEM

log = logging.getLogger("xunlong.elevation")

USER_AGENT = "XunlongEngine/0.3 (DEM fetcher; research GIS)"

# AWS open terrain tiles (Terrarium)
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# ESRI fallback（绝对高程在部分区域不可靠，仅应急）
ESRI_EXPORT_URL = (
    "https://elevation3d.arcgis.com/arcgis/rest/services/"
    "WorldElevation3D/Terrain3D/ImageServer/exportImage"
)

_MEM_CACHE: dict[str, Tuple[float, "DEM"]] = {}
_MEM_CACHE_MAX = 16


# ============== 范围计算 ==============

def bbox_from_center(
    center_lon: float, center_lat: float, radius_km: float
) -> Tuple[float, float, float, float]:
    """从中心 + 半径 (km) 算 bbox (EPSG:4326)."""
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(0.0001, math.cos(math.radians(center_lat))))
    return (
        center_lon - dlon, center_lat - dlat,
        center_lon + dlon, center_lat + dlat,
    )


def suggest_resolution(bbox: Tuple[float, float, float, float]) -> float:
    minx, miny, maxx, maxy = bbox
    span_deg = max(maxx - minx, maxy - miny)
    span_m = span_deg * 111_000
    if span_m < 8_000:
        return 30.0
    if span_m < 20_000:
        return 40.0
    if span_m < 40_000:
        return 60.0
    return 100.0


# ============== Terrarium tiles ==============

def _lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[int, int]:
    lat = max(min(lat, 85.0511), -85.0511)
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    x = max(0, min(n - 1, x))
    y = max(0, min(n - 1, y))
    return x, y


def _tile_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """返回 tile 的 (west, south, east, north) EPSG:4326."""
    n = 2.0 ** z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    def y_to_lat(ty):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    north = y_to_lat(y)
    south = y_to_lat(y + 1)
    return west, south, east, north


def _terrarium_decode(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    elev = r * 256.0 + g + b / 256.0 - 32768.0
    # Terrarium 海洋/无效常为 -32768 附近
    elev[elev < -500] = np.nan
    return elev


def _download_tile(z: int, x: int, y: int, timeout: float = 20.0) -> np.ndarray:
    url = TERRARIUM_URL.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return _terrarium_decode(np.asarray(im))


def _zoom_for_resolution(resolution_m: float) -> int:
    """经验：z12≈38m, z13≈19m, z11≈76m (赤道)."""
    if resolution_m <= 25:
        return 13
    if resolution_m <= 45:
        return 12
    if resolution_m <= 90:
        return 11
    return 10


def fetch_dem_terrarium(
    bbox: Tuple[float, float, float, float],
    resolution_m: float = 40.0,
    timeout: float = 25.0,
) -> DEM:
    """从 Terrarium 瓦片拼接 DEM (EPSG:4326)."""
    minx, miny, maxx, maxy = bbox
    z = _zoom_for_resolution(resolution_m)
    n = 2 ** z

    x0, y1 = _lonlat_to_tile(minx, miny, z)  # SW → y larger
    x1, y0 = _lonlat_to_tile(maxx, maxy, z)  # NE → y smaller
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0

    # 限制瓦片数量，过大则降 zoom
    while (x1 - x0 + 1) * (y1 - y0 + 1) > 64 and z > 9:
        z -= 1
        x0, y1 = _lonlat_to_tile(minx, miny, z)
        x1, y0 = _lonlat_to_tile(maxx, maxy, z)
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

    tile_w = tile_h = 256
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    mosaic = np.full((ny * tile_h, nx * tile_w), np.nan, dtype=np.float64)

    for iy, ty in enumerate(range(y0, y1 + 1)):
        for ix, tx in enumerate(range(x0, x1 + 1)):
            try:
                elev = _download_tile(z, tx % n, ty, timeout=timeout)
                mosaic[iy * tile_h:(iy + 1) * tile_h, ix * tile_w:(ix + 1) * tile_w] = elev
            except Exception as e:
                log.warning("tile z=%s x=%s y=%s failed: %s", z, tx, ty, e)

    # mosaic 覆盖的地理范围
    west, _, _, north = _tile_bounds(z, x0, y0)
    _, south, east, _ = _tile_bounds(z, x1, y1)

    # 裁剪到请求 bbox
    full_h, full_w = mosaic.shape
    def lon_to_col(lon):
        return (lon - west) / (east - west) * full_w
    def lat_to_row(lat):
        return (north - lat) / (north - south) * full_h

    c0 = max(0, int(math.floor(lon_to_col(minx))))
    c1 = min(full_w, int(math.ceil(lon_to_col(maxx))))
    r0 = max(0, int(math.floor(lat_to_row(maxy))))
    r1 = min(full_h, int(math.ceil(lat_to_row(miny))))
    if c1 <= c0 + 2 or r1 <= r0 + 2:
        raise RuntimeError("Terrarium crop empty")

    data = mosaic[r0:r1, c0:c1].copy()
    # 对应 bounds
    out_west = west + (c0 / full_w) * (east - west)
    out_east = west + (c1 / full_w) * (east - west)
    out_north = north - (r0 / full_h) * (north - south)
    out_south = north - (r1 / full_h) * (north - south)

    # 可选重采样到目标分辨率
    mid_lat = (out_south + out_north) / 2
    width_m = abs(out_east - out_west) * 111_320 * math.cos(math.radians(mid_lat))
    height_m = abs(out_north - out_south) * 111_320
    tw = max(32, min(1024, int(round(width_m / resolution_m))))
    th = max(32, min(1024, int(round(height_m / resolution_m))))
    if abs(tw - data.shape[1]) > 5 or abs(th - data.shape[0]) > 5:
        from PIL import Image as PILImage
        # 用 nearest 保高程，nan 用邻域均值填充后再缩
        filled = data.copy()
        mask = ~np.isfinite(filled)
        if mask.any() and (~mask).any():
            filled[mask] = np.nanmean(filled)
        img = PILImage.fromarray(filled.astype(np.float32), mode="F")
        img = img.resize((tw, th), resample=PILImage.BILINEAR)
        data = np.array(img, dtype=np.float64)
        # 原 nan 区域大致保留：极低值不处理

    h, w = data.shape
    transform = from_bounds(out_west, out_south, out_east, out_north, w, h)
    return DEM(
        data=data,
        transform=transform,
        crs="EPSG:4326",
        nodata=np.nan,
        bounds=(out_west, out_south, out_east, out_north),
        resolution=(abs(transform.a), abs(transform.e)),
    )


# ============== ESRI fallback ==============

def _esri_export_metadata(
    bbox: Tuple[float, float, float, float],
    size: Tuple[int, int],
    timeout: float = 30.0,
) -> dict:
    minx, miny, maxx, maxy = bbox
    w, h = size
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{w},{h}",
        "format": "tiff",
        "f": "json",
    }
    url = ESRI_EXPORT_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"ESRI exportImage HTTP {resp.status}")
        text = resp.read().decode("utf-8")
    meta = json.loads(text)
    if "href" not in meta:
        raise RuntimeError(f"ESRI exportImage no href: {text[:200]}")
    return meta


def _download_tiff(href: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(href, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _dem_from_bytes(tif_bytes: bytes) -> DEM:
    with rasterio.open(io.BytesIO(tif_bytes)) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data < -1000] = np.nan
        transform = src.transform
        crs = src.crs
        b = src.bounds
        resolution = (abs(src.transform.a), abs(src.transform.e))
    return DEM(
        data=data,
        transform=transform,
        crs=crs,
        nodata=nodata,
        bounds=(b.left, b.bottom, b.right, b.top),
        resolution=resolution,
    )


def fetch_dem_esri(
    bbox: Tuple[float, float, float, float],
    resolution_m: float = 100.0,
    timeout: float = 30.0,
) -> DEM:
    minx, miny, maxx, maxy = bbox
    mid_lat = (miny + maxy) / 2
    width_m = (maxx - minx) * 111_320 * math.cos(math.radians(mid_lat))
    height_m = (maxy - miny) * 111_320
    w = max(10, min(2048, int(round(width_m / resolution_m))))
    h = max(10, min(2048, int(round(height_m / resolution_m))))
    meta = _esri_export_metadata(bbox, (w, h), timeout=timeout)
    tif_bytes = _download_tiff(meta["href"], timeout=timeout)
    return _dem_from_bytes(tif_bytes)


# ============== 主入口 ==============

def fetch_dem(
    bbox: Tuple[float, float, float, float],
    resolution_m: float = 40.0,
    timeout: float = 30.0,
    use_cache: bool = True,
    source: str = "auto",
) -> DEM:
    """按 bbox 拉 DEM (EPSG:4326).

    source: auto | terrarium | esri
    """
    minx, miny, maxx, maxy = bbox
    key = f"{source}:{round(minx,5)},{round(miny,5)},{round(maxx,5)},{round(maxy,5)},{resolution_m}"
    if use_cache and key in _MEM_CACHE:
        ts, dem = _MEM_CACHE[key]
        del _MEM_CACHE[key]
        _MEM_CACHE[key] = (ts, dem)
        return dem

    dem: DEM | None = None
    errors: list[str] = []

    order = []
    if source == "esri":
        order = ["esri"]
    elif source == "terrarium":
        order = ["terrarium"]
    else:
        order = ["terrarium", "esri"]

    for src in order:
        try:
            if src == "terrarium":
                dem = fetch_dem_terrarium(bbox, resolution_m=resolution_m, timeout=timeout)
            else:
                dem = fetch_dem_esri(bbox, resolution_m=resolution_m, timeout=timeout)
            # 基本质量检查：中心高程若落在离谱范围且 relief 极小，换源
            valid = dem.data[np.isfinite(dem.data)]
            if valid.size == 0:
                raise RuntimeError("no valid pixels")
            log.info(
                "DEM source=%s shape=%s elev=%.1f..%.1f",
                src, dem.data.shape, float(valid.min()), float(valid.max()),
            )
            break
        except Exception as e:
            errors.append(f"{src}: {e}")
            log.warning("DEM source %s failed: %s", src, e)
            dem = None

    if dem is None:
        raise RuntimeError("All DEM sources failed: " + "; ".join(errors))

    if use_cache:
        _MEM_CACHE[key] = (time.time(), dem)
        if len(_MEM_CACHE) > _MEM_CACHE_MAX:
            oldest = min(_MEM_CACHE, key=lambda k: _MEM_CACHE[k][0])
            _MEM_CACHE.pop(oldest, None)
    return dem


def fetch_dem_for_analysis(
    center_lon: float,
    center_lat: float,
    radius_km: float,
    resolution_m: Optional[float] = None,
    target_crs: str = "EPSG:3857",
) -> DEM:
    """为分析拉 DEM: Terrarium 主源 + 投影到米制 CRS 便于坡度/四象。"""
    bbox = bbox_from_center(center_lon, center_lat, radius_km)
    if resolution_m is None:
        if radius_km <= 6:
            resolution_m = 30.0
        elif radius_km <= 12:
            resolution_m = 40.0
        elif radius_km <= 20:
            resolution_m = 50.0
        else:
            resolution_m = 80.0
    dem = fetch_dem(bbox, resolution_m=resolution_m)
    if target_crs and str(dem.crs).upper() != target_crs.upper():
        from engine.io.dem import reproject_dem
        try:
            dem = reproject_dem(dem, target_crs)
        except Exception as e:
            log.warning("reproject failed: %s", e)
    return dem


def ping() -> dict:
    """健康检查: 阆中一点附近小块."""
    try:
        dem = fetch_dem(
            bbox=(105.90, 31.55, 106.00, 31.62),
            resolution_m=60.0,
            timeout=25.0,
            use_cache=False,
        )
        return {
            "ok": True,
            "shape": list(dem.shape),
            "bounds": list(dem.bounds),
            "elevation": [float(np.nanmin(dem.data)), float(np.nanmax(dem.data))],
            "source": "terrarium/auto",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
