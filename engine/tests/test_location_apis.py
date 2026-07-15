"""测试 elevation_api (ESRI DEM) 和 overpass (OSM 水系) + API 路由.

这些测试需要联网, 但都有超时保护. 离线/防火墙环境 skip.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest
import numpy as np
from fastapi.testclient import TestClient

from engine.io.elevation_api import (
    fetch_dem, fetch_dem_for_analysis, bbox_from_center, ping,
)
from engine.io.overpass import fetch_water, fetch_water_for_analysis, ping as ping_water
from engine.api.main import app


# 网络依赖: ESRI + Overpass. 网络不可用时 skip
def _esri_reachable() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen(
            "https://elevation3d.arcgis.com/arcgis/rest/services/"
            "WorldElevation3D/Terrain3D/ImageServer?f=json",
            timeout=5,
        )
        return True
    except Exception:
        return False


def _overpass_reachable() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("https://overpass-api.de/api/status", timeout=5)
        return True
    except Exception:
        return False


requires_esri = pytest.mark.skipif(not _esri_reachable(), reason="ESRI 不可达")
requires_overpass = pytest.mark.skipif(not _overpass_reachable(), reason="Overpass 不可达")


# ============ bbox_from_center ============

class TestBboxFromCenter:
    def test_10km_around_eq(self):
        # 赤道 10km ≈ 0.09°
        bb = bbox_from_center(0.0, 0.0, 10.0)
        assert abs(bb[0] + 0.09) < 0.01
        assert abs(bb[1] + 0.09) < 0.01
        assert abs(bb[2] - 0.09) < 0.01
        assert abs(bb[3] - 0.09) < 0.01

    def test_at_30N_lon_smaller(self):
        # 在 30° 纬度, 1° 经度 ≈ 96 km, 1° 纬度 ≈ 111 km
        # 10km 半径 → 跨度 20km
        bb = bbox_from_center(116.0, 30.0, 10.0)
        span_lon = bb[2] - bb[0]
        span_lat = bb[3] - bb[1]
        # 跨度 = 2 * radius_km / (111.32) ≈ 0.18
        assert abs(span_lat - 0.18) < 0.01
        # 经度跨度 0.18 / cos(30°) ≈ 0.208
        assert abs(span_lon - 0.21) < 0.02
        # 验证: span_lon > span_lat (因为 cos(30°) < 1)
        assert span_lon > span_lat


# ============ ESRI elevation API ============

@requires_esri
class TestElevationAPI:
    def test_ping(self):
        result = ping()
        assert result["ok"] is True
        assert "shape" in result
        assert result["shape"][0] > 0
        assert result["shape"][1] > 0

    def test_fetch_dem_small(self):
        dem = fetch_dem(
            bbox=(105.80, 31.50, 105.90, 31.60),
            resolution_m=100.0,
            timeout=30.0,
            use_cache=False,
        )
        assert dem.shape[0] > 0
        assert dem.shape[1] > 0
        # 高程应该有有效值
        valid = dem.data[np.isfinite(dem.data)]
        assert valid.size > 0
        assert 100 < valid.mean() < 8000  # 合理高程范围

    def test_fetch_dem_for_analysis_reproject(self):
        dem = fetch_dem_for_analysis(105.85, 31.55, radius_km=5)
        # 应该是 EPSG:3857
        assert "3857" in str(dem.crs).upper()
        assert dem.shape[0] > 10
        assert dem.shape[1] > 10

    def test_resolution_affects_size(self):
        dem_lo = fetch_dem(bbox=(105.80, 31.50, 105.90, 31.60), resolution_m=200, timeout=30, use_cache=False)
        dem_hi = fetch_dem(bbox=(105.80, 31.50, 105.90, 31.60), resolution_m=50, timeout=30, use_cache=False)
        # 高分辨率应该更多像素
        assert dem_hi.shape[0] * dem_hi.shape[1] > dem_lo.shape[0] * dem_lo.shape[1]


# ============ OSM Overpass ============

@requires_overpass
class TestOverpassAPI:
    def test_ping(self):
        result = ping_water()
        assert result["ok"] is True
        assert result["count"] > 0

    def test_fetch_water(self):
        gdf = fetch_water(
            bbox=(105.80, 31.50, 105.90, 31.60),
            timeout=30.0,
            use_cache=False,
        )
        assert len(gdf) > 0
        assert "name" in gdf.columns
        assert "waterway" in gdf.columns

    def test_water_crs_3857(self):
        gdf = fetch_water_for_analysis(105.85, 31.55, radius_km=5)
        if not gdf.empty:
            assert "3857" in str(gdf.crs).upper()


# ============ API 路由 ============

@requires_esri
class TestLocationAPI:
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)

    def test_search_chinese(self, client):
        r = client.post(
            "/api/location/search",
            json={"query": "Beijing", "limit": 3},
        )
        assert r.status_code == 200, r.text[:200]
        d = r.json()
        assert "results" in d
        assert d["count"] > 0
        first = d["results"][0]
        assert "lat" in first
        assert "lon" in first

    def test_elevation_fetch(self, client):
        r = client.post(
            "/api/elevation/fetch",
            json={"lon": 105.85, "lat": 31.55, "radius_km": 5},
        )
        assert r.status_code == 200, r.text[:200]
        d = r.json()
        assert d["shape"][0] > 10
        assert d["crs"] == "EPSG:3857"
        assert "geotiff_base64" in d
        # 验证 base64 可解码
        import base64
        decoded = base64.b64decode(d["geotiff_base64"])
        assert len(decoded) > 100
        # TIFF magic: 49 49 2A 00 (little endian) or 4D 4D 00 2A
        assert decoded[:4] in (b"II*\x00", b"MM\x00*")

    def test_water_fetch(self, client):
        r = client.post(
            "/api/water/fetch",
            json={"lon": 105.85, "lat": 31.55, "radius_km": 5},
        )
        assert r.status_code == 200, r.text[:200]
        d = r.json()
        assert "count" in d
        assert d["count"] >= 0
        assert "features" in d

    def test_cache_info(self, client):
        r = client.get("/api/cache/info")
        assert r.status_code == 200
        d = r.json()
        assert "dem" in d
        assert "water" in d
        assert "services" in d

    def test_elevation_validation(self, client):
        # 半径超业务上限 → 400（aoi_limits 校验）
        r = client.post(
            "/api/elevation/fetch",
            json={"lon": 0, "lat": 0, "radius_km": 999},
        )
        assert r.status_code == 400
        # 半径过小
        r2 = client.post(
            "/api/elevation/fetch",
            json={"lon": 0, "lat": 0, "radius_km": 1},
        )
        assert r2.status_code == 400

    def test_search_validation(self, client):
        # 空 query
        r = client.post(
            "/api/location/search",
            json={"query": ""},
        )
        assert r.status_code == 422

    def test_save_tmp_helper(self, client):
        """测试 /api/cache/save_tmp 接收 GeoTIFF 字节."""
        # 构造一个最小 GeoTIFF
        import io
        import rasterio
        from rasterio.transform import from_bounds
        import numpy as np

        buf = io.BytesIO()
        profile = {
            "driver": "GTiff", "height": 4, "width": 4, "count": 1,
            "dtype": "float32", "crs": "EPSG:4326",
            "transform": from_bounds(0, 0, 1, 1, 4, 4),
        }
        with rasterio.open(buf, "w", **profile) as dst:
            dst.write(np.zeros((4, 4), dtype="float32"), 1)
        body = buf.getvalue()
        r = client.post(
            "/api/cache/save_tmp",
            content=body,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["size"] == len(body)
        assert "path" in d
        assert Path(d["path"]).exists()
        # 清理
        Path(d["path"]).unlink(missing_ok=True)

    def test_save_text_helper(self, client):
        """测试 /api/cache/save_text 接收 GeoJSON."""
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "test"},
                    "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                }
            ],
        }
        r = client.post(
            "/api/cache/save_text",
            json={"content": json.dumps(geojson)},
        )
        assert r.status_code == 200, r.text
        d = r.json()
        assert "path" in d
        assert Path(d["path"]).exists()
        # 验证
        loaded = json.loads(Path(d["path"]).read_text(encoding="utf-8"))
        assert loaded["type"] == "FeatureCollection"
        assert len(loaded["features"]) == 1
        Path(d["path"]).unlink(missing_ok=True)

    def test_cache_clear(self, client):
        r = client.post("/api/cache/clear")
        assert r.status_code == 200
        assert r.json()["cleared"] is True


# ============ 全流程: 拉 DEM + 水系 -> 调 /api/layers/all ============

@requires_esri
@requires_overpass
class TestEndToEndPipeline:
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)

    def test_full_pipeline(self, client):
        """端到端: 拉阆中 5km DEM + 水系, 保存临时文件, 调 /api/layers/all."""
        import time as _t
        # 1) 拉 DEM
        r1 = client.post(
            "/api/elevation/fetch",
            json={"lon": 105.85, "lat": 31.55, "radius_km": 5},
        )
        assert r1.status_code == 200
        dem = r1.json()

        # 2) 拉水系
        r2 = client.post(
            "/api/water/fetch",
            json={"lon": 105.85, "lat": 31.55, "radius_km": 5},
        )
        assert r2.status_code == 200
        water = r2.json()

        # 3) 保存临时文件
        r3 = client.post(
            "/api/cache/save_tmp",
            content=__import__("base64").b64decode(dem["geotiff_base64"]),
            headers={"Content-Type": "application/octet-stream"},
        )
        assert r3.status_code == 200
        dem_path = r3.json()["path"]

        r4 = client.post(
            "/api/cache/save_text",
            json={"content": json.dumps({"type": "FeatureCollection", "features": water.get("features", [])})},
        )
        assert r4.status_code == 200
        water_path = r4.json()["path"]

        try:
            # 4) 调 /api/layers/all
            r5 = client.get(
                "/api/layers/all",
                params={
                    "dem_path": dem_path,
                    "water_path": water_path,
                    "top_k": 5,
                    "mode": "elevation",
                },
            )
            assert r5.status_code == 200, r5.text[:500]
            d = r5.json()
            assert "basemap" in d
            assert "water" in d
            assert "structured" in d
            assert len(d["structured"]["candidates"]) > 0
            assert d["structured"]["four_beasts"]["xuanwu"] is not None
        finally:
            Path(dem_path).unlink(missing_ok=True)
            Path(water_path).unlink(missing_ok=True)


# 必需 import json
import json
