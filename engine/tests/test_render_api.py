"""测试 render 函数和 API layer 端点。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from engine.io.dem import load_dem
from engine.io.rivers import load_water, WaterNetwork
from engine.core.render import (
    render_basemap, render_slope_basemap, render_contours,
    render_water, render_water_influence, render_score_grid,
    four_beasts_geojson, candidates_geojson, ridges_geojson,
)
from engine.core.fengshui_score import (
    _sanitize, find_and_rank_candidates,
)
from engine.tests.fixtures.make_synthetic import make_synthetic_dem, make_synthetic_rivers

FIXTURES = Path(__file__).parent / "fixtures"


# ============ Fixtures ============

@pytest.fixture(scope="module")
def synth_dem_path():
    p = FIXTURES / "synth_dem.tif"
    if not p.exists():
        make_synthetic_dem(p)
    return p


@pytest.fixture(scope="module")
def synth_rivers_path():
    p = FIXTURES / "synth_rivers.geojson"
    if not p.exists():
        make_synthetic_rivers(p)
    return p


@pytest.fixture(scope="module")
def dem(synth_dem_path):
    return load_dem(synth_dem_path)


@pytest.fixture(scope="module")
def water(synth_rivers_path):
    return load_water(synth_rivers_path)


@pytest.fixture(scope="module")
def app():
    """Build a minimal test app for API endpoint tests."""
    from engine.api.main import app
    return app


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


# ============ _sanitize ============

class TestSanitize:
    def test_handles_inf(self):
        assert _sanitize(float("inf")) is None
        assert _sanitize(float("-inf")) is None

    def test_handles_nan(self):
        assert _sanitize(float("nan")) is None

    def test_passes_normal_floats(self):
        assert _sanitize(3.14) == 3.14
        assert _sanitize(0.0) == 0.0

    def test_handles_dict_with_inf(self):
        d = {"a": 1.0, "b": float("inf"), "c": float("nan")}
        cleaned = _sanitize(d)
        assert cleaned["a"] == 1.0
        assert cleaned["b"] is None
        assert cleaned["c"] is None

    def test_handles_nested(self):
        d = {"x": [float("inf"), 1.0, {"y": float("-inf")}]}
        cleaned = _sanitize(d)
        assert cleaned["x"][0] is None
        assert cleaned["x"][1] == 1.0
        assert cleaned["x"][2]["y"] is None

    def test_handles_numpy_float(self):
        assert _sanitize(np.float64(float("inf"))) is None
        assert _sanitize(np.float32(1.5)) == 1.5

    def test_handles_numpy_int(self):
        assert _sanitize(np.int32(42)) == 42


# ============ Render functions ============

class TestRenderBasemap:
    def test_render_elevation(self, dem):
        result = render_basemap(dem, dpi=50)
        assert result.width > 0
        assert result.height > 0
        assert len(result.bbox) == 4
        assert len(result.png_base64) > 100
        assert result.legend is not None
        assert "vmin" in result.legend

    def test_render_slope(self, dem):
        result = render_slope_basemap(dem, dpi=50)
        assert result.width > 0
        assert result.height > 0
        assert len(result.bbox) == 4
        assert len(result.png_base64) > 100

    def test_render_without_dem_fails(self):
        with pytest.raises((TypeError, AttributeError, Exception)):
            render_basemap(None)  # type: ignore


class TestRenderWater:
    def test_render_with_water(self, dem, water):
        gdf = water.gdf
        result = render_water(gdf, dem.bounds, dpi=50)
        assert result.width > 0
        assert len(result.png_base64) > 100

    def test_render_without_water(self, dem):
        result = render_water(None, dem.bounds, dpi=50)
        if result is not None:
            assert len(result.png_base64) >= 0  # just don't crash


class TestRenderContours:
    def test_render_contours(self, dem):
        result = render_contours(dem, contour_interval=30, dpi=50)
        assert result.width > 0
        assert len(result.png_base64) > 100
        assert result.geojson is not None


class TestRenderWaterInfluence:
    def test_render_influence(self, dem, water):
        gdf = water.gdf
        result = render_water_influence(gdf, dem.bounds, buffer_m=200, dpi=50)
        assert result.width > 0
        assert len(result.png_base64) > 100


class TestRenderScoreGrid:
    def test_render_score(self, dem):
        from engine.core.four_beasts_detect import compute_score_grid
        grid = compute_score_grid(dem, sample_step=10)
        result = render_score_grid(dem, grid, dpi=50)
        assert result.width > 0
        assert len(result.png_base64) > 100


# ============ GeoJSON helpers ============

class TestGeoJSONHelpers:
    def test_four_beasts_geojson(self):
        fb = {
            "shaozu": {"x": 100.0, "y": 200.0},
            "xuanwu": {"x": 150.0, "y": 180.0},
            "zhuque": {"x": 140.0, "y": 220.0},
            "qinglong": {"x": 170.0, "y": 190.0},
            "baihu": {"x": 130.0, "y": 195.0},
        }
        gj = four_beasts_geojson(fb)
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 5
        names = {f["properties"]["label"] for f in gj["features"]}
        assert "少祖" in names
        assert "玄武" in names

    def test_four_beasts_empty(self):
        gj = four_beasts_geojson({})
        assert len(gj["features"]) == 0

    def test_candidates_geojson(self):
        cands = [
            {"id": "C-001", "x": 100, "y": 200, "overall_score": 75,
             "form_type": "窝穴", "rank": 1},
            {"id": "C-002", "x": 110, "y": 210, "overall_score": 60,
             "form_type": "钳穴", "rank": 2},
        ]
        gj = candidates_geojson(cands)
        assert len(gj["features"]) == 2
        assert gj["features"][0]["properties"]["id"] == "C-001"
        assert gj["features"][0]["geometry"]["coordinates"] == [100, 200]

    def test_ridges_geojson(self):
        ridges_data = [
            {"coords": [[0, 0], [10, 10], [20, 5]], "rank": 1},
            {"coords": [[5, 5], [15, 15]], "rank": 2},
        ]
        gj = ridges_geojson(ridges_data)
        assert len(gj["features"]) == 2
        assert gj["features"][0]["properties"]["rank"] == 1


# ============ API endpoints (synthetic DEM) ============

# These tests need the real server with paths to synthetic data.
# We use the FastAPI TestClient but need to construct URLs with dem_path/water_path.
# The TestClient can't mount static files, so we test the API directly.

synth_dem_str = str(FIXTURES / "synth_dem.tif")
synth_water_str = str(FIXTURES / "synth_rivers.geojson")


class TestAPILayers:
    """Integration tests for API layer endpoints."""

    def test_four_beasts_endpoint(self, client):
        """GET /api/layers/four-beasts returns 200 + valid structure."""
        resp = client.get(
            "/api/layers/four-beasts",
            params={
                "dem_path": synth_dem_str,
                "water_path": synth_water_str,
                "top_k": 3,
            },
        )
        assert resp.status_code == 200, resp.text[:200]
        data = resp.json()
        assert "center" in data
        assert "facing" in data
        assert "four_beasts" in data
        assert "geojson" in data
        beasts = data["four_beasts"]
        assert "shaozu" in beasts
        assert "xuanwu" in beasts
        assert "zhuque" in beasts
        assert "qinglong" in beasts
        assert "baihu" in beasts

    def test_four_beasts_no_water(self, client):
        """Without water, should still work (water=None)."""
        resp = client.get(
            "/api/layers/four-beasts",
            params={"dem_path": synth_dem_str, "top_k": 3},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "four_beasts" in data

    def test_candidates_endpoint(self, client):
        """GET /api/layers/candidates returns 200 + ranked candidates."""
        resp = client.get(
            "/api/layers/candidates",
            params={
                "dem_path": synth_dem_str,
                "water_path": synth_water_str,
                "top_k": 5,
                "min_score": 0,
            },
        )
        assert resp.status_code == 200, resp.text[:200]
        data = resp.json()
        assert "candidates" in data
        assert "geojson" in data
        cands = data["candidates"]
        assert len(cands) > 0
        # Ranks should be sequential
        for i, c in enumerate(cands):
            assert c["rank"] == i + 1
        # Should be no inf/nan in JSON (sanitizer worked)
        import json
        json.dumps(data)  # should not raise

    def test_all_endpoint(self, client):
        """GET /api/layers/all returns 200 + all layers."""
        resp = client.get(
            "/api/layers/all",
            params={
                "dem_path": synth_dem_str,
                "water_path": synth_water_str,
                "top_k": 3,
                "mode": "elevation",
            },
        )
        assert resp.status_code == 200, resp.text[:500]
        data = resp.json()
        # Top-level keys
        assert "bbox" in data
        assert "dem" in data
        assert "basemap" in data
        assert "water" in data
        assert "water_influence" in data
        assert "score" in data
        assert "contours" in data
        assert "structured" in data
        # Structured
        s = data["structured"]
        assert "four_beasts" in s
        assert "candidates" in s
        assert "center" in s
        assert "facing" in s
        assert len(s["candidates"]) > 0
        # 场评最高点 ≡ 四象穴心
        peak = data["score"].get("peak_xy") or (data["score"].get("legend") or {}).get("peak_xy")
        assert peak is not None
        assert s["center"] is not None
        assert abs(peak[0] - s["center"]["x"]) < 1e-6
        assert abs(peak[1] - s["center"]["y"]) < 1e-6
        assert s.get("center_source") in (
            "score_field_peak", "explicit", "top_candidate", "dem_center",
        )
        # PNG base64 strings should be valid
        import base64
        base64.b64decode(data["basemap"]["png_base64"])  # no error
        base64.b64decode(data["water"]["png_base64"])
        base64.b64decode(data["score"]["png_base64"])

    def test_all_endpoint_slope_mode(self, client):
        """Slope mode should produce a different basemap legend."""
        resp = client.get(
            "/api/layers/all",
            params={
                "dem_path": synth_dem_str,
                "water_path": synth_water_str,
                "top_k": 3,
                "mode": "slope",
            },
        )
        assert resp.status_code == 200, resp.text[:200]
        data = resp.json()
        assert data["basemap"]["mode"] == "slope"

    def test_all_no_water(self, client):
        """Without water path, all should still work."""
        resp = client.get(
            "/api/layers/all",
            params={"dem_path": synth_dem_str, "top_k": 3},
        )
        assert resp.status_code == 200, resp.text[:200]
        data = resp.json()
        assert "structured" in data


class TestAPIHealth:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ============ _all_structured_layers integration ============

class TestAllStructuredLayers:
    def test_returns_valid_structure(self, dem, water):
        from engine.api.routers.layers import _all_structured_layers
        from engine.api.schemas.models import AnalyzeRequest

        req = AnalyzeRequest(
            dem_path=str(FIXTURES / "synth_dem.tif"),
            water_path=str(FIXTURES / "synth_rivers.geojson"),
            top_k=5, min_score=0,
        )
        result = _all_structured_layers(req, dem, water)
        assert "center" in result
        assert "facing" in result
        assert "four_beasts" in result
        assert "four_beasts_geojson" in result
        assert "ridges_geojson" in result
        assert "candidates" in result
        assert "candidates_geojson" in result
        assert len(result["candidates"]) > 0

    def test_handles_no_water(self, dem):
        from engine.api.routers.layers import _all_structured_layers
        from engine.api.schemas.models import AnalyzeRequest

        req = AnalyzeRequest(
            dem_path=str(FIXTURES / "synth_dem.tif"),
            water_path=None,
            top_k=3, min_score=0,
        )
        result = _all_structured_layers(req, dem, None)
        assert "candidates" in result
        assert len(result["candidates"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
