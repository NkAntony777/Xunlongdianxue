"""AOI 半径约束测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core.aoi_limits import (
    MIN_RADIUS_KM, MAX_RADIUS_KM, validate_radius_km, radius_quality,
)
from engine.api.main import app


class TestAoiLimitsCore:
    def test_validate_ok(self):
        assert validate_radius_km(8) == 8.0

    def test_validate_too_small(self):
        with pytest.raises(ValueError):
            validate_radius_km(MIN_RADIUS_KM - 0.1)

    def test_validate_too_large(self):
        with pytest.raises(ValueError):
            validate_radius_km(MAX_RADIUS_KM + 1)

    def test_quality(self):
        assert radius_quality(8) == "ok"
        assert radius_quality(3.5) == "small"
        assert radius_quality(20) == "large"
        assert radius_quality(1) == "invalid"


class TestAoiAPI:
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(app)

    def test_limits(self, client):
        r = client.get("/api/aoi/limits")
        assert r.status_code == 200
        d = r.json()
        assert d["min_radius_km"] == MIN_RADIUS_KM
        assert d["max_radius_km"] == MAX_RADIUS_KM
        assert "rationale" in d

    def test_validate_endpoint(self, client):
        bad = client.post("/api/aoi/validate", json={"radius_km": 1})
        assert bad.json()["ok"] is False
        good = client.post("/api/aoi/validate", json={"radius_km": 10})
        assert good.json()["ok"] is True
        assert good.json()["quality"] == "ok"

    def test_elevation_rejects_small(self, client):
        r = client.post(
            "/api/elevation/fetch",
            json={"lon": 105.9, "lat": 31.55, "radius_km": 1},
        )
        assert r.status_code == 400
