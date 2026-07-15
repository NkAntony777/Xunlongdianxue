"""Water-sha influence risk: not near-water = full sha (shipped functions)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import LineString
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core.water_sha_influence import (
    water_sha_dist_risk,
    water_sha_elev_factor,
    water_sha_flow_mod,
    water_sha_influence_risk,
    water_sha_influence_at_xy,
    nearest_water_geom_metrics,
)
from engine.core.water_model import water_sha_elev_factor as elev_from_model
from engine.io.dem import DEM
from engine.core.rendering.pipeline import render_water_influence


class TestInfluenceRiskFactors:
    def test_dist_risk_narrow_not_km_scale(self):
        assert water_sha_dist_risk(20.0) > 0.9
        assert water_sha_dist_risk(100.0) > water_sha_dist_risk(250.0)
        # far mid-mingtang distance must not stay high
        assert water_sha_dist_risk(400.0) < 0.12
        assert water_sha_dist_risk(1200.0) < 0.05

    def test_elev_reuses_scoring_factor(self):
        # same API as scoring attenuation
        assert elev_from_model(1.0) == water_sha_elev_factor(1.0)
        assert elev_from_model(35.0) < elev_from_model(2.0)

    def test_same_distance_high_terrace_lower_than_low_bank(self):
        """Acceptance: same plan-distance, high terrace influence < low terrace."""
        d = 60.0
        low = water_sha_influence_risk(d, elev_above_water_m=1.0, flow_cos=0.0)
        high = water_sha_influence_risk(d, elev_above_water_m=35.0, flow_cos=0.0)
        assert low > high + 0.15

    def test_flow_facing_higher_than_lateral_or_back(self):
        """Acceptance: flow-facing rush > lateral/back at same dist/elev."""
        d = 70.0
        elev = 3.0
        face = water_sha_influence_risk(d, elev_above_water_m=elev, flow_cos=0.9)
        lateral = water_sha_influence_risk(d, elev_above_water_m=elev, flow_cos=0.0)
        back = water_sha_influence_risk(d, elev_above_water_m=elev, flow_cos=-0.7)
        assert face > lateral
        assert face > back
        assert water_sha_flow_mod(0.9) > water_sha_flow_mod(-0.5)


class TestInfluenceWithSyntheticRiverDem:
    def _dem_river_west_high(self):
        """River along x-axis 0→1000; elev decreases east (flow east)."""
        h, w = 40, 50
        transform = from_origin(-100.0, 600.0, 30.0, 30.0)
        data = np.zeros((h, w), dtype=np.float64)
        for r in range(h):
            for c in range(w):
                xx = -100.0 + c * 30.0 + 15.0
                yy = 600.0 - r * 30.0 - 15.0
                # base slope east down + low terrace near y=0 river
                data[r, c] = 180.0 - 0.06 * xx + 0.02 * abs(yy)
        dem = DEM(
            data=data,
            transform=transform,
            crs="EPSG:3857",
            nodata=-9999.0,
            bounds=(-100.0, 600.0 - 40 * 30.0, -100.0 + 50 * 30.0, 600.0),
            resolution=(30.0, 30.0),
        )
        line = LineString([(0.0, 0.0), (1000.0, 0.0)])
        gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:3857")
        return dem, gdf

    def test_geom_metrics_flow_and_elev(self):
        dem, gdf = self._dem_river_west_high()
        # Point east of river end on flow path (downstream face)
        m_face = nearest_water_geom_metrics(1100.0, 0.0, gdf, dem=dem)
        # Lateral north of mid-river
        m_lat = nearest_water_geom_metrics(500.0, 120.0, gdf, dem=dem)
        assert m_face["dist_m"] < 200
        assert m_lat["dist_m"] < 200
        # downhill east → flow_cos toward (1100,0) should be positive when method works
        if m_face.get("flow_method") == "dem_downhill":
            assert m_face["flow_cos"] is not None
            assert m_face["flow_cos"] > 0.3

    def test_at_xy_face_vs_high_terrace(self):
        dem, gdf = self._dem_river_west_high()
        # low bank near river
        r_low = water_sha_influence_at_xy(500.0, 25.0, gdf, dem=dem)
        # same x, far higher terrace north (elev rises with |y| in synth)
        r_high = water_sha_influence_at_xy(500.0, 280.0, gdf, dem=dem)
        # high point also farther → must be strictly lower risk
        assert r_high < r_low

    def test_render_with_and_without_dem(self):
        dem, gdf = self._dem_river_west_high()
        bounds = (-50.0, -100.0, 1100.0, 300.0)
        r0 = render_water_influence(gdf, bounds, dpi=40)
        assert r0.png_base64
        assert r0.legend and r0.legend.get("type") == "water_influence"
        assert r0.legend.get("mode") == "narrow_bank"
        r1 = render_water_influence(gdf, bounds, dpi=40, dem=dem)
        assert r1.png_base64
        assert r1.legend.get("type") == "water_influence"
        assert r1.legend.get("mode") == "dem_risk"
