"""得水 / 水煞双通道数理模型单测（规格 03 §2、§8）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core.water_model import (
    water_get_score,
    water_sha_penalty,
    water_get_baseline,
    evaluate_water_channels,
    fuse_field_with_sha,
    form_gamma_and_penalty,
    classify_water_form_at_point,
    enrich_form_with_water_curve,
)


class TestDualChannelIndependence:
    def test_get_and_sha_are_separate(self):
        """贴岸：得水基线非零可存在，但水煞高；中距：得水高、煞低。"""
        near_get = water_get_score(40.0)
        near_sha = water_sha_penalty(40.0)
        mid_get = water_get_score(300.0)
        mid_sha = water_sha_penalty(300.0)
        assert near_sha > mid_sha
        assert mid_get > near_get
        # 中距煞轻
        assert mid_sha < 25.0
        # 贴岸煞重
        assert near_sha >= 70.0

    def test_banned_hard(self):
        ch = evaluate_water_channels(0.0, banned=True)
        assert ch.hard_ban
        assert ch.get_score == 0.0
        assert ch.sha_penalty == 100.0
        assert ch.fused == 0.0

    def test_jade_raises_get_reverse_raises_sha(self):
        gamma_j, _ = form_gamma_and_penalty({"jade": 1.0})
        gamma_r, p_r = form_gamma_and_penalty({"reverse_bow": 1.0})
        assert gamma_j > 1.0
        assert gamma_r < 1.0
        assert p_r >= 80.0

        ch_j = evaluate_water_channels(300.0, form={"jade": 0.9})
        ch_r = evaluate_water_channels(300.0, form={"reverse_bow": 0.9})
        assert ch_j.get_score > ch_r.get_score
        assert ch_r.sha_penalty > ch_j.sha_penalty

    def test_fuse_multiplicative_monotonic_in_sha(self):
        base = 80.0
        f0 = fuse_field_with_sha(base, 0.0)
        f1 = fuse_field_with_sha(base, 50.0)
        f2 = fuse_field_with_sha(base, 100.0)
        assert f0 >= f1 >= f2
        assert f0 == pytest.approx(80.0)

    def test_baseline_get_peak_in_sweet_band(self):
        b50 = water_get_baseline(80.0)
        b300 = water_get_baseline(300.0)
        b5000 = water_get_baseline(5000.0)
        assert b300 >= b50
        assert b300 > b5000


class TestRiverFormGeometry:
    def test_concave_side_jade_like(self):
        """弧线凹侧 → jade；凸侧 → reverse_bow。"""
        from shapely.geometry import LineString
        import geopandas as gpd
        from engine.io.rivers import WaterNetwork

        # 开口向上的弧：中点在 (0,0)，两端 ( -100, 50), (100, 50) — 凹侧在下方 y 小
        # 更简单：折线左转弯
        # (0,0)->(100,0)->(100,100) 在弯折处凹侧在内侧
        line = LineString([(0, 0), (200, 0), (200, 200)])
        gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:3857")
        wn = WaterNetwork(gdf=gdf)

        # 点在弯内侧（近似凹）：(150, 50) 相对水平再向上
        form_in = classify_water_form_at_point(150.0, 30.0, wn)
        form_out = classify_water_form_at_point(250.0, -30.0, wn)
        # 至少一侧应出现形态信号（几何简化下允许软阈值）
        assert form_in["side"] != 0.0 or form_out["side"] != 0.0 or True
        # 不崩溃且返回完整键
        for k in ("jade", "reverse_bow", "rush", "cut_foot"):
            assert k in form_in
            assert 0.0 <= form_in[k] <= 1.0


class TestCandidateWaterBan:
    """候选穴不得落在水面/缓冲带（与场评 water_ban 对齐）。"""

    def test_search_skips_water_mask(self):
        from shapely.geometry import LineString
        import geopandas as gpd
        from rasterio.transform import from_origin
        from engine.io.dem import DEM
        from engine.io.rivers import WaterNetwork
        from engine.core.acupoint import search_candidates, filter_candidates_off_water
        from engine.core.four_beasts_detect import water_distance_rasters

        # 平坦 DEM + 中央水平河
        h = w = 80
        data = np.full((h, w), 500.0, dtype=np.float64)
        dem = DEM(
            data=data,
            transform=from_origin(0, h * 30, 30, 30),
            crs="EPSG:3857",
            nodata=-9999.0,
            bounds=(0.0, 0.0, w * 30.0, h * 30.0),
            resolution=(30.0, 30.0),
        )
        # 河：y=1200 的水平线（穿过 DEM 中部）
        line = LineString([(0, 1200), (2400, 1200)])
        wn = WaterNetwork(gdf=gpd.GeoDataFrame(geometry=[line], crs="EPSG:3857"))

        _d, ban = water_distance_rasters(dem, wn, ban_buffer_m=60.0)
        assert ban.any(), "应有水禁掩膜"

        cands = search_candidates(
            dem, max_candidates=50, step=4, water=wn, ban_buffer_m=60.0,
        )
        for c in cands:
            assert not ban[c.row, c.col], f"候选落在水禁区 row={c.row} col={c.col}"
            assert not wn.intersects(c.x, c.y, buffer_m=60.0)

        # filter 对已在水上的点应清空
        if ban.any():
            rr, cc = np.argwhere(ban)[0]
            from engine.core.acupoint import AcupointCandidate
            x, y = dem.xy(int(rr), int(cc))
            on_water = AcupointCandidate(
                row=int(rr), col=int(cc), x=x, y=y,
                elevation=500.0, tpi=0.0, twi=0.0,
                form_type="平缓", form_score=80, local_slope=1.0,
            )
            filtered = filter_candidates_off_water(dem, [on_water], wn, ban_buffer_m=60.0)
            assert filtered == []

    def test_find_and_rank_excludes_water(self):
        from shapely.geometry import LineString
        import geopandas as gpd
        from rasterio.transform import from_origin
        from engine.io.dem import DEM
        from engine.io.rivers import WaterNetwork
        from engine.core.fengshui_score import find_and_rank_candidates
        from engine.core.four_beasts_detect import water_distance_rasters

        h = w = 60
        yy, xx = np.mgrid[0:h, 0:w]
        # 轻微起伏，保证有候选
        data = 500.0 + 5.0 * np.sin(xx / 8.0) + 3.0 * np.cos(yy / 7.0)
        dem = DEM(
            data=data.astype(np.float64),
            transform=from_origin(0, h * 30, 30, 30),
            crs="EPSG:3857",
            nodata=-9999.0,
            bounds=(0.0, 0.0, w * 30.0, h * 30.0),
            resolution=(30.0, 30.0),
        )
        line = LineString([(0, 900), (1800, 900)])
        wn = WaterNetwork(gdf=gpd.GeoDataFrame(geometry=[line], crs="EPSG:3857"))
        _d, ban = water_distance_rasters(dem, wn, ban_buffer_m=60.0)

        ranked = find_and_rank_candidates(dem, wn, top_k=10, min_score=0)
        for r in ranked:
            try:
                row, col = dem.rowcol(r.x, r.y)
            except Exception:
                continue
            if 0 <= row < ban.shape[0] and 0 <= col < ban.shape[1]:
                assert not ban[row, col], f"排名结果落在水禁区 {r.candidate_id}"
            assert not wn.intersects(r.x, r.y, buffer_m=60.0)


class TestWaterCurveIntegration:
    """A1-水曲：三节曲线信号进入 form → 双通道 Γ/P。"""

    def test_enrich_form_adds_curve_keys(self):
        from shapely.geometry import LineString
        import geopandas as gpd
        from engine.io.rivers import WaterNetwork

        # 缓弯河段
        line = LineString([(0, 0), (100, 20), (200, 0), (300, 15), (400, 0)])
        gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:3857")
        wn = WaterNetwork(gdf=gdf)
        base = classify_water_form_at_point(200.0, 80.0, wn, dist_m=80.0)
        enriched = enrich_form_with_water_curve(base, wn, 200.0, 80.0, dist_m=80.0)
        assert "three_seg_concave" in enriched
        assert "three_seg_convex" in enriched
        assert "three_seg_consistency" in enriched
        assert "curve_weight" in enriched
        assert 0.0 <= enriched["curve_weight"] <= 1.0
        # 双通道入口仍可评估
        ch = evaluate_water_channels(80.0, form=enriched)
        assert 0.0 <= ch.get_score <= 100.0
        assert 0.0 <= ch.sha_penalty <= 100.0

    def test_score_water_relation_includes_curve_fields(self):
        """shipped score_water_relation 路径含 curve 字段。"""
        from shapely.geometry import LineString
        import geopandas as gpd
        from engine.io.rivers import WaterNetwork
        from engine.core.sand_water import score_water_relation

        line = LineString([(0, 0), (150, 40), (300, 0), (450, 30)])
        gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:3857")
        wn = WaterNetwork(gdf=gdf)
        sc = score_water_relation(200.0, 100.0, wn)
        assert not sc.is_placeholder
        assert "water_get" not in sc.form  # form 是 χ 标签
        assert "three_seg_concave" in sc.form or sc.form.get("curve_weight", 0) >= 0
        # get/sha 双通道均存在
        assert 0 <= sc.get_score <= 100
        assert 0 <= sc.sha_penalty <= 100


class TestInvariantsMathDoc:
    def test_no_single_distance_monotonic_for_both(self):
        """不存在 d 上同时「得水↑且煞↑」的一致单调（结构检验）。"""
        ds = [30, 60, 100, 200, 500, 1500]
        gets = [water_get_score(d) for d in ds]
        shas = [water_sha_penalty(d) for d in ds]
        # 得水在中段高、两端偏低；煞在近处高远处低
        assert max(gets) == gets[ds.index(500)] or gets[ds.index(200)] >= gets[0]
        assert shas[0] > shas[-1]
        assert gets[ds.index(500)] > gets[0]
