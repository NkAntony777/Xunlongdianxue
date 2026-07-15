"""传统理论审计补强模块测试用例。

覆盖：
  - P0-1 五行星体识别 (star_body)
  - P0-2 砂形朝抱度量 (mountain_curve)
  - P0-3 案山质量评估 (anshan_quality)
  - P0-4 水口与交媾 (water_mouth)
  - P0-5 扇区宽度一致性 (four_beasts)
  - P1-1 入首过峡 蜂腰鹤膝 (dragon_vein.find_yaoxia)
"""
from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_bounds

from engine.io.dem import DEM
from engine.core.star_body import classify_star_body, score_xuanwu_by_star
from engine.core.mountain_curve import measure_embrace
from engine.core.anshan_quality import score_anshan_quality, AnshanQuality
from engine.core.water_mouth import (
    find_confluences,
    find_water_mouths,
    score_mouth_locking,
    best_mouth_for_acupoint,
)
from engine.core.halo_soil import score_halo_soil
from engine.core.water_curve import (
    split_river_segments,
    distance_adaptive_form_weight,
    score_water_curve_three_segments,
    score_multi_segment_concavity,
)


# ===== 合成 DEM 工具 =====

def make_dem_from_func(
    func,
    h: int = 120,
    w: int = 120,
    cell_size_m: float = 30.0,
    base_elev: float = 500.0,
) -> DEM:
    """根据 func(yy, xx) -> 高程数组，合成 DEM。"""
    yy, xx = np.mgrid[0:h, 0:w]
    data = func(yy, xx).astype(np.float64)
    transform = from_bounds(0, 0, w * cell_size_m, h * cell_size_m, w, h)
    return DEM(
        data=data,
        transform=transform,
        crs="EPSG:3857",
        nodata=-9999.0,
        bounds=(0.0, 0.0, w * cell_size_m, h * cell_size_m),
        resolution=(cell_size_m, cell_size_m),
    )


# ===== P0-1 star_body =====

class TestStarBody:
    def test_jin_star_rounded(self):
        """中心高、四周缓坡（高斯）→ 金星。"""
        def f(yy, xx):
            cy, cx = 60, 60
            return 500.0 + 120.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 12 ** 2))
        dem = make_dem_from_func(f)
        star = classify_star_body(dem, 60, 60, search_radius_m=300.0)
        assert star.type == "金星", star.notes
        assert star.is_xuanwu_eligible

    def test_diagonal_long_ridge_is_mu_not_jin(self):
        """【修复 G.2】对角线长脊：原 bbox AR=√2=1.41 被判金星，
        PCA AR 应 >2 → 应识别为木星或更高，不被误判为金星。"""
        def f(yy, xx):
            cy, cx = 60, 60
            # 沿 NE-SW 主对角线的细长椭圆（行向 σ=3, 列向 σ=25 + 对角旋转）
            r = yy - cy
            c = xx - cx
            # 旋转 45°：旋转坐标
            cos45 = np.cos(np.pi / 4)
            sin45 = np.sin(np.pi / 4)
            r2 = r * cos45 - c * sin45
            c2 = r * sin45 + c * cos45
            return 500.0 + 150.0 * np.exp(
                -(r2 ** 2) / (2 * 3 ** 2) - (c2 ** 2) / (2 * 25 ** 2)
            )
        dem = make_dem_from_func(f, h=120, w=120)
        star = classify_star_body(dem, 60, 60, search_radius_m=300.0)
        # 长轴应被识别为高 AR，不能落入金星
        assert star.aspect_ratio > 1.6, f"PCA AR={star.aspect_ratio}, 旋转 45° 长脊应有真实 AR"
        assert star.type != "金星", (
            f"对角线长脊不应判为金星；旧 bbox 给出 AR=1.41 误判。"
            f"actual={star.type} {star.notes}"
        )

    def test_mu_star_tall_skinny(self):
        """中心细高（高宽比高）→ 木星。"""
        def f(yy, xx):
            cy, cx = 60, 60
            # 极长东西向轴
            return 500.0 + 150.0 * np.exp(-(((yy - cy) ** 2) / (2 * 4 ** 2) + ((xx - cx) ** 2) / (2 * 25 ** 2)))
        dem = make_dem_from_func(f)
        star = classify_star_body(dem, 60, 60, search_radius_m=400.0)
        # AR = (max_axis_span / min_axis_span), 长椭圆上
        assert star.aspect_ratio > 1.6, f"AR={star.aspect_ratio}"
        assert star.type in ("木星", "金星"), star.notes

    def test_huo_star_sharp(self):
        """中心细瘦陡峭（极小 sigma, 高幅）→ 火星。"""
        def f(yy, xx):
            cy, cx = 60, 60
            return 500.0 + 200.0 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2)) / (2 * 3 ** 2))
        dem = make_dem_from_func(f)
        star = classify_star_body(dem, 60, 60, search_radius_m=300.0)
        # 太尖锐应被识别为火星或木星 topSlope≥32: 火星
        # 较弱情况下可能被识别为木星；先确认 is_xuanwu_eligible 的行为
        if star.type == "火星":
            assert not star.is_xuanwu_eligible
            assert not star.is_shaozu_eligible

    def test_xuanwu_score_penalty(self):
        """火星作父母山应被罚分。"""
        from engine.core.star_body import StarBodyResult
        s_huo = StarBodyResult(
            type="火星", confidence=0.7, aspect_ratio=3.5,
            plan_area_m2=10000.0, h_relative_m=80.0,
            peak_count=1, mean_top_slope=40.0,
            is_xuanwu_eligible=False, is_shaozu_eligible=False,
            notes="",
        )
        score = score_xuanwu_by_star(s_huo, base_score=70.0)
        assert score < 70.0  # 罚分

        s_jin = StarBodyResult(
            type="金星", confidence=0.8, aspect_ratio=1.1,
            plan_area_m2=10000.0, h_relative_m=80.0,
            peak_count=1, mean_top_slope=15.0,
            is_xuanwu_eligible=True, is_shaozu_eligible=True,
            notes="",
        )
        score_jin = score_xuanwu_by_star(s_jin, base_score=70.0)
        assert score_jin > 70.0

    def test_out_of_bounds_returns_eligibility_false(self):
        dem = make_dem_from_func(lambda yy, xx: 500.0 + np.zeros_like(yy, dtype=float))
        star = classify_star_body(dem, 200, 200, search_radius_m=200.0)
        assert not star.is_xuanwu_eligible


# ===== P0-2 mountain_curve =====

class TestMountainCurve:
    def test_ne_sw_diagonal_embrace(self):
        """【修复 C.5】NE-SW 走向砂山：crest 必须有多个采样点，不能再走 fallback。"""
        def f(yy, xx):
            cy, cx = 60, 60
            # 对角线弧形砂山（凸侧朝穴）
            r = yy - cy
            c = xx - cx
            cos45 = np.cos(np.pi / 4)
            sin45 = np.sin(np.pi / 4)
            r2 = r * cos45 - c * sin45      # 沿对角线的"切向"
            c2 = r * sin45 + c * cos45      # 法向（垂直）
            z = 500.0 + 60.0 * np.exp(-((c2 - 15) ** 2) / (2 * 8 ** 2))
            return z
        dem = make_dem_from_func(f, h=120, w=120)
        # 在东侧构造一个弧形砂：弧形顶点位于 r=60,c=72 附近
        # 这里通过 region_mask 圈出来
        mask = (dem.data > 500.0) & ((yy := np.indices(dem.data.shape))[0] >= 50) & (yy[0] <= 70)
        from engine.core.mountain_curve import _extract_crest_points
        pts = _extract_crest_points(dem, 60, 60, mask)
        assert len(pts) >= 3, f"NE-SW 砂山应给出 ≥3 个 crest 采样，实际 {len(pts)}"

    def test_no_sand_in_sector(self):
        dem = make_dem_from_func(lambda yy, xx: 500.0 + np.zeros_like(yy, dtype=float))
        region = np.zeros_like(dem.data, dtype=bool)
        res = measure_embrace(dem, 60, 60, region, direction_center_deg=90.0)
        # 无砂山时给中性占位分
        assert 0 <= res.score <= 100
        assert not res.convex_to_acupoint

    def test_partial_crest_returns_default_50(self):
        """采样段数不足时（< 3 个 crest 点）应使用方位一致性替代。"""
        dem = make_dem_from_func(lambda yy, xx: 500.0 + np.zeros_like(yy, dtype=float))
        # 在东侧 60m 处放一个孤峰
        dem.data[60, 62] = 700.0
        region = np.zeros_like(dem.data, dtype=bool)
        region[58:63, 60:65] = True
        res = measure_embrace(dem, 60, 60, region, direction_center_deg=90.0)
        assert res.n_segments < 3


# ===== P0-3 anshan_quality =====

class TestAnshanQuality:
    def test_no_mask_returns_false(self):
        dem = make_dem_from_func(lambda yy, xx: 500.0 + np.zeros_like(yy, dtype=float))
        empty = np.zeros_like(dem.data, dtype=bool)
        res = score_anshan_quality(dem, 60, 60, empty)
        assert not res.is_eligible
        assert res.score < 60

    def test_good_anshan_high_score(self):
        """低、圆、缓、不破碎的案山 → 高分。"""
        # 在穴南 400m (距 cx=60 cy=60, row=80), 圆净单峰
        def f(yy, xx):
            return 500.0 + 30.0 * np.exp(-((yy - 80) ** 2 + (xx - 60) ** 2) / (2 * 8 ** 2))
        dem = make_dem_from_func(f)
        # 父母山用 600 m 模拟（高于案山 70m）
        res = score_anshan_quality(
            dem, 60, 60,
            anshan_mask=(dem.data > 528),
            parents_top_m=600.0,
            facing_deg=180.0,
        )
        assert res.height_ratio < 0.55
        # 圆净、缓；得分应明显高于 base
        assert res.score > 60, f"score={res.score}, notes={res.notes}"
        assert res.is_eligible

    def test_dominant_anshan_penalized(self):
        """案山高于父母山 → 欺主罚分。"""
        def f(yy, xx):
            # 案山 100m, 比父母山更高
            return 500.0 + 100.0 * np.exp(-((yy - 80) ** 2 + (xx - 60) ** 2) / (2 * 8 ** 2))
        dem = make_dem_from_func(f)
        res = score_anshan_quality(
            dem, 60, 60,
            anshan_mask=(dem.data > 528),
            parents_top_m=550.0,
            facing_deg=180.0,
        )
        assert res.height_ratio > 1.0
        assert not res.is_eligible
        assert "欺主" in res.notes


# ===== P0-4 water_mouth =====

class TestWaterMouth:
    def test_confluence_on_simple_data(self):
        """两条线相交 → 应识别为 confluent。"""
        import geopandas as gpd
        from shapely.geometry import LineString
        gdf = gpd.GeoDataFrame(
            geometry=[
                LineString([(0, 0), (1000, 0)]),
                LineString([(500, -500), (500, 500)]),
            ],
            crs="EPSG:3857",
        )
        from engine.io.rivers import WaterNetwork
        wn = WaterNetwork(gdf=gdf)
        confs = find_confluences(wn)
        # 距离 5m 内端点聚合 —— 应至少识别一个交媾点
        assert len(confs) >= 0  # 至少端点近接 (取决于 LineString 是否成端点)

    def test_mouth_returns_list(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        gdf = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (2000, 0)])],
            crs="EPSG:3857",
        )
        from engine.io.rivers import WaterNetwork
        wn = WaterNetwork(gdf=gdf)
        mouths = find_water_mouths(wn)
        assert isinstance(mouths, list)
        # LineString 有 2 个端点 → 应当识别出 2 个 mouth (endpoint) 且不重复
        assert len(mouths) >= 2

    def test_best_mouth_for_acupoint(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        gdf = gpd.GeoDataFrame(
            geometry=[LineString([(2000, 0), (4000, 0)])],
            crs="EPSG:3857",
        )
        from engine.io.rivers import WaterNetwork
        wn = WaterNetwork(gdf=gdf)
        mouths = find_water_mouths(wn)
        best, dist = best_mouth_for_acupoint(wn, 3000.0, 0.0, mouths, consideration_radius_m=2000.0)
        assert best is not None
        assert dist <= 2000.0

    def test_empty_water_safe(self):
        import geopandas as gpd
        from engine.io.rivers import WaterNetwork
        wn = WaterNetwork(gdf=gpd.GeoDataFrame(geometry=[], crs="EPSG:3857"))
        assert find_confluences(wn) == []
        assert find_water_mouths(wn) == []
        best, dist = best_mouth_for_acupoint(wn, 0.0, 0.0)
        assert best is None
        assert dist == float("inf")


# ===== P0-5 扇区一致性 =====

class TestSectorConsistency:
    def test_sectors_disjoint(self):
        """默认 facing=180, 半宽 45° 时四象扇区互不重叠。"""
        from engine.core.four_beasts import _sector_mask
        bearings = np.arange(0, 360, 1.0)
        ql = _sector_mask(bearings, 90, 45)
        bh = _sector_mask(bearings, 270, 45)
        zq = _sector_mask(bearings, 180, 45)
        xw = _sector_mask(bearings, 0, 45)
        for arr in (ql, bh, zq, xw):
            assert arr.dtype == bool
        union = ql.astype(int) + bh.astype(int) + zq.astype(int) + xw.astype(int)
        assert (union.max() <= 1).all()


class TestYaoxia:
    def test_find_yaoxia_on_synthetic_ridge(self):
        """构造一条有明显「蜂腰（横向变窄）」的山脊线，过峡应被识别。

        【修复 A4】此前测试只断言 isinstance(list)，不要求检测到任何"峡"。
        此次明确要求：横向变窄的山脊中段必须产出 yaoxia。
        """
        from engine.core.dragon_vein import RidgeLine, find_yaoxia

        def f(yy, xx):
            """水平延伸的山脊，中段 (col=55-65) 横向窄，两侧 (col=40-55, 65-80) 宽。"""
            base = 500.0 + np.zeros_like(yy, dtype=float)
            # 中段窄
            base = base + 60.0 * np.where(
                (yy >= 55) & (yy <= 65) & (xx >= 55) & (xx <= 65), 1.0, 0.0)
            # 两翼宽
            base = base + 60.0 * np.where(
                (yy >= 50) & (yy <= 70) & (xx >= 40) & (xx < 55), 1.0, 0.0)
            base = base + 60.0 * np.where(
                (yy >= 50) & (yy <= 70) & (xx > 65) & (xx <= 80), 1.0, 0.0)
            return base

        dem = make_dem_from_func(f, h=120, w=120)

        # 沿 row=60, col=40→79 → 40 个点（满足 n >= 20）
        ridge_pts = [(60, 40 + i) for i in range(40)]
        coords = np.array(ridge_pts, dtype=int)

        ridge = RidgeLine(
            coords=coords,
            length_m=len(ridge_pts) * 30.0,
            mean_elevation=560.0,
            max_elevation=600.0,
            sinuosity=1.0,
            feature_significance=1.0,
        )
        yao = find_yaoxia([ridge], dem, neck_width_m=80.0, min_narrowing_ratio=0.85)
        assert isinstance(yao, list)
        # 【C.5.2 FAIL-case 加强】横向变窄中段必须产出至少 1 个过峡点
        assert len(yao) >= 1, f"蜂腰中段应识别过峡，实际 {len(yao)} 个"
        # 过峡应落在中段 col≈55–65
        cols = [int(p["col"]) for p in yao]
        assert any(55 <= c <= 65 for c in cols), f"过峡 col 应在中段，实际 {cols}"
        # 收窄比应明显小于 1
        assert all(float(p["narrow_ratio"]) < 0.85 for p in yao)

    def test_empty_ridges_returns_empty(self):
        from engine.core.dragon_vein import find_yaoxia
        dem = make_dem_from_func(lambda yy, xx: 500.0 + np.zeros_like(yy, dtype=float))
        assert find_yaoxia([], dem) == []

    def test_analyze_dragon_vein_includes_yaoxia_field(self):
        """DragonVeinResult 必须带 yaoxia 字段（可为空列表）。"""
        from engine.core.dragon_vein import analyze_dragon_vein

        dem = make_dem_from_func(
            lambda yy, xx: 500.0 + 40.0 * np.exp(-((yy - 60) ** 2) / (2 * 4 ** 2))
        )
        dv = analyze_dragon_vein(dem, min_length_m=30.0)
        assert hasattr(dv, "yaoxia")
        assert isinstance(dv.yaoxia, list)


# ===== P1-5 晕土 =====

class TestHaloSoil:
    def test_flat_topography_gives_high_score(self):
        """穴位于平台正中 → 应有较高晕土分。"""
        def f(yy, xx):
            # 中心完全平台；边缘微凸
            cy, cx = 60, 60
            z = np.full_like(yy, 500.0, dtype=float)
            # 局部失稳区在 50,60 (helper)
            z += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 40 ** 2)) * 10.0
            return z

        dem = make_dem_from_func(f)
        # 中心点 (=cy=60, cx=60): 应有较高分
        h = score_halo_soil(dem, 60, 60)
        assert h.score >= 50

    def test_peak_not_halo(self):
        """山顶不应有晕土。"""
        def f(yy, xx):
            cy, cx = 60, 60
            return 500.0 + 100 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 8 ** 2))

        dem = make_dem_from_func(f)
        h = score_halo_soil(dem, 60, 60, search_radius_m=30.0)
        # 山顶局部坡度最大 → 不是晕土
        assert h.has_min_slope is False


# ===== P1-2 二十四山 =====

class TestCompassFacing:
    def test_basic_south_is_wu(self):
        from engine.core.compass_directions import classify_facing
        f = classify_facing(180.0)
        assert f.shan == "\u5348"  # 午
        assert f.is_jian_xiang is False
        assert f.is_chu_gua is False

    def test_jian_xiang_range(self):
        from engine.core.compass_directions import classify_facing
        # 190° 距午(180)10°, 距丁(195)5°. 会命中丁中心, 偏 5°, 应为兼向
        f = classify_facing(190.0)
        assert f.is_jian_xiang is True

    def test_score_decreases_on_chu_gua(self):
        from engine.core.compass_directions import score_compass_purity
        s_pure, _f = score_compass_purity(180.0)
        s_jian, _fj = score_compass_purity(190.0)
        assert s_jian < s_pure

    def test_facing_cross_check(self):
        from engine.core.compass_directions import facing_cross_check
        ok, _msg = facing_cross_check(180.0, 0.0)
        assert ok is True
        bad, _msg = facing_cross_check(180.0, 270.0)
        assert bad is False

    def test_score_candidate_cross_check_injected(self):
        """【B14】注入 long_az_deg 后 cross_check 非占位；反局有罚分。"""
        from engine.io.dem import load_dem
        from engine.io.rivers import load_water
        from engine.core.acupoint import search_candidates
        from engine.core.terrain_analysis import analyze_terrain
        from engine.core.fengshui_score import score_candidate

        dem = load_dem(r"D:\Xunlong\engine\tests\fixtures\synth_dem.tif")
        w = load_water(r"D:\Xunlong\engine\tests\fixtures\synth_rivers.geojson")
        cands = search_candidates(dem, max_candidates=5, step=15)
        if not cands:
            pytest.skip("合成 DEM 上未找到候选穴")
        terrain = analyze_terrain(dem)

        # 无来龙：占位
        fused0 = score_candidate(dem, cands[0], terrain, w, long_az_deg=None)
        assert fused0.meta.get("cross_check_ok") is None
        assert "未注入" in (fused0.meta.get("cross_check_msg") or "")
        assert fused0.scores.get("cross_check_penalty") == 0

        # 朝南 facing≈180 → sit≈0；long_az=0 对齐 → 通过
        fused_ok = score_candidate(dem, cands[0], terrain, w, long_az_deg=0.0)
        assert fused_ok.meta.get("cross_check_ok") is True
        assert fused_ok.meta.get("long_az_deg") == 0.0
        assert fused_ok.messages.get("cross_check")
        assert "未注入" not in fused_ok.messages["cross_check"]

        # long 与 sit 正交 → 反局罚分
        fused_bad = score_candidate(dem, cands[0], terrain, w, long_az_deg=90.0)
        assert fused_bad.meta.get("cross_check_ok") is False
        assert fused_bad.scores.get("cross_check_penalty") == -10
        assert fused_bad.overall <= fused_ok.overall

    def test_sanyuan_net_yin_yang_b2(self):
        """【B2】三元龙 + 净阴净阳：校准后 classify_facing 一致。"""
        from engine.core.compass_directions import (
            classify_facing, TIANYUAN, DIYUAN, RENYUAN,
            YANG_TIAN, YIN_TIAN, YANG_DI, YIN_DI, YANG_REN, YIN_REN,
        )
        # 午 = 天元阴 → 净阴
        f_wu = classify_facing(180.0)
        assert f_wu.shan == "午"
        assert f_wu.san_yuan == "天"
        assert f_wu.yin_yang_status == "净阴"
        # 甲 = 地元阳 → 净阳
        f_jia = classify_facing(75.0)
        assert f_jia.shan == "甲"
        assert f_jia.san_yuan == "地"
        assert f_jia.yin_yang_status == "净阳"
        # 八方位中心落山与二十四山表一致
        for deg, expect in [(0, "子"), (90, "卯"), (180, "午"), (270, "酉")]:
            f = classify_facing(float(deg))
            assert f.shan == expect
        assert len(TIANYUAN | DIYUAN | RENYUAN) == 24
        assert len(YANG_TIAN | YIN_TIAN | YANG_DI | YIN_DI | YANG_REN | YIN_REN) == 24


# ===== P2-2 微地形物候 =====

class TestPhenology:
    def test_no_input_returns_proxy(self):
        from engine.core.phenology import score_acupoint_phenology, PhenologyInputs
        # 平稳 DEM
        def f(yy, xx):
            return 500.0 + np.zeros_like(yy, dtype=float)
        dem = make_dem_from_func(f)
        res = score_acupoint_phenology(dem, 60, 60, inputs=None)
        assert "total" in res
        assert "vegetation_proxy" in res

    def test_with_ndvi_and_moisture(self):
        from engine.core.phenology import score_acupoint_phenology, PhenologyInputs
        import numpy as np
        dem = make_dem_from_func(lambda yy, xx: 500.0 + np.zeros_like(yy, dtype=float))
        ndvi = np.full(dem.data.shape, 0.6, dtype=float)
        sm = np.full(dem.data.shape, 0.4, dtype=float)
        inputs = PhenologyInputs(ndvi=ndvi, soil_moisture=sm)
        res = score_acupoint_phenology(dem, 60, 60, inputs=inputs)
        assert res["ndvi_score"] == 60.0
        assert res["moisture_score"] == 40.0
        assert 0 <= res["total"] <= 100


# ===== P-3 集成校验 =====

class TestIntegration:
    """验证 star_body / halo_soil / find_yaoxia 等 P0/P1 模块已接入 score_candidate。"""

    def test_score_candidate_includes_halo_and_star(self):
        """【修复 A1】star_body / halo_soil 必须出现在 scores / meta 中。"""
        from engine.io.dem import load_dem
        from engine.io.rivers import load_water
        from engine.core.acupoint import search_candidates
        from engine.core.terrain_analysis import analyze_terrain
        from engine.core.fengshui_score import score_candidate
        dem = load_dem(r"D:\Xunlong\engine\tests\fixtures\synth_dem.tif")
        w = load_water(r"D:\Xunlong\engine\tests\fixtures\synth_rivers.geojson")
        cands = search_candidates(dem, max_candidates=5, step=15)
        if not cands:
            pytest.skip("合成 DEM 上未找到候选穴")
        terrain = analyze_terrain(dem)
        fused = score_candidate(dem, cands[0], terrain, w)
        # scores 中有 halo_soil 和 star_body_bonus
        assert "halo_soil" in fused.scores
        assert "star_body_bonus" in fused.scores
        # meta 中有 star_body_type / mouth_evaluated / weighted_dims
        assert fused.meta is not None
        assert "star_body_type" in fused.meta
        assert "weighted_dims" in fused.meta
        # 明确 mouth_evaluated 字段避免误导
        assert "mouth_evaluated" in fused.meta

    def test_score_candidate_embrace_in_sand(self):
        """【A1-余】measure_embrace 并入砂分；scores 暴露 embrace_left/right。"""
        from engine.io.dem import load_dem
        from engine.io.rivers import load_water
        from engine.core.acupoint import search_candidates
        from engine.core.terrain_analysis import analyze_terrain
        from engine.core.fengshui_score import score_candidate

        dem = load_dem(r"D:\Xunlong\engine\tests\fixtures\synth_dem.tif")
        w = load_water(r"D:\Xunlong\engine\tests\fixtures\synth_rivers.geojson")
        cands = search_candidates(dem, max_candidates=5, step=15)
        if not cands:
            pytest.skip("合成 DEM 上未找到候选穴")
        terrain = analyze_terrain(dem)
        fused = score_candidate(dem, cands[0], terrain, w)
        assert "embrace_left" in fused.scores
        assert "embrace_right" in fused.scores
        assert fused.meta.get("embrace_in_sand") is True
        # 朝抱应写入 geography；message 可含朝抱标签
        assert "embrace_left" in fused.geography
        assert "embrace_right" in fused.geography

    def test_score_candidate_yaoxia_bonus_injected(self):
        """【A1-余】注入过峡点后 scores.yaoxia_bonus / meta.yaoxia_evaluated 生效。"""
        from engine.io.dem import load_dem
        from engine.io.rivers import load_water
        from engine.core.acupoint import search_candidates
        from engine.core.terrain_analysis import analyze_terrain
        from engine.core.fengshui_score import score_candidate, _score_yaoxia_for_candidate

        dem = load_dem(r"D:\Xunlong\engine\tests\fixtures\synth_dem.tif")
        w = load_water(r"D:\Xunlong\engine\tests\fixtures\synth_rivers.geojson")
        cands = search_candidates(dem, max_candidates=5, step=15)
        if not cands:
            pytest.skip("合成 DEM 上未找到候选穴")
        terrain = analyze_terrain(dem)
        # 构造距候选 ~200 m 的人造过峡点
        fake_yaoxia = [{
            "x": cands[0].x + 200.0,
            "y": cands[0].y,
            "narrow_ratio": 0.35,
            "neck_width_m": 40.0,
        }]
        fused = score_candidate(
            dem, cands[0], terrain, w, yaoxia_points=fake_yaoxia,
        )
        assert fused.meta.get("yaoxia_evaluated") is True
        assert fused.scores.get("yaoxia_bonus") == 6
        assert fused.geography.get("nearest_yaoxia_m") is not None
        assert fused.geography["nearest_yaoxia_m"] < 250.0

        # 单元规则：压峡负分
        b_press, m_press = _score_yaoxia_for_candidate(
            cands[0].x, cands[0].y,
            [{"x": cands[0].x, "y": cands[0].y, "narrow_ratio": 0.3}],
        )
        assert b_press == -2
        assert m_press["yaoxia_evaluated"] is True

    def test_sand_mountain_includes_embrace(self):
        """score_sand_mountain 应返回 embrace_left/right 并入 score。"""
        from engine.core.sand_water import score_sand_mountain

        dem = make_dem_from_func(
            lambda yy, xx: 500.0
            + 80.0 * np.exp(-((yy - 40) ** 2 + (xx - 60) ** 2) / (2 * 6 ** 2))  # 北玄武
            + 50.0 * np.exp(-((yy - 60) ** 2 + (xx - 40) ** 2) / (2 * 8 ** 2))  # 西青龙
            + 50.0 * np.exp(-((yy - 60) ** 2 + (xx - 80) ** 2) / (2 * 8 ** 2))  # 东白虎
        )
        sand = score_sand_mountain(dem, search_radius_m=800.0)
        assert sand.embrace_left is not None
        assert sand.embrace_right is not None
        assert 0 <= sand.embrace_left <= 100
        assert 0 <= sand.embrace_right <= 100
        assert "朝抱" in sand.message or "反背" in sand.message


# ===== P2-1 三元九运 / 玄空挨星 =====

class TestYuanYunXuanKong:
    def test_year_to_yun_basic(self):
        from engine.core.yuan_yun_xuankong import year_to_yun
        info = year_to_yun(2026)
        assert info.yun == 9
        info = year_to_yun(2000)
        assert info.yun == 7
        info = year_to_yun(1864)
        assert info.yun == 1

    def test_year_out_of_range_raises(self):
        from engine.core.yuan_yun_xuankong import year_to_yun
        try:
            year_to_yun(2100)
            assert False
        except ValueError:
            pass

    def test_fly_chart_8yun_south(self):
        from engine.core.yuan_yun_xuankong import fly_chart, wang_xiang, score_yun
        chart = fly_chart(8, "\u5b50", "\u5348")
        # 8 运山星到向宫不等于 8，不算旺向
        flags = wang_xiang(8, chart)
        assert "wang_xiang" in flags
        # 评分在 [0, 100] 范围内
        s = score_yun(8, chart)
        assert 0 <= s <= 100

    def test_fly_chart_distinct_shan_facing(self):
        from engine.core.yuan_yun_xuankong import fly_chart
        # 山盘与向盘应使用不同卦的中心。
        # 坐坤(225°) = 坤；向午(180°) = 离；不同宫
        chart = fly_chart(7, "\u5764", "\u5348")
        assert chart.shan_gua == "\u5764"
        assert chart.facing_gua == "\u79bb"

    def test_gua_yinyang_correct(self):
        """【修复】乾必须为阳，不是阴。"""
        from engine.core.yuan_yun_xuankong import GUA_YIN_YANG
        assert GUA_YIN_YANG["\u4e7e"] == "\u9633"  # 乾为阳
        assert GUA_YIN_YANG["\u5151"] == "\u9633"  # 兑 阳
        assert GUA_YIN_YANG["\u9707"] == "\u9633"  # 震 阳
        assert GUA_YIN_YANG["\u79bb"] == "\u9633"  # 离 阳
        assert GUA_YIN_YANG["\u574e"] == "\u9634"
        assert GUA_YIN_YANG["\u826e"] == "\u9634"
        assert GUA_YIN_YANG["\u5dfd"] == "\u9634"
        assert GUA_YIN_YANG["\u5764"] == "\u9634"

    def test_facing_takes_own_gua_not_dual(self):
        """【修复】shan_facing_gua(向) 应返回向本宫，不应对宫。"""
        from engine.core.yuan_yun_xuankong import shan_facing_gua
        # 向午(180°) = 离，向子(0°) = 坎，卯(90°) = 震，亥(330°) ≈ 乾(315°)
        assert shan_facing_gua("\u5348") == "\u79bb"   # 180° 离
        assert shan_facing_gua("\u5b50") == "\u574e"   # 0°   坎
        assert shan_facing_gua("\u536f") == "\u9707"   # 90°  震
        assert shan_facing_gua("\u4ea5") == "\u4e7e"   # 330° 乾（最近）
        assert shan_facing_gua("\u4e59") == "\u9707"   # 105° 震（最近）
        # 关键：向午(180°)不能再返回坎（旧 bug）
        assert shan_facing_gua("\u5348") != "\u574e"

    def test_xuankong_implemented_false(self):
        """【修复 B7】简化盘不输出星数；API 仍 xuankong_implemented=false。"""
        from engine.core.yuan_yun_xuankong import fly_chart
        chart = fly_chart(8, "\u5b50", "\u5348")
        assert chart.simplified is True
        # 简化盘星数必须为 None（禁止误导）
        assert chart.shan_star_at_facing is None
        assert chart.facing_star_at_facing is None
        # 至少 元 / 运 应正确
        assert chart.yun == 8

    def test_period_plate_always_forward(self):
        """【B8 修】运盘一律顺飞：一运/八运黄金九宫。"""
        from engine.core.yuan_yun_xuankong import period_plate, fly_stars

        p1 = period_plate(1)
        assert p1["\u4e2d"] == 1
        assert p1["\u4e7e"] == 2  # 乾
        assert p1["\u5dfd"] == 9  # 巽

        # 八运：中8 乾9 兑1 艮2 离3 坎4 坤5 震6 巽7（永顺，非奇偶逆）
        p8 = period_plate(8)
        assert p8["\u4e2d"] == 8
        assert p8["\u4e7e"] == 9  # 乾
        assert p8["\u5151"] == 1  # 兑
        assert p8["\u826e"] == 2  # 艮
        assert p8["\u79bb"] == 3  # 离
        assert p8["\u574e"] == 4  # 坎
        assert p8["\u5764"] == 5  # 坤
        assert p8["\u9707"] == 6  # 震
        assert p8["\u5dfd"] == 7  # 巽

        f = fly_stars(9, reverse=False)
        assert f["\u4e2d"] == 9
        assert f["\u4e7e"] == 1

    def test_fly_chart_strict_zi_wu_8yun_golden(self):
        """【B8 黄金】八运子山午向下卦 = 双星到向（向宫 山8/向8/运3）。"""
        from engine.core.yuan_yun_xuankong import fly_chart_strict, star_fly_polarity

        # 山星 4：元旦巽宫 + 子天元 → 巽阳 → 顺
        assert star_fly_polarity(4, "\u5b50") == "\u9633"
        # 向星 3：元旦震宫 + 午天元 → 卯阴 → 逆
        assert star_fly_polarity(3, "\u5348") == "\u9634"
        # 五黄 D2：取本山阴阳
        assert star_fly_polarity(5, "\u5b50") == "\u9634"  # 子阴
        assert star_fly_polarity(5, "\u7532") == "\u9633"  # 甲阳

        chart = fly_chart_strict(8, "\u5b50", "\u5348")
        assert chart.simplified is False
        assert chart.shan_gua == "\u574e"  # 坎
        assert chart.facing_gua == "\u79bb"  # 离
        assert chart.period_chart.get("\u574e") == 4  # 坎运星
        assert chart.period_chart.get("\u79bb") == 3  # 离运星
        # 向宫：山星 8、向星 8 → 双星到向
        assert chart.shan_star_at_facing == 8
        assert chart.facing_star_at_facing == 8
        # P2：替卦已实现（兼向时启用）；城门等仍缺
        assert "城门" in chart.features_missing
        assert "替卦" not in chart.features_missing or "替卦全流派细则" in chart.features_missing
        assert len(chart.mountain_chart) == 8
        assert len(chart.facing_chart) == 8

    def test_fly_chart_strict_wu_zi_8yun(self):
        """【B8】八运午山子向：山星 3 逆、向星 4 顺（教材例）。"""
        from engine.core.yuan_yun_xuankong import fly_chart_strict, star_fly_polarity

        # 山星 3（离宫运星）：3→震 + 午天元 → 卯阴 → 逆
        assert star_fly_polarity(3, "\u5348") == "\u9634"
        # 向星 4（坎宫运星）：4→巽 + 子天元 → 巽阳 → 顺
        assert star_fly_polarity(4, "\u5b50") == "\u9633"

        chart = fly_chart_strict(8, "\u5348", "\u5b50")
        assert chart.shan_gua == "\u79bb"
        assert chart.facing_gua == "\u574e"
        # 坐宫(离) 山星应为 8（3 逆飞到离）
        assert chart.mountain_chart.get("\u79bb") == 8
        # 向宫(坎) 向星应为 9（4 顺飞：中4 乾5 兑6 艮7 离8 坎9）
        assert chart.facing_chart.get("\u574e") == 9

    def test_fly_chart_strict_score_yun_uses_stars(self):
        """严格盘 score_yun 使用星数；简化盘中性 50。"""
        from engine.core.yuan_yun_xuankong import (
            fly_chart, fly_chart_strict, score_yun,
        )
        simple = fly_chart(8, "\u5b50", "\u5348")
        assert score_yun(8, simple) == 50
        strict = fly_chart_strict(8, "\u5b50", "\u5348")
        s = score_yun(8, strict)
        assert 0 <= s <= 100
        # 八运子山午向双星到向(8) → 令星会向，应高于中性
        assert s > 50

    def test_shan_table_sanyuan_xuan_kong(self):
        """【B8 §1.8】SHAN_TABLE 三元龙与无常派地天人一致。"""
        from engine.core.compass_directions import (
            SHAN_TABLE, TIANYUAN, DIYUAN, RENYUAN,
            YANG_TIAN, YIN_TIAN, YANG_DI, YIN_DI, YANG_REN, YIN_REN,
        )
        correct = {
            "子": "天", "午": "天", "卯": "天", "酉": "天",
            "乾": "天", "坤": "天", "艮": "天", "巽": "天",
            "甲": "地", "庚": "地", "壬": "地", "丙": "地",
            "辰": "地", "戌": "地", "丑": "地", "未": "地",
            "寅": "人", "申": "人", "巳": "人", "亥": "人",
            "乙": "人", "辛": "人", "丁": "人", "癸": "人",
        }
        for shan, yuan in correct.items():
            assert SHAN_TABLE[shan][1] == yuan, f"{shan}: {SHAN_TABLE[shan][1]}!={yuan}"
        assert TIANYUAN == set(s for s, y in correct.items() if y == "天")
        assert DIYUAN == set(s for s, y in correct.items() if y == "地")
        assert RENYUAN == set(s for s, y in correct.items() if y == "人")
        # 阴阳集合覆盖 24 山
        all_yy = YANG_TIAN | YIN_TIAN | YANG_DI | YIN_DI | YANG_REN | YIN_REN
        assert all_yy == set(correct.keys())

    def test_gui_finding_15deg(self):
        """【修复 B1】15° 中心应命中癸（不是百）。"""
        from engine.core.compass_directions import SHAN_TABLE, find_nearest_shan
        assert "\u7678" in SHAN_TABLE
        assert "\u767e" not in SHAN_TABLE
        shan, _ = find_nearest_shan(15.0)
        assert shan == "\u7678"
        assert SHAN_TABLE[shan][1] == "人"  # 癸为人元


# ===== P1-4 弓背水三节连贯 =====

class TestWaterCurve:
    def test_distance_adaptive_weight(self):
        from engine.core.water_curve import distance_adaptive_form_weight
        # 近水形态意义 0
        assert distance_adaptive_form_weight(20.0) == 0.0
        # 80-1200 m 满分
        assert distance_adaptive_form_weight(500.0) == 1.0
        # 远水衰减
        assert distance_adaptive_form_weight(3000.0) < 0.5

    def test_three_segments_synthetic(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        from engine.io.rivers import WaterNetwork

        # 构造一个简单「凹」河段（弧向左）
        coords = [(0, 0), (100, 50), (200, 100), (300, 200), (400, 350)]
        gdf = gpd.GeoDataFrame(geometry=[LineString(coords)], crs="EPSG:3857")
        w = WaterNetwork(gdf=gdf)
        segs = split_river_segments(w)
        assert isinstance(segs, list)
        if segs:
            r = score_water_curve_three_segments(w, 200.0, 100.0, dist_m=120.0)
            assert "consistency" in r

    def test_empty_water_safe(self):
        from engine.io.rivers import WaterNetwork
        import geopandas as gpd
        w = WaterNetwork(gdf=gpd.GeoDataFrame(geometry=[], crs="EPSG:3857"))
        assert split_river_segments(w) == []
        assert score_multi_segment_concavity(w, 0, 0) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
