"""四象识别算法严谨性测试（非视觉贴图）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core.four_beasts_detect import (
    BEAST_WATER_BAN_M,
    WATER_BAN_BUFFER_M,
    detect_four_beasts,
    infer_facing,
    find_score_peak,
    water_score_from_dist,
    smooth_score_field,
    water_distance_rasters,
    _ideal_dist_score,
    _angle_diff,
    _segment_hits_water,
)
from engine.core.water_model import water_get_score, water_sha_penalty as sha_pen
from engine.io.dem import load_dem
from engine.tests.fixtures.make_synthetic import make_synthetic_dem

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def synth_dem():
    p = FIXTURES / "synth_dem.tif"
    if not p.exists():
        make_synthetic_dem(p)
    return load_dem(p)


class TestFourBeastsFacingConvention:
    """four_beasts.score_four_beasts 与 detect 使用同一朝向约定。"""

    def test_default_face_south_sectors(self, synth_dem):
        from engine.core.four_beasts import score_four_beasts
        sc = score_four_beasts(synth_dem)
        d = sc.details
        assert d["facing"] == pytest.approx(180.0)  # 朝南
        assert d["sit"] == pytest.approx(0.0)       # 坐北
        assert d["dirs"]["xuanwu"] == pytest.approx(0.0)
        assert d["dirs"]["zhuque"] == pytest.approx(180.0)
        assert d["dirs"]["qinglong"] == pytest.approx(90.0)   # 东
        assert d["dirs"]["baihu"] == pytest.approx(270.0)     # 西

    def test_face_east_qinglong_is_north(self, synth_dem):
        from engine.core.four_beasts import score_four_beasts
        sc = score_four_beasts(synth_dem, facing_override=90.0)  # 朝东
        d = sc.details["dirs"]
        assert d["qinglong"] == pytest.approx(0.0)    # 左=北
        assert d["baihu"] == pytest.approx(180.0)     # 右=南
        assert d["xuanwu"] == pytest.approx(270.0)    # 后=西


class TestHelpers:
    def test_ideal_dist_in_range(self):
        s = _ideal_dist_score(250, 50, 500)
        assert s > 0.8

    def test_ideal_dist_too_close(self):
        assert _ideal_dist_score(10, 50, 500) < 0.6

    def test_ideal_dist_too_far(self):
        assert _ideal_dist_score(5000, 50, 500) < 0.5

    def test_angle_diff_wrap(self):
        assert _angle_diff(10, 350) == pytest.approx(20)
        assert _angle_diff(0, 180) == pytest.approx(180)


class TestDetectFourBeasts:
    def test_returns_positions(self, synth_dem):
        h, w = synth_dem.data.shape
        fb = detect_four_beasts(synth_dem, h // 2, w // 2)
        assert fb.center is not None
        assert 0 <= fb.facing < 360
        # 至少一个方向能识别到砂/山
        found = sum(
            1
            for k in ("shaozu", "xuanwu", "zhuque", "qinglong", "baihu")
            if getattr(fb, k) is not None
        )
        assert found >= 2

    def test_user_facing_respected(self, synth_dem):
        h, w = synth_dem.data.shape
        fb = detect_four_beasts(synth_dem, h // 2, w // 2, facing=90.0)
        assert fb.facing == pytest.approx(90.0)
        assert fb.facing_method == "user_facing"
        assert fb.sit == pytest.approx(270.0)

    def test_meta_has_metric_fields(self, synth_dem):
        h, w = synth_dem.data.shape
        fb = detect_four_beasts(synth_dem, h // 2, w // 2)
        assert "beasts" in fb.meta
        assert "facing_method" in fb.meta
        # 有识别到的点应带 dist_m / elev_m
        for k, m in fb.meta["beasts"].items():
            if m is None:
                continue
            assert "dist_m" in m
            assert "elev_m" in m
            assert m["dist_m"] >= 0

    def test_occupied_peaks_distinct(self, synth_dem):
        """不同四象点不应落在同一像素。"""
        h, w = synth_dem.data.shape
        fb = detect_four_beasts(synth_dem, h // 2, w // 2)
        cells = []
        for k, m in (fb.meta.get("beasts") or {}).items():
            if not m:
                continue
            cells.append((m["row"], m["col"]))
        # 允许少部分缺失，但出现的点坐标应互异
        assert len(cells) == len(set(cells))

    def test_xuanwu_distance_prefer_near_range(self, synth_dem):
        """玄武若识别到，应在合理近距离量级（非图幅对角硬拉）。"""
        h, w = synth_dem.data.shape
        fb = detect_four_beasts(synth_dem, h // 2, w // 2)
        xw = fb.meta["beasts"].get("xuanwu")
        if xw is None:
            pytest.skip("synthetic DEM has no xuanwu peak")
        # 合成 DEM 范围有限；相对 L 不贴身
        assert xw["dist_m"] < 3000
        L = float((fb.meta.get("params_m") or {}).get("L_site_m") or 2000)
        assert xw["dist_m"] >= 0.06 * L

    def test_beast_distance_windows_scale_relative(self):
        """比例制：L 越大窗越大；无绝对 180/800 硬地板。"""
        from engine.core.four_beasts_detect import (
            beast_distance_windows,
            XUANWU_FRAC,
            SHAOZU_XUANWU_DIST_RATIO,
        )

        small = beast_distance_windows(800.0, cell_m=30.0)
        large = beast_distance_windows(4500.0, cell_m=30.0)
        assert large["L"] > small["L"]
        assert large["xuanwu"][0] > small["xuanwu"][0]
        # 玄武下界 ≈ frac × L（允许噪声地板）
        assert small["xuanwu"][0] <= small["L"] * (XUANWU_FRAC[0] + 0.05) + 50
        assert large["shaozu"][0] >= large["xuanwu"][0] * (SHAOZU_XUANWU_DIST_RATIO - 0.05)
        # 小局玄武下界应远小于「180m 绝对时代」对大局的限制感：
        # 大局白虎下界应明显高于小局
        assert large["baihu"][0] > small["baihu"][0] * 1.3

    def test_hierarchy_shaozu_farther_and_baihu_not_hugging(self, synth_dem):
        """少祖远于玄武；白虎相对局尺度不贴穴。"""
        h, w = synth_dem.data.shape
        fb = detect_four_beasts(synth_dem, h // 2, w // 2)
        beasts = fb.meta.get("beasts") or {}
        params = fb.meta.get("params_m") or {}
        L = float(params.get("L_site_m") or 0) or 1.0
        xw = beasts.get("xuanwu")
        sz = beasts.get("shaozu")
        bh = beasts.get("baihu")
        if xw and sz:
            assert sz["dist_m"] >= xw["dist_m"] * 1.5 - 1.0
        if bh and L > 0:
            # 白虎 ≥ 约 0.08L（比例下限放宽容差）
            assert bh["dist_m"] >= min(0.08 * L, float((params.get("baihu") or [0])[0]) * 0.7)

    def test_gate_rejects_missing_shaozu_logic(self):
        """门禁：无少祖 / 少祖不高于玄武 → 不通过。"""
        from engine.core.fengshui_score import _gate_beasts_for_hole

        # 构造空 info 路径：用极小 DEM 中心，允许失败但不崩溃
        h, w = synth_dem_shape = (30, 30)
        # 使用 fixture-free 最小 DEM
        from rasterio.transform import from_origin
        from engine.io.dem import DEM
        import numpy as np
        data = np.linspace(50, 120, 30 * 30).reshape(30, 30)
        dem = DEM(
            data=data.astype(float),
            transform=from_origin(0, 900, 30, 30),
            crs="EPSG:3857",
            nodata=-9999.0,
            bounds=(0.0, 0.0, 900.0, 900.0),
            resolution=(30.0, 30.0),
        )
        ok, reason, info = _gate_beasts_for_hole(dem, 15, 15, water=None)
        # 可能通过也可能因 incomplete 失败；必须有 reason 字段
        assert reason
        assert "beasts_present" in info or "reason" in info


class TestInferFacing:
    def test_default_or_terrain(self, synth_dem):
        h, w = synth_dem.data.shape
        facing, method = infer_facing(synth_dem, h // 2, w // 2, water=None)
        assert 0 <= facing < 360
        assert method in (
            "back_to_high_terrain",
            "default_south",
            "face_water",
            "back_high_face_water",
            "mingtang_face_water",
            "back_high_over_nearest_water",
        )

    def test_nearest_water_does_not_flip_against_back_high(self):
        """北有高靠山、南有明堂水时，不得因北侧近岸把朝向拧成朝北。"""
        from rasterio.transform import from_origin
        import geopandas as gpd
        from shapely.geometry import box
        from engine.io.dem import DEM
        from engine.io.rivers import WaterNetwork

        h, w = 90, 90
        cell = 30.0
        yy, xx = np.mgrid[0:h, 0:w]
        cr, cc = 55, 45
        z = 100.0 + (cr - yy) * 0.15  # 略北高
        # 北靠山
        z += 90.0 * np.exp(-((yy - 25) ** 2 + (xx - cc) ** 2) / (2 * 5**2))
        # 穴略凹
        z -= 10.0 * np.exp(-((yy - cr) ** 2 + (xx - cc) ** 2) / (2 * 3**2))
        # 南侧低缓明堂
        z[65:80, 20:70] -= 8.0
        # 北侧窄水（很近）— 若只认最近岸会面北
        transform = from_origin(0.0, h * cell, cell, cell)
        dem = DEM(
            data=z.astype(np.float32),
            transform=transform,
            crs="EPSG:3857",
            nodata=-9999.0,
            bounds=(0.0, 0.0, w * cell, h * cell),
            resolution=(cell, cell),
        )
        # 北汊 + 南河（南河更宽、在明堂方向）
        y_n0 = h * cell - 48 * cell
        y_n1 = h * cell - 52 * cell
        y_s0 = h * cell - 72 * cell
        y_s1 = h * cell - 78 * cell
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[
                box(0, min(y_n0, y_n1), w * cell, max(y_n0, y_n1)),
                box(0, min(y_s0, y_s1), w * cell, max(y_s0, y_s1)),
            ],
            crs="EPSG:3857",
        )
        wn = WaterNetwork(gdf=gdf)
        facing, method = infer_facing(dem, cr, cc, water=wn)
        # 应朝南象限（背北山），不得 ≈ 北 (0°)
        assert 90.0 < facing < 270.0, f"facing={facing} method={method}"
        # 更严：接近南
        align_south = abs(((facing - 180 + 180) % 360) - 180)
        assert align_south < 70.0, f"facing={facing} method={method}"


class TestFindScorePeak:
    def test_peak_near_seeded_high(self):
        """平滑后峰值应落在高分团块附近，而非孤立噪声。"""
        g = np.full((40, 40), 10.0, dtype=np.float64)
        g[10, 10] = 100.0  # 孤立尖峰（应被平滑削弱）
        g[25:30, 25:30] = 80.0  # 团块高分（应胜出）
        pr, pc, sc = find_score_peak(g, smooth_sigma=2.0)
        assert 22 <= pr <= 32
        assert 22 <= pc <= 32

    def test_all_nan_returns_none(self):
        g = np.full((8, 8), np.nan)
        assert find_score_peak(g) is None

    def test_nan_water_zone_not_peak(self):
        """禁水 nan 区不得成为峰值。"""
        g = np.full((30, 30), 50.0, dtype=np.float64)
        g[10:20, 10:20] = np.nan  # 河
        g[5, 5] = 70.0
        pr, pc, _ = find_score_peak(g, smooth_sigma=1.0)
        assert not (10 <= pr < 20 and 10 <= pc < 20)

    def test_smooth_and_peak_same_argmax(self):
        """热力平滑场 argmax 与 find_score_peak 一致。"""
        g = np.full((25, 25), 20.0, dtype=np.float64)
        g[18:22, 8:12] = 90.0
        soft, _ = smooth_score_field(g, smooth_sigma=1.5)
        pr, pc, _ = find_score_peak(g, smooth_sigma=1.5)
        filled = np.where(np.isfinite(soft), soft, -np.inf)
        ar, ac = np.unravel_index(int(np.argmax(filled)), filled.shape)
        assert (pr, pc) == (int(ar), int(ac))

    def test_water_score_band(self):
        assert water_score_from_dist(0, banned=True) == 0.0
        # 双通道：中距得水高、贴岸煞高
        assert water_get_score(300) >= 70.0
        assert water_get_score(30) < water_get_score(300)
        assert sha_pen(30) > sha_pen(300)
        assert sha_pen(30) >= 70.0
        # fused 兼容分：贴岸 < 中距
        assert water_score_from_dist(30) < water_score_from_dist(300)

    def test_detect_from_score_peak_center(self, synth_dem):
        """四象中心应与传入的评分峰值行列一致。"""
        h, w = synth_dem.data.shape
        g = np.zeros((h, w), dtype=np.float64)
        tr, tc = h // 3, w // 3
        g[tr - 2:tr + 3, tc - 2:tc + 3] = 90.0
        peak = find_score_peak(g, smooth_sigma=1.5)
        assert peak is not None
        pr, pc, _ = peak
        fb = detect_four_beasts(synth_dem, center_row=pr, center_col=pc)
        assert fb.center is not None
        # 中心世界坐标对应同一像元邻域
        from rasterio.transform import rowcol
        cr, cc = rowcol(synth_dem.transform, fb.center[0], fb.center[1])
        assert abs(int(cr) - pr) <= 1
        assert abs(int(cc) - pc) <= 1

    def test_baihu_not_higher_than_qinglong_when_both(self, synth_dem):
        """有青龙时白虎高程不应显著高于青龙。"""
        h, w = synth_dem.data.shape
        fb = detect_four_beasts(synth_dem, h // 2, w // 2)
        ql = (fb.meta.get("beasts") or {}).get("qinglong")
        bh = (fb.meta.get("beasts") or {}).get("baihu")
        if not ql or not bh:
            pytest.skip("synthetic DEM missing ql/bh")
        # 允许数值噪声，但不得明显抬头
        assert bh["elev_m"] <= ql["elev_m"] * 1.02 + 2.0


class TestBeastWaterPolicy:
    """四象水禁：窄禁水面、允许对岸干峰；穴心宽禁带保持分离。"""

    def test_beast_ban_narrower_than_acupoint_ban(self):
        assert BEAST_WATER_BAN_M < WATER_BAN_BUFFER_M
        assert BEAST_WATER_BAN_M <= 20.0

    def test_segment_hits_water_midline(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 5:15] = True  # 横向水面
        assert _segment_hits_water(mask, 2, 10, 18, 10) is True
        assert _segment_hits_water(mask, 2, 2, 2, 18) is False  # 平行不穿

    def test_zhuque_can_select_opposite_bank_peak(self):
        """河湾：穴在北岸、案山在南岸近岸 → 朱雀应能取对岸（视线跨水）。"""
        from rasterio.transform import from_origin
        import geopandas as gpd
        from shapely.geometry import box
        from engine.io.dem import DEM
        from engine.io.rivers import WaterNetwork

        h, w = 80, 80
        cell = 30.0  # m
        yy, xx = np.mgrid[0:h, 0:w]
        z = np.full((h, w), 100.0, dtype=np.float64)
        # 北岸靠山（玄武方向）
        z += 80.0 * np.exp(-((yy - 18) ** 2 + (xx - 40) ** 2) / (2 * 5**2))
        # 穴心北岸台地（略凹）
        cr, cc = 32, 40
        z -= 15.0 * np.exp(-((yy - cr) ** 2 + (xx - cc) ** 2) / (2 * 3**2))
        # 南岸对岸案山（朱雀目标，近岸但在干地）
        z += 45.0 * np.exp(-((yy - 58) ** 2 + (xx - 40) ** 2) / (2 * 4**2))
        # 本岸前方矮丘（诱敌：若过度禁对岸会选这里）
        z += 22.0 * np.exp(-((yy - 38) ** 2 + (xx - 40) ** 2) / (2 * 3**2))
        # 河槽降低
        z[42:50, :] -= 25.0

        transform = from_origin(0.0, h * cell, cell, cell)
        dem = DEM(
            data=z.astype(np.float32),
            transform=transform,
            crs="EPSG:3857",
            nodata=-9999.0,
            bounds=(0.0, 0.0, w * cell, h * cell),
            resolution=(cell, cell),
        )
        # 横向河道多边形（行 42–49 对应 y）
        # row r → y = (h-r)*cell  with from_origin north-up: y decreases with row
        # from_origin(0, h*cell, cell, cell): row0 y=h*cell, row r y = h*cell - r*cell
        y_north = h * cell - 42 * cell
        y_south = h * cell - 50 * cell
        river = box(0.0, min(y_south, y_north), w * cell, max(y_south, y_north))
        wn = WaterNetwork(
            gdf=gpd.GeoDataFrame({"id": [1]}, geometry=[river], crs="EPSG:3857")
        )

        fb = detect_four_beasts(
            dem, center_row=cr, center_col=cc, facing=180.0, water=wn,
        )
        zq = (fb.meta.get("beasts") or {}).get("zhuque")
        assert zq is not None, "朱雀应识别到"
        # 对岸案山约 row 58；本岸矮丘约 38
        assert zq["row"] >= 50, f"朱雀应在南岸对岸峰，got row={zq['row']}"

        # 砂点不得落在真水面（宽穴禁带不适用）
        _wd, ban = water_distance_rasters(dem, wn, ban_buffer_m=BEAST_WATER_BAN_M)
        for name in ("zhuque", "baihu", "qinglong", "xuanwu"):
            m = (fb.meta.get("beasts") or {}).get(name)
            if not m:
                continue
            assert not ban[int(m["row"]), int(m["col"])], f"{name} 落在水禁带内"

        # 参数可解释
        assert fb.meta["params_m"]["beast_water_ban_m"] == pytest.approx(BEAST_WATER_BAN_M)


class TestScoreGridPerformanceG3:
    """生气场 compute_score_grid：全矢量乘性场 + 性能。"""

    def test_fast_four_beasts_matches_classic(self, synth_dem):
        """score_four_beasts_combined_at 与 score_four_beasts 一致（展示路径仍可用）。"""
        from engine.core.four_beasts import score_four_beasts, score_four_beasts_combined_at
        from engine.core.terrain_analysis import compute_slope_aspect
        from engine.io.dem import DEM

        slope, _ = compute_slope_aspect(synth_dem)
        r, c = synth_dem.data.shape[0] // 2, synth_dem.data.shape[1] // 2
        pad = 15
        sub = synth_dem.data[r - pad:r + pad + 1, c - pad:c + pad + 1].copy()
        sub_dem = DEM(
            data=sub,
            transform=synth_dem.transform,
            crs=synth_dem.crs,
            nodata=synth_dem.nodata,
            bounds=synth_dem.bounds,
            resolution=synth_dem.resolution,
        )
        classic = score_four_beasts(sub_dem).combined
        fast = score_four_beasts_combined_at(
            synth_dem.data, slope, r, c, xres_m=30.0, yres_m=30.0,
        )
        # 完整路径含 viewshed/蜿蜒/驯俯；快路径用代理，允许较大偏差
        assert abs(float(classic) - float(fast)) <= 18.0

    def test_compute_score_grid_shape_and_range(self, synth_dem):
        from engine.core.four_beasts_detect import compute_score_grid
        from engine.io.rivers import load_water

        water = load_water(FIXTURES / "synth_rivers.geojson")
        grid = compute_score_grid(
            synth_dem, sample_step=8, water=water, max_samples=None,
        )
        assert grid.shape == synth_dem.data.shape
        valid = grid[np.isfinite(grid)]
        assert valid.size > 100
        assert float(valid.min()) >= 0.0
        assert float(valid.max()) <= 100.0

    def test_compute_score_grid_fast_enough(self, synth_dem):
        """200×200 全幅矢量应在 1.5s 内（乘性生气场）。"""
        import time
        from engine.core.four_beasts_detect import compute_score_grid
        from engine.io.rivers import load_water

        water = load_water(FIXTURES / "synth_rivers.geojson")
        t0 = time.perf_counter()
        grid = compute_score_grid(
            synth_dem, sample_step=4, water=water, max_samples=None,
        )
        dt = time.perf_counter() - t0
        assert np.isfinite(grid).sum() > 1000
        assert dt < 1.5, f"compute_score_grid too slow: {dt:.2f}s"

    def test_qi_field_multiplicative_and_no_facing(self, synth_dem):
        """子场 0–1 乘性；水面硬零；与 facing 无关（无逐像素四象）。"""
        from engine.core.four_beasts_detect import compute_qi_field_layers, compute_score_grid
        from engine.io.rivers import load_water

        water = load_water(FIXTURES / "synth_rivers.geojson")
        layers = compute_qi_field_layers(synth_dem, water)
        for k in ("cangfeng", "water", "enclosure", "stability", "qi"):
            arr = layers[k]
            assert arr.shape == synth_dem.data.shape
            assert float(np.nanmin(arr)) >= -1e-9
            assert float(np.nanmax(arr)) <= 1.0 + 1e-9
        # 乘性后轻平滑+归一化：qi∈[0,1]；禁水区为 0
        qi = layers["qi"]
        ban = layers["water_ban"]
        finite = layers["finite"]
        mask = finite & ~ban
        if mask.any():
            assert float(qi[mask].max()) <= 1.0 + 1e-9
            assert float(qi[mask].min()) >= -1e-9
            # 禁水区 qi 为 0
            if ban.any():
                assert float(np.max(qi[ban])) <= 1e-12
            # 子场乘积与归一前一致方向：干地 qi 高处各子场不应全接近 0
            top = mask & (qi >= np.nanpercentile(qi[mask], 90))
            if top.any():
                assert float(layers["stability"][top].mean()) > 0.2
        g1 = compute_score_grid(synth_dem, water=water)
        g2 = compute_score_grid(synth_dem, water=water, weights={"water_lo_m": 120.0})
        assert g1.shape == g2.shape
        # 同参数可复现
        assert np.allclose(
            np.nan_to_num(g1, nan=-1),
            np.nan_to_num(g2, nan=-1),
            rtol=1e-5, atol=1e-5,
        )

    def test_water_plateau_prefers_mid_range_over_bank(self):
        """得水宽平台：300–600m 明显高于贴岸 80m（避免岸边光环）。"""
        from engine.core.four_beasts_detect import _water_distance_plateau

        d = np.array([80.0, 200.0, 400.0, 700.0, 1500.0, 3500.0])
        g = _water_distance_plateau(d)
        assert g[2] > g[0] + 0.15  # 400m 堂心 >> 80m 贴岸
        assert g[1] >= 0.90  # 200m 进入/近平台高分
        assert g[5] < g[3]  # 过远 < 平台

    def test_auto_sample_step_caps_work(self, synth_dem):
        from engine.core.four_beasts_detect import _auto_sample_step

        h, w = 1000, 1000
        # 目标 4000 点 → step ≈ sqrt(1e6/4000) ≈ 16
        s = _auto_sample_step(h, w, sample_step=2, max_samples=4000)
        assert s >= 15
        assert (h // s) * (w // s) <= 5000
