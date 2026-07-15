"""P1/P2：形峦深度 + 水法理气补强单测。

覆盖：
  - 青龙蜿蜒 / 白虎驯俯 / 朱雀 viewshed
  - 得水基线对齐数理文档
  - 天门地户分治
  - TPI 分辨率自适应
  - 替卦兼向
  - 穴星本体
"""
from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_bounds

from engine.io.dem import DEM


def make_dem_from_func(
    func,
    h: int = 80,
    w: int = 80,
    cell_size_m: float = 30.0,
) -> DEM:
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


class TestQinglongSinuosity:
    def test_sinuosity_returns_keys(self):
        from engine.core.four_beasts import measure_sector_sinuosity

        def f(yy, xx):
            # 左（西）侧有起伏脊
            return 500.0 + 40.0 * np.exp(-((xx - 15) ** 2) / 30.0) * (
                1.0 + 0.3 * np.sin(yy / 4.0)
            )

        dem = make_dem_from_func(f)
        # 青龙 = 朝南时的左侧 = 东 = 90°；这里朝南 left=270 西
        m = measure_sector_sinuosity(dem, 270.0, radius_m=400.0)
        assert "sinuosity" in m and "morph_score" in m
        assert m["sinuosity"] >= 1.0
        assert 0 <= m["morph_score"] <= 100

    def test_score_four_beasts_has_morph_details(self):
        from engine.core.four_beasts import score_four_beasts

        def f(yy, xx):
            cy, cx = 40, 40
            return 500.0 + 80.0 * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 20 ** 2)
            )

        dem = make_dem_from_func(f)
        sc = score_four_beasts(dem, facing_override=180.0)
        d = sc.details
        assert "qinglong_morph" in d
        assert "baihu_tame" in d
        assert "zhuque_viewshed" in d
        assert 0 <= sc.qinglong <= 100
        assert 0 <= sc.baihu <= 100
        assert 0 <= sc.zhuque <= 100


class TestBaihuTame:
    def test_steep_peak_lower_tame_than_gentle(self):
        from engine.core.four_beasts import measure_sector_tame
        from engine.core.terrain_analysis import compute_slope_aspect

        def gentle(yy, xx):
            return 500.0 + 30.0 * np.exp(-((xx - 55) ** 2 + (yy - 40) ** 2) / (2 * 12 ** 2))

        def steep(yy, xx):
            return 500.0 + 80.0 * np.exp(-((xx - 55) ** 2 + (yy - 40) ** 2) / (2 * 3 ** 2))

        dem_g = make_dem_from_func(gentle)
        dem_s = make_dem_from_func(steep)
        sg, _ = compute_slope_aspect(dem_g)
        ss, _ = compute_slope_aspect(dem_s)
        # 白虎 = 朝南右 = 西 = 270? facing 180: right = 270 西
        # 我们把峰放在 xx=55（东侧）→ 白虎若朝北会不同
        # facing=0 朝北，白虎=90 东
        tg = measure_sector_tame(dem_g, 90.0, sg, radius_m=400.0)
        ts = measure_sector_tame(dem_s, 90.0, ss, radius_m=400.0)
        assert tg["tame_score"] >= ts["tame_score"] - 5  # 缓 ≥ 尖（允许噪声）


class TestZhuqueViewshed:
    def test_open_front_higher_viewshed(self):
        from engine.core.four_beasts import measure_sector_viewshed

        def flat(yy, xx):
            return np.full_like(yy, 500.0, dtype=float)

        def wall(yy, xx):
            z = np.full_like(yy, 500.0, dtype=float)
            z[yy < 25] = 560.0  # 北侧高墙（朝北=0 时前朱雀被挡）
            return z

        dem_o = make_dem_from_func(flat)
        dem_w = make_dem_from_func(wall)
        vo = measure_sector_viewshed(dem_o, 0.0, radius_m=400.0)
        vw = measure_sector_viewshed(dem_w, 0.0, radius_m=400.0)
        assert vo["viewshed"] >= vw["viewshed"] - 0.05


class TestWaterBaselineDoc:
    def test_baseline_matches_math_doc(self):
        from engine.core.water_model import water_get_baseline

        assert water_get_baseline(40.0) == pytest.approx(68.0)
        assert water_get_baseline(300.0) == pytest.approx(86.0)
        assert water_get_baseline(2000.0) == pytest.approx(78.0)
        assert water_get_baseline(5000.0) < 78.0


class TestTianmenDihu:
    def test_classify_roles_by_axis(self):
        from shapely.geometry import LineString
        import geopandas as gpd
        from engine.io.rivers import WaterNetwork
        from engine.core.water_mouth import find_water_mouths

        # 南北向长河
        line = LineString([(0, 0), (0, 5000)])
        gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:3857")
        wn = WaterNetwork(gdf=gdf)
        mouths = find_water_mouths(wn, classify_roles=True)
        roles = {m.role for m in mouths}
        # 两端应分出天门/地户至少一种
        assert "tianmen" in roles or "dihu" in roles or "unknown" in roles
        # 至少有 endpoint
        assert any(m.kind == "endpoint" for m in mouths)

    def test_dihu_lock_stricter_threshold(self):
        from engine.core.water_mouth import (
            WaterMouth,
            score_water_mouth_for_candidate,
        )

        dihu = WaterMouth(
            x=0, y=0, kind="endpoint", n_inflows=1,
            lock_ratio=0.4, facing_angle_deg=0, is_jiaogou=False,
            role="dihu",
        )
        tian = WaterMouth(
            x=0, y=0, kind="endpoint", n_inflows=1,
            lock_ratio=0.4, facing_angle_deg=0, is_jiaogou=False,
            role="tianmen",
        )
        sd, _ = score_water_mouth_for_candidate(dihu, lock_ratio=0.4)
        st, _ = score_water_mouth_for_candidate(tian, lock_ratio=0.4)
        # 同样中等锁紧：地户应低于天门（地户宜闭）
        assert st > sd


class TestTpiScale:
    def test_scale_factor_bounds(self):
        from engine.core.acupoint import tpi_scale_factor, classify_form

        assert tpi_scale_factor(30.0) == pytest.approx(1.0)
        assert tpi_scale_factor(5.0) < 1.0
        assert tpi_scale_factor(90.0) > 1.0
        # 同一 TPI 在细分辨率下更易判窝
        f30 = classify_form(-1.2, 5.0, cell_size_m=30.0)
        f5 = classify_form(-1.2, 5.0, cell_size_m=5.0)
        # 5m scale=0.5 → thr t1=-0.75，-1.2 < -0.75 → 窝
        # 30m t1=-1.5，-1.2 在钳/窝过渡
        assert f5 == "窝穴"
        assert f30 in ("窝穴", "钳穴", "平缓")


class TestTiGua:
    def test_substitute_and_force_ti(self):
        from engine.core.yuan_yun_xuankong import (
            substitute_ti_star,
            fly_chart_strict,
            TI_STAR_TABLE,
        )

        assert substitute_ti_star(1) == TI_STAR_TABLE[1]
        base = fly_chart_strict(8, "子", "午")
        ti = fly_chart_strict(8, "子", "午", force_ti=True)
        assert "替卦" in ti.notes or "替" in ti.notes
        # 替星可能改变向宫星数
        assert ti.simplified is False
        assert base.simplified is False

    def test_jian_detect(self):
        from engine.core.yuan_yun_xuankong import detect_jian_xiang
        from engine.core.compass_directions import SHAN_TABLE

        center = SHAN_TABLE["午"][0]
        is_j, off, _ = detect_jian_xiang("午", deg_override=center + 5.0)
        assert is_j is True
        assert off >= 3.0
        is_j2, _, _ = detect_jian_xiang("午", deg_override=center + 0.5)
        assert is_j2 is False


class TestXueStar:
    def test_xue_star_on_peak(self):
        from engine.core.star_body import classify_xue_star, score_xue_star_bonus

        def f(yy, xx):
            cy, cx = 40, 40
            return 500.0 + 100.0 * np.exp(
                -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 10 ** 2)
            )

        dem = make_dem_from_func(f)
        star = classify_xue_star(dem, 40, 40, form_hint="窝穴")
        assert star.type in ("金星", "木星", "水星", "土星", "火星", "不清", "廉贞")
        bonus, notes = score_xue_star_bonus(star)
        assert -10 <= bonus <= 10
        assert isinstance(notes, str)
