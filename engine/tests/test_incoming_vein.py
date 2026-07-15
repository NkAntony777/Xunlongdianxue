"""来龙筛选 + 脊上少祖/玄武：轻量路径与四象耦合。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core.dragon_vein import (
    light_ridge_mask,
    select_incoming_vein,
)
from engine.core.four_beasts_detect import detect_four_beasts
from engine.io.dem import DEM


def _make_ns_ridge_dem(
    h: int = 100,
    w: int = 80,
    cell: float = 30.0,
) -> tuple[DEM, int, int]:
    """北高南低主脊 + 南侧穴：来龙宜 N→S。"""
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w // 2
    z = 100.0 + (h - yy) * 0.35  # 北高南低底坡
    # 中央南北脊（高 TPI）
    z += 55.0 * np.exp(-((xx - cx) ** 2) / (2 * 3.5**2))
    # 穴：脊南端略凹（先定穴位，再按距离布峰）
    cr, cc = 70, cx
    # 玄武/父母：穴北约 300–400 m（~12 像元 ×30m）
    z += 45.0 * np.exp(-((yy - 58) ** 2 + (xx - cx) ** 2) / (2 * 3.2**2))
    # 少祖：更北约 1.2 km
    z += 75.0 * np.exp(-((yy - 28) ** 2 + (xx - cx) ** 2) / (2 * 4**2))
    z -= 12.0 * np.exp(-((yy - cr) ** 2 + (xx - cc) ** 2) / (2 * 2.5**2))
    # 干扰：东侧孤立高峰（不应抢来龙）
    z += 90.0 * np.exp(-((yy - 45) ** 2 + (xx - (cx + 28)) ** 2) / (2 * 4**2))

    transform = from_origin(0.0, h * cell, cell, cell)
    dem = DEM(
        data=z.astype(np.float32),
        transform=transform,
        crs="EPSG:3857",
        nodata=-9999.0,
        bounds=(0.0, 0.0, w * cell, h * cell),
        resolution=(cell, cell),
    )
    return dem, cr, cc


class TestLightRidgeMask:
    def test_shape_and_spine(self):
        dem, _, cx = _make_ns_ridge_dem()
        mask = light_ridge_mask(dem)
        assert mask.shape == dem.data.shape
        assert mask.any()
        # 中央列应有脊带像素
        assert mask[:, cx].sum() > mask[:, 5].sum()


class TestSelectIncomingVein:
    def test_picks_north_ridge_not_east_outlier(self):
        dem, cr, cc = _make_ns_ridge_dem()
        # 坐北朝南：sit=0, facing=180
        vein = select_incoming_vein(
            dem, cr, cc, sit_deg=0.0, facing_deg=180.0,
        )
        assert vein.method.startswith("light") or vein.method == "ridge_lines"
        assert vein.xuanwu is not None
        # 玄武应在穴北（row 更小）
        assert vein.xuanwu.row < cr
        # 方位大致朝北扇区
        assert abs(((vein.xuanwu.bearing_deg + 180) % 360) - 180) < 90 or \
            vein.xuanwu.bearing_deg < 60 or vein.xuanwu.bearing_deg > 300

        if vein.shaozu is not None:
            assert vein.shaozu.dist_m > vein.xuanwu.dist_m * 0.9
            assert vein.shaozu.row < vein.xuanwu.row + 5  # 更靠北
            # 不应落到东侧干扰峰（col 远离中央）
            assert abs(vein.shaozu.col - cc) < 15

        if vein.incoming_azimuth_deg is not None:
            # 龙气走向宜接近朝南 (180°)
            align = abs(((vein.incoming_azimuth_deg - 180 + 180) % 360) - 180)
            assert align < 55.0

    def test_downhill_flag(self):
        dem, cr, cc = _make_ns_ridge_dem()
        vein = select_incoming_vein(dem, cr, cc, sit_deg=0.0, facing_deg=180.0)
        if vein.xuanwu and vein.shaozu:
            assert vein.downhill_ok is True


class TestPrimaryDragonFirst:
    def test_select_primary_on_ns_ridge(self):
        from engine.core.dragon_vein import select_primary_dragon, analyze_dragon_vein

        dem, cr, cc = _make_ns_ridge_dem()
        # 合成脊较短，降低 min_length
        dv = analyze_dragon_vein(dem, min_length_m=30.0)
        primary = select_primary_dragon(
            dem, water=None, dragon_vein=dv, min_length_m=30.0,
        )
        # 若脊提取成功，主龙应存在且龙气大致南行
        if primary is None:
            pytest.skip("synthetic DEM produced no ridge lines")
        assert primary.entrance_row >= 0
        assert 0.0 <= primary.flow_azimuth_deg < 360.0
        # 锚点选龙：应带回 meta
        assert primary.meta is not None
        assert primary.length_m >= 0

    def test_rank_prefers_near_entrance(self):
        from engine.core.fengshui_score import find_and_rank_candidates
        from engine.core.dragon_vein import analyze_dragon_vein, select_primary_dragon

        dem, cr, cc = _make_ns_ridge_dem()
        dv = analyze_dragon_vein(dem, min_length_m=30.0)
        primary = select_primary_dragon(
            dem, water=None, dragon_vein=dv, min_length_m=30.0,
        )
        ranked, ctx = find_and_rank_candidates(
            dem, water=None, top_k=5, min_score=0,
            dragon_vein=dv, primary_dragon=primary, return_context=True,
        )
        assert len(ranked) >= 1
        assert ctx.get("primary_dragon") is not None or primary is None
        if primary is not None and ranked:
            top = ranked[0]
            meta = top.meta or {}
            # top 应带龙对齐字段
            assert "dragon_align" in meta or meta.get("qi_field") is not None


class TestOrientRidgeToHole:
    def test_source_not_flipped_to_front(self):
        """相对穴定向：更高但在「前」的一端不得抢源。"""
        from engine.core.dragon_vein import orient_ridge_to_hole, _m_per_px_dem

        dem, cr, cc = _make_ns_ridge_dem()
        # 人造脊：北(源侧)低一点、南(前)更高 —— 旧算法会把源钉在南
        h, w = dem.data.shape
        cx = w // 2
        coords = np.array(
            [[r, cx] for r in range(15, min(85, h - 5), 2)],
            dtype=np.int32,
        )
        # 抬高南端
        for r, c in coords[-5:]:
            dem.data[r, c] = float(dem.data[r, c]) + 80.0
        mpx, mpy = _m_per_px_dem(dem)
        ordered, meta = orient_ridge_to_hole(
            dem, coords, cr, cc, mpx, mpy, water_dist=None,
        )
        assert meta.get("ok")
        src = meta["source"]
        # 源应在穴北（row 更小），不能因南端更高就选南
        assert src[0] < cr, f"source should be north of hole, got {src} hole={(cr,cc)}"


class TestClassicalDragonSit:
    def test_shaozu_on_source_not_compass_east(self):
        """坐靠来龙：少祖在脊源端，sit 指向源，与绝对东无关。"""
        from engine.core.dragon_vein import (
            analyze_dragon_vein,
            select_primary_dragon,
            beasts_from_primary_dragon,
        )
        from engine.core.four_beasts_detect import detect_four_beasts

        dem, cr, cc = _make_ns_ridge_dem()
        dv = analyze_dragon_vein(dem, min_length_m=30.0)
        primary = select_primary_dragon(
            dem, water=None, dragon_vein=dv, min_length_m=30.0,
        )
        if primary is None:
            pytest.skip("no primary ridge")
        vein = beasts_from_primary_dragon(dem, cr, cc, primary)
        assert vein.method == "primary_dragon_classical"
        # 坐向应大致指向龙源（相对穴）
        if vein.shaozu is not None:
            assert vein.shaozu.dist_m > 100
            # 少祖应比玄武更远（若两者都有）
            if vein.xuanwu is not None:
                assert vein.shaozu.dist_m >= vein.xuanwu.dist_m * 0.9

        fb = detect_four_beasts(
            dem, cr, cc, water=None,
            dragon_vein=dv, primary_dragon=primary,
        )
        assert fb.facing_method in (
            "sit_to_dragon_source",
            "dragon_sit_face_water",
            "dragon_source_sit",
        ) or fb.meta.get("incoming_vein", {}).get("method") == "primary_dragon_classical"
        # 不应再出现「先面东再坐西」的用户误解路径作为默认
        assert fb.facing_method != "face_water" or primary is None


class TestDetectCoupledVein:
    def test_meta_has_incoming_vein(self):
        dem, cr, cc = _make_ns_ridge_dem()
        fb = detect_four_beasts(dem, cr, cc, facing=180.0)
        assert "incoming_vein" in fb.meta
        assert fb.meta["incoming_vein"].get("used") is True
        xw = fb.meta["beasts"].get("xuanwu")
        assert xw is not None
        assert "on_ridge" in xw
        # 玄武在北
        assert xw["row"] < cr

    def test_can_disable_vein(self):
        dem, cr, cc = _make_ns_ridge_dem()
        fb = detect_four_beasts(
            dem, cr, cc, facing=180.0, use_incoming_vein=False,
        )
        assert fb.meta["incoming_vein"]["used"] is False
        assert fb.xuanwu is not None or fb.meta["beasts"]["xuanwu"] is not None
