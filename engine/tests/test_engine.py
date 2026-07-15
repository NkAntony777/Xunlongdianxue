"""单元测试。"""
import sys
from pathlib import Path

# 让测试可以从 engine/ 目录外运行
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from engine.io.dem import DEM, load_dem, fill_pits, clip_dem
from engine.io.rivers import load_water
from engine.core.terrain_analysis import (
    compute_slope_aspect, analyze_terrain, tpi
)
from engine.core.four_beasts import score_four_beasts
from engine.core.acupoint import (
    search_candidates, score_form, classify_form
)
from engine.core.sand_water import (
    score_water_relation, score_sand_mountain
)
from engine.core.fengshui_score import (
    find_and_rank_candidates, score_candidate, to_json
)
from engine.tests.fixtures.make_synthetic import make_synthetic_dem, make_synthetic_rivers


FIXTURES = Path(__file__).parent / "fixtures"


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
def synth_dem(synth_dem_path):
    return load_dem(synth_dem_path)


@pytest.fixture(scope="module")
def synth_water(synth_rivers_path):
    return load_water(synth_rivers_path)


def test_load_dem(synth_dem):
    assert synth_dem.shape == (200, 200)
    assert synth_dem.resolution == (30.0, 30.0)
    assert synth_dem.crs is not None
    valid = synth_dem.data[np.isfinite(synth_dem.data)]
    assert valid.size > 0
    assert valid.max() > 500


def test_fill_pits(synth_dem):
    filled = fill_pits(synth_dem)
    assert filled.data.shape == synth_dem.data.shape
    assert not np.isnan(filled.data).all()
    # 填洼后最低点应比原来高（或相等）
    orig_min = np.nanmin(synth_dem.data)
    new_min = np.nanmin(filled.data)
    assert new_min >= orig_min - 0.1


def test_compute_slope_aspect(synth_dem):
    slope, aspect = compute_slope_aspect(synth_dem)
    assert slope.shape == synth_dem.shape
    assert aspect.shape == synth_dem.shape
    valid = slope[np.isfinite(slope)]
    assert valid.min() >= 0
    assert valid.max() <= 90
    aspect_valid = aspect[np.isfinite(aspect)]
    assert aspect_valid.min() >= 0
    assert aspect_valid.max() <= 360


def test_analyze_terrain(synth_dem):
    m = analyze_terrain(synth_dem)
    assert m.mean_elevation > 400
    assert m.relief > 100
    assert m.mean_slope > 0
    assert m.dominant_aspect in ["北", "东北", "东", "东南", "南", "西南", "西", "西北", "未主向"]
    assert m.terrain_position in ["缓坡台地", "丘陵区", "山地区", "高山区"]


def test_all_nan_dem_e5_safe():
    """【E.5】全 NaN DEM：分析中性、搜索空、排名空，不崩溃不造高分。"""
    from rasterio.transform import from_origin
    from engine.io.dem import DEM
    from engine.core.acupoint import search_candidates
    from engine.core.fengshui_score import find_and_rank_candidates

    data = np.full((40, 40), np.nan, dtype=np.float64)
    dem = DEM(
        data=data,
        transform=from_origin(0, 40 * 30, 30, 30),
        crs="EPSG:3857",
        nodata=-9999.0,
        bounds=(0.0, 0.0, 40 * 30.0, 40 * 30.0),
        resolution=(30.0, 30.0),
    )
    m = analyze_terrain(dem)
    assert m.terrain_position == "无效DEM"
    assert not (m.mean_elevation == m.mean_elevation)  # NaN
    assert search_candidates(dem, max_candidates=10, step=2) == []
    ranked = find_and_rank_candidates(dem, water=None, top_k=5, min_score=0)
    assert ranked == []


def test_tpi(synth_dem):
    tpi_arr = tpi(synth_dem, radius_m=100)
    valid = tpi_arr[np.isfinite(tpi_arr)]
    assert valid.size > 0
    assert valid.std() > 0


def test_score_four_beasts(synth_dem):
    score = score_four_beasts(synth_dem, search_radius_m=300)
    assert 0 <= score.qinglong <= 100
    assert 0 <= score.baihu <= 100
    assert 0 <= score.zhuque <= 100
    assert 0 <= score.xuanwu <= 100
    assert 0 <= score.combined <= 100
    # 扇区半宽修正为 45°，九象象不重叠。玄武落北 / 案山落南测得分正向。
    # 朱雀扇区中心 = facing (默认朝南 180)，应捕捉到朝山；玄武扇区中心 = sit (0)
    # 朝山的 max 应在朱雀扇区、玄武主峰 max 应在玄武扇区。
    details = score.details
    assert details["qinglong_sector"]["max_height"] >= details["baihu_sector"]["max_height"] - 30
    # 玄武扇区确实捕获到 (86, 100) 主峰的最大高程：
    assert details["xuanwu_sector"]["max_height"] >= 600
    # 朱雀扇区确实捕获到朝/案山：
    assert details["zhuque_sector"]["max_height"] >= 590


def test_classify_form():
    assert classify_form(-2.0) == "窝穴"
    assert classify_form(1.5) == "突穴"
    assert classify_form(0.5) == "乳穴"
    assert classify_form(-0.5, local_slope=20) == "钳穴"


def test_score_form():
    assert score_form(-2.0, "窝穴") > 80
    # 平缓明堂：TPI≈0 应高分（不再当废地）
    assert score_form(0.0, "平缓") >= 80


def test_search_candidates(synth_dem):
    cands = search_candidates(synth_dem, max_candidates=20, step=10)
    assert len(cands) > 0
    assert len(cands) <= 20
    for c in cands:
        assert np.isfinite(c.elevation)
        assert c.form_type in ["窝穴", "钳穴", "突穴", "乳穴", "平缓"]


def test_score_water_relation_no_water():
    from engine.io.rivers import WaterNetwork
    import geopandas as gpd
    empty = WaterNetwork(gdf=gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    s = score_water_relation(105.0, 31.0, empty)
    assert s.is_placeholder
    assert s.score == 60


def test_score_water_relation_with_water(synth_water):
    # 合成水系位于 y=1450-1700（正南玉带水），x=2000-4000
    # 查询 (3000, 1500) 附近 → 应非常近
    s = score_water_relation(3000, 1500, synth_water)
    assert not s.is_placeholder
    assert s.nearest_distance_m < 100
    assert 0 <= s.score <= 100


def test_score_sand_mountain(synth_dem):
    score = score_sand_mountain(synth_dem, search_radius_m=500)
    assert 0 <= score.score <= 100
    assert score.left_peak_count >= 0
    assert score.right_peak_count >= 0


def test_score_candidate(synth_dem, synth_water):
    from engine.core.acupoint import search_candidates
    from engine.core.terrain_analysis import analyze_terrain
    cands = search_candidates(synth_dem, max_candidates=5, step=15)
    terrain = analyze_terrain(synth_dem)
    if cands:
        fused = score_candidate(synth_dem, cands[0], terrain, synth_water)
        assert 0 <= fused.overall <= 100
        assert "four_beasts" in fused.scores
        assert "water" in fused.scores
        assert fused.geography["qinglong"] is not None


def test_find_and_rank_candidates(synth_dem, synth_water):
    results = find_and_rank_candidates(synth_dem, synth_water, top_k=5, min_score=0)
    assert len(results) > 0
    # 排名应递减
    for i in range(len(results) - 1):
        assert results[i].overall >= results[i + 1].overall
    # rank 应递增
    for i, r in enumerate(results):
        assert r.rank == i + 1


def test_to_json(synth_dem, synth_water):
    results = find_and_rank_candidates(synth_dem, synth_water, top_k=3, min_score=0)
    out = to_json(results, metadata={"bbox": [0, 0, 6000, 6000]})
    assert "candidates" in out
    assert "metadata" in out
    assert len(out["candidates"]) == len(results)
    for c in out["candidates"]:
        assert "overall_score" in c
        assert "x" in c
        assert "y" in c
        assert "scores" in c


def test_clip_dem(synth_dem):
    bbox = synth_dem.bounds
    # 裁剪中心 1000x1000 m
    minx, miny = bbox[0] + 1000, bbox[1] + 1000
    maxx, maxy = minx + 1000, miny + 1000
    sub = clip_dem(synth_dem, (minx, miny, maxx, maxy))
    assert sub.bounds == (minx, miny, maxx, maxy)
    assert sub.width < synth_dem.width
    assert sub.height < synth_dem.height


def test_reproject_dem(synth_dem):
    from engine.io.dem import reproject_dem
    reproj = reproject_dem(synth_dem, "EPSG:4326")
    assert reproj.crs is not None
    assert reproj.data.shape == synth_dem.data.shape or abs(
        reproj.width - synth_dem.width
    ) < 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
