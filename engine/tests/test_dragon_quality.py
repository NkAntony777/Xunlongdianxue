"""Tier-1 龙脉质量：拓扑累积、平地解算、入首、主龙链路。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core.dragon_vein import (
    compute_flow_accumulation,
    compute_flow_direction,
    resolve_flats,
    _flow_direction_numpy,
    analyze_dragon_vein,
    select_primary_dragon,
    find_entrance_on_ridge,
    RidgeLine,
)
from engine.io.dem import DEM


def _dem_from_z(z: np.ndarray, cell: float = 30.0) -> DEM:
    h, w = z.shape
    transform = from_origin(0.0, h * cell, cell, cell)
    return DEM(
        data=z.astype(np.float64),
        transform=transform,
        crs="EPSG:3857",
        nodata=-9999.0,
        bounds=(0.0, 0.0, w * cell, h * cell),
        resolution=(cell, cell),
    )


class TestTopoFlowAccumulation:
    def test_fishbone_upstream_sums(self):
        """鱼骨：上游两源汇入下游，acc 应为拓扑序正确值。"""
        # 3 列：左源、中汇、右源 → 都流向中心再向下
        # 简化：一行链 0→1→2→3 流向东 (code=1)
        h, w = 1, 5
        flow = np.zeros((h, w), dtype=np.int32)
        # E=1
        for c in range(w - 1):
            flow[0, c] = 1
        flow[0, w - 1] = 0  # sink
        acc = compute_flow_accumulation(flow)
        # 每格自带 1，向东传递：acc[0]=1, acc[1]=2, ... acc[4]=5
        assert acc[0, 0] == pytest.approx(1.0)
        assert acc[0, 1] == pytest.approx(2.0)
        assert acc[0, 4] == pytest.approx(5.0)

    def test_row_scan_would_fail_on_reverse_order(self):
        """若流向朝西，拓扑序仍应在源头 acc=1、汇合端更大。"""
        h, w = 1, 4
        flow = np.zeros((h, w), dtype=np.int32)
        # W=16：每格流向左
        for c in range(1, w):
            flow[0, c] = 16
        flow[0, 0] = 0
        acc = compute_flow_accumulation(flow)
        assert acc[0, 3] == pytest.approx(1.0)
        assert acc[0, 0] == pytest.approx(4.0)


class TestResolveFlats:
    def test_flat_gets_micro_slope(self):
        z = np.ones((20, 20), dtype=np.float64) * 100.0
        # 中心平台，边缘更低
        z[0, :] = 90.0
        z[-1, :] = 90.0
        z[:, 0] = 90.0
        z[:, -1] = 90.0
        z[5:15, 5:15] = 100.0
        out = resolve_flats(z, epsilon=1e-3)
        # 平台内部应有微抬差异
        plat = out[6:14, 6:14]
        assert plat.std() > 0 or plat.max() > plat.min()
        # 流向在 resolve 后不应全 0
        fd = _flow_direction_numpy(out)
        assert (fd[6:14, 6:14] != 0).sum() > 10


class TestEntranceCurvature:
    def test_prefers_drop_not_only_min(self):
        h, w = 40, 10
        z = np.full((h, w), 200.0)
        # 南北脊：高→缓→急降→低
        for r in range(h):
            z[r, 5] = 300.0 - r * 1.5
        z[28:35, 5] = z[28:35, 5] - np.linspace(0, 40, 7)  # 急降
        dem = _dem_from_z(z)
        coords = np.array([[r, 5] for r in range(2, 38)], dtype=np.int32)
        ridge = RidgeLine(
            coords=coords,
            length_m=1000.0,
            mean_elevation=250.0,
            max_elevation=300.0,
            sinuosity=1.2,
            feature_significance=1.0,
        )
        pt = find_entrance_on_ridge(ridge, dem, window=5)
        assert pt is not None
        # 入首应落在急降段附近（末端 1/3）
        assert pt[0] >= 20


class TestTier2StrahlerMerge:
    def test_merge_and_strahler(self):
        from engine.core.dragon_strahler import grade_and_merge_ridges, merge_ridge_lines
        from engine.core.dragon_vein import RidgeLine

        h, w = 50, 30
        z = np.full((h, w), 100.0)
        for r in range(5, 45):
            z[r, 10] = 150.0 + r * 0.2
            z[r, 12] = 140.0 + r * 0.15
        dem = _dem_from_z(z)
        # 两段接近的脊，应可合并
        c1 = np.array([[r, 10] for r in range(5, 25)], dtype=np.int32)
        c2 = np.array([[r, 10] for r in range(25, 45)], dtype=np.int32)
        r1 = RidgeLine(c1, 600.0, 160.0, 180.0, 1.1, 1.0)
        r2 = RidgeLine(c2, 600.0, 170.0, 190.0, 1.1, 1.0)
        merged = merge_ridge_lines([r1, r2], dem, merge_dist_m=50.0, merge_dh_m=40.0)
        assert len(merged) <= 2
        graded = grade_and_merge_ridges([r1, r2], dem)
        assert len(graded) >= 1
        assert all(hasattr(g, "strahler_order") for g in graded)


class TestTier3ViewshedDual:
    def test_viewshed_open_line(self):
        from engine.core.dragon_vein import sector_viewshed_score, dual_signal_anchor

        h, w = 40, 40
        z = np.full((h, w), 100.0)
        dem = _dem_from_z(z)
        # 平地视线应开阔
        s = sector_viewshed_score(dem, 20, 10, 20, 30)
        assert s > 0.8
        # 中间高墙遮挡
        z[:, 20] = 200.0
        dem2 = _dem_from_z(z)
        s2 = sector_viewshed_score(dem2, 20, 5, 20, 35)
        assert s2 < s

    def test_dual_anchor_pull(self):
        from engine.core.dragon_vein import dual_signal_anchor

        a = dual_signal_anchor((10, 10), (10, 10), 30.0, 30.0)
        assert a == (10, 10)
        b = dual_signal_anchor((10, 10), (100, 100), 30.0, 30.0, pull_m=100.0)
        assert b is not None
        # 过远应向入首拉
        assert b[0] > 10 or b[1] > 10


class TestPrimaryDragonMeta:
    def test_rank_meta_has_primary_dragon(self):
        from engine.core.fengshui_score import find_and_rank_candidates

        h, w = 60, 60
        yy, xx = np.mgrid[0:h, 0:w]
        z = 100.0 + (30 - yy) * 0.5
        z += 40.0 * np.exp(-((yy - 15) ** 2 + (xx - 30) ** 2) / 40.0)
        z += 25.0 * np.exp(-((yy - 35) ** 2 + (xx - 30) ** 2) / 30.0)
        dem = _dem_from_z(z, cell=30.0)
        ranked, ctx = find_and_rank_candidates(
            dem, water=None, top_k=5, min_score=0, return_context=True,
        )
        assert len(ranked) >= 1
        # primary 应在 context
        assert "primary_dragon" in ctx or ctx.get("dragon_vein") is not None
        # 至少一个候选带 long_az / primary_dragon meta（若龙脉成功）
        if ctx.get("primary_dragon") is not None:
            m = ranked[0].meta or {}
            assert "dragon_align" in m or "qi_field" in m
