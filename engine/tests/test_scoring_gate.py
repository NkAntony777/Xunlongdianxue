"""门禁 + 排名：重构后保留重构前可用性（橙心/热峰有候选）。"""
from __future__ import annotations

import numpy as np
import pytest
from rasterio.transform import from_origin

from engine.io.dem import DEM
from engine.core.fengshui_score import (
    _gate_beasts_for_hole,
    find_and_rank_candidates,
)


def _flatish_dem(h: int = 80, w: int = 80, cell: float = 30.0) -> DEM:
    """中心略凹 + 北侧靠山，模拟明堂堂心。"""
    yy, xx = np.mgrid[0:h, 0:w]
    cr, cc = h // 2, w // 2
    z = 120.0 + (cr - yy) * 0.08
    # 北靠
    z += 45.0 * np.exp(-((yy - h * 0.22) ** 2 + (xx - cc) ** 2) / (2 * 6**2))
    # 少祖更远更高
    z += 70.0 * np.exp(-((yy - h * 0.08) ** 2 + (xx - cc) ** 2) / (2 * 5**2))
    # 堂心微凹
    z -= 8.0 * np.exp(-((yy - cr) ** 2 + (xx - cc) ** 2) / (2 * 4**2))
    # 南案
    z += 25.0 * np.exp(-((yy - h * 0.78) ** 2 + (xx - cc) ** 2) / (2 * 5**2))
    # 左右砂
    z += 20.0 * np.exp(-((yy - cr) ** 2 + (xx - w * 0.28) ** 2) / (2 * 4**2))
    z += 18.0 * np.exp(-((yy - cr) ** 2 + (xx - w * 0.72) ** 2) / (2 * 4**2))
    transform = from_origin(0.0, h * cell, cell, cell)
    return DEM(
        data=z.astype(np.float64),
        transform=transform,
        crs="EPSG:3857",
        nodata=-9999.0,
        bounds=(0.0, 0.0, w * cell, h * cell),
        resolution=(cell, cell),
    )


class TestGateSoftElev:
    def test_soft_keep_passes_when_no_xuanwu(self):
        """soft_keep：缺玄武仍通过，记 meta。"""
        dem = _flatish_dem(40, 40)
        # 极平 DEM 中心可能缺峰；强制 soft
        ok, reason, info = _gate_beasts_for_hole(
            dem, 20, 20, water=None, soft_keep=True, min_side_beasts=0,
        )
        assert ok is True
        assert info.get("soft_keep") is True
        assert reason  # 有 reason 字符串

    def test_require_shaozu_higher_hard_when_enabled(self, monkeypatch):
        """兼容旧行为：require_shaozu_higher=True 且祖低 → 硬否（非 soft）。"""
        dem = _flatish_dem(60, 60)

        class FakeFb:
            facing = 180.0
            sit = 0.0
            meta = {
                "beasts": {
                    "shaozu": {"elev_m": 100.0, "dist_m": 800.0},
                    "xuanwu": {"elev_m": 150.0, "dist_m": 300.0},
                    "zhuque": {"elev_m": 90.0, "dist_m": 200.0},
                    "qinglong": {"elev_m": 95.0, "dist_m": 180.0},
                    "baihu": {"elev_m": 92.0, "dist_m": 190.0},
                }
            }

        def _fake_detect(*_a, **_k):
            return FakeFb()

        monkeypatch.setattr(
            "engine.core.four_beasts_detect.detect_four_beasts",
            _fake_detect,
        )
        ok, reason, info = _gate_beasts_for_hole(
            dem, 30, 30, water=None,
            require_shaozu_higher=True,
            soft_keep=False,
            min_side_beasts=0,
        )
        assert ok is False
        assert reason == "shaozu_not_higher"
        assert info.get("shaozu_minus_xuanwu_m") == pytest.approx(-50.0)

    def test_soft_elev_bonus_when_shaozu_higher(self, monkeypatch):
        """默认：祖高于玄 → 通过 + elev_bonus。"""
        dem = _flatish_dem(60, 60)

        class FakeFb:
            facing = 180.0
            sit = 0.0
            meta = {
                "beasts": {
                    "shaozu": {"elev_m": 200.0, "dist_m": 900.0},
                    "xuanwu": {"elev_m": 140.0, "dist_m": 280.0},
                    "zhuque": {"elev_m": 90.0, "dist_m": 220.0},
                    "qinglong": {"elev_m": 95.0, "dist_m": 180.0},
                    "baihu": {"elev_m": 92.0, "dist_m": 190.0},
                }
            }

        monkeypatch.setattr(
            "engine.core.four_beasts_detect.detect_four_beasts",
            lambda *_a, **_k: FakeFb(),
        )
        ok, reason, info = _gate_beasts_for_hole(
            dem, 30, 30, water=None,
            require_shaozu_higher=False,
            soft_keep=False,
            min_side_beasts=1,
        )
        assert ok is True
        assert reason == "ok"
        assert int(info.get("shaozu_higher_bonus") or 0) > 0

    def test_soft_elev_penalty_not_kill_when_shaozu_lower(self, monkeypatch):
        """默认：祖低于玄 → 仍通过，负 bonus。"""
        dem = _flatish_dem(60, 60)

        class FakeFb:
            facing = 180.0
            sit = 0.0
            meta = {
                "beasts": {
                    "shaozu": {"elev_m": 100.0, "dist_m": 800.0},
                    "xuanwu": {"elev_m": 150.0, "dist_m": 300.0},
                    "zhuque": {"elev_m": 90.0, "dist_m": 200.0},
                    "qinglong": None,
                    "baihu": None,
                }
            }

        monkeypatch.setattr(
            "engine.core.four_beasts_detect.detect_four_beasts",
            lambda *_a, **_k: FakeFb(),
        )
        ok, reason, info = _gate_beasts_for_hole(
            dem, 30, 30, water=None,
            require_shaozu_higher=False,
            soft_keep=False,
            min_side_beasts=0,
        )
        assert ok is True
        assert reason == "ok"
        assert int(info.get("shaozu_higher_bonus") or 0) < 0
        assert info.get("shaozu_lower_soft") is True


class TestRankKeepsPeak:
    def test_rank_returns_candidates_with_require_beasts(self):
        """重构后 require_beasts=True 仍应返回候选（不空杀）。"""
        dem = _flatish_dem(100, 100, cell=40.0)
        results = find_and_rank_candidates(
            dem, water=None, top_k=5, min_score=0, require_beasts=True,
        )
        assert len(results) >= 1
        assert all(0 <= r.overall <= 100 for r in results)
        # 应有 rank / id
        assert results[0].rank == 1
        assert results[0].candidate_id.startswith("C-")

    def test_rank_without_beasts_also_works(self):
        dem = _flatish_dem(80, 80, cell=40.0)
        results = find_and_rank_candidates(
            dem, water=None, top_k=3, min_score=0, require_beasts=False,
        )
        assert len(results) >= 1
