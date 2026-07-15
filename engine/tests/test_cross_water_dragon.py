"""水界龙止 / 龙气不过水：同岸约束单测。"""
from __future__ import annotations

import numpy as np
from rasterio.transform import from_bounds

from engine.io.dem import DEM
from engine.core.four_beasts_detect import (
    _segment_hits_water,
    same_bank_as_hole,
    _select_peak_in_sector,
    _local_maxima_mask,
)


def _dem(h=80, w=80, cell=30.0):
    data = np.full((h, w), 100.0, dtype=float)
    # 左岸丘 + 右岸丘，中间河道列 38-42 低
    data[:, 15:25] += 80.0  # left hills
    data[:, 55:70] += 120.0  # right higher "祖"
    data[35:45, 35:45] = 105.0  # center plateau (hole)
    transform = from_bounds(0, 0, w * cell, h * cell, w, h)
    return DEM(
        data=data,
        transform=transform,
        crs="EPSG:3857",
        nodata=-9999.0,
        bounds=(0.0, 0.0, w * cell, h * cell),
        resolution=(cell, cell),
    )


class TestSegmentCrossWater:
    def test_hits_water_midline(self):
        dem = _dem()
        # water surface: wider river columns 36-45
        ws = np.zeros(dem.data.shape, dtype=bool)
        ws[:, 36:46] = True
        # left peak col 20 → right peak col 60 crosses water
        assert _segment_hits_water(ws, 40, 20, 40, 60) is True
        # same bank left
        assert _segment_hits_water(ws, 40, 20, 40, 22) is False
        # hole on left dry bank col 30
        assert same_bank_as_hole(ws, 40, 30, 40, 20) is True
        assert same_bank_as_hole(ws, 40, 30, 40, 60) is False

    def test_single_pixel_river_still_detected(self):
        """OSM 河常仅 1 像元宽：中段命中 1 点即跨水（回归漏检）。"""
        ws = np.zeros((80, 80), dtype=bool)
        ws[:, 40] = True  # 单列河
        assert _segment_hits_water(ws, 40, 10, 40, 70, n_samples=24) is True
        assert same_bank_as_hole(ws, 40, 20, 40, 60) is False


class TestShaozuRejectCrossWater:
    def test_select_peak_rejects_opposite_bank(self):
        dem = _dem()
        h, w = dem.data.shape
        # peaks
        peaks = _local_maxima_mask(dem.data, size=5)
        # force peaks on both banks
        peaks[40, 20] = True
        peaks[40, 62] = True
        dem.data[40, 20] = 200.0
        dem.data[40, 62] = 250.0  # higher opposite

        ws = np.zeros((h, w), dtype=bool)
        ws[:, 38:43] = True

        rejected = []
        # sit west → direction 270, left bank is col small... 
        # hole at 40,40; back west = 270 → toward smaller col (left bank col 20)
        # facing east 90, sit west 270
        bp = _select_peak_in_sector(
            dem, 40, 40, 270.0, 60.0,
            (100.0, 5000.0), peaks,
            water_surface_mask=ws,
            reject_cross_water=True,
            cross_water_penalty=3.5,
            out_rejected_cross=rejected,
            border_margin=2,
            min_elev_above_cand=5.0,
        )
        # should prefer left bank or none, not col 62
        if bp is not None:
            assert bp.col < 38, f"少祖不得跨水到对岸 col={bp.col}"
        # opposite bank should be in rejected if scanned
        # (may or may not depending on sector — west sector includes left)

        # east sector sit 90 would include right bank and reject
        rejected2 = []
        bp2 = _select_peak_in_sector(
            dem, 40, 40, 90.0, 60.0,
            (100.0, 5000.0), peaks,
            water_surface_mask=ws,
            reject_cross_water=True,
            out_rejected_cross=rejected2,
            border_margin=2,
            min_elev_above_cand=5.0,
        )
        if bp2 is not None:
            assert bp2.col > 43 or bp2.col < 38  # if selected, same logic
        # right bank peak should be rejected when facing that way
        # (direction 90 goes east to col 62)
        assert any(p.col > 50 for p in rejected2) or bp2 is None or bp2.col < 38
