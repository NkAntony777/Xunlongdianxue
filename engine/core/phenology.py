"""微地形物候（NDVI / 土壤湿度 / 植被覆盖）接口。

传统理据：
  - 「穴上草木自然茂盛」：土气所钟，植物生长更旺。
  - 「土壤湿润，颜色深沉」：地下水近，地温恒。
  - 「山石黄白间紫」：矿物/微量元素沉淀。
  - 「晨雾先蒸、夕露迟下」：气化热力学差异。

DEM 之外的可选栅格（非必需；缺省时使用 DEM 估算的代理指标）：
  - ndvi.tif: 归一化植被指数，0-1。
  - soil_moisture.tif: 0-1 相对湿度。
  - vegetation_cover.tif: 植被覆盖率 0-1。
  - dem_proxy: 缺省代理——基于局部坡度 + 凸性 + TWI。

接口：
  - PhenologyInputs：聚合多个栅格
  - score_acupoint_phenology：在 (row, col) 周围做综合评估
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.io.dem import DEM


@dataclass
class PhenologyInputs:
    """物候数据聚合。"""

    ndvi: np.ndarray | None = None
    soil_moisture: np.ndarray | None = None
    vegetation_cover: np.ndarray | None = None
    soil_temperature_anomaly: np.ndarray | None = None  # K; 正 = 高于邻域


def align_raster_to_dem(arr: np.ndarray, dem: DEM) -> np.ndarray:
    """将外部栅格重采样/对齐到 DEM 形状（简化：同形状则直接裁剪/整流）。"""
    if arr.shape != dem.data.shape:
        # 仅支持同形状（不重采样）；其他情况返回 None
        return None
    return arr


def score_acupoint_phenology(
    dem: DEM,
    center_row: int,
    center_col: int,
    inputs: PhenologyInputs | None = None,
    *,
    search_radius_m: float = 30.0,
) -> dict[str, float]:
    """穴位物候打分。

    返回 dict{ndvi_score, moisture_score, vegetation_proxy, temperature_proxy, total}
    0-100 综合。
    """
    h, w = dem.data.shape
    if not (0 <= center_row < h and 0 <= center_col < w):
        return {"total": 50.0, "notes": "中心点越界"}

    # 计算 mpx, mpy（与 terrain_analysis._m_per_px 等价但避免私用）
    from engine.core.terrain_analysis import _is_geographic
    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs):
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        import math
        cos_lat = max(0.2, abs(math.cos(math.radians(mid_lat))))
        mpx = xres * 111_000.0 * cos_lat
        mpy = yres * 111_000.0
    else:
        mpx, mpy = xres, yres
    r_px = max(3, int(round(search_radius_m / max(mpx, mpy))))
    r0 = max(0, center_row - r_px)
    r1 = min(h, center_row + r_px + 1)
    c0 = max(0, center_col - r_px)
    c1 = min(w, center_col + r_px + 1)

    notes = []
    ndvi_score = 50.0
    moist_score = 50.0
    if inputs is not None and inputs.ndvi is not None:
        nd = align_raster_to_dem(inputs.ndvi, dem)
        if nd is not None and np.isfinite(nd[r0:r1, c0:c1]).any():
            local = float(np.nanmean(nd[r0:r1, c0:c1]))
            ndvi_score = float(np.clip(local * 100.0, 0.0, 100.0))
            notes.append(f"NDVI={local:.2f}")

    if inputs is not None and inputs.soil_moisture is not None:
        sm = align_raster_to_dem(inputs.soil_moisture, dem)
        if sm is not None and np.isfinite(sm[r0:r1, c0:c1]).any():
            local = float(np.nanmean(sm[r0:r1, c0:c1]))
            moist_score = float(np.clip(local * 100.0, 0.0, 100.0))
            notes.append(f"soil_m={local:.2f}")

    # 代理：基于 DEM + TWI 的物候指示（无 NDVI 时）
    veg_proxy = 50.0
    temp_proxy = 50.0
    try:
        from engine.core.terrain_analysis import compute_slope_aspect
        slope_arr, _ = compute_slope_aspect(dem)
        local_slope = float(np.nanmean(slope_arr[r0:r1, c0:c1]))
        # 缓坡 + 局部低洼 → 植被茂盛（代理）
        sub_elev = dem.data[r0:r1, c0:c1]
        cand_elev = float(dem.data[center_row, center_col])
        rel_h = float(np.nanmean(sub_elev) - cand_elev)
        veg_proxy = float(np.clip(70 - abs(local_slope) * 0.8 - rel_h * 0.4, 0, 100))
        temp_proxy = float(np.clip(60 - abs(local_slope) * 0.6, 0, 100))
    except Exception:
        pass

    # 综合：NDVI/湿/植被/温度按权重融合
    weights = (0.30, 0.25, 0.25, 0.20)
    total = (
        weights[0] * ndvi_score
        + weights[1] * moist_score
        + weights[2] * veg_proxy
        + weights[3] * temp_proxy
    )
    return {
        "ndvi_score": round(ndvi_score, 1),
        "moisture_score": round(moist_score, 1),
        "vegetation_proxy": round(veg_proxy, 1),
        "temperature_proxy": round(temp_proxy, 1),
        "total": round(total, 1),
        "notes": "; ".join(notes) if notes else "代理指标",
    }
