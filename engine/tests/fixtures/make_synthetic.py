"""合成测试数据：生成符合风水格局的合成 DEM。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds


def make_synthetic_dem(
    path: str | Path,
    h: int = 200,
    w: int = 200,
    cell_size_m: float = 30.0,
    crs: str = "EPSG:3857",
) -> None:
    """教科书式背山面水格局。
    特征在正确四兽方位，中心为窝穴且被四兽环绕。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h // 2, w // 2

    # 1. 平坦基底（避免北高南低梯度导致中心被高估）
    z = 500.0 + (cy - yy) * 0.05

    # 特征距离中心（行像素差×30m），按传统四兽严格方位对齐：
    #   少祖 (35,100) → 1950m (+200m)  北
    #   玄武 (86,100) → 420m  (+110m) 北
    #   青龙 (100,118) → 540m (+80m)  东  (90°)
    #   白虎 (100, 82) → 540m (+50m)  西  (270°)
    #   案山 (118,100) → 540m (+55m)  南  (180°)
    #   朝山 (155,100) → 1650m (+90m) 南
    #   窝穴 (98,100)  → 60m  (-40m)   中央

    z += 200.0 * np.exp(-((yy - 35) ** 2 + (xx - cx) ** 2) / (2 * 20**2))   # 少祖
    z += 110.0 * np.exp(-((yy - 86) ** 2 + (xx - cx) ** 2) / (2 * 16**2))   # 玄武
    z += 80.0 * np.exp(-((yy - cx) ** 2 + (xx - 118) ** 2) / (2 * 15**2))  # 青龙 (正东)
    z += 50.0 * np.exp(-((yy - cx) ** 2 + (xx - 82) ** 2) / (2 * 15**2))   # 白虎 (正西)
    z += 55.0 * np.exp(-((yy - 118) ** 2 + (xx - cx) ** 2) / (2 * 15**2))  # 案山
    z += 90.0 * np.exp(-((yy - 155) ** 2 + (xx - cx) ** 2) / (2 * 14**2))  # 朝山
    z -= 40.0 * np.exp(-((yy - 98) ** 2 + (xx - cx) ** 2) / (2 * 7**2))    # 窝穴

    # 写 GeoTIFF
    transform = from_bounds(0, 0, w * cell_size_m, h * cell_size_m, w, h)
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(z.astype(np.float32), 1)


def make_synthetic_rivers(path: str | Path, cell_size_m: float = 30.0) -> None:
    """生成合成水系 GeoJSON：南向一条玉带水。"""
    import geopandas as gpd
    from shapely.geometry import LineString

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 穴心在 (3000, 3000)，玉带水在正南 y=1450-1700
    coords = [
        (2000, 1700), (2200, 1600), (2400, 1530), (2600, 1490),
        (2800, 1470), (3000, 1450), (3200, 1470), (3400, 1490),
        (3600, 1530), (3800, 1600), (4000, 1700),
    ]
    gdf = gpd.GeoDataFrame(
        {"name": ["jade_belt"]},
        geometry=[LineString(coords)],
        crs="EPSG:3857",
    )
    gdf.to_file(path, driver="GeoJSON")
