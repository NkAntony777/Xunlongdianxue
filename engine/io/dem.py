"""DEM 数据加载、裁剪、重投影、填洼。

提供栅格数据的统一访问接口。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject


@dataclass
class DEM:
    """数字高程模型。

    Attributes:
        data: 高程矩阵, shape=(H, W), 单位: m
        transform: 仿射变换, (x, y) ↔ (row, col)
        crs: 坐标系 (EPSG)
        nodata: NoData 值
        bounds: (minx, miny, maxx, maxy)
        resolution: (xres, yres) 米/像素
    """

    data: np.ndarray
    transform: Any
    crs: Any
    nodata: float | None
    bounds: tuple[float, float, float, float]
    resolution: tuple[float, float]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def rowcol(self, x: float, y: float) -> tuple[int, int]:
        """经纬度 → 行列号。"""
        inv = ~self.transform
        col, row = inv * (x, y)
        return int(row), int(col)

    def xy(self, row: int, col: int) -> tuple[float, float]:
        """行列号 → 经纬度。"""
        x, y = self.transform * (col, row)
        return float(x), float(y)

    def sample(self, x: float, y: float) -> float:
        """经纬度 → 高程值（双线性插值）。"""
        from rasterio.transform import rowcol

        row, col = rowcol(self.transform, x, y)
        if 0 <= row < self.height - 1 and 0 <= col < self.width - 1:
            r0, c0 = int(row), int(col)
            fr, fc = row - r0, col - c0
            v00 = self.data[r0, c0]
            v01 = self.data[r0, c0 + 1]
            v10 = self.data[r0 + 1, c0]
            v11 = self.data[r0 + 1, c0 + 1]
            v0 = v00 * (1 - fc) + v01 * fc
            v1 = v10 * (1 - fc) + v11 * fc
            return float(v0 * (1 - fr) + v1 * fr)
        if 0 <= row < self.height and 0 <= col < self.width:
            return float(self.data[int(row), int(col)])
        return float(self.nodata) if self.nodata is not None else float("nan")

    def mask_valid(self) -> np.ndarray:
        """有效像元掩膜（排除 NoData 和 NaN）。"""
        m = np.isfinite(self.data)
        if self.nodata is not None:
            m &= self.data != self.nodata
        return m


def load_dem(path: str | Path) -> DEM:
    """加载 GeoTIFF DEM。

    Args:
        path: GeoTIFF 文件路径

    Returns:
        DEM 数据对象
    """
    path = Path(path)
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        bounds = src.bounds
        resolution = (abs(src.transform.a), abs(src.transform.e))

    if nodata is not None:
        data[data == nodata] = np.nan
    return DEM(
        data=data,
        transform=transform,
        crs=crs,
        nodata=nodata,
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        resolution=resolution,
    )


def clip_dem(dem: DEM, bbox: tuple[float, float, float, float]) -> DEM:
    """按 bbox 裁剪 DEM。

    Args:
        dem: 源 DEM
        bbox: (minx, miny, maxx, maxy), 必须与 DEM 同坐标系
    """
    minx, miny, maxx, maxy = bbox
    inv = ~dem.transform
    r0, c0 = inv * (minx, maxy)
    r1, c1 = inv * (maxx, miny)
    r0, r1 = sorted([int(np.floor(r0)), int(np.ceil(r1))])
    c0, c1 = sorted([int(np.floor(c0)), int(np.ceil(c1))])
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(dem.height, r1)
    c1 = min(dem.width, c1)

    sub = dem.data[r0:r1, c0:c1].copy()
    new_transform = rasterio.transform.from_bounds(
        minx, miny, maxx, maxy, sub.shape[1], sub.shape[0]
    )
    return DEM(
        data=sub,
        transform=new_transform,
        crs=dem.crs,
        nodata=dem.nodata,
        bounds=(minx, miny, maxx, maxy),
        resolution=(abs(new_transform.a), abs(new_transform.e)),
    )


def reproject_dem(dem: DEM, target_crs: str | int) -> DEM:
    """重投影到目标坐标系（建议用 EPSG:3857 用于距离计算）。"""
    from rasterio.crs import CRS

    target_crs = CRS.from_user_input(target_crs)
    if dem.crs == target_crs:
        return dem

    transform, width, height = calculate_default_transform(
        dem.crs, target_crs, dem.width, dem.height, *dem.bounds
    )
    dst = np.full((height, width), np.nan, dtype=np.float64)
    reproject(
        source=dem.data,
        destination=dst,
        src_transform=dem.transform,
        src_crs=dem.crs,
        dst_transform=transform,
        dst_crs=target_crs,
        resampling=rasterio.enums.Resampling.bilinear,
    )
    from rasterio.transform import array_bounds

    bnds = array_bounds(height, width, transform)
    return DEM(
        data=dst,
        transform=transform,
        crs=target_crs,
        nodata=dem.nodata,
        bounds=bnds,
        resolution=(abs(transform.a), abs(transform.e)),
    )


def fill_pits(dem: DEM, max_depth: float = 1e6) -> DEM:
    """Wang & Liu 风格填洼：递归填平所有凹陷。

    简化实现：直接调用 pysheds（若可用），否则用 numpy 邻域最小值。
    """
    try:
        from pysheds.grid import Grid

        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            _write_geotiff(dem, tmp_path)
            grid = Grid.from_raster(tmp_path, data_name="dem")
            grid.fill_depressions("dem", out_name="flooded")
            with rasterio.open(tmp_path) as src:
                arr = src.read(1)
            filled = grid.view("flooded")
            result = filled if hasattr(filled, "filled") else np.array(filled)
            if hasattr(result, "filled"):
                result = result.filled(np.nan)
            out = DEM(
                data=np.asarray(result, dtype=np.float64),
                transform=dem.transform,
                crs=dem.crs,
                nodata=dem.nodata,
                bounds=dem.bounds,
                resolution=dem.resolution,
            )
            return out
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except ImportError:
        return _fill_pits_numpy(dem)


def _fill_pits_numpy(dem: DEM) -> DEM:
    """纯 numpy 填洼（迭代，直到无变化）。"""
    from scipy.ndimage import maximum_filter

    data = dem.data.copy()
    max_iter = 50
    for _ in range(max_iter):
        max_neighbor = maximum_filter(data, size=3)
        diff = max_neighbor - data
        sink_mask = (diff > 1e-6) & np.isfinite(data) & np.isfinite(max_neighbor)
        if not sink_mask.any():
            break
        data[sink_mask] = max_neighbor[sink_mask]
    return DEM(
        data=data,
        transform=dem.transform,
        crs=dem.crs,
        nodata=dem.nodata,
        bounds=dem.bounds,
        resolution=dem.resolution,
    )


def _write_geotiff(dem: DEM, path: str | Path) -> None:
    """将 DEM 写出为 GeoTIFF（用于 pysheds 中转）。"""
    path = Path(path)
    profile = {
        "driver": "GTiff",
        "height": dem.height,
        "width": dem.width,
        "count": 1,
        "dtype": "float64",
        "crs": dem.crs,
        "transform": dem.transform,
        "nodata": dem.nodata if dem.nodata is not None else -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        data = dem.data.copy()
        if dem.nodata is not None:
            data[np.isnan(data)] = dem.nodata
        else:
            data[np.isnan(data)] = -9999.0
        dst.write(data, 1)


def cell_area_m2(dem: DEM) -> np.ndarray:
    """每个栅格的近似面积（平方米），用于面积加权统计。"""
    xres, yres = dem.resolution
    return np.full(dem.shape, xres * yres, dtype=np.float64)
