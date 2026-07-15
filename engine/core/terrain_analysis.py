"""地形分析：坡度、坡向、高差、粗糙度、地形位置。

参考：
  - Horn (1981) 坡度算法
  - TPI (Topographic Position Index) — Weiss 2001
  - shanshui-mingtang-fengshui-gis terrain_analysis.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter

from engine.io.dem import DEM


def _is_geographic(crs) -> bool:
    """判断 CRS 是否为地理坐标系（度）。"""
    if crs is None:
        return False
    try:
        from rasterio.crs import CRS
        c = CRS.from_user_input(crs) if not isinstance(crs, CRS) else crs
        return c.is_geographic
    except Exception:
        return str(crs).upper().endswith("4326") or "GEOG" in str(crs).upper()


def _radius_px(dem: DEM, radius_m: float) -> int:
    """根据 DEM 的 CRS 决定邻域半径（像素）。

    地理坐标（度）：按 1° ≈ 111000 m 估算。
    """
    xres, yres = dem.resolution
    if _is_geographic(dem.crs):
        m_per_deg = 111000.0
        ref = min(xres, yres) * m_per_deg
    else:
        ref = min(xres, yres)
    return max(1, int(round(radius_m / ref)))


@dataclass
class TerrainMetrics:
    """区域级地形统计量。"""

    mean_elevation: float
    max_elevation: float
    min_elevation: float
    relief: float
    mean_slope: float
    max_slope: float
    dominant_aspect: str
    aspect_degree: float
    terrain_position: str
    terrain_roughness: float
    slope: np.ndarray
    aspect: np.ndarray


def compute_slope_aspect(dem: DEM) -> tuple[np.ndarray, np.ndarray]:
    """Horn (1981) 3x3 邻域算法计算坡度（度）和坡向（度, 0-360, 北=0）。

    地理坐标系下把像元尺寸换算为米，避免 dz/d(degree) 导致坡度爆炸/失真。

    Returns:
        slope: 0-90 度
        aspect: 0-360 度（北=0, 东=90, 南=180, 西=270）
    """
    z = dem.data
    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    h, w = z.shape

    # 地理 CRS：度 → 米
    if _is_geographic(dem.crs):
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        xres_m = xres * 111_320.0 * cos_lat
        yres_m = yres * 111_320.0
    else:
        xres_m, yres_m = xres, yres
    xres_m = max(xres_m, 1e-3)
    yres_m = max(yres_m, 1e-3)

    # 邻域值（边界用 reflect padding）
    pad = np.pad(z, 1, mode="edge")
    # dz/dx 列方向（东为正）
    dz_dx = (
        (pad[0:-2, 2:] + 2 * pad[1:-1, 2:] + pad[2:, 2:])
        - (pad[0:-2, 0:-2] + 2 * pad[1:-1, 0:-2] + pad[2:, 0:-2])
    ) / (8.0 * xres_m)
    # dz/dy 行方向（北为正 = 行减小的方向）
    dz_dy = (
        (pad[0:-2, 0:-2] + 2 * pad[0:-2, 1:-1] + pad[0:-2, 2:])
        - (pad[2:, 0:-2] + 2 * pad[2:, 1:-1] + pad[2:, 2:])
    ) / (8.0 * yres_m)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    aspect_rad = np.arctan2(dz_dx, -dz_dy)
    aspect_deg = (np.degrees(aspect_rad) + 360) % 360
    # 平地（坡度<0.5°）的坡向设为 NaN
    flat = slope_deg < 0.5
    aspect_deg = np.where(flat, np.nan, aspect_deg)
    return slope_deg, aspect_deg


def tpi(dem: DEM, radius_m: float = 100.0) -> np.ndarray:
    """Topographic Position Index = z0 - mean(邻域)。

    Args:
        radius_m: 邻域半径（米）
    """
    r = _radius_px(dem, radius_m)
    size = 2 * r + 1
    mean_z = uniform_filter(dem.data, size=size, mode="reflect")
    return dem.data - mean_z


def terrain_roughness(dem: DEM, radius_m: float = 50.0) -> np.ndarray:
    """地形粗糙度 = 邻域高程标准差 / 邻域高程均值。

    使用均值-平方均值法计算局部标准差，避免 generic_filter 的 Python 循环。
    """
    r = _radius_px(dem, radius_m)
    size = 2 * r + 1
    data = dem.data
    mean = uniform_filter(data, size=size, mode="reflect")
    sq_mean = uniform_filter(data ** 2, size=size, mode="reflect")
    var = np.maximum(0, sq_mean - mean ** 2)
    std = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mean != 0, std / mean, 0.0)


def aspect_name(deg: float) -> str:
    """角度 → 八方位名。"""
    if np.isnan(deg):
        return "无主向"
    names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    return names[int(((deg + 22.5) % 360) / 45)]


def analyze_terrain(
    dem: DEM,
    mask: np.ndarray | None = None,
    radius_m: float = 100.0,
) -> TerrainMetrics:
    """计算区域级地形统计量。

    Args:
        dem: DEM 数据
        mask: 有效像元掩膜（True=有效），None 时使用 mask_valid()
        radius_m: TPI / 粗糙度邻域半径（米）
    """
    if mask is None:
        mask = dem.mask_valid()
    valid = dem.data[mask]
    if valid.size == 0:
        # E.5：全 NaN / 无有效像元 → 中性占位，不抛异常（调用方可再判 empty）
        z = np.zeros_like(dem.data, dtype=np.float64)
        nan_a = np.full_like(dem.data, np.nan, dtype=np.float64)
        return TerrainMetrics(
            mean_elevation=float("nan"),
            max_elevation=float("nan"),
            min_elevation=float("nan"),
            relief=0.0,
            mean_slope=0.0,
            max_slope=0.0,
            dominant_aspect="未识别",
            aspect_degree=float("nan"),
            terrain_position="无效DEM",
            terrain_roughness=0.0,
            slope=z,
            aspect=nan_a,
        )

    slope, aspect = compute_slope_aspect(dem)
    slope_valid = slope[mask]
    aspect_valid = aspect[mask & ~np.isnan(aspect)]

    relief = float(np.nanmax(valid) - np.nanmin(valid))
    mean_slope = float(np.nanmean(slope_valid))
    max_slope = float(np.nanmax(slope_valid))
    median_aspect = float(np.nanmedian(aspect_valid)) if aspect_valid.size else float("nan")
    roughness_arr = terrain_roughness(dem, radius_m=radius_m)
    mean_roughness = float(np.nanmean(roughness_arr[mask]))

    if mean_slope < 10:
        position = "缓坡台地"
    elif mean_slope < 20:
        position = "丘陵区"
    elif mean_slope < 35:
        position = "山地区"
    else:
        position = "高山区"

    return TerrainMetrics(
        mean_elevation=float(np.nanmean(valid)),
        max_elevation=float(np.nanmax(valid)),
        min_elevation=float(np.nanmin(valid)),
        relief=relief,
        mean_slope=mean_slope,
        max_slope=max_slope,
        dominant_aspect=aspect_name(median_aspect),
        aspect_degree=median_aspect,
        terrain_position=position,
        terrain_roughness=mean_roughness,
        slope=slope,
        aspect=aspect,
    )


def aspect_to_unit_vector(aspect_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """坡向角度 → 单位向量 (dx, dy)，用于平面曲率 / 凹凸性分析。"""
    rad = np.deg2rad(aspect_deg)
    dx = np.sin(rad)
    dy = np.cos(rad)
    return dx, dy
