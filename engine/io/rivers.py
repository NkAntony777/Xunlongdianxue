"""水系 / 道路矢量数据加载与查询。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import make_valid
from shapely.geometry import Point


def _clean_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """去掉空/无效几何，尽量 make_valid。"""
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf.copy()
    geoms = []
    keep = []
    for i, g in enumerate(gdf.geometry):
        if g is None or g.is_empty:
            continue
        try:
            if not g.is_valid:
                g = make_valid(g)
            if g is None or g.is_empty:
                continue
            # make_valid 可能变成 GeometryCollection，取最大线/面
            if g.geom_type == "GeometryCollection":
                parts = [p for p in g.geoms if p.geom_type in (
                    "LineString", "MultiLineString", "Polygon", "MultiPolygon"
                ) and not p.is_empty]
                if not parts:
                    continue
                from shapely.ops import unary_union
                g = unary_union(parts)
                if g.is_empty:
                    continue
            geoms.append(g)
            keep.append(i)
        except Exception:
            continue
    if not keep:
        return gpd.GeoDataFrame(columns=list(gdf.columns), crs=gdf.crs)
    out = gdf.iloc[keep].copy()
    out = out.set_geometry(geoms)
    return out


def _safe_representative_point(geom):
    """安全取代表点，避免 empty Point 崩溃。"""
    if geom is None or geom.is_empty:
        return None
    try:
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom is None or geom.is_empty:
            return None
        # 优先 representative_point（总在几何上）
        if hasattr(geom, "representative_point"):
            p = geom.representative_point()
            if p is not None and not p.is_empty:
                return p
        c = geom.centroid
        if c is not None and not c.is_empty:
            return c
        # 线：取中点坐标
        if geom.geom_type == "LineString" and len(geom.coords) >= 1:
            coords = list(geom.coords)
            mid = coords[len(coords) // 2]
            return Point(mid[0], mid[1])
        if geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                if not line.is_empty and len(line.coords) >= 1:
                    mid = list(line.coords)[len(line.coords) // 2]
                    return Point(mid[0], mid[1])
    except Exception:
        return None
    return None


@dataclass
class WaterNetwork:
    """水系网络（多边形或线）。

    Attributes:
        gdf: 原始 GeoDataFrame（任意 CRS）
        projected_gdf: 投影到 EPSG:3857 的副本（用于距离计算）
        total_bounds: (minx, miny, maxx, maxy)
    """

    gdf: gpd.GeoDataFrame
    projected_gdf: gpd.GeoDataFrame = field(init=False)
    total_bounds: tuple[float, float, float, float] = field(init=False)

    def __post_init__(self):
        if self.gdf is None:
            self.gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        if self.gdf.crs is None:
            # 根据坐标数值猜 CRS：|x|>180 视为投影米制
            try:
                xs = self.gdf.geometry.centroid.x
                if len(xs) and np.nanmax(np.abs(xs.astype(float))) > 180:
                    self.gdf = self.gdf.set_crs("EPSG:3857")
                else:
                    self.gdf = self.gdf.set_crs("EPSG:4326")
            except Exception:
                self.gdf = self.gdf.set_crs("EPSG:4326")

        self.gdf = _clean_gdf(self.gdf)
        if self.gdf.empty:
            self.projected_gdf = self.gdf.copy()
            self.total_bounds = (0.0, 0.0, 0.0, 0.0)
            return

        try:
            if "3857" in str(self.gdf.crs).upper():
                self.projected_gdf = _clean_gdf(self.gdf)
            else:
                self.projected_gdf = _clean_gdf(self.gdf.to_crs("EPSG:3857"))
        except Exception:
            self.projected_gdf = self.gdf.copy()

        if self.projected_gdf.empty:
            self.total_bounds = (0.0, 0.0, 0.0, 0.0)
        else:
            b = self.projected_gdf.total_bounds
            self.total_bounds = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    @property
    def empty(self) -> bool:
        return self.gdf is None or self.gdf.empty or self.projected_gdf.empty

    def _to_3857(self, x: float, y: float) -> Point:
        """输入坐标 → EPSG:3857。

        启发式：
          - |x| > 180 或 |y| > 90 → 已是投影坐标（米）
          - gdf 已是 3857 → 原样
          - 否则按 4326 转换
        """
        from pyproj import Transformer

        if abs(x) > 180 or abs(y) > 90:
            return Point(x, y)
        if self.gdf.crs and "3857" in str(self.gdf.crs).upper():
            return Point(x, y)
        try:
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xx, yy = transformer.transform(x, y)
            return Point(xx, yy)
        except Exception:
            return Point(x, y)

    def distance_to_nearest_m(self, x: float, y: float) -> float:
        """到最近水系要素的平面距离（米）。"""
        if self.empty:
            return float("inf")
        try:
            pt = self._to_3857(x, y)
            dists = self.projected_gdf.distance(pt)
            dists = dists[np.isfinite(dists)]
            if dists.empty:
                return float("inf")
            return float(dists.min())
        except Exception:
            return float("inf")

    def nearest_direction(self, x: float, y: float) -> str:
        """最近水系要素相对当前点的方位（八方位）。"""
        if self.empty:
            return "未识别"
        try:
            pt = self._to_3857(x, y)
            distances = self.projected_gdf.distance(pt)
            # 过滤 inf/nan
            valid = distances.replace([np.inf, -np.inf], np.nan).dropna()
            if valid.empty:
                return "未识别"
            idx = valid.idxmin()
            geom = self.projected_gdf.geometry.loc[idx]
            c = _safe_representative_point(geom)
            if c is None:
                return "未识别"
            dx = float(c.x) - float(pt.x)
            dy = float(c.y) - float(pt.y)
            if abs(dx) + abs(dy) < 1e-9:
                return "未识别"
            deg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
            names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
            return names[int(((deg + 22.5) % 360) / 45)]
        except Exception:
            return "未识别"

    def intersects(self, x: float, y: float, buffer_m: float = 0) -> bool:
        """当前点 + buffer 内是否与水系相交。"""
        if self.empty:
            return False
        try:
            pt = self._to_3857(x, y)
            if buffer_m > 0:
                pt = pt.buffer(buffer_m)
            return bool(self.projected_gdf.intersects(pt).any())
        except Exception:
            return False


def load_water(path: str | Path, layer: str | None = None) -> WaterNetwork:
    """加载水系数据（GeoJSON / Shapefile / GPKG）。"""
    path = Path(path)
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    gdf = _clean_gdf(gdf)
    return WaterNetwork(gdf=gdf)
