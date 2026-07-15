"""可视化渲染：分图层渲染，生成与参考图风格一致的可叠加 PNG。

输出格式：JSON (栅格数据用 base64 PNG) 或 直接 PNG 字节。

参考图风格：
  - 底图：白色背景 + 浅灰蓝 hypsometric + 极淡 hillshade
  - 等高线：细灰线，间距 30m，标高 6 号字
  - 水系：muted steel-blue (#6e92a8) 软填充 + 蓝色描边
  - 水煞影响带：水系附近圆点紫灰色（#9c8aa0）半透明
  - 砂层（可建城片区）：tan/khaki 圆点（#b89568）半透明
  - 评分热力：紫 (#8a6cb5) → 橙 (#e07a32) 径向渐变（与参考图一致）
  - 四象：小圆点 + 中文标签
  - 龙脉：细黑实线
  - 候选穴：白底红描边圆圈 + 中文标签
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")
# 让 matplotlib 支持中文（系统已安装 Noto Sans SC / Microsoft YaHei / SimHei）
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "Noto Sans SC", "SimHei", "PingFang SC",
    "Hiragino Sans GB", "WenQuanYi Micro Hei", "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np

try:
    from matplotlib import colormaps as _cmap_registry
    def _get_cmap(name):
        return _cmap_registry[name]
except ImportError:
    from matplotlib import cm as _cm
    def _get_cmap(name):
        return _cm.get_cmap(name)


# ========== 颜色与样式常量（参考图配色） ==========
WATER_FILL = "#6e8fad"        # 水系填充：steel-blue（参考图）
WATER_EDGE = "#4a6d8a"        # 水系描边
WATER_INFLUENCE = "#9b8bb0"   # 水煞影响带：紫灰色圆点
SAND_FILL = "#c4a06a"         # 砂层/可建城片区：tan 方点
SAND_EDGE = "#8c6a3e"
SCORE_LOW = "#8a6cb5"         # 评分低端：紫
SCORE_HIGH = "#e07a32"        # 评分高端：橙
GROUND_LOW = "#f3f4f5"        # 底图低端色：极淡灰
GROUND_HIGH = "#d3d8de"       # 底图高端色：浅 steel-grey
CONTOUR_COLOR = "#6a737c"     # 等高线
BUILDING_OVERLAY = "#c5a572"  # 可建城实际覆盖片区
TEXT_COLOR = "#2b323a"


def _is_geo_bounds(bounds: tuple[float, float, float, float]) -> bool:
    """根据 bbox 跨度粗判是否为经纬度（度）。"""
    minx, miny, maxx, maxy = bounds
    return (maxx - minx) < 20 and (maxy - miny) < 20 and abs(minx) <= 180 and abs(maxx) <= 180


def _unit_scale_m(bounds: tuple[float, float, float, float]) -> float:
    """1 个 CRS 单位对应多少米（地理坐标按纬度近似）。"""
    if _is_geo_bounds(bounds):
        return 111_000.0
    return 1.0


def _meters_to_units(meters: float, bounds: tuple[float, float, float, float]) -> float:
    return meters / _unit_scale_m(bounds)


@dataclass
class RenderResult:
    """单帧渲染结果。"""

    width: int
    height: int
    bbox: tuple[float, float, float, float]  # minx, miny, maxx, maxy (in CRS units)
    png_base64: str
    legend: dict[str, Any] | None = None
    geojson: dict[str, Any] | None = None  # for vector layers


def _figsize_for(bounds: tuple[float, float, float, float],
                 base: float = 10.0) -> tuple[float, float]:
    """按 bbox 宽高比生成 figsize，避免图层被拉伸变形。"""
    minx, miny, maxx, maxy = bounds
    w = max(maxx - minx, 1e-12)
    h = max(maxy - miny, 1e-12)
    ar = w / h
    if ar >= 1:
        return (base, base / ar)
    return (base * ar, base)


def _to_png_base64(fig, dpi: int = 100) -> str:
    """导出 PNG。强制 axes 铺满 figure，禁止 tight crop，保证各图层像素对齐。"""
    import matplotlib.pyplot as plt
    # 各图层必须同尺寸、同 extent，否则前端 object-fit 叠加会错位
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for ax in fig.get_axes():
        ax.set_position([0, 0, 1, 1])
        # 用 auto 填满画布（bounds 已控制地理范围）
        try:
            ax.set_aspect("auto")
        except Exception:
            pass
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi,
        bbox_inches=None, pad_inches=0,
        facecolor="none", transparent=True,
    )
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _hillshade(dem_data: np.ndarray, azimuth: float = 315.0,
               altitude: float = 45.0, dx: float | None = None,
               dy: float | None = None) -> np.ndarray:
    """计算简易 hillshade（0-1 灰度）。"""
    data = dem_data.copy()
    valid = np.isfinite(data)
    fill = np.nanmean(data) if valid.any() else 0
    filled = np.where(valid, data, fill)
    if dx is None:
        dx, dy = 1.0, 1.0
    az = np.radians(azimuth)
    alt = np.radians(altitude)
    # 梯度
    gy, gx = np.gradient(filled, dy, dx)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    intensity = (np.sin(alt) * np.cos(slope)
                 + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    intensity = np.clip(intensity, 0, 1)
    intensity[~valid] = np.nan
    return intensity


def _hillshade_rgb(dem_data: np.ndarray) -> np.ndarray:
    """返回 (H,W,3) RGB hillshade 数组 (0-1)。"""
    h1 = _hillshade(dem_data, azimuth=315, altitude=45)
    h2 = _hillshade(dem_data, azimuth=225, altitude=30) * 0.4
    h3 = _hillshade(dem_data, azimuth=90, altitude=20) * 0.2
    rgb = (h1 + h2 + h3)
    rgb = (rgb - np.nanmin(rgb)) / max(np.nanmax(rgb) - np.nanmin(rgb), 1e-9)
    rgb = np.clip(rgb, 0, 1)
    rgb = np.where(np.isfinite(rgb), rgb, np.nan)
    return rgb


def _extent_for(dem) -> list[float]:
    """matplotlib imshow extent = [left, right, bottom, top]"""
    b = dem.bounds
    return [b[0], b[2], b[1], b[3]]


# ========== 底图：hypsometric + 轻 hillshade ==========

def render_basemap(
    dem,
    colormap: str = "Greys",
    figsize: tuple[float, float] | None = None,
    dpi: int = 110,
) -> RenderResult:
    """基础底图：极淡 hypsometric tint + 轻 hillshade，输出透明 PNG。"""
    data = dem.data.copy()
    valid = np.isfinite(data)
    if not valid.any():
        raise ValueError("DEM has no valid pixels")
    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    h, w = data.shape
    if figsize is None:
        figsize = _figsize_for(dem.bounds)

    # 构建 (H,W,4) RGBA 数组
    norm = np.clip((data - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    base_rgb = np.full((h, w, 3), 1.0, dtype=np.float64)
    base_rgb[..., 0] = 1 - 0.05 * norm  # 微暗
    base_rgb[..., 1] = 1 - 0.06 * norm
    base_rgb[..., 2] = 1 - 0.08 * norm

    # hillshade 叠层（在 base_rgb 之上）
    shade = _hillshade_rgb(data)
    shade_val = np.where(np.isfinite(shade), shade, 0.5)

    # 让 hillshade 调制 base_rgb（在阴影处稍微加深，高光处稍微提亮）
    base_rgb = base_rgb * (0.85 + 0.30 * shade_val[..., None])
    base_rgb = np.clip(base_rgb, 0, 1)

    rgba = np.dstack([base_rgb, np.where(valid, 1.0, 0.0).astype(np.float64)])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("white")
    ax.imshow(rgba, extent=_extent_for(dem), origin="upper", aspect="equal",
              interpolation="bilinear")
    ax.set_xlim(dem.bounds[0], dem.bounds[2])
    ax.set_ylim(dem.bounds[1], dem.bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return RenderResult(
        width=w, height=h,
        bbox=dem.bounds,
        png_base64=_to_png_base64(fig, dpi=dpi),
        legend={"vmin": vmin, "vmax": vmax, "colormap": colormap,
                "type": "basemap"},
    )


# ========== 底图：坡度图（用于切换底图） ==========

def render_slope_basemap(
    dem,
    figsize: tuple[float, float] | None = None,
    dpi: int = 110,
) -> RenderResult:
    """坡度底图。"""
    from engine.core.terrain_analysis import compute_slope_aspect
    slope, _ = compute_slope_aspect(dem)
    data = slope
    valid = np.isfinite(data)
    if not valid.any():
        raise ValueError("no valid pixels")
    vmax = float(np.nanpercentile(data, 98))
    norm = np.clip(data / max(vmax, 1e-3), 0, 1)
    if figsize is None:
        figsize = _figsize_for(dem.bounds)

    rgba = _get_cmap("YlOrBr")(norm)
    rgba[..., 3] = np.where(valid, 0.85, 0.0)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("white")
    ax.imshow(rgba, extent=_extent_for(dem), origin="upper", aspect="equal",
              interpolation="bilinear")
    ax.set_xlim(dem.bounds[0], dem.bounds[2])
    ax.set_ylim(dem.bounds[1], dem.bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    h, w = data.shape
    return RenderResult(
        width=w, height=h, bbox=dem.bounds,
        png_base64=_to_png_base64(fig, dpi=dpi),
        legend={"vmin": 0, "vmax": vmax, "type": "slope_basemap"},
    )


# ========== 等高线（PNG + GeoJSON） ==========

def render_contours(
    dem,
    contour_interval: float = 30.0,
    figsize: tuple[float, float] | None = None,
    dpi: int = 110,
) -> RenderResult:
    """渲染等高线图层（透明背景上的细灰线 + 标高）。"""
    data = dem.data.copy()
    valid = np.isfinite(data)
    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    h, w = data.shape
    if figsize is None:
        figsize = _figsize_for(dem.bounds)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("none")
    levels = np.arange(np.floor(vmin / contour_interval) * contour_interval,
                       vmax + contour_interval, contour_interval)
    xs = np.linspace(dem.bounds[0], dem.bounds[2], w)
    ys = np.linspace(dem.bounds[1], dem.bounds[3], h)
    geo_features: list[dict] = []
    try:
        cs = ax.contour(
            xs, ys, np.flipud(data),
            levels=levels, colors=CONTOUR_COLOR, linewidths=0.35, alpha=0.85,
        )
        ax.clabel(cs, inline=True, fontsize=6, fmt="%d", colors="#5c6770")

        # 把每条 contour 序列化为 GeoJSON LineString
        # matplotlib 返回的 collection 是按 level 分组的，每个里有若干 segments
        if hasattr(cs, "allsegs"):
            for level_idx, level in enumerate(levels):
                for seg in cs.allsegs[level_idx]:
                    if len(seg) < 2:
                        continue
                    geo_features.append({
                        "type": "Feature",
                        "properties": {"elevation": float(level)},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": seg.tolist(),
                        },
                    })
    except Exception:
        pass
    ax.set_xlim(dem.bounds[0], dem.bounds[2])
    ax.set_ylim(dem.bounds[1], dem.bounds[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    return RenderResult(
        width=w, height=h, bbox=dem.bounds,
        png_base64=_to_png_base64(fig, dpi=dpi),
        legend={"vmin": vmin, "vmax": vmax,
                "contour_interval": contour_interval, "type": "contours"},
        geojson={"type": "FeatureCollection", "features": geo_features},
    )


# ========== 水系 ==========

def _water_rank(row) -> str:
    """水系等级：major / minor / pond，用于参考图风格分层渲染。

    地理坐标下 length/area 为「度」，需换算到米再分级，否则主河会被误判为支流。
    """
    name = str(row.get("name") or row.get("NAME") or "")
    ww = str(row.get("waterway") or row.get("fclass") or row.get("type") or "")
    nat = str(row.get("natural") or "")
    geom = row.geometry
    if nat == "water" or (geom is not None and geom.geom_type in ("Polygon", "MultiPolygon")):
        try:
            area = float(geom.area)
        except Exception:
            area = 0.0
        # 度² → 粗略 m²；投影 CRS 下 area 已是 m²
        area_m2 = area * (111_000.0 ** 2) if area < 2.0 else area
        return "pond" if area_m2 < 5e5 else "major"
    if ww in ("river", "riverbank", "canal", "water") or any(
        k in name for k in ("江", "河", "嘉陵", "湖", "溪")
    ):
        return "major"
    if ww in ("stream", "drain", "ditch", "brook"):
        return "minor"
    try:
        length = float(geom.length) if geom is not None else 0.0
    except Exception:
        length = 0.0
    length_m = length * 111_000.0 if length < 2.0 else length
    return "major" if length_m > 1500 else "minor"


def render_water(
    water_gdf,
    bounds: tuple[float, float, float, float],
    figsize: tuple[float, float] | None = None,
    dpi: int = 110,
    width_m: float = 260.0,
) -> RenderResult:
    """水系图层：参考图风格——主河道宽蓝带 + 支流细线。"""
    import matplotlib.pyplot as plt

    minx, miny, maxx, maxy = bounds
    if figsize is None:
        figsize = _figsize_for(bounds)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("none")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    if water_gdf is None or water_gdf.empty:
        return RenderResult(
            width=100, height=100, bbox=bounds,
            png_base64=_to_png_base64(fig, dpi=dpi),
            legend={"type": "water", "count": 0},
        )

    half_major = _meters_to_units(width_m, bounds)
    half_minor = _meters_to_units(max(40.0, width_m * 0.22), bounds)
    n = 0

    majors, minors, ponds = [], [], []
    for _, row in water_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        rank = _water_rank(row)
        if rank == "major":
            majors.append(geom)
        elif rank == "pond":
            ponds.append(geom)
        else:
            minors.append(geom)

    # 支流细线
    for geom in minors:
        try:
            if geom.geom_type == "LineString":
                xs, ys = geom.xy
                ax.plot(xs, ys, color=WATER_FILL, linewidth=1.2,
                        solid_capstyle="round", alpha=0.45, zorder=2)
            elif geom.geom_type == "MultiLineString":
                for line in geom.geoms:
                    xs, ys = line.xy
                    ax.plot(xs, ys, color=WATER_FILL, linewidth=1.2,
                            solid_capstyle="round", alpha=0.45, zorder=2)
            n += 1
        except Exception:
            continue

    # 水塘（过滤过小）
    min_pond = _meters_to_units(100, bounds) ** 2
    for geom in ponds:
        try:
            if float(geom.area) < min_pond:
                continue
            ax.add_collection(_poly_collection(
                [geom], facecolor=WATER_FILL, edgecolor=WATER_EDGE,
                alpha=0.55, linewidth=0.4,
            ))
            n += 1
        except Exception:
            continue

    # 主河道：先 dissolve 再 buffer，避免 OSM 多段重叠成双线
    if majors:
        try:
            from shapely.ops import unary_union, linemerge
            merged = unary_union(majors)
            # 尽量把 MultiLineString 合并
            try:
                if merged.geom_type == "MultiLineString":
                    merged = linemerge(merged)
            except Exception:
                pass
            soft = merged.buffer(half_major * 1.2, cap_style=1, join_style=1)
            buf = merged.buffer(half_major, cap_style=1, join_style=1)
            if not soft.is_empty:
                ax.add_collection(_poly_collection(
                    [soft], facecolor=WATER_FILL, edgecolor="none", alpha=0.32,
                ))
            if not buf.is_empty:
                ax.add_collection(_poly_collection(
                    [buf], facecolor=WATER_FILL, edgecolor=WATER_EDGE,
                    alpha=0.85, linewidth=0.5,
                ))
                n += 1
        except Exception:
            for geom in majors:
                try:
                    buf = geom.buffer(half_major, cap_style=1, join_style=1)
                    if not buf.is_empty:
                        ax.add_collection(_poly_collection(
                            [buf], facecolor=WATER_FILL, edgecolor=WATER_EDGE,
                            alpha=0.8, linewidth=0.5,
                        ))
                        n += 1
                except Exception:
                    continue

    return RenderResult(
        width=int((maxx - minx) / 30) or 100,
        height=int((maxy - miny) / 30) or 100,
        bbox=bounds,
        png_base64=_to_png_base64(fig, dpi=dpi),
        legend={"type": "water", "count": n, "width_m": width_m},
    )


def _poly_collection(geoms, **kwargs):
    from matplotlib.collections import PolyCollection
    polys = []
    for g in geoms:
        if g.geom_type == "Polygon":
            polys.append(list(g.exterior.coords))
            for hole in g.interiors:
                polys.append(list(hole.coords))
        elif g.geom_type == "MultiPolygon":
            for p in g.geoms:
                polys.append(list(p.exterior.coords))
    return PolyCollection(polys, **kwargs)


# ========== 水煞影响带 ==========

def render_water_influence(
    water_gdf,
    bounds: tuple[float, float, float, float],
    buffer_m: float | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 110,
    dem=None,
) -> RenderResult:
    """水系「水煞影响带」紫灰点阵 — 非「近水即满煞」。

    - 无 DEM：窄岸缘条带（割脚示意），非 km 级填充
    - 有 DEM：采样点按 距离×海拔衰减×流向冲 调 alpha/是否绘制
      （与 water_sha_influence_risk 一致；玉带堂心/高台淡出）

    dem: 可选 DEM（与 water_gdf 同 CRS 时海拔/流向最准）
    buffer_m: 兼容旧参数；有 DEM 时作采样外缘上限，无 DEM 时覆盖默认窄带。
    """
    import matplotlib.pyplot as plt
    from shapely.geometry import Point as SPoint
    from shapely.ops import unary_union

    from engine.core.water_sha_influence import (
        INFLUENCE_BANK_MAJOR_M,
        INFLUENCE_BANK_MINOR_M,
        INFLUENCE_SAMPLE_MAJOR_M,
        INFLUENCE_SAMPLE_MINOR_M,
        INFLUENCE_DRAW_MIN,
        water_sha_influence_at_xy,
        water_sha_dist_risk,
    )

    minx, miny, maxx, maxy = bounds
    if figsize is None:
        figsize = _figsize_for(bounds)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("none")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    if water_gdf is None or water_gdf.empty:
        return RenderResult(
            width=100, height=100, bbox=bounds,
            png_base64=_to_png_base64(fig, dpi=dpi),
            legend={"type": "water_influence", "count": 0, "mode": "empty"},
        )

    has_dem = dem is not None and getattr(dem, "data", None) is not None
    # 窄岸默认；buffer_m 仅作上限/兼容旧宽缓冲请求时夹紧
    if buffer_m is not None and float(buffer_m) > 0:
        major_m = min(float(buffer_m), INFLUENCE_SAMPLE_MAJOR_M if has_dem else INFLUENCE_BANK_MAJOR_M)
        minor_m = min(float(buffer_m) * 0.55, INFLUENCE_SAMPLE_MINOR_M if has_dem else INFLUENCE_BANK_MINOR_M)
    else:
        major_m = INFLUENCE_SAMPLE_MAJOR_M if has_dem else INFLUENCE_BANK_MAJOR_M
        minor_m = INFLUENCE_SAMPLE_MINOR_M if has_dem else INFLUENCE_BANK_MINOR_M

    buf_major = _meters_to_units(major_m, bounds)
    buf_minor = _meters_to_units(minor_m, bounds)
    spacing = _meters_to_units(55.0 if has_dem else 70.0, bounds)
    if spacing <= 0:
        spacing = (maxx - minx) / 100.0

    majors, minors = [], []
    for _, row in water_gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        rank = _water_rank(row)
        if rank == "minor":
            minors.append(g)
        else:
            majors.append(g)
    if not majors and not minors:
        return RenderResult(
            width=100, height=100, bbox=bounds,
            png_base64=_to_png_base64(fig, dpi=dpi),
            legend={"type": "water_influence", "count": 0, "mode": "empty"},
        )

    parts = []
    try:
        if majors:
            parts.append(unary_union(majors).buffer(buf_major))
        if minors:
            parts.append(unary_union(minors).buffer(buf_minor))
        merged = unary_union(parts) if parts else None
    except Exception:
        merged = None

    if merged is None or merged.is_empty:
        return RenderResult(
            width=100, height=100, bbox=bounds,
            png_base64=_to_png_base64(fig, dpi=dpi),
            legend={"type": "water_influence", "count": 0, "mode": "empty"},
        )

    try:
        from shapely.geometry import box as shapely_box
        merged = merged.intersection(shapely_box(minx, miny, maxx, maxy))
    except Exception:
        pass
    if merged is None or merged.is_empty:
        return RenderResult(
            width=100, height=100, bbox=bounds,
            png_base64=_to_png_base64(fig, dpi=dpi),
            legend={"type": "water_influence", "count": 0, "mode": "empty"},
        )

    minxg, minyg, maxxg, maxyg = merged.bounds
    minxg = max(minxg, minx)
    minyg = max(minyg, miny)
    maxxg = min(maxxg, maxx)
    maxyg = min(maxyg, maxy)

    xs = np.arange(minxg, maxxg + spacing * 0.5, spacing)
    ys = np.arange(minyg, maxyg + spacing * 0.5, spacing)
    if len(xs) == 0 or len(ys) == 0:
        return RenderResult(
            width=100, height=100, bbox=bounds,
            png_base64=_to_png_base64(fig, dpi=dpi),
            legend={"type": "water_influence", "count": 0, "mode": "empty"},
        )
    if len(xs) * len(ys) > 50_000:
        step = int(np.ceil(np.sqrt(len(xs) * len(ys) / 50_000.0)))
        xs = xs[::step]
        ys = ys[::step]

    xx, yy = np.meshgrid(xs, ys)
    pts_x, pts_y, alphas, sizes = [], [], [], []
    n_cand = 0
    try:
        from shapely import prepared
        prep = prepared.prep(merged)
        candidates = [
            (float(px), float(py))
            for px, py in zip(xx.ravel(), yy.ravel())
            if prep.contains(SPoint(px, py))
        ]
    except Exception:
        polys = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
        candidates = []
        for px, py in zip(xx.ravel(), yy.ravel()):
            p = SPoint(float(px), float(py))
            if any(poly.contains(p) for poly in polys):
                candidates.append((float(px), float(py)))

    n_cand = len(candidates)
    # 点数过多时降采样风险计算
    if n_cand > 12_000:
        stride = int(np.ceil(n_cand / 12_000.0))
        candidates = candidates[::stride]

    raw_water = None
    if not has_dem:
        try:
            raw_water = unary_union(majors + minors)
        except Exception:
            raw_water = None

    for px, py in candidates:
        if has_dem:
            risk = water_sha_influence_at_xy(px, py, water_gdf, dem=dem)
        else:
            # 无 DEM：仅距离窄带（割脚示意），非 km 填充
            d_m = 40.0
            try:
                from shapely.geometry import Point as P2
                if raw_water is not None:
                    d_m = float(P2(px, py).distance(raw_water))
            except Exception:
                pass
            risk = water_sha_dist_risk(d_m)
        if risk < INFLUENCE_DRAW_MIN:
            continue
        pts_x.append(px)
        pts_y.append(py)
        # alpha: 0.18–0.72 by risk
        alphas.append(float(0.18 + 0.54 * risk))
        sizes.append(float(8.0 + 10.0 * risk))

    if pts_x:
        # scatter with per-point alpha via rgba
        from matplotlib.colors import to_rgba
        base = to_rgba(WATER_INFLUENCE)
        colors = [(base[0], base[1], base[2], a) for a in alphas]
        ax.scatter(
            pts_x, pts_y, s=sizes, c=colors, marker="s",
            linewidths=0, zorder=4,
        )

    return RenderResult(
        width=int((maxx - minx) / 30) or 100,
        height=int((maxy - miny) / 30) or 100,
        bbox=bounds,
        png_base64=_to_png_base64(fig, dpi=dpi),
        legend={
            "type": "water_influence",
            "buffer_m_major": major_m,
            "buffer_m_minor": minor_m,
            "count": len(pts_x),
            "n_candidates": n_cand,
            "n_major": len(majors),
            "n_minor": len(minors),
            "mode": "dem_risk" if has_dem else "narrow_bank",
            "draw_min": INFLUENCE_DRAW_MIN,
        },
    )


# ========== 评分热力 (紫→橙) ==========

def render_score_grid(
    dem,
    score_grid: np.ndarray,
    figsize: tuple[float, float] | None = None,
    dpi: int = 110,
    alpha: float = 0.72,
) -> RenderResult:
    """渲染风水评分热力：平滑评分场本身（橙心 ≡ find_score_peak）。"""
    if score_grid.shape != dem.data.shape:
        raise ValueError(f"score_grid shape {score_grid.shape} != DEM shape {dem.data.shape}")

    h, w = dem.data.shape
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if figsize is None:
        figsize = _figsize_for(dem.bounds)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("none")

    valid = np.isfinite(score_grid)
    if not valid.any():
        return RenderResult(
            width=w, height=h, bbox=dem.bounds,
            png_base64=_to_png_base64(fig, dpi=dpi),
            legend={"type": "score"},
        )

    # 与穴心取峰共用同一平滑场 → 所见即所得
    from engine.core.four_beasts_detect import find_score_peak, smooth_score_field

    soft_field, _sigma = smooth_score_field(score_grid)
    peak = find_score_peak(score_grid)
    if peak is None:
        filled = np.where(np.isfinite(soft_field), soft_field, -np.inf)
        pr, pc = np.unravel_index(np.nanargmax(filled), filled.shape)
        peak_val = float(filled[pr, pc])
    else:
        pr, pc, peak_val = peak
        pr, pc = int(pr), int(pc)

    soft_valid = np.isfinite(soft_field)
    if soft_valid.any():
        smax = float(np.nanpercentile(soft_field[soft_valid], 97))
        smin = float(np.nanpercentile(soft_field[soft_valid], 55))
    else:
        smax, smin = 1.0, 0.0

    # 分位拉伸：高分暖橙，低分透明紫
    norm = np.clip((soft_field - smin) / max(smax - smin, 1e-6), 0, 1)
    norm = np.where(soft_valid, norm, 0.0)
    # 压掉外围杂色，保留场心
    vis = norm ** 1.05
    vis = vis * (vis > 0.12)
    # 轻微高斯仅用于抗锯齿（不改变峰位：sigma 很小）
    from scipy.ndimage import gaussian_filter
    vis = gaussian_filter(vis, sigma=max(0.6, min(h, w) / 200.0))
    vis = vis / max(float(vis.max()), 1e-9)
    vis = np.where(soft_valid, vis, 0.0)

    cmap = LinearSegmentedColormap.from_list(
        "xunlong_score", [
            (0.00, "#ffffff00"),
            (0.12, "#c4b0e000"),
            (0.30, "#8a6cb5"),
            (0.50, "#c45a7a"),
            (0.70, "#e07030"),
            (0.88, "#f08a28"),
            (1.00, "#ffc060"),
        ], N=512,
    )
    rgba = cmap(vis)
    rgba[..., 3] = np.where(soft_valid, alpha * np.clip(vis * 1.15, 0, 1), 0.0)

    ax.imshow(rgba, extent=_extent_for(dem), origin="upper", aspect="equal",
              interpolation="bilinear")
    ax.set_xlim(dem.bounds[0], dem.bounds[2])
    ax.set_ylim(dem.bounds[1], dem.bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    peak_xy = None
    try:
        peak_xy = dem.xy(int(pr), int(pc))
    except Exception:
        pass

    return RenderResult(
        width=w, height=h, bbox=dem.bounds,
        png_base64=_to_png_base64(fig, dpi=dpi),
        legend={"vmin": float(smin), "vmax": float(smax), "type": "score",
                "peak_xy": list(peak_xy) if peak_xy else None,
                "peak_score": float(peak_val) if np.isfinite(peak_val) else None,
                "peak_row": int(pr), "peak_col": int(pc)},
    )


# ========== 砂层 / 可建城片区 ==========

def render_buildings(
    sand_polys_gdf,
    bounds: tuple[float, float, float, float],
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 110,
) -> RenderResult:
    """可建城实际覆盖片区：tan/khaki 方点带（参考图风格）。"""
    import matplotlib.pyplot as plt
    from shapely.geometry import Point as SPoint

    minx, miny, maxx, maxy = bounds
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("none")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    if sand_polys_gdf is None or getattr(sand_polys_gdf, "empty", True):
        return RenderResult(
            width=100, height=100, bbox=bounds,
            png_base64=_to_png_base64(fig, dpi=dpi),
            legend={"type": "buildings"},
        )

    spacing = _meters_to_units(180.0, bounds)
    total = 0
    for _, row in sand_polys_gdf.iterrows():
        g = row.geometry
        if g is None:
            continue
        polys = [g] if g.geom_type == "Polygon" else list(g.geoms) \
            if g.geom_type == "MultiPolygon" else []
        for poly in polys:
            if not hasattr(poly, "exterior"):
                continue
            minxg, minyg, maxxg, maxyg = poly.bounds
            xx, yy = np.meshgrid(
                np.arange(minxg, maxxg, spacing),
                np.arange(minyg, maxyg, spacing),
            )
            inside_x, inside_y = [], []
            for px, py in zip(xx.ravel(), yy.ravel()):
                if poly.contains(SPoint(px, py)):
                    inside_x.append(px)
                    inside_y.append(py)
            if inside_x:
                ax.scatter(
                    inside_x, inside_y, s=11, c=SAND_FILL, marker="s",
                    alpha=0.62, linewidths=0, zorder=4,
                )
                total += len(inside_x)

    return RenderResult(
        width=int((maxx - minx) / 30) or 100,
        height=int((maxy - miny) / 30) or 100,
        bbox=bounds,
        png_base64=_to_png_base64(fig, dpi=dpi),
        legend={"type": "buildings", "count": total},
    )


def render_buildable_from_dem(
    dem,
    water_gdf=None,
    max_slope_deg: float = 12.0,
    max_elev_percentile: float = 55.0,
    figsize: tuple[float, float] | None = None,
    dpi: int = 110,
) -> RenderResult:
    """从 DEM 推导可建城片区：缓坡 + 中低海拔 + 近水廊道，参考图 tan 方点。

    不依赖外部建筑数据，纯地形规则。
    """
    import matplotlib.pyplot as plt
    from engine.core.terrain_analysis import compute_slope_aspect

    bounds = dem.bounds
    minx, miny, maxx, maxy = bounds
    if figsize is None:
        figsize = _figsize_for(bounds)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("none")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    slope, _ = compute_slope_aspect(dem)
    elev = dem.data
    valid = np.isfinite(elev) & np.isfinite(slope)
    if not valid.any():
        return RenderResult(
            width=dem.width, height=dem.height, bbox=bounds,
            png_base64=_to_png_base64(fig, dpi=dpi),
            legend={"type": "buildings", "source": "dem_rule", "count": 0},
        )

    elev_cut = float(np.nanpercentile(elev[valid], max_elev_percentile))
    mask = valid & (slope <= max_slope_deg) & (elev <= elev_cut)

    # 排除水系本身（近岸缓冲内不画建城区点）
    water_u = None
    if water_gdf is not None and not getattr(water_gdf, "empty", True):
        try:
            from shapely.ops import unary_union
            geoms = [g for g in water_gdf.geometry if g is not None and not g.is_empty]
            if geoms:
                water_u = unary_union(geoms)
        except Exception:
            water_u = None

    # 点阵采样（参考图约 220–260m 一方点，更疏、更像"可建城片区"）
    step_m = 240.0
    xres = abs(dem.resolution[0])
    yres = abs(dem.resolution[1])
    unit_m = 111_000.0 if _is_geo_bounds(bounds) else 1.0
    step_px_x = max(2, int(round(step_m / max(xres * unit_m, 1e-9))))
    step_px_y = max(2, int(round(step_m / max(yres * unit_m, 1e-9))))

    # 仅保留连通大片区（去掉零星噪点）
    try:
        from scipy.ndimage import binary_opening, binary_closing, label
        mask = binary_closing(mask, structure=np.ones((3, 3)))
        mask = binary_opening(mask, structure=np.ones((5, 5)))
        labeled, nlab = label(mask)
        if nlab > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            keep = sizes >= max(20, int(0.004 * mask.size))
            mask = keep[labeled]
    except Exception:
        pass

    pts_x, pts_y = [], []
    h, w = elev.shape
    river_buf_u = _meters_to_units(120.0, bounds)
    # 准备 STRtree 加速距离判断
    water_prep = None
    if water_u is not None:
        try:
            from shapely import prepared
            water_prep = prepared.prep(water_u.buffer(river_buf_u))
        except Exception:
            water_prep = None

    for r in range(step_px_y // 2, h, step_px_y):
        for c in range(step_px_x // 2, w, step_px_x):
            if not mask[r, c]:
                continue
            x, y = dem.xy(r, c)
            if water_prep is not None:
                try:
                    from shapely.geometry import Point as SPoint
                    if water_prep.contains(SPoint(x, y)):
                        continue
                except Exception:
                    pass
            elif water_u is not None:
                try:
                    from shapely.geometry import Point as SPoint
                    if water_u.distance(SPoint(x, y)) < river_buf_u:
                        continue
                except Exception:
                    pass
            pts_x.append(x)
            pts_y.append(y)

    if pts_x:
        ax.scatter(
            pts_x, pts_y, s=14, c=SAND_FILL, marker="s",
            alpha=0.55, linewidths=0, zorder=4,
        )

    return RenderResult(
        width=dem.width, height=dem.height, bbox=bounds,
        png_base64=_to_png_base64(fig, dpi=dpi),
        legend={
            "type": "buildings",
            "source": "dem_rule",
            "count": len(pts_x),
            "max_slope_deg": max_slope_deg,
        },
    )


# ========== 矢量图层（直接输出 GeoJSON） ==========

def four_beasts_geojson(four_beasts: dict) -> dict:
    """四象 → GeoJSON FeatureCollection。"""
    feats = []
    style_map = {
        "shaozu":   {"label": "少祖", "color": "#34495e", "size": 100},
        "xuanwu":   {"label": "玄武", "color": "#1a1a1a", "size": 120},
        "qinglong": {"label": "青龙", "color": "#16a085", "size": 90},
        "baihu":    {"label": "白虎", "color": "#d35400", "size": 90},
        "zhuque":   {"label": "朱雀", "color": "#c0392b", "size": 90},
    }
    for key, st in style_map.items():
        if key in four_beasts:
            p = four_beasts[key]
            feats.append({
                "type": "Feature",
                "properties": {"id": key, "label": st["label"],
                               "color": st["color"], "size": st["size"]},
                "geometry": {"type": "Point",
                             "coordinates": [p["x"], p["y"]]},
            })
    return {"type": "FeatureCollection", "features": feats}


def candidates_geojson(candidates: list[dict]) -> dict:
    feats = []
    for c in candidates:
        feats.append({
            "type": "Feature",
            "properties": {
                "id": c.get("id", "?"),
                "overall_score": c.get("overall_score", 0),
                "form_type": c.get("form_type", ""),
                "rank": c.get("rank", 0),
            },
            "geometry": {"type": "Point",
                         "coordinates": [c["x"], c["y"]]},
        })
    return {"type": "FeatureCollection", "features": feats}


def ridges_geojson(ridges: list[dict]) -> dict:
    feats = []
    for i, r in enumerate(ridges):
        coords = r.get("coords", [])
        if len(coords) < 2:
            continue
        if isinstance(coords[0], (list, tuple)) and len(coords[0]) == 2:
            props = {
                "id": i,
                "rank": r.get("rank", i + 1),
                "is_primary": bool(r.get("is_primary", False)),
            }
            feats.append({
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "LineString",
                             "coordinates": coords},
            })
    return {"type": "FeatureCollection", "features": feats}


# ========== 综合渲染（保留旧 API，调用各子图层） ==========

def render_combined(
    dem,
    four_beasts: dict | None = None,
    score_grid: np.ndarray | None = None,
    contour_interval: float = 30.0,
    ridges: list | None = None,
    candidates: list | None = None,
    water_gdf=None,
    sand_polys_gdf=None,
    water_influence: bool = True,
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 110,
) -> RenderResult:
    """综合渲染：叠合所有图层（一次性输出 PNG）。"""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("white")
    ax.set_xlim(dem.bounds[0], dem.bounds[2])
    ax.set_ylim(dem.bounds[1], dem.bounds[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # 1. 底图
    base = render_basemap(dem, dpi=dpi)
    import base64 as _b64
    from PIL import Image
    img = Image.open(io.BytesIO(_b64.b64decode(base.png_base64)))
    ax.imshow(np.asarray(img), extent=_extent_for(dem), origin="upper",
              aspect="equal", interpolation="bilinear")

    # 2. 水煞影响带（在水体下方，更柔和）
    if water_influence and water_gdf is not None:
        wi = render_water_influence(water_gdf, dem.bounds, dpi=dpi, dem=dem)
        img = Image.open(io.BytesIO(_b64.b64decode(wi.png_base64)))
        ax.imshow(np.asarray(img), extent=_extent_for(dem), origin="upper",
                  aspect="equal", interpolation="bilinear")

    # 3. 水系
    if water_gdf is not None:
        w = render_water(water_gdf, dem.bounds, dpi=dpi)
        img = Image.open(io.BytesIO(_b64.b64decode(w.png_base64)))
        ax.imshow(np.asarray(img), extent=_extent_for(dem), origin="upper",
                  aspect="equal", interpolation="bilinear")

    # 4. 砂层
    if sand_polys_gdf is not None:
        b = render_buildings(sand_polys_gdf, dem.bounds, dpi=dpi)
        img = Image.open(io.BytesIO(_b64.b64decode(b.png_base64)))
        ax.imshow(np.asarray(img), extent=_extent_for(dem), origin="upper",
                  aspect="equal", interpolation="bilinear")

    # 5. 评分热力
    if score_grid is not None and score_grid.shape == dem.data.shape:
        sg = render_score_grid(dem, score_grid, dpi=dpi)
        img = Image.open(io.BytesIO(_b64.b64decode(sg.png_base64)))
        ax.imshow(np.asarray(img), extent=_extent_for(dem), origin="upper",
                  aspect="equal", interpolation="bilinear")

    # 6. 等高线
    if contour_interval > 0:
        cs_layer = render_contours(dem, contour_interval, dpi=dpi)
        img = Image.open(io.BytesIO(_b64.b64decode(cs_layer.png_base64)))
        ax.imshow(np.asarray(img), extent=_extent_for(dem), origin="upper",
                  aspect="equal", interpolation="bilinear")

    # 7. 龙脉
    if ridges:
        for r in ridges[:5]:
            coords = r.get("coords", [])
            if len(coords) >= 2:
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                ax.plot(xs, ys, "-", color="#1a1a1a", linewidth=1.6,
                        alpha=0.85, zorder=6)

    # 8. 四象 + 9. 候选穴
    if four_beasts:
        style_map = {
            "shaozu":   ("少祖", "#34495e", 70),
            "xuanwu":   ("玄武", "#1a1a1a", 90),
            "qinglong": ("青龙", "#16a085", 60),
            "baihu":    ("白虎", "#d35400", 60),
            "zhuque":   ("朱雀", "#c0392b", 60),
        }
        for key, (label, color, size) in style_map.items():
            if key in four_beasts:
                x, y = four_beasts[key]["x"], four_beasts[key]["y"]
                ax.scatter([x], [y], s=size, c=color, edgecolors="white",
                           linewidths=1.5, zorder=10)
                ax.annotate(label, (x, y), xytext=(8, 4),
                            textcoords="offset points",
                            fontsize=10, color=TEXT_COLOR, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", ec="#bfc4ca", alpha=0.85,
                                      lw=0.5),
                            zorder=11)

    if candidates:
        for c in candidates:
            ax.scatter([c["x"]], [c["y"]], s=70, c="white",
                       edgecolors="#e74c3c", linewidths=1.8, zorder=12)
            ax.annotate(c.get("id", ""), (c["x"], c["y"]),
                        xytext=(10, 4), textcoords="offset points",
                        fontsize=9, color=TEXT_COLOR, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="white", ec="#e74c3c", alpha=0.85,
                                  lw=0.5),
                        zorder=13)

    h, w = dem.data.shape
    return RenderResult(
        width=w, height=h, bbox=dem.bounds,
        png_base64=_to_png_base64(fig, dpi=dpi),
        legend={"vmin": float(np.nanmin(dem.data)),
                "vmax": float(np.nanmax(dem.data)),
                "contour_interval": contour_interval,
                "type": "combined"},
    )


# ========== 兼容旧 API（保留部分函数签名） ==========

def render_dem_overlay(
    dem,
    contour_interval: float = 30.0,
    colormap: str = "Greys",
    figsize: tuple[float, float] = (8, 8),
    dpi: int = 100,
) -> RenderResult:
    """旧 API：DEM 灰度 + 等高线。现在转为底图 + 等高线合并。"""
    return render_basemap(dem, colormap, figsize, dpi)
