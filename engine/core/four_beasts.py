"""四兽（青龙/白虎/朱雀/玄武）评分。

坐标约定（与 four_beasts_detect / 数理模型 03 统一）：
  - facing = **朝向**（人面朝方向）：北=0°，东=90°，南=180°，西=270°
  - 坐向 sit = (facing + 180) % 360（玄武方向）
  - 左青龙 = (facing + 270) % 360
  - 右白虎 = (facing + 90) % 360
  - 前朱雀 = facing
  - 后玄武 = sit

参考:
  - research/02_four_beasts/00_四兽量化规则.md
  - research/99_summary/03_数理模型_点穴抽象与公式体系.md
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.core.terrain_analysis import (
    aspect_name,
    compute_slope_aspect,
)
from engine.io.dem import DEM
from engine.utils.helpers import clamp_score, linear_normalize

# 白虎/青龙相对高差比上限（抬头阈值）
BAIHU_QL_RATIO = 0.85

# 青龙蜿蜒：sinuosity 甜区 [1.08, 1.55]；白虎顶坡驯俯上限（度）
QL_SINU_SWEET = (1.08, 1.55)
BH_TAME_SLOPE_MAX = 22.0


@dataclass
class FourBeastsScore:
    """四象评分。"""

    qinglong: int  # 青龙（左）
    baihu: int  # 白虎（右）
    zhuque: int  # 朱雀（前）
    xuanwu: int  # 玄武（后）
    combined: int  # 综合
    details: dict


def _sector_mask(
    aspect_deg: np.ndarray,
    center_deg: float,
    half_width_deg: float = 45.0,
) -> np.ndarray:
    """给定中心方位 ± 半宽，返回该扇区掩膜。

    默认半宽 45° ⇒ 全宽 90°，四象扇区平分 360°，边界相邻元素不重叠。
    与 four_beasts_detect 的窄扇区（用于定位单峰）不同：本函数用于「象内
    统计评分」，理想全宽 90°。
        朱雀 (180±45) ∪ 青龙 (90±45) ∪ 白虎 (270±45) ∪ 玄武 (0±45)
        覆盖 135-225 / 45-135 / 225-315 / 315-45（wrap），互不重叠。

    边界处理：上界用 `<`（不含 half_width_deg），下界用 `>=`，
    确保相邻扇区边界处只归入"下半侧"扇区，不重复计算。

    历史说明：旧版本 half_width=90° 会让四象各占 180°，相邻象严重重叠，
    同一像元同时计入 2 个象分，导致评分虚高。现已修复。

    Args:
        aspect_deg: 方位角数组（北=0）
        center_deg: 扇区中心方位
        half_width_deg: 扇区半宽（默认 45° → 90° 扇区）
    """
    # 用「半宽 +1° 的下界，做包含」再「半宽严格 < 上界」
    # 实现：算 signed diff in [-180, 180)，用 [-half, half)
    sd = ((aspect_deg - center_deg + 180.0) % 360.0) - 180.0
    # -half_width_deg <= sd < half_width_deg
    return (sd >= -half_width_deg) & (sd < half_width_deg)


def _sector_stats(
    dem: DEM,
    aspect_deg_center: float,
    radius_m: float = 300.0,
    slope_arr: np.ndarray | None = None,
    aspect_arr: np.ndarray | None = None,
) -> dict:
    """计算以候选点为中心、某方位扇区内的地形统计量。"""
    if slope_arr is None or aspect_arr is None:
        slope_arr, aspect_arr = compute_slope_aspect(dem)

    h, w = dem.data.shape
    cy, cx = h // 2, w // 2
    from engine.core.terrain_analysis import _is_geographic
    if _is_geographic(dem.crs):
        m_per_unit = 111000.0
    else:
        m_per_unit = 1.0
    yres, xres = abs(dem.resolution[1]) * m_per_unit, abs(dem.resolution[0]) * m_per_unit

    # 构造坐标网格（米）：行号向下增大 → 北 = row 减小 → dy 北正
    yy, xx = np.mgrid[0:h, 0:w]
    dx_m = (xx - cx) * xres
    dy_m = (cy - yy) * yres
    dist_m = np.sqrt(dx_m**2 + dy_m**2)

    # 候选点到每点的方位角（北=0, 东=90）
    bearing = (np.degrees(np.arctan2(dx_m, dy_m)) + 360) % 360

    sector = _sector_mask(bearing, aspect_deg_center, half_width_deg=45.0)
    region = sector & (dist_m <= radius_m) & np.isfinite(dem.data)

    if not region.any():
        return {
            "max_height": np.nan,
            "mean_slope": np.nan,
            "mean_aspect_deg": np.nan,
            "rel_height": np.nan,
            "n_pixels": 0,
            "radius_m": radius_m,
        }

    heights = dem.data[region]
    slopes = slope_arr[region]
    aspects = aspect_arr[region]
    valid_aspect = aspects[~np.isnan(aspects)]

    return {
        "max_height": float(np.nanmax(heights)),
        "mean_height": float(np.nanmean(heights)),
        "mean_slope": float(np.nanmean(slopes)),
        "mean_aspect_deg": float(np.nanmean(valid_aspect)) if valid_aspect.size else float("nan"),
        "rel_height": float(np.nanmax(heights) - dem.data[cy, cx]),
        "n_pixels": int(region.sum()),
        "radius_m": radius_m,
        "region_mask": region,
        "dx_m": dx_m,
        "dy_m": dy_m,
        "dist_m": dist_m,
        "bearing": bearing,
        "cy": cy,
        "cx": cx,
        "xres_m": xres,
        "yres_m": yres,
    }


def _m_per_px_local(dem: DEM) -> tuple[float, float]:
    from engine.core.terrain_analysis import _is_geographic

    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs):
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        return xres * 111_000.0 * cos_lat, yres * 111_000.0
    return xres, yres


def measure_sector_sinuosity(
    dem: DEM,
    center_bearing_deg: float,
    *,
    radius_m: float = 300.0,
    half_width_deg: float = 45.0,
    n_radial: int = 12,
) -> dict:
    """青龙蜿蜒度量：扇区脊线近似 sinuosity + 峰谷起伏频次。

    沿方位中心线两侧取「径向最高点」折线，计算：
      sinuosity = 折线弧长 / 端点弦长（≥1）
      undulation = 高程沿脊二阶变号次数 / 段数（0–1）

    Returns:
        sinuosity, undulation, morph_score(0–100), notes
    """
    h, w = dem.data.shape
    cy, cx = h // 2, w // 2
    mpx, mpy = _m_per_px_local(dem)
    if not np.isfinite(dem.data[cy, cx]):
        return {
            "sinuosity": 1.0,
            "undulation": 0.0,
            "morph_score": 50.0,
            "notes": "中心无效",
        }

    # 径向采样：每环取扇区内最高点
    pts: list[tuple[float, float, float]] = []  # (x_m, y_m, z)
    for i in range(1, n_radial + 1):
        r_m = radius_m * i / n_radial
        best_z = -1e18
        best_xy = None
        # 扇区角向采样
        for k in range(9):
            ang = center_bearing_deg + (-half_width_deg + k * half_width_deg / 4.0)
            rad = np.radians(ang)
            dx = np.sin(rad) * r_m
            dy = np.cos(rad) * r_m
            col = int(round(cx + dx / max(mpx, 1e-6)))
            row = int(round(cy - dy / max(mpy, 1e-6)))
            if not (0 <= row < h and 0 <= col < w):
                continue
            z = float(dem.data[row, col])
            if np.isfinite(z) and z > best_z:
                best_z = z
                best_xy = (dx, dy, z)
        if best_xy is not None:
            pts.append(best_xy)

    if len(pts) < 3:
        return {
            "sinuosity": 1.0,
            "undulation": 0.0,
            "morph_score": 50.0,
            "notes": "脊采样不足",
        }

    # 折线弧长 vs 弦长
    arc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        arc += float(np.hypot(b[0] - a[0], b[1] - a[1]))
    chord = float(np.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]))
    sinu = float(arc / max(chord, 1e-3))
    sinu = float(np.clip(sinu, 1.0, 3.0))

    zs = np.array([p[2] for p in pts], dtype=float)
    # 起伏：相邻高差变号次数
    dz = np.diff(zs)
    sign_changes = 0
    for a, b in zip(dz[:-1], dz[1:]):
        if a * b < 0 and abs(a) > 0.5 and abs(b) > 0.5:
            sign_changes += 1
    und = float(sign_changes / max(len(dz) - 1, 1))

    # 蜿蜒分：sinu 甜区高分；过直(<1.05)或过乱(>2.0)降分
    lo, hi = QL_SINU_SWEET
    if sinu <= 1.02:
        sinu_s = 35.0  # 直硬
    elif sinu < lo:
        sinu_s = 35.0 + 50.0 * (sinu - 1.02) / max(lo - 1.02, 1e-3)
    elif sinu <= hi:
        sinu_s = 85.0 + 15.0 * (1.0 - abs(sinu - (lo + hi) / 2) / max((hi - lo) / 2, 1e-3))
    elif sinu <= 2.0:
        sinu_s = 85.0 - 40.0 * (sinu - hi) / max(2.0 - hi, 1e-3)
    else:
        sinu_s = 40.0

    # 适度峰谷起伏加分（und 0.2–0.6 佳）
    if 0.15 <= und <= 0.65:
        und_s = 80.0 + 20.0 * (1.0 - abs(und - 0.4) / 0.4)
    elif und < 0.15:
        und_s = 50.0 + und / 0.15 * 25.0
    else:
        und_s = max(35.0, 80.0 - (und - 0.65) * 60.0)

    morph = float(clamp_score(0.65 * sinu_s + 0.35 * und_s))
    return {
        "sinuosity": sinu,
        "undulation": und,
        "morph_score": morph,
        "notes": f"sinu={sinu:.2f} und={und:.2f}",
    }


def measure_sector_tame(
    dem: DEM,
    center_bearing_deg: float,
    slope_arr: np.ndarray,
    *,
    radius_m: float = 300.0,
    half_width_deg: float = 45.0,
    cand_elev: float | None = None,
) -> dict:
    """白虎驯俯：圆净低伏 vs 尖削低陷。

    指标：
      - mean_top_slope：顶区平均坡度（缓=圆驯，陡=尖削）
      - peakiness：最高点相对扇区均值的突出度
      - roundness_proxy：1 - clip(peakiness/40,0,1) 近似圆净
    """
    h, w = dem.data.shape
    cy, cx = h // 2, w // 2
    mpx, mpy = _m_per_px_local(dem)
    if cand_elev is None:
        cand_elev = float(dem.data[cy, cx]) if np.isfinite(dem.data[cy, cx]) else 0.0

    yy, xx = np.mgrid[0:h, 0:w]
    dx_m = (xx - cx) * mpx
    dy_m = (cy - yy) * mpy
    dist_m = np.hypot(dx_m, dy_m)
    bearing = (np.degrees(np.arctan2(dx_m, dy_m)) + 360.0) % 360.0
    region = _sector_mask(bearing, center_bearing_deg, half_width_deg)
    region = region & (dist_m <= radius_m) & (dist_m >= 30.0) & np.isfinite(dem.data)
    if not region.any():
        return {
            "mean_top_slope": float("nan"),
            "peakiness": 0.0,
            "tame_score": 50.0,
            "notes": "扇区空",
        }

    heights = dem.data[region]
    slopes = slope_arr[region]
    mean_h = float(np.nanmean(heights))
    max_h = float(np.nanmax(heights))
    peakiness = max(0.0, max_h - mean_h)
    mean_slope = float(np.nanmean(slopes[np.isfinite(slopes)])) if np.isfinite(slopes).any() else 25.0

    # 顶区：高于 70 分位
    thr = float(np.nanpercentile(heights, 70))
    top = region & (dem.data >= thr)
    top_slope = float(np.nanmean(slope_arr[top])) if top.any() and np.isfinite(slope_arr[top]).any() else mean_slope

    # 驯俯分：缓坡圆净高分；尖削(顶坡>32°)低分；过低且尖(peakiness高且rel低)略减
    if top_slope <= 12.0:
        slope_s = 92.0
    elif top_slope <= BH_TAME_SLOPE_MAX:
        slope_s = 92.0 - (top_slope - 12.0) / max(BH_TAME_SLOPE_MAX - 12.0, 1e-3) * 35.0
    elif top_slope <= 35.0:
        slope_s = 57.0 - (top_slope - BH_TAME_SLOPE_MAX) / max(35.0 - BH_TAME_SLOPE_MAX, 1e-3) * 30.0
    else:
        slope_s = 20.0  # 尖削

    # peakiness：适度(5–25m)圆润，>40m 尖峰
    if peakiness <= 8.0:
        pk_s = 70.0 + peakiness  # 太平略减
    elif peakiness <= 25.0:
        pk_s = 88.0
    elif peakiness <= 45.0:
        pk_s = 88.0 - (peakiness - 25.0) / 20.0 * 40.0
    else:
        pk_s = 40.0

    tame = float(clamp_score(0.60 * slope_s + 0.40 * pk_s))
    return {
        "mean_top_slope": top_slope,
        "peakiness": peakiness,
        "tame_score": tame,
        "notes": f"topSlope={top_slope:.1f}° peak={peakiness:.1f}m",
    }


def measure_sector_viewshed(
    dem: DEM,
    center_bearing_deg: float,
    *,
    radius_m: float = 300.0,
    n_rays: int = 9,
    half_width_deg: float = 45.0,
) -> dict:
    """朱雀/明堂 viewshed：前向多射线开阔度 0–1。"""
    from engine.core.dragon_vein import sector_viewshed_score

    h, w = dem.data.shape
    cy, cx = h // 2, w // 2
    mpx, mpy = _m_per_px_local(dem)
    scores: list[float] = []
    for k in range(n_rays):
        ang = center_bearing_deg + (-half_width_deg + k * (2 * half_width_deg) / max(n_rays - 1, 1))
        rad = np.radians(ang)
        # 目标点：半径 70% 处
        r_m = radius_m * 0.75
        dx = np.sin(rad) * r_m
        dy = np.cos(rad) * r_m
        tc = int(round(cx + dx / max(mpx, 1e-6)))
        tr = int(round(cy - dy / max(mpy, 1e-6)))
        if not (0 <= tr < h and 0 <= tc < w):
            continue
        scores.append(sector_viewshed_score(dem, cy, cx, tr, tc, n_samples=24))
    if not scores:
        return {"viewshed": 0.5, "viewshed_score": 50.0, "n_rays": 0}
    vs = float(np.mean(scores))
    return {
        "viewshed": vs,
        "viewshed_score": float(clamp_score(vs * 100.0)),
        "n_rays": len(scores),
    }


def score_four_beasts(
    dem: DEM,
    slope_arr: np.ndarray | None = None,
    aspect_arr: np.ndarray | None = None,
    search_radius_m: float = 300.0,
    facing_override: str | float | None = None,
) -> FourBeastsScore:
    """对 DEM 中心点进行四象评分。

    Args:
        dem: DEM 数据（中心点默认为候选穴）
        slope_arr: 预计算的坡度（可选）
        aspect_arr: 预计算的坡向（可选）
        search_radius_m: 砂山搜索半径（米）
        facing_override: **朝向**名（"南"/"东南"…）或角度；None 默认朝南 180°

    Returns:
        FourBeastsScore
    """
    if slope_arr is None or aspect_arr is None:
        slope_arr, aspect_arr = compute_slope_aspect(dem)

    h, w = dem.data.shape
    cy, cx = h // 2, w // 2
    cand_elev = float(dem.data[cy, cx])

    # 1. 朝向 facing（与 four_beasts_detect 一致）
    if facing_override is None:
        facing = 180.0  # 默认坐北朝南 → 面朝南
    elif isinstance(facing_override, (int, float)):
        facing = float(facing_override) % 360.0
    else:
        facing_map = {
            "北": 0, "东北": 45, "东": 90, "东南": 135,
            "南": 180, "西南": 225, "西": 270, "西北": 315,
        }
        facing = float(facing_map.get(str(facing_override), 180))

    sit = (facing + 180.0) % 360.0
    left_dir = (facing + 270.0) % 360.0   # 青龙
    right_dir = (facing + 90.0) % 360.0   # 白虎
    front_dir = facing                    # 朱雀
    back_dir = sit                        # 玄武

    # 2. 四扇区（方位用「从穴指向该点」的 bearing，北=0）
    ql = _sector_stats(dem, left_dir, search_radius_m, slope_arr, aspect_arr)
    bh = _sector_stats(dem, right_dir, search_radius_m, slope_arr, aspect_arr)
    zq = _sector_stats(dem, front_dir, search_radius_m, slope_arr, aspect_arr)
    xw = _sector_stats(dem, back_dir, search_radius_m, slope_arr, aspect_arr)

    # 3. 玄武（靠山）评分：高度可观 + 坡度合适
    if np.isnan(xw["max_height"]):
        xw_score = 30
    else:
        xw_height_score = linear_normalize(xw["max_height"] - cand_elev, 0, 100)
        xw_slope_score = 100 - linear_normalize(xw["mean_slope"], 0, 35)
        xw_score = clamp_score(0.7 * xw_height_score + 0.3 * xw_slope_score)

    # 4. 朱雀（明堂）：坡度代理 + viewshed 真开阔度（P1）
    vs = measure_sector_viewshed(dem, front_dir, radius_m=search_radius_m)
    if np.isnan(zq["mean_slope"]):
        zq_base = 30.0
    else:
        zq_slope_score = 100 - linear_normalize(zq["mean_slope"], 0, 25)
        zq_height_penalty = linear_normalize(zq["max_height"] - cand_elev, 0, 60, invert=True)
        zq_base = 0.7 * zq_slope_score + 0.3 * zq_height_penalty
    # 坡度 55% + viewshed 45%
    zq_score = clamp_score(0.55 * zq_base + 0.45 * vs["viewshed_score"])

    # 5. 青龙（左）：高度 + 蜿蜒形态（P1）
    ql_morph = measure_sector_sinuosity(dem, left_dir, radius_m=search_radius_m)
    if np.isnan(ql["max_height"]):
        ql_height_score = 50.0
    else:
        ql_height_score = linear_normalize(ql["max_height"] - cand_elev, 0, 150)
    # 高度 60% + 蜿蜒 40%（直硬高脊不得满分）
    ql_score = clamp_score(0.60 * ql_height_score + 0.40 * ql_morph["morph_score"])

    # 6. 白虎（右）：高差比 + 驯俯形态（P1，非越低越好）
    bh_tame = measure_sector_tame(
        dem, right_dir, slope_arr, radius_m=search_radius_m, cand_elev=cand_elev,
    )
    if np.isnan(bh["max_height"]):
        bh_ratio_score = 50.0
    else:
        ql_h = ql["max_height"] if not np.isnan(ql["max_height"]) else cand_elev
        bh_h = bh["max_height"]
        ql_rel = max(ql_h - cand_elev, 1.0)
        bh_rel = bh_h - cand_elev
        ratio = bh_rel / ql_rel if ql_rel > 0 else 1.0
        if ratio > 1.0:
            bh_ratio_score = 25.0
        elif ratio > BAIHU_QL_RATIO:
            t = (ratio - BAIHU_QL_RATIO) / (1.0 - BAIHU_QL_RATIO)
            bh_ratio_score = float(clamp_score(70 * (1.0 - t) + 30 * t))
        else:
            # 适度低于青龙为吉；过低(ratio≈0)略减（缺砂）
            # 最佳 ratio ≈ 0.35–0.70
            if ratio < 0.15:
                bh_ratio_score = float(clamp_score(55 + ratio / 0.15 * 20))
            elif ratio <= 0.70:
                bh_ratio_score = float(clamp_score(88 + (0.5 - abs(ratio - 0.5)) * 20))
            else:
                bh_ratio_score = float(
                    clamp_score(100 * (1.0 - 0.30 * ratio / BAIHU_QL_RATIO))
                )
    bh_score = clamp_score(0.55 * bh_ratio_score + 0.45 * bh_tame["tame_score"])

    # 7. 综合
    combined = clamp_score(0.25 * ql_score + 0.25 * bh_score + 0.25 * xw_score + 0.25 * zq_score)

    # 清理不可序列化的临时键
    def _clean_sector(s: dict) -> dict:
        skip = {"region_mask", "dx_m", "dy_m", "dist_m", "bearing"}
        return {k: v for k, v in s.items() if k not in skip and not isinstance(v, np.ndarray)}

    return FourBeastsScore(
        qinglong=ql_score,
        baihu=bh_score,
        zhuque=zq_score,
        xuanwu=xw_score,
        combined=combined,
        details={
            "qinglong_sector": _clean_sector(ql),
            "baihu_sector": _clean_sector(bh),
            "zhuque_sector": _clean_sector(zq),
            "xuanwu_sector": _clean_sector(xw),
            "qinglong_morph": ql_morph,
            "baihu_tame": bh_tame,
            "zhuque_viewshed": vs,
            "facing": facing,  # 朝向
            "sit": sit,
            "dirs": {
                "qinglong": left_dir,
                "baihu": right_dir,
                "zhuque": front_dir,
                "xuanwu": back_dir,
            },
            "search_radius_m": search_radius_m,
        },
    )


def score_four_beasts_combined_at(
    elev: np.ndarray,
    slope: np.ndarray,
    row: int,
    col: int,
    *,
    xres_m: float,
    yres_m: float,
    search_radius_m: float = 300.0,
    facing: float = 180.0,
) -> float:
    """G.3 快速路径：在整幅栅格上对像元 (row, col) 计算四象综合分。

    与 score_four_beasts 评分规则一致，但：
      - 不分配 DEM / 不重算坡度
      - 单次窗口 mgrid + 四扇区掩膜
      - 仅返回 combined（热力场用）

    供 compute_score_grid 等全图扫描调用。
    """
    h, w = elev.shape
    if not (0 <= row < h and 0 <= col < w):
        return 30.0
    cand = float(elev[row, col])
    if not np.isfinite(cand):
        return 30.0

    facing = float(facing) % 360.0
    sit = (facing + 180.0) % 360.0
    dirs = (
        (facing + 270.0) % 360.0,  # 青龙
        (facing + 90.0) % 360.0,   # 白虎
        facing,                    # 朱雀
        sit,                       # 玄武
    )

    # 窗口：半径 + 1 像素边
    pr = max(1, int(np.ceil(search_radius_m / max(yres_m, 1e-6))))
    pc = max(1, int(np.ceil(search_radius_m / max(xres_m, 1e-6))))
    r0 = max(0, row - pr)
    r1 = min(h, row + pr + 1)
    c0 = max(0, col - pc)
    c1 = min(w, col + pc + 1)

    sub_e = elev[r0:r1, c0:c1]
    sub_s = slope[r0:r1, c0:c1]
    yy, xx = np.mgrid[r0:r1, c0:c1]
    dx_m = (xx - col) * xres_m
    dy_m = (row - yy) * yres_m  # 北 = row 减小
    dist_m = np.hypot(dx_m, dy_m)
    bearing = (np.degrees(np.arctan2(dx_m, dy_m)) + 360.0) % 360.0
    finite = np.isfinite(sub_e) & (dist_m <= search_radius_m) & (dist_m > 0)

    def _sector_mh_ms(center_deg: float) -> tuple[float, float]:
        """返回 (max_height, mean_slope)，空扇区 → (nan, nan)。"""
        mask = finite & _sector_mask(bearing, center_deg, half_width_deg=45.0)
        if not mask.any():
            return float("nan"), float("nan")
        heights = sub_e[mask]
        slopes = sub_s[mask]
        mh = float(np.max(heights))
        # 坡度可能含 nan
        if slopes.size and np.isfinite(slopes).any():
            ms = float(np.nanmean(slopes))
        else:
            ms = float("nan")
        return mh, ms

    ql_h, _ql_s = _sector_mh_ms(dirs[0])
    bh_h, bh_s = _sector_mh_ms(dirs[1])
    zq_h, zq_s = _sector_mh_ms(dirs[2])
    xw_h, xw_s = _sector_mh_ms(dirs[3])
    _bh_s = bh_s

    # 玄武
    if np.isnan(xw_h):
        xw_score = 30.0
    else:
        xw_height_score = linear_normalize(xw_h - cand, 0, 100)
        xw_slope_score = 100.0 - linear_normalize(
            0.0 if np.isnan(xw_s) else xw_s, 0, 35
        )
        xw_score = float(clamp_score(0.7 * xw_height_score + 0.3 * xw_slope_score))

    # 朱雀（热力快路径：坡度代理；完整路径另含 viewshed）
    if np.isnan(zq_s) and np.isnan(zq_h):
        zq_score = 30.0
    else:
        zq_slope_score = 100.0 - linear_normalize(
            0.0 if np.isnan(zq_s) else zq_s, 0, 25
        )
        zq_height_penalty = linear_normalize(
            (0.0 if np.isnan(zq_h) else zq_h) - cand, 0, 60, invert=True
        )
        zq_base = 0.7 * zq_slope_score + 0.3 * zq_height_penalty
        # 快路径无 viewshed：以坡度开阔代理 0.85 中性补齐，逼近完整分
        zq_score = float(clamp_score(0.55 * zq_base + 0.45 * 70.0))

    # 青龙（快路径：高度为主 + 中性蜿蜒 70）
    if np.isnan(ql_h):
        ql_h_s = 50.0
    else:
        ql_h_s = float(linear_normalize(ql_h - cand, 0, 150))
    ql_score = float(clamp_score(0.60 * ql_h_s + 0.40 * 70.0))

    # 白虎（与完整路径同 ratio 甜区 + 驯俯用顶坡近似）
    if np.isnan(bh_h):
        bh_ratio_score = 50.0
    else:
        ql_rel = max((ql_h if not np.isnan(ql_h) else cand) - cand, 1.0)
        bh_rel = bh_h - cand
        ratio = bh_rel / ql_rel if ql_rel > 0 else 1.0
        if ratio > 1.0:
            bh_ratio_score = 25.0
        elif ratio > BAIHU_QL_RATIO:
            t = (ratio - BAIHU_QL_RATIO) / (1.0 - BAIHU_QL_RATIO)
            bh_ratio_score = float(clamp_score(70 * (1.0 - t) + 30 * t))
        else:
            if ratio < 0.15:
                bh_ratio_score = float(clamp_score(55 + ratio / 0.15 * 20))
            elif ratio <= 0.70:
                bh_ratio_score = float(clamp_score(88 + (0.5 - abs(ratio - 0.5)) * 20))
            else:
                bh_ratio_score = float(
                    clamp_score(100 * (1.0 - 0.30 * ratio / BAIHU_QL_RATIO))
                )
    # 白虎扇区 mean slope 作驯俯代理
    _bh_s = _bh_s if not np.isnan(_bh_s) else 18.0
    if _bh_s <= 12.0:
        tame_s = 90.0
    elif _bh_s <= BH_TAME_SLOPE_MAX:
        tame_s = 90.0 - (_bh_s - 12.0) / max(BH_TAME_SLOPE_MAX - 12.0, 1e-3) * 30.0
    else:
        tame_s = max(25.0, 60.0 - (_bh_s - BH_TAME_SLOPE_MAX) * 2.0)
    bh_score = float(clamp_score(0.55 * bh_ratio_score + 0.45 * tame_s))

    return float(clamp_score(0.25 * ql_score + 0.25 * bh_score + 0.25 * xw_score + 0.25 * zq_score))
