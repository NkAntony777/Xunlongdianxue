"""Qi / score field rasters (field layer).

Moved from four_beasts_detect: compute_qi_field_layers, compute_score_grid, peaks.
"""
from __future__ import annotations

import numpy as np

from engine.io.dem import DEM
from engine.core.field.water_raster import water_distance_rasters

def _auto_sample_step(h: int, w: int, sample_step: int, max_samples: int | None) -> int:
    """兼容旧 API；生气场路径已全幅矢量化，一般不再依赖采样步长。"""
    step = max(1, int(sample_step))
    if max_samples is None or max_samples <= 0:
        return step
    n = max(1, (h // step) * (w // step))
    if n <= max_samples:
        return step
    s = int(np.ceil(np.sqrt((h * w) / float(max_samples))))
    return max(step, s)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """稳定 sigmoid，避免 overflow。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _water_distance_plateau(
    dist_m: np.ndarray,
    *,
    d_lo: float = 180.0,
    d_hi: float = 900.0,
    d_far: float = 2800.0,
    near_floor: float = 0.22,
    far_floor: float = 0.28,
) -> np.ndarray:
    """得水宽平台：明堂腹地高分，避免贴岸光环。

    - [d_lo, d_hi] → 1.0（有情界水甜区，默认自 ~180m 起）
    - < d_lo：从 near_floor 抬升到 1（贴岸明显低于堂心）
    - > d_hi：缓降到 far_floor
    """
    d = np.asarray(dist_m, dtype=np.float64)
    g = np.full_like(d, far_floor, dtype=np.float64)
    # 近侧肩：ban 外到 d_lo
    near = d < d_lo
    g = np.where(
        near,
        near_floor + (1.0 - near_floor) * np.clip(d / max(d_lo, 1e-6), 0.0, 1.0),
        g,
    )
    # 甜区平台
    g = np.where((d >= d_lo) & (d <= d_hi), 1.0, g)
    # 远侧肩
    mid_far = (d > d_hi) & (d <= d_far)
    t = (d - d_hi) / max(d_far - d_hi, 1e-6)
    g = np.where(mid_far, 1.0 - (1.0 - far_floor) * np.clip(t, 0.0, 1.0), g)
    g = np.where(d > d_far, far_floor, g)
    return np.clip(g, 0.0, 1.0)


def compute_qi_field_layers(
    dem: DEM,
    water=None,
    *,
    tpi_fine_m: float = 80.0,
    tpi_mid_m: float = 400.0,
    enclosure_radius_m: float = 500.0,
    water_opt_m: float = 300.0,
    water_sigma_m: float = 400.0,
    water_lo_m: float = 180.0,
    water_hi_m: float = 900.0,
    water_far_m: float = 2800.0,
    slope_opt_deg: float = 5.0,
    slope_sigma_deg: float = 14.0,
    floor_cangfeng: float = 0.30,
    floor_water: float = 0.18,
    floor_enclosure: float = 0.52,
    floor_stability: float = 0.30,
    mingtang_boost: float = 0.42,
) -> dict[str, np.ndarray]:
    """全矢量生气子场（0–1），方向无关。

    设计目标（对标河湾平坦明堂 / 参考图橙心）::
      - 藏风：中尺度略凹 + **细尺度平台**（大凹中小平）——平台权重大
      - 得水：**宽平台** [water_lo, water_hi]，非 300m 尖峰；贴岸肩低
      - 围合：**开阔平坦优先**（低起伏 + 缓坡），弱化「高墙夹峙」
      - 稳定：0–8° 宜穴，近平给高分
      - 明堂通道：平坦×开阔×有情水距 再乘性抬升（mingtang_boost）
      - 乘性 + 各通道地板，避免单项否决整场

    water_opt_m / water_sigma_m 保留兼容；优先使用 lo/hi/far 平台参数。
    """
    from scipy.ndimage import maximum_filter
    from engine.core.terrain_analysis import (
        compute_slope_aspect,
        tpi as _tpi,
        _is_geographic,
    )

    _ = (water_opt_m, water_sigma_m)  # 兼容旧 weights 键名

    elev = np.asarray(dem.data, dtype=np.float64)
    finite = np.isfinite(elev)
    if not finite.any():
        z = np.zeros_like(elev, dtype=np.float64)
        ban = np.zeros_like(elev, dtype=bool)
        return {
            "cangfeng": z,
            "water": z,
            "enclosure": z,
            "stability": z,
            "qi": z,
            "water_ban": ban,
            "finite": finite,
        }

    slope_arr, _ = compute_slope_aspect(dem)
    tpi_fine = _tpi(dem, radius_m=tpi_fine_m)
    tpi_mid = _tpi(dem, radius_m=tpi_mid_m)
    tpi_fine = np.where(np.isfinite(tpi_fine), tpi_fine, 0.0)
    tpi_mid = np.where(np.isfinite(tpi_mid), tpi_mid, 0.0)

    from scipy.ndimage import uniform_filter

    if _is_geographic(dem.crs):
        m_per_unit = 111000.0
    else:
        m_per_unit = 1.0
    xres_m = abs(dem.resolution[0]) * m_per_unit
    yres_m = abs(dem.resolution[1]) * m_per_unit
    mpp = max(0.5 * (xres_m + yres_m), 1e-6)
    r_px = max(1, int(round(enclosure_radius_m / mpp)))
    # 邻域窗：围合 / 弯内对比
    win = 2 * r_px + 1
    win_small = max(3, 2 * max(1, r_px // 3) + 1)

    # 1) 藏风：中尺度略凹 + 细尺度平台（明堂）；平台权重再提高
    g_basin = _sigmoid(-0.35 * tpi_mid)
    g_platform = np.exp(-(tpi_fine ** 2) / (2.0 * 0.55 ** 2))  # 更贴 |TPI|≈0
    g_cf = 0.25 * g_basin + 0.75 * g_platform
    g_cf = np.clip(np.maximum(g_cf, floor_cangfeng), 0.0, 1.0)

    # 2) 稳定：0–6° 宜穴；近平给高分；陡坡明显衰减
    slope_safe = np.where(np.isfinite(slope_arr), slope_arr, 45.0)
    g_stab = np.exp(
        -((slope_safe - slope_opt_deg) / max(slope_sigma_deg, 1e-6)) ** 2
    )
    g_stab = np.where(slope_safe <= 3.0, np.maximum(g_stab, 0.97), g_stab)
    g_stab = np.where(slope_safe <= 6.0, np.maximum(g_stab, 0.90), g_stab)
    # 陡岸/坡麓（>16°）强压——参考图热力不在山坡
    g_stab = np.where(slope_safe > 16.0, g_stab * 0.38, g_stab)
    g_stab = np.where(slope_safe > 22.0, g_stab * 0.55, g_stab)
    g_stab = np.clip(np.maximum(g_stab, floor_stability), 0.0, 1.0)

    # 3) 得水：宽平台 + 弯内/陆心（距水局部极大 → 半岛心）
    water_dist, water_ban = water_distance_rasters(dem, water)
    has_water = bool(np.isfinite(water_dist).any() and np.any(water_dist < 1e12))
    if has_water:
        d = np.where(np.isfinite(water_dist), water_dist, 1.0e6)
        g_water = _water_distance_plateau(
            d, d_lo=water_lo_m, d_hi=water_hi_m, d_far=water_far_m,
            near_floor=0.20,
        )
        # 贴岸额外肩衰减：禁带外 70–210m 仍压光环，逼热力入堂心
        bank_fade = np.clip((d - 70.0) / 140.0, 0.0, 1.0)
        g_water = g_water * (0.32 + 0.68 * bank_fade)
        # 弯内：距水大于邻域均值 → 离岸、居陆心（河环内侧台地）
        d_land = np.where(water_ban, 0.0, np.clip(d, 0.0, water_hi_m * 1.2))
        d_nb = uniform_filter(d_land, size=win_small, mode="nearest")
        inland_excess = np.clip((d_land - d_nb) / 30.0, 0.0, 1.0)
        # 仅在有情距离带内抬弯内 / 堂心
        in_band = (d_land >= water_lo_m * 0.70) & (d_land <= water_hi_m * 1.2)
        g_inland = np.where(in_band, 0.38 + 0.62 * inland_excess, 0.36)
        g_water = g_water * (0.48 + 0.52 * g_inland)
        g_water = np.where(water_ban, 0.0, np.clip(g_water, 0.0, 1.0))
        g_water = np.where(water_ban, 0.0, np.maximum(g_water, floor_water))
        g_water = np.where(water_ban, 0.0, g_water)
    else:
        g_water = np.full_like(elev, 0.72, dtype=np.float64)
        water_ban = np.zeros_like(elev, dtype=bool)
        d = np.full_like(elev, 500.0)
        in_band = np.ones_like(elev, dtype=bool)

    # 4) 围合：开阔平坦优先（参考图明堂），靠山为辅、弱化高差奖山脚
    fill_val = float(np.nanmedian(elev[finite]))
    elev_f = np.where(finite, elev, fill_val)
    surrounding_max = maximum_filter(elev_f, size=win, mode="nearest")
    surrounding_mean = uniform_filter(elev_f, size=win, mode="nearest")
    relief_max = np.maximum(surrounding_max - elev_f, 0.0)
    # 局部起伏（小窗 std 代理）：明堂腹地应低
    elev_sq = elev_f * elev_f
    local_mean = uniform_filter(elev_f, size=win_small, mode="nearest")
    local_var = np.maximum(
        uniform_filter(elev_sq, size=win_small, mode="nearest") - local_mean ** 2,
        0.0,
    )
    local_std = np.sqrt(local_var)
    g_flat_local = np.exp(-(local_std ** 2) / (2.0 * 6.0 ** 2))  # σ≈6m 内高分
    g_flat_local = np.maximum(g_flat_local, 0.35)

    # 靠山：穴低于邻域平均（背后有高），权重降低——避免热力爬坡
    below_mean = surrounding_mean - elev_f
    g_back = _sigmoid(0.05 * below_mean)
    # 开阔：低 max 高差 + 低局地起伏
    g_open_relief = np.exp(-(np.maximum(relief_max - 18.0, 0.0) ** 2) / (2.0 * 55.0 ** 2))
    g_open_relief = np.maximum(g_open_relief, 0.58)
    g_open_relief = np.where(relief_max > 140.0, g_open_relief * 0.50, g_open_relief)
    g_open = 0.55 * g_open_relief + 0.45 * g_flat_local
    # 合成：开阔平坦为主，靠山为辅；缓坡再抬
    g_enc = 0.32 * g_back + 0.68 * g_open
    g_enc = g_enc * (0.70 + 0.30 * np.exp(-(slope_safe / 12.0) ** 2))
    g_enc = np.clip(np.maximum(g_enc, floor_enclosure), 0.0, 1.0)

    # 5) 明堂通道：平坦平台 × 开阔 × 缓坡 × 有情水距
    g_mt_flat = g_platform * g_flat_local
    g_mt_open = g_open * np.exp(-(slope_safe / 10.0) ** 2)
    if has_water:
        g_mt_water = np.where(in_band, g_water, g_water * 0.75)
    else:
        g_mt_water = g_water
    g_mingtang = np.clip(
        g_mt_flat * (0.40 + 0.60 * g_mt_open) * (0.50 + 0.50 * g_mt_water),
        0.0, 1.0,
    )
    g_mingtang = np.where(water_ban, 0.0, g_mingtang)

    # 乘性融合 + 明堂乘性抬升（参考图：橙心在平坦明堂腹地，非贴岸）
    qi = g_cf * g_water * g_enc * g_stab
    boost = float(np.clip(mingtang_boost, 0.0, 0.60))
    qi = qi * (1.0 - boost + boost * (0.28 + 0.72 * g_mingtang))
    qi = np.where(finite & ~water_ban, qi, 0.0)
    # 平滑：略加大，凝聚单团明堂心（对标参考图大团橙心）
    from scipy.ndimage import gaussian_filter

    qi_s = gaussian_filter(qi, sigma=max(1.0, min(r_px, 10) * 0.50))
    # 保持禁水与无效
    qi = np.where(finite & ~water_ban, qi_s, 0.0)
    # 归一到 [0,1] 保持动态范围
    qmax = float(np.max(qi)) if qi.size else 0.0
    if qmax > 1e-9:
        qi = qi / qmax
    qi = np.clip(qi, 0.0, 1.0)

    return {
        "cangfeng": g_cf.astype(np.float64),
        "water": g_water.astype(np.float64),
        "enclosure": g_enc.astype(np.float64),
        "stability": g_stab.astype(np.float64),
        "mingtang": g_mingtang.astype(np.float64),
        "qi": qi.astype(np.float64),
        "water_ban": water_ban,
        "finite": finite,
    }


def compute_score_grid(
    dem: DEM,
    weights: dict[str, float] | None = None,
    tpi_radius_m: float = 100.0,
    sample_step: int = 4,
    water=None,
    *,
    use_water_form: bool = False,
    max_samples: int | None = 12_000,
    search_radius_m: float = 300.0,
) -> np.ndarray:
    """计算全图生气评分场（热力 + 穴心）。

    生气场（乘性 · 全矢量，方向无关）::

        F(p) = G_藏风(p) × G_得水(p) × G_围合(p) × G_稳定(p)

    - 藏风：中尺度略凹 + 细尺度平台（明堂）
    - 得水：宽距离平台；水面+缓冲硬零
    - 围合：宽高差甜区 + 开阔底分
    - 稳定：缓坡/近平为吉

    输出 0–100（qi×100）；水面/无效像元为 nan。
    穴心 = find_score_peak(场)；四象/少祖在峰值处再 detect。
    """
    _ = (tpi_radius_m, sample_step, use_water_form, max_samples, search_radius_m)
    w = dict(weights or {})
    layers = compute_qi_field_layers(
        dem,
        water,
        tpi_fine_m=float(w.get("tpi_fine_m", 80.0)),
        tpi_mid_m=float(w.get("tpi_mid_m", 400.0)),
        enclosure_radius_m=float(
            w.get("enclosure_radius_m", w.get("search_radius_m", 500.0))
        ),
        water_opt_m=float(w.get("water_opt_m", 300.0)),
        water_sigma_m=float(w.get("water_sigma_m", 400.0)),
        water_lo_m=float(w.get("water_lo_m", 120.0)),
        water_hi_m=float(w.get("water_hi_m", 900.0)),
        water_far_m=float(w.get("water_far_m", 2800.0)),
        slope_opt_deg=float(w.get("slope_opt_deg", 5.0)),
        slope_sigma_deg=float(w.get("slope_sigma_deg", 14.0)),
        floor_cangfeng=float(w.get("floor_cangfeng", 0.32)),
        floor_water=float(w.get("floor_water", 0.22)),
        floor_enclosure=float(w.get("floor_enclosure", 0.48)),
        floor_stability=float(w.get("floor_stability", 0.32)),
    )
    qi = layers["qi"]
    ban = layers["water_ban"]
    finite = layers["finite"]
    score = qi * 100.0
    score = np.where(finite & ~ban, score, np.nan)
    return score.astype(np.float64)


def smooth_score_field(
    score_grid: np.ndarray,
    *,
    smooth_sigma: float | None = None,
) -> tuple[np.ndarray, float]:
    """返回 (平滑后评分场, sigma)。无效区保持 nan。"""
    from scipy.ndimage import gaussian_filter

    h, w = score_grid.shape
    valid = np.isfinite(score_grid)
    if smooth_sigma is None:
        smooth_sigma = max(1.2, min(h, w) / 80.0)
    sigma = float(smooth_sigma)
    if not valid.any():
        return np.full_like(score_grid, np.nan, dtype=np.float64), sigma
    filled = np.where(valid, score_grid.astype(np.float64), 0.0)
    soft = gaussian_filter(filled, sigma=sigma)
    # 边界：无效像元不扩散有效分
    weight = gaussian_filter(valid.astype(np.float64), sigma=sigma)
    soft = soft / np.maximum(weight, 1e-9)
    soft = np.where(valid, soft, np.nan)
    return soft, sigma


def find_score_peak(
    score_grid: np.ndarray,
    *,
    smooth_sigma: float | None = None,
) -> tuple[int, int, float] | None:
    """在评分场上取平滑后的最高点，作为「穴」中心。

    与 render_score_grid 共用 smooth_score_field，保证橙心 ≡ 场评最高点。

    Returns:
        (row, col, smoothed_score) 或 None（全无效）
    """
    if score_grid is None or score_grid.size == 0:
        return None
    soft, _ = smooth_score_field(score_grid, smooth_sigma=smooth_sigma)
    valid = np.isfinite(soft)
    if not valid.any():
        return None
    filled = np.where(valid, soft, -np.inf)
    pr, pc = np.unravel_index(int(np.argmax(filled)), filled.shape)
    pr, pc = int(pr), int(pc)
    return pr, pc, float(soft[pr, pc])


def score_peak_xy(
    dem: DEM,
    score_grid: np.ndarray,
    *,
    smooth_sigma: float | None = None,
) -> tuple[tuple[int, int], list[float], float] | None:
    """评分场峰值 → (row,col) + 世界坐标 [x,y] + 分数。"""
    peak = find_score_peak(score_grid, smooth_sigma=smooth_sigma)
    if peak is None:
        return None
    pr, pc, sc = peak
    try:
        x, y = dem.xy(pr, pc)
    except Exception:
        return None
    return (pr, pc), [float(x), float(y)], float(sc)
