"""Viewshed proxy and dual-signal anchor."""
from __future__ import annotations

import numpy as np

from engine.io.dem import DEM


def sector_viewshed_score(
    dem: DEM,
    center_row: int,
    center_col: int,
    target_row: int,
    target_col: int,
    *,
    n_samples: int = 32,
) -> float:
    """简化视线：穴→目标线段上是否被中间地形遮挡。

    返回 0–1：1=全程开阔/目标为视线高点，0=严重遮挡。
    Tier 3 H：Viewshed 朝案粗实现。
    """
    h, w = dem.data.shape
    if not (0 <= center_row < h and 0 <= center_col < w):
        return 0.0
    if not (0 <= target_row < h and 0 <= target_col < w):
        return 0.0
    z0 = float(dem.data[center_row, center_col])
    z1 = float(dem.data[target_row, target_col])
    if not (np.isfinite(z0) and np.isfinite(z1)):
        return 0.0
    # 观察点抬高 2m
    eye = z0 + 2.0
    max_block = 0.0
    for i in range(1, n_samples):
        t = i / float(n_samples)
        r = int(round(center_row + t * (target_row - center_row)))
        c = int(round(center_col + t * (target_col - center_col)))
        if not (0 <= r < h and 0 <= c < w):
            continue
        z = float(dem.data[r, c])
        if not np.isfinite(z):
            continue
        # 视线高度（线性插值到目标）
        line_z = eye + t * (z1 - eye)
        block = z - line_z
        if block > max_block:
            max_block = block
    if max_block <= 0:
        return 1.0
    # 遮挡越多分越低
    return float(np.clip(1.0 - max_block / 40.0, 0.0, 1.0))


def dual_signal_anchor(
    qi_peak: tuple[int, int] | None,
    entrance: tuple[int, int] | None,
    mpx: float,
    mpy: float,
    *,
    pull_m: float = 700.0,
) -> tuple[int, int] | None:
    """Tier 2 G：热峰 + 入首双信号锚点。

    近则信热峰；过远则向入首轻微拉回，避免主龙与橙心脱节。
    """
    if qi_peak is None and entrance is None:
        return None
    if qi_peak is None:
        return entrance
    if entrance is None:
        return qi_peak
    pr, pc = int(qi_peak[0]), int(qi_peak[1])
    er, ec = int(entrance[0]), int(entrance[1])
    d = float(np.hypot((pr - er) * mpy, (pc - ec) * mpx))
    if d <= pull_m:
        return (pr, pc)
    # 65% 热峰 + 35% 入首
    ar = int(round(0.65 * pr + 0.35 * er))
    ac = int(round(0.65 * pc + 0.35 * ec))
    return (ar, ac)


