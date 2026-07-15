"""通用工具函数。"""
from __future__ import annotations

import numpy as np


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """将值限制到 [lo, hi] 区间。"""
    return float(max(lo, min(hi, x)))


def clamp_score(x: float) -> int:
    """限制到 0-100 并四舍五入为 int。"""
    return int(round(clamp(x, 0, 100)))


def linear_normalize(
    x: float | np.ndarray,
    ref_min: float,
    ref_max: float,
    invert: bool = False,
) -> float | np.ndarray:
    """线性归一化到 0-100。ref_min → 0, ref_max → 100（或反向）。"""
    if ref_max == ref_min:
        return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
    t = (np.asarray(x) - ref_min) / (ref_max - ref_min)
    t = np.clip(t, 0.0, 1.0)
    if invert:
        t = 1.0 - t
    return (t * 100.0) if isinstance(x, np.ndarray) else float(t * 100.0)


def degree_to_8(deg: float) -> str:
    """0-360° → 八方位名。"""
    names = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    return names[int(((deg + 22.5) % 360) / 45)]


def degree_to_24(deg: float) -> str:
    """0-360° → 24 山向（壬子癸丑艮寅甲卯乙辰巽巳丙午丁未坤申庚酉辛戌乾亥）。"""
    mountains = [
        "壬", "子", "癸", "丑", "艮", "寅", "甲", "卯", "乙", "辰",
        "巽", "巳", "丙", "午", "丁", "未", "坤", "申", "庚", "酉",
        "辛", "戌", "乾", "亥",
    ]
    return mountains[int(((deg + 7.5) % 360) / 15)]


def angular_diff_deg(a: float, b: float) -> float:
    """两角度之间的最小角差（度，0-180）。"""
    d = (a - b + 180) % 360 - 180
    return abs(float(d))
