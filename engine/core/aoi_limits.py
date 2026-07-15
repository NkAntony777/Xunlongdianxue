"""分析区 (AOI) 尺度约束。

依据调研 (research/02_four_beasts、04_acupoint、99_summary/01_速查卡)：

  | 要素           | 距离量级        |
  |----------------|-----------------|
  | 玄武（靠山）   | 50–500 m        |
  | 案山           | 200–1000 m      |
  | 朝山 / 外明堂  | 1–5 km          |
  | 少祖           | 可达数 km       |
  | 龙虎砂         | 约 0.1–2.5 km   |

因此：
  - **过小**（半径 < 3 km）：朝山/少祖常落在图外，四象与主轴失真。
  - **推荐** 5–15 km：能覆盖案朝山与近祖山，又保持 DEM 分辨率与耗时可控。
  - **过大**（半径 > 25 km）：ESRI 单次导出像素变稀、Overpass 易超时，且格局被稀释。

半径定义为：以分析中心为圆心的圆半径（km），拉取 DEM/水系的半宽。
"""
from __future__ import annotations

from typing import Any


# 硬约束（API 拒绝）
MIN_RADIUS_KM = 3.0
MAX_RADIUS_KM = 25.0

# 推荐区间（前端提示，API 不拒绝）
REC_MIN_RADIUS_KM = 5.0
REC_MAX_RADIUS_KM = 15.0

# 默认圈选
DEFAULT_RADIUS_KM = 8.0


def validate_radius_km(radius_km: float) -> float:
    """校验并返回 float；不合法抛 ValueError。"""
    try:
        r = float(radius_km)
    except (TypeError, ValueError) as e:
        raise ValueError(f"radius_km 无效: {radius_km}") from e
    if r < MIN_RADIUS_KM:
        raise ValueError(
            f"分析半径过小（{r:.2f} km < {MIN_RADIUS_KM} km）。"
            f"四象/朝山通常需要 1–5 km 视野，建议 ≥ {REC_MIN_RADIUS_KM} km。"
        )
    if r > MAX_RADIUS_KM:
        raise ValueError(
            f"分析半径过大（{r:.2f} km > {MAX_RADIUS_KM} km）。"
            f"过大将降低 DEM 有效分辨率并显著增加拉取/分析时间，建议 ≤ {REC_MAX_RADIUS_KM} km。"
        )
    return r


def radius_quality(radius_km: float) -> str:
    """返回 ok | small | large | invalid。"""
    try:
        r = float(radius_km)
    except (TypeError, ValueError):
        return "invalid"
    if r < MIN_RADIUS_KM or r > MAX_RADIUS_KM:
        return "invalid"
    if r < REC_MIN_RADIUS_KM:
        return "small"   # 可用但偏小
    if r > REC_MAX_RADIUS_KM:
        return "large"   # 可用但偏大
    return "ok"


def aoi_limits_payload() -> dict[str, Any]:
    """供前端 / 文档使用的配置。"""
    return {
        "min_radius_km": MIN_RADIUS_KM,
        "max_radius_km": MAX_RADIUS_KM,
        "recommended_min_km": REC_MIN_RADIUS_KM,
        "recommended_max_km": REC_MAX_RADIUS_KM,
        "default_radius_km": DEFAULT_RADIUS_KM,
        "unit": "km",
        "shape": "circle",
        "rationale": {
            "min": (
                "朝山 1–5 km、少祖更远；半径 < 3 km 时四象/主轴易截断，"
                "山川格局统计不可靠。"
            ),
            "recommended": (
                "5–15 km 可覆盖案山/朝山与近祖山，并保持约 30–100 m DEM 分辨率与可接受耗时。"
            ),
            "max": (
                "超过 25 km 时单次 ESRI 导出像元变稀、Overpass 易超时，"
                "且一局分析内格局被过度平均。"
            ),
        },
        "diameter_km": {
            "min": MIN_RADIUS_KM * 2,
            "max": MAX_RADIUS_KM * 2,
            "recommended_min": REC_MIN_RADIUS_KM * 2,
            "recommended_max": REC_MAX_RADIUS_KM * 2,
        },
    }
