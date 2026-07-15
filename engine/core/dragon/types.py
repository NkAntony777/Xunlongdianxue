"""Dragon domain types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RidgeLine:
    """山脊线段。"""

    coords: np.ndarray  # shape (N, 2), (row, col)
    length_m: float
    mean_elevation: float
    max_elevation: float
    sinuosity: float  # 蜿蜒度 = 实际长 / 直线长
    feature_significance: float
    # Tier 2：Strahler-like（1=父母/叶, 2=少祖, 3+=太祖）
    strahler_order: int = 1
    parent_idx: int | None = None
    role: str = "branch"  # parent_leaf | shaozu | taizu | branch



@dataclass
class DragonVeinResult:
    """龙脉识别结果。"""

    ridge_mask: np.ndarray
    ridge_lines: list[RidgeLine]
    flow_acc: np.ndarray
    flow_dir: np.ndarray
    entrance_point: tuple[int, int] | None  # 入首点 (row, col)
    entrance_xy: tuple[float, float] | None  # 入首点经纬度
    major_ridges: list[RidgeLine]  # 一级龙脉
    # A1-余：蜂腰鹤膝过峡点（find_yaoxia 输出）
    yaoxia: list[dict[str, Any]] = field(default_factory=list)
    # Tier 2/3 元信息
    meta: dict[str, Any] = field(default_factory=dict)



@dataclass
class PrimaryDragon:
    """图幅主来龙（势来方向 + 入首），供搜穴/四象共用。"""

    ridge_idx: int
    ordered_coords: np.ndarray  # 远源(高) → 入首(低)，shape (N,2) row,col
    entrance_row: int
    entrance_col: int
    entrance_xy: tuple[float, float]
    source_row: int
    source_col: int
    flow_azimuth_deg: float  # 龙气：源 → 入首
    sit_deg: float           # 坐向 ≈ 入首看向源（背靠来龙）
    facing_deg: float        # 朝向 ≈ flow（气往前送）
    score: float
    method: str
    length_m: float
    downhill_m: float
    dragon_vein: DragonVeinResult | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def entrance_point(self) -> tuple[int, int]:
        return (self.entrance_row, self.entrance_col)



@dataclass
class RidgePoint:
    """脊/峰上一点（相对穴）。"""

    row: int
    col: int
    elev_m: float
    dist_m: float
    bearing_deg: float
    score: float = 0.0
    on_ridge: bool = True



@dataclass
class IncomingVeinSelection:
    """相对穴的主来龙选取结果。"""

    xuanwu: RidgePoint | None
    shaozu: RidgePoint | None
    incoming_azimuth_deg: float | None  # 龙气走向：少祖→穴（或玄武→穴）
    sit_align_deg: float | None         # 与坐向偏差
    downhill_ok: bool
    method: str
    score: float
    meta: dict[str, Any] = field(default_factory=dict)


