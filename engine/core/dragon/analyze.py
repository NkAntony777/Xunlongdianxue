"""One-shot dragon vein analysis pipeline."""
from __future__ import annotations

from typing import Any

import numpy as np
from skimage.morphology import skeletonize

from engine.io.dem import DEM
from engine.core.dragon.types import DragonVeinResult, RidgeLine
from engine.core.dragon.hydro import (
    resolve_flats,
    compute_flow_direction,
    compute_flow_accumulation,
)
from engine.core.dragon.ridge import (
    extract_ridges,
    multi_scale_ridge_mask,
    vectorize_ridges,
    feature_significance_filter,
)
from engine.core.dragon.entrance import find_entrance_point
from engine.core.dragon.yaoxia import find_yaoxia


def analyze_dragon_vein(
    dem: DEM,
    filled_dem: np.ndarray | None = None,
    min_length_m: float = 100.0,
    water=None,
) -> DragonVeinResult:
    """一站式龙脉识别：填洼 → 平地解算 → D8 → 拓扑累积 → 提脊 → 入首。

    Args:
        dem: 原始 DEM
        filled_dem: 填洼后的 DEM（None 时自动填洼）
        min_length_m: 最短山脊线长度（米）
        water: 可选水系，用于入首近水评分
    """
    if filled_dem is None:
        from engine.io.dem import fill_pits

        filled = fill_pits(dem)
        filled_dem = filled.data

    # 平地微坡，避免流向全 0
    resolved = resolve_flats(np.asarray(filled_dem, dtype=np.float64))
    flow_dir = compute_flow_direction(dem, resolved)
    flow_acc = compute_flow_accumulation(flow_dir)
    # Tier 3：水文脊 ∪ 多尺度 TPI/剖面
    hydro = extract_ridges(flow_acc, dem)
    multi = multi_scale_ridge_mask(dem, flow_acc)
    ridge_mask = skeletonize(np.asarray(hydro | multi, dtype=bool))
    ridges = vectorize_ridges(ridge_mask, dem, min_length_m=min_length_m)
    ridges = feature_significance_filter(ridges, keep_top=48)
    # Tier 2：合并 + Strahler 分级
    try:
        from engine.core.dragon_strahler import grade_and_merge_ridges

        ridges = grade_and_merge_ridges(ridges, dem)
    except Exception:
        ridges.sort(key=lambda x: -x.feature_significance)
    major = [
        r for r in ridges
        if getattr(r, "strahler_order", 1) >= 2 or r.length_m >= 400
    ][:12]
    if not major:
        major = ridges[:8]
    entrance = find_entrance_point(ridges, dem, water=water)
    entrance_xy = dem.xy(*entrance) if entrance else None
    yaoxia = find_yaoxia(ridges, dem)
    return DragonVeinResult(
        ridge_mask=ridge_mask,
        ridge_lines=ridges,
        flow_acc=flow_acc,
        flow_dir=flow_dir,
        entrance_point=entrance,
        entrance_xy=entrance_xy,
        major_ridges=major,
        yaoxia=yaoxia,
        meta={
            "pipeline": "tier2_3",
            "n_ridges": len(ridges),
            "n_major": len(major),
            "multi_scale": True,
            "strahler": True,
        },
    )


