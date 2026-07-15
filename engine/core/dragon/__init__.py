"""Dragon vein domain package (split from dragon_vein.py)."""
from __future__ import annotations

from engine.core.dragon.types import (
    RidgeLine,
    DragonVeinResult,
    PrimaryDragon,
    RidgePoint,
    IncomingVeinSelection,
)
from engine.core.dragon.hydro import (
    resolve_flats,
    compute_flow_direction,
    compute_flow_accumulation,
    _flow_direction_numpy,
)
from engine.core.dragon.ridge import (
    extract_ridges,
    multi_scale_ridge_mask,
    feature_significance_filter,
    vectorize_ridges,
    light_ridge_mask,
)
from engine.core.dragon.viewshed import (
    sector_viewshed_score,
    dual_signal_anchor,
)
from engine.core.dragon.yaoxia import find_yaoxia, rel_drop_default
from engine.core.dragon.entrance import (
    refine_entrance_on_ordered,
    find_entrance_on_ridge,
    find_entrance_point,
)
from engine.core.dragon.analyze import analyze_dragon_vein
from engine.core.dragon.primary import (
    orient_ridge_to_hole,
    reorient_primary_to_hole,
    select_primary_dragon,
    dist_to_ridge_m,
    dragon_alignment_score,
    score_ridge_as_incoming,
)
from engine.core.dragon.incoming import (
    pick_xuanwu_shaozu_on_ridge,
    beasts_from_primary_dragon,
    select_incoming_vein,
)
from engine.core.dragon.util import _m_per_px_dem, _bearing_rc, _ang_diff

__all__ = [
    "RidgeLine",
    "DragonVeinResult",
    "PrimaryDragon",
    "RidgePoint",
    "IncomingVeinSelection",
    "resolve_flats",
    "compute_flow_direction",
    "compute_flow_accumulation",
    "_flow_direction_numpy",
    "extract_ridges",
    "multi_scale_ridge_mask",
    "feature_significance_filter",
    "vectorize_ridges",
    "light_ridge_mask",
    "sector_viewshed_score",
    "dual_signal_anchor",
    "find_yaoxia",
    "rel_drop_default",
    "refine_entrance_on_ordered",
    "find_entrance_on_ridge",
    "find_entrance_point",
    "analyze_dragon_vein",
    "orient_ridge_to_hole",
    "reorient_primary_to_hole",
    "select_primary_dragon",
    "dist_to_ridge_m",
    "dragon_alignment_score",
    "score_ridge_as_incoming",
    "pick_xuanwu_shaozu_on_ridge",
    "beasts_from_primary_dragon",
    "select_incoming_vein",
    "_m_per_px_dem",
    "_bearing_rc",
    "_ang_diff",
]
