"""综合评分：兼容门面（实现已拆到 engine.core.scoring.*）。

历史路径保留：
  from engine.core.fengshui_score import score_candidate, find_and_rank_candidates
"""
from __future__ import annotations

from engine.core.scoring.weights import (  # noqa: F401
    DEFAULT_WEIGHTS,
    LUNTOU_ONLY_WEIGHTS,
)
from engine.core.scoring.candidate import (  # noqa: F401
    FusedScore,
    score_candidate,
    _score_yaoxia_for_candidate,
    _bearing_from_to,
    _find_nearest_shan_by_deg,
    _facing_to_sit,
)
from engine.core.scoring.gate import _gate_beasts_for_hole  # noqa: F401
from engine.core.scoring.rank import find_and_rank_candidates  # noqa: F401
from engine.core.scoring.serialize import (  # noqa: F401
    to_geojson,
    to_json,
    _sanitize,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "LUNTOU_ONLY_WEIGHTS",
    "FusedScore",
    "score_candidate",
    "find_and_rank_candidates",
    "to_geojson",
    "to_json",
    "_sanitize",
    "_gate_beasts_for_hole",
]
