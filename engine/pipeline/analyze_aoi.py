"""AOI analysis use-case (orchestration, no HTTP).

Callers: api.routers.layers / future CLI.
Keeps steps explicit for testability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from engine.core.field.qi import compute_score_grid, find_score_peak, score_peak_xy
from engine.core.four_beasts_detect import detect_four_beasts
from engine.core.fengshui_score import find_and_rank_candidates
from engine.io.dem import DEM


@dataclass
class AnalyzeAoiResult:
    dem: DEM
    water: Any = None
    score_grid: Optional[np.ndarray] = None
    peak_rowcol: Optional[tuple[int, int]] = None
    peak_xy: Optional[tuple[float, float]] = None
    peak_score: Optional[float] = None
    four_beasts: Any = None
    four_beasts_meta: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    context: dict = field(default_factory=dict)


def analyze_aoi(
    dem: DEM,
    water=None,
    *,
    top_k: int = 10,
    min_score: int = 0,
    sample_step: int = 4,
    require_beasts: bool = True,
    dragon_vein=None,
    primary_dragon=None,
) -> AnalyzeAoiResult:
    """Compute score field, peak, four beasts, ranked candidates.

    Does not render PNGs (that stays in rendering / layers router for now).
    """
    grid = compute_score_grid(dem, sample_step=sample_step, water=water)
    peak_info = score_peak_xy(dem, grid)
    if peak_info is not None:
        (pr, pc), peak_xy, peak_score = peak_info
    else:
        pr = pc = None
        peak_xy = None
        peak_score = None
        peak = find_score_peak(grid)
        if peak is not None:
            pr, pc, peak_score = int(peak[0]), int(peak[1]), float(peak[2])
            try:
                peak_xy = dem.xy(pr, pc)
            except Exception:
                peak_xy = None

    fb = None
    fb_meta: dict = {}
    if pr is not None and pc is not None:
        try:
            fb = detect_four_beasts(
                dem,
                center_row=pr,
                center_col=pc,
                water=water,
                dragon_vein=dragon_vein,
                primary_dragon=primary_dragon,
            )
            fb_meta = getattr(fb, "meta", None) or {}
        except Exception as e:
            fb_meta = {"error": str(e)}

    ranked, ctx = find_and_rank_candidates(
        dem,
        water=water,
        top_k=top_k,
        min_score=min_score,
        dragon_vein=dragon_vein,
        primary_dragon=primary_dragon,
        return_context=True,
        require_beasts=require_beasts,
    )

    return AnalyzeAoiResult(
        dem=dem,
        water=water,
        score_grid=grid,
        peak_rowcol=(pr, pc) if pr is not None else None,
        peak_xy=peak_xy if isinstance(peak_xy, tuple) else (
            (float(peak_xy[0]), float(peak_xy[1])) if peak_xy is not None else None
        ),
        peak_score=float(peak_score) if peak_score is not None else None,
        four_beasts=fb,
        four_beasts_meta=fb_meta,
        candidates=list(ranked or []),
        context=dict(ctx or {}),
    )
