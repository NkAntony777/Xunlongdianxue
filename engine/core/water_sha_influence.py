"""Water-sha influence risk factors for rendering (and optional scoring reuse).

Theory: 水煞 ≠ 近水. Risk concentrates on cut-foot banks, low terraces,
flow-facing (直冲) or reverse-bow sides — not the open mingtang / jade-belt
interior at mid distance.

Pure functions here are unit-testable without matplotlib.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.water_model import water_sha_elev_factor


# Narrow bank defaults (metres). No-DEM path uses these as hard buffer.
INFLUENCE_BANK_MAJOR_M = 120.0
INFLUENCE_BANK_MINOR_M = 70.0
# With DEM we may sample a bit farther so elev/flow can still mark low rush banks
INFLUENCE_SAMPLE_MAJOR_M = 280.0
INFLUENCE_SAMPLE_MINOR_M = 140.0
# Risk below this → do not draw a purple point
INFLUENCE_DRAW_MIN = 0.22


def water_sha_dist_risk(
    dist_m: float,
    *,
    d_cut_m: float = 80.0,
    d_far_m: float = 220.0,
) -> float:
    """Distance risk ∈ [0, 1]: full near bank, near-zero beyond ~d_far (not km-scale)."""
    if not np.isfinite(dist_m) or dist_m < 0:
        return 1.0
    if dist_m <= 0:
        return 1.0
    if dist_m <= 40.0:
        return 1.0
    if dist_m <= d_cut_m:
        # 40→80: 1.0 → 0.75
        t = (dist_m - 40.0) / max(d_cut_m - 40.0, 1e-6)
        return float(1.0 - 0.25 * t)
    if dist_m <= d_far_m:
        # 80→220: 0.75 → 0.08
        t = (dist_m - d_cut_m) / max(d_far_m - d_cut_m, 1e-6)
        return float(0.75 - 0.67 * np.clip(t, 0.0, 1.0))
    # beyond: very weak residual (will usually fail DRAW_MIN without rush)
    over = (dist_m - d_far_m) / 200.0
    return float(max(0.0, 0.08 * np.exp(-over)))


def water_sha_flow_mod(flow_cos: float | None) -> float:
    """Flow alignment modulator ≥ 0.

    flow_cos = unit_flow · unit(foot→point):
      >0 toward point (rush) → boost
      ~0 lateral → neutral/slight damp
      <0 back/downstream away → damp (not full rush just for near-water)
    """
    if flow_cos is None or not np.isfinite(float(flow_cos)):
        return 1.0
    c = float(np.clip(flow_cos, -1.0, 1.0))
    if c >= 0.55:
        # strong rush face: 1.15 → 1.55
        return float(1.15 + 0.40 * (c - 0.55) / 0.45)
    if c >= 0.15:
        return float(0.85 + 0.30 * (c - 0.15) / 0.40)
    if c >= -0.25:
        # lateral / weak: 0.55 → 0.85
        return float(0.55 + 0.30 * (c + 0.25) / 0.40)
    # back side of flow
    return float(0.35 + 0.20 * max(0.0, c + 1.0))


def water_sha_influence_risk(
    dist_m: float,
    *,
    elev_above_water_m: float | None = None,
    flow_cos: float | None = None,
    reverse_bow: float = 0.0,
) -> float:
    """Combined influence intensity ∈ [0, 1] for purple layer.

    Multiplicative: dist × elev_atten × flow_mod, plus soft reverse-bow lift.
    """
    d_r = water_sha_dist_risk(dist_m)
    e_f = water_sha_elev_factor(elev_above_water_m)
    f_m = water_sha_flow_mod(flow_cos)
    rb = float(np.clip(reverse_bow, 0.0, 1.0))
    # reverse bow: allow a bit more residual even mid-range
    base = d_r * e_f * f_m
    if rb > 0.2:
        base = max(base, d_r * e_f * (0.70 + 0.45 * rb) * 0.85)
    return float(np.clip(base, 0.0, 1.0))


def _sample_elev(dem, x: float, y: float) -> float:
    try:
        return float(dem.sample(x, y))
    except Exception:
        return float("nan")


def nearest_water_geom_metrics(
    x: float,
    y: float,
    water_gdf,
    dem=None,
) -> dict[str, Any]:
    """Nearest water distance, elev above water, flow_cos at (x,y) in gdf CRS."""
    out: dict[str, Any] = {
        "dist_m": float("inf"),
        "elev_above_water_m": None,
        "flow_cos": None,
        "flow_method": "",
    }
    if water_gdf is None or getattr(water_gdf, "empty", True):
        return out
    try:
        from shapely.geometry import Point, LineString, MultiLineString
        from shapely.ops import nearest_points, unary_union

        pt = Point(float(x), float(y))
        # Planar distance in CRS units; for projected CRS = metres
        geoms = [g for g in water_gdf.geometry if g is not None and not g.is_empty]
        if not geoms:
            return out
        union = unary_union(geoms)
        foot = nearest_points(pt, union)[1]
        dist = float(pt.distance(foot))
        out["dist_m"] = dist

        # Find a line segment near the foot for flow direction
        line = None
        best_d = 1e18
        for g in geoms:
            try:
                d = g.distance(foot)
            except Exception:
                continue
            if d < best_d:
                best_d = d
                if g.geom_type == "LineString":
                    line = g
                elif g.geom_type == "MultiLineString":
                    line = min(g.geoms, key=lambda lg: lg.distance(foot))
                elif g.geom_type in ("Polygon", "MultiPolygon"):
                    b = g.boundary
                    if b.geom_type == "LineString":
                        line = b
                    elif b.geom_type == "MultiLineString" and len(b.geoms):
                        line = min(b.geoms, key=lambda lg: lg.distance(foot))

        elev_fn = None
        if dem is not None:
            from engine.core.water_model import _elev_fn_for_projected_xy
            # Prefer dem.sample in dem CRS; if gdf CRS matches dem use sample
            crs_dem = str(getattr(dem, "crs", "") or "")
            crs_gdf = str(getattr(water_gdf, "crs", "") or "")
            if crs_gdf and crs_dem and crs_gdf.upper() == crs_dem.upper():
                elev_fn = lambda xx, yy: _sample_elev(dem, xx, yy)
            else:
                elev_fn = _elev_fn_for_projected_xy(dem)

        z_pt = elev_fn(float(x), float(y)) if elev_fn else float("nan")
        z_w = elev_fn(float(foot.x), float(foot.y)) if elev_fn else float("nan")
        if np.isfinite(z_pt) and np.isfinite(z_w):
            out["elev_above_water_m"] = float(z_pt - z_w)

        if line is not None and line.length > 1e-6:
            from engine.core.water_model import _flow_tangent_dem

            L = float(line.length)
            s_star = float(line.project(foot))
            ds = max(30.0, min(L * 0.08, 120.0))
            s0 = max(0.0, s_star - ds)
            s1 = min(L, s_star + ds)
            p0 = line.interpolate(s0)
            p2 = line.interpolate(s1)
            tx, ty, method = _flow_tangent_dem(
                line, s_star, L, elev_fn,
                p0_xy=(float(p0.x), float(p0.y)),
                p2_xy=(float(p2.x), float(p2.y)),
            )
            rdx = float(x) - float(foot.x)
            rdy = float(y) - float(foot.y)
            rn = max(np.hypot(rdx, rdy), 1e-9)
            cos = float(tx * (rdx / rn) + ty * (rdy / rn))
            out["flow_cos"] = cos
            out["flow_method"] = method
    except Exception:
        pass
    return out


def water_sha_influence_at_xy(
    x: float,
    y: float,
    water_gdf,
    dem=None,
) -> float:
    """End-to-end risk at a map point using shipped metrics + influence formula."""
    m = nearest_water_geom_metrics(x, y, water_gdf, dem=dem)
    return water_sha_influence_risk(
        float(m["dist_m"]),
        elev_above_water_m=m.get("elev_above_water_m"),
        flow_cos=m.get("flow_cos"),
    )
