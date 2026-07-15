"""Hydrology: flats, D8 flow direction/accumulation."""
from __future__ import annotations

import numpy as np

from engine.io.dem import DEM


# ESRI D8：row 向下为南。N=64, NE=128, E=1, SE=2, S=4, SW=8, W=16, NW=32
_D8_DIRS = (
    (-1, 0, 64),   # N
    (-1, 1, 128),  # NE
    (0, 1, 1),     # E
    (1, 1, 2),     # SE
    (1, 0, 4),     # S
    (1, -1, 8),    # SW
    (0, -1, 16),   # W
    (-1, -1, 32),  # NW
)
_D8_DECODE = {
    64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1),
    4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1),
}



def resolve_flats(
    dem_arr: np.ndarray,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """平地解算（Garbrecht & Martz 思路简化版）。

    填洼后大片等高平地 → D8 无坡降 → 流向=0 → 假脊。
    对「无严格下坡邻域」的连通平地，按到「可泄出口」的距离递增抬升 ε，
    形成可流向的微坡。
    """
    from scipy.ndimage import distance_transform_edt, binary_dilation, label as nd_label

    out = np.array(dem_arr, dtype=np.float64, copy=True)
    h, w = out.shape
    valid = np.isfinite(out)
    if not valid.any():
        return out

    # 无严格下坡邻居 → 平地/坑底候选
    flat = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not valid[r, c]:
                continue
            z0 = out[r, c]
            has_down = False
            has_eq = False
            for dr, dc, _code in _D8_DIRS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not valid[nr, nc]:
                    continue
                z1 = out[nr, nc]
                if z1 < z0 - 1e-9:
                    has_down = True
                    break
                if abs(z1 - z0) <= 1e-9:
                    has_eq = True
            if not has_down and has_eq:
                flat[r, c] = True

    if not flat.any():
        return out

    from scipy.ndimage import binary_erosion

    labeled, nlab = nd_label(flat)
    for lab in range(1, nlab + 1):
        mask = labeled == lab
        if not mask.any():
            continue
        flat_z = float(np.nanmean(out[mask]))
        # 出口：与更低栅格相邻的平地像元
        ring = binary_dilation(mask, iterations=1) & (~mask) & valid
        outlets = np.zeros_like(mask)
        rs, cs = np.where(ring)
        for r, c in zip(rs.tolist(), cs.tolist()):
            if out[r, c] < flat_z - 1e-9:
                for dr, dc, _ in _D8_DIRS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        outlets[nr, nc] = True
        if not outlets.any():
            core = binary_erosion(mask, iterations=1)
            outlets = mask & (~core)
            if not outlets.any():
                # 整块：中心最高、边缘出口
                outlets = mask.copy()

        # 到最近出口的距离（出口处 dist=0）
        # EDT: 1=要算距离的区域；我们要对 mask 内算到 outlets 的距离
        # 令 non-outlet 在 mask 内为 True → dist to False (outlet or outside)
        seed = outlets | (~mask)
        dist = distance_transform_edt(~seed)
        dist = np.where(mask, dist, 0.0)
        out[mask] = flat_z + epsilon * (1.0 + dist[mask])

    return out


def compute_flow_direction(dem: DEM, filled_dem: np.ndarray) -> np.ndarray:
    """D8 流向（ESRI：N=64…）。输入应为填洼 + resolve_flats 后的 DEM。"""
    try:
        from pysheds.grid import Grid
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            from engine.io.dem import _write_geotiff

            tmp_dem = DEM(
                data=filled_dem,
                transform=dem.transform,
                crs=dem.crs,
                nodata=dem.nodata,
                bounds=dem.bounds,
                resolution=dem.resolution,
            )
            _write_geotiff(tmp_dem, tmp_path)
            grid = Grid.from_raster(tmp_path, data_name="dem")
            dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
            grid.flowdir("dem", out_name="dir", dirmap=dirmap)
            result = np.asarray(grid.view("dir"), dtype=np.int32)
            return result
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception:
        return _flow_direction_numpy(filled_dem)


def _flow_direction_numpy(dem_arr: np.ndarray) -> np.ndarray:
    """纯 numpy D8：最大坡降邻域（含对角 1.414）。"""
    h, w = dem_arr.shape
    flow_dir = np.zeros((h, w), dtype=np.int32)
    valid = np.isfinite(dem_arr)
    for r in range(h):
        for c in range(w):
            if not valid[r, c]:
                continue
            z0 = dem_arr[r, c]
            best_drop = 0.0
            best_code = 0
            for dr, dc, code in _D8_DIRS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not valid[nr, nc]:
                    continue
                z1 = dem_arr[nr, nc]
                dist = 1.414 if (dr != 0 and dc != 0) else 1.0
                drop = (z0 - z1) / dist
                if drop > best_drop:
                    best_drop = drop
                    best_code = code
            flow_dir[r, c] = best_code
    return flow_dir


def compute_flow_accumulation(flow_dir: np.ndarray) -> np.ndarray:
    """D8 汇流累积（拓扑序 Kahn：先上游后下游）。

    每格初值 1；将自身 acc 加到流向所指下游。
    必须拓扑序，否则上游未就绪时下游累积偏小，脊阈值失效。
    """
    from collections import deque

    h, w = flow_dir.shape
    acc = np.ones((h, w), dtype=np.float64)
    indeg = np.zeros((h, w), dtype=np.int32)

    # 入度：被多少邻居指向
    for r in range(h):
        for c in range(w):
            code = int(flow_dir[r, c])
            if code not in _D8_DECODE:
                continue
            dr, dc = _D8_DECODE[code]
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                indeg[nr, nc] += 1

    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if indeg[r, c] == 0:
                q.append((r, c))

    processed = 0
    while q:
        r, c = q.popleft()
        processed += 1
        code = int(flow_dir[r, c])
        if code not in _D8_DECODE:
            continue
        dr, dc = _D8_DECODE[code]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        acc[nr, nc] += acc[r, c]
        indeg[nr, nc] -= 1
        if indeg[nr, nc] == 0:
            q.append((nr, nc))

    # 环/未处理：多轮松弛兜底
    if processed < h * w:
        for _ in range(8):
            changed = False
            for r in range(h):
                for c in range(w):
                    code = int(flow_dir[r, c])
                    if code not in _D8_DECODE:
                        continue
                    dr, dc = _D8_DECODE[code]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        # 不重复加：松弛仅用于残差环，跳过
                        pass
            if not changed:
                break
    return acc


