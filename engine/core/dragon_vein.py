"""龙脉识别：山脊线提取 + 入首点定位。

参考:
  - 调研报告 03_dragon_vein/00_龙脉识别算法.md
  - 黄培之 2001 论文
  - 特征显著度法
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

from engine.io.dem import DEM


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


def extract_ridges(
    flow_acc: np.ndarray,
    dem: DEM,
    min_length_m: float = 50.0,
    smooth_sigma: float = 0.8,
) -> np.ndarray:
    """提取山脊线栅格（分水岭：上游贡献极小）。

    拓扑序正确后，真源/脊 ≈ acc 接近 1；阈值过宽会满图假脊。
    """
    from scipy.ndimage import gaussian_filter, binary_opening

    contributions = np.clip(flow_acc - 1.0, 0, None)
    if smooth_sigma > 0:
        smoothed = gaussian_filter(contributions.astype(np.float64), sigma=smooth_sigma)
    else:
        smoothed = contributions

    # 上游贡献 < 0.5：近似无上游（真分水线）
    ridge = (smoothed < 0.5) & np.isfinite(dem.data)
    ridge = binary_opening(ridge, iterations=1)
    return skeletonize(ridge)


def multi_scale_ridge_mask(
    dem: DEM,
    flow_acc: np.ndarray | None = None,
    *,
    tpi_min: float = 0.6,
) -> np.ndarray:
    """Tier 3：多信号脊带融合。

    - 水文低累积（若提供 flow_acc）
    - TPI 正脊带（特征显著度代理）
    - 断面局部极大（LANDMARK/剖面极值简化）
    """
    from scipy.ndimage import gaussian_filter, binary_opening, maximum_filter

    data = dem.data.astype(np.float64)
    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros(data.shape, dtype=bool)
    fill = np.where(valid, data, float(np.nanmean(data[valid])))

    # TPI
    local = gaussian_filter(fill, sigma=1.0)
    base = gaussian_filter(fill, sigma=5.0)
    tpi = local - base
    tpi_ridge = valid & (tpi >= tpi_min)

    # 断面极大：行/列 3 邻域峰值
    mx_r = maximum_filter(fill, size=(1, 3), mode="nearest")
    mx_c = maximum_filter(fill, size=(3, 1), mode="nearest")
    prof = valid & ((fill >= mx_r - 1e-6) | (fill >= mx_c - 1e-6)) & (tpi > 0.2)

    hydro = np.zeros(data.shape, dtype=bool)
    if flow_acc is not None and flow_acc.shape == data.shape:
        contrib = np.clip(flow_acc - 1.0, 0, None)
        sm = gaussian_filter(contrib.astype(np.float64), sigma=0.8)
        hydro = valid & (sm < 0.5)

    combined = hydro | (tpi_ridge & prof) | (tpi_ridge & hydro)
    if not combined.any():
        combined = tpi_ridge
    combined = binary_opening(combined, iterations=1)
    return skeletonize(combined)


def feature_significance_filter(
    ridges: list[RidgeLine],
    *,
    min_sig_ratio: float = 0.15,
    keep_top: int = 40,
) -> list[RidgeLine]:
    """Tier 3：按特征显著度裁剪弱脊，保留主脉候选。"""
    if not ridges:
        return []
    sigs = np.array([max(r.feature_significance, 1e-9) for r in ridges], dtype=np.float64)
    thr = float(np.nanpercentile(sigs, 100 * (1.0 - min(min_sig_ratio * 3, 0.85))))
    # 至少保留较长的 top
    ranked = sorted(ridges, key=lambda r: -r.feature_significance)
    kept = [r for r in ranked if r.feature_significance >= thr * 0.5 or r.length_m > 400]
    if len(kept) < 3:
        kept = ranked[: min(keep_top, len(ranked))]
    return kept[:keep_top]


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


def rel_drop_default(dem: DEM, ridge: "RidgeLine") -> float:
    """估算"蜂腰"判定阈值：脊线最高-最低（米）的 50%，最小 2 m。"""
    elevs = []
    for r, c in ridge.coords:
        if 0 <= r < dem.data.shape[0] and 0 <= c < dem.data.shape[1]:
            e = float(dem.data[r, c])
            if np.isfinite(e):
                elevs.append(e)
    if not elevs:
        return 5.0
    drop = max(elevs) - min(elevs)
    return max(2.0, drop * 0.5)


def find_yaoxia(
    ridges: list[RidgeLine],
    dem: DEM,
    *,
    neck_width_m: float = 60.0,
    min_narrowing_ratio: float = 0.55,
) -> list[dict[str, Any]]:
    """识别「蜂腰鹤膝」过峡点。

    在山脊线（ridge）上寻找"突然变窄"的位置：
      - 该点上下游一段长度内，脊线平均宽度显著小于其他段宽度
      - 峡两侧高低差不悬殊（已"脱煞"），即过峡处坡度缓

    返回 list[{ridge_idx, pos_idx, pos_xy, neck_width_m, ...}]
    """
    yaoxia: list[dict[str, Any]] = []
    if not ridges:
        return yaoxia

    mpx = dem.resolution[0] if dem.crs is None else dem.resolution[0]
    from engine.core.terrain_analysis import _is_geographic
    if _is_geographic(dem.crs):
        m_per_deg = 111000.0
        mpx = mpx * m_per_deg
        mpy = dem.resolution[1] * m_per_deg
    else:
        mpy = dem.resolution[1]
    xres, yres = mpx, mpy

    for r_idx, ridge in enumerate(ridges):
        n = len(ridge.coords)
        if n < 20:
            continue

        # 沿脊线计算每点的"横向脊宽"：从脊线点出发，沿垂直脊方向双侧爬，
        # 找到降高度至 threshold 处的距离 ×2 = 局部"等高线半宽"×2。
        elevs: list[float] = []
        widths_m: list[float] = []   # 横向距离（米）
        window = 5
        max_walk_px = 15  # 最大搜索距离（像素）

        # 脊线局部切向量
        for i in range(n):
            r, c = ridge.coords[i]
            if not (0 <= r < dem.data.shape[0] and 0 <= c < dem.data.shape[1]):
                elevs.append(float("nan"))
                widths_m.append(float("nan"))
                continue
            peak_elev = float(dem.data[r, c])  # 【修复】局部声明
            elevs.append(peak_elev)

            # 切向：用 i ± window 中心差分
            il = max(0, i - 1)
            ir = min(n - 1, i + 1)
            rL, cL = ridge.coords[il]
            rR, cR = ridge.coords[ir]
            tx = (cR - cL) * xres
            ty = (rL - rR) * yres
            t_norm = np.hypot(tx, ty)
            if t_norm < 1e-9:
                widths_m.append(float("nan"))
                continue
            tx /= t_norm
            ty /= t_norm

            # 法向（旋转 90°）
            nx = -ty
            ny = tx

            # 双向爬升至"半山腰"——取全局脊线 50% 高差作为阈值
            target_drop = rel_drop_default(dem, ridge)

            left_dist = 0.0
            right_dist = 0.0
            for d in range(1, max_walk_px + 1):
                px_r = int(round(r - ny * d))
                px_c_l = int(round(c + nx * d))  # 左侧
                px_c_r = int(round(c - nx * d))  # 右侧
                if not (0 <= px_r < dem.data.shape[0]):
                    continue
                if 0 <= px_c_l < dem.data.shape[1]:
                    e_l = float(dem.data[px_r, px_c_l])
                    if e_l < peak_elev - target_drop and left_dist == 0:
                        left_dist = d * (xres + yres) / 2.0
                if 0 <= px_c_r < dem.data.shape[1]:
                    e_r = float(dem.data[px_r, px_c_r])
                    if e_r < peak_elev - target_drop and right_dist == 0:
                        right_dist = d * (xres + yres) / 2.0
                if left_dist and right_dist:
                    break
            # 总横向距离：左右之和（如一侧未找到则用单侧 ×2）
            if left_dist and right_dist:
                widths_m.append(left_dist + right_dist)
            elif left_dist:
                widths_m.append(left_dist * 2.0)
            elif right_dist:
                widths_m.append(right_dist * 2.0)
            else:
                widths_m.append(float(max_walk_px * (xres + yres)))

        widths_arr = np.array(widths_m, dtype=np.float64)
        elev_arr = np.array(elevs, dtype=np.float64)

        if not np.isfinite(widths_arr).any():
            continue

        # 全脊线宽度中位数
        med_w = float(np.nanmedian(widths_arr))
        if med_w <= 0:
            continue

        for i in range(window, n - window):
            local_w = widths_arr[i]
            if not np.isfinite(local_w) or local_w <= 0:
                continue
            ratio = local_w / med_w
            if ratio > min_narrowing_ratio:
                continue
            # 峡点位置（地理坐标）
            r_pt, c_pt = ridge.coords[i]
            x_pt, y_pt = dem.xy(r_pt, c_pt)
            # 峡两侧高点差（不超过 60 m 为宜，过大的说明未"脱煞"）
            left_idx = max(0, i - 30)
            right_idx = min(n - 1, i + 30)
            L_elev = float(np.nanmax(elev_arr[left_idx:i + 1]))
            R_elev = float(np.nanmax(elev_arr[i:right_idx + 1]))
            dh = abs(L_elev - R_elev)
            yaoxia.append({
                "ridge_idx": r_idx,
                "pos_idx": i,
                "row": int(r_pt),
                "col": int(c_pt),
                "x": float(x_pt),
                "y": float(y_pt),
                "neck_width_m": round(float(local_w), 1),
                "median_width_m": round(med_w, 1),
                "narrow_ratio": round(float(ratio), 3),
                "side_relief_diff_m": round(dh, 1),
            })
    return yaoxia


def vectorize_ridges(
    ridge_mask: np.ndarray,
    dem: DEM,
    min_length_m: float = 50.0,
) -> list[RidgeLine]:
    """将山脊线栅格矢量化并计算属性。"""
    from skimage.measure import label as ski_label
    from engine.core.terrain_analysis import _is_geographic

    labeled = ski_label(ridge_mask, connectivity=2)
    xres, yres = dem.resolution
    if _is_geographic(dem.crs):
        m_per_unit = 111000.0
    else:
        m_per_unit = 1.0
    min_pixels = int(min_length_m / (min(xres, yres) * m_per_unit))

    ridges: list[RidgeLine] = []
    for region in regionprops(labeled):
        if region.area < min_pixels:
            continue
        coords = region.coords  # (N, 2) (row, col)
        # 计算蜿蜒度
        if len(coords) < 2:
            continue
        actual_length = 0.0
        for i in range(1, len(coords)):
            dr = (coords[i, 0] - coords[i - 1, 0]) * yres * m_per_unit
            dc = (coords[i, 1] - coords[i - 1, 1]) * xres * m_per_unit
            actual_length += np.sqrt(dr ** 2 + dc ** 2)
        ys0 = coords[0, 0] * yres * m_per_unit
        xs0 = coords[0, 1] * xres * m_per_unit
        ys1 = coords[-1, 0] * yres * m_per_unit
        xs1 = coords[-1, 1] * xres * m_per_unit
        straight_length = np.sqrt(
            (ys0 - ys1) ** 2 + (xs0 - xs1) ** 2
        )
        sinuosity = float(actual_length / straight_length) if straight_length > 0 else 1.0

        # 沿线高程统计
        elevs = dem.data[coords[:, 0], coords[:, 1]]
        valid_elevs = elevs[np.isfinite(elevs)]
        if valid_elevs.size == 0:
            continue

        # 特征显著度（粗略：高程均值 × 蜿蜒度 / 长度）
        feature_significance = float(
            np.nanmean(valid_elevs) * sinuosity / max(actual_length, 1)
        )

        ridges.append(
            RidgeLine(
                coords=coords,
                length_m=float(actual_length),
                mean_elevation=float(np.nanmean(valid_elevs)),
                max_elevation=float(np.nanmax(valid_elevs)),
                sinuosity=sinuosity,
                feature_significance=feature_significance,
            )
        )
    return ridges


def find_entrance_on_ridge(
    ridge: RidgeLine,
    dem: DEM,
    *,
    window: int = 7,
    water_dist: np.ndarray | None = None,
) -> tuple[int, int] | None:
    """单条脊上的入首：末端 1/3 内 高程急降 ∩ 曲率极值（+ 近水软加分）。

    调研 §5.6：末端节点 + 局部曲率突变，非简单最低点。
    """
    coords = ridge.coords
    n = len(coords)
    if n < 20:
        if n < 3:
            return None
        # 短脊：取较低端点
        e0 = float(dem.data[int(coords[0, 0]), int(coords[0, 1])])
        e1 = float(dem.data[int(coords[-1, 0]), int(coords[-1, 1])])
        i = 0 if (np.isfinite(e0) and e0 <= e1) else n - 1
        return (int(coords[i, 0]), int(coords[i, 1]))

    elevs = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        r, c = int(coords[i, 0]), int(coords[i, 1])
        if 0 <= r < dem.data.shape[0] and 0 <= c < dem.data.shape[1]:
            elevs[i] = dem.data[r, c]

    # 保证「从头到尾」大致高→低，便于末端=入首侧
    if np.nanmean(elevs[: max(3, n // 5)]) < np.nanmean(elevs[-max(3, n // 5):]):
        coords = coords[::-1].copy()
        elevs = elevs[::-1].copy()

    dh = np.diff(elevs, prepend=elevs[0])
    dh = np.where(np.isfinite(dh), dh, 0.0)
    # 下降强度（正=降）
    ker = np.ones(max(3, window)) / max(3, window)
    drop_signal = -np.convolve(dh, ker, mode="same")
    d2 = np.diff(dh, prepend=dh[0])
    curv_signal = np.abs(np.convolve(d2, ker, mode="same"))

    tail_n = max(8, n // 3)
    start = n - tail_n
    best_i = start
    best_s = -1e18
    for i in range(start, n):
        r, c = int(coords[i, 0]), int(coords[i, 1])
        s = float(drop_signal[i]) + 0.5 * float(curv_signal[i])
        # 近水甜区
        if water_dist is not None and 0 <= r < water_dist.shape[0] and 0 <= c < water_dist.shape[1]:
            dw = float(water_dist[r, c]) if np.isfinite(water_dist[r, c]) else 1e9
            if 40.0 <= dw <= 900.0:
                s += 1.2
            elif dw < 25.0:
                s -= 0.8
        if s > best_s:
            best_s = s
            best_i = i
    return (int(coords[best_i, 0]), int(coords[best_i, 1]))


def find_entrance_point(
    ridges: list[RidgeLine],
    dem: DEM,
    water=None,
) -> tuple[int, int] | None:
    """全局入首：在主要脊线上选「急降+曲率」最优末端点。"""
    if not ridges:
        return None
    wd = None
    try:
        from engine.core.four_beasts_detect import water_distance_rasters
        if water is not None and not getattr(water, "empty", True):
            wd, _ = water_distance_rasters(dem, water, ban_buffer_m=0.0)
    except Exception:
        wd = None

    best = None
    best_score = -1e18
    for ridge in ridges[:15]:
        pt = find_entrance_on_ridge(ridge, dem, water_dist=wd)
        if pt is None:
            continue
        r, c = pt
        elev = float(dem.data[r, c]) if np.isfinite(dem.data[r, c]) else 0.0
        # 脊越长、落差越大越好
        head = ridge.coords[0]
        head_e = float(dem.data[int(head[0]), int(head[1])]) if np.isfinite(
            dem.data[int(head[0]), int(head[1])]
        ) else elev
        drop = max(0.0, head_e - elev)
        sc = drop * max(ridge.sinuosity, 1.0) * min(ridge.length_m / 500.0, 3.0)
        if sc > best_score:
            best_score = sc
            best = pt
    return best


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


# ---------------------------------------------------------------------------
# 先定龙：全图主来龙（不依赖穴点）→ 入首 → 再点穴
# ---------------------------------------------------------------------------

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


def _ridge_end_elev(
    dem: DEM, coords: np.ndarray, at_head: bool
) -> float:
    """脊一端邻域均高。"""
    if coords is None or len(coords) < 1:
        return float("-inf")
    n = len(coords)
    k = max(1, n // 7)
    seg = coords[:k] if at_head else coords[-k:]
    es = []
    for r, c in seg:
        r, c = int(r), int(c)
        if 0 <= r < dem.data.shape[0] and 0 <= c < dem.data.shape[1]:
            e = float(dem.data[r, c])
            if np.isfinite(e):
                es.append(e)
    return float(np.mean(es)) if es else float("-inf")


def _ridge_high_low_ends(
    dem: DEM, coords: np.ndarray
) -> tuple[tuple[int, int], tuple[int, int], float, float]:
    """脊两端粗分：更高端暂作源（仅全图初筛；相对穴时必须 reorient）。"""
    if coords is None or len(coords) < 2:
        return (0, 0), (0, 0), 0.0, 0.0
    h0 = (int(coords[0, 0]), int(coords[0, 1]))
    t0 = (int(coords[-1, 0]), int(coords[-1, 1]))
    eh = _ridge_end_elev(dem, coords, True)
    et = _ridge_end_elev(dem, coords, False)
    if eh >= et:
        return h0, t0, eh, et
    return t0, h0, et, eh


def orient_ridge_to_hole(
    dem: DEM,
    coords: np.ndarray,
    center_row: int,
    center_col: int,
    mpx: float,
    mpy: float,
    water_dist: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """相对穴点定向脊：源=坐后/更远更高/离水更远；入首=近穴且更近水。

    解决「南丘更高就把源钉在南」导致少祖落前的问题。
    不绑绝对东/南/西/北。
    """
    if coords is None or len(coords) < 2:
        return coords, {"ok": False}

    coords = np.asarray(coords, dtype=np.int32)
    e0 = (int(coords[0, 0]), int(coords[0, 1]))
    e1 = (int(coords[-1, 0]), int(coords[-1, 1]))
    elev0 = _ridge_end_elev(dem, coords, True)
    elev1 = _ridge_end_elev(dem, coords, False)

    def _d_hole(r: int, c: int) -> float:
        return float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))

    def _d_water(r: int, c: int) -> float:
        if water_dist is None:
            return 500.0
        if 0 <= r < water_dist.shape[0] and 0 <= c < water_dist.shape[1]:
            v = float(water_dist[r, c])
            return v if np.isfinite(v) else 5000.0
        return 5000.0

    d0, d1 = _d_hole(*e0), _d_hole(*e1)
    w0, w1 = _d_water(*e0), _d_water(*e1)

    # 穴→两端方位
    brg0 = _bearing_rc(center_row, center_col, e0[0], e0[1], mpx, mpy)
    brg1 = _bearing_rc(center_row, center_col, e1[0], e1[1], mpx, mpy)
    # 两端是否大致相对（一前一后）
    ends_opposite = _ang_diff(brg0, brg1) > 100.0

    def _src_score(elev: float, d_h: float, d_w: float) -> float:
        # 源：以「距穴远」为主（祖远），高程次之——防近端突峰抢源
        s = d_h / 180.0 + elev / 90.0 + min(d_w, 2500.0) / 2500.0
        return s

    def _ent_score(elev: float, d_h: float, d_w: float) -> float:
        # 入首：近穴、近水甜区、不宜过高欺穴
        s = 2.5 / (1.0 + d_h / 120.0) - elev / 100.0
        if 40.0 <= d_w <= 1000.0:
            s += 2.2
        elif d_w < 30.0:
            s -= 1.5
        else:
            s += max(0.0, 1.0 - (d_w - 1000.0) / 2000.0)
        return s

    # 方案 A：e0 源 e1 入首；方案 B 对调
    sc_a = _src_score(elev0, d0, w0) + _ent_score(elev1, d1, w1)
    sc_b = _src_score(elev1, d1, w1) + _ent_score(elev0, d0, w0)

    # 硬偏好：更远的一端作源（除非远侧明显更低且近侧也远）
    if d0 > d1 * 1.15:
        sc_a += 2.5
        sc_b -= 1.0
    elif d1 > d0 * 1.15:
        sc_b += 2.5
        sc_a -= 1.0

    if ends_opposite:
        sc_a += 0.5
        sc_b += 0.5

    # 落势：源应不低于入首太多
    if elev0 >= elev1 - 15.0:
        sc_a += 0.6
    else:
        sc_a -= 0.8
    if elev1 >= elev0 - 15.0:
        sc_b += 0.6
    else:
        sc_b -= 0.8

    # 近水一端不宜作源
    if w0 < 80.0 and w1 > 150.0:
        sc_a -= 2.0
        sc_b += 0.5
    if w1 < 80.0 and w0 > 150.0:
        sc_b -= 2.0
        sc_a += 0.5

    # 近端突峰：距穴很近却很高 → 禁止作源
    if d0 < max(d1 * 0.55, 200.0) and elev0 > elev1 + 20.0:
        sc_a -= 3.0
    if d1 < max(d0 * 0.55, 200.0) and elev1 > elev0 + 20.0:
        sc_b -= 3.0

    if sc_a >= sc_b:
        ordered = coords
        src, ent = e0, e1
        e_src, e_ent = elev0, elev1
        choice = "head_source"
        conf = sc_a - sc_b
    else:
        ordered = coords[::-1].copy()
        src, ent = e1, e0
        e_src, e_ent = elev1, elev0
        choice = "tail_source"
        conf = sc_b - sc_a

    flow_az = _bearing_rc(src[0], src[1], ent[0], ent[1], mpx, mpy)
    # 坐：穴看向源
    sit = _bearing_rc(center_row, center_col, src[0], src[1], mpx, mpy)
    facing = (sit + 180.0) % 360.0

    meta = {
        "ok": True,
        "choice": choice,
        "confidence": float(conf),
        "source": src,
        "entrance": ent,
        "source_elev": float(e_src),
        "entrance_elev": float(e_ent),
        "flow_azimuth_deg": float(flow_az),
        "sit_deg": float(sit),
        "facing_deg": float(facing),
        "dist_source_m": float(_d_hole(*src)),
        "dist_entrance_end_m": float(_d_hole(*ent)),
        "ends_opposite": ends_opposite,
    }
    return ordered, meta


def reorient_primary_to_hole(
    dem: DEM,
    primary: PrimaryDragon,
    center_row: int,
    center_col: int,
    water=None,
) -> PrimaryDragon:
    """把已选主龙相对本穴重定向源/入首（防祖落前）。"""
    mpx, mpy = _m_per_px_dem(dem)
    wd = None
    try:
        from engine.core.four_beasts_detect import water_distance_rasters
        if water is not None and not getattr(water, "empty", True):
            wd, _ = water_distance_rasters(dem, water, ban_buffer_m=0.0)
            if not np.isfinite(wd).any():
                wd = None
    except Exception:
        wd = None

    coords = primary.ordered_coords
    if coords is None or len(coords) < 2:
        # 从原 dragon_vein 取脊
        dv = primary.dragon_vein
        if dv is not None and 0 <= primary.ridge_idx < len(dv.ridge_lines):
            coords = dv.ridge_lines[primary.ridge_idx].coords
        else:
            return primary

    ordered, om = orient_ridge_to_hole(
        dem, coords, center_row, center_col, mpx, mpy, water_dist=wd,
    )
    if not om.get("ok"):
        return primary

    sr, sc_ = om["source"]
    er, ec = om["entrance"]
    ex, ey = dem.xy(int(er), int(ec))
    meta = dict(primary.meta or {})
    meta["reoriented_to_hole"] = True
    meta["orient"] = om
    meta["direction_note"] = "source/entrance relative to hole (not absolute compass)"

    return PrimaryDragon(
        ridge_idx=primary.ridge_idx,
        ordered_coords=ordered,
        entrance_row=int(er),
        entrance_col=int(ec),
        entrance_xy=(float(ex), float(ey)),
        source_row=int(sr),
        source_col=int(sc_),
        flow_azimuth_deg=float(om["flow_azimuth_deg"]),
        sit_deg=float(om["sit_deg"]),
        facing_deg=float(om["facing_deg"]),
        score=float(primary.score),
        method=primary.method + "+hole_orient",
        length_m=primary.length_m,
        downhill_m=float(om["source_elev"] - om["entrance_elev"]),
        dragon_vein=primary.dragon_vein,
        meta=meta,
    )


def select_primary_dragon(
    dem: DEM,
    water=None,
    dragon_vein: DragonVeinResult | None = None,
    *,
    min_length_m: float = 120.0,
    anchor_row: int | None = None,
    anchor_col: int | None = None,
) -> PrimaryDragon | None:
    """选主来龙。

    若给定 anchor（热峰/穴），优先选「脊贴近锚点、源在锚点背后更远更高、
    入首近锚点且近水」的脊；否则全图粗选后再靠 reorient。

    不假定北来南入；方向由落势+相对锚点+得水决定。
    """
    if dragon_vein is None:
        dragon_vein = analyze_dragon_vein(dem, min_length_m=min_length_m)
    ridges = list(dragon_vein.ridge_lines or [])
    if not ridges:
        return None

    mpx, mpy = _m_per_px_dem(dem)
    hh, ww = dem.data.shape
    if anchor_row is None:
        anchor_row = hh // 2
    if anchor_col is None:
        anchor_col = ww // 2
    ar, ac = int(np.clip(anchor_row, 0, hh - 1)), int(np.clip(anchor_col, 0, ww - 1))

    wd = None
    try:
        from engine.core.four_beasts_detect import water_distance_rasters

        if water is not None and not getattr(water, "empty", True):
            wd, _ban = water_distance_rasters(dem, water, ban_buffer_m=0.0)
            if not np.isfinite(wd).any():
                wd = None
    except Exception:
        wd = None

    water_face_az: float | None = None
    if water is not None and not getattr(water, "empty", True):
        try:
            from engine.core.four_beasts_detect import _nearest_water_bearing

            nw = _nearest_water_bearing(dem, ar, ac, water)
            if nw is not None:
                water_face_az = float(nw[0])
        except Exception:
            water_face_az = None

    best: PrimaryDragon | None = None
    best_sc = -1e18

    for idx, ridge in enumerate(ridges):
        coords = ridge.coords
        if coords is None or len(coords) < 8:
            continue
        if ridge.length_m < min_length_m * 0.5:
            continue

        # 相对锚点定向源/入首
        ordered, om = orient_ridge_to_hole(
            dem, coords, ar, ac, mpx, mpy, water_dist=wd,
        )
        if not om.get("ok"):
            continue
        sr, sc_ = int(om["source"][0]), int(om["source"][1])
        er, ec = int(om["entrance"][0]), int(om["entrance"][1])
        e_src = float(om["source_elev"])
        e_ent = float(om["entrance_elev"])
        downhill = e_src - e_ent
        flow_az = float(om["flow_azimuth_deg"])
        sit = float(om["sit_deg"])
        facing = float(om["facing_deg"])

        # 脊到锚点距离（越近越好 = 此龙服务于该穴/热峰）
        d_ridge = dist_to_ridge_m(ar, ac, ordered, mpx, mpy)
        d_src = float(om.get("dist_source_m", 1e9))
        d_ent_end = float(om.get("dist_entrance_end_m", 1e9))

        # 入首得水
        water_sc = 0.35
        d_w = None
        if wd is not None and 0 <= er < wd.shape[0] and 0 <= ec < wd.shape[1]:
            d_w = float(wd[er, ec]) if np.isfinite(wd[er, ec]) else 1e9
            if d_w < 25.0:
                water_sc = 0.05
            elif 40.0 <= d_w <= 900.0:
                water_sc = 1.3
            elif d_w < 40.0:
                water_sc = 0.4
            else:
                water_sc = max(0.15, 0.7 * np.exp(-(d_w - 900.0) / 1200.0))
        if wd is not None and 0 <= sr < wd.shape[0] and 0 <= sc_ < wd.shape[1]:
            d_ws = float(wd[sr, sc_]) if np.isfinite(wd[sr, sc_]) else 1e9
            if d_ws < 50.0:
                water_sc -= 1.2  # 源贴水：假龙

        face_water_sc = 0.0
        if water_face_az is not None:
            # 龙气（源→入首）宜与「穴→水」大致同向（面水收气）
            face_water_sc = 1.0 - _ang_diff(flow_az, water_face_az) / 180.0

        # 锚点贴脊 + 源远于入首端（祖远父近）
        near_ridge_sc = max(0.0, 1.0 - d_ridge / 600.0)
        geometry_sc = 0.0
        if d_src > d_ent_end * 1.05:
            geometry_sc += 1.2  # 源更远
        if d_ent_end < 800.0:
            geometry_sc += 0.8  # 入首端靠近锚点
        if downhill > 8.0:
            geometry_sc += 0.6

        length_sc = min(ridge.length_m / 2500.0, 1.3)
        sinu_sc = float(np.clip((ridge.sinuosity - 1.0) / 0.5, 0.0, 1.3))
        down_sc = float(np.clip(downhill / 50.0, -0.5, 1.6))
        edge = min(er, ec, hh - 1 - er, ww - 1 - ec)
        edge_sc = 0.0 if edge < 5 else min(edge / 20.0, 1.0)

        # 置信：相对穴定向置信
        conf = float(om.get("confidence", 0.0))
        # Tier 2：主脉优先（Strahler 高 = 太祖/少祖级）
        so = int(getattr(ridge, "strahler_order", 1) or 1)
        strahler_sc = min(so, 4) / 4.0
        role = getattr(ridge, "role", "branch")
        if role in ("shaozu", "taizu"):
            strahler_sc = max(strahler_sc, 0.75)

        sc = (
            2.8 * near_ridge_sc          # 服务本穴/热峰最重要
            + 1.6 * geometry_sc
            + 1.5 * length_sc
            + 1.0 * sinu_sc
            + 1.4 * max(0.0, down_sc)
            + 1.8 * water_sc
            + 0.8 * max(0.0, face_water_sc)
            + 0.5 * edge_sc
            + 0.4 * min(max(conf, 0.0) / 3.0, 1.0)
            + 0.2 * min(ridge.feature_significance / 50.0, 1.0)
            + 1.2 * strahler_sc          # 真祖级脊加权
        )
        if downhill < 3.0:
            sc -= 0.8
        if d_ridge > 900.0:
            sc -= 1.5  # 离热峰/穴太远的脊降权

        if sc > best_sc:
            best_sc = sc
            ex, ey = dem.xy(er, ec)
            best = PrimaryDragon(
                ridge_idx=idx,
                ordered_coords=ordered,
                entrance_row=er,
                entrance_col=ec,
                entrance_xy=(float(ex), float(ey)),
                source_row=sr,
                source_col=sc_,
                flow_azimuth_deg=float(flow_az),
                sit_deg=float(sit),
                facing_deg=float(facing),
                score=float(sc),
                method="primary_anchor_ridge",
                length_m=float(ridge.length_m),
                downhill_m=float(downhill),
                dragon_vein=dragon_vein,
                meta={
                    "sinuosity": float(ridge.sinuosity),
                    "water_score": water_sc,
                    "face_water_align": face_water_sc,
                    "water_face_az": water_face_az,
                    "dist_water_at_entrance_m": d_w,
                    "source_elev": e_src,
                    "entrance_elev": e_ent,
                    "dist_ridge_to_anchor_m": d_ridge,
                    "dist_source_m": d_src,
                    "dist_entrance_end_m": d_ent_end,
                    "anchor": (ar, ac),
                    "orient": om,
                    "direction_note": "select by proximity to anchor + hole-relative source",
                },
            )

    if best is not None:
        best.dragon_vein = dragon_vein
        # 再 reorient 一次保证一致
        best = reorient_primary_to_hole(dem, best, ar, ac, water=water)
    return best


def dist_to_ridge_m(
    row: int,
    col: int,
    ordered_coords: np.ndarray,
    mpx: float,
    mpy: float,
) -> float:
    """点到脊折线的近似最短距离（米）。"""
    if ordered_coords is None or len(ordered_coords) == 0:
        return 1e9
    best = 1e18
    step = max(1, len(ordered_coords) // 80)
    for i in range(0, len(ordered_coords), step):
        r, c = int(ordered_coords[i, 0]), int(ordered_coords[i, 1])
        d = float(np.hypot((r - row) * mpy, (c - col) * mpx))
        if d < best:
            best = d
    return float(best)


def dragon_alignment_score(
    row: int,
    col: int,
    primary: PrimaryDragon,
    mpx: float,
    mpy: float,
    *,
    entrance_sweet_m: tuple[float, float] = (30.0, 1200.0),
    ridge_max_m: float = 800.0,
) -> dict[str, float]:
    """候选相对主来龙对齐分 0–100。

    宽松：只要贴脊/近入首就给高分；不因略偏入首坐标把热峰打到很低。
    """
    er, ec = primary.entrance_row, primary.entrance_col
    d_ent = float(np.hypot((row - er) * mpy, (col - ec) * mpx))
    d_ridge = dist_to_ridge_m(row, col, primary.ordered_coords, mpx, mpy)
    d_src = float(np.hypot(
        (row - primary.source_row) * mpy, (col - primary.source_col) * mpx,
    ))

    # 贴脊：主信号（半岛上沿北来脊的橙心应接近脊）
    if d_ridge <= 120.0:
        s_ridge = 95.0
    elif d_ridge <= 300.0:
        s_ridge = 85.0 - 20.0 * (d_ridge - 120.0) / 180.0
    elif d_ridge <= ridge_max_m:
        s_ridge = 65.0 * (1.0 - (d_ridge - 300.0) / max(ridge_max_m - 300.0, 1.0))
    else:
        s_ridge = max(15.0, 40.0 * np.exp(-(d_ridge - ridge_max_m) / 600.0))

    # 入首邻域：宽甜区（热峰可略离几何入首端点）
    lo, hi = entrance_sweet_m
    if d_ent <= hi:
        s_ent = 90.0 - 25.0 * min(d_ent, hi) / max(hi, 1.0)
    else:
        s_ent = max(20.0, 70.0 * np.exp(-(d_ent - hi) / 800.0))

    # 不宜压在源上（过远骑龙未结）
    if d_src < 150.0 and d_ent > 400.0:
        s_pos = 45.0
    elif d_src > d_ent:
        # 穴比源端更靠近入首侧 = 结穴位合理
        s_pos = 85.0
    else:
        s_pos = 60.0

    # 综合：贴脊权重大，入首距离次之
    total = 0.50 * s_ridge + 0.30 * s_ent + 0.20 * s_pos
    # 底分：有主龙时不要给灾难性低分（避免热峰被龙分打穿）
    total = max(total, 40.0 if d_ridge < 1000.0 else 25.0)

    return {
        "dragon_align": float(np.clip(total, 0.0, 100.0)),
        "dist_entrance_m": d_ent,
        "dist_ridge_m": d_ridge,
        "dist_source_m": d_src,
    }


# ---------------------------------------------------------------------------
# 来龙筛选 + 脊上切少祖/玄武（相对穴；有全量脊时优先）
# ---------------------------------------------------------------------------

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


def _m_per_px_dem(dem: DEM) -> tuple[float, float]:
    from engine.core.terrain_analysis import _is_geographic

    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs):
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        return xres * 111_000.0 * cos_lat, yres * 111_000.0
    return float(xres), float(yres)


def _bearing_rc(
    r0: int, c0: int, r1: int, c1: int, mpx: float, mpy: float
) -> float:
    """从 (r0,c0) 指向 (r1,c1) 的方位角，北=0 东=90。"""
    dx = (c1 - c0) * mpx
    dy = (r0 - r1) * mpy  # 行号向下 → 北为 row 减小
    return float((np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0)


def _ang_diff(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def light_ridge_mask(
    dem: DEM,
    sigma_local: float = 1.0,
    sigma_base: float = 5.0,
    tpi_min: float = 0.8,
    dilate_px: int = 1,
) -> np.ndarray:
    """轻量脊带：TPI（相对中尺度基底抬升）为正的分水岭带。

    比全量 D8 龙脉快 2 个数量级，适合四象实时路径。
    """
    from scipy.ndimage import gaussian_filter, binary_dilation

    data = dem.data.astype(np.float64)
    valid = np.isfinite(data)
    if not valid.any():
        return np.zeros(data.shape, dtype=bool)
    fill = np.where(valid, data, np.nanmean(data[valid]))
    local = gaussian_filter(fill, sigma=sigma_local)
    base = gaussian_filter(fill, sigma=sigma_base)
    tpi = local - base
    ridge = valid & (tpi >= float(tpi_min))
    if dilate_px > 0:
        ridge = binary_dilation(ridge, iterations=int(dilate_px))
    return ridge


def _ridge_path_fraction(
    ridge_mask: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    n_samples: int = 20,
) -> float:
    """两点间折线落在脊带上的比例（端点邻域略忽略）。"""
    if ridge_mask is None or not np.any(ridge_mask):
        return 0.0
    h, w = ridge_mask.shape
    n = max(6, int(n_samples))
    hit = 0
    tot = 0
    for i in range(1, n - 1):
        t = i / float(n - 1)
        if t < 0.08 or t > 0.92:
            continue
        r = int(round(r0 + t * (r1 - r0)))
        c = int(round(c0 + t * (c1 - c0)))
        if 0 <= r < h and 0 <= c < w:
            tot += 1
            if ridge_mask[r, c]:
                hit += 1
    return float(hit / tot) if tot else 0.0


def _order_ridge_by_dist_to_hole(
    coords: np.ndarray,
    center_row: int,
    center_col: int,
    mpx: float,
    mpy: float,
) -> np.ndarray:
    """按到穴距离排序脊点（近→远），便于从入首向外取父母/少祖。"""
    if coords is None or len(coords) == 0:
        return coords
    dr = (coords[:, 0] - center_row) * mpy
    dc = (coords[:, 1] - center_col) * mpx
    d = np.hypot(dr, dc)
    order = np.argsort(d)
    return coords[order]


def score_ridge_as_incoming(
    dem: DEM,
    ridge_coords: np.ndarray,
    center_row: int,
    center_col: int,
    sit_deg: float,
    facing_deg: float,
    mpx: float,
    mpy: float,
    *,
    max_entrance_m: float = 900.0,
    sector_half: float = 55.0,
) -> dict[str, Any]:
    """评估一条脊是否适合作为本穴来龙。"""
    if ridge_coords is None or len(ridge_coords) < 5:
        return {"score": -1e9, "ok": False}

    cand_elev = float(dem.data[center_row, center_col])
    ordered = _order_ridge_by_dist_to_hole(
        ridge_coords, center_row, center_col, mpx, mpy
    )
    # 入首：距穴最近的脊点
    er, ec = int(ordered[0, 0]), int(ordered[0, 1])
    e_dist = float(
        np.hypot((er - center_row) * mpy, (ec - center_col) * mpx)
    )
    if e_dist > max_entrance_m:
        return {"score": -1e9, "ok": False, "entrance_dist_m": e_dist}

    # 脊点方位：相对穴落在坐向扇区的比例
    in_sector = 0
    n_pts = 0
    elevs = []
    dists = []
    for r, c in ordered:
        r, c = int(r), int(c)
        if not (0 <= r < dem.data.shape[0] and 0 <= c < dem.data.shape[1]):
            continue
        e = float(dem.data[r, c])
        if not np.isfinite(e):
            continue
        d = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
        brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
        n_pts += 1
        elevs.append(e)
        dists.append(d)
        if _ang_diff(brg, sit_deg) <= sector_half:
            in_sector += 1
    if n_pts < 5:
        return {"score": -1e9, "ok": False}

    sector_frac = in_sector / n_pts
    # 远高近低：远端 30% 均高 vs 近端 30%
    k = max(2, n_pts // 3)
    near_e = float(np.mean(elevs[:k]))
    far_e = float(np.mean(elevs[-k:]))
    downhill = far_e - near_e  # >0 势来向穴
    # 龙气走向：最远脊点 → 穴
    fr, fc = int(ordered[-1, 0]), int(ordered[-1, 1])
    flow_az = _bearing_rc(fr, fc, center_row, center_col, mpx, mpy)
    align_face = 1.0 - _ang_diff(flow_az, facing_deg) / 180.0
    align_sit = 1.0 - abs(sector_frac - 1.0)  # 扇区内越多越好

    entrance_score = max(0.0, 1.0 - e_dist / max_entrance_m)
    downhill_score = float(np.clip(downhill / 40.0, -0.5, 1.5))
    length_proxy = float(dists[-1]) if dists else 0.0

    score = (
        2.2 * entrance_score
        + 1.8 * sector_frac
        + 1.5 * max(0.0, downhill_score)
        + 1.4 * align_face
        + 0.4 * min(length_proxy / 2000.0, 1.0)
        + 0.3 * (far_e - cand_elev) / 80.0
    )
    return {
        "score": float(score),
        "ok": sector_frac >= 0.25 and e_dist <= max_entrance_m,
        "entrance": (er, ec),
        "entrance_dist_m": e_dist,
        "sector_frac": sector_frac,
        "downhill_m": downhill,
        "flow_azimuth_deg": flow_az,
        "align_face": align_face,
        "far_elev": far_e,
        "near_elev": near_e,
        "ordered": ordered,
        "length_proxy_m": length_proxy,
    }


def pick_xuanwu_shaozu_on_ridge(
    dem: DEM,
    ordered_coords: np.ndarray,
    center_row: int,
    center_col: int,
    sit_deg: float | None,
    mpx: float,
    mpy: float,
    *,
    xuanwu_dist: tuple[float, float] = (50.0, 500.0),
    shaozu_dist: tuple[float, float] = (500.0, 8000.0),
    xw_dh_sweet: tuple[float, float] = (15.0, 150.0),
    forbid_mask: np.ndarray | None = None,
    sector_half: float | None = 50.0,
    require_sector: bool = True,
    source_first: bool = False,
) -> tuple[RidgePoint | None, RidgePoint | None, dict[str, Any]]:
    """在脊点上切父母山与少祖。

    Args:
        ordered_coords: 若 source_first=True 则为「源→入首」；否则「近穴→远」
        sit_deg / sector_half: 传统坐向扇区约束；主龙路径可 require_sector=False
        source_first: 少祖优先取脊**源端**高峰（峦头：祖在来龙源头）
    """
    cand_elev = float(dem.data[center_row, center_col])
    h, w = dem.data.shape
    meta: dict[str, Any] = {"source_first": source_first, "require_sector": require_sector}
    n = len(ordered_coords) if ordered_coords is not None else 0

    def _ok_cell(r: int, c: int) -> bool:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if forbid_mask is not None and forbid_mask.shape == (h, w) and forbid_mask[r, c]:
            return False
        return bool(np.isfinite(dem.data[r, c]))

    def _mk(r: int, c: int, sc: float) -> RidgePoint:
        elev = float(dem.data[r, c])
        dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
        brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
        return RidgePoint(
            row=r, col=c, elev_m=elev, dist_m=dist,
            bearing_deg=brg, score=sc, on_ridge=True,
        )

    def _in_sector(brg: float) -> bool:
        if not require_sector or sit_deg is None or sector_half is None:
            return True
        return _ang_diff(brg, float(sit_deg)) <= float(sector_half)

    # —— 玄武：距穴父母窗 + 高差甜区（主龙上不强制绝对方位）——
    best_xw: RidgePoint | None = None
    best_xw_s = -1e18
    for r, c in ordered_coords:
        r, c = int(r), int(c)
        if not _ok_cell(r, c):
            continue
        dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
        if dist < xuanwu_dist[0] * 0.6 or dist > xuanwu_dist[1] * 1.25:
            continue
        brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
        if not _in_sector(brg):
            continue
        elev = float(dem.data[r, c])
        rel = elev - cand_elev
        if rel < 2.0:
            continue
        d_lo, d_hi = xuanwu_dist
        if d_lo <= dist <= d_hi:
            s_dist = 1.0 - 0.15 * abs(dist - 0.5 * (d_lo + d_hi)) / max(d_hi - d_lo, 1.0)
        elif dist < d_lo:
            s_dist = 0.5 * dist / d_lo
        else:
            s_dist = 0.7 * np.exp(-(dist - d_hi) / max(d_hi, 1.0))
        dh_lo, dh_hi = xw_dh_sweet
        if dh_lo <= rel <= dh_hi:
            s_dh = 1.0
        elif rel < dh_lo:
            s_dh = 0.5 * rel / max(dh_lo, 1.0)
        else:
            s_dh = max(-0.3, 0.8 * np.exp(-(rel - dh_hi) / 80.0))
        s_az = 1.0
        if sit_deg is not None and sector_half:
            s_az = 1.0 - _ang_diff(brg, float(sit_deg)) / max(float(sector_half), 1.0)
        s = 1.4 * s_dist + 1.3 * s_dh + 0.5 * s_az + 0.3 * min(rel / 80.0, 1.2)
        if s > best_xw_s:
            best_xw_s = s
            best_xw = _mk(r, c, float(s))

    if best_xw is None and not require_sector:
        # 放宽：脊上最近「高于穴」的点
        for r, c in ordered_coords:
            r, c = int(r), int(c)
            if not _ok_cell(r, c):
                continue
            dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
            if dist < 30.0 or dist > max(xuanwu_dist[1] * 1.5, 800.0):
                continue
            elev = float(dem.data[r, c])
            if elev < cand_elev + 1.0:
                continue
            s = 1.0 / max(dist, 1.0) + (elev - cand_elev) / 80.0
            if s > best_xw_s:
                best_xw_s = s
                best_xw = _mk(r, c, float(s))

    if best_xw is None:
        return None, None, {"reason": "no_xuanwu_on_ridge"}

    # —— 少祖 ——
    best_sz: RidgePoint | None = None
    best_sz_s = -1e18
    sz_lo = max(shaozu_dist[0], best_xw.dist_m * 1.15)
    sz_hi = shaozu_dist[1]

    for i, (r, c) in enumerate(ordered_coords):
        r, c = int(r), int(c)
        if not _ok_cell(r, c):
            continue
        dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
        if dist < sz_lo * 0.85 or dist > sz_hi:
            continue
        if dist <= best_xw.dist_m * 1.05:
            continue
        brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
        if require_sector and not _in_sector(brg):
            # 主龙源端可略放宽
            if not (source_first and sit_deg is not None
                    and _ang_diff(brg, float(sit_deg)) <= (sector_half or 55) + 35):
                continue
        elev = float(dem.data[r, c])
        if elev < best_xw.elev_m - 20.0:
            continue
        colinear = 1.0 - _ang_diff(brg, best_xw.bearing_deg) / 60.0
        s_elev = float(np.clip((elev - best_xw.elev_m) / 50.0, -0.3, 1.5))
        s_az = 1.0
        if sit_deg is not None:
            s_az = 1.0 - _ang_diff(brg, float(sit_deg)) / 90.0
        s_dist = float(np.clip(
            1.0 - abs(dist - min(2000.0, 0.5 * (sz_lo + sz_hi))) / 3000.0, 0.0, 1.0
        ))
        # 源端优先：ordered 前段（源）加权
        s_src = 0.0
        if source_first and n > 0:
            # i 越小越靠源
            s_src = 1.2 * (1.0 - i / max(n - 1, 1))
        s = (
            1.0 * max(colinear, 0.0)
            + 1.0 * s_elev
            + 0.5 * max(s_az, 0.0)
            + 0.4 * s_dist
            + s_src
        )
        if s > best_sz_s:
            best_sz_s = s
            best_sz = _mk(r, c, float(s))

    # 主龙：若仍无少祖，取源端有效最高点
    if best_sz is None and source_first and n > 0:
        k = max(1, n // 5)
        best_e = -1e18
        best_rc = None
        for r, c in ordered_coords[:k]:
            r, c = int(r), int(c)
            if not _ok_cell(r, c):
                continue
            elev = float(dem.data[r, c])
            dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
            if dist < max(200.0, best_xw.dist_m * 1.1):
                continue
            if elev > best_e:
                best_e = elev
                best_rc = (r, c)
        if best_rc is not None:
            best_sz = _mk(best_rc[0], best_rc[1], 0.5)

    meta["xuanwu_score"] = best_xw.score
    meta["shaozu_score"] = best_sz.score if best_sz else None
    return best_xw, best_sz, meta


def beasts_from_primary_dragon(
    dem: DEM,
    center_row: int,
    center_col: int,
    primary: PrimaryDragon,
    *,
    forbid_mask: np.ndarray | None = None,
    xuanwu_dist: tuple[float, float] = (50.0, 500.0),
    shaozu_dist: tuple[float, float] = (500.0, 8000.0),
    water=None,
) -> IncomingVeinSelection:
    """峦头正法：坐靠来龙，少祖=龙源，父母=近穴脊峰。

    先相对本穴 reorient 源/入首，避免南丘更高把祖钉在前。
    """
    mpx, mpy = _m_per_px_dem(dem)
    primary = reorient_primary_to_hole(
        dem, primary, center_row, center_col, water=water,
    )
    ordered = primary.ordered_coords  # 源 → 入首（已相对穴）
    if ordered is None or len(ordered) < 3:
        return IncomingVeinSelection(
            xuanwu=None, shaozu=None,
            incoming_azimuth_deg=None, sit_align_deg=None,
            downhill_ok=False, method="primary_empty",
            score=-1e9, meta={},
        )

    # 坐向：穴看向龙源；朝向：对向
    sit = float(primary.sit_deg)
    facing = float(primary.facing_deg)
    flow_az = float(primary.flow_azimuth_deg)

    xw, sz, pm = pick_xuanwu_shaozu_on_ridge(
        dem, ordered, center_row, center_col, sit, mpx, mpy,
        xuanwu_dist=xuanwu_dist,
        shaozu_dist=shaozu_dist,
        forbid_mask=forbid_mask,
        sector_half=80.0,
        require_sector=False,
        source_first=True,
    )

    # 硬约束：少祖必须在「坐后」半区（与朝向差 > 90°），禁止落在前朱雀半区
    def _is_behind(bp: RidgePoint | None) -> bool:
        if bp is None:
            return False
        # 穴→祖 与 朝向 应接近对向（差≈180°）
        return _ang_diff(bp.bearing_deg, facing) > 90.0

    if sz is not None and not _is_behind(sz):
        # 丢弃前侧假祖，强制取源端
        n = len(ordered)
        k = max(2, n // 4)
        best_e = -1e18
        best_rc = None
        for r, c in ordered[:k]:
            r, c = int(r), int(c)
            if forbid_mask is not None and forbid_mask.shape == dem.data.shape:
                if forbid_mask[r, c]:
                    continue
            if not np.isfinite(dem.data[r, c]):
                continue
            elev = float(dem.data[r, c])
            dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
            if dist < 120.0:
                continue
            brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
            if _ang_diff(brg, facing) <= 90.0:
                continue  # 仍在前半区
            if elev > best_e:
                best_e = elev
                best_rc = (r, c, dist, brg, elev)
        if best_rc is not None:
            r, c, dist, brg, elev = best_rc
            sz = RidgePoint(
                row=r, col=c, elev_m=elev, dist_m=dist,
                bearing_deg=brg, score=1.0, on_ridge=True,
            )
            pm = dict(pm or {})
            pm["shaozu_forced_behind"] = True
        else:
            sz = None
            pm = dict(pm or {})
            pm["shaozu_rejected_front"] = True

    if sz is not None:
        sit = float(sz.bearing_deg)
        facing = (sit + 180.0) % 360.0
    elif xw is not None and _is_behind(xw):
        sit = float(xw.bearing_deg)
        facing = (sit + 180.0) % 360.0

    downhill_ok = True
    if sz is not None and xw is not None:
        downhill_ok = sz.elev_m >= xw.elev_m - 20.0

    return IncomingVeinSelection(
        xuanwu=xw,
        shaozu=sz,
        incoming_azimuth_deg=flow_az,
        sit_align_deg=_ang_diff(flow_az, facing),
        downhill_ok=downhill_ok,
        method="primary_dragon_classical",
        score=float(primary.score),
        meta={
            "theory": "坐靠来龙；少祖龙源（相对穴后）；禁前侧假祖",
            "sit_deg": sit,
            "facing_deg": facing,
            "primary_ridge_idx": primary.ridge_idx,
            "primary_flow_az": flow_az,
            "reoriented": bool((primary.meta or {}).get("reoriented_to_hole")),
            "orient": (primary.meta or {}).get("orient"),
            "pick": pm,
        },
    )


def select_incoming_vein(
    dem: DEM,
    center_row: int,
    center_col: int,
    sit_deg: float,
    facing_deg: float | None = None,
    *,
    peaks_mask: np.ndarray | None = None,
    forbid_mask: np.ndarray | None = None,
    ridge_lines: list[RidgeLine] | None = None,
    ridge_mask: np.ndarray | None = None,
    xuanwu_dist: tuple[float, float] = (50.0, 500.0),
    shaozu_dist: tuple[float, float] = (500.0, 8000.0),
    sector_half: float = 55.0,
) -> IncomingVeinSelection:
    """相对穴筛选主来龙，并在脊上切玄武（父母）与少祖。

    优先级：
      1. 已有 ridge_lines（全量龙脉结果）→ 评分为来龙 → 脊上切点
      2. 轻量 TPI 脊带 + 局部峰 → 伪脊链评分 → 切点
      3. 失败则 method=failed，由调用方扇区回退

    龙气走向：少祖→穴 宜接近 facing（由朝向/落势决定，不限定南北）。
    """
    if facing_deg is None:
        facing_deg = (float(sit_deg) + 180.0) % 360.0
    sit_deg = float(sit_deg) % 360.0
    facing_deg = float(facing_deg) % 360.0
    mpx, mpy = _m_per_px_dem(dem)
    h, w = dem.data.shape
    center_row = int(np.clip(center_row, 0, h - 1))
    center_col = int(np.clip(center_col, 0, w - 1))

    # —— 路径 1：矢量化脊线 ——
    if ridge_lines:
        best_sc = -1e18
        best_info = None
        best_idx = -1
        for i, rl in enumerate(ridge_lines):
            info = score_ridge_as_incoming(
                dem, rl.coords, center_row, center_col, sit_deg, facing_deg,
                mpx, mpy, sector_half=sector_half,
            )
            if info.get("ok") and info["score"] > best_sc:
                best_sc = info["score"]
                best_info = info
                best_idx = i
        if best_info is not None:
            ordered = best_info["ordered"]
            xw, sz, pm = pick_xuanwu_shaozu_on_ridge(
                dem, ordered, center_row, center_col, sit_deg, mpx, mpy,
                xuanwu_dist=xuanwu_dist, shaozu_dist=shaozu_dist,
                forbid_mask=forbid_mask, sector_half=sector_half,
            )
            flow_az = best_info.get("flow_azimuth_deg")
            if sz is not None:
                flow_az = _bearing_rc(
                    sz.row, sz.col, center_row, center_col, mpx, mpy
                )
            elif xw is not None:
                flow_az = _bearing_rc(
                    xw.row, xw.col, center_row, center_col, mpx, mpy
                )
            downhill_ok = bool(best_info.get("downhill_m", 0) > -5.0)
            sit_align = (
                _ang_diff(flow_az, facing_deg) if flow_az is not None else None
            )
            return IncomingVeinSelection(
                xuanwu=xw,
                shaozu=sz,
                incoming_azimuth_deg=float(flow_az) if flow_az is not None else None,
                sit_align_deg=float(sit_align) if sit_align is not None else None,
                downhill_ok=downhill_ok,
                method="ridge_lines",
                score=float(best_sc),
                meta={
                    "ridge_idx": best_idx,
                    "entrance_dist_m": best_info.get("entrance_dist_m"),
                    "sector_frac": best_info.get("sector_frac"),
                    "downhill_m": best_info.get("downhill_m"),
                    "pick": pm,
                },
            )

    # —— 路径 2：轻量 TPI 脊 + 峰 ——
    if ridge_mask is None:
        ridge_mask = light_ridge_mask(dem)
    if peaks_mask is not None and peaks_mask.shape == ridge_mask.shape:
        cand_cells = peaks_mask & ridge_mask
    else:
        # 脊带上的局部高点
        from scipy.ndimage import maximum_filter

        data = dem.data
        valid = np.isfinite(data) & ridge_mask
        filled = np.where(valid, data, -np.inf)
        mx = maximum_filter(filled, size=5, mode="nearest")
        cand_cells = valid & (filled == mx) & (filled > -1e17)

    if forbid_mask is not None and forbid_mask.shape == cand_cells.shape:
        cand_cells = cand_cells & (~forbid_mask.astype(bool))

    rs, cs = np.where(cand_cells)
    if rs.size < 3:
        return IncomingVeinSelection(
            xuanwu=None, shaozu=None,
            incoming_azimuth_deg=None, sit_align_deg=None,
            downhill_ok=False, method="failed_no_ridge_peaks",
            score=-1e9, meta={},
        )

    # 伪脊：坐向扇区内候选峰，按距穴排序
    pts = []
    for r, c in zip(rs.tolist(), cs.tolist()):
        dist = float(np.hypot((r - center_row) * mpy, (c - center_col) * mpx))
        if dist < 25.0 or dist > max(shaozu_dist[1], 8000.0):
            continue
        brg = _bearing_rc(center_row, center_col, r, c, mpx, mpy)
        if _ang_diff(brg, sit_deg) > sector_half + 15:
            continue
        elev = float(dem.data[r, c])
        pts.append((dist, r, c, elev, brg))
    if len(pts) < 2:
        return IncomingVeinSelection(
            xuanwu=None, shaozu=None,
            incoming_azimuth_deg=None, sit_align_deg=None,
            downhill_ok=False, method="failed_few_back_peaks",
            score=-1e9, meta={"n_back_peaks": len(pts)},
        )

    pts.sort(key=lambda t: t[0])  # 近→远
    coords = np.array([[p[1], p[2]] for p in pts], dtype=np.int32)

    # 用峰链当「脊」评分
    chain_info = score_ridge_as_incoming(
        dem, coords, center_row, center_col, sit_deg, facing_deg,
        mpx, mpy, sector_half=sector_half, max_entrance_m=1200.0,
    )
    # 共线/脊连通加分：远峰到近峰路径落在 ridge_mask
    path_bonus = 0.0
    if len(pts) >= 2:
        far = pts[-1]
        near = pts[0]
        frac = _ridge_path_fraction(
            ridge_mask, far[1], far[2], near[1], near[2]
        )
        path_bonus = 0.8 * frac
        # 落势：远峰高程 ≥ 近峰
        if far[3] >= near[3] - 5:
            path_bonus += 0.4

    xw, sz, pm = pick_xuanwu_shaozu_on_ridge(
        dem, coords, center_row, center_col, sit_deg, mpx, mpy,
        xuanwu_dist=xuanwu_dist, shaozu_dist=shaozu_dist,
        forbid_mask=forbid_mask, sector_half=sector_half,
    )

    # 若少祖与玄武之间脊连通差，降权但仍可用
    if xw and sz:
        frac_xz = _ridge_path_fraction(
            ridge_mask, sz.row, sz.col, xw.row, xw.col
        )
        pm["ridge_frac_sz_xw"] = frac_xz
        if frac_xz < 0.25:
            # 仍保留点，标记弱连通
            pm["weak_ridge_link"] = True

    flow_az = None
    if sz is not None:
        flow_az = _bearing_rc(sz.row, sz.col, center_row, center_col, mpx, mpy)
    elif xw is not None:
        flow_az = _bearing_rc(xw.row, xw.col, center_row, center_col, mpx, mpy)

    base_sc = float(chain_info.get("score", 0.0)) + path_bonus
    if xw is None:
        return IncomingVeinSelection(
            xuanwu=None, shaozu=None,
            incoming_azimuth_deg=flow_az, sit_align_deg=None,
            downhill_ok=False, method="failed_no_xuanwu",
            score=base_sc, meta={"chain": chain_info, "pick": pm},
        )

    downhill_ok = True
    if sz is not None:
        downhill_ok = sz.elev_m >= xw.elev_m - 15.0
    sit_align = _ang_diff(flow_az, facing_deg) if flow_az is not None else None

    return IncomingVeinSelection(
        xuanwu=xw,
        shaozu=sz,
        incoming_azimuth_deg=float(flow_az) if flow_az is not None else None,
        sit_align_deg=float(sit_align) if sit_align is not None else None,
        downhill_ok=downhill_ok,
        method="light_ridge_peaks",
        score=base_sc + (xw.score if xw else 0) * 0.2,
        meta={
            "n_back_peaks": len(pts),
            "path_bonus": path_bonus,
            "chain_downhill_m": chain_info.get("downhill_m"),
            "pick": pm,
        },
    )

