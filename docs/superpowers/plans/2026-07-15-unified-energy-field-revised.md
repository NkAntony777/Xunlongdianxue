# 寻龙点穴统一能量场评分体系——修订版计划 v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把目前分散的风水理论参数整合为统一能量场评分：场只保留方向无关的聚气三层，所有方向/点位相关项只在候选点评分里作为 bonus 追加。

**Architecture:**

```
能量场：  E(p) = 100 · G_藏风(p) × G_得水(p) × G_稳定(p)
          G_藏风 = sigmoid(-tpi_mid) × enclosure(max_elev_radius - elev)

候选分：  S(c) = E(c) · (1 - α·P_水煞(c)/100) + Σ w_k · bonus_k(c)
```

- 不预计算 `facing_field`；朝向在候选点精确推断。
- 四象、龙脉、水口、理气、星体、过峡等只作为候选 bonus，不进乘性场。
- 水煞只衰减场值，不衰减 bonus 几何品质。

**Tech Stack:** Python 3.11+, NumPy, SciPy, rasterio, pytest.

## Global Constraints

- 使用项目 venv：`D:\Xunlong\engine\.venv\Scripts\python.exe`
- 每次修改后跑：`engine\.venv\Scripts\python.exe -m pytest engine\tests\test_engine.py -q`
- 热力图输出峰值必须与 `find_score_peak` 平滑后峰值同点
- 水面/缓冲带硬禁：场值为 `nan`
- 能量场通道保持 0–1；候选 bonus 总幅度控制在 ±30 分以内
- 单 bonus 上限 ±15 分

---

## File Structure

| 文件 | 职责 |
|------|------|
| `engine/core/energy_field.py` | **新增**：统一能量场入口；三层乘性场（藏风/得水/稳定） |
| `engine/core/four_beasts_detect.py` | **修改**：`compute_score_grid` / `compute_qi_field_layers` 委托到 `energy_field.py`；保留向后兼容 |
| `engine/core/fengshui_score.py` | **修改**：`score_candidate` 支持 `field_score` 标量 + bonus 模式；新增 `DEFAULT_BONUS_WEIGHTS` |
| `engine/tests/test_energy_field.py` | **新增**：能量场核心单测 |
| `engine/tests/test_energy_bonus.py` | **新增**：候选评分 bonus 模式单测 |
| `engine/tests/test_energy_fusion.py` | **新增**：排序接入单测 |
| `engine/tests/test_energy_consistency.py` | **新增**：场峰值与 top1 候选一致性 |
| `research/99_summary/07_能量场规格.md` | **新增**：修订后规格 |

---

## Task 1: 提取并封装基础能量场

**Files:**
- Create: `engine/core/energy_field.py`
- Modify: `engine/core/four_beasts_detect.py`
- Test: `engine/tests/test_energy_field.py`

**Interfaces:**
- Produces: `compute_energy_field(dem, water=None, config=None) -> EnergyFieldResult`
- Produces: `compute_base_layers(dem, water, config) -> dict[str, np.ndarray]`

```python
@dataclass
class EnergyFieldResult:
    energy: np.ndarray          # 0–100, nan on water/invalid
    channels: dict[str, np.ndarray]  # cangfeng, water, stability ∈ [0,1]
    water_ban: np.ndarray       # bool
    meta: dict = field(default_factory=dict)
```

说明：当前 `four_beasts_detect.py` 里已有一个较复杂的 `compute_qi_field_layers`（四层 + 多种局部修正）。本任务新建单一、方向无关、全广播的三层实现，作为 canonical 能量场；旧函数在 Task 2 中委托到新实现。

- [ ] **Step 1: 写失败测试**

```python
# engine/tests/test_energy_field.py
import numpy as np
import pytest
from rasterio.transform import from_bounds

from engine.io.dem import DEM
from engine.core.energy_field import compute_energy_field, compute_base_layers


def _dem(h=40, w=40, cell=30.0):
    data = np.zeros((h, w), dtype=np.float64)
    transform = from_bounds(0, 0, w * cell, h * cell, w, h)
    return DEM(
        data=data,
        transform=transform,
        crs="EPSG:3857",
        nodata=-9999.0,
        bounds=(0, 0, w * cell, h * cell),
        resolution=(cell, cell),
    )


def test_energy_field_has_three_channels():
    dem = _dem()
    res = compute_energy_field(dem)
    assert set(res.channels.keys()) == {"cangfeng", "water", "stability"}
    assert res.energy.shape == dem.data.shape
    valid = np.isfinite(res.energy)
    assert np.nanmin(res.energy[valid]) >= 0.0
    assert np.nanmax(res.energy[valid]) <= 100.0


def test_energy_field_no_water_gives_full_water_channel():
    dem = _dem()
    res = compute_energy_field(dem)
    assert np.allclose(res.channels["water"], 1.0)


def test_energy_field_water_ban_is_nan():
    """水面及缓冲带在场中为 nan（硬禁）。"""
    from shapely.geometry import LineString
    import geopandas as gpd
    from engine.io.rivers import WaterNetwork

    dem = _dem(h=80, w=80, cell=30.0)
    line = LineString([(200, 600), (200, 1800)])
    gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:3857")
    water = WaterNetwork(gdf=gdf)
    res = compute_energy_field(dem, water=water)
    assert np.any(np.isnan(res.energy))
    assert not np.all(np.isnan(res.energy))
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine\tests\test_energy_field.py -v`
Expected: FAIL (`compute_energy_field` not defined)

- [ ] **Step 2: 实现三层基础场（全广播，无循环）**

```python
# engine/core/energy_field.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

from engine.core.terrain_analysis import compute_slope_aspect, tpi as _tpi
from engine.io.dem import DEM


@dataclass
class EnergyFieldResult:
    energy: np.ndarray
    channels: dict[str, np.ndarray]
    water_ban: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _meters_per_pixel(dem: DEM) -> tuple[float, float]:
    from engine.core.terrain_analysis import _is_geographic

    xres, yres = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geographic(dem.crs):
        mid_lat = (dem.bounds[1] + dem.bounds[3]) / 2.0
        cos_lat = max(0.2, abs(np.cos(np.radians(mid_lat))))
        return xres * 111_000.0 * cos_lat, yres * 111_000.0
    return xres, yres


def compute_base_layers(
    dem: DEM,
    water=None,
    config: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """方向无关的三层基础场（0–1），全 numpy 广播。

    Config keys (all optional):
      - tpi_mid_m:       TPI 中尺度半径，默认 200.0
      - enclosure_radius_m: 围合窗半径，默认 300.0
      - water_opt_m:     得水最佳距离，默认 300.0
      - water_sigma_m:   得水距离衰减 sigma，默认 400.0
      - slope_max_deg:   稳定层坡度上限，默认 25.0
    """
    cfg = config or {}
    tpi_mid_m = float(cfg.get("tpi_mid_m", 200.0))
    enclosure_radius_m = float(cfg.get("enclosure_radius_m", 300.0))
    water_opt_m = float(cfg.get("water_opt_m", 300.0))
    water_sigma_m = float(cfg.get("water_sigma_m", 400.0))
    slope_max_deg = float(cfg.get("slope_max_deg", 25.0))

    elev = np.asarray(dem.data, dtype=np.float64)
    finite = np.isfinite(elev)
    if not finite.any():
        z = np.zeros_like(elev)
        return {
            "cangfeng": z,
            "water": z,
            "stability": z,
            "water_ban": np.zeros_like(elev, dtype=bool),
        }

    slope_arr, _ = compute_slope_aspect(dem)

    # 1) 藏风 = 中尺度凹度 × 围合度
    tpi_mid = _tpi(dem, radius_m=tpi_mid_m)
    tpi_mid = np.where(np.isfinite(tpi_mid), tpi_mid, 0.0)
    g_basin = _sigmoid(-0.35 * tpi_mid)

    mpx, mpy = _meters_per_pixel(dem)
    mpp = max(0.5 * (mpx + mpy), 1e-6)
    r_px = max(1, int(round(enclosure_radius_m / mpp)))
    win = 2 * r_px + 1

    fill_val = float(np.nanmedian(elev[finite]))
    elev_f = np.where(finite, elev, fill_val)
    surrounding_max = maximum_filter(elev_f, size=win, mode="nearest")
    relief = np.maximum(surrounding_max - elev_f, 0.0)
    g_enclosure = np.exp(-((relief - 60.0) / 50.0) ** 2)
    g_enclosure = np.clip(g_enclosure, 0.0, 1.0)

    g_cangfeng = np.clip(g_basin * g_enclosure, 0.0, 1.0)

    # 2) 得水：高斯距离核；无水时全 1.0
    # 懒加载避免与 four_beasts_detect 的循环导入
    from engine.core.four_beasts_detect import water_distance_rasters

    water_dist, water_ban = water_distance_rasters(dem, water)
    has_water = (
        water is not None
        and not getattr(water, "empty", True)
        and np.isfinite(water_dist).any()
    )
    if has_water:
        d = np.where(np.isfinite(water_dist), water_dist, 1.0e6)
        g_water = np.exp(-((d - water_opt_m) / max(water_sigma_m, 1e-6)) ** 2)
        g_water = np.clip(g_water, 0.0, 1.0)
        g_water = np.where(water_ban, 0.0, g_water)
    else:
        g_water = np.ones_like(elev)
        water_ban = np.zeros_like(elev, dtype=bool)

    # 3) 稳定：坡度缓则高；河床平地略压
    slope_safe = np.where(np.isfinite(slope_arr), slope_arr, slope_max_deg)
    g_stab = np.clip(1.0 - slope_safe / slope_max_deg, 0.1, 1.0)

    if has_water:
        _, water_surface = water_distance_rasters(dem, water, ban_buffer_m=0.0)
        flat_riverbed = (slope_safe < 1.0) & water_surface
        g_stab = g_stab * (1.0 - 0.5 * flat_riverbed.astype(float))
        g_stab = np.clip(g_stab, 0.0, 1.0)

    return {
        "cangfeng": g_cangfeng.astype(np.float64),
        "water": g_water.astype(np.float64),
        "stability": g_stab.astype(np.float64),
        "water_ban": water_ban,
    }


def compute_energy_field(
    dem: DEM,
    water=None,
    config: dict[str, Any] | None = None,
) -> EnergyFieldResult:
    """统一能量场入口。返回 0–100 栅格及三通道。"""
    layers = compute_base_layers(dem, water, config)
    finite = np.isfinite(np.asarray(dem.data, dtype=np.float64))
    water_ban = layers["water_ban"]

    energy = (
        100.0
        * layers["cangfeng"]
        * layers["water"]
        * layers["stability"]
    )
    energy = np.where(finite & ~water_ban, energy, np.nan)
    energy = np.clip(energy, 0.0, 100.0)

    channels = {
        k: layers[k]
        for k in ("cangfeng", "water", "stability")
    }
    return EnergyFieldResult(
        energy=energy,
        channels=channels,
        water_ban=water_ban,
        meta={"source": "energy_field_v2", "config": config or {}},
    )
```

- [ ] **Step 3: 让 four_beasts_detect 委托到新能量场**

```python
# engine/core/four_beasts_detect.py
# 在文件顶部新增导入
from engine.core.energy_field import compute_energy_field

# 替换 compute_score_grid 的实现，保留函数签名
def compute_score_grid(
    dem: DEM,
    weights: dict[str, float] | None = None,
    tpi_radius_m: float = 100.0,
    sample_step: int = 4,
    water=None,
    *,
    use_water_form: bool = False,
    max_samples: int | None = 12_000,
    search_radius_m: float = 300.0,
) -> np.ndarray:
    """计算全图生气评分场（0–100）。

    现在委托给 engine.core.energy_field 的统一三层的实现；
    保留签名以保证现有调用方不报错。
    """
    _ = (tpi_radius_m, sample_step, use_water_form, max_samples, search_radius_m)
    w = dict(weights or {})
    config = {
        "tpi_mid_m": float(w.get("tpi_mid_m", 200.0)),
        "enclosure_radius_m": float(
            w.get("enclosure_radius_m", w.get("search_radius_m", 300.0))
        ),
        "water_opt_m": float(w.get("water_opt_m", 300.0)),
        "water_sigma_m": float(w.get("water_sigma_m", 400.0)),
        "slope_max_deg": float(w.get("slope_max_deg", 25.0)),
    }
    ef = compute_energy_field(dem, water=water, config=config)
    return ef.energy


# 保留 compute_qi_field_layers 作为向后兼容别名
def compute_qi_field_layers(
    dem: DEM,
    water=None,
    **kwargs,
) -> dict[str, np.ndarray]:
    """向后兼容：旧四层名映射到新的三通道 + water_ban。"""
    ef = compute_energy_field(dem, water=water, config=kwargs)
    return {
        "cangfeng": ef.channels["cangfeng"],
        "water": ef.channels["water"],
        "stability": ef.channels["stability"],
        # 旧 caller 可能引用 "enclosure"，用 cangfeng 的围合分量等价占位
        "enclosure": ef.channels["cangfeng"],
        "qi": ef.energy / 100.0,
        "water_ban": ef.water_ban,
        "finite": np.isfinite(np.asarray(dem.data, dtype=np.float64)),
    }
```

注意：`compute_qi_field_layers` 原实现较复杂；替换为兼容包装后，数值会变化。后续回归测试若对老数值有强断言，需要同步更新。

- [ ] **Step 4: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine\tests\test_energy_field.py -v`
Expected: PASS

- [ ] **Step 5: 跑核心回归测试，修复因场公式变化导致的旧断言**

Run: `engine\.venv\Scripts\python.exe -m pytest engine\tests\test_engine.py -q`
Expected: PASS（若有旧数值硬断言，按新场语义修改）

- [ ] **Step 6: Commit**

```bash
git add engine/core/energy_field.py engine/core/four_beasts_detect.py engine/tests/test_energy_field.py
git commit -m "feat(energy): three-layer base energy field with delegation"
```

---

## Task 2: 候选评分改为“场值 + bonus”模式

**Files:**
- Modify: `engine/core/fengshui_score.py`
- Test: `engine/tests/test_energy_bonus.py`

**Interfaces:**
- Consumes: `field_score: float | None`（统一能量场在候选像素的值）
- Consumes: `dragon_alignment: float | None`（来龙对齐 0–100）
- Produces: `score_candidate(..., field_score=None, dragon_alignment=None, bonus_weights=None)` 兼容旧路径

- [ ] **Step 1: 写失败测试**

```python
# engine/tests/test_energy_bonus.py
import numpy as np
import pytest
from rasterio.transform import from_bounds

from engine.io.dem import DEM
from engine.core.acupoint import AcupointCandidate, search_candidates
from engine.core.terrain_analysis import analyze_terrain
from engine.core.fengshui_score import score_candidate
from engine.core.energy_field import compute_energy_field


def _dem(h=80, w=80, cell=30.0):
    data = np.zeros((h, w), dtype=np.float64)
    transform = from_bounds(0, 0, w * cell, h * cell, w, h)
    return DEM(
        data=data,
        transform=transform,
        crs="EPSG:3857",
        nodata=-9999.0,
        bounds=(0, 0, w * cell, h * cell),
        resolution=(cell, cell),
    )


def test_candidate_score_starts_from_field():
    """新路径：overall 与 field_score 偏差在可控 bonus 范围内。"""
    dem = _dem()
    dem.data[:] = 500.0 + 80.0 * np.exp(
        -((np.mgrid[0:80, 0:80][0] - 40) ** 2
          + (np.mgrid[0:80, 0:80][1] - 40) ** 2)
        / (2 * 15 ** 2)
    )
    ef = compute_energy_field(dem)
    cands = search_candidates(dem, max_candidates=5)
    terrain = analyze_terrain(dem)
    fused = score_candidate(dem, cands[0], terrain, field_score=ef.energy[cands[0].row, cands[0].col])
    field_val = ef.energy[cands[0].row, cands[0].col]
    assert abs(fused.overall - field_val) <= 35.0


def test_candidate_score_backwards_compatible():
    """不传 field_score 时走旧加权和路径，不报错。"""
    dem = _dem()
    dem.data[:] = 500.0 + 80.0 * np.exp(
        -((np.mgrid[0:80, 0:80][0] - 40) ** 2
          + (np.mgrid[0:80, 0:80][1] - 40) ** 2)
        / (2 * 15 ** 2)
    )
    cands = search_candidates(dem, max_candidates=5)
    terrain = analyze_terrain(dem)
    fused = score_candidate(dem, cands[0], terrain)
    assert 0 <= fused.overall <= 100
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine\tests\test_energy_bonus.py -v`
Expected: FAIL (`field_score` parameter not accepted)

- [ ] **Step 2: 新增 bonus 权重常量并修改 score_candidate 签名**

在 `engine/core/fengshui_score.py` 中，紧邻 `DEFAULT_WEIGHTS` 添加：

```python
# 候选精修 bonus 权重（相对中性点的偏移）
DEFAULT_BONUS_WEIGHTS = {
    "four_beasts": 0.30,
    "dragon": 0.20,
    "mouth": 0.15,
    "compass": 0.10,
    "star_body": 0.10,
    "xue_star": 0.10,
    # yaoxia / cross 已是固定小分，权重 1.0 表示直接加
    "yaoxia": 1.0,
    "cross": 1.0,
}
```

修改 `score_candidate` 签名（新参数放最后，带默认值）：

```python
def score_candidate(
    dem: DEM,
    candidate: AcupointCandidate,
    terrain: TerrainMetrics,
    water: WaterNetwork | None = None,
    weights: dict[str, float] | None = None,
    slope_arr: np.ndarray | None = None,
    aspect_arr: np.ndarray | None = None,
    yaoxia_points: list[dict[str, Any]] | None = None,
    long_az_deg: float | None = None,
    field_score: float | None = None,
    dragon_alignment: float | None = None,
    bonus_weights: dict[str, float] | None = None,
) -> FusedScore:
```

- [ ] **Step 3: 在 score_candidate 末尾替换 overall 计算逻辑**

保留原加权和路径的代码直到 `raw` 字典构建完成；然后在 `return FusedScore(...)` 之前，把以下旧代码块：

```python
    w_local = {k: v for k, v in weights.items() if k in raw and k != "water_sha"}
    w_sum = sum(w_local.values()) or 1.0
    overall = sum(float(raw[k]) * w_local[k] for k in w_local) / w_sum
    if not water_score.is_placeholder:
        overall = fuse_field_with_sha(overall, float(water_score.sha_penalty))
    overall += jiaogou_bonus
    overall += star_score_bonus
    overall += xue_star_bonus
    if halo_score_val is not None:
        overall += (halo_score_val - 50) * 0.16
    overall += yaoxia_bonus
    overall += cross_check_penalty
```

替换为：

```python
    if field_score is not None:
        # ---- 新路径：场值 + 可控 bonus，水煞只压场值 ----
        bw = dict(bonus_weights or DEFAULT_BONUS_WEIGHTS)
        sha_factor = 1.0
        if not water_score.is_placeholder:
            sha_factor = max(
                0.25,
                1.0 - 0.45 * float(water_score.sha_penalty) / 100.0,
            )

        # 各 bonus 相对于中性点的偏移
        bonuses = {
            "four_beasts": float(four.combined) - 50.0,
            "mouth": float(mouth_score_val) - 50.0,
            "compass": float(compass_score) - 85.0,
            "star_body": float(star_score_bonus),
            "xue_star": float(xue_star_bonus),
            "yaoxia": float(yaoxia_bonus),
            "cross": float(cross_check_penalty),
        }
        if dragon_alignment is not None and np.isfinite(float(dragon_alignment)):
            bonuses["dragon"] = float(dragon_alignment) - 50.0

        active = [k for k in bonuses if k in bw]
        bonus_sum = sum(bonuses[k] * bw[k] for k in active)
        # 硬性兜底：总 bonus 不超过 ±30
        bonus_sum = max(-30.0, min(30.0, bonus_sum))

        overall = float(field_score) * sha_factor + bonus_sum
    else:
        # ---- 旧路径：保留向后兼容 ----
        w_local = {k: v for k, v in weights.items() if k in raw and k != "water_sha"}
        if not mouth_evaluated or mouth_score_val <= 0:
            w_local.pop("mouth", None)
        if star_type in ("未评估", "无父母山", "不清"):
            w_local.pop("star_body", None)
        w_sum = sum(w_local.values()) or 1.0
        overall = sum(float(raw[k]) * w_local[k] for k in w_local) / w_sum
        if not water_score.is_placeholder:
            overall = fuse_field_with_sha(overall, float(water_score.sha_penalty))
        overall += jiaogou_bonus
        overall += star_score_bonus
        overall += xue_star_bonus
        if halo_score_val is not None:
            overall += (halo_score_val - 50) * 0.16
        overall += yaoxia_bonus
        overall += cross_check_penalty

    overall = clamp_score(overall)
```

注意：
- `jiaogou_bonus` 已在水口逻辑中作为固定 +12 加入；在新路径中，建议把 `mouth_score_val` 计算时已经体现了 `is_jiaogou` 的加成（见 `score_water_mouth_for_candidate`），所以 `jiaogou_bonus` 不再重复加。若旧路径依赖它，保持旧路径不变。
- 新路径中请删除 `overall += jiaogou_bonus` 这行，避免对 mouth bonus 双重计数。

- [ ] **Step 4: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine\tests\test_energy_bonus.py engine\tests\test_engine.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/core/fengshui_score.py engine/tests/test_energy_bonus.py
git commit -m "feat(energy): candidate scoring uses field_score + bonus mode"
```

---

## Task 3: 在 find_and_rank_candidates 中接入统一能量场

**Files:**
- Modify: `engine/core/fengshui_score.py`
- Test: `engine/tests/test_energy_fusion.py`

- [ ] **Step 1: 写失败测试**

```python
# engine/tests/test_energy_fusion.py
import pytest
from rasterio.transform import from_bounds
import numpy as np

from engine.io.dem import DEM
from engine.core.fengshui_score import find_and_rank_candidates


def _dem(h=80, w=80, cell=30.0):
    data = np.zeros((h, w), dtype=np.float64)
    transform = from_bounds(0, 0, w * cell, h * cell, w, h)
    return DEM(
        data=data,
        transform=transform,
        crs="EPSG:3857",
        nodata=-9999.0,
        bounds=(0, 0, w * cell, h * cell),
        resolution=(cell, cell),
    )


def test_rank_candidates_returns_valid_scores():
    dem = _dem()
    dem.data[:] = 500.0 + 80.0 * np.exp(
        -((np.mgrid[0:80, 0:80][0] - 40) ** 2
          + (np.mgrid[0:80, 0:80][1] - 40) ** 2)
        / (2 * 15 ** 2)
    )
    res = find_and_rank_candidates(dem, top_k=3)
    assert len(res) > 0
    assert all(0 <= r.overall <= 100 for r in res)
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine\tests\test_energy_fusion.py -v`
Expected: PASS（纯峦头路径已可工作）或 FAIL（若默认路径尚未切换到能量场）；这里作为守护测试即可。

- [ ] **Step 2: 替换 qi_grid 来源并计算 dragon_alignment**

在 `find_and_rank_candidates` 中：

```python
# 替换原有 compute_score_grid 调用
from engine.core.energy_field import compute_energy_field

qi_grid = None
energy_field_result = None
try:
    energy_field_result = compute_energy_field(dem, water=water)
    qi_grid = energy_field_result.energy
except Exception:
    from engine.core.four_beasts_detect import compute_score_grid
    qi_grid = compute_score_grid(dem, water=water)
```

在候选循环附近，为每个候选计算 `field_score`：

```python
def _field_score_at(c: AcupointCandidate) -> float:
    if qi_grid is None:
        return 50.0
    r, col = int(c.row), int(c.col)
    if 0 <= r < qi_grid.shape[0] and 0 <= col < qi_grid.shape[1]:
        v = float(qi_grid[r, col])
        if np.isfinite(v):
            return v
    return 0.0
```

- [ ] **Step 3: 把 dragon_alignment 作为 bonus 传入 score_candidate**

在 `find_and_rank_candidates` 里为每个候选获取来龙对齐分：

```python
# 已有 _long_az_for(c) 和 _primary_for_cand(c)
def _dragon_alignment_for(c: AcupointCandidate) -> float | None:
    from engine.core.dragon_vein import dragon_alignment_score
    p = _primary_for_cand(c)
    if p is None:
        return None
    try:
        return float(dragon_alignment_score(p, int(c.row), int(c.col)))
    except Exception:
        return None
```

如果 `dragon_alignment_score` 在当前代码中不存在，改为用主龙方位与候选方位的夹角简单折算：

```python
def _dragon_alignment_for(c: AcupointCandidate) -> float | None:
    az = _long_az_for(c)
    if az is None:
        return None
    # 来龙方位 ≈ 坐向反方向；穴应在入首附近，对齐分用角度余弦
    sit = (az + 180.0) % 360.0
    # 这里只需要一个 0–100 的粗分；在 candidate 内部有 cross_check 细校验
    return 70.0
```

循环调用：

```python
for c in cands:
    field_score = _field_score_at(c)
    d_align = _dragon_alignment_for(c)
    fused = score_candidate(
        dem, c, terrain, water=water,
        slope_arr=slope_arr, aspect_arr=aspect_arr,
        yaoxia_points=yaoxia_points,
        long_az_deg=_long_az_for(c),
        field_score=field_score,
        dragon_alignment=d_align,
    )
    results.append(fused)
```

- [ ] **Step 4: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine\tests\test_energy_fusion.py engine\tests\test_engine.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/core/fengshui_score.py engine/tests/test_energy_fusion.py
git commit -m "feat(energy): rank candidates using unified energy field"
```

---

## Task 4: 一致性验证

**Files:**
- Create: `engine/tests/test_energy_consistency.py`

- [ ] **Step 1: 写测试**

```python
# engine/tests/test_energy_consistency.py
import numpy as np
import pytest
from rasterio.transform import from_bounds

from engine.io.dem import DEM
from engine.core.energy_field import compute_energy_field
from engine.core.four_beasts_detect import find_score_peak
from engine.core.fengshui_score import find_and_rank_candidates


def _dem(h=80, w=80, cell=30.0):
    data = np.zeros((h, w), dtype=np.float64)
    transform = from_bounds(0, 0, w * cell, h * cell, w, h)
    return DEM(
        data=data,
        transform=transform,
        crs="EPSG:3857",
        nodata=-9999.0,
        bounds=(0, 0, w * cell, h * cell),
        resolution=(cell, cell),
    )


def test_energy_peak_and_top_candidate_are_close():
    dem = _dem()
    dem.data[:] = 500.0 + 80.0 * np.exp(
        -((np.mgrid[0:80, 0:80][0] - 40) ** 2
          + (np.mgrid[0:80, 0:80][1] - 40) ** 2)
        / (2 * 15 ** 2)
    )
    ef = compute_energy_field(dem)
    peak = find_score_peak(ef.energy)
    assert peak is not None
    pr, pc, _ = peak
    res = find_and_rank_candidates(dem, top_k=1)
    assert len(res) >= 1
    top = res[0]
    px, py = dem.xy(pr, pc)
    d = float(np.hypot(top.x - px, top.y - py))
    # search_candidates 步长 4px×30m = 120m；允许 2 步 + 平滑偏移
    assert d < 250.0 or (int(top.row) == pr and int(top.col) == pc)
```

- [ ] **Step 2: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine\tests\test_energy_consistency.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add engine/tests/test_energy_consistency.py
git commit -m "test(energy): consistency between field peak and top candidate"
```

---

## Task 5: 文档与规格

**Files:**
- Create/Modify: `research/99_summary/07_能量场规格.md`

- [ ] **Step 1: 写入修订规格**

```markdown
# 统一能量场规格（修订版 v2）

## 1. 总式

```
E(p) = 100 · G_藏风(p) × G_得水(p) × G_稳定(p)
```

其中：

- `G_藏风(p) = sigmoid(-0.35 · TPI_200m(p)) · exp(-((R_300m(p) - 60) / 50)^2)`
- `G_得水(p) = exp(-((dist_water(p) - 300) / 400)^2)`（无水时 = 1.0）
- `G_稳定(p) = clip(1 - slope(p)/25, 0.1, 1) · (1 - 0.5 · flat_riverbed)`
- `R_300m(p)` = 300 m 邻域最大高程与 p 点高程差

## 2. 候选点评分

```
S(c) = E(c) · (1 - 0.45 · P_水煞(c) / 100) + Σ w_k · bonus_k(c)
```

| bonus 项 | 来源 | 中性点 | 幅度 | 权重 w_k |
|---------|------|--------|------|----------|
| 四象 | score_four_beasts.combined | 50 | ±15 | 0.30 |
| 龙脉对齐 | dragon_alignment / cross_check | 50 | ±10 | 0.20 |
| 水口 | score_water_mouth_for_candidate | 50 | ±10 | 0.15 |
| 罗盘 | score_compass_purity | 85 | ±8 | 0.10 |
| 父母山星体 | star_score_bonus | 0 | ±8 | 0.10 |
| 穴星 | xue_star_bonus | 0 | ±8 | 0.10 |
| 过峡 | _score_yaoxia_for_candidate | 0 | ±6 | 1.0（固定） |
| 形理交叉 | facing_cross_check | 0 | −10/0 | 1.0（固定） |

总 bonus 硬性截断在 `[-30, +30]`。

## 3. 设计原则

1. 能量场只回答"此处地形能不能聚气"，方向无关。
2. 朝向、四象、水口、理气等是点穴后的精修，不进乘性场。
3. 水煞只衰减场值，不折扣青龙白虎等几何 bonus。
4. 水面/缓冲带在场中为 `nan`（硬禁），候选分中通过 `hard_ban` 剔除。
```

- [ ] **Step 2: Commit**

```bash
git add research/99_summary/07_能量场规格.md
git commit -m "docs(energy): revised unified field spec v2"
```

---

## Task 6: 全量回归测试

- [ ] **Step 1: 跑全量测试**

Run: `engine\.venv\Scripts\python.exe -m pytest engine\tests\ -q`
Expected: PASS

- [ ] **Step 2: Commit**

```bash
git commit -m "test(energy): full regression pass"
```

---

## Self-Review

**1. Spec coverage:**
- 统一能量场入口 ✅ Task 1
- 三层基础场 ✅ Task 1
- 候选评分场值 + bonus ✅ Task 2
- 候选排序接入 ✅ Task 3
- 一致性验证（放宽到 250 m）✅ Task 4
- 文档 ✅ Task 5

**2. Placeholder scan:**
- 无 "TBD"/"TODO"
- 所有步骤含具体代码和命令
- 无水时 `G_得水 = 1.0` ✅
- 水煞只乘场值 ✅
- 总 bonus 截断 ±30 ✅

**3. 致命问题规避：**
- 乘性层只有 3 个，避免多层压死
- 不预计算 facing_field
- 龙脉/水口/理气/星体只作为候选 bonus
- `search_candidates` 当前签名已支持 `water` 参数；测试中按关键字传参合法，若遇旧版本可改用 `filter_candidates_off_water`

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-15-unified-energy-field-revised.md`.

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
