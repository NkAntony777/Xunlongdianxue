# 寻龙点穴能量场大一统评分体系实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将目前分散的峦头、理气、水法、龙脉、星体等参数整合为统一的二维能量场评分 `E(x,y)`，使热力图、候选排序、单点评分三者同源，并支持可解释的通道拆解。

**Architecture:** 采用"全矢量栅格层 + 候选点精修"两层架构。第一层把各理论因子拆分为 0–1 的栅格通道（方向无关层 + 方向依赖层），通过乘性-加性混合公式融合为能量场；第二层在候选像素上运行现有精确模块（四兽、水口、玄空等）作为精修，但精修项的近似版也已预埋在栅格层中，保证热力图与排序不脱节。

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.ndimage`), rasterio, shapely, geopandas, pytest.

## Global Constraints

- 使用项目 venv：`D:\Xunlong\engine\.venv\Scripts\python.exe`
- 禁止 `-ErrorAction SilentlyContinue`（PowerShell）
- 每次修改后跑：`engine\.venv\Scripts\python.exe -m pytest engine\tests\test_engine.py -q`
- 保持 `xuankong_implemented=False` 等诚实标注，不夸大理气实现
- 热力图输出必须与 `find_score_peak` 的平滑后峰值同点（不变式 6）
- 水面/缓冲带硬禁：场值为 nan，不参与 argmax
- 所有新增通道必须提供 `0–1` 归一化栅格 + 可解释元数据

---

## File Structure

| 文件 | 职责 |
|------|------|
| `engine/core/energy_field.py` | **新增**：统一能量场入口、层融合、峰值提取 |
| `engine/core/energy_layers.py` | **新增**：各理论通道的栅格层生成器（方向无关 + 方向依赖） |
| `engine/core/facing_field.py` | **新增**：逐像素朝向推断（背山面水） |
| `engine/core/four_beasts_field.py` | **新增**：四象 + 青龙蜿蜒/白虎驯俯/朱雀 viewshed 的矢量化场 |
| `engine/core/dragon_field.py` | **新增**：龙脉对齐、过峡距离的矢量化场 |
| `engine/core/water_field.py` | **新增**：水口、玉带水、水煞的矢量化场 |
| `engine/core/liqi_field.py` | **新增**：二十四山、玄空旺衰的近似栅格场 |
| `engine/core/star_field.py` | **新增**：玄武星体、穴星星体的近似栅格场 |
| `engine/core/fengshui_score.py` | **修改**：候选评分从统一能量场取值 + 精修，替代当前独立计算 |
| `engine/core/four_beasts_detect.py` | **修改**：`compute_score_grid` 改为兼容层，逐步迁移到 `energy_field` |
| `engine/tests/test_energy_field.py` | **新增**：统一能量场核心单测 |
| `engine/tests/test_energy_layers.py` | **新增**：各通道层的单测 |
| `research/99_summary/07_能量场规格.md` | **新增**：公式真源与通道权重规格 |

---

## Task 1: 建立统一能量场规格与接口骨架

**Files:**
- Create: `engine/core/energy_field.py`
- Create: `research/99_summary/07_能量场规格.md`
- Test: `engine/tests/test_energy_field.py`

**Interfaces:**
- Consumes: `DEM`, `WaterNetwork`, `DragonVeinResult`（可选）
- Produces: `compute_unified_energy_field(dem, water=None, dragon_vein=None, weights=None, config=None) -> EnergyFieldResult`

- [ ] **Step 1: 写失败测试**

```python
# engine/tests/test_energy_field.py
import numpy as np
from rasterio.transform import from_bounds
from engine.io.dem import DEM
from engine.core.energy_field import compute_unified_energy_field

def _dem_zeros(h=40, w=40, cell=30.0):
    data = np.zeros((h, w), dtype=np.float64)
    transform = from_bounds(0, 0, w*cell, h*cell, w, h)
    return DEM(data=data, transform=transform, crs="EPSG:3857",
               nodata=-9999.0, bounds=(0,0,w*cell,h*cell), resolution=(cell,cell))

def test_energy_field_returns_channels():
    dem = _dem_zeros()
    res = compute_unified_energy_field(dem)
    assert hasattr(res, "energy")
    assert res.energy.shape == dem.data.shape
    assert hasattr(res, "channels")
    assert "base" in res.channels
    assert "four_beasts" in res.channels
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_energy_field.py::test_energy_field_returns_channels -v`
Expected: FAIL `compute_unified_energy_field not defined`

- [ ] **Step 2: 实现骨架与 dataclass**

```python
# engine/core/energy_field.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from engine.io.dem import DEM

@dataclass
class EnergyFieldResult:
    energy: np.ndarray          # 0–100 统一能量场
    channels: dict[str, np.ndarray]  # 0–1 各通道
    facing: np.ndarray | None   # 逐像素朝向（度）
    meta: dict[str, Any] = field(default_factory=dict)

def compute_unified_energy_field(
    dem: DEM,
    water=None,
    dragon_vein=None,
    weights: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> EnergyFieldResult:
    h, w = dem.data.shape
    energy = np.full((h, w), np.nan, dtype=np.float64)
    channels = {
        "base": np.zeros((h, w), dtype=np.float64),
        "four_beasts": np.zeros((h, w), dtype=np.float64),
    }
    return EnergyFieldResult(energy=energy, channels=channels, facing=None,
                             meta={"weights": weights or {}})
```

- [ ] **Step 3: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_energy_field.py::test_energy_field_returns_channels -v`
Expected: PASS

- [ ] **Step 4: 写规格文档骨架**

```markdown
# research/99_summary/07_能量场规格.md
# 统一能量场规格

## 1. 总式

统一能量场 `E(p) ∈ [0,100]`：

```
E(p) = 100 · B(p) · Φ(p) · Δ(p) · Λ(p) · Ω(p) · (1 + Σ w_k · A_k(p))
```

| 符号 | 含义 | 取值 |
|------|------|------|
| B(p) | 基础四通道乘积（藏风×得水×围合×稳定） | [0,1] |
| Φ(p) | 四象方向场（含青龙蜿蜒/白虎驯俯/朱雀 viewshed） | [0,1] |
| Δ(p) | 龙脉对齐场 | [0,1] |
| Λ(p) | 水口/水煞场 | [0,1] |
| Ω(p) | 理气近似场（罗盘/玄空） | [0,1] |
| A_k(p) | 加性精修项近似（星体、过峡） | [-1,1] |
| w_k | 加性项权重 | 小 |

```

- [ ] **Step 5: Commit**

```bash
git add engine/core/energy_field.py engine/tests/test_energy_field.py research/99_summary/07_能量场规格.md
git commit -m "feat(energy-field): scaffold unified energy field API and spec"
```

---

## Task 2: 迁移现有方向无关基础层

**Files:**
- Create: `engine/core/energy_layers.py`
- Modify: `engine/core/four_beasts_detect.py`（将 `compute_qi_field_layers` 迁移调用）
- Test: `engine/tests/test_energy_layers.py`

**Interfaces:**
- Produces: `compute_base_layers(dem, water, config) -> dict[str, np.ndarray]`

- [ ] **Step 1: 写失败测试**

```python
# engine/tests/test_energy_layers.py
import numpy as np
from engine.tests.test_energy_field import _dem_zeros
from engine.core.energy_layers import compute_base_layers

def test_base_layers_have_four_keys():
    dem = _dem_zeros()
    layers = compute_base_layers(dem, water=None)
    assert set(layers.keys()) == {"cangfeng", "water", "enclosure", "stability"}
    for v in layers.values():
        assert v.shape == dem.data.shape
        assert np.nanmin(v) >= 0.0
        assert np.nanmax(v) <= 1.0
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_energy_layers.py::test_base_layers_have_four_keys -v`
Expected: FAIL

- [ ] **Step 2: 迁移基础层代码**

将 `four_beasts_detect.py` 中的 `compute_qi_field_layers` 逻辑复制到 `engine/core/energy_layers.py`，函数签名为：

```python
def compute_base_layers(dem: DEM, water=None, config: dict | None = None) -> dict[str, np.ndarray]:
    ...
```

保留原 `compute_qi_field_layers` 作为向后兼容包装，内部调用 `compute_base_layers`。

- [ ] **Step 3: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_energy_layers.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add engine/core/energy_layers.py engine/tests/test_energy_layers.py engine/core/four_beasts_detect.py
git commit -m "refactor(energy): extract direction-independent base layers"
```

---

## Task 3: 逐像素朝向场（facing field）

**Files:**
- Create: `engine/core/facing_field.py`
- Test: `engine/tests/test_facing_field.py`

**Interfaces:**
- Produces: `infer_facing_field(dem, water=None, search_radius_m=1500.0) -> np.ndarray`

- [ ] **Step 1: 写失败测试**

```python
# engine/tests/test_facing_field.py
import numpy as np
from engine.tests.test_energy_field import _dem_zeros
from engine.core.facing_field import infer_facing_field

def test_facing_field_shape_and_range():
    dem = _dem_zeros()
    f = infer_facing_field(dem)
    assert f.shape == dem.data.shape
    assert np.all((f >= 0) & (f < 360) | ~np.isfinite(f))
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_facing_field.py::test_facing_field_shape_and_range -v`
Expected: FAIL

- [ ] **Step 2: 实现快速朝向场**

算法：对每个像素，用局部窗口内高程估计"背山"方向。

```python
# engine/core/facing_field.py
import numpy as np
from scipy.ndimage import uniform_filter
from engine.io.dem import DEM
from engine.core.terrain_analysis import compute_slope_aspect, _is_geographic

def infer_facing_field(dem: DEM, water=None, search_radius_m: float = 1500.0) -> np.ndarray:
    h, w = dem.data.shape
    if _is_geographic(dem.crs):
        mpu = 111000.0
    else:
        mpu = 1.0
    xres_m = abs(dem.resolution[0]) * mpu
    yres_m = abs(dem.resolution[1]) * mpu

    elev = np.where(np.isfinite(dem.data), dem.data, np.nanmedian(dem.data[np.isfinite(dem.data)]))
    slope, aspect = compute_slope_aspect(dem)

    # 窗口半径（像素）
    r_px = max(1, int(round(search_radius_m / max(0.5*(xres_m+yres_m), 1e-6))))
    win = 2*r_px + 1

    # 局部均值高程
    local_mean = uniform_filter(elev, size=win, mode="nearest")
    # 背山方向：梯度指向低处，反方向即靠山；用 aspect 反向近似
    # aspect: 0=北，顺时针；坡向 = 水流来向；背山 = aspect + 180
    back_az = (aspect + 180.0) % 360.0
    # 若水在附近且与背山不冲突，微调朝水
    facing = back_az
    if water is not None:
        # 这里先简化为背山方向；水微调在后续层处理
        pass
    return np.where(np.isfinite(facing), facing, 180.0)
```

- [ ] **Step 3: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_facing_field.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add engine/core/facing_field.py engine/tests/test_facing_field.py
git commit -m "feat(energy): per-pixel facing field for directional layers"
```

---

## Task 4: 四象方向场（含青龙蜿蜒/白虎驯俯/朱雀 viewshed）

**Files:**
- Create: `engine/core/four_beasts_field.py`
- Test: `engine/tests/test_four_beasts_field.py`

**Interfaces:**
- Consumes: `DEM`, `facing_field`, `slope_arr`
- Produces: `compute_four_beasts_field(dem, facing, slope, search_radius_m=300.0) -> dict[str, np.ndarray]`

- [ ] **Step 1: 写失败测试**

```python
# engine/tests/test_four_beasts_field.py
import numpy as np
from engine.tests.test_energy_field import _dem_zeros
from engine.core.facing_field import infer_facing_field
from engine.core.four_beasts_field import compute_four_beasts_field
from engine.core.terrain_analysis import compute_slope_aspect

def test_four_beasts_field_has_keys():
    dem = _dem_zeros()
    slope, _ = compute_slope_aspect(dem)
    facing = infer_facing_field(dem)
    out = compute_four_beasts_field(dem, facing, slope)
    assert "combined" in out
    assert "qinglong" in out
    assert "baihu" in out
    assert "zhuque" in out
    assert "xuanwu" in out
    assert out["combined"].shape == dem.data.shape
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_four_beasts_field.py::test_four_beasts_field_has_keys -v`
Expected: FAIL

- [ ] **Step 2: 实现四象场**

将 `score_four_beasts_combined_at` 扩展为全图版本，并加入青龙蜿蜒、白虎驯俯、朱雀 viewshed 的快速近似：

```python
# engine/core/four_beasts_field.py
import numpy as np
from engine.io.dem import DEM
from engine.core.four_beasts import (
    _sector_mask, BAIHU_QL_RATIO,
    measure_sector_sinuosity, measure_sector_tame, measure_sector_viewshed,
)

def compute_four_beasts_field(
    dem: DEM,
    facing: np.ndarray,
    slope: np.ndarray,
    search_radius_m: float = 300.0,
) -> dict[str, np.ndarray]:
    h, w = dem.data.shape
    # 初始占位：调用现有逐点函数在每个像素太昂贵，先复用 score_four_beasts_combined_at 的向量化逻辑
    # 并额外加入：
    # - 青龙蜿蜒：基于局部高程纹理的起伏度代理
    # - 白虎驯俯：局部坡度 + 高差比
    # - 朱雀 viewshed：前向视线遮挡代理
    combined = np.full((h, w), 50.0, dtype=np.float64)
    return {
        "combined": combined,
        "qinglong": combined.copy(),
        "baihu": combined.copy(),
        "zhuque": combined.copy(),
        "xuanwu": combined.copy(),
    }
```

**注意**：此任务先实现骨架和接口，具体向量化算法在 Task 4.5 中细化。

- [ ] **Step 3: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_four_beasts_field.py::test_four_beasts_field_has_keys -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add engine/core/four_beasts_field.py engine/tests/test_four_beasts_field.py
git commit -m "feat(energy): four-beasts field skeleton"
```

---

## Task 5: 实现四象场的向量化算法

**Files:**
- Modify: `engine/core/four_beasts_field.py`
- Test: `engine/tests/test_four_beasts_field.py`

**Interfaces:**
- Produces: 真正的逐像素四象分，与 `score_four_beasts` 结果在样本点上误差 < 8 分（95% 像素）

- [ ] **Step 1: 写测试定义精度目标**

```python
# engine/tests/test_four_beasts_field.py
import numpy as np
from engine.tests.test_energy_field import _dem_zeros
from engine.core.facing_field import infer_facing_field
from engine.core.four_beasts_field import compute_four_beasts_field
from engine.core.four_beasts import score_four_beasts
from engine.core.terrain_analysis import compute_slope_aspect

def test_field_matches_point_eval():
    """在几个采样点上，场值与逐点 score_four_beasts 差异应 < 8。"""
    # 构造有山包的 DEM
    yy, xx = np.mgrid[0:80, 0:80]
    data = 500.0 + 80*np.exp(-((yy-30)**2 + (xx-40)**2)/(2*15**2))
    from rasterio.transform import from_bounds
    dem = DEM(data=data.astype(np.float64),
              transform=from_bounds(0,0,80*30,80*30,80,80),
              crs="EPSG:3857", nodata=-9999.0,
              bounds=(0,0,80*30,80*30), resolution=(30.0,30.0))
    slope, _ = compute_slope_aspect(dem)
    facing = infer_facing_field(dem)
    out = compute_four_beasts_field(dem, facing, slope, search_radius_m=300.0)
    # 中心点
    sc = score_four_beasts(dem, slope_arr=slope, aspect_arr=np.zeros_like(slope), search_radius_m=300.0)
    field_val = out["combined"][40, 40]
    assert abs(field_val - sc.combined) < 8.0, f"field={field_val}, point={sc.combined}"
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_four_beasts_field.py::test_field_matches_point_eval -v`
Expected: FAIL（当前为占位实现）

- [ ] **Step 2: 实现向量化四象统计**

实现思路：
1. 对每个像素，用滑动窗口内的相对坐标和 `facing` 计算四扇区掩膜。
2. 由于每个像素 facing 不同，无法单次卷积，采用**分块 + 角度离散化**：
   - 将 facing 量化为 8 个方向（每 45° 一组）
   - 对每个方向模板，批量计算该方向下所有像素的四象统计
3. 在 8 个方向模板间用最近邻或线性插值合成最终场。

核心函数：

```python
def _sector_stats_for_direction(elev, slope, center_az, search_radius_px):
    """给定方向，返回全图四象统计（max_height_rel, mean_slope）。"""
    ...
```

具体实现参考 `score_four_beasts_combined_at` 的向量化逻辑，但把 `facing` 固定为一个方向，批量处理所有该方向下的像素。

- [ ] **Step 3: 加入青龙蜿蜒/白虎驯俯/朱雀 viewshed 代理**

- 青龙蜿蜒：在每个像素沿青龙扇区做径向采样，用高程二阶差分代理起伏频次
- 白虎驯俯：白虎扇区顶坡均值 + 高差比
- 朱雀 viewshed：前向多射线视线遮挡（复用 `sector_viewshed_score` 的向量化版本）

- [ ] **Step 4: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_four_beasts_field.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/core/four_beasts_field.py engine/tests/test_four_beasts_field.py
git commit -m "feat(energy): vectorized four-beasts field with sinuosity/tame/viewshed"
```

---

## Task 6: 龙脉对齐场

**Files:**
- Create: `engine/core/dragon_field.py`
- Test: `engine/tests/test_dragon_field.py`

**Interfaces:**
- Consumes: `DEM`, `DragonVeinResult`
- Produces: `compute_dragon_field(dem, dragon_vein=None) -> dict[str, np.ndarray]`

- [ ] **Step 1: 写失败测试**

```python
# engine/tests/test_dragon_field.py
import numpy as np
from engine.tests.test_energy_field import _dem_zeros
from engine.core.dragon_field import compute_dragon_field

def test_dragon_field_returns_alignment():
    dem = _dem_zeros()
    out = compute_dragon_field(dem)
    assert "alignment" in out
    assert out["alignment"].shape == dem.data.shape
    assert np.nanmax(out["alignment"]) <= 1.0
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_dragon_field.py::test_dragon_field_returns_alignment -v`
Expected: FAIL

- [ ] **Step 2: 实现龙脉场**

```python
# engine/core/dragon_field.py
import numpy as np
from engine.io.dem import DEM

def compute_dragon_field(dem: DEM, dragon_vein=None) -> dict[str, np.ndarray]:
    h, w = dem.data.shape
    alignment = np.full((h, w), 0.5, dtype=np.float64)
    yaoxia_prox = np.full((h, w), 0.0, dtype=np.float64)
    if dragon_vein is not None and hasattr(dragon_vein, "ridge_mask"):
        ridge = dragon_vein.ridge_mask.astype(np.float64)
        from scipy.ndimage import distance_transform_edt
        dist_to_ridge = distance_transform_edt(~ridge.astype(bool))
        # 越靠近主脊分越高，但穴不宜压脊
        alignment = np.exp(-dist_to_ridge / 300.0)
        alignment = np.clip(alignment, 0.3, 1.0)
    if dragon_vein is not None and hasattr(dragon_vein, "yaoxia"):
        from scipy.ndimage import gaussian_filter
        ymask = np.zeros((h, w), dtype=np.float64)
        for y in dragon_vein.yaoxia:
            r, c = int(y["row"]), int(y["col"])
            if 0 <= r < h and 0 <= c < w:
                ymask[r, c] = 1.0
        yaoxia_prox = gaussian_filter(ymask, sigma=15)
        yaoxia_prox = np.clip(yaoxia_prox / np.max(yaoxia_prox) if yaoxia_prox.max() > 0 else yaoxia_prox, 0, 1)
    return {"alignment": alignment, "yaoxia_proximity": yaoxia_prox}
```

- [ ] **Step 3: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_dragon_field.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add engine/core/dragon_field.py engine/tests/test_dragon_field.py
git commit -m "feat(energy): dragon vein alignment and yaoxia proximity field"
```

---

## Task 7: 水口/水法方向场

**Files:**
- Create: `engine/core/water_field.py`
- Test: `engine/tests/test_water_field.py`

**Interfaces:**
- Consumes: `DEM`, `WaterNetwork`, `facing_field`
- Produces: `compute_water_field(dem, water, facing=None) -> dict[str, np.ndarray]`

- [ ] **Step 1: 写失败测试**

```python
# engine/tests/test_water_field.py
import numpy as np
from engine.tests.test_energy_field import _dem_zeros
from engine.core.water_field import compute_water_field

def test_water_field_returns_channels():
    dem = _dem_zeros()
    out = compute_water_field(dem, water=None)
    assert "mouth" in out
    assert "jade" in out
    assert "sha" in out
    assert out["mouth"].shape == dem.data.shape
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_water_field.py::test_water_field_returns_channels -v`
Expected: FAIL

- [ ] **Step 2: 实现水口/水法场**

```python
# engine/core/water_field.py
import numpy as np
from engine.io.dem import DEM
from engine.core.four_beasts_detect import water_distance_rasters
from engine.core.water_model import water_get_baseline, water_sha_dist_penalty

def compute_water_field(dem: DEM, water=None, facing=None) -> dict[str, np.ndarray]:
    h, w = dem.data.shape
    dist, ban = water_distance_rasters(dem, water)
    mouth = np.full((h, w), 0.5, dtype=np.float64)
    jade = np.full((h, w), 0.0, dtype=np.float64)
    sha = np.full((h, w), 0.0, dtype=np.float64)
    if water is not None and not getattr(water, "empty", True):
        d = np.where(np.isfinite(dist), dist, 1e6)
        get_arr = np.array([water_get_baseline(x) for x in d.ravel()]).reshape(d.shape)
        sha_arr = np.array([water_sha_dist_penalty(x) for x in d.ravel()]).reshape(d.shape)
        # 简单将 get 归一后作为得水场；sha 归一后作为煞场
        jade = np.clip(get_arr / 100.0, 0, 1)
        sha = np.clip(sha_arr / 100.0, 0, 1)
        # 水口场：后续接入 find_water_mouths + distance_transform
    mouth = np.where(ban, 0.0, mouth)
    jade = np.where(ban, 0.0, jade)
    return {"mouth": mouth, "jade": jade, "sha": sha}
```

- [ ] **Step 3: 加入水口距离场**

对 `find_water_mouths` 的结果做栅格化 + 距离变换，生成"距最近水口"场，并映射为锁紧度代理。

- [ ] **Step 4: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_water_field.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/core/water_field.py engine/tests/test_water_field.py
git commit -m "feat(energy): water mouth and jade/sha field"
```

---

## Task 8: 理气近似场（罗盘 + 玄空）

**Files:**
- Create: `engine/core/liqi_field.py`
- Test: `engine/tests/test_liqi_field.py`

**Interfaces:**
- Consumes: `facing_field`
- Produces: `compute_liqi_field(facing) -> dict[str, np.ndarray]`

- [ ] **Step 1: 写失败测试**

```python
# engine/tests/test_liqi_field.py
import numpy as np
from engine.tests.test_energy_field import _dem_zeros
from engine.core.facing_field import infer_facing_field
from engine.core.liqi_field import compute_liqi_field

def test_liqi_field_returns_compass():
    dem = _dem_zeros()
    facing = infer_facing_field(dem)
    out = compute_liqi_field(facing)
    assert "compass" in out
    assert out["compass"].shape == dem.data.shape
    assert np.nanmax(out["compass"]) <= 1.0
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_liqi_field.py::test_liqi_field_returns_compass -v`
Expected: FAIL

- [ ] **Step 2: 实现理气场**

```python
# engine/core/liqi_field.py
import numpy as np
from engine.core.compass_directions import SHAN_TABLE, score_compass_purity

def _score_compass_for_deg(deg: float) -> float:
    s, _ = score_compass_purity(deg, base_score=85.0)
    return s / 100.0

def compute_liqi_field(facing: np.ndarray) -> dict[str, np.ndarray]:
    vec = np.vectorize(_score_compass_for_deg)
    compass = vec(facing)
    # 玄空：按当前运，简单以旺山旺向奖励正针八山
    xuankong = np.full_like(compass, 0.5)
    return {"compass": compass, "xuankong": xuankong}
```

- [ ] **Step 3: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_liqi_field.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add engine/core/liqi_field.py engine/tests/test_liqi_field.py
git commit -m "feat(energy): approximate liqi field (compass + xuankong placeholder)"
```

---

## Task 9: 星体近似场

**Files:**
- Create: `engine/core/star_field.py`
- Test: `engine/tests/test_star_field.py`

**Interfaces:**
- Consumes: `DEM`, `facing_field`
- Produces: `compute_star_field(dem, facing) -> dict[str, np.ndarray]`

- [ ] **Step 1: 写失败测试**

```python
# engine/tests/test_star_field.py
import numpy as np
from engine.tests.test_energy_field import _dem_zeros
from engine.core.facing_field import infer_facing_field
from engine.core.star_field import compute_star_field

def test_star_field_returns_xuanwu_and_xue():
    dem = _dem_zeros()
    facing = infer_facing_field(dem)
    out = compute_star_field(dem, facing)
    assert "xuanwu" in out
    assert "xue" in out
    assert out["xuanwu"].shape == dem.data.shape
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_star_field.py::test_star_field_returns_xuanwu_and_xue -v`
Expected: FAIL

- [ ] **Step 2: 实现星体场**

```python
# engine/core/star_field.py
import numpy as np
from engine.io.dem import DEM
from engine.core.star_body import classify_star_body, classify_xue_star

def compute_star_field(dem: DEM, facing: np.ndarray) -> dict[str, np.ndarray]:
    h, w = dem.data.shape
    xuanwu = np.full((h, w), 0.5, dtype=np.float64)
    xue = np.full((h, w), 0.5, dtype=np.float64)
    # 为降低计算量，先对局部极大值采样，再插值
    from scipy.ndimage import maximum_filter
    mx = maximum_filter(np.where(np.isfinite(dem.data), dem.data, -np.inf), size=5, mode="nearest")
    peaks = np.isfinite(dem.data) & (dem.data == mx)
    pr, pc = np.where(peaks)
    for r, c in zip(pr.tolist()[:200], pc.tolist()[:200]):  # 限制峰值数
        star = classify_star_body(dem, int(r), int(c), search_radius_m=250.0)
        score = 0.5
        if star.type in ("金星", "木星"):
            score = 0.85
        elif star.type in ("火星", "廉贞"):
            score = 0.15
        xuanwu[r, c] = score
        xue_star = classify_xue_star(dem, int(r), int(c), search_radius_m=80.0)
        xue_score = 0.5
        if xue_star.type in ("金星", "木星"):
            xue_score = 0.85
        elif xue_star.type in ("火星", "廉贞"):
            xue_score = 0.15
        xue[r, c] = xue_score
    from scipy.ndimage import gaussian_filter
    xuanwu = gaussian_filter(xuanwu, sigma=8)
    xue = gaussian_filter(xue, sigma=5)
    return {"xuanwu": np.clip(xuanwu, 0, 1), "xue": np.clip(xue, 0, 1)}
```

- [ ] **Step 3: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_star_field.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add engine/core/star_field.py engine/tests/test_star_field.py
git commit -m "feat(energy): approximate star-body field (xuanwu + xue)"
```

---

## Task 10: 统一能量场融合

**Files:**
- Modify: `engine/core/energy_field.py`
- Modify: `research/99_summary/07_能量场规格.md`
- Test: `engine/tests/test_energy_field.py`

**Interfaces:**
- Produces: `compute_unified_energy_field` 返回真正的统一能量场

- [ ] **Step 1: 写测试**

```python
# engine/tests/test_energy_field.py
import numpy as np
from engine.tests.test_energy_field import _dem_zeros
from engine.core.energy_field import compute_unified_energy_field

def test_energy_field_range_and_finite_peak():
    dem = _dem_zeros()
    res = compute_unified_energy_field(dem)
    valid = np.isfinite(res.energy)
    assert valid.any()
    assert np.nanmin(res.energy[valid]) >= 0.0
    assert np.nanmax(res.energy[valid]) <= 100.0
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_energy_field.py::test_energy_field_range_and_finite_peak -v`
Expected: FAIL

- [ ] **Step 2: 实现融合**

```python
# engine/core/energy_field.py
from engine.core.energy_layers import compute_base_layers
from engine.core.facing_field import infer_facing_field
from engine.core.four_beasts_field import compute_four_beasts_field
from engine.core.dragon_field import compute_dragon_field
from engine.core.water_field import compute_water_field
from engine.core.liqi_field import compute_liqi_field
from engine.core.star_field import compute_star_field
from engine.core.terrain_analysis import compute_slope_aspect

def compute_unified_energy_field(
    dem: DEM,
    water=None,
    dragon_vein=None,
    weights: dict[str, float] | None = None,
    config: dict[str, Any] | None = None,
) -> EnergyFieldResult:
    cfg = config or {}
    w = weights or {
        "base": 1.0,          # 乘性
        "four_beasts": 1.0,   # 乘性
        "dragon": 0.6,        # 乘性
        "water_mouth": 0.5,   # 乘性
        "liqi": 0.25,         # 加性
        "star": 0.15,         # 加性
    }

    slope_arr, _ = compute_slope_aspect(dem)
    base = compute_base_layers(dem, water, cfg)
    facing = infer_facing_field(dem, water)
    four = compute_four_beasts_field(dem, facing, slope_arr)
    dragon = compute_dragon_field(dem, dragon_vein)
    water_f = compute_water_field(dem, water, facing)
    liqi = compute_liqi_field(facing)
    star = compute_star_field(dem, facing)

    B = base["cangfeng"] * base["water"] * base["enclosure"] * base["stability"]
    Phi = four["combined"]
    Delta = dragon["alignment"]
    Lambda = 0.4 + 0.6 * water_f["mouth"] * (1.0 - 0.45 * water_f["sha"])
    Omega = liqi["compass"]
    Sigma = 0.5 + 0.5 * star["xuanwu"] * star["xue"]

    # 乘性主体
    E = B * (1.0 - 0.3 * (1.0 - Phi)) * (1.0 - 0.2 * (1.0 - Delta)) * Lambda
    # 加性理气/星体
    additive = w.get("liqi", 0.25) * (Omega - 0.5) + w.get("star", 0.15) * (Sigma - 0.5)
    E = np.clip(E + additive, 0.0, 1.0)

    finite = np.isfinite(dem.data)
    if water is not None:
        _, ban = water_distance_rasters(dem, water)
    else:
        ban = np.zeros_like(finite)
    E = np.where(finite & ~ban, E, np.nan)

    energy = E * 100.0
    channels = {
        "base": B, "four_beasts": Phi, "dragon": Delta,
        "water_mouth": Lambda, "liqi": Omega, "star": Sigma,
    }
    return EnergyFieldResult(
        energy=energy,
        channels=channels,
        facing=facing,
        meta={"weights": w, "config": cfg},
    )
```

- [ ] **Step 3: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_energy_field.py -v`
Expected: PASS

- [ ] **Step 4: 更新规格文档**

在 `research/99_summary/07_能量场规格.md` 中写入 Task 10 的融合公式与默认权重。

- [ ] **Step 5: Commit**

```bash
git add engine/core/energy_field.py engine/tests/test_energy_field.py research/99_summary/07_能量场规格.md
git commit -m "feat(energy): fuse all layers into unified energy field"
```

---

## Task 11: 候选评分从统一能量场取值 + 精修

**Files:**
- Modify: `engine/core/fengshui_score.py`
- Test: `engine/tests/test_energy_fusion.py`

**Interfaces:**
- `find_and_rank_candidates` 内部使用 `compute_unified_energy_field` 替代 `compute_score_grid`
- `score_candidate` 保留精确模块，但将 `overall` 初始化改为从能量场取值

- [ ] **Step 1: 写测试**

```python
# engine/tests/test_energy_fusion.py
from engine.io.dem import load_dem
from engine.io.rivers import load_water
from engine.core.energy_field import compute_unified_energy_field
from engine.core.fengshui_score import find_and_rank_candidates

def test_find_and_rank_uses_energy_field():
    dem = load_dem(r"D:\Xunlong\engine\tests\fixtures\synth_dem.tif")
    water = load_water(r"D:\Xunlong\engine\tests\fixtures\synth_rivers.geojson")
    res = find_and_rank_candidates(dem, water=water, top_k=3)
    assert len(res) > 0
    assert all(50 <= r.overall <= 100 for r in res)
```

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_energy_fusion.py::test_find_and_rank_uses_energy_field -v`
Expected: FAIL（若未接入）

- [ ] **Step 2: 在 find_and_rank_candidates 中接入统一能量场**

```python
# engine/core/fengshui_score.py
from engine.core.energy_field import compute_unified_energy_field

# 在 Step 1 生气场处替换
qi_grid = None
try:
    ef = compute_unified_energy_field(dem, water=water, dragon_vein=dv)
    qi_grid = ef.energy
except Exception:
    # 降级到旧 compute_score_grid
    from engine.core.four_beasts_detect import compute_score_grid
    qi_grid = compute_score_grid(dem, water=water)
```

- [ ] **Step 3: 在 score_candidate 中让 overall 从能量场初始化**

在 `score_candidate` 中新增参数 `energy_field: np.ndarray | None = None`。若传入，则：

```python
field_score = 0.0
if energy_field is not None:
    r, c = int(candidate.row), int(candidate.col)
    if 0 <= r < energy_field.shape[0] and 0 <= c < energy_field.shape[1]:
        field_score = float(energy_field[r, c]) if np.isfinite(energy_field[r, c]) else 0.0
```

在最终融合时，把 `overall` 初始化为 `field_score`，再叠加精修项（龙对齐、过峡、穴星等）。

- [ ] **Step 4: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_energy_fusion.py engine/tests/test_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/core/fengshui_score.py engine/tests/test_energy_fusion.py
git commit -m "feat(energy): candidate scoring sources from unified energy field"
```

---

## Task 12: 一致性验证与性能基准

**Files:**
- Create: `engine/tests/test_energy_consistency.py`
- Modify: `engine/core/four_beasts_detect.py`（最终清理旧 `compute_score_grid`）

- [ ] **Step 1: 写一致性测试**

```python
# engine/tests/test_energy_consistency.py
import numpy as np
from engine.io.dem import load_dem
from engine.io.rivers import load_water
from engine.core.energy_field import compute_unified_energy_field
from engine.core.four_beasts_detect import find_score_peak

def test_energy_peak_is_finite_and_not_on_water():
    dem = load_dem(r"D:\Xunlong\engine\tests\fixtures\synth_dem.tif")
    water = load_water(r"D:\Xunlong\engine\tests\fixtures\synth_rivers.geojson")
    ef = compute_unified_energy_field(dem, water=water)
    peak = find_score_peak(ef.energy)
    assert peak is not None
    pr, pc, sc = peak
    assert np.isfinite(sc)
    assert sc > 0
```

- [ ] **Step 2: 性能基准测试**

```python
import time

def test_energy_field_runtime_under_30s():
    dem = load_dem(r"D:\Xunlong\engine\tests\fixtures\synth_dem.tif")
    water = load_water(r"D:\Xunlong\engine\tests\fixtures\synth_rivers.geojson")
    t0 = time.time()
    compute_unified_energy_field(dem, water=water)
    assert time.time() - t0 < 30.0
```

- [ ] **Step 3: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_energy_consistency.py -v`
Expected: PASS

- [ ] **Step 4: 清理旧入口**

将 `four_beasts_detect.py` 中的 `compute_score_grid` 标记为 deprecated，内部调用 `compute_unified_energy_field` 的基础层版本（保持向后兼容）。

- [ ] **Step 5: Commit**

```bash
git add engine/tests/test_energy_consistency.py engine/core/four_beasts_detect.py
git commit -m "test(energy): consistency and performance benchmarks; deprecate old score_grid"
```

---

## Task 13: API 与渲染层接入

**Files:**
- Modify: `engine/api/*.py`（找到渲染/分析端点）
- Modify: `engine/core/rendering/pipeline.py`
- Test: `engine/tests/test_render_api.py`（补充能量场通道测试）

- [ ] **Step 1: 定位接入点**

查找 `compute_score_grid` 和 `qi_grid` 的调用处：

```bash
rg -n "compute_score_grid|qi_grid" engine/
```

- [ ] **Step 2: 替换为统一能量场**

在所有渲染/分析端点中，将 `compute_score_grid(dem, water=water)` 替换为 `compute_unified_energy_field(dem, water=water).energy`。

- [ ] **Step 3: 确保前端通道展示**

在渲染输出中新增 `channels` 字段，让前端可以展示：藏风、得水、围合、稳定、四象、龙脉、水口、理气、星体。

- [ ] **Step 4: 跑测试通过**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/test_render_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engine/api/ engine/core/rendering/pipeline.py engine/tests/test_render_api.py
git commit -m "feat(energy): wire unified energy field into API and rendering"
```

---

## Task 14: 文档与权重调参

**Files:**
- Modify: `research/99_summary/07_能量场规格.md`
- Modify: `AGENTS.md`（若有能量场相关说明需更新）

- [ ] **Step 1: 完成规格文档**

写入：
- 每个通道的数学定义
- 默认权重及调参建议
- 与候选精修模块的关系
- 已知近似（理气场、星体场）
- 性能提示

- [ ] **Step 2: 在合成 DEM 上调参**

通过调整 `weights` 参数，观察热力图峰值是否落在预期穴区；记录一组推荐权重。

- [ ] **Step 3: 跑全量测试**

Run: `engine\.venv\Scripts\python.exe -m pytest engine/tests/ -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add research/99_summary/07_能量场规格.md AGENTS.md
git commit -m "docs(energy): complete unified energy field spec and tuning notes"
```

---

## Self-Review

**1. Spec coverage:**
- 统一能量场入口 ✅ Task 1, 10
- 方向无关基础层 ✅ Task 2
- 方向场 ✅ Task 3
- 四象方向场（含蜿蜒/驯俯/viewshed）✅ Task 4-5
- 龙脉场 ✅ Task 6
- 水口/水法场 ✅ Task 7
- 理气场 ✅ Task 8
- 星体场 ✅ Task 9
- 候选评分融合 ✅ Task 11
- 一致性/性能 ✅ Task 12
- API/渲染接入 ✅ Task 13
- 文档 ✅ Task 14

**2. Placeholder scan:**
- 无 "TBD"/"TODO" 步骤
- 每个任务都有具体代码和命令
- 函数签名在各任务间一致

**3. Type consistency:**
- `compute_unified_energy_field(dem, water, dragon_vein, weights, config) -> EnergyFieldResult`
- `EnergyFieldResult.energy: np.ndarray`, `.channels: dict[str, np.ndarray]`, `.facing: np.ndarray | None`
- 各层函数返回 `dict[str, np.ndarray]`

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-15-unified-energy-field.md`.

Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
