# Xunlong Engine (寻龙点穴地形分析引擎)

> 基于 DEM + 水系/道路等矢量数据，把传统风水（形势派）"龙砂穴水向"五诀翻译为可计算的 GIS 算法，自动输出候选"龙穴"+ 综合评分。

**本项目纯属传统文化研究 + GIS 地形分析技术实践，不构成任何投资、医疗、法律、婚姻、商业选址决策建议。**

---

## 1. 状态

✅ **核心算法完成**：
- DEM 加载 / 裁剪 / 重投影 / 填洼
- 坡度 / 坡向 / 高差 / 粗糙度 / TPI
- 四兽（青龙白虎朱雀玄武）扇区评分
- 砂山识别（玄武 + 龙虎 + 案山 + 朝山）
- 水系距离 / 方向 / 玉带水检测
- 候选穴搜索（TPI / TWI / 形态）
- 综合评分 + Top-K 排名
- 山脊线 / 龙脉提取（基础版）

✅ **传统审计第一轮（P0/P1 全做，2026-07）**：
- 五行星体识别 `core/star_body.py`
- 砂形朝抱 / 反背 `core/mountain_curve.py`
- 案山圆净 / 破碎度 / 高度型 `core/anshan_quality.py`
- 水口 / 龙水交媾 `core/water_mouth.py`
- 弓背水三节连贯 + 距离自适应 `core/water_curve.py`
- 二十四山向 + 兼向 / 出卦判别 `core/compass_directions.py`
- 玄空九运 + 简化挨星 `core/yuan_yun_xuankong.py`
- 微地形物候（NDVI / 土壤湿度接口）`core/phenology.py`
- 入首过峡「蜂腰鹤膝」识别 `dragon_vein.find_yaoxia`
- 晕土（晕界）粗略识别 `core/halo_soil.py`

✅ **集成到 API**：`fengshui_score.py` 已吸纳 compass / mouth / xuankong / phenology，候选响应新增对应字段。

**测试覆盖：145 passed, 4 skipped**（含 G.3、B8 下卦黄金局、朝抱/过峡集成）。

---

## 模块集成状态（v0.3+ 审查后）

| 模块 | 文件 | 进入 overall 加权？ | 备注 |
|---|---|---|---|
| 坡度/坡向 | `terrain_analysis.py` | 间接 | 各模块基础 |
| 四兽扇区评分 | `four_beasts.py` | ✅ | weighted |
| 砂山统计 | `sand_water.py` | ✅ | weighted |
| 形态判读 | `acupoint.py` | ✅ | weighted |
| 明堂开阔 | `acupoint.py` | ✅ | weighted |
| 稳定性 | `acupoint.py` | ✅ | weighted |
| 得水/水煞 | `water_model.py` + `water_mouth.py` | ✅ + 乘性 | 综合通道 |
| 水口/龙水交媾 | `water_mouth.py` | +12 固定加成 | jiaogou_bonus |
| 二十四山向 | `compass_directions.py` | 显示 | `scores.compass` |
| **晕土识别** | `halo_soil.py` | ✅ 加成 | `(halo-50) × 0.16` |
| **五行星体** | `star_body.py` | ✅ 加成 | `+5/0/-8` |
| **砂形朝抱** | `mountain_curve.measure_embrace` | ✅ 并入 sand | 左右砂 60% 峰数 + 40% 朝抱 |
| **入首过峡** | `dragon_vein.find_yaoxia()` | ✅ 固定加减 | `-2…+6`；`DragonVeinResult.yaoxia`；排名整幅一次 |
| **三节弓背水** | `water_curve.py` | ✅ 进 form→Γ/P | `enrich_form_with_water_curve`；双通道不混维 |
| 玄空九运 | `yuan_yun_xuankong.py` | 仅展示 | 简化盘对外；`fly_chart_strict` **下卦已修对**（运永顺+同元龙）；替卦未做，**仍禁止 implemented=true** |
| 微地形物候 | `phenology.py` | 仅展示 | DEM 代理；`is_proxy=True` |

**已修复**（子代理审计关键问题）：
- ✅ B1 修正癸 typo `\u767e → \u7678`
- ✅ B3 修正乾 yin/yang（阴→阳）
- ✅ B4 修正 LUO_SHUN_ORDER 顺序
- ✅ B5/B6 玄空输出明确为简化盘，`implemented=False`
- ✅ A4 `find_yaoxia` 重写为横向脊宽估计
- ✅ D1 mouth/compass/xuankong 加权项展示分 vs 入和分 明确标注
- ✅ E7 真正的 `sand_dist_fn` 实施
- ✅ G.3 / 生气场 `compute_score_grid`：全矢量乘性场（藏风×得水×围合×稳定），穴=峰值后 detect 四象
- ✅ B8 下卦 `fly_chart_strict` 修对（运盘永顺、元旦同元龙、SHAN_TABLE 三元龙；替卦未做，implemented 仍 false）

---

## 2. 项目结构

```
engine/
├── __init__.py
├── cli.py                  # 命令行入口
├── requirements.txt
├── core/                   # 核心算法
│   ├── __init__.py
│   ├── terrain_analysis.py # 坡度/坡向/TPI/粗糙度
│   ├── four_beasts.py      # 龙虎朱雀玄武评分
│   ├── acupoint.py         # 候选穴搜索/TPI/形态
│   ├── sand_water.py       # 砂山+水系
│   ├── dragon_vein.py      # 山脊线/龙脉
│   └── fengshui_score.py   # 综合评分
├── io/                     # 数据 I/O
│   ├── dem.py              # DEM 加载/裁剪/重投影
│   └── rivers.py           # 水系加载
├── utils/
│   └── helpers.py          # 通用工具
├── tests/
│   ├── test_engine.py      # 单元测试 (17)
│   └── fixtures/           # 测试数据
└── examples/
    └── demo.py             # 端到端 demo
```

---

## 3. 安装

```bash
# 创建虚拟环境（推荐用 uv）
uv venv engine\.venv --python 3.12
uv pip install --python engine\.venv\Scripts\python.exe -r engine\requirements.txt
```

依赖：
- numpy / scipy / scikit-image
- rasterio / geopandas / shapely
- matplotlib（可选）
- pysheds（可选加速，默认走 numpy 回退）

---

## 4. 快速开始

### 4.1 跑 Demo（合成数据）

```bash
engine\.venv\Scripts\python.exe -m engine.examples.demo
```

输出：
- `engine/examples/output/report.json` - 候选穴报告
- `engine/examples/output/candidates.geojson` - GeoJSON

### 4.2 CLI

```bash
engine\.venv\Scripts\python.exe -m engine.cli \
    --dem path/to/dem.tif \
    --water path/to/rivers.geojson \
    --top-k 10 \
    --min-score 50 \
    --out-json report.json \
    --out-geojson candidates.geojson \
    --dragon-vein
```

### 4.3 Python API

```python
from engine.io.dem import load_dem
from engine.io.rivers import load_water
from engine.core.fengshui_score import find_and_rank_candidates

dem = load_dem("dem.tif")
water = load_water("rivers.geojson")  # 可选
results = find_and_rank_candidates(dem, water, top_k=10, min_score=60)

for r in results:
    print(f"{r.candidate_id}: ({r.x:.0f}, {r.y:.0f}) score={r.overall} form={r.form_type}")
```

### 4.4 单元测试

```bash
cd engine
..\engine\.venv\Scripts\python.exe -m pytest tests\test_engine.py -v
```

---

## 5. 输出格式

### 5.1 JSON 报告

```json
{
  "metadata": {
    "dem": "dem.tif",
    "bbox": [0, 0, 6000, 6000],
    "resolution_m": 30.0,
    "terrain": { "mean_elevation": 500, "relief": 250, ... }
  },
  "candidates": [
    {
      "id": "C-001",
      "rank": 1,
      "x": 3000.0,
      "y": 2700.0,
      "elevation_m": 562.3,
      "form_type": "窝穴",
      "overall_score": 87,
      "scores": {
        "four_beasts": 88,
        "form": 86,
        "sand": 90,
        "water": 85,
        "openness": 80,
        "stability": 92
      },
      "geography": {
        "tpi": -1.85,
        "twi": 9.2,
        "local_slope": 6.5,
        "back_mountain_height_m": 167,
        "back_mountain_distance_m": 250,
        "nearest_water_m": 850,
        "nearest_water_dir": "南",
        "qinglong": 92, "baihu": 88, "zhuque": 85, "xuanwu": 90
      }
    }
  ]
}
```

### 5.2 GeoJSON

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Point", "coordinates": [3000, 2700] },
      "properties": { "id": "C-001", "overall_score": 87, "form_type": "窝穴", ... }
    }
  ]
}
```

---

## 6. 算法原理

详见 `D:\Xunlong\research\99_summary\00_总报告.md`。

核心映射：

| 传统 | 算法 |
|------|------|
| 龙脉 | DEM 山脊线（D8 流向 + 累积=0 + 骨架化） |
| 青龙/白虎 | DEM 扇区统计（候选点东/西 90° 扇区） |
| 朱雀/玄武 | DEM 扇区统计（候选点南/北 90° 扇区） |
| 砂山 | 局部高程峰值检测 |
| 案山/朝山 | 距离分层（200-1000m / 1-5km）的局部极大值 |
| 水口 | D8 流域出水口 |
| 玉带水 | 河曲凹向候选穴 |
| 穴位（TPI）| z0 - mean(邻域) |
| 聚气（TWI）| ln(汇流累积面积 / tan(坡度)) |
| 穴形态 | 窝/钳/乳/突（基于 TPI 与局部坡度） |

---

## 7. 评分公式

```
overall = (
    0.25 × four_beasts_score    # 龙虎朱雀玄武
  + 0.15 × form_score            # 穴位形态
  + 0.18 × sand_score            # 砂山
  + 0.15 × water_score           # 得水
  + 0.10 × openness_score        # 明堂开阔
  + 0.17 × stability_score       # 稳定性
)
```

水系占位时从总分中剔除并归一化。

---

## 8. 数据源

| 数据 | 推荐来源 |
|------|---------|
| DEM | Copernicus DEM 30m, ALOS PALSAR 12.5m, USGS 3DEP |
| 水系 | OpenStreetMap, HydroRIVERS, 中国 1:5万 |
| 道路 | OpenStreetMap |

---

## 9. Roadmap

- [x] **v0.1**：核心算法 + 单元测试
- [ ] **v0.2**：FastAPI 后端 + Vue 3 前端（参考 shanshui-mingtang-fengshui-gis 风格）
- [ ] **v0.3**：阆中真实样本测试 + 调参
- [ ] **v0.4**：LLM 报告生成（DeepSeek 接入）
- [ ] **v0.5**：理气派扩展（八宅、三元九运）
- [ ] **v1.0**：Cesium 3D + 阆中真实寻龙 Demo 复刻

---

## 10. 调研材料

`D:\Xunlong\research\` 目录下 8 份 Markdown 文档（67 KB）：

- `01_core_theory/00_术语与理论总览.md`
- `02_four_beasts/00_四兽量化规则.md`
- `03_dragon_vein/00_龙脉识别算法.md`
- `04_acupoint/00_穴位判定模型.md`
- `05_sand_water/00_砂水分形量化.md`
- `06_dem_gis/00_参考项目与工具链.md`
- `99_summary/00_总报告.md`
- `99_summary/01_速查卡.md`

---

## 11. 致谢

- [limi124/shanshui-mingtang-fengshui-gis](https://github.com/limi124/shanshui-mingtang-fengshui-gis) — 评分公式参考
- 黄培之《提取山脊线和山谷线的一种新方法》
- pysheds, scikit-image, shapely 等开源库

---

## 12. License & Disclaimer

MIT License. 仅供学术研究、WebGIS 实践、传统文化娱乐参考，**不构成任何现实决策建议**。
