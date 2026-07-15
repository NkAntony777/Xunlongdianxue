# 寻龙点穴引擎 v0.3+ — Handoff-Ready 接手必读

> **生成/核对日期**：2026-07-14  
> **仓库根**：`D:\Xunlong`  
> **测试基线**：`154 passed, 4 skipped`（qi 弯内/靠山+开阔；候选⊂高 qi；2026-07-15）  
> **规则真源**：`AGENTS.md`（强制）+ `research/99_summary/03_数理模型_*.md`（公式）

---

## 0. 60 秒速览

```powershell
$env:PYTHONIOENCODING = "utf-8"
cd D:\Xunlong

# 测试（按范围；全量较慢）
& "D:\Xunlong\engine\.venv\Scripts\python.exe" -X utf8 scripts\run_tests.py auto
# 全量: scripts\run_tests.py all   |  仅前端: frontend  |  仅引擎: engine

# 传统模块 demo
& "D:\Xunlong\engine\.venv\Scripts\python.exe" -X utf8 -m engine.examples.demo_traditional_modules --top-k 3

# 后端（默认 8765）
Start-Process -FilePath "D:\Xunlong\engine\.venv\Scripts\python.exe" `
  -ArgumentList "D:\Xunlong\engine\run_server.py" `
  -RedirectStandardOutput "D:\Xunlong\engine\server.log" `
  -RedirectStandardError  "D:\Xunlong\engine\server.err.log" `
  -NoNewWindow -PassThru | Select-Object Id
Start-Sleep 5
Get-Content D:\Xunlong\engine\server.log -Encoding UTF8
Get-Content D:\Xunlong\engine\server.err.log -Encoding UTF8
# 健康检查：应打印 Status + body
$r = Invoke-WebRequest http://127.0.0.1:8765/api/health -UseBasicParsing
"Status: $($r.StatusCode)"; $r.Content
```

**一句话定位**：峦头骨架（龙砂穴水向 + 四象 + 得水/水煞双通道）已可跑端到端；**形局×理气交叉、玄空严格盘、日课择时**未达「专业堪舆」宣称门槛（目标约 v0.5）。

---

## 1. 项目架构

```
D:\Xunlong\
├── AGENTS.md                 # 操作规则（强制）
├── HANDOFF.md                # 本文件
├── engine/                   # 主算法包
│   ├── core/                 # 核心算法（约 19 个 .py）
│   ├── io/                   # DEM / 水系 I/O
│   ├── api/                  # FastAPI（routers + schemas）
│   ├── llm/                  # LLM 报告（DeepSeek 等）
│   ├── tests/                # 单元/集成测试
│   ├── examples/             # demo / smoke
│   ├── data/                 # 默认 DEM 等（或 data/ 在仓库根）
│   ├── run_server.py         # uvicorn 入口
│   └── README.md
├── frontend/                 # Web 前端（HTML/JS/Cesium 等）
├── research/                 # 调研 + 数理 + 审核
├── data/                     # 阆中等真实/合成数据
└── third_party/              # 外部开源对照（非运行时依赖）
```

---

## 2. 核心模块清单（`engine/core/`）

行数为 2026-07-14 本地 `Measure-Object -Line` 近似值。

| 模块 | 约行数 | 功能 |
|------|--------|------|
| `terrain_analysis.py` | 168 | 坡度/坡向/Horn/TPI/粗糙度 |
| `four_beasts.py` | 213 | 四象**扇区评分**（朝向=facing） |
| `four_beasts_detect.py` | 816 | 四兽**点位** + `compute_score_grid` 热力场 |
| `sand_water.py` | 258 | 砂山统计 + 得水关系 |
| `water_model.py` | 338 | **得水 / 水煞双通道** |
| `water_mouth.py` | 303 | 水口 / 交媾点 |
| `water_curve.py` | 185 | 三节弓背水等 |
| `acupoint.py` | 217 | 候选穴 + TPI/TWI/形态 |
| `dragon_vein.py` | 421 | 脊线/入首 + `find_yaoxia` |
| `fengshui_score.py` | ~650 | 综合评分总装（含朝抱/过峡加减） |
| `star_body.py` | 356 | 五行星体 |
| `mountain_curve.py` | 282 | 砂形朝抱 `measure_embrace`（已进砂分） |
| `anshan_quality.py` | 191 | 案山圆净/破碎 |
| `halo_soil.py` | 126 | 晕土（DEM 代理） |
| `compass_directions.py` | 166 | 24 山 + 兼向/出卦 |
| `yuan_yun_xuankong.py` | 207 | 三元九运 + 玄空简化盘 |
| `phenology.py` | 110 | NDVI/土湿接口 + DEM 代理 |
| `aoi_limits.py` | 85 | AOI 半径/面积限制 |
| `render.py` / `rendering/` | — | 分图层 PNG/GeoJSON |

**朝向约定（强制）**：`facing` = **朝向**（北=0…南=180）；青龙=`facing+270`，白虎=`facing+90`，玄武=`facing+180`。详见 `research/99_summary/03_数理模型_*.md`、`04_代码审核报告_*.md`。

---

## 3. 已修复历史项（摘要）

含传统模块审计与 B/A/G 系列（详见历次审计笔记）。代表项：

| 编号 | 项 |
|------|-----|
| B1–B7 | 癸 typo、乾阴阳、洛书序、山向卦、玄空 simplified + 星数置空 |
| D1 | mouth/compass/xuankong **不进 overall**；`meta.weighted_dims` |
| E7 / G.1 | `sand_dist_fn` 真实分辨率与 CRS |
| A2 / G.2 | `star_body` PCA 主轴 |
| C.5 | `mountain_curve` skeletonize + BFS 脊线 |
| A1 部分 | `halo_soil` / `star_body` 已有加减分；其余见「集成缺口」 |
| 朝向 CRITICAL | `four_beasts` 与 detect 统一朝向 + 北向 bearing |
| A1-余 朝抱 | `measure_embrace` → 左右砂 60% 峰数 + 40% 朝抱，进 `sand` 加权维 |
| A1-余 过峡 | `find_yaoxia` → `DragonVeinResult.yaoxia` + overall 固定加减；排名时整幅只跑一次龙脉 |
| C.5.2 | `find_yaoxia` 合成蜂腰 FAIL-case 强化为 `len >= 1` + 中段 col 断言 |
| **G.3** | `compute_score_grid`：快速四象 + 默认跳过水形态 + max_samples 自适应步长；200×200 step=4 ≈0.8s（优化前 ~2.4s+） |
| **B8 下卦** | 已按核验+用户审核修对：运盘永顺、元旦同元龙顺逆、`SHAN_TABLE` 三元龙 12 处、五黄取本山阴阳；黄金局八运子山午向 8-8-3 通过。替卦未做 → **仍禁止** `xuankong_implemented=true` |

---

## 4. 测试覆盖

```text
pytest engine/tests -q
→ 145 passed, 4 skipped  (~73s, 2026-07-14 核对)
```

| 文件 | 内容 |
|------|------|
| `test_engine.py` | 基础集成 |
| `test_four_beasts_detect.py` | 四兽识别 + 朝向约定 |
| `test_water_model.py` | 得水/水煞双通道 |
| `test_aoi_limits.py` | AOI 限制 |
| `test_render_api.py` | 图层/渲染 API |
| `test_location_apis.py` | 在线位置相关 |
| `test_traditional_audit.py` | 传统模块审计（P0–P2 覆盖） |

---

## 5. API 字段语义（候选穴）

`POST /api/candidates/search` → `CandidateItem` 类结构（实现以 `fengshui_score.FusedScore` + schema 为准）。

### 5.1 `scores`

| 键 | 含义 | 进 overall？ |
|----|------|----------------|
| `four_beasts` | 四象扇区综合 | **是** |
| `form` | 形态 | **是** |
| `sand` | 砂（可含案山质量） | **是** |
| `water` / `water_get` | **得水**加分通道 | **是**（water） |
| `openness` | 明堂开阔 | **是** |
| `stability` | 稳定 | **是** |
| `water_sha` | **水煞**罚分 | 乘性衰减，非加权平均项 |
| `mouth` | 水口 | **展示**（交媾 bonus 等规则见代码） |
| `compass` | 罗盘纯净 | **展示** |
| `xuankong` | 玄空 | **强制 None / 不实现完整盘** |
| `halo_soil` | 晕土 | 有加减计入 overall |
| `star_body_bonus` | 星体档位 | 有加减计入 overall |
| `embrace_left` / `embrace_right` | 左右砂朝抱分 | **已并入 sand 加权**；字段仅展示 |
| `yaoxia_bonus` | 过峡入首加减 | 固定加减计入 overall（-2…+6） |

### 5.2 `meta`（前端硬约束，勿删）

```text
weighted_dims          # 实际加权维度列表
xuankong_implemented   # 恒为 false，直至严格盘完成
phenology_is_proxy     # DEM 代理物候
mouth_evaluated
star_body_type / star_body_is_xuanwu_eligible
halo_soil_notes
embrace_in_sand        # 朝抱已并入 sand 维
yaoxia_evaluated / yaoxia_count / yaoxia_notes
```

### 5.3 `geography`（节选）

- 峦头：靠山高差/距离、四象分  
- 罗盘：`compass_shan`, `deviation`, 兼向/出卦  
- 水口：`water_mouth_lock_ratio`, `mouth_evaluated`  
- 玄空：运/元/卦名可有；**星数字段一律 null**  
- 物候：total/ndvi/moisture + `is_proxy=true`  
- 水：`water_get` / `water_sha` / `water_form`  
- 朝抱：`embrace_left` / `embrace_right`（已并入 sand）  
- 过峡：`nearest_yaoxia_m` / `yaoxia_count` / `yaoxia_evaluated`

---

## 6. 待办清单（接手优先）

### P0

| 编号 | 任务 | 量级 | 风险 |
|------|------|------|------|
| ~~**G.3**~~ | ~~`compute_score_grid` 性能~~ | — | **已完成** → 乘性生气场 v2：宽平台得水、围合底分、近平不罚、细尺度平台藏风；候选 TPI 阈值 0；平缓形态高分 |
| ~~**A1-余** 朝抱~~ | ~~`measure_embrace` → 砂分~~ | — | **已完成**（左右砂 60/40 融合） |
| ~~**A1-余** 过峡~~ | ~~`find_yaoxia` → 评分 + 字段~~ | — | **已完成**（`DragonVeinResult.yaoxia` + overall 加减） |
| ~~**B8 修**~~ | ~~下卦三盘按核验修正~~ | — | **已完成**（运盘永顺 + 同元龙 + SHAN_TABLE + 五黄 D2；黄金 8-8-3） |
| **B8 余** | 替卦 / 完整反伏吟 / 城门 / 零正神 / 接入 overall | 天 | 下卦已通；**仍禁止** `xuankong_implemented=true` |
| ~~**C.5.2**~~ | ~~`find_yaoxia` FAIL-case 测试加强~~ | — | **已完成**（`len>=1` + 中段 col） |

### P1

| 编号 | 任务 | 状态 |
|------|------|------|
| ~~A1-水曲~~ | ~~`score_water_curve_*` → 得水 form 通道~~ | **已完成**（`enrich_form_with_water_curve` → Γ/P；双通道不混维） |
| ~~B14~~ | ~~来龙方位注入 `facing_cross_check`~~ | **已完成**（排名时入首→穴方位；反局 -10） |
| ~~E.5~~ | ~~全 NaN DEM 空结果回退~~ | **已完成**（`analyze_terrain` 中性 / search+rank 空列表） |
| ~~B2~~ | ~~三元龙 + 净阴净阳~~ | **已完成**（随 SHAN_TABLE 校准 + 测试） |
| ~~C.4~~ | ~~八方位 vs 二十四山~~ | **已完成**（0/90/180/270 落子卯午酉断言） |
| C.2 | `compute_flow_accumulation` → 拓扑序 / pysheds | **延期**（需 pysheds/生产 DEM 标定，非本轮） |
| C.3 | `classify_form` 与 `tpi_radius_m` 标定 | **延期**（需生产样本标定） |
| D.2–D.5 | 出卦文案、i18n、性能 timeout 标记 | **延期**（产品/文案向） |

### P2–P3（学术完整 · 明确延期）

日课择时、三合局并列、护龙/夹送水、龙身八格、喝形 100+、曜星、Pydantic 全强类型等。

### 已写未进 overall（集成缺口）

| 函数 | 模块 | 状态 |
|------|------|------|
| `measure_embrace` | `mountain_curve` | **已进 sand 加权** |
| `find_yaoxia` | `dragon_vein` | **已进 overall 固定加减** |
| `score_water_curve_*` | `water_curve` | **已进 form→Γ_get/P_form**（经 `enrich_form_with_water_curve`） |
| `facing_cross_check` | `compass_directions` | **已进 overall 罚分**（来龙注入时） |
| `score_yun` / `wang_xiang` / `fly_chart_strict` | `yuan_yun_xuankong` | 下卦已修；候选仍简化盘展示；**未进 overall**；`xuankong_implemented=false` |
| `score_xuanwu_by_star` | `star_body` | 部分：type 奖惩已进；函数本身未直调 |

---

## 7. 固化原则（不可破）

1. **玄空**：对外候选路径用 `fly_chart` 简化盘（星数 **None**）；`fly_chart_strict` 为研究用下卦（已修对）；替卦未做前 **不得** `xuankong_implemented=true`。  
2. **mouth / compass / xuankong / phenology**：以展示为主；`meta.weighted_dims` 明示真实加权项。  
3. **禁止** PowerShell `-ErrorAction SilentlyContinue`（`AGENTS.md`）。  
4. 改 `score_candidate` **必须保留** `FusedScore.meta` 字段（前端硬约束）。  
5. 调用 venv Python 时：`PYTHONIOENCODING=utf-8` + `-X utf8`。  
6. 得水与水煞 **双通道**，禁止再用单一 `f(距水)` 混维（`water_model.py` + 数理 03）。

---

## 8. 已知风险

| 风险 | 说明 |
|------|------|
| 细分辨率 DEM | 12.5m 上 mouth 等需生产验证 |
| 跨 CRS | 4326 DEM + 3857 水系端到端 smoke 不足 |
| LLM | `llm/generator.py` 是否透传 `geography.xuankong_*` 需 review |
| 评分场大 DEM | G.3 已优化；超大 AOI 仍靠 `sample_step`/`max_samples`；`use_water_form=True` 会明显变慢 |

---

## 9. 相关文档索引

| 文档 | 用途 |
|------|------|
| `AGENTS.md` | 启动/测试/编码规范 |
| `research/99_summary/03_数理模型_*.md` | 公式真源 |
| `research/99_summary/04_代码审核报告_*.md` | 朝向 CRITICAL 等 |
| `research/99_summary/05_理论核验_玄空飞星与形局_*.md` | B8/形局理论核验（实现前必读） |
| `research/99_summary/02_调研索引_*.md` | 调研地图 |
| `research/01_core_theory/01_传统点穴总论_*.md` | 自然语言总论 |
| `research/05_sand_water/01_得水与水煞_*.md` | 水法分离语义 |
| `engine/README.md` | 引擎说明 |
| `third_party/README.md` | 开源对照仓库 |

---

## 10. 版本判断

| 版本感 | 含义 |
|--------|------|
| **v0.3+（当前）** | 峦头 + 双通道水 + 四象点位/评分 + 罗盘/水口/简化玄空展示骨架完整 |
| **v0.5 目标** | 形局×理气交叉校验、玄空严格盘、关键集成缺口入 overall、评分场性能可生产 |

> 在 v0.5 前 **勿对外宣称「专业堪舆水准」**。

---

## 11. 免责

本项目仅供学术研究、WebGIS 实践与传统文化娱乐参考。不构成投资、医疗、婚姻、法律或商业选址决策建议。
