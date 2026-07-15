# API 路由模块

按领域拆分，避免单文件堆积。

| 模块 | 前缀 | 职责 |
|------|------|------|
| `search.py` | `/api/location` | Nominatim 搜索 |
| `elevation.py` | `/api/elevation` | ESRI DEM 拉取 |
| `water.py` | `/api/water` | OSM 水系拉取（可降级） |
| `aoi.py` | `/api/aoi` | 分析区半径约束 |
| `cache.py` | `/api/cache` | 缓存与临时文件 |
| `layers.py` | `/api/layers` | 分图层分析渲染 |
| `analysis.py` | `/api/terrain` | 地形分析 |
| `candidates.py` | `/api/candidates` | 候选穴 |
| `dragon_vein.py` | `/api/dragon-vein` | 龙脉 |
| `render.py` | `/api/render` | 旧渲染 API |
| `report.py` / `llm.py` | 报告 / LLM | |

`location.py` 仅为兼容 re-export，新代码请直接引用具体模块。
