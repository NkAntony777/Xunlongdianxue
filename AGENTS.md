# AGENTS.md

> 寻龙点穴引擎项目（Xunlong Engine）的开发规则与上下文。

## 接手必读

- **现状与待办**：根目录 [`HANDOFF.md`](HANDOFF.md)（v0.3+ handoff-ready）
- **公式真源**：`research/99_summary/03_数理模型_点穴抽象与公式体系.md`
- **审核纪要**：`research/99_summary/04_代码审核报告_2026-07.md`

## 项目目标

在指定区域，基于 DEM 与水系等数据，按"四兽"与"龙砂穴水向"等风水参数，自动分析并输出候选龙穴地点 + 综合评分。

## 重要：操作规则

### 1. 禁止使用 `-ErrorAction SilentlyContinue`

**所有 PowerShell 命令禁止使用 `SilentlyContinue`**。原因：会静默吞掉错误，调试时看不到问题。

```powershell
# 错误（禁止）
Get-Content file.log -ErrorAction SilentlyContinue

# 正确（推荐）
Get-Content file.log
# 或
$ErrorActionPreference = "Stop"; Get-Content file.log
```

### 2. 服务状态必须显式输出

- 启动后台服务时，必须 `Get-Content` 日志文件验证服务确实起来
- 测试 API 时必须打印 `Status` 和响应体
- 任何 `Start-Process` 后必须 sleep + 检查日志

### 3. 终端编码

PowerShell 5.1 默认 GBK 编码，UTF-8 中文会乱码。Read 工具显示也偶有乱码。文件实际内容是正确的，可用以下方式验证：

```powershell
Get-Content -LiteralPath "xxx.json" -Encoding UTF8
```

### 4. Python 解释器

使用项目 venv：`D:\Xunlong\engine\.venv\Scripts\python.exe`
不要用系统 Python（uv 管理、PEP 668 保护、不能装包）。

### 5. 测试

每次改完代码后必须跑：

```powershell
cd D:\Xunlong
engine\.venv\Scripts\python.exe -m pytest engine\tests\test_engine.py -q
```

## 目录结构

```
D:\Xunlong\
├── AGENTS.md                  # 本文件
├── research/                  # 调研材料（8 份 Markdown）
├── data/                      # 阆中真实/合成数据
├── engine/                    # 寻龙点穴引擎
│   ├── core/                  # 核心算法 (terrain / four_beasts / acupoint / ...)
│   ├── io/                    # 数据 I/O
│   ├── api/                   # FastAPI 后端
│   ├── llm/                   # LLM 集成
│   ├── tests/                 # 单元测试
│   ├── examples/              # Demo / 测试脚本
│   ├── cli.py                 # CLI 入口
│   ├── run_server.py          # uvicorn 启动器
│   ├── requirements.txt
│   └── README.md
└── frontend/                  # Web 前端 (HTML + JS + Cesium)
    ├── index.html
    ├── css/style.css
    └── js/
```

## 启动服务

```powershell
cd D:\Xunlong
# 启动后端（默认 8765 端口）
Start-Process -FilePath "D:\Xunlong\engine\.venv\Scripts\python.exe" -ArgumentList "D:\Xunlong\engine\run_server.py" -RedirectStandardOutput "D:\Xunlong\engine\server.log" -RedirectStandardError "D:\Xunlong\engine\server.err.log" -NoNewWindow -PassThru | Select-Object Id
Start-Sleep 4
Get-Content D:\Xunlong\engine\server.log
Get-Content D:\Xunlong\engine\server.err.log
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/info` | 服务元信息 |
| POST | `/api/terrain/analyze` | 地形分析 |
| POST | `/api/candidates/search` | 候选穴搜索 |
| POST | `/api/candidates/geojson` | 候选穴 GeoJSON |
| POST | `/api/dragon-vein/extract` | 龙脉识别 |
| POST | `/api/report/generate` | 生成报告 |
| POST | `/api/llm/interpret` | LLM 解读 |

## 数据格式

- DEM: GeoTIFF (.tif)
- 水系: GeoJSON / Shapefile
- 报告: JSON

## 免责

本项目仅供学术研究、WebGIS 实践、传统文化娱乐参考。
不构成任何投资、医疗、婚姻、法律、商业选址决策建议。
