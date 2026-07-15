# 前端模块说明

`index.html` 仅保留页面结构；样式与逻辑均已拆分。

## 目录

```
frontend/
├── index.html          # 页面骨架
├── css/app.css         # 全局样式
├── assets/             # leaflet 等第三方静态资源
└── js/
    ├── app.js          # 入口：组装与启动
    ├── config.js       # 常量
    ├── state.js        # 全局状态单例
    ├── utils.js        # DOM / 格式化
    ├── api.js          # 后端 HTTP 封装
    ├── progress.js     # 分步加载 UI
    ├── aoi.js          # 高德圈选 AOI
    ├── analysis-map.js # 分析结果画布平移缩放
    ├── render-ui.js    # 图层 / SVG / 右侧面板
    ├── analysis.js     # Demo / 在线分析流程
    └── search.js       # 地点搜索
```

## 依赖关系（简）

```
app.js
 ├─ aoi.js ── api.js, state, utils
 ├─ analysis-map.js ── state, utils
 ├─ render-ui.js ── analysis-map, config, state
 ├─ analysis.js ── api, progress, aoi, render-ui
 └─ search.js ── api, aoi
```

## 约定

- 使用原生 ES modules（`type="module"`），不经打包。
- 可变状态集中在 `state.js`，避免跨文件隐式全局。
- 网络请求只走 `api.js`，便于 mock 与改基址。
