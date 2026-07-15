# third_party — 外部开源参考（独立目录）

本目录存放 **寻龙点穴引擎** 调研用的第三方开源仓库，**不作为引擎运行时依赖**。
请勿把本目录代码直接 import 进 `engine/`；需要时对照算法后在本仓库内重写。

## 已克隆仓库

| 目录 | 来源 | 许可/用途 | 与本项目关系 |
|------|------|-----------|--------------|
| `shanshui-mingtang-fengshui-gis/` | [limi124/shanshui-mingtang-fengshui-gis](https://github.com/limi124/shanshui-mingtang-fengshui-gis) | 民俗文化/WebGIS 实践 | **主要代码参考**：围栏级 DEM 指标、靠山/明堂/围合/得水评分、水系距离规则 |
| `yijing-fengshui/` | [wolke/yijing-fengshui](https://github.com/wolke/yijing-fengshui) | CC BY-NC-SA 4.0 | 阳宅/卦象方位，**无 DEM 四象定位**；仅作文化方位对照 |

## 更新方式

```powershell
cd D:\Xunlong\third_party\shanshui-mingtang-fengshui-gis
git pull
```

## 调研笔记

算法对照与落地建议见：

- `../research/02_four_beasts/01_开源与文献对照_四象定位.md`

## 说明

- 克隆日期约 2026-07-14（shallow clone `--depth 1`）
- 若体积过大，可将本目录加入 `.gitignore` 后由各环境自行 clone
- 免责：外部项目声明仅供文化娱乐与 WebGIS 实践；本项目同样不作选址决策依据
