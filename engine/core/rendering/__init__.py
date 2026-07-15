"""渲染子系统。

子模块（逐步拆分中）:
  - pipeline.py  当前完整实现（兼容迁移）
  - 后续: basemap / hydro / score / buildable / vectors

对外 API 仍从 engine.core.render 导入。
"""
from engine.core.rendering.pipeline import *  # noqa: F403
