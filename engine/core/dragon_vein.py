"""龙脉识别：兼容门面（实现见 engine.core.dragon.*）。

历史路径保留:
  from engine.core.dragon_vein import analyze_dragon_vein, select_primary_dragon
"""
from __future__ import annotations

from engine.core.dragon import *  # noqa: F403
from engine.core.dragon import __all__  # noqa: F401
