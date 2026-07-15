"""兼容入口：engine.core.render → engine.core.rendering。

请优先::

    from engine.core.rendering import render_basemap, ...
"""
from engine.core.rendering import *  # noqa: F403
