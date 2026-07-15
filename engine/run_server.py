"""直接启动 uvicorn（开发用）。"""
import sys
from pathlib import Path

# 添加项目根目录到 PYTHONPATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "engine.api.main:app",
        host="127.0.0.1",
        port=8765,
        reload=False,
        log_level="info",
    )
